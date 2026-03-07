"""
evolving_ising.runner
=====================
High-level evolution loop: `EvoRunner` wires together an `IsingModel`,
a `TemperatureDiffuser`, and a `SeparableCMAES` optimizer into a single
`run()` call that returns the best coupling matrix J_nk found.

This module is the integration layer — it owns the parameter-space
representation (theta → J_nk mapping) and the fitness evaluation loop,
but delegates all physics and optimisation details to the sub-modules.

Typical usage
-------------
    cfg   = EvoConfig(pop_size=32, iters=200, chains_per_eval=64)
    runner = EvoRunner(ising, pin_mask, pin_values, cfg)
    J_best, fitness = runner.run()
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp

from .diffusion import TemperatureDiffuser
from .model import IsingModel
from .optim import SeparableCMAES

Array = jnp.ndarray


@dataclass
class EvoConfig:
    """
    Hyperparameters for `EvoRunner`.

    Fields
    ------
    pop_size : int
        CMA-ES population size λ.
    iters : int
        Number of CMA-ES generations.
    sigma_init : float
        Initial CMA-ES step-size σ₀.
    j_scale : float
        Multiplier applied after the theta→J transform.  Controls the
        overall magnitude of the learned couplings.
    j_transform : str
        Nonlinearity mapping unconstrained theta → positive J values:
          "softplus" — log(1 + e^θ), smooth and always > 0  (default)
          "sigmoid"  — σ(θ), bounded in (0, 1)
          "relu"     — max(θ, 0), sparse (zero for negative θ)
          "tanh01"   — (tanh(θ)+1)/2, bounded in (0, 1), saturates
    warmup_sweeps : int
        Number of Metropolis+diffusion steps discarded before measuring
        fitness.  Allows the spin/temperature state to reach approximate
        equilibrium before the fitness signal is collected.
    measure_sweeps : int
        Number of sweeps over which fitness is averaged after warmup.
        Averaging reduces variance in the fitness estimate.
    diffusion_steps_per_sweep : int
        How many diffusion steps to run per Metropolis sweep.
    alpha_diffusion : float
        Diffusion rate α passed to TemperatureDiffuser.
    conductance_mode : str
        Passed to TemperatureDiffuser; controls how J → conductance.
    normalize_mode : str
        Passed to TemperatureDiffuser; "row" or "none".
    chains_per_eval : int
        Number of independent Markov chains run in parallel per candidate.
        More chains reduce fitness variance but cost more memory/compute.
    seed : int
        PRNG seed for both the CMA-ES and the Metropolis chain keys.
    """
    pop_size: int = 32
    iters: int = 200
    sigma_init: float = 0.3
    j_scale: float = 1.0
    j_transform: str = "softplus"
    warmup_sweeps: int = 10
    measure_sweeps: int = 10
    diffusion_steps_per_sweep: int = 2
    alpha_diffusion: float = 0.5
    conductance_mode: str = "abs"
    normalize_mode: str = "row"
    chains_per_eval: int = 64
    seed: int = 0


class EvoRunner:
    """
    High-level evolution loop combining physics simulation with CMA-ES.

    Architecture
    ------------
    The CMA-ES optimises over a flat parameter vector theta of length D,
    where D = number of valid (non-masked) entries in J_nk.  At each
    generation:

      1. Sample population: theta_i ~ N(mean, σ² diagC)   [D-dimensional]
      2. Map theta → J_nk via `vec_to_Jnk` (softplus + j_scale)
      3. For each candidate J_nk, run `warmup_sweeps` steps to equilibrate,
         then average energy over `measure_sweeps` steps as the fitness.
      4. Update CMA-ES from (theta_i, fitness_i) pairs.

    Fitness = negative mean Ising energy (lower energy = better alignment),
    so the optimizer is maximising -E, which minimises E.

    Parameters
    ----------
    ising      : IsingModel        the physical grid
    pin_mask   : (N,) bool         sites with fixed temperatures
    pin_values : (N,) float32      their fixed values
    cfg        : EvoConfig         all hyperparameters
    """

    def __init__(self, ising: IsingModel, pin_mask: Array, pin_values: Array, cfg: EvoConfig):
        self.ising = ising
        self.cfg = cfg
        self.pin_mask   = pin_mask.astype(bool)
        self.pin_values = pin_values.astype(jnp.float32)
        self.N = ising.n
        self.K = ising.K

        # flat_idx: 1D indices into J_nk.reshape(-1) where mask is True.
        # Only these D positions are optimised; the rest stay zero.
        self.flat_idx = jnp.where(ising.mask.reshape(-1))[0]
        self.D = int(self.flat_idx.shape[0])

        self.es = SeparableCMAES(dim=self.D, pop_size=cfg.pop_size, sigma_init=cfg.sigma_init, seed=cfg.seed)
        self._diffuser = TemperatureDiffuser(
            alpha=cfg.alpha_diffusion,
            conductance_mode=cfg.conductance_mode,
            normalize_mode=cfg.normalize_mode,
        )

    def _theta_to_Jpos(self, theta: Array) -> Array:
        """
        Apply the configured nonlinearity to map unconstrained theta → J > 0.

        The transform ensures J_nk is always non-negative (for conductance
        modes that require it) and scales the output by cfg.j_scale.
        """
        m = self.cfg.j_transform
        if m == "softplus":
            return jax.nn.softplus(theta) * self.cfg.j_scale
        elif m == "sigmoid":
            return jax.nn.sigmoid(theta) * self.cfg.j_scale
        elif m == "relu":
            return jnp.maximum(theta, 0.0) * self.cfg.j_scale
        elif m == "tanh01":
            return 0.5 * (jnp.tanh(theta) + 1.0) * self.cfg.j_scale
        else:
            return jax.nn.softplus(theta) * self.cfg.j_scale

    def vec_to_Jnk(self, theta: Array) -> Array:
        """
        Expand a flat parameter vector theta (D,) → J_nk (N, K).

        Only the D valid positions (given by flat_idx) are filled;
        all other slots remain zero and are zeroed further by the mask
        in downstream computations.

        Parameters
        ----------
        theta : (D,) float32  unconstrained CMA-ES parameters

        Returns
        -------
        Array (N, K) float32
        """
        J = jnp.zeros((self.N * self.K,), dtype=jnp.float32)
        J = J.at[self.flat_idx].set(self._theta_to_Jpos(theta))
        return J.reshape(self.N, self.K)

    def _diffuse(self, J_nk: Array, T_b: Array) -> Array:
        """Run one round of diffusion (cfg.diffusion_steps_per_sweep steps)."""
        return self._diffuser.diffuse(
            self.ising.neighbors, J_nk, self.ising.mask, T_b,
            steps=self.cfg.diffusion_steps_per_sweep,
            pin_mask=self.pin_mask,
            pin_values=self.pin_values,
        )

    def _evaluate_single(self, key: Array, theta: Array) -> Array:
        """
        Evaluate one candidate theta: warmup then measure mean energy.

        The fitness is the *negative* mean energy (so CMA-ES maximises it,
        driving the system toward lower energy / more aligned spins).

        Warmup
        ------
        Run `warmup_sweeps` interleaved diffusion+Metropolis steps, starting
        from a random spin configuration and an approximately linear T profile.
        The warmup is discarded — it just brings the system close to stationarity.

        Measurement
        -----------
        Run `measure_sweeps` additional steps and average the energy across
        them and across the `chains_per_eval` parallel chains.  Averaging
        reduces the noise in the fitness signal.

        Returns
        -------
        Array ()  float32  scalar fitness value
        """
        J_nk = self.vec_to_Jnk(theta)
        B, N = self.cfg.chains_per_eval, self.N
        key_s, key_perm = jax.random.split(key)
        spins0 = self.ising.init_spins(key_s, B)

        # Initialise temperature: mean of pin_values everywhere, then
        # overwrite pinned sites with their fixed values
        T0 = jnp.where(
            jnp.broadcast_to(self.pin_mask[None], (B, N)),
            jnp.broadcast_to(self.pin_values[None], (B, N)),
            jnp.full((B, N), jnp.mean(self.pin_values)),
        )

        def do_sweep(carry, _):
            """One interleaved diffusion + Metropolis step."""
            spins_c, T_c, key_c = carry
            T_c = self._diffuse(J_nk, T_c)
            key_c, sub = jax.random.split(key_c)
            spins_c, _ = self.ising.metropolis_checkerboard_sweeps(sub, spins_c, J_nk, T_c, num_sweeps=1)
            return (spins_c, T_c, key_c), None

        # ── Warmup (discarded) ───────────────────────────────────────────
        (spins_w, T_w, key_w), _ = jax.lax.scan(
            do_sweep, (spins0, T0, key_perm), None, length=self.cfg.warmup_sweeps
        )

        def measure(carry, _):
            """One measurement step: advance the chain and accumulate energy."""
            spins_c, T_c, key_c, e_sum = carry
            (spins_c, T_c, key_c), _ = jax.lax.scan(do_sweep, (spins_c, T_c, key_c), None, length=1)
            e_sum += jnp.mean(self.ising.energy(J_nk, spins_c))
            return (spins_c, T_c, key_c, e_sum), None

        # ── Measurement (averaged) ───────────────────────────────────────
        (_, _, _, e_acc), _ = jax.lax.scan(
            measure, (spins_w, T_w, key_w, 0.0), None, length=self.cfg.measure_sweeps
        )
        return -(e_acc / float(self.cfg.measure_sweeps))  # negate: maximise = minimise energy

    def _batched_evaluate(self, key: Array, thetas: Array) -> Array:
        """
        Evaluate the entire population in parallel via vmap.

        Each candidate gets an independent PRNG key (split from the shared
        key) so the chains don't share randomness.
        """
        keys = jax.random.split(key, thetas.shape[0])
        return jax.jit(jax.vmap(self._evaluate_single, in_axes=(0, 0)))(keys, thetas)

    def run(self) -> Tuple[Array, Array]:
        """
        Run the full evolution loop for cfg.iters generations.

        Returns
        -------
        J_best    : (N, K) float32  best coupling matrix found
        best_fit  : ()     float32  corresponding fitness value
        """
        key = jax.random.PRNGKey(self.cfg.seed)
        best_theta, best_fit = None, -jnp.inf

        for _ in range(self.cfg.iters):
            X = self.es.ask()                         # sample population
            key, sub = jax.random.split(key)
            fitness = self._batched_evaluate(sub, X)  # evaluate all candidates
            self.es.tell(X, fitness)                  # update distribution

            idx = int(jnp.argmax(fitness))
            if best_theta is None or fitness[idx] > best_fit:
                best_theta = X[idx]
                best_fit   = fitness[idx]

        return self.vec_to_Jnk(best_theta), best_fit