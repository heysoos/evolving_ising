"""
evolving_ising.objectives
=========================
Defines the experiment registry (`EXPERIMENTS`) and the fitness evaluation
machinery (`make_eval_fn`) used by the loss-function comparison experiments.

Separation of concerns
-----------------------
This module owns *what to optimise* — the mapping from a candidate parameter
vector theta to a scalar fitness.  It does not own *how to optimise* (that's
`SeparableCMAES` in optim.py) or *how to save results* (that's experiment.py).

Parameter representation
------------------------
The CMA-ES operates on a flat vector theta ∈ ℝᴰ, where D is the number of
valid (non-masked) entries in J_nk.  `vec_to_Jnk` maps theta → J_nk via:

    J_nk[flat_idx] = softplus(theta) * j_scale

softplus ensures J > 0 everywhere (important for conductance-based diffusion
modes like "abs" or "relu" which are identity for positive inputs).

PhysicsSetup
------------
Rather than passing ~10 separate arrays to every function, the shared
physical configuration is bundled into a `PhysicsSetup` dataclass.  This
makes function signatures clean and ensures all eval functions see the same
grid, boundary conditions, and hyperparameters.

Adding a new experiment
-----------------------
1. Add an entry to the `EXPERIMENTS` dict with keys "title", "description",
   "formula".
2. Add a matching `elif objective == "<key>":` branch in `make_eval_fn` that
   defines `eval_single(key, theta) -> scalar`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from .diffusion import TemperatureDiffuser
from .model import IsingModel

Array = jnp.ndarray


# ── PhysicsSetup ────────────────────────────────────────────────────────────

@dataclass
class PhysicsSetup:
    """
    Immutable bundle of all shared physical configuration needed to run
    and evaluate a simulation.

    Passed into `make_eval_fn` and `run_experiment` so that the physical
    setup is defined once (in `run_experiments.py`) and reused across
    all objectives without repetition.

    Fields
    ------
    ising : IsingModel
        The grid geometry and neighbourhood structure.
    diffuser : TemperatureDiffuser
        Configured diffusion operator (conductance mode, alpha, etc.).
    T0 : (N,) float32
        Initial temperature profile for each chain.  Typically a linear
        gradient from cold_t (top) to hot_t (bottom).
    pin_mask : (N,) bool
        True for sites whose temperature is held fixed each diffusion step
        (e.g. the bottom row acting as a heat reservoir).
    pin_values : (N,) float32
        Temperature values restored at pinned sites after each step.
    top_idx : (W,) int32
        Flat indices of the top row (row 0).  Used to measure how much
        heat has been transported up from the hot boundary.
    flat_idx : (D,) int32
        Indices into J_nk.reshape(-1) of the valid (non-masked) bond slots.
        The CMA-ES parameter vector has exactly D entries, one per slot.
    iters_eval : int
        Total number of coupled diffusion+Metropolis steps in one fitness
        evaluation.  More steps → more equilibration → less noisy fitness,
        but slower evaluation.
    steps_per_iter : int
        Number of diffusion steps per Metropolis sweep within each iter.
    sweeps_per_iter : int
        Number of Metropolis sweeps per iter.
    chains_per_eval : int
        Parallel Markov chains per candidate.  More chains average out
        stochastic noise in the fitness estimate.
    j_scale : float
        Multiplier on the softplus output; controls the overall coupling scale.
    """
    ising: IsingModel
    diffuser: TemperatureDiffuser
    T0: Array
    pin_mask: Array
    pin_values: Array
    top_idx: Array
    flat_idx: Array
    iters_eval: int = 80
    steps_per_iter: int = 2
    sweeps_per_iter: int = 2
    chains_per_eval: int = 8
    j_scale: float = 1.0

    @property
    def N(self) -> int:
        """Total number of sites (H * W)."""
        return self.ising.n

    @property
    def K(self) -> int:
        """Number of neighbour slots per site."""
        return self.ising.K

    @property
    def D(self) -> int:
        """Dimensionality of the CMA-ES parameter vector (number of valid bond slots)."""
        return int(self.flat_idx.shape[0])


# ── Parameter-space → J_nk ──────────────────────────────────────────────────

def vec_to_Jnk(theta: Array, setup: PhysicsSetup) -> Array:
    """
    Expand a flat CMA-ES parameter vector into a full coupling matrix.

    Mapping:  theta (D,)  →  J_nk (N, K)

    Only the D valid bond positions (setup.flat_idx) are filled.  All other
    entries (invalid boundary slots, enforced by mask) remain zero.

    The softplus nonlinearity maps ℝ → ℝ₊, ensuring J > 0.  Combined with
    conductance_mode="abs" (which is identity for positive inputs) this means
    the learned couplings are always positive, acting as pure conductances.

    Parameters
    ----------
    theta : (D,) float32  unconstrained parameter vector from CMA-ES
    setup : PhysicsSetup

    Returns
    -------
    Array (N, K) float32
    """
    J = jnp.zeros((setup.N * setup.K,), dtype=jnp.float32)
    J = J.at[setup.flat_idx].set(jax.nn.softplus(theta.astype(jnp.float32)) * setup.j_scale)
    return J.reshape(setup.N, setup.K)


# ── Experiment registry ──────────────────────────────────────────────────────

# Each entry maps a string key to display metadata.
# The key must match an `elif` branch in `make_eval_fn` below.
EXPERIMENTS: dict = {
    "max_top_temp": {
        "title": "Maximize Top-Row Temperature",
        "description": (
            "Maximize mean temperature on the top row.  The baseline heat-channeling "
            "objective: evolution should learn couplings that conduct heat upward."
        ),
        "formula": "fitness = mean(T_top_row)",
    },
    "max_mean_temp": {
        "title": "Maximize Mean Temperature",
        "description": (
            "Maximize the average temperature across the entire grid.  Since the only "
            "heat source is the pinned bottom row, this rewards conductors that spread "
            "heat widely, not just channel it to the top."
        ),
        "formula": "fitness = mean(T_all)",
    },
    "min_temp_variance": {
        "title": "Minimize Temperature Variance",
        "description": (
            "Drive the temperature field toward spatial uniformity.  Low variance means "
            "the grid has equalised — either all hot or all cold.  Interesting comparison "
            "with the top-row objective."
        ),
        "formula": "fitness = -std(T_all)",
    },
    "max_neg_energy": {
        "title": "Minimize Ising Energy",
        "description": (
            "Minimize the Ising energy directly, favouring spin configurations where "
            "neighbours are aligned.  The temperature field is used but the objective "
            "ignores it — evolution cares only about spin order."
        ),
        "formula": "fitness = -mean(E) / N",
    },
    "max_top_temp_low_energy": {
        "title": "Top-Row Temp + Low Energy (Multi-Objective)",
        "description": (
            "Weighted combination of heat transport and spin alignment.  The 0.1 "
            "coefficient keeps the energy term as a regulariser rather than dominating."
        ),
        "formula": "fitness = mean(T_top) + 0.1 * (-mean(E) / N)",
    },
}


# ── Fitness function factory ─────────────────────────────────────────────────

def make_eval_fn(
    objective: str, setup: PhysicsSetup
) -> Tuple[Callable, Callable]:
    """
    Build a pair of JIT-compiled evaluation functions for the given objective.

    The returned functions close over `setup` — they capture the grid,
    diffuser, and all hyperparameters at definition time so there is no
    overhead from passing them at each call.

    Shared dynamics
    ---------------
    All objectives run the same coupled diffusion + Metropolis loop via
    `_dynamics`, then extract different scalar summaries from the final state.
    The loop uses `jax.lax.scan` so it compiles to a single fused XLA program.

    Parameters
    ----------
    objective : str   key from EXPERIMENTS
    setup     : PhysicsSetup

    Returns
    -------
    eval_single     : (key, theta) -> scalar float32
        Evaluate a single candidate theta.
    eval_population : (key, thetas) -> (pop_size,) float32
        Evaluate the whole population in parallel via vmap.
    """
    if objective not in EXPERIMENTS:
        raise ValueError(f"Unknown objective '{objective}'. Valid: {list(EXPERIMENTS)}")

    # Unpack for cleaner closure (avoids repeated attribute lookups in traced code)
    ising    = setup.ising
    diffuser = setup.diffuser

    def _dynamics(key, theta):
        """
        Run the full simulation for one candidate.

        Returns (spins_f, T_f, J_nk) after `setup.iters_eval` coupled steps:
          - T is diffused `steps_per_iter` times
          - Then spins are updated `sweeps_per_iter` Metropolis sweeps
          - The bottom-row boundary is re-pinned to HOT_T after each diffusion

        All chains start from the same T0 profile but independent random spins.
        `jax.lax.scan` fuses the entire loop into one XLA computation.
        """
        J_nk = vec_to_Jnk(theta, setup)
        B = setup.chains_per_eval
        key_s, key_loop = jax.random.split(key)
        spins = ising.init_spins(key_s, B)
        T = jnp.broadcast_to(setup.T0[None], (B, setup.N))

        def body(carry, _):
            spins_c, T_c, key_c = carry
            # 1. Diffuse temperature (re-pins the hot boundary internally)
            T_c = diffuser.diffuse(
                ising.neighbors, J_nk, ising.mask, T_c,
                steps=setup.steps_per_iter,
                pin_mask=setup.pin_mask,
                pin_values=setup.pin_values,
            )
            # 2. Update spins at the new local temperatures
            key_c, sub = jax.random.split(key_c)
            spins_c, _ = ising.metropolis_checkerboard_sweeps(
                sub, spins_c, J_nk, T_c, num_sweeps=setup.sweeps_per_iter,
            )
            return (spins_c, T_c, key_c), None

        (spins_f, T_f, _), _ = jax.lax.scan(body, (spins, T, key_loop), None, length=setup.iters_eval)
        return spins_f, T_f, J_nk

    # ── Objective-specific eval_single functions ─────────────────────────
    # Each is decorated with @jax.jit independently, producing a separate
    # compiled kernel per objective.  The _dynamics closure is traced into
    # each kernel at compile time.

    if objective == "max_top_temp":
        @jax.jit
        def eval_single(key, theta):
            # Average temperature on the top row, averaged over all chains
            _, T_f, _ = _dynamics(key, theta)
            return jnp.mean(T_f[:, setup.top_idx])

    elif objective == "max_mean_temp":
        @jax.jit
        def eval_single(key, theta):
            # Global mean temperature across the entire grid and all chains
            _, T_f, _ = _dynamics(key, theta)
            return jnp.mean(T_f)

    elif objective == "min_temp_variance":
        @jax.jit
        def eval_single(key, theta):
            # Negative std dev — minimising variance drives uniformity
            _, T_f, _ = _dynamics(key, theta)
            return -jnp.std(T_f)

    elif objective == "max_neg_energy":
        @jax.jit
        def eval_single(key, theta):
            # Negative normalised energy — minimising E maximises this
            spins_f, _, J_nk = _dynamics(key, theta)
            return -jnp.mean(ising.energy(J_nk, spins_f)) / setup.N

    elif objective == "max_top_temp_low_energy":
        @jax.jit
        def eval_single(key, theta):
            # Weighted sum: heat transport (primary) + spin alignment (regulariser)
            spins_f, T_f, J_nk = _dynamics(key, theta)
            top_temp   = jnp.mean(T_f[:, setup.top_idx])
            neg_energy = -jnp.mean(ising.energy(J_nk, spins_f)) / setup.N
            return top_temp + 0.1 * neg_energy

    @jax.jit
    def eval_population(key, thetas):
        """
        Evaluate all candidates in parallel.

        Uses `jax.vmap` to vectorise `eval_single` over the population axis,
        giving each candidate an independent PRNG key so chains are decorrelated.

        Parameters
        ----------
        key    : PRNGKey
        thetas : (pop_size, D) float32

        Returns
        -------
        Array (pop_size,) float32
        """
        keys = jax.random.split(key, thetas.shape[0])
        return jax.vmap(eval_single, in_axes=(0, 0))(keys, thetas)

    return eval_single, eval_population