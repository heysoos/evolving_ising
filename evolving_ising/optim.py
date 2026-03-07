"""
evolving_ising.optim
====================
Implements `SeparableCMAES`: a self-contained, array-only separable
(diagonal-covariance) CMA-ES optimizer that runs entirely on JAX arrays
and requires no external evosax dependency.

CMA-ES background
-----------------
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is a black-box
optimiser for continuous, non-convex problems.  It maintains a multivariate
Gaussian N(m, σ² C) over the search space and iteratively:
  1. Samples a population  X ~ N(m, σ² C)      [ask]
  2. Evaluates fitness f(X_i) for each sample
  3. Updates m, σ, C from the top-μ winners    [tell]

The *separable* variant restricts C to a diagonal matrix (stored as a
vector `diagC`), reducing O(D²) memory/compute to O(D).  This is a good
approximation when the problem is approximately axis-aligned, and is much
cheaper for the high-dimensional parameter vectors used here (D ~ 10⁴).

Adaptation equations (Hansen, 2016)
-------------------------------------
Evolution path for step-size control:
    p_s ← (1-c_s) p_s + √(c_s(2-c_s) μ_eff) · C^{-1/2} · (m_new - m_old)/σ

Heaviside indicator (stagnation guard):
    h_σ = 1  if  ‖p_s‖/√(1-(1-c_s)^{2t}) < (1.4 + 2/(D+1)) · E‖N(0,I)‖

Evolution path for rank-1 covariance update:
    p_c ← (1-c_c) p_c + h_σ √(c_c(2-c_c) μ_eff) · (m_new - m_old)/σ

Diagonal covariance update (rank-1 + rank-μ):
    diagC ← (1-c_1-c_μ) diagC + c_1 p_c² + c_μ Σ_i w_i z_i²

Step-size update (CSA):
    σ ← σ · exp(c_s/d_σ · (‖p_s‖/E‖N(0,I)‖ - 1))

Usage
-----
    es = SeparableCMAES(dim=D, pop_size=32, sigma_init=0.5)
    for _ in range(num_iters):
        X = es.ask()                  # (pop_size, D)
        fitness = evaluate(X)         # (pop_size,) — higher is better
        es.tell(X, fitness)
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

Array = jnp.ndarray


@dataclass
class CMAState:
    """
    Mutable state of the CMA-ES distribution.

    Fields
    ------
    mean  : (D,) float32   current distribution mean
    sigma : ()   float32   global step-size σ
    diagC : (D,) float32   diagonal of the covariance matrix C
    pc    : (D,) float32   evolution path for rank-1 covariance update
    ps    : (D,) float32   evolution path for step-size control (CSA)
    rng   : PRNGKey        JAX PRNG state advanced each ask() call
    """
    mean:  Array  # (D,)
    sigma: Array  # scalar
    diagC: Array  # (D,)
    pc:    Array  # (D,)
    ps:    Array  # (D,)
    rng:   Array  # PRNGKey


class SeparableCMAES:
    """
    Separable (diagonal-covariance) CMA-ES.  Maximises fitness.

    Parameters
    ----------
    dim      : int    dimensionality of the search space
    pop_size : int    number of samples per generation (λ)
    sigma_init : float  initial step-size σ₀
    seed     : int    seed for the internal JAX PRNG
    """

    def __init__(self, dim: int, pop_size: int, sigma_init: float, seed: int = 0):
        self.dim = dim
        self.lam = pop_size          # λ: total population size
        self.mu  = max(1, pop_size // 2)  # μ: number of elites used for update

        # ── Recombination weights ────────────────────────────────────────
        # Logarithmic weights down-weight lower-ranked individuals.
        # Normalised so they sum to 1.
        i = jnp.arange(self.mu, dtype=jnp.float32)
        w = jnp.log(self.mu + 0.5) - jnp.log(i + 1.0)
        w = w / jnp.sum(w)
        self.w = w

        # Effective number of parents (variance effective selection mass)
        self.mu_eff = 1.0 / jnp.sum(w ** 2)

        # ── Adaptation coefficients (Hansen, 2016, Table 1) ─────────────
        D = float(dim)

        # c_c: learning rate for the rank-1 evolution path p_c
        self.cc    = (4 + self.mu_eff / D) / (D + 4 + 2 * self.mu_eff / D)

        # c_s: learning rate for the step-size evolution path p_s
        self.cs    = (self.mu_eff + 2.0) / (D + self.mu_eff + 5.0)

        # c_1: learning rate for the rank-1 covariance update
        self.c1    = 2.0 / ((D + 1.3) ** 2 + self.mu_eff)

        # c_μ: learning rate for the rank-μ covariance update
        self.cmu   = jnp.minimum(
            1.0 - self.c1,
            2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / ((D + 2.0) ** 2 + self.mu_eff),
        )

        # d_σ: damping coefficient for step-size control
        self.damps = 1.0 + 2.0 * jnp.maximum(0.0, jnp.sqrt((self.mu_eff - 1.0) / (D + 1.0)) - 1.0) + self.cs

        # Expected L2 norm of an isotropic Gaussian sample in D dimensions.
        # Used as the target ‖p_s‖ in the step-size update.
        self.E_norm = jnp.sqrt(D) * (1.0 - 1.0 / (4.0 * D) + 1.0 / (21.0 * D * D))

        # ── Initial state ────────────────────────────────────────────────
        self.state = CMAState(
            mean  = jnp.zeros((dim,), dtype=jnp.float32),   # start at origin
            sigma = jnp.asarray(sigma_init, dtype=jnp.float32),
            diagC = jnp.ones((dim,), dtype=jnp.float32),    # isotropic initial covariance
            pc    = jnp.zeros((dim,), dtype=jnp.float32),
            ps    = jnp.zeros((dim,), dtype=jnp.float32),
            rng   = jax.random.PRNGKey(seed),
        )

    def ask(self) -> Array:
        """
        Sample a population from the current distribution.

        Samples:  X_i = mean + σ · √diagC ⊙ z_i,   z_i ~ N(0, I)

        The PRNG key is advanced and stored back into state so that
        successive ask() calls produce independent samples.

        Returns
        -------
        Array (pop_size, dim) float32
        """
        key, sub = jax.random.split(self.state.rng)
        z = jax.random.normal(sub, (self.lam, self.dim), dtype=jnp.float32)
        # Scale each dimension by its standard deviation √diagC[d]
        X = self.state.mean[None] + self.state.sigma * z * jnp.sqrt(self.state.diagC)[None]
        # Advance the RNG in state (everything else is unchanged)
        self.state = CMAState(
            mean=self.state.mean, sigma=self.state.sigma, diagC=self.state.diagC,
            pc=self.state.pc, ps=self.state.ps, rng=key,
        )
        return X

    def tell(self, X: Array, fitness: Array) -> None:
        """
        Update the distribution from the evaluated population.

        Selects the top-μ individuals by fitness (maximisation), computes
        the weighted mean, then updates the evolution paths, diagonal
        covariance, and step-size according to the CMA-ES equations.

        Parameters
        ----------
        X       : (pop_size, dim) float32  the population returned by ask()
        fitness : (pop_size,)     float32  objective values (higher = better)
        """
        # ── Elite selection ──────────────────────────────────────────────
        idx   = jnp.argsort(-fitness)          # descending rank
        X_sel = X[idx[: self.mu]]              # (μ, D) top individuals
        w = self.w[:, None]                    # (μ, 1) for broadcasting

        mean_old = self.state.mean
        mean_new = jnp.sum(w * X_sel, axis=0)  # weighted centroid

        # ── Normalised steps in whitened space ───────────────────────────
        # y: steps from old mean, normalised by current step-size
        y = (X_sel - mean_old[None]) / jnp.maximum(self.state.sigma, 1e-8)
        # z: further whitened by the diagonal standard deviations
        z = y / jnp.sqrt(self.state.diagC)[None]
        # Weighted sum of whitened steps (used for p_s update)
        z_w = jnp.sum(self.w[:, None] * z, axis=0)

        # ── Step-size evolution path (CSA) ───────────────────────────────
        ps = (1 - self.cs) * self.state.ps + jnp.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * z_w
        norm_ps = jnp.linalg.norm(ps)

        # Heaviside h_σ: suppress rank-1 update when p_s is too long,
        # which indicates that the step-size is still adapting
        h_sigma = (
            norm_ps / jnp.sqrt(1 - (1 - self.cs) ** 2) < (1.4 + 2.0 / (self.dim + 1.0)) * self.E_norm
        ).astype(jnp.float32)

        # ── Rank-1 evolution path ────────────────────────────────────────
        y_w = jnp.sum(w * y, axis=0)  # weighted mean step (un-whitened)
        pc = (1 - self.cc) * self.state.pc + h_sigma * jnp.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * y_w

        # ── Diagonal covariance update ───────────────────────────────────
        # Three contributions:
        #   (1-c1-cμ) diagC  — decay (prevents indefinite growth)
        #   c1 · pc²          — rank-1 update from evolution path
        #   cμ · Σ w_i z_i²   — rank-μ update from elite whitened steps
        diagC = (
            (1 - self.c1 - self.cmu) * self.state.diagC
            + self.c1 * pc ** 2
            + self.cmu * jnp.sum(w * z ** 2, axis=0)
        )

        # ── Step-size update (exponential CSA) ───────────────────────────
        # If ‖p_s‖ > E_norm, the distribution is moving fast → increase σ.
        # If ‖p_s‖ < E_norm, progress is slow → decrease σ.
        sigma = self.state.sigma * jnp.exp((self.cs / self.damps) * (norm_ps / self.E_norm - 1.0))

        self.state = CMAState(mean=mean_new, sigma=sigma, diagC=diagC, pc=pc, ps=ps, rng=self.state.rng)