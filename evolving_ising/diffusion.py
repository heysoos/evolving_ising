"""
evolving_ising.diffusion
========================
Defines `TemperatureDiffuser`: a graph-based heat diffusion model that
propagates local temperatures across the Ising neighbour graph.

Physics
-------
At each step the temperature field evolves as:

    T_{t+1}[i] = (1 - α) * T_t[i]  +  α * Σ_k  W_norm[i,k] * T_t[nbr[i,k]]

where W_norm[i,k] is a normalised conductance derived from the bond coupling
J_nk.  The conductance mode controls how J values map to non-negative weights
(e.g. abs(J), relu(J), softplus(J), …).  With row-normalisation the weighted
average over neighbours is a proper convex combination, keeping temperatures
bounded in [min_pin, max_pin].

The diffuser is deliberately kept stateless — all graph structure (neighbors,
mask) is passed at call time, not stored, so a single diffuser instance can
be reused across different IsingModel configurations.
"""
from __future__ import annotations

from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax

Array = jnp.ndarray


class TemperatureDiffuser:
    """
    Diffuses temperature over the Ising neighbour graph.

    Parameters
    ----------
    alpha : float
        Diffusion rate in (0, 1].  At α=1 each site fully adopts the weighted
        average of its neighbours in one step; at α→0 the field barely moves.
    conductance_mode : str
        How to derive non-negative conductance weights from J_nk:
          "abs"      — |J|          (symmetric, default)
          "relu"     — max(J, 0)    (only positive couplings conduct)
          "softplus" — log(1+e^J)   (smooth, always positive)
          "square"   — J²           (emphasises large couplings)
          "sigmoid"  — σ(J)         (bounded in (0,1))
    normalize_mode : str
        "row" (default) — row-normalise W so each site's neighbourhood
                          weights sum to 1.  Temperature stays bounded.
        "none"          — use raw conductances; temperatures can grow.
    eps : float
        Small constant added to row sums before dividing, preventing NaN
        at isolated sites with no valid neighbours.
    use_abs : bool | None
        Deprecated back-compat flag.  If set, overrides conductance_mode
        with "abs" (True) or "relu" (False).
    """

    def __init__(
        self,
        alpha: float = 0.5,
        conductance_mode: str = "abs",
        normalize_mode: str = "row",
        eps: float = 1e-8,
        use_abs: Optional[bool] = None,
    ):
        if use_abs is not None:
            conductance_mode = "abs" if use_abs else "relu"
        self.alpha = alpha
        self._c_mode = conductance_mode
        self._n_mode = normalize_mode
        self.eps = eps

    @partial(jax.jit, static_argnums=(0,))
    def _conductance(self, J_nk: Array) -> Array:
        """
        Convert raw coupling weights J_nk → non-negative conductances.

        The conductance_mode is baked into the JIT trace as a static string,
        so each mode produces a distinct compiled kernel.

        Returns
        -------
        Array (N, K) float32  non-negative conductance values
        """
        mode = self._c_mode
        if mode == "abs":
            return jnp.abs(J_nk)
        elif mode == "relu":
            return jnp.maximum(J_nk, 0.0)
        elif mode == "softplus":
            return jax.nn.softplus(J_nk)
        elif mode == "square":
            return J_nk * J_nk
        elif mode == "sigmoid":
            return jax.nn.sigmoid(J_nk)
        else:
            return jnp.abs(J_nk)  # safe default

    @partial(jax.jit, static_argnums=(0,))
    def _normalize(self, W: Array, mask: Array) -> Array:
        """
        Zero out invalid neighbour slots then optionally row-normalise.

        Row-normalisation divides each row by its sum + eps, so the weights
        form a probability distribution over the valid neighbours.  This keeps
        the diffusion step a convex combination, bounding T in [T_min, T_max].

        Returns
        -------
        Array (N, K) float32  normalised conductance weights
        """
        Wm = W * mask.astype(W.dtype)   # zero invalid slots
        if self._n_mode == "none":
            return Wm
        deg = jnp.sum(Wm, axis=1, keepdims=True) + self.eps
        return Wm / deg

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        neighbors: Array,               # (N, K) int32
        J_nk: Array,                    # (N, K) float32
        mask: Array,                    # (N, K) bool
        T: Array,                       # (B, N) or (N,) float32
        pin_mask: Optional[Array] = None,    # (N,) bool
        pin_values: Optional[Array] = None,  # (N,) float32
    ) -> Array:
        """
        Apply one diffusion step and return the updated temperature field.

        After the weighted averaging, any sites whose index appears in
        `pin_mask` are hard-reset to `pin_values`.  This enforces fixed
        boundary temperatures (e.g. a hot reservoir at the bottom row).

        Parameters
        ----------
        neighbors  : (N, K) int32  — neighbour flat indices (from IsingModel)
        J_nk       : (N, K) float32 — coupling weights
        mask       : (N, K) bool    — True where neighbour slot is valid
        T          : (B, N) or (N,) float32 — current temperature field;
                     if (N,) it is broadcast to (1, N) internally.
        pin_mask   : (N,) bool   — sites to hold fixed after each step
        pin_values : (N,) float32 — values to restore at pinned sites

        Returns
        -------
        Array (B, N) float32  updated temperature field
        """
        Wn = self._normalize(self._conductance(J_nk), mask)  # (N, K)
        if T.ndim == 1:
            T = T[None]   # (N,) -> (1, N) so batch indexing works
        # Gather neighbour temperatures: T[:, neighbors] has shape (B, N, K).
        # Multiply by normalised weights and sum over the K neighbour slots.
        T_flow = jnp.sum(T[:, neighbors] * Wn[None], axis=2)  # (B, N)
        T_new = (1.0 - self.alpha) * T + self.alpha * T_flow
        # Restore pinned boundary sites (e.g. hot reservoir)
        if pin_mask is not None and pin_values is not None:
            T_new = jnp.where(pin_mask[None], pin_values[None], T_new)
        return T_new

    @partial(jax.jit, static_argnums=(0, 5))
    def diffuse(
        self,
        neighbors: Array,
        J_nk: Array,
        mask: Array,
        T0: Array,
        steps: int,                          # static — determines loop unrolling
        pin_mask: Optional[Array] = None,
        pin_values: Optional[Array] = None,
    ) -> Array:
        """
        Run `steps` sequential diffusion steps via `jax.lax.scan`.

        `steps` is a compile-time constant so XLA can fully pipeline the scan
        body.  The same compiled kernel is reused for repeated calls with the
        same `steps` value.

        Parameters
        ----------
        neighbors  : (N, K) int32
        J_nk       : (N, K) float32
        mask       : (N, K) bool
        T0         : (B, N) or (N,) float32  initial temperature field
        steps      : int  (static)
        pin_mask   : (N,) bool   optional
        pin_values : (N,) float32 optional

        Returns
        -------
        Array (B, N) float32  temperature field after `steps` steps
        """
        def body(T, _):
            return self.step(neighbors, J_nk, mask, T, pin_mask, pin_values), None
        T, _ = lax.scan(body, T0, None, length=steps)
        return T