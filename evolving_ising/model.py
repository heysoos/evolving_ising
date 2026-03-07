"""
evolving_ising.model
====================
Defines `IsingModel`: a 2D Ising spin system on a rectangular grid, designed
for batch parallel simulation on GPU via JAX.

Design choices
--------------
- Spins are stored as int8 in {-1, +1} to minimise memory bandwidth.
- Neighbour connectivity is precomputed into dense (N, K) index and mask
  arrays so that all gather/scatter operations are simple matmuls or
  indexing — no sparse formats needed.
- Metropolis updates use a *grouped (checkerboard/coloured) scheme*: sites
  are partitioned into independent colour classes so every site in a class
  can be flipped simultaneously without creating data races.  Von Neumann
  neighbourhoods need 2 colours; Moore neighbourhoods need 4.
- All heavy methods are `@jax.jit`'d with `static_argnums=(0,)` so the
  model object (which owns the static graph arrays) acts as a compile-time
  constant.  The same compiled kernel is reused across calls with the same
  batch size and sweep count.
"""
from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax

Array = jnp.ndarray


class IsingModel:
    """
    GPU-friendly Ising model on a 2D grid (H × W) with configurable
    neighbourhood and boundary conditions.

    Attributes
    ----------
    h, w : int
        Grid height and width.
    n : int
        Total number of sites (H * W).
    K : int
        Number of neighbour slots per site (4 for von_neumann, 8 for moore).
    neighbors : Array (N, K)  int32
        Flat index of each neighbour.  Invalid slots (boundary sites with no
        wrap) point to index 0 and are masked out by `mask`.
    mask : Array (N, K)  bool
        True wherever the corresponding neighbor slot is a real neighbour.
        Always zero-out J_nk at False slots before using neighbours.
    rev_slot : Array (N, K)  int32
        For site n and neighbour slot k, `rev_slot[n, k]` is the slot index
        in the neighbour's J array that points back to n.  Used when you need
        symmetric J without storing both directions explicitly.

    Parameters
    ----------
    grid_hw : (H, W)
        Grid dimensions.
    neighborhood : {"moore", "von_neumann"}
        8-neighbour (Moore) or 4-neighbour (von Neumann) stencil.
    boundary : {"open", "periodic", "periodic_lr", "periodic_tb"}
        open        — no wrapping; edge sites have fewer valid neighbours.
        periodic    — wrap in both dimensions (torus).
        periodic_lr — wrap left↔right (columns) only.
        periodic_tb — wrap top↔bottom (rows) only.
    """

    def __init__(self, grid_hw: Tuple[int, int], neighborhood: str = "moore", boundary: str = "open"):
        self.h, self.w = grid_hw
        self.n = self.h * self.w
        assert neighborhood in {"moore", "von_neumann"}, "neighborhood must be 'moore' or 'von_neumann'"
        assert boundary in {"open", "periodic", "periodic_lr", "periodic_tb"}, (
            "boundary must be one of 'open', 'periodic', 'periodic_lr', 'periodic_tb'"
        )
        self.neighborhood = neighborhood
        self.boundary = boundary

        # Fixed (row, col) offsets for each neighbour slot, in a canonical order.
        # The order matters: slot k for site n always corresponds to the same
        # spatial direction, enabling the reverse-slot lookup below.
        if neighborhood == "moore":
            offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        else:  # von_neumann
            offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]

        self.K = len(offsets)
        self._offsets = tuple(offsets)

        # Build a map: offset -> slot index, then derive the reverse map.
        # _rev_slot_map[off] = slot index of the *opposite* direction (-off),
        # i.e. if slot k of site n points to neighbour m, then rev_slot_map[off]
        # gives the slot of m that points back to n.
        offset_to_slot = {off: i for i, off in enumerate(offsets)}
        self._rev_slot_map = {off: offset_to_slot.get((-off[0], -off[1]), -1) for off in offsets}

        def idx(i, j):
            """Flat row-major index for grid position (i, j)."""
            return i * self.w + j

        # wrap_h=True: columns wrap left↔right (used by periodic and periodic_lr).
        # wrap_v=True: rows wrap top↔bottom   (used by periodic and periodic_tb).
        wrap_h = self.boundary in ("periodic", "periodic_lr")
        wrap_v = self.boundary in ("periodic", "periodic_tb")
        self._wrap_h = wrap_h
        self._wrap_v = wrap_v

        # Build the neighbour table in Python (done once at construction).
        # For each site (i, j) and each offset, determine the flat index of
        # the neighbour (with wrapping), or mark it invalid for open boundaries.
        rows, masks, rev_slots = [], [], []
        for i in range(self.h):
            for j in range(self.w):
                row_neigh, row_mask, row_rev = [], [], []
                for off in offsets:
                    ni, nj = i + off[0], j + off[1]
                    # Apply per-axis wrapping where enabled
                    ni_w = (ni % self.h) if wrap_v else ni
                    nj_w = (nj % self.w) if wrap_h else nj
                    # A neighbour is valid if it stays inside the grid
                    # after applying the (possibly absent) wrapping
                    valid = (wrap_v or 0 <= ni < self.h) and (wrap_h or 0 <= nj < self.w)
                    if valid:
                        row_neigh.append(idx(ni_w, nj_w))
                        row_mask.append(True)
                        rev = self._rev_slot_map[off]
                        row_rev.append(rev if rev >= 0 else 0)
                    else:
                        # Invalid slot: point to site 0 (harmless; mask=False prevents use)
                        row_neigh.append(0)
                        row_mask.append(False)
                        row_rev.append(0)
                rows.append(row_neigh)
                masks.append(row_mask)
                rev_slots.append(row_rev)

        self.neighbors = jnp.array(rows, dtype=jnp.int32)    # (N, K)
        self.mask      = jnp.array(masks, dtype=bool)         # (N, K)
        self.rev_slot  = jnp.array(rev_slots, dtype=jnp.int32)  # (N, K)

        # ── Colour masks for parallel Metropolis updates ─────────────────
        # Sites are partitioned into colour classes such that no two sites
        # in the same class are neighbours of each other.  Within a class,
        # all flips are independent and can be applied simultaneously.
        #
        # Von Neumann (4-nbr): standard checkerboard 2-colouring suffices —
        #   colour = (row + col) mod 2.
        # Moore (8-nbr): diagonals also connect same-checkerboard-parity
        #   sites, so we need a 4-colouring keyed on (row%2, col%2).
        rows_idx = jnp.arange(self.n, dtype=jnp.int32) // self.w
        cols_idx = jnp.arange(self.n, dtype=jnp.int32) % self.w
        if neighborhood == "von_neumann":
            cid = (rows_idx + cols_idx) & 1                           # 0 or 1
            self._color_masks = jnp.stack([cid == 0, cid == 1], axis=0)  # (2, N)
            self._num_colors = 2
        else:
            cid = (rows_idx & 1) * 2 + (cols_idx & 1)                # 0, 1, 2, or 3
            self._color_masks = jnp.stack([cid == g for g in range(4)], axis=0)  # (4, N)
            self._num_colors = 4

    # ── Initialisation ───────────────────────────────────────────────────

    def init_spins(self, key: Array, batch_size: int) -> Array:
        """
        Sample a random spin configuration.

        Returns
        -------
        Array (batch_size, N)  int8
            Spins uniformly drawn from {-1, +1}.
        """
        s = jax.random.bernoulli(key, 0.5, (batch_size, self.n))
        return s.astype(jnp.int8) * 2 - 1  # {0,1} -> {-1,+1}

    # ── Energy computation ───────────────────────────────────────────────

    @partial(jax.jit, static_argnums=(0,))
    def energy(self, J_nk: Array, spins: Array) -> Array:
        """
        Total Ising energy per chain.

        E = -0.5 * Σ_i  s_i * h_i,   where  h_i = Σ_k J[i,k] * s[nbr[i,k]]

        The factor 0.5 corrects for double-counting each bond (i→j and j→i).
        Invalid neighbour slots are zeroed via `mask` before the sum.

        Parameters
        ----------
        J_nk : (N, K) float32
            Per-bond coupling strengths.  Invalid slots should be 0.
        spins : (B, N) int8
            Spin configuration batch.

        Returns
        -------
        Array (B,) float32
            Total energy for each chain in the batch.
        """
        spins_f = spins.astype(jnp.float32)
        J_masked = J_nk * self.mask.astype(J_nk.dtype)   # zero out invalid slots
        h = jnp.sum(spins_f[:, self.neighbors] * J_masked[None], axis=2)  # (B, N)
        return -0.5 * jnp.einsum("bi,bi->b", spins_f, h)

    @partial(jax.jit, static_argnums=(0,))
    def local_energy(self, J_nk: Array, spins: Array) -> Array:
        """
        Per-site energy contribution.

        Returns e_i = -0.5 * s_i * h_i for every site, so that
        `energy(J_nk, spins) == local_energy(J_nk, spins).sum(axis=-1)`.

        Returns
        -------
        Array (B, N) float32
        """
        spins_f = spins.astype(jnp.float32)
        J_masked = J_nk * self.mask.astype(J_nk.dtype)
        h = jnp.sum(spins_f[:, self.neighbors] * J_masked[None], axis=2)
        return -0.5 * spins_f * h

    @partial(jax.jit, static_argnums=(0,))
    def _compute_field(self, J_nk: Array, spins: Array) -> Array:
        """
        Local effective field at every site.

        h_i = Σ_k  J[i,k] * s[neighbors[i,k]]

        Used internally by the Metropolis update to compute ΔE = 2 s_i h_i.

        Returns
        -------
        Array (B, N) float32
        """
        spins_f = spins.astype(jnp.float32)
        J_masked = J_nk * self.mask.astype(J_nk.dtype)
        return jnp.sum(spins_f[:, self.neighbors] * J_masked[None], axis=2)

    # ── Metropolis updates ───────────────────────────────────────────────

    def _metropolis_update_masked_broadcasted(
        self, key: Array, spins: Array, J_nk: Array, T_b: Array, site_mask: Array
    ) -> Array:
        """
        Propose flipping every site simultaneously, accept according to the
        Metropolis criterion, but only actually apply flips to sites in
        `site_mask` (i.e. the current colour class).

        Algorithm per site i
        --------------------
        1. Compute ΔE_i = 2 s_i h_i  (energy cost of flipping s_i)
        2. Accept flip if ΔE_i ≤ 0  (always), or with probability
           exp(-ΔE_i / T_i)  otherwise.  T_i is clipped to ≥ 1e-8 to
           prevent division by zero at T=0.
        3. Only sites where site_mask[i]=True are eligible to flip.

        The exp() argument is clipped to [-50, 50] to prevent overflow;
        the lower clip is irrelevant since that case is always accepted.

        Parameters
        ----------
        key : PRNGKey
        spins : (B, N) int8
        J_nk  : (N, K) float32
        T_b   : (B, N) float32  per-site temperatures, already broadcast
        site_mask : (N,) bool   True for sites in the active colour class

        Returns
        -------
        Array (B, N) int8  updated spins
        """
        spins_f = spins.astype(jnp.float32)
        dE = 2.0 * spins_f * self._compute_field(J_nk, spins)  # (B, N)
        T_b = jnp.maximum(T_b, jnp.asarray(1e-8, dtype=T_b.dtype))
        accept_prob = jnp.exp(-jnp.clip(dE / T_b, a_min=-50.0, a_max=50.0))
        u = jax.random.uniform(key, shape=spins_f.shape, dtype=spins_f.dtype)
        # Accept if energy-lowering OR if random draw passes Boltzmann threshold,
        # AND only for sites in the active colour class
        accept = ((dE <= 0.0) | (u < accept_prob)) & jnp.broadcast_to(site_mask[None], spins_f.shape)
        return jnp.where(accept, -spins_f, spins_f).astype(spins.dtype)

    @partial(jax.jit, static_argnums=(0, 5))
    def metropolis_checkerboard_sweeps(
        self, key: Array, spins: Array, J_nk: Array, T: Array, num_sweeps: int = 1,
    ) -> Tuple[Array, Array]:
        """
        Run `num_sweeps` full Metropolis sweeps using the grouped checkerboard
        scheme.  Within each sweep, colour classes are updated sequentially;
        within each class, all sites are updated in parallel.

        `num_sweeps` is a compile-time constant (static_argnums) so JAX
        unrolls the outer loop at trace time, enabling full fusion.

        Temperature `T` is broadcast to (B, N) automatically:
          - scalar  → same temperature everywhere
          - (N,)    → same profile for each chain
          - (B, N)  → fully per-site, per-chain

        Parameters
        ----------
        key        : PRNGKey
        spins      : (B, N) int8
        J_nk       : (N, K) float32
        T          : scalar | (N,) | (B, N) float32
        num_sweeps : int  (static)

        Returns
        -------
        new_spins : (B, N) int8
        energies  : (B,)  float32   energy of the final configuration
        """
        B, N = spins.shape
        spins = spins.astype(jnp.int8)
        J_nk  = J_nk.astype(jnp.float32)

        # Broadcast T to (B, N) once before the sweep loop
        if T.ndim == 0:
            T_b = jnp.full((B, N), T, dtype=J_nk.dtype)
        elif T.ndim == 1:
            T_b = jnp.broadcast_to(T[None], (B, N))
        else:
            T_b = T

        G = self._num_colors
        color_masks = self._color_masks  # (G, N)

        def one_sweep(spins_c, key_sweep):
            """Apply one complete sweep: update each colour class in turn."""
            keys_g = jax.random.split(key_sweep, G)  # independent key per colour
            def apply_color(g, s):
                return self._metropolis_update_masked_broadcasted(keys_g[g], s, J_nk, T_b, color_masks[g])
            return jax.lax.fori_loop(0, G, apply_color, spins_c)

        # Pre-split keys for all sweeps to keep the loop body pure (no side effects)
        keys_sweeps = jax.random.split(key, num_sweeps)
        spins = jax.lax.fori_loop(0, num_sweeps, lambda t, s: one_sweep(s, keys_sweeps[t]), spins)
        return spins, self.energy(J_nk, spins)

    @partial(jax.jit, static_argnums=(0, 5))
    def metropolis_checkerboard_sweeps_with_history(
        self, key: Array, spins: Array, J_nk: Array, T: Array, num_sweeps: int = 1,
    ) -> Tuple[Array, Array, Array]:
        """
        Same as `metropolis_checkerboard_sweeps`, but records the spin
        configuration after every sweep using `lax.scan`.

        Returns
        -------
        spins   : (B, N) int8         final configuration
        energy  : (B,)   float32      energy of the final configuration
        history : (num_sweeps, B, N) int8  snapshot after each sweep
        """
        B, N = spins.shape
        spins = spins.astype(jnp.int8)
        J_nk  = J_nk.astype(jnp.float32)

        if T.ndim == 0:
            T_b = jnp.full((B, N), T, dtype=J_nk.dtype)
        elif T.ndim == 1:
            T_b = jnp.broadcast_to(T[None], (B, N))
        else:
            T_b = T

        G = self._num_colors
        color_masks = self._color_masks

        def one_sweep(spins_c, key_sweep):
            keys_g = jax.random.split(key_sweep, G)
            def apply_color(g, s):
                return self._metropolis_update_masked_broadcasted(keys_g[g], s, J_nk, T_b, color_masks[g])
            spins_c = jax.lax.fori_loop(0, G, apply_color, spins_c)
            return spins_c, spins_c  # (carry, output) — output is stacked into history

        keys_sweeps = jax.random.split(key, num_sweeps)
        spins_f, hist = jax.lax.scan(one_sweep, spins, keys_sweeps)  # hist: (num_sweeps, B, N)
        return spins_f, self.energy(J_nk, spins_f), hist

    # ── Backwards-compatible aliases ─────────────────────────────────────

    @partial(jax.jit, static_argnums=(0, 5))
    def metropolis_sweeps(self, key, spins, J_nk, T, num_sweeps=1):
        """Alias for metropolis_checkerboard_sweeps."""
        return self.metropolis_checkerboard_sweeps(key, spins, J_nk, T, num_sweeps)

    @partial(jax.jit, static_argnums=(0, 5))
    def metropolis_sweeps_with_history(self, key, spins, J_nk, T, num_sweeps=1):
        """Alias for metropolis_checkerboard_sweeps_with_history."""
        return self.metropolis_checkerboard_sweeps_with_history(key, spins, J_nk, T, num_sweeps)

    # ── Utility ──────────────────────────────────────────────────────────

    def wrap_flags(self) -> Tuple[bool, bool]:
        """Return (wrap_h, wrap_v): True if columns / rows wrap respectively."""
        return self._wrap_h, self._wrap_v

    def vertical_edge_masks(self) -> Tuple[Array, Array]:
        """
        Return (has_up, has_down) boolean masks of shape (N,).

        A site has `has_up=True` if its up-neighbour slot is valid (i.e. it is
        not on the top boundary, or top-bottom wrapping is enabled).  Useful
        for identifying interior vs boundary sites along the temperature gradient.

        For Moore neighbourhoods where up/down cannot be cleanly separated from
        diagonals, returns a conservative approximation: True if any vertically
        displaced neighbour exists.
        """
        up_idx = down_idx = -1
        for k, off in enumerate(self._offsets):
            if off == (-1, 0):
                up_idx = k
            elif off == (1, 0):
                down_idx = k
        if up_idx >= 0 and down_idx >= 0:
            # Von Neumann: exact up/down slots exist
            return self.mask[:, up_idx], self.mask[:, down_idx]
        # Moore fallback: mark any site with a vertically-displaced neighbour
        rows = jnp.arange(self.n, dtype=jnp.int32) // self.w
        nbr_rows = self.neighbors // self.w
        vert = jnp.any((nbr_rows != rows[:, None]) & self.mask, axis=1)
        return vert, vert