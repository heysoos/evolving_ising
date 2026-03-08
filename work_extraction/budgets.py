"""Budget classes for gating bond remodelling (Phase 3).

Three budget strategies with a shared BaseBudget interface:
- BondBudget: per-bond accumulation from ordering events (eq. 11)
- NeighbourhoodBudget: site budgets pooled over neighbourhood (eq. 12)
- DiffusingBudget: reaction-diffusion field on the lattice (eq. 13)
- NoBudget: always returns inf (used for Experiment 0)

All update methods use JAX for GPU-accelerated vectorized operations.
Budget state is stored as JAX arrays to avoid host↔device transfers
during the simulation loop.
"""

import numpy as np
import jax
import jax.numpy as jnp


def _prep_spins(spins):
    """Ensure spins are a 1D JAX float32 array, averaging over batch."""
    s = jnp.asarray(spins, dtype=jnp.float32)
    if s.ndim == 2:
        s = s.mean(axis=0)
    return s


@jax.jit
def _compute_ordering(s_before, s_after, neighbors, mask):
    """Vectorized ordering event computation (GPU).

    Returns (N, K) array of ordering magnitudes (>= 0).
    """
    nbr_before = s_before[neighbors]   # (N, K)
    nbr_after = s_after[neighbors]

    corr_before = s_before[:, None] * nbr_before
    corr_after = s_after[:, None] * nbr_after

    delta_corr = corr_after - corr_before
    return jnp.maximum(0.0, delta_corr) * mask


class BaseBudget:
    """Base class for budget strategies."""

    def update(self, spins_before, spins_after, J_nk, T):
        raise NotImplementedError

    def get_budget(self, i, j):
        raise NotImplementedError

    def spend(self, i, j, cost):
        raise NotImplementedError

    def get_all_budgets_for_bonds(self, bond_i, bond_j):
        """Return budgets for an array of bonds as a JAX array."""
        return jnp.array([self.get_budget(int(i), int(j))
                          for i, j in zip(bond_i, bond_j)])

    def spend_all(self, bond_i, bond_j, costs):
        """Deduct costs for an array of bonds."""
        for i, j, c in zip(bond_i, bond_j, costs):
            self.spend(int(i), int(j), float(c))


class NoBudget(BaseBudget):
    """Always returns inf budget. Used for Experiment 0 (no gating)."""

    def update(self, spins_before, spins_after, J_nk, T):
        pass

    def get_budget(self, i, j):
        return float('inf')

    def spend(self, i, j, cost):
        pass

    def get_all_budgets_for_bonds(self, bond_i, bond_j):
        return jnp.full(len(bond_i), jnp.inf)

    def spend_all(self, bond_i, bond_j, costs):
        pass


class BondBudget(BaseBudget):
    """Per-bond budget from ordering events (eq. 11).

    dB_ij/dt = alpha * max(0, Delta(s_i * s_j)) - C(delta_J_ij)

    State stored as JAX array (N, K). All operations GPU-accelerated.
    """

    def __init__(self, neighbors, mask, alpha=0.1):
        self.neighbors = jnp.asarray(neighbors)
        self.mask = jnp.asarray(mask, dtype=jnp.float32)
        self.alpha = alpha
        self.N, self.K = neighbors.shape
        self._budget = jnp.zeros((self.N, self.K), dtype=jnp.float32)

    def update(self, spins_before, spins_after, J_nk, T):
        s_before = _prep_spins(spins_before)
        s_after = _prep_spins(spins_after)
        ordering = _compute_ordering(s_before, s_after,
                                     self.neighbors, self.mask)
        self._budget = self._budget + self.alpha * ordering

    def get_budget(self, i, j):
        for k in range(self.K):
            if self.mask[i, k] and self.neighbors[i, k] == j:
                return max(0.0, float(self._budget[i, k]))
        return 0.0

    def spend(self, i, j, cost):
        for k in range(self.K):
            if self.mask[i, k] and self.neighbors[i, k] == j:
                new_val = max(0.0, float(self._budget[i, k]) - cost)
                self._budget = self._budget.at[i, k].set(new_val)
                return

    def get_budget_array(self):
        """Return full budget array (N, K) as numpy."""
        return np.maximum(0.0, np.asarray(self._budget))

    def get_all_budgets_for_bonds_nk(self, sel_i, sel_k):
        """Get budgets by (site, slot) indices. Returns JAX array."""
        return jnp.maximum(0.0, self._budget[sel_i, sel_k])

    def get_all_budgets_for_bonds(self, bond_i, bond_j):
        """Get budgets by (site_i, site_j). Returns numpy array.

        Falls back to per-bond lookup since we need to find the slot.
        For hot-path code, use get_all_budgets_for_bonds_nk with
        precomputed slot indices instead.
        """
        return np.array([self.get_budget(int(i), int(j))
                         for i, j in zip(bond_i, bond_j)])

    def spend_all_nk(self, sel_i, sel_k, costs, apply_mask):
        """Vectorized spend by (site, slot) indices (GPU).

        Parameters
        ----------
        sel_i, sel_k : array (n,) int
        costs : array (n,) float
        apply_mask : array (n,) bool — which bonds to spend on
        """
        # Use JAX scatter-subtract
        spend_amounts = jnp.where(apply_mask, costs, 0.0)
        self._budget = self._budget.at[sel_i, sel_k].add(-spend_amounts)
        self._budget = jnp.maximum(0.0, self._budget)


class NeighbourhoodBudget(BaseBudget):
    """Site budgets pooled over neighbourhood (eq. 12).

    B_i^nbhd = B_i + gamma * sum_{j~i} B_j
    Budget for bond (i,j) = min(B_i^nbhd, B_j^nbhd).

    State stored as JAX array (N,).
    """

    def __init__(self, neighbors, mask, alpha=0.1, gamma=0.25):
        self.neighbors = jnp.asarray(neighbors)
        self.mask = jnp.asarray(mask, dtype=jnp.float32)
        self.alpha = alpha
        self.gamma = gamma
        self.N, self.K = neighbors.shape
        self._budget = jnp.zeros(self.N, dtype=jnp.float32)

    def update(self, spins_before, spins_after, J_nk, T):
        s_before = _prep_spins(spins_before)
        s_after = _prep_spins(spins_after)
        ordering = _compute_ordering(s_before, s_after,
                                     self.neighbors, self.mask)
        self._budget = self._budget + self.alpha * ordering.sum(axis=1)

    def _neighbourhood_budget(self):
        nbr_budgets = self._budget[self.neighbors] * self.mask
        return self._budget + self.gamma * nbr_budgets.sum(axis=1)

    def get_budget(self, i, j):
        nbhd = self._neighbourhood_budget()
        return max(0.0, float(jnp.minimum(nbhd[i], nbhd[j])))

    def spend(self, i, j, cost):
        half = cost / 2.0
        new_i = max(0.0, float(self._budget[i]) - half)
        new_j = max(0.0, float(self._budget[j]) - half)
        self._budget = self._budget.at[i].set(new_i)
        self._budget = self._budget.at[j].set(new_j)

    def get_all_budgets_for_bonds(self, bond_i, bond_j):
        nbhd = self._neighbourhood_budget()
        return jnp.minimum(nbhd[bond_i], nbhd[bond_j])

    def spend_all(self, bond_i, bond_j, costs):
        """Vectorized spend via JAX scatter."""
        half = jnp.asarray(costs, dtype=jnp.float32) / 2.0
        self._budget = self._budget.at[bond_i].add(-half)
        self._budget = self._budget.at[bond_j].add(-half)
        self._budget = jnp.maximum(0.0, self._budget)


class DiffusingBudget(BaseBudget):
    """Reaction-diffusion budget field on the lattice (eq. 13).

    d mu_i/dt = D * nabla^2 mu_i + eta_i - mu_i/tau_mu - R_i

    State stored as JAX array (N,). Laplacian computed on GPU.
    """

    def __init__(self, neighbors, mask, alpha=0.1, D=0.1, tau_mu=20.0):
        self.neighbors = jnp.asarray(neighbors)
        self.mask = jnp.asarray(mask, dtype=jnp.float32)
        self.alpha = alpha
        self.D = D
        self.tau_mu = tau_mu
        self.N, self.K = neighbors.shape
        self._mu = jnp.zeros(self.N, dtype=jnp.float32)

        # Precompute effective neighbor count (constant)
        self._K_eff = jnp.sum(self.mask, axis=1)

    def update(self, spins_before, spins_after, J_nk, T):
        s_before = _prep_spins(spins_before)
        s_after = _prep_spins(spins_after)
        ordering = _compute_ordering(s_before, s_after,
                                     self.neighbors, self.mask)
        eta = self.alpha * ordering.sum(axis=1)

        # Discrete Laplacian (GPU)
        nbr_mu = self._mu[self.neighbors] * self.mask
        laplacian = nbr_mu.sum(axis=1) - self._K_eff * self._mu

        self._mu = self._mu + self.D * laplacian + eta - self._mu / self.tau_mu
        self._mu = jnp.maximum(self._mu, 0.0)

    def get_budget(self, i, j):
        return max(0.0, float(jnp.minimum(self._mu[i], self._mu[j])))

    def spend(self, i, j, cost):
        half = cost / 2.0
        new_i = max(0.0, float(self._mu[i]) - half)
        new_j = max(0.0, float(self._mu[j]) - half)
        self._mu = self._mu.at[i].set(new_i)
        self._mu = self._mu.at[j].set(new_j)

    def get_all_budgets_for_bonds(self, bond_i, bond_j):
        return jnp.minimum(
            jnp.maximum(0.0, self._mu[bond_i]),
            jnp.maximum(0.0, self._mu[bond_j])
        )

    def spend_all(self, bond_i, bond_j, costs):
        half = jnp.asarray(costs, dtype=jnp.float32) / 2.0
        self._mu = self._mu.at[bond_i].add(-half)
        self._mu = self._mu.at[bond_j].add(-half)
        self._mu = jnp.maximum(0.0, self._mu)

    def get_field(self):
        """Return the chemical potential field (N,) as numpy."""
        return np.maximum(0.0, np.asarray(self._mu))