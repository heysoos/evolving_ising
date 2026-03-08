"""Budget classes for gating bond remodelling (Phase 3).

Three budget strategies with a shared BaseBudget interface:
- BondBudget: per-bond accumulation from ordering events (eq. 11)
- NeighbourhoodBudget: site budgets pooled over neighbourhood (eq. 12)
- DiffusingBudget: reaction-diffusion field on the lattice (eq. 13)
- NoBudget: always returns inf (used for Experiment 0)
"""

import numpy as np


class BaseBudget:
    """Base class for budget strategies."""

    def update(self, spins_before, spins_after, J_nk, T):
        """Update budgets based on spin changes.

        Parameters
        ----------
        spins_before, spins_after : array (N,) or (B, N)
            Spin configurations before and after Metropolis sweeps.
            Converted to numpy and averaged over batch if needed.
        J_nk : array (N, K)
            Current coupling array.
        T : float
            Current bath temperature.
        """
        raise NotImplementedError

    def get_budget(self, i, j):
        """Return available budget for bond between sites i and j.

        Parameters
        ----------
        i, j : int
            Site indices.

        Returns
        -------
        float : available budget (non-negative).
        """
        raise NotImplementedError

    def spend(self, i, j, cost):
        """Deduct cost from the budget for bond (i, j).

        Parameters
        ----------
        i, j : int
            Site indices.
        cost : float
            Amount to deduct (non-negative).
        """
        raise NotImplementedError

    def get_all_budgets_for_bonds(self, bond_i, bond_j):
        """Return budgets for an array of bonds.

        Parameters
        ----------
        bond_i, bond_j : array (n_bonds,)
            Site indices for each bond.

        Returns
        -------
        budgets : array (n_bonds,)
        """
        return np.array([self.get_budget(int(i), int(j))
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
        return np.full(len(bond_i), float('inf'))


class BondBudget(BaseBudget):
    """Per-bond budget from ordering events (eq. 11).

    dB_ij/dt = alpha * max(0, Delta(s_i * s_j)) - C(delta_J_ij)

    Bonds accumulate budget when their spin correlation increases.

    Parameters
    ----------
    neighbors : array (N, K)
        Neighbor indices from IsingModel.
    mask : array (N, K)
        Valid neighbor mask from IsingModel.
    alpha : float
        Budget accumulation rate.
    """

    def __init__(self, neighbors, mask, alpha=0.1):
        self.neighbors = np.asarray(neighbors)
        self.mask = np.asarray(mask)
        self.alpha = alpha
        self.N, self.K = neighbors.shape

        # Per-bond budget: stored as (N, K) matching J_nk layout
        self._budget = np.zeros((self.N, self.K), dtype=np.float64)

    def update(self, spins_before, spins_after, J_nk, T):
        s_before = np.asarray(spins_before, dtype=np.float32)
        s_after = np.asarray(spins_after, dtype=np.float32)

        if s_before.ndim == 2:
            s_before = s_before.mean(axis=0)
        if s_after.ndim == 2:
            s_after = s_after.mean(axis=0)

        # Compute correlation change for each bond
        for i in range(self.N):
            for k in range(self.K):
                if not self.mask[i, k]:
                    continue
                j = self.neighbors[i, k]
                corr_before = s_before[i] * s_before[j]
                corr_after = s_after[i] * s_after[j]
                delta_corr = corr_after - corr_before

                # Ordering event: correlation increased
                if delta_corr > 0:
                    self._budget[i, k] += self.alpha * delta_corr

    def get_budget(self, i, j):
        # Find bond slot for (i, j)
        for k in range(self.K):
            if self.mask[i, k] and self.neighbors[i, k] == j:
                return max(0.0, self._budget[i, k])
        return 0.0

    def spend(self, i, j, cost):
        for k in range(self.K):
            if self.mask[i, k] and self.neighbors[i, k] == j:
                self._budget[i, k] = max(0.0, self._budget[i, k] - cost)
                return

    def get_budget_array(self):
        """Return full budget array (N, K)."""
        return np.maximum(0.0, self._budget.copy())

    def update_vectorized(self, spins_before, spins_after, J_nk, T):
        """Vectorized version of update for performance."""
        s_before = np.asarray(spins_before, dtype=np.float32)
        s_after = np.asarray(spins_after, dtype=np.float32)

        if s_before.ndim == 2:
            s_before = s_before.mean(axis=0)
        if s_after.ndim == 2:
            s_after = s_after.mean(axis=0)

        # Vectorized correlation computation
        nbr_before = s_before[self.neighbors]  # (N, K)
        nbr_after = s_after[self.neighbors]    # (N, K)

        corr_before = s_before[:, None] * nbr_before  # (N, K)
        corr_after = s_after[:, None] * nbr_after      # (N, K)

        delta_corr = corr_after - corr_before  # (N, K)
        ordering = np.maximum(0.0, delta_corr) * self.mask  # (N, K)

        self._budget += self.alpha * ordering


class NeighbourhoodBudget(BaseBudget):
    """Site budgets pooled over neighbourhood (eq. 12).

    B_i^nbhd = B_i + gamma * sum_{j~i} B_j

    Budget for bond (i,j) = min(B_i^nbhd, B_j^nbhd).

    Parameters
    ----------
    neighbors : array (N, K)
    mask : array (N, K)
    alpha : float
        Budget accumulation rate.
    gamma : float
        Neighbour sharing weight (0 = isolated, 1 = strong sharing).
    """

    def __init__(self, neighbors, mask, alpha=0.1, gamma=0.25):
        self.neighbors = np.asarray(neighbors)
        self.mask = np.asarray(mask)
        self.alpha = alpha
        self.gamma = gamma
        self.N, self.K = neighbors.shape

        # Per-site budget
        self._budget = np.zeros(self.N, dtype=np.float64)

    def update(self, spins_before, spins_after, J_nk, T):
        s_before = np.asarray(spins_before, dtype=np.float32)
        s_after = np.asarray(spins_after, dtype=np.float32)

        if s_before.ndim == 2:
            s_before = s_before.mean(axis=0)
        if s_after.ndim == 2:
            s_after = s_after.mean(axis=0)

        # Accumulate from ordering events at each site
        nbr_before = s_before[self.neighbors]
        nbr_after = s_after[self.neighbors]

        corr_before = s_before[:, None] * nbr_before
        corr_after = s_after[:, None] * nbr_after

        delta_corr = corr_after - corr_before
        ordering = np.maximum(0.0, delta_corr) * self.mask

        # Sum ordering events over all bonds of each site
        self._budget += self.alpha * ordering.sum(axis=1)

    def _neighbourhood_budget(self):
        """Compute pooled neighbourhood budgets."""
        nbr_budgets = self._budget[self.neighbors] * self.mask  # (N, K)
        return self._budget + self.gamma * nbr_budgets.sum(axis=1)

    def get_budget(self, i, j):
        nbhd = self._neighbourhood_budget()
        return max(0.0, min(nbhd[i], nbhd[j]))

    def spend(self, i, j, cost):
        # Split cost equally between the two sites
        half = cost / 2.0
        self._budget[i] = max(0.0, self._budget[i] - half)
        self._budget[j] = max(0.0, self._budget[j] - half)

    def get_all_budgets_for_bonds(self, bond_i, bond_j):
        nbhd = self._neighbourhood_budget()
        return np.minimum(nbhd[bond_i], nbhd[bond_j])


class DiffusingBudget(BaseBudget):
    """Reaction-diffusion budget field on the lattice (eq. 13).

    d mu_i/dt = D * nabla^2 mu_i + eta_i - mu_i/tau_mu - R_i

    Parameters
    ----------
    neighbors : array (N, K)
    mask : array (N, K)
    alpha : float
        Source strength from ordering events.
    D : float
        Diffusion coefficient.
    tau_mu : float
        Decay timescale.
    """

    def __init__(self, neighbors, mask, alpha=0.1, D=0.1, tau_mu=20.0):
        self.neighbors = np.asarray(neighbors)
        self.mask = np.asarray(mask)
        self.alpha = alpha
        self.D = D
        self.tau_mu = tau_mu
        self.N, self.K = neighbors.shape

        # Per-site chemical potential field
        self._mu = np.zeros(self.N, dtype=np.float64)

    def update(self, spins_before, spins_after, J_nk, T):
        s_before = np.asarray(spins_before, dtype=np.float32)
        s_after = np.asarray(spins_after, dtype=np.float32)

        if s_before.ndim == 2:
            s_before = s_before.mean(axis=0)
        if s_after.ndim == 2:
            s_after = s_after.mean(axis=0)

        # Source: ordering events
        nbr_before = s_before[self.neighbors]
        nbr_after = s_after[self.neighbors]

        corr_before = s_before[:, None] * nbr_before
        corr_after = s_after[:, None] * nbr_after

        delta_corr = corr_after - corr_before
        ordering = np.maximum(0.0, delta_corr) * self.mask
        eta = self.alpha * ordering.sum(axis=1)  # (N,)

        # Discrete Laplacian: nabla^2 mu_i = sum_{j~i} mu_j - K_eff * mu_i
        nbr_mu = self._mu[self.neighbors] * self.mask  # (N, K)
        K_eff = self.mask.sum(axis=1)  # actual number of valid neighbors
        laplacian = nbr_mu.sum(axis=1) - K_eff * self._mu

        # Update: d mu/dt = D * nabla^2 mu + eta - mu/tau_mu
        self._mu += self.D * laplacian + eta - self._mu / self.tau_mu

        # Enforce non-negativity
        np.maximum(self._mu, 0.0, out=self._mu)

    def get_budget(self, i, j):
        return max(0.0, min(self._mu[i], self._mu[j]))

    def spend(self, i, j, cost):
        half = cost / 2.0
        self._mu[i] = max(0.0, self._mu[i] - half)
        self._mu[j] = max(0.0, self._mu[j] - half)

    def get_all_budgets_for_bonds(self, bond_i, bond_j):
        return np.minimum(
            np.maximum(0.0, self._mu[bond_i]),
            np.maximum(0.0, self._mu[bond_j])
        )

    def get_field(self):
        """Return the chemical potential field (N,)."""
        return np.maximum(0.0, self._mu.copy())