"""Tests for budgets.py — Phase 3 verification.

Checks:
1. Budget non-negativity after 1000 random spend calls.
2. NoBudget never blocks a remodel.
3. DiffusingBudget with D=0, tau_mu=inf reduces to BondBudget behaviour.
"""

import numpy as np
import pytest

from evolving_ising.model import IsingModel
from work_extraction.budgets import NoBudget, BondBudget, NeighbourhoodBudget, DiffusingBudget


@pytest.fixture
def model():
    return IsingModel((8, 8), neighborhood="von_neumann", boundary="periodic")


@pytest.fixture
def neighbors_mask(model):
    return np.asarray(model.neighbors), np.asarray(model.mask)


def test_no_budget_always_inf():
    """NoBudget never blocks a remodel."""
    nb = NoBudget()
    rng = np.random.default_rng(42)

    for _ in range(1000):
        i, j = rng.integers(0, 64, size=2)
        assert nb.get_budget(i, j) == float('inf')
        nb.spend(i, j, rng.uniform(0, 100))
        assert nb.get_budget(i, j) == float('inf')


def test_no_budget_update_noop():
    """NoBudget.update does nothing."""
    nb = NoBudget()
    nb.update(np.ones(64), -np.ones(64), np.ones((64, 4)), 2.5)
    assert nb.get_budget(0, 1) == float('inf')


def test_bond_budget_non_negativity(neighbors_mask):
    """Budget >= 0 after 1000 random spend calls."""
    neighbors, mask = neighbors_mask
    bb = BondBudget(neighbors, mask, alpha=0.1)
    rng = np.random.default_rng(42)
    N = neighbors.shape[0]

    # Generate some budget via ordering events
    for _ in range(50):
        s_before = rng.choice([-1, 1], size=N).astype(np.float32)
        s_after = rng.choice([-1, 1], size=N).astype(np.float32)
        bb.update(s_before, s_after, np.ones_like(neighbors, dtype=np.float32), 2.5)

    # Spend randomly
    for _ in range(1000):
        i = rng.integers(0, N)
        k = rng.integers(0, neighbors.shape[1])
        if mask[i, k]:
            j = neighbors[i, k]
            bb.spend(int(i), int(j), rng.uniform(0, 1))

    # Check non-negativity
    budget_arr = bb.get_budget_array()
    assert np.all(budget_arr >= 0), f"Negative budget found: min={budget_arr.min()}"


def test_neighbourhood_budget_non_negativity(neighbors_mask):
    """Budget >= 0 after 1000 random spend calls."""
    neighbors, mask = neighbors_mask
    nb = NeighbourhoodBudget(neighbors, mask, alpha=0.1, gamma=0.25)
    rng = np.random.default_rng(42)
    N = neighbors.shape[0]

    # Generate some budget
    for _ in range(50):
        s_before = rng.choice([-1, 1], size=N).astype(np.float32)
        s_after = rng.choice([-1, 1], size=N).astype(np.float32)
        nb.update(s_before, s_after, np.ones_like(neighbors, dtype=np.float32), 2.5)

    # Spend randomly
    for _ in range(1000):
        i = rng.integers(0, N)
        k = rng.integers(0, neighbors.shape[1])
        if mask[i, k]:
            j = neighbors[i, k]
            nb.spend(int(i), int(j), rng.uniform(0, 1))

    # Check non-negativity via get_budget
    for i in range(N):
        for k in range(neighbors.shape[1]):
            if mask[i, k]:
                j = neighbors[i, k]
                assert nb.get_budget(int(i), int(j)) >= 0


def test_diffusing_budget_non_negativity(neighbors_mask):
    """Budget >= 0 after 1000 random spend calls."""
    neighbors, mask = neighbors_mask
    db = DiffusingBudget(neighbors, mask, alpha=0.1, D=0.1, tau_mu=20.0)
    rng = np.random.default_rng(42)
    N = neighbors.shape[0]

    # Generate some budget
    for _ in range(50):
        s_before = rng.choice([-1, 1], size=N).astype(np.float32)
        s_after = rng.choice([-1, 1], size=N).astype(np.float32)
        db.update(s_before, s_after, np.ones_like(neighbors, dtype=np.float32), 2.5)

    # Spend randomly
    for _ in range(1000):
        i = rng.integers(0, N)
        k = rng.integers(0, neighbors.shape[1])
        if mask[i, k]:
            j = neighbors[i, k]
            db.spend(int(i), int(j), rng.uniform(0, 1))

    # Check non-negativity
    field = db.get_field()
    assert np.all(field >= 0), f"Negative field found: min={field.min()}"


def test_diffusing_no_diffusion_no_decay(neighbors_mask):
    """DiffusingBudget with D=0, tau_mu=inf behaves like BondBudget.

    With no diffusion and no decay, the source accumulation should match
    the BondBudget accumulation pattern (local ordering events only).
    """
    neighbors, mask = neighbors_mask
    mask_np = np.asarray(mask)
    N = neighbors.shape[0]

    db = DiffusingBudget(neighbors, mask, alpha=0.1, D=0.0, tau_mu=float('inf'))
    bb = BondBudget(neighbors, mask, alpha=0.1)

    rng = np.random.default_rng(99)

    # Run the same sequence of updates
    for _ in range(20):
        s_before = rng.choice([-1, 1], size=N).astype(np.float32)
        s_after = rng.choice([-1, 1], size=N).astype(np.float32)
        J_dummy = np.ones_like(neighbors, dtype=np.float32)

        db.update(s_before, s_after, J_dummy, 2.5)
        bb.update(s_before, s_after, J_dummy, 2.5)

    # DiffusingBudget stores per-site, BondBudget stores per-bond.
    # With D=0, tau_mu=inf: mu_i accumulates sum of ordering events at site i.
    # Both should be non-negative and correlated.
    db_field = db.get_field()
    bb_array = bb.get_budget_array()

    assert np.all(db_field >= 0)
    assert np.all(bb_array >= 0)

    # Sites with high bond budget should have high diffusing budget
    site_bb = (bb_array * mask_np).sum(axis=1)
    corr = np.corrcoef(db_field, site_bb)[0, 1]
    print(f"Correlation between DiffusingBudget field and BondBudget site totals: {corr:.4f}")
    assert corr > 0.9, f"Budgets should be highly correlated: corr={corr}"


def test_bond_budget_update_is_vectorized(neighbors_mask):
    """Verify update correctly accumulates from ordering events."""
    neighbors, mask = neighbors_mask
    N = neighbors.shape[0]

    bb = BondBudget(neighbors, mask, alpha=0.1)

    # All aligned before, mixed after — should produce ordering events
    s_before = -np.ones(N, dtype=np.float32)
    s_after = np.ones(N, dtype=np.float32)

    bb.update(s_before, s_after, np.ones_like(neighbors, dtype=np.float32), 2.5)

    # All bonds should have increased budget (correlation went from -1*-1=1
    # to 1*1=1, so delta_corr = 0, no ordering for same-sign flips)
    # Actually: s_before=-1, so corr_before = (-1)*(-1) = 1
    # s_after=+1, so corr_after = (+1)*(+1) = 1
    # delta_corr = 0 — no ordering event. Let's use a case that generates events.
    bb2 = BondBudget(neighbors, mask, alpha=0.5)
    s_before2 = np.ones(N, dtype=np.float32)
    s_before2[::2] = -1  # alternating spins
    s_after2 = np.ones(N, dtype=np.float32)  # all aligned

    bb2.update(s_before2, s_after2, np.ones_like(neighbors, dtype=np.float32), 2.5)

    budget_arr = bb2.get_budget_array()
    assert budget_arr.sum() > 0, "Should have accumulated some budget"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
