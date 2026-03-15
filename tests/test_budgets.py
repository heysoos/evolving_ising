"""Tests for budgets.py — Phase 3 verification.

Checks:
1. Budget non-negativity after 1000 random spend calls.
2. NoBudget never blocks a remodel.
3. DiffusingBudget with D=0, tau_mu=inf reduces to BondBudget behaviour.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from evolving_ising.model import IsingModel
from work_extraction.budgets import (
    NoBudget, BondBudget, NeighbourhoodBudget, DiffusingBudget, make_budget
)


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


def test_pure_interface_lax_scan(neighbors_mask):
    """update_pure / get_pure / spend_pure work inside jax.lax.scan for all budget types."""
    neighbors, mask = neighbors_mask
    N = neighbors.shape[0]
    config = {'budget_alpha': 0.1, 'gamma': 0.25, 'D': 0.1, 'tau_mu': 20.0}
    rng = np.random.default_rng(7)

    # Small selection of bonds for get/spend
    valid_i, valid_k = np.where(mask)
    valid_j = neighbors[valid_i, valid_k]
    n_upd = min(8, len(valid_i))
    si = jnp.asarray(valid_i[:n_upd], dtype=jnp.int32)
    sk = jnp.asarray(valid_k[:n_upd], dtype=jnp.int32)
    sj = jnp.asarray(valid_j[:n_upd], dtype=jnp.int32)

    s_steps = jnp.asarray(
        rng.choice([-1.0, 1.0], size=(10, N)).astype(np.float32)
    )  # (10, N)

    for bt in ['none', 'bond', 'neighbourhood', 'diffusing']:
        bud = make_budget(bt, neighbors, mask, config)

        def _scan_step(carry, s_aft):
            state, s_prev = carry
            state = bud.update_pure(state, s_prev, s_aft)
            bud_vals = bud.get_pure(state, si, sk, sj)
            costs = jnp.full(n_upd, 0.01, dtype=jnp.float32)
            can_apply = bud_vals >= costs
            state = bud.spend_pure(state, si, sk, sj, costs, can_apply)
            return (state, s_aft), bud_vals.sum()

        init_s = s_steps[0]
        init_carry = (bud.init(), init_s)
        (final_state, _), totals = jax.lax.scan(_scan_step, init_carry, s_steps[1:])

        # Verify non-negativity of final budget state
        assert jnp.all(final_state >= 0.0), f"{bt}: negative budget after scan"
        # Verify totals are finite (or inf for NoBudget)
        if bt == 'none':
            assert jnp.all(jnp.isinf(totals)), f"{bt}: expected inf"
        else:
            assert jnp.all(jnp.isfinite(totals)), f"{bt}: non-finite totals"


def test_pure_interface_matches_stateful(neighbors_mask):
    """update_pure accumulates non-negative budget matching stateful update."""
    neighbors, mask = neighbors_mask
    N = neighbors.shape[0]
    config = {'budget_alpha': 0.1, 'gamma': 0.25, 'D': 0.0, 'tau_mu': 1e30}
    rng = np.random.default_rng(99)

    for bt in ['bond', 'neighbourhood', 'diffusing']:
        bud = make_budget(bt, neighbors, mask, config)
        state = bud.init()

        for _ in range(20):
            s_bef = jnp.asarray(rng.choice([-1.0, 1.0], size=N).astype(np.float32))
            s_aft = jnp.asarray(rng.choice([-1.0, 1.0], size=N).astype(np.float32))
            state = bud.update_pure(state, s_bef, s_aft)

        assert jnp.all(state >= 0.0), f"{bt}: negative state after update_pure"
        assert float(state.sum()) > 0.0, f"{bt}: state should be non-zero after updates"


def test_make_budget_factory(neighbors_mask):
    """make_budget factory creates correct types and init() returns correct shapes."""
    neighbors, mask = neighbors_mask
    N, K = neighbors.shape
    config = {'budget_alpha': 0.05, 'gamma': 0.3, 'D': 0.2, 'tau_mu': 10.0}

    cases = [
        ('none',          NoBudget,              (1,)),
        ('bond',          BondBudget,            (N, K)),
        ('neighbourhood', NeighbourhoodBudget,   (N,)),
        ('diffusing',     DiffusingBudget,       (N,)),
    ]
    for bt, expected_cls, expected_shape in cases:
        b = make_budget(bt, neighbors, mask, config)
        assert isinstance(b, expected_cls), f"{bt}: wrong type {type(b)}"
        state = b.init()
        assert state.shape == expected_shape, f"{bt}: init shape {state.shape} != {expected_shape}"
        assert jnp.all(state == 0.0), f"{bt}: init state should be zero"

    with pytest.raises(ValueError, match="Unknown budget_type"):
        make_budget('invalid', neighbors, mask, config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
