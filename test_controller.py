"""Tests for controller.py — Phase 2 verification.

Checks:
1. Round-trip: set_params(get_params()) leaves weights identical.
2. Output always in [-delta_J_max, delta_J_max] for 10,000 random inputs.
"""

import numpy as np
import pytest

from controller import LocalController, LocalMagnetisationTracker


def test_round_trip():
    """set_params(get_params()) leaves all weights identical."""
    ctrl = LocalController(delta_J_max=0.1, hidden_size=8)

    # Set random weights
    rng = np.random.default_rng(42)
    params = rng.standard_normal(ctrl.n_params).astype(np.float32)
    ctrl.set_params(params)

    # Round-trip
    retrieved = ctrl.get_params()
    ctrl.set_params(retrieved)
    final = ctrl.get_params()

    np.testing.assert_array_equal(params, retrieved)
    np.testing.assert_array_equal(retrieved, final)


def test_output_bounds():
    """Output is always in [-delta_J_max, delta_J_max] for 10,000 random inputs."""
    delta_J_max = 0.1
    ctrl = LocalController(delta_J_max=delta_J_max, hidden_size=8)

    rng = np.random.default_rng(123)

    # Set random weights (large to stress-test saturation)
    params = rng.standard_normal(ctrl.n_params).astype(np.float32) * 5.0
    ctrl.set_params(params)

    # Generate 10,000 random inputs
    x = rng.standard_normal((10000, 5)).astype(np.float32)
    out = ctrl.forward(x)

    assert out.shape == (10000, 1)
    assert np.all(out >= -delta_J_max), f"Min output: {out.min()}"
    assert np.all(out <= delta_J_max), f"Max output: {out.max()}"


def test_param_count():
    """Parameter count matches architecture."""
    ctrl = LocalController(delta_J_max=0.1, hidden_size=8)
    # 5*8 + 8 + 8*8 + 8 + 8*1 + 1 = 40 + 8 + 64 + 8 + 8 + 1 = 129
    assert ctrl.n_params == 129


def test_propose_updates():
    """propose_updates returns correct shape."""
    ctrl = LocalController(delta_J_max=0.1, hidden_size=8)
    rng = np.random.default_rng(42)
    ctrl.set_params(rng.standard_normal(ctrl.n_params).astype(np.float32))

    n_bonds = 50
    s_i = rng.choice([-1, 1], size=n_bonds).astype(np.float32)
    s_j = rng.choice([-1, 1], size=n_bonds).astype(np.float32)
    m_bar = rng.standard_normal(n_bonds).astype(np.float32) * 0.5
    budget_norm = rng.uniform(-1, 1, size=n_bonds).astype(np.float32)

    delta_J = ctrl.propose_updates(s_i, s_j, m_bar, T_norm=0.5, budget_norm=budget_norm)
    assert delta_J.shape == (n_bonds,)
    assert np.all(np.abs(delta_J) <= 0.1)


def test_magnetisation_tracker():
    """LocalMagnetisationTracker EMA works correctly."""
    n = 100
    alpha = 0.1
    tracker = LocalMagnetisationTracker(n, alpha=alpha)

    # Initial values should be zero
    np.testing.assert_array_equal(tracker.get(), np.zeros(n))

    # After one update with all +1
    spins = np.ones(n, dtype=np.float32)
    tracker.update(spins)
    expected = alpha * 1.0  # 0 * (1-alpha) + 1 * alpha
    np.testing.assert_allclose(tracker.get(), np.full(n, expected))

    # After another update with all -1
    spins = -np.ones(n, dtype=np.float32)
    tracker.update(spins)
    expected2 = alpha * (-1.0) + (1 - alpha) * expected
    np.testing.assert_allclose(tracker.get(), np.full(n, expected2), rtol=1e-6)


def test_magnetisation_tracker_batched():
    """Tracker handles batched spins by averaging over batch."""
    n = 50
    tracker = LocalMagnetisationTracker(n, alpha=0.1)

    # Batched input: mean of two opposite chains should be ~0
    spins = np.array([np.ones(n), -np.ones(n)], dtype=np.float32)  # (2, N)
    tracker.update(spins)
    np.testing.assert_allclose(tracker.get(), np.zeros(n), atol=1e-7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])