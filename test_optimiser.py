"""Tests for optimiser.py — Phase 4 verification.

Check: On a convex toy function (negative L2 norm), the optimiser
reaches within 1% of the optimum within 200 generations.
"""

import numpy as np
import pytest

from optimiser import WorkExtractionES


def test_cma_on_sphere():
    """CMA-ES reaches near-optimum on negative sphere function.

    Fitness = -||x||^2, optimum at x=0 with fitness 0.
    """
    dim = 10
    es = WorkExtractionES(n_params=dim, pop_size=20, sigma=1.0, seed=42)

    for gen in range(200):
        params_list = es.ask()
        fitnesses = [-np.sum(p**2) for p in params_list]
        es.tell(params_list, fitnesses)

    best = es.best_params
    best_fitness = -np.sum(best**2)

    print(f"Best fitness after 200 gens: {best_fitness:.6f}")
    print(f"Best params norm: {np.linalg.norm(best):.6f}")

    # Should be very close to 0
    assert best_fitness > -0.01, \
        f"Optimizer didn't converge: best_fitness={best_fitness}"


def test_history_tracking():
    """History records mean_fitness, best_fitness, sigma per generation."""
    dim = 5
    es = WorkExtractionES(n_params=dim, pop_size=10, sigma=0.5, seed=0)

    for _ in range(10):
        params_list = es.ask()
        fitnesses = [-np.sum(p**2) for p in params_list]
        es.tell(params_list, fitnesses)

    assert len(es.history) == 10
    for entry in es.history:
        assert 'mean_fitness' in entry
        assert 'best_fitness' in entry
        assert 'sigma' in entry
        assert entry['best_fitness'] >= entry['mean_fitness']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])