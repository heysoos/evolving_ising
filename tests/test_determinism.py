"""Tests that run_experiment produces identical results across repeated runs."""

import tempfile
import numpy as np
import pytest

from work_extraction.train import run_experiment, DEFAULT_CONFIG

SMALL_CFG = {
    **DEFAULT_CONFIG,
    'L': 8,
    'n_generations': 5,
    'n_eval_cycles': 3,
    'n_eval_chains': 2,
    'pop_size': 6,
    'warmup_sweeps': 10,
    'steps_per_cycle': 20,
    'log_interval': 1,
}


def _run(budget_type='none'):
    with tempfile.TemporaryDirectory() as d:
        r = run_experiment(SMALL_CFG, budget_type=budget_type,
                           name='det_test', results_dir=d, verbose=False)
    return r.training_log


@pytest.mark.parametrize('budget_type', ['none', 'bond'])
def test_deterministic(budget_type):
    """Two identical runs must produce bit-for-bit identical fitness trajectories."""
    log1 = _run(budget_type)
    log2 = _run(budget_type)

    np.testing.assert_array_equal(
        log1['best_fitness'], log2['best_fitness'],
        err_msg=f"best_fitness differs between runs (budget={budget_type})",
    )
    np.testing.assert_array_equal(
        log1['mean_fitness'], log2['mean_fitness'],
        err_msg=f"mean_fitness differs between runs (budget={budget_type})",
    )
    np.testing.assert_array_equal(
        log1['sigma'], log2['sigma'],
        err_msg=f"sigma differs between runs (budget={budget_type})",
    )
