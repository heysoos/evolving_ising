"""Experiment 1: Bond Budget.

Use J0 = T_mean / 2.269 as initial uniform coupling.
Sweep lambda over {0.0, 0.01, 0.1, 0.5} and alpha over {0.05, 0.1, 0.3}.
For each combination run 500 generations of evolutionary optimisation.

Expected: W_net of best individual exceeds W_net_opt from Experiment 0
at low lambda. Increasing lambda degrades performance monotonically.
"""

import os
import numpy as np
from work_extraction.train import run_experiment, DEFAULT_CONFIG


def run_exp1(config=None, results_dir='../results/exp1', n_generations=500):
    """Run Experiment 1: Bond Budget sweep."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    T_mean = cfg['T_mean']
    J_init = T_mean / 2.269

    lambda_values = [0.0, 0.01, 0.1, 0.5]
    alpha_values = [0.05, 0.1, 0.3]

    results = {}

    for lam in lambda_values:
        for alpha in alpha_values:
            name = f"lam_{lam:.2f}_alpha_{alpha:.2f}"
            print(f"\n{'='*60}")
            print(f"Experiment 1: lambda={lam}, alpha={alpha}")
            print(f"{'='*60}")

            exp_config = {
                **cfg,
                'J_init': J_init,
                'lambda': lam,
                'budget_alpha': alpha,
                'n_generations': n_generations,
            }

            result = run_experiment(
                config=exp_config,
                budget_type='bond',
                name=name,
                results_dir=results_dir,
            )
            results[(lam, alpha)] = result

    return results


if __name__ == '__main__':
    run_exp1(results_dir='../results/exp1_batched')