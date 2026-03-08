"""Experiment 2: Neighbourhood Budget.

Same setup as Experiment 1 but use NeighbourhoodBudget.
Sweep gamma over {0.0, 0.1, 0.25, 0.5, 1.0} at best (lambda, alpha)
from Experiment 1. Additionally sweep tau at best gamma.

Expected: Non-monotonic W_net vs. gamma with a peak at some gamma*.
"""

import os
import numpy as np
from train import run_experiment, DEFAULT_CONFIG


def run_exp2(best_lambda=0.01, best_alpha=0.1, config=None,
             results_dir='results/exp2', n_generations=500):
    """Run Experiment 2: Neighbourhood Budget sweep."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    T_mean = cfg['T_mean']
    J_init = T_mean / 2.269

    gamma_values = [0.0, 0.1, 0.25, 0.5, 1.0]

    results = {}

    # Gamma sweep
    for gamma in gamma_values:
        name = f"gamma_{gamma:.2f}"
        print(f"\n{'='*60}")
        print(f"Experiment 2: gamma={gamma}")
        print(f"{'='*60}")

        exp_config = {
            **cfg,
            'J_init': J_init,
            'lambda': best_lambda,
            'budget_alpha': best_alpha,
            'gamma': gamma,
            'n_generations': n_generations,
        }

        result = run_experiment(
            config=exp_config,
            budget_type='neighbourhood',
            name=name,
            results_dir=results_dir,
        )
        results[('gamma', gamma)] = result

    # Find best gamma
    best_gamma = max(
        gamma_values,
        key=lambda g: results[('gamma', g)].training_log['best_fitness'].max()
    )
    print(f"\nBest gamma: {best_gamma}")

    # Tau sweep at best gamma
    tau_values = [100, 200, 500]
    for tau in tau_values:
        name = f"gamma_{best_gamma:.2f}_tau_{tau}"
        print(f"\n{'='*60}")
        print(f"Experiment 2: gamma={best_gamma}, tau={tau}")
        print(f"{'='*60}")

        exp_config = {
            **cfg,
            'J_init': J_init,
            'lambda': best_lambda,
            'budget_alpha': best_alpha,
            'gamma': best_gamma,
            'tau': tau,
            'steps_per_cycle': tau,
            'n_generations': n_generations,
        }

        result = run_experiment(
            config=exp_config,
            budget_type='neighbourhood',
            name=name,
            results_dir=results_dir,
        )
        results[('tau', tau)] = result

    return results


if __name__ == '__main__':
    run_exp2()