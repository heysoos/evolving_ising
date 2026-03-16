"""Experiment 2: Neighbourhood Budget.

Same setup as Experiment 1 but use NeighbourhoodBudget.
Sweep gamma over {0.0, 0.1, 0.25, 0.5, 1.0} at best (lambda, alpha)
from Experiment 1. Additionally sweep tau at best gamma.

Expected: Non-monotonic W_net vs. gamma with a peak at some gamma*.
"""

import os
import numpy as np
from work_extraction.train import run_experiment, DEFAULT_CONFIG


def run_exp2(best_lambda=0.01, best_alpha=0.1, config=None,
             results_dir='../results/exp2', n_generations=500, resume=False):
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
            resume=resume,
        )
        results[('gamma', gamma)] = result

        try:
            import os as _os, sys as _sys
            _HERE = _os.path.dirname(_os.path.abspath(__file__))
            if _HERE not in _sys.path:
                _sys.path.insert(0, _HERE)
            from report_utils import generate_training_report
            generate_training_report(results_dir)
        except Exception:
            pass

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
            resume=resume,
        )
        results[('tau', tau)] = result

        try:
            import os as _os, sys as _sys
            _HERE = _os.path.dirname(_os.path.abspath(__file__))
            if _HERE not in _sys.path:
                _sys.path.insert(0, _HERE)
            from report_utils import generate_training_report
            generate_training_report(results_dir)
        except Exception:
            pass

    return results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Experiment 2: Neighbourhood Budget sweep')
    p.add_argument('--results-dir', default='../results/exp2')
    p.add_argument('--n-generations', type=int, default=500)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--auto-report', action='store_true')
    p.add_argument('--no-animate', action='store_true')
    args = p.parse_args()

    results = run_exp2(results_dir=args.results_dir, n_generations=args.n_generations,
                       resume=args.resume)

    if args.auto_report:
        import os, sys
        _HERE = os.path.dirname(os.path.abspath(__file__))
        if _HERE not in sys.path:
            sys.path.insert(0, _HERE)
        from exp2_report import generate_report
        generate_report(args.results_dir, animate=not args.no_animate)