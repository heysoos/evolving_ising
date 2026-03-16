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


def run_exp1(config=None, results_dir='../results/exp1', n_generations=500, resume=False):
    """Run Experiment 1: Bond Budget sweep."""
    import os, sys
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    J_init = 0.92  # T_mean / 2.269

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
                resume=resume,
            )
            results[(lam, alpha)] = result

            # Sweep-level training report after each run
            try:
                _HERE = os.path.dirname(os.path.abspath(__file__))
                if _HERE not in sys.path:
                    sys.path.insert(0, _HERE)
                from report_utils import generate_training_report
                generate_training_report(results_dir)
            except Exception:
                pass

    return results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Experiment 1: Bond Budget sweep')
    p.add_argument('--results-dir', default='../results/exp1')
    p.add_argument('--n-generations', type=int, default=500)
    p.add_argument('--resume', action='store_true',
                   help='Resume from checkpoint (unstarted runs start fresh)')
    p.add_argument('--auto-report', action='store_true',
                   help='Run exp1_report.py after experiment completes')
    p.add_argument('--no-animate', action='store_true',
                   help='Skip GIF animations in auto-report')
    args = p.parse_args()

    results = run_exp1(results_dir=args.results_dir, n_generations=args.n_generations,
                       resume=args.resume)

    if args.auto_report:
        import os, sys
        _HERE = os.path.dirname(os.path.abspath(__file__))
        if _HERE not in sys.path:
            sys.path.insert(0, _HERE)
        from exp1_report import generate_report
        generate_report(args.results_dir, animate=not args.no_animate)