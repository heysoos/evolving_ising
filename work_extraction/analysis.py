"""Analysis and plotting for work extraction experiments (Phase 7).

All functions save figures to figures/ directory.
Run as a script to regenerate all figures from saved results.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .thermodynamics import temperature_schedule_np


FIGURES_DIR = 'figures'


def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def plot_learning_curves(results_dict, exp_name, baseline_ceiling=None,
                         figures_dir=FIGURES_DIR):
    """W_net vs. generation for all runs in an experiment.

    Parameters
    ----------
    results_dict : dict
        Maps run_name -> ExperimentResult or training_log dict.
    exp_name : str
        Experiment name for title.
    baseline_ceiling : float, optional
        W_net_opt from Experiment 0.
    """
    _ensure_dir(figures_dir)
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, result in results_dict.items():
        log = result.training_log if hasattr(result, 'training_log') else result
        ax.plot(log['generation'], log['best_fitness'], label=str(name))

    if baseline_ceiling is not None:
        ax.axhline(baseline_ceiling, color='red', linestyle='--',
                   label=f'Fixed-J ceiling ({baseline_ceiling:.4f})')

    ax.set_xlabel('Generation')
    ax.set_ylabel('W_net (best)')
    ax.set_title(f'Learning Curves — {exp_name}')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f'{exp_name}_learning_curves.png'), dpi=150)
    plt.close()


def plot_J_phase_portrait(J_bar_trace, T_trace, tau, T_mean, delta_T,
                          figures_dir=FIGURES_DIR):
    """J_bar(t) vs T(t) over one cycle.

    Parameters
    ----------
    J_bar_trace : array (steps,)
        Spatially averaged coupling over one cycle.
    T_trace : array (steps,)
        Temperature over one cycle.
    """
    _ensure_dir(figures_dir)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(T_trace, J_bar_trace, 'b-', alpha=0.8)
    ax.plot(T_trace[0], J_bar_trace[0], 'go', markersize=10, label='Start')
    ax.plot(T_trace[-1], J_bar_trace[-1], 'rs', markersize=10, label='End')
    ax.set_xlabel('T(t)')
    ax.set_ylabel('J_bar(t)')
    ax.set_title('J-T Phase Portrait')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'J_phase_portrait.png'), dpi=150)
    plt.close()


def compute_phase_lag(T_trace, J_bar_trace):
    """Cross-correlation phase lag between T(t) and J_bar(t).

    Returns
    -------
    phi : float
        Phase lag in radians.
    """
    T_c = T_trace - T_trace.mean()
    J_c = J_bar_trace - J_bar_trace.mean()

    # Normalise
    T_norm = T_c / (np.std(T_c) + 1e-12)
    J_norm = J_c / (np.std(J_c) + 1e-12)

    # Cross-correlation
    corr = np.correlate(T_norm, J_norm, mode='full')
    lags = np.arange(-len(T_trace) + 1, len(T_trace))

    # Find peak
    peak_idx = np.argmax(corr)
    peak_lag = lags[peak_idx]

    # Convert lag to phase
    tau = len(T_trace)
    phi = 2.0 * np.pi * peak_lag / tau

    return phi


def plot_J_spatial_map(J_mean, L, figures_dir=FIGURES_DIR):
    """Heatmap of time-averaged J_ij on the lattice.

    Parameters
    ----------
    J_mean : array (N,) or (L, L)
        Time-averaged coupling per site (mean over neighbors).
    L : int
        Lattice side length.
    """
    _ensure_dir(figures_dir)

    if J_mean.ndim == 1:
        J_map = J_mean.reshape(L, L)
    else:
        J_map = J_mean

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(J_map, cmap='viridis', interpolation='nearest')
    ax.set_title('Time-averaged J spatial map')
    plt.colorbar(im, ax=ax, label='<J>')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'J_spatial_map.png'), dpi=150)
    plt.close()

    # Heterogeneity index
    het = np.std(J_map) / np.mean(J_map) if np.mean(J_map) > 0 else 0
    return het


def plot_budget_vs_domain_walls(budget_field, spins, L, figures_dir=FIGURES_DIR):
    """Mean budget at domain wall vs interior sites (eq. 16).

    Parameters
    ----------
    budget_field : array (N,)
        Budget/mu field per site.
    spins : array (N,) or (L, L)
        Spin configuration.
    L : int
    """
    _ensure_dir(figures_dir)

    s = spins.reshape(L, L) if spins.ndim == 1 else spins
    b = budget_field.reshape(L, L) if budget_field.ndim == 1 else budget_field

    # Domain walls: sites where at least one neighbor has opposite spin
    wall_mask = np.zeros_like(s, dtype=bool)
    for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        shifted = np.roll(np.roll(s, di, axis=0), dj, axis=1)
        wall_mask |= (s != shifted)

    wall_budget = b[wall_mask].mean() if wall_mask.any() else 0
    interior_budget = b[~wall_mask].mean() if (~wall_mask).any() else 0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(s, cmap='RdBu', interpolation='nearest')
    axes[0].set_title('Spin configuration')
    im = axes[1].imshow(b, cmap='hot', interpolation='nearest')
    axes[1].set_title('Budget field')
    plt.colorbar(im, ax=axes[1])

    fig.suptitle(f'Wall budget: {wall_budget:.4f}, Interior: {interior_budget:.4f}')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'budget_vs_domain_walls.png'), dpi=150)
    plt.close()

    return wall_budget, interior_budget


def plot_entropy_production_map(sigma_map, L, figures_dir=FIGURES_DIR):
    """Per-site entropy production averaged over cycles.

    Parameters
    ----------
    sigma_map : array (N,) or (L, L)
    L : int
    """
    _ensure_dir(figures_dir)

    s_map = sigma_map.reshape(L, L) if sigma_map.ndim == 1 else sigma_map

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(s_map, cmap='hot', interpolation='nearest')
    ax.set_title('Per-site entropy production')
    plt.colorbar(im, ax=ax, label='sigma_i')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'entropy_production_map.png'), dpi=150)
    plt.close()


def plot_lambda_sweep(results_dict, figures_dir=FIGURES_DIR):
    """W_net vs Lambda from Experiment 3.

    Parameters
    ----------
    results_dict : dict
        Maps label -> ExperimentResult with .extra['Lambda'] set.
    """
    _ensure_dir(figures_dir)

    lambdas = []
    w_nets = []
    labels = []

    for key, result in results_dict.items():
        if 'Lambda' in result.extra:
            lambdas.append(result.extra['Lambda'])
            w_nets.append(result.training_log['best_fitness'].max())
            labels.append(str(key))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(lambdas, w_nets, s=80, zorder=5)
    for i, label in enumerate(labels):
        ax.annotate(label, (lambdas[i], w_nets[i]), fontsize=8)

    ax.axvline(1.0, color='red', linestyle='--', alpha=0.5, label='Lambda=1')
    ax.set_xlabel('Lambda = D * tau_mu / xi^2')
    ax.set_ylabel('Best W_net')
    ax.set_title('W_net vs Lambda (Experiment 3)')
    ax.set_xscale('log')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'lambda_sweep.png'), dpi=150)
    plt.close()


def plot_efficiency_vs_sigma(efficiencies, sigmas, figures_dir=FIGURES_DIR):
    """eta vs Sigma_cycle across generations.

    Parameters
    ----------
    efficiencies : array
        Efficiency values.
    sigmas : array
        Entropy production values.
    """
    _ensure_dir(figures_dir)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(sigmas, efficiencies, alpha=0.5, s=20)
    ax.set_xlabel('Sigma_cycle')
    ax.set_ylabel('Efficiency eta')
    ax.set_title('Efficiency vs Entropy Production')

    # Linear fit
    if len(sigmas) > 2:
        coeffs = np.polyfit(sigmas, efficiencies, 1)
        x_fit = np.linspace(min(sigmas), max(sigmas), 100)
        ax.plot(x_fit, np.polyval(coeffs, x_fit), 'r--',
                label=f'Fit: slope={coeffs[0]:.4f}')
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'efficiency_vs_sigma.png'), dpi=150)
    plt.close()


def load_and_plot_all(results_base='results', figures_dir=FIGURES_DIR):
    """Regenerate all figures from saved results."""
    _ensure_dir(figures_dir)

    # Experiment 0
    exp0_path = os.path.join(results_base, 'exp0', 'sweep.npz')
    if os.path.exists(exp0_path):
        data = np.load(exp0_path)
        from experiments.exp0_baseline import plot_heatmap
        plot_heatmap({
            'J0_values': data['J0_values'],
            'tau_values': data['tau_values'],
            'W_net_grid': data['W_net_grid'],
            'J0_opt': float(data['J0_opt']),
            'tau_opt': int(data['tau_opt']),
        }, figures_dir=figures_dir)
        print(f"Exp 0: J0_opt={data['J0_opt']:.4f}, W_net_opt={data['W_net_opt']:.4f}")
    else:
        print(f"Exp 0 results not found at {exp0_path}")

    # Experiment 1, 2, 3: load training logs
    for exp_name in ['exp1', 'exp2', 'exp3']:
        exp_dir = os.path.join(results_base, exp_name)
        if not os.path.exists(exp_dir):
            print(f"{exp_name} results not found")
            continue

        results = {}
        for subdir in sorted(os.listdir(exp_dir)):
            log_path = os.path.join(exp_dir, subdir, 'training_log.npz')
            if os.path.exists(log_path):
                results[subdir] = np.load(log_path)

        if results:
            baseline = float(data['W_net_opt']) if os.path.exists(exp0_path) else None
            plot_learning_curves(results, exp_name, baseline_ceiling=baseline,
                                figures_dir=figures_dir)
            print(f"{exp_name}: plotted {len(results)} runs")


if __name__ == '__main__':
    load_and_plot_all()