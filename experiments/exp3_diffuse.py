"""Experiment 3: Diffusing Budget.

Sweep D over {0.01, 0.1, 0.5, 2.0} at fixed tau_mu=20.
Sweep tau_mu over {5, 20, 100} at fixed D=0.1.
Sweep T_mean over {2.0, 2.5, 3.0} to vary xi and test Lambda ~ 1.

Expected: W_net peaks near Lambda = 1.
"""

import os
import numpy as np
import jax
import jax.numpy as jnp

from evolving_ising.model import IsingModel
from train import run_experiment, DEFAULT_CONFIG


def estimate_correlation_length(model, J0, T, n_samples=200):
    """Estimate spin-spin correlation length xi.

    Computed from exponential decay of connected correlation function:
    C(r) = <s_i s_{i+r}> - <s_i>^2

    Returns
    -------
    xi : float
        Correlation length estimate.
    """
    L = model.h
    N = model.n
    K = model.K

    J_nk = jnp.ones((N, K), dtype=jnp.float32) * J0
    J_nk = J_nk * jnp.array(model.mask, dtype=jnp.float32)

    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)
    spins = model.init_spins(init_key, batch_size=1)

    # Equilibrate
    for _ in range(500):
        key, subkey = jax.random.split(key)
        spins, _ = model.metropolis_checkerboard_sweeps(subkey, spins, J_nk, T, 1)

    # Collect samples
    corr_samples = []
    for _ in range(n_samples):
        key, subkey = jax.random.split(key)
        spins, _ = model.metropolis_checkerboard_sweeps(subkey, spins, J_nk, T, 5)

        s = np.asarray(spins[0]).reshape(L, L)

        # Compute horizontal correlation
        max_r = L // 2
        corrs = np.zeros(max_r)
        for r in range(max_r):
            corrs[r] = np.mean(s * np.roll(s, r, axis=1))

        m_sq = np.mean(s) ** 2
        corrs -= m_sq
        corr_samples.append(corrs)

    mean_corr = np.mean(corr_samples, axis=0)

    # Fit exponential decay: C(r) ~ exp(-r/xi)
    # Use log-linear fit on positive correlations
    r_vals = np.arange(1, len(mean_corr))
    c_vals = mean_corr[1:]
    positive = c_vals > 0
    if positive.sum() < 2:
        return 1.0  # fallback

    r_fit = r_vals[positive]
    c_fit = np.log(c_vals[positive])

    # Linear regression: log(C) = -r/xi + const
    A = np.vstack([r_fit, np.ones_like(r_fit)]).T
    slope, _ = np.linalg.lstsq(A, c_fit, rcond=None)[0]

    xi = -1.0 / slope if slope < 0 else float(L)
    return max(1.0, min(xi, float(L)))


def run_exp3(best_lambda=0.01, best_alpha=0.1, config=None,
             results_dir='results/exp3', n_generations=500):
    """Run Experiment 3: Diffusing Budget sweeps."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    T_mean = cfg['T_mean']
    J_init = T_mean / 2.269

    L = cfg['L']
    model = IsingModel(
        (L, L),
        neighborhood=cfg['neighborhood'],
        boundary=cfg['boundary'],
    )

    results = {}

    # D sweep at fixed tau_mu=20
    D_values = [0.01, 0.1, 0.5, 2.0]
    tau_mu_fixed = 20.0

    xi = estimate_correlation_length(model, J_init, T_mean)
    print(f"Estimated correlation length xi = {xi:.2f}")

    for D in D_values:
        Lambda = D * tau_mu_fixed / xi**2
        name = f"D_{D:.2f}_taumu_{tau_mu_fixed:.0f}"
        print(f"\n{'='*60}")
        print(f"Experiment 3: D={D}, tau_mu={tau_mu_fixed}, Lambda={Lambda:.3f}")
        print(f"{'='*60}")

        exp_config = {
            **cfg,
            'J_init': J_init,
            'lambda': best_lambda,
            'budget_alpha': best_alpha,
            'D': D,
            'tau_mu': tau_mu_fixed,
            'n_generations': n_generations,
        }

        result = run_experiment(
            config=exp_config,
            budget_type='diffusing',
            name=name,
            results_dir=results_dir,
        )
        result.extra['xi'] = xi
        result.extra['Lambda'] = Lambda
        results[('D', D)] = result

    # tau_mu sweep at fixed D=0.1
    D_fixed = 0.1
    tau_mu_values = [5, 20, 100]

    for tau_mu in tau_mu_values:
        Lambda = D_fixed * tau_mu / xi**2
        name = f"D_{D_fixed:.2f}_taumu_{tau_mu:.0f}"
        print(f"\n{'='*60}")
        print(f"Experiment 3: D={D_fixed}, tau_mu={tau_mu}, Lambda={Lambda:.3f}")
        print(f"{'='*60}")

        exp_config = {
            **cfg,
            'J_init': J_init,
            'lambda': best_lambda,
            'budget_alpha': best_alpha,
            'D': D_fixed,
            'tau_mu': tau_mu,
            'n_generations': n_generations,
        }

        result = run_experiment(
            config=exp_config,
            budget_type='diffusing',
            name=name,
            results_dir=results_dir,
        )
        result.extra['xi'] = xi
        result.extra['Lambda'] = Lambda
        results[('tau_mu', tau_mu)] = result

    # T_mean sweep to vary xi
    T_mean_values = [2.0, 2.5, 3.0]
    D_test = 0.1
    tau_mu_test = 20.0

    for T_m in T_mean_values:
        J_init_t = T_m / 2.269
        xi_t = estimate_correlation_length(model, J_init_t, T_m)
        Lambda = D_test * tau_mu_test / xi_t**2

        name = f"Tmean_{T_m:.1f}"
        print(f"\n{'='*60}")
        print(f"Experiment 3: T_mean={T_m}, xi={xi_t:.2f}, Lambda={Lambda:.3f}")
        print(f"{'='*60}")

        exp_config = {
            **cfg,
            'T_mean': T_m,
            'J_init': J_init_t,
            'lambda': best_lambda,
            'budget_alpha': best_alpha,
            'D': D_test,
            'tau_mu': tau_mu_test,
            'n_generations': n_generations,
        }

        result = run_experiment(
            config=exp_config,
            budget_type='diffusing',
            name=name,
            results_dir=results_dir,
        )
        result.extra['xi'] = xi_t
        result.extra['Lambda'] = Lambda
        results[('T_mean', T_m)] = result

    return results


if __name__ == '__main__':
    run_exp3()
