"""Experiment 0: Baseline — Fixed J sweep.

Sweep J0 over [0.2, 2.0] (10 values) and tau over [10, 1000] (8 log-spaced).
For each combination run 50 cycles and record W_net, Sigma_cycle, m(t).

Expected: W_net peaks near J0 ≈ T_mean / 2.269.
Outputs: results/exp0/sweep.npz, figures/exp0_heatmap.png
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evolving_ising.model import IsingModel
from work_extraction.thermodynamics import run_cycles_jax
from work_extraction.train import DEFAULT_CONFIG


def run_baseline_sweep(config=None, results_dir='../results/exp0'):
    """Run the baseline J0 × tau sweep using JAX vmap + nested lax.scan.

    All J0 values are evaluated in parallel via jax.vmap.  A single
    jax.jit compilation covers the entire (warmup + n_cycles × steps_per_cycle)
    computation; subsequent tau values reuse the cached XLA binary.

    Returns
    -------
    dict with sweep results.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    J0_values = np.linspace(0.2, 2.0, 100)
    tau_values = np.unique(np.logspace(np.log10(10), np.log10(1000), 80).astype(int))

    L = cfg['L']
    model = IsingModel(
        (L, L),
        neighborhood=cfg['neighborhood'],
        boundary=cfg['boundary'],
    )

    T_mean        = cfg['T_mean']
    delta_T       = cfg['delta_T']
    n_cycles      = 50  # cfg['n_eval_cycles'] * 5
    num_sweeps    = cfg['num_sweeps']    # Python int — static (scan unrolling)
    warmup_sweeps = cfg['warmup_sweeps'] # Python int — static (single GPU call)
    batch_size    = 40  # cfg['n_eval_chains'] * 8

    mask_f = jnp.array(model.mask, dtype=jnp.float32)
    J0_jax = jnp.array(J0_values, dtype=jnp.float32)

    W_net_grid = np.zeros((len(J0_values), len(tau_values)))
    sigma_grid  = np.zeros((len(J0_values), len(tau_values)))

    # JIT + vmap over J0.  Compiles once for the fixed (n_cycles, num_sweeps)
    # structure.  tau and steps_per_cycle are dynamic JAX arguments so
    # different tau values in the Python loop share the same XLA binary.
    @jax.jit
    def sweep_tau(J0_batch, keys_batch, spins_batch, tau, steps_per_cycle):
        def run_one(J0, key, spins):
            J_nk = J0 * mask_f
            # Warmup: single fused GPU call — no Python loop
            key, subkey = jax.random.split(key)
            spins, _ = model.metropolis_checkerboard_sweeps(
                subkey, spins, J_nk, T_mean, warmup_sweeps
            )
            # Cycles: fori_loop (dynamic steps) inside lax.scan (static cycles)
            W_nets, sigmas, _, _ = run_cycles_jax(
                model, key, spins, J_nk,
                T_mean, delta_T, tau,
                steps_per_cycle, n_cycles, num_sweeps,
            )
            return jnp.mean(W_nets), jnp.mean(sigmas)

        return jax.vmap(run_one)(J0_batch, keys_batch, spins_batch)

    master_key = jax.random.PRNGKey(42)

    for j, tau in enumerate(tau_values):
        master_key, *init_keys = jax.random.split(master_key, len(J0_values) + 1)
        init_keys   = jnp.stack(init_keys)
        spins_batch = jax.vmap(lambda k: model.init_spins(k, batch_size))(init_keys)

        # steps_per_cycle = tau so each cycle covers exactly one full period.
        # Passed as a dynamic JAX int — no recompilation across tau values.
        W_nets, sigmas = sweep_tau(
            J0_jax, init_keys, spins_batch,
            jnp.float32(tau), jnp.int32(tau),
        )

        W_net_grid[:, j] = np.asarray(W_nets)
        sigma_grid[:, j]  = np.asarray(sigmas)

        best_i = int(np.argmax(W_net_grid[:, j]))
        print(f"tau={tau:5d}: W_net=[{W_net_grid[:,j].min():.3f} … "
              f"{W_net_grid[:,j].max():.3f}]  "
              f"best J0={J0_values[best_i]:.2f} "
              f"(W_net={W_net_grid[best_i,j]:.4f})")

    # Find optimum
    best_idx = np.unravel_index(W_net_grid.argmax(), W_net_grid.shape)
    J0_opt = J0_values[best_idx[0]]
    tau_opt = tau_values[best_idx[1]]
    W_net_opt = W_net_grid[best_idx]

    print(f"\nOptimal: J0={J0_opt:.4f}, tau={tau_opt}, W_net={W_net_opt:.4f}")
    print(f"Expected J0_opt ≈ T_mean/2.269 = {T_mean/2.269:.4f}")

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    np.savez(
        os.path.join(results_dir, 'sweep.npz'),
        J0_values=J0_values,
        tau_values=tau_values,
        W_net_grid=W_net_grid,
        sigma_grid=sigma_grid,
        J0_opt=J0_opt,
        tau_opt=tau_opt,
        W_net_opt=W_net_opt,
    )

    return {
        'J0_values': J0_values,
        'tau_values': tau_values,
        'W_net_grid': W_net_grid,
        'sigma_grid': sigma_grid,
        'J0_opt': J0_opt,
        'tau_opt': tau_opt,
        'W_net_opt': W_net_opt,
    }


def plot_heatmap(results, figures_dir='figures'):
    """Plot W_net heatmap from sweep results."""
    os.makedirs(figures_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(
        results['W_net_grid'],
        aspect='auto',
        origin='lower',
        extent=[
            np.log10(results['tau_values'][0]),
            np.log10(results['tau_values'][-1]),
            results['J0_values'][0],
            results['J0_values'][-1],
        ],
    )
    ax.set_xlabel('log10(tau)')
    ax.set_ylabel('J0')
    ax.set_title('W_net vs J0 and tau (Experiment 0: Baseline)')
    plt.colorbar(im, ax=ax, label='W_net')

    # Mark optimum
    best_idx = np.unravel_index(
        results['W_net_grid'].argmax(),
        results['W_net_grid'].shape
    )
    ax.plot(
        np.log10(results['tau_values'][best_idx[1]]),
        results['J0_values'][best_idx[0]],
        'r*', markersize=15, label=f"Optimum: J0={results['J0_opt']:.2f}"
    )

    # Mark expected J0
    T_mean = 2.5
    ax.axhline(T_mean / 2.269, color='white', linestyle='--',
               label=f'T_mean/2.269={T_mean/2.269:.2f}')

    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'exp0_heatmap.png'), dpi=150)
    plt.close()


if __name__ == '__main__':
    results = run_baseline_sweep()
    plot_heatmap(results)
