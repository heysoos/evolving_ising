"""Generate a self-contained HTML report for Experiment 1 (BondBudget sweep).

Usage
-----
python experiments/exp1_report.py [--results-dir results/exp1] [--baseline-path results/exp0/sweep.npz]

The script scans ``results_dir`` for subdirectories named
``lam_{lam:.2f}_alpha_{alpha:.2f}``, loads ``training_log.npz`` from each,
and optionally loads a baseline W_net from ``baseline_path``.

Output: ``results_dir/report.html`` — fully self-contained (no external URLs).
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure project root and experiments/ are on the path
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from report_utils import (  # noqa: E402
    REPORT_CSS as _REPORT_CSS,
    fig_to_b64 as _fig_to_b64_util,
    load_config,
    config_table_html,
    run_anim_frames,
    frames_to_gif_b64,
    canvas_chart_html,
    scenario_selector_html,
    img_tag as _img_tag,
    gif_tag as _gif_tag,
    PALETTE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_run(run_dir):
    """Load training_log.npz from a run directory.  Returns dict or None."""
    p = Path(run_dir) / 'training_log.npz'
    if not p.exists():
        return None
    d = np.load(p)
    return {k: d[k] for k in d.files}


def _parse_run_name(name):
    """Parse 'lam_0.10_alpha_0.05' -> (lam, alpha) floats or None."""
    m = re.fullmatch(r'lam_([0-9.eE+\-]+)_alpha_([0-9.eE+\-]+)', name)
    if m is None:
        return None
    return float(m.group(1)), float(m.group(2))


# ---------------------------------------------------------------------------
# Figure generators — core 4 plots
# ---------------------------------------------------------------------------

def fig_learning_curves(runs, baseline_W):
    """One subplot per lambda, curves per alpha, baseline dashed line."""
    lam_vals = sorted({lam for lam, _ in runs})
    n_lam = len(lam_vals)
    if n_lam == 0:
        return None

    ncols = min(n_lam, 4)
    nrows = (n_lam + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                             squeeze=False)
    fig.suptitle('Learning Curves — W_net vs Generation', fontsize=13, y=1.01)

    alpha_vals = sorted({alpha for _, alpha in runs})
    cmap = matplotlib.colormaps['viridis'].resampled(max(len(alpha_vals), 1))
    alpha_color = {a: cmap(i) for i, a in enumerate(alpha_vals)}

    for ax_idx, lam in enumerate(lam_vals):
        row, col = divmod(ax_idx, ncols)
        ax = axes[row][col]
        ax.set_title(f'λ = {lam:.2f}', fontsize=10)
        ax.set_xlabel('Generation')
        ax.set_ylabel('W_net')

        for (l, alpha), data in sorted(runs.items()):
            if l != lam:
                continue
            best_hist = data.get('best_fitness')
            if best_hist is None:
                continue
            gens = np.arange(len(best_hist))
            ax.plot(gens, best_hist, color=alpha_color[alpha],
                    label=f'α={alpha:.2f}', linewidth=1.2)

        if baseline_W is not None:
            ax.axhline(baseline_W, color='crimson', linestyle='--',
                       linewidth=1.2, label='baseline')

        ax.legend(fontsize=7, loc='lower right')

    # Hide unused subplots
    for ax_idx in range(n_lam, nrows * ncols):
        row, col = divmod(ax_idx, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    return fig


def fig_heatmap(runs, baseline_W):
    """Alpha × lambda heatmap of best W_net; star marks best cell."""
    lam_vals = sorted({lam for lam, _ in runs})
    alpha_vals = sorted({alpha for _, alpha in runs})
    if not lam_vals or not alpha_vals:
        return None

    grid = np.full((len(alpha_vals), len(lam_vals)), np.nan)
    for (lam, alpha), data in runs.items():
        r = alpha_vals.index(alpha)
        c = lam_vals.index(lam)
        best_hist = data.get('best_fitness')
        if best_hist is not None and len(best_hist) > 0:
            grid[r, c] = float(best_hist[-1])

    fig, ax = plt.subplots(figsize=(max(4, len(lam_vals) * 0.9 + 1.5),
                                    max(3, len(alpha_vals) * 0.8 + 1.2)))

    vmin = np.nanmin(grid) if not np.all(np.isnan(grid)) else 0
    vmax = np.nanmax(grid) if not np.all(np.isnan(grid)) else 1
    im = ax.imshow(grid, aspect='auto', origin='lower',
                   vmin=vmin, vmax=vmax, cmap='RdYlGn')
    fig.colorbar(im, ax=ax, label='Best W_net (final gen)')

    ax.set_xticks(range(len(lam_vals)))
    ax.set_xticklabels([f'{v:.2f}' for v in lam_vals], fontsize=8)
    ax.set_yticks(range(len(alpha_vals)))
    ax.set_yticklabels([f'{v:.2f}' for v in alpha_vals], fontsize=8)
    ax.set_xlabel('Lambda (λ)')
    ax.set_ylabel('Alpha (α)')
    ax.set_title('Best W_net — Alpha × Lambda Grid', fontsize=11)

    # Mark best cell with a star
    if not np.all(np.isnan(grid)):
        best_idx = np.unravel_index(np.nanargmax(grid), grid.shape)
        ax.plot(best_idx[1], best_idx[0], '*', color='navy',
                markersize=14, label='best')
        ax.legend(fontsize=9)

    if baseline_W is not None:
        fig.text(0.5, -0.04,
                 f'Baseline W_net = {baseline_W:.4f}  (dashed reference)',
                 ha='center', fontsize=9, color='crimson')

    fig.tight_layout()
    return fig


def fig_wnet_vs_lambda(runs, baseline_W):
    """W_net vs lambda, one curve per alpha."""
    lam_vals = sorted({lam for lam, _ in runs})
    alpha_vals = sorted({alpha for _, alpha in runs})
    if not lam_vals or not alpha_vals:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    cmap = matplotlib.colormaps['viridis'].resampled(max(len(alpha_vals), 1))

    for i, alpha in enumerate(alpha_vals):
        xs, ys = [], []
        for lam in lam_vals:
            data = runs.get((lam, alpha))
            if data is None:
                continue
            best_hist = data.get('best_fitness')
            if best_hist is not None and len(best_hist) > 0:
                xs.append(lam)
                ys.append(float(best_hist[-1]))
        if xs:
            ax.plot(xs, ys, 'o-', color=cmap(i), label=f'α={alpha:.2f}',
                    linewidth=1.5, markersize=5)

    if baseline_W is not None:
        ax.axhline(baseline_W, color='crimson', linestyle='--',
                   linewidth=1.4, label='baseline')

    ax.set_xlabel('Lambda (λ) — thermodynamic cost weight')
    ax.set_ylabel('Best W_net (final generation)')
    ax.set_title('W_net vs Lambda by Alpha')
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_sigma_convergence(runs):
    """CMA-ES sigma on semilogy for all runs."""
    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = matplotlib.colormaps['tab20'].resampled(max(len(runs), 1))

    for idx, ((lam, alpha), data) in enumerate(sorted(runs.items())):
        sigma = data.get('sigma')
        if sigma is None or len(sigma) == 0:
            continue
        label = f'λ={lam:.2f} α={alpha:.2f}'
        ax.semilogy(np.arange(len(sigma)), sigma,
                    color=cmap(idx % 20), linewidth=1.0, label=label, alpha=0.75)

    ax.set_xlabel('Generation')
    ax.set_ylabel('CMA-ES sigma (log scale)')
    ax.set_title('CMA-ES Sigma Convergence — All Runs')
    if len(runs) <= 12:
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Controller strategy analysis figures
# ---------------------------------------------------------------------------

def fig_controller_strategy(params_flat, config):
    """2-panel heatmap: δJ as function of T_norm and m_bar for two bond types.

    Panel 1 (left):  aligned bond  (s_i=+1, s_j=+1)
    Panel 2 (right): anti-aligned bond (s_i=+1, s_j=-1)
    Budget fixed at 0.5 normalised.
    """
    try:
        from work_extraction.controller import LocalController

        ctrl = LocalController(
            delta_J_max=config.get('delta_J_max', 0.1),
            hidden_size=config.get('hidden_size', 8),
        )
        ctrl.set_params(params_flat)

        n_pts = 40
        T_norm_vals = np.linspace(-1, 1, n_pts)   # cold -> hot
        m_bar_vals  = np.linspace(-1, 1, n_pts)   # anti-aligned -> aligned memory
        TT, MM = np.meshgrid(T_norm_vals, m_bar_vals)  # (n_pts, n_pts)
        TT_flat = TT.ravel()
        MM_flat = MM.ravel()
        budget_fixed = np.full(len(TT_flat), 0.5, dtype=np.float32)

        # Aligned bond: s_i=+1, s_j=+1
        s_i_aligned  = np.ones(len(TT_flat), dtype=np.float32)
        s_j_aligned  = np.ones(len(TT_flat), dtype=np.float32)

        # Anti-aligned bond: s_i=+1, s_j=-1
        s_i_anti = np.ones(len(TT_flat),  dtype=np.float32)
        s_j_anti = np.full(len(TT_flat), -1.0, dtype=np.float32)

        # Doesn't get used?
        # T_mean  = float(config.get('T_mean', 2.5))
        # delta_T = float(config.get('delta_T', 1.5))

        # We pass T_norm per-bond via propose_updates; internally it scales
        # so we pass each unique T_norm as a scalar repeated over bonds.
        # Because propose_updates accepts a scalar T_norm, we loop row by row.
        # For grid efficiency, build the full (N,5) input directly.
        import jax.numpy as jnp

        J_norm_fixed = np.zeros(len(TT_flat), dtype=np.float32)  # J = J_crit

        def _run_grid(s_i, s_j):
            x = np.stack([
                s_i.astype(np.float32),
                s_j.astype(np.float32),
                MM_flat.astype(np.float32),
                TT_flat.astype(np.float32),
                budget_fixed,
                J_norm_fixed,
            ], axis=-1)
            x_jax = jnp.asarray(x)
            out = ctrl.forward(x_jax)  # (..., 1)
            return np.asarray(out).ravel().reshape(n_pts, n_pts)

        dJ_aligned = _run_grid(s_i_aligned, s_j_aligned)
        dJ_anti    = _run_grid(s_i_anti, s_j_anti)

        vmax = max(np.abs(dJ_aligned).max(), np.abs(dJ_anti).max(), 1e-6)
        vmin = -vmax

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, dJ, title in [
            (axes[0], dJ_aligned, 'δJ for aligned bond (s_i·s_j = +1)'),
            (axes[1], dJ_anti,    'δJ for anti-aligned bond (s_i·s_j = −1)'),
        ]:
            im = ax.imshow(
                dJ, origin='lower', aspect='auto',
                extent=[-1, 1, -1, 1],
                cmap='RdBu_r', vmin=vmin, vmax=vmax,
            )
            fig.colorbar(im, ax=ax, label='δJ')
            ax.set_xlabel('T_norm (cold → hot)')
            ax.set_ylabel('m̄_i (magnetisation EMA)')
            ax.set_title(title, fontsize=10)
            ax.axvline(0, color='k', linewidth=0.5, linestyle='--')
            ax.axhline(0, color='k', linewidth=0.5, linestyle='--')

        fig.suptitle('Controller Strategy: δJ Heatmaps (budget_norm = 0.5 fixed)',
                     fontsize=12)
        fig.tight_layout()
        return fig

    except Exception:
        return None


def fig_controller_budget_sensitivity(params_flat, config):
    """δJ vs budget_norm for 4 combinations: 2 T conditions × 2 spin states."""
    try:
        from work_extraction.controller import LocalController
        import jax.numpy as jnp

        ctrl = LocalController(
            delta_J_max=config.get('delta_J_max', 0.1),
            hidden_size=config.get('hidden_size', 8),
        )
        ctrl.set_params(params_flat)

        n_pts = 60
        bud_vals = np.linspace(0, 1, n_pts, dtype=np.float32)
        m_bar_zero = np.zeros(n_pts, dtype=np.float32)

        combos = [
            ('Cold (T_norm=−1), aligned',    -1.0, +1.0, +1.0, 'steelblue',  '-'),
            ('Cold (T_norm=−1), anti-aligned',-1.0, +1.0, -1.0, 'cornflowerblue', '--'),
            ('Hot  (T_norm=+1), aligned',     +1.0, +1.0, +1.0, 'firebrick',  '-'),
            ('Hot  (T_norm=+1), anti-aligned', +1.0, +1.0, -1.0, 'salmon',    '--'),
        ]

        fig, ax = plt.subplots(figsize=(7, 4))
        for label, t_norm, s_i_val, s_j_val, color, ls in combos:
            s_i = np.full(n_pts, s_i_val, dtype=np.float32)
            s_j = np.full(n_pts, s_j_val, dtype=np.float32)
            t_arr = np.full(n_pts, t_norm, dtype=np.float32)
            J_norm_zero = np.zeros(n_pts, dtype=np.float32)  # J = J_crit
            x = np.stack([s_i, s_j, m_bar_zero, t_arr, bud_vals, J_norm_zero], axis=-1)
            dJ = np.asarray(ctrl.forward(jnp.asarray(x))).ravel()
            ax.plot(bud_vals, dJ, color=color, linestyle=ls, linewidth=1.5,
                    label=label)

        ax.axhline(0, color='k', linewidth=0.5)
        ax.set_xlabel('budget_norm (tanh-scaled remaining budget)')
        ax.set_ylabel('Proposed δJ')
        ax.set_title('Controller Budget Sensitivity\n(m̄ = 0 fixed)')
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Connectivity / J analysis
# ---------------------------------------------------------------------------

def _simulate_final_J(run_dir, config, n_cycles=5):
    """Run a short simulation with the saved best controller and return J state.

    Uses @jax.jit + lax.scan (same pattern as make_jax_eval_fn in optimiser.py).
    No Python for loops in the hot path.

    Returns a dict with keys: J_final, J_bar_trace, T_trace
    or None on any failure.
    """
    try:
        import jax
        import jax.numpy as jnp
        from evolving_ising.model import IsingModel
        from work_extraction.controller import _mlp_forward, make_layer_specs
    except ImportError:
        return None

    try:
        ctrl_path = Path(run_dir) / 'best_controller.npz'
        if not ctrl_path.exists():
            return None
        ctrl_data = np.load(ctrl_path)
        params_flat = ctrl_data['params']

        L               = int(config.get('L', 32))
        T_mean          = float(config.get('T_mean', 2.5))
        delta_T         = float(config.get('delta_T', 1.5))
        tau             = float(config.get('tau', 200))
        steps_per_cycle = int(config.get('steps_per_cycle', 200))
        num_sweeps      = int(config.get('num_sweeps', 1))
        J_init_val      = float(config.get('J_init', T_mean / 2.269))
        J_min           = float(config.get('J_min', 0.01))
        J_max           = float(config.get('J_max', 5.0))
        bond_update_frac = float(config.get('bond_update_frac', 0.1))
        delta_J_max     = float(config.get('delta_J_max', 0.1))
        hidden_size     = int(config.get('hidden_size', 8))
        mag_alpha       = float(config.get('mag_ema_alpha', 0.05))
        B_scale         = float(config.get('B_scale', 2.0))
        lam             = float(config.get('lambda', 0.05))
        budget_alpha    = float(config.get('budget_alpha', 0.1))

        model = IsingModel(
            (L, L),
            neighborhood=config.get('neighborhood', 'von_neumann'),
            boundary=config.get('boundary', 'periodic'),
        )
        N = model.n
        K = model.K
        neighbors_np = np.asarray(model.neighbors)    # (N, K)
        mask_np      = np.asarray(model.mask, dtype=bool)  # (N, K)

        neighbors_jax = jnp.asarray(neighbors_np)
        mask_f        = jnp.asarray(mask_np, dtype=jnp.float32)
        valid_count   = jnp.float32(mask_np.sum())

        valid_i_np, valid_k_np = np.where(mask_np)
        valid_j_np  = neighbors_np[valid_i_np, valid_k_np]
        n_bonds_total = len(valid_i_np)
        n_updates     = max(1, int(n_bonds_total * bond_update_frac))

        valid_i_jax = jnp.asarray(valid_i_np, dtype=jnp.int32)
        valid_k_jax = jnp.asarray(valid_k_np, dtype=jnp.int32)
        valid_j_jax = jnp.asarray(valid_j_np, dtype=jnp.int32)

        J_init_jax = jnp.full((N, K), J_init_val, dtype=jnp.float32) * mask_f
        params_jax = jnp.asarray(params_flat)

        layer_specs = make_layer_specs(hidden_size)

        J_crit = T_mean / 2.269   # critical coupling

        # --- Pure JAX bond budget closures (mirrors make_jax_eval_fn) ---
        def bud_init():
            return jnp.zeros((N, K), dtype=jnp.float32)

        def bud_update(bud, s_bef, s_aft):
            nbr_bef = s_bef[neighbors_jax]
            nbr_aft = s_aft[neighbors_jax]
            ordering = jnp.maximum(0.0, s_aft[:, None] * nbr_aft
                                   - s_bef[:, None] * nbr_bef) * mask_f
            return bud + budget_alpha * ordering

        def bud_get(bud, si, sk, sj):
            return jnp.maximum(0.0, bud[si, sk])

        def bud_spend(bud, si, sk, sj, costs, can_apply):
            spend = jnp.where(can_apply, costs, 0.0)
            return jnp.maximum(0.0, bud.at[si, sk].add(-spend))

        # --- Warm up then run all cycles with lax.scan ---
        key = jax.random.PRNGKey(42)
        key, init_key, warmup_key = jax.random.split(key, 3)
        spins = model.init_spins(init_key, batch_size=1)
        spins, _ = model.metropolis_checkerboard_sweeps(
            warmup_key, spins, J_init_jax, T_mean, 200
        )

        def _step_fn(carry, t):
            spins_c, key_c, J_c, bud_c, mag_c = carry
            T_t = T_mean + delta_T * jnp.sin(2.0 * jnp.pi * t / tau)

            s_bef_f = jnp.mean(spins_c.astype(jnp.float32), axis=0)
            key_c, sub_m, sub_b = jax.random.split(key_c, 3)
            spins_c, _ = model.metropolis_checkerboard_sweeps(
                sub_m, spins_c, J_c, T_t, num_sweeps
            )
            s_aft_f = jnp.mean(spins_c.astype(jnp.float32), axis=0)

            bud_c = bud_update(bud_c, s_bef_f, s_aft_f)
            mag_c = mag_alpha * s_aft_f + (1.0 - mag_alpha) * mag_c

            perm = jax.random.permutation(sub_b, n_bonds_total)[:n_updates]
            si = valid_i_jax[perm]
            sk = valid_k_jax[perm]
            sj = valid_j_jax[perm]

            T_norm   = (T_t - T_mean) / delta_T
            bud_vals = bud_get(bud_c, si, sk, sj)
            bud_norm   = jnp.tanh(bud_vals / B_scale)
            J_norm_arr = jnp.tanh(J_c[si, sk] / J_crit - 1.0)
            x = jnp.stack([s_aft_f[si], s_aft_f[sj], mag_c[si],
                           jnp.full(n_updates, T_norm, dtype=jnp.float32),
                           bud_norm, J_norm_arr], axis=-1)

            dJ   = _mlp_forward(params_jax, x, layer_specs, delta_J_max).ravel()
            costs = jnp.abs(s_aft_f[si] * s_aft_f[sj] * dJ) + lam * jnp.abs(dJ)
            can_apply = bud_vals >= costs

            J_c = jnp.clip(
                J_c.at[si, sk].add(jnp.where(can_apply, dJ, 0.0)),
                J_min, J_max
            ) * mask_f
            bud_c = bud_spend(bud_c, si, sk, sj, costs, can_apply)

            J_bar = jnp.sum(J_c * mask_f) / jnp.maximum(valid_count, 1.0)
            return (spins_c, key_c, J_c, bud_c, mag_c), (J_bar, T_t)

        def _cycle_fn(carry, _):
            t_arr = jnp.arange(steps_per_cycle, dtype=jnp.int32)
            carry, (J_bars, T_ts) = jax.lax.scan(_step_fn, carry, t_arr)
            return carry, (J_bars, T_ts)

        @jax.jit
        def _run_sim(spins, key):
            init_carry = (spins, key, J_init_jax, bud_init(),
                          jnp.zeros(N, dtype=jnp.float32))
            final_carry, (J_bars_all, T_ts_all) = jax.lax.scan(
                _cycle_fn, init_carry, None, length=n_cycles
            )
            spins_f, key_f, J_final, bud_f, mag_f = final_carry
            return J_final, J_bars_all, T_ts_all

        J_final_jax, J_bars_all, T_ts_all = _run_sim(spins, key)

        # J_bars_all, T_ts_all: (n_cycles, steps_per_cycle) → flatten
        return {
            'J_final':     np.asarray(J_final_jax),
            'J_bar_trace': np.asarray(J_bars_all).ravel(),
            'T_trace':     np.asarray(T_ts_all).ravel(),
        }

    except Exception:
        return None


def fig_J_spatial(J_final, L, J_init, T_mean):
    """Left: spatial heatmap of mean J per site.  Right: row slices."""
    try:
        mask_np = J_final != 0
        K_eff = mask_np.sum(axis=1)
        K_eff = np.where(K_eff == 0, 1, K_eff)  # avoid divide-by-zero
        J_site = (J_final * mask_np).sum(axis=1) / K_eff
        J_map = J_site.reshape(L, L)
        J_c = T_mean / 2.269

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: spatial heatmap
        im = axes[0].imshow(J_map, cmap='coolwarm', aspect='equal',
                            vmin=J_map.min(), vmax=J_map.max())
        cb = fig.colorbar(im, ax=axes[0], label='Mean J per site')
        cb.ax.axhline(J_c, color='red', linewidth=1.5, linestyle='--')
        axes[0].set_title('Mean J per site (final state)')
        axes[0].set_xlabel('Column')
        axes[0].set_ylabel('Row')
        axes[0].text(0.02, 0.02, f'J_c = {J_c:.3f}', color='red',
                     transform=axes[0].transAxes, fontsize=8)

        # Right: row slices
        row_indices = [0, L // 4, L // 2, 3 * L // 4]
        cols = np.arange(L)
        for r in row_indices:
            axes[1].plot(cols, J_map[r, :], label=f'row {r}', linewidth=1.2)
        axes[1].axhline(J_c, color='red', linestyle='--', linewidth=1.2,
                        label=f'J_c={J_c:.3f}')
        axes[1].axhline(J_init, color='gray', linestyle=':', linewidth=1.0,
                        label=f'J_init={J_init:.3f}')
        axes[1].set_xlabel('Column index')
        axes[1].set_ylabel('Mean J')
        axes[1].set_title('Row slices of J spatial map')
        axes[1].legend(fontsize=8)

        fig.tight_layout()
        return fig

    except Exception:
        return None


def fig_J_histogram(J_final, J_init, T_mean):
    """Histogram of final coupling strengths."""
    try:
        mask_np = J_final != 0
        J_vals = J_final[mask_np].ravel()
        J_c = T_mean / 2.269

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(J_vals, bins=50, color='steelblue', alpha=0.75, edgecolor='white')
        ax.axvline(J_init, color='gray', linestyle='--', linewidth=1.4,
                   label=f'J_init = {J_init:.3f}')
        ax.axvline(J_c, color='crimson', linestyle='--', linewidth=1.4,
                   label=f'J_c = T_mean/2.269 = {J_c:.3f}')
        ax.set_xlabel('Coupling strength J')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of final coupling strengths J')
        ax.legend(fontsize=9)
        fig.tight_layout()
        return fig

    except Exception:
        return None


def fig_J_phase_portrait(J_bar_trace, T_trace):
    """J̄–T phase portrait of all simulated cycles."""
    try:
        fig, ax = plt.subplots(figsize=(6, 5))
        # Draw the loop with a gradient of colors to show direction
        n = len(T_trace)
        from matplotlib.collections import LineCollection
        points = np.array([T_trace, J_bar_trace]).T.reshape(-1, 1, 2)
        segs = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segs, cmap='viridis', linewidth=1.8)
        lc.set_array(np.linspace(0, 1, len(segs)))
        ax.add_collection(lc)
        fig.colorbar(lc, ax=ax, label='Simulation progress (start → end)')

        ax.plot(T_trace[0],  J_bar_trace[0],  'go', markersize=8, label='start')
        ax.plot(T_trace[-1], J_bar_trace[-1], 'rs', markersize=8, label='end')
        ax.set_xlim(T_trace.min() * 0.99, T_trace.max() * 1.01)
        ax.set_ylim(J_bar_trace.min() * 0.99, J_bar_trace.max() * 1.01)
        ax.set_xlabel('Bath temperature T')
        ax.set_ylabel('Mean coupling J̄')
        ax.set_title('J̄–T Phase Portrait (all cycles)')
        ax.legend(fontsize=9)
        fig.tight_layout()
        return fig

    except Exception:
        return None


# ---------------------------------------------------------------------------
# HTML content
# ---------------------------------------------------------------------------

_EXPLANATION = """
<h2>1. About This Experiment</h2>

<div class="card">
<h3>Physical Setup</h3>
<p>
We study a 2D Ising model (a lattice of binary spins s<sub>i</sub> ∈ {−1, +1})
coupled to a <em>single oscillating thermal bath</em>:
</p>
<p class="formula">T(t) = T<sub>mean</sub> + ΔT · sin(2π t / τ)</p>
<p>
During the <strong>cold phase</strong> (T &lt; T<sub>mean</sub>) the bath is below the Ising
critical temperature T<sub>c</sub> ≈ 2.269 J.  The system tends to <em>order</em>: spins
align into ferromagnetic domains, lowering the internal energy and <em>releasing heat</em>
into the bath (Q<sub>out</sub> &gt; 0, ΔE &lt; 0).
</p>
<p>
During the <strong>hot phase</strong> (T &gt; T<sub>mean</sub>) the system undergoes
<em>disordering</em>: domains break up, internal energy rises, and the system
<em>absorbs heat</em> from the bath (Q<sub>in</sub> &gt; 0, ΔE &gt; 0).
</p>
<p>
For a <strong>fixed coupling J</strong>, detailed balance ensures that the total heat
absorbed over a cycle equals the total heat released — there is no net work extraction:
Q<sub>in</sub> = Q<sub>out</sub>, W<sub>net</sub> = 0.
The system acts like a reversible heat sponge with no thermodynamic asymmetry.
</p>
<p>
<strong>W<sub>net</sub> = Q<sub>out</sub> − Q<sub>in</sub> − W<sub>remodel</sub></strong>
measures the net work delivered to an external load after paying the metabolic cost of
changing the bonds.
</p>
</div>

<div class="card">
<h3>Why Fixed J Fails</h3>
<p>
Without adaptation, the coupling J is calibrated for some intermediate temperature and
is equally "wrong" during both the hot and cold phases.  Because the Metropolis dynamics
satisfy detailed balance at each temperature, the cycle is thermodynamically reversible
in the limit of slow driving and Q<sub>in</sub> ≈ Q<sub>out</sub>.  The symmetry
between absorption (hot) and emission (cold) events cannot be broken without an
external information source that knows the current phase.
</p>
<p>
In practice, with a fixed J near J<sub>c</sub> = T<sub>mean</sub>/2.269, the system
spends both phases partially ordered, and the oscillation produces approximately equal
and opposite energy flows.  Net work extraction is negligible.
</p>
</div>

<div class="card">
<h3>The Controller's Opportunity</h3>
<p>
An adaptive controller that can <em>observe</em> the current temperature and local
spin state can break the Q<sub>in</sub> = Q<sub>out</sub> symmetry:
</p>
<ul>
  <li>
    <strong>During the cold phase</strong>: <em>raise J</em> on bonds that are already
    aligned.  A stronger coupling makes ferromagnetic ordering energetically more
    favourable, concentrating the Metropolis acceptance of ordering moves and increasing
    Q<sub>out</sub> per unit time.  The system "locks in" order more aggressively.
  </li>
  <li>
    <strong>During the hot phase</strong>: <em>lower J</em> on bonds that are aligned.
    A weaker coupling makes disordering energetically cheaper, reducing the energy that
    must be absorbed from the bath to break a domain.  Q<sub>in</sub> is reduced.
  </li>
</ul>
<p>
The net effect is Q<sub>out</sub> &gt; Q<sub>in</sub> even after subtracting the
mechanical work W<sub>remodel</sub> spent changing the bonds — a genuine thermodynamic
engine operating between hot and cold phases of a single oscillating bath.
</p>
</div>

<div class="card">
<h3>Controller Architecture</h3>
<p>
The controller is a compact <strong>5 → 8 → 8 → 1 MLP</strong>
(tanh activations, tanh-scaled output) evaluated <em>independently for each bond</em>
at each time step.  The five inputs encode the local physical state:
</p>
<ul>
  <li><strong>s<sub>i</sub></strong> — spin at site i ∈ {−1, +1}: tells the controller
      whether site i is currently up or down.</li>
  <li><strong>s<sub>j</sub></strong> — spin at neighbour j ∈ {−1, +1}: together with
      s<sub>i</sub> this determines alignment (s<sub>i</sub>·s<sub>j</sub> = ±1).</li>
  <li><strong>m̄<sub>i</sub></strong> — exponential moving average of s<sub>i</sub>
      (smoothing α<sub>EMA</sub> ≈ 0.05): encodes recent ordering history and acts as
      a local "memory" of whether site i has been persistently ordered or fluctuating.</li>
  <li><strong>T<sub>norm</sub></strong> = (T − T<sub>mean</sub>) / ΔT ∈ [−1, +1] — the
      bath temperature normalised to the oscillation amplitude.  Negative means cold,
      positive means hot; this is the key signal for phase-aware adaptation.</li>
  <li><strong>b<sub>norm</sub></strong> = tanh(B<sub>ij</sub> / B<sub>scale</sub>) ∈ [0, 1) —
      normalised remaining budget for bond (i, j): tells the controller whether it has
      "credits" to spend on remodelling this bond.</li>
</ul>
<p>
The output δJ ∈ [−δJ<sub>max</sub>, +δJ<sub>max</sub>] is added to the current
coupling: J<sub>ij</sub> ← clip(J<sub>ij</sub> + δJ, J<sub>min</sub>, J<sub>max</sub>).
</p>
</div>

<div class="card">
<h3>Bond Budget Mechanism</h3>
<p>
Remodelling bonds costs free energy — changing J<sub>ij</sub> while spins are
correlated changes the Hamiltonian energy and therefore costs thermodynamic work.
The <strong>BondBudget</strong> tracks a per-bond account of "ordering credit":
</p>
<p class="formula">B<sub>ij</sub>(t+1) = B<sub>ij</sub>(t) + α · max(0, Δ(s<sub>i</sub>·s<sub>j</sub>))</p>
<p>
Budget is <em>earned</em> whenever a Metropolis sweep increases the local spin
correlation on bond (i, j) — i.e., when an ordering event occurs at that bond.
Budget is <em>spent</em> proportionally to the cost of the proposed change:
</p>
<p class="formula">C(δJ) = |s<sub>i</sub>·s<sub>j</sub>·δJ| + λ·|δJ|</p>
<p>
A bond update is only applied if B<sub>ij</sub> ≥ C(δJ).  This grounds the
controller in thermodynamic accounting: it can only spend the ordering work that
has been deposited in the budget.
</p>
<ul>
  <li><strong>α (alpha)</strong> — accumulation rate: sets how quickly budget refills
      per ordering event.  High α = permissive; low α = tightly gated.  Expected to
      see a weak effect unless α is very small.</li>
  <li><strong>λ (lambda)</strong> — basal penalty weight: flat cost per unit |δJ|
      regardless of spin state.  This is the primary sweep parameter.  At λ = 0
      only spin-correlated remodelling costs anything; at high λ every bond change
      is expensive even if the spins are disordered.  Monotone degradation of W<sub>net</sub>
      with λ is the expected signal.</li>
</ul>
</div>

<div class="card">
<h3>CMA-ES Optimisation</h3>
<p>
The MLP weights (185 parameters for 5→8→8→1) are evolved by
<strong>Separable CMA-ES</strong> (diagonal covariance, population size 20).
Each fitness evaluation runs <em>n_eval_cycles = 10</em> complete temperature cycles
and returns the mean W<sub>net</sub>.  Fitness is <strong>maximised</strong>.
Training runs for 500 generations; the best individual across all generations
is reported.
</p>
</div>

<div class="card">
<h3>What to Look For in Results</h3>
<ul>
  <li><strong>Low λ should give the best W<sub>net</sub></strong>, ideally exceeding
      the exp0 ceiling (fixed-J optimum).  If it does not, the controller has not
      discovered the hot/cold phase asymmetry.</li>
  <li><strong>Monotone degradation with λ</strong>: each step up in λ should hurt
      performance.  A non-monotone response suggests the penalty is actually helping
      regularise noisy updates.</li>
  <li><strong>α effect</strong>: moderate values (0.1) should be near-optimal.
      Very low α starves the budget; very high α trivialises the gating.</li>
  <li><strong>Sigma convergence</strong>: σ should decrease monotonically (or
      plateau near a minimum scale) for well-converged runs.  Runs that plateau
      early or at large σ have not found a good optimum.</li>
  <li><strong>Controller strategy (see below)</strong>: an evolved controller should
      show δJ &gt; 0 (red, increase J) at T<sub>norm</sub> &lt; 0 for aligned bonds,
      and δJ &lt; 0 (blue, decrease J) at T<sub>norm</sub> &gt; 0.  This is the
      mechanistic signature of heat-engine operation.</li>
</ul>
</div>
"""


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

def generate_report(results_dir, baseline_path=None, animate=True):
    results_dir = Path(results_dir)
    runs = {}  # (lam, alpha) -> data dict
    run_dirs = {}  # (lam, alpha) -> Path

    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue
        parsed = _parse_run_name(subdir.name)
        if parsed is None:
            continue
        data = _load_run(subdir)
        if data is None:
            print(f'  [warn] no training_log.npz in {subdir}', file=sys.stderr)
            continue
        runs[parsed] = data
        run_dirs[parsed] = subdir

    if not runs:
        print(f'No valid runs found under {results_dir}', file=sys.stderr)
        sys.exit(1)

    # Load baseline
    baseline_W    = None
    baseline_J0   = None
    baseline_tau  = None
    if baseline_path is not None:
        bp = Path(baseline_path)
        if bp.exists():
            bd = np.load(bp)
            if 'W_net_opt' in bd:
                baseline_W = float(bd['W_net_opt'])
            else:
                for key in ('W_net', 'best_W_net', 'mean_W_net', 'w_net'):
                    if key in bd:
                        baseline_W = float(bd[key])
                        break
                if baseline_W is None and len(bd.files) > 0:
                    baseline_W = float(np.max(bd[bd.files[0]]))
            if 'J0_opt' in bd:
                baseline_J0  = float(bd['J0_opt'])
            if 'tau_opt' in bd:
                baseline_tau = float(bd['tau_opt'])
        else:
            print(f'  [warn] baseline not found: {bp}', file=sys.stderr)

    # Identify best run
    best_key = max(runs, key=lambda k: (
        float(runs[k]['best_fitness'][-1])
        if runs[k].get('best_fitness') is not None and len(runs[k]['best_fitness']) > 0
        else -np.inf
    ))
    best_data = runs[best_key]
    best_lam, best_alpha = best_key
    best_W = float(best_data['best_fitness'][-1]) if (
        best_data.get('best_fitness') is not None and len(best_data['best_fitness']) > 0
    ) else float('nan')

    # Try loading best controller params for analysis figures
    best_params_flat = None
    best_run_dir = run_dirs.get(best_key)
    if best_run_dir is not None:
        ctrl_path = best_run_dir / 'best_controller.npz'
        if ctrl_path.exists():
            try:
                best_params_flat = np.load(ctrl_path)['params']
            except Exception:
                pass

    # Build a config dict for the best run (from DEFAULT_CONFIG + parsed key)
    best_config = {
        'L': 32, 'T_mean': 2.5, 'delta_T': 1.5, 'tau': 200,
        'J_init': 2.5 / 2.269, 'J_min': 0.01, 'J_max': 5.0,
        'steps_per_cycle': 200, 'bond_update_frac': 0.1,
        'delta_J_max': 0.1, 'hidden_size': 8, 'mag_ema_alpha': 0.05,
        'B_scale': 2.0, 'lambda': best_lam, 'budget_alpha': best_alpha,
        'neighborhood': 'von_neumann', 'boundary': 'periodic',
    }

    # --- Figures ---
    figs = {}

    f = fig_learning_curves(runs, baseline_W)
    if f:
        figs['learning_curves'] = _fig_to_b64_util(f)

    f = fig_heatmap(runs, baseline_W)
    if f:
        figs['heatmap'] = _fig_to_b64_util(f)

    f = fig_wnet_vs_lambda(runs, baseline_W)
    if f:
        figs['wnet_vs_lambda'] = _fig_to_b64_util(f)

    f = fig_sigma_convergence(runs)
    if f:
        figs['sigma'] = _fig_to_b64_util(f)

    # Controller strategy (only if we have params)
    if best_params_flat is not None:
        f = fig_controller_strategy(best_params_flat, best_config)
        if f:
            figs['ctrl_strategy'] = _fig_to_b64_util(f)

        f = fig_controller_budget_sensitivity(best_params_flat, best_config)
        if f:
            figs['ctrl_budget'] = _fig_to_b64_util(f)

    # Connectivity/J analysis (requires JAX simulation)
    j_sim = None
    if best_run_dir is not None:
        print('  [info] simulating final J state (this may take a moment)...',
              file=sys.stderr)
        j_sim = _simulate_final_J(best_run_dir, best_config, n_cycles=5)

    if j_sim is not None:
        J_final = j_sim['J_final']
        J_bar   = j_sim['J_bar_trace']
        T_tr    = j_sim['T_trace']
        L       = best_config['L']
        J_init  = best_config['J_init']
        T_mean  = best_config['T_mean']

        f = fig_J_spatial(J_final, L, J_init, T_mean)
        if f:
            figs['J_spatial'] = _fig_to_b64_util(f)

        f = fig_J_histogram(J_final, J_init, T_mean)
        if f:
            figs['J_hist'] = _fig_to_b64_util(f)

        f = fig_J_phase_portrait(J_bar, T_tr)
        if f:
            figs['J_portrait'] = _fig_to_b64_util(f)

    # --- Interactive canvas chart (all training curves) ---
    chart_series = []
    for idx, ((lam, alpha), data) in enumerate(sorted(runs.items())):
        bh = data.get('best_fitness')
        if bh is None or len(bh) == 0:
            continue
        chart_series.append({
            'label': f'λ={lam:.2f} α={alpha:.2f}',
            'x': list(range(len(bh))),
            'y': [float(v) for v in bh],
            'color': PALETTE[idx % len(PALETTE)],
        })
    interactive_curves_html = canvas_chart_html(
        chart_series, 'exp1_curves',
        title='Training Curves — W_net vs Generation (hover to inspect, click legend to toggle)',
        xlabel='Generation', ylabel='W_net',
        baseline=baseline_W,
    )

    # --- Per-run scenario selector panels ---
    model_shared = None
    try:
        from evolving_ising.model import IsingModel
        model_shared = IsingModel(
            (best_config['L'], best_config['L']),
            neighborhood=best_config['neighborhood'],
            boundary=best_config['boundary'],
        )
    except Exception as _e:
        print(f'  [warn] could not build IsingModel for animations: {_e}', file=sys.stderr)

    scenario_ids = []
    scenario_labels = []
    scenario_panels = {}

    for idx, ((lam, alpha), data) in enumerate(sorted(runs.items())):
        sid = f'sc_lam{lam:.2f}_alpha{alpha:.2f}'.replace('.', 'p')
        label = f'λ={lam:.2f}, α={alpha:.2f}'
        scenario_ids.append(sid)
        scenario_labels.append(label)

        run_dir = run_dirs.get((lam, alpha))
        ctrl_data = None
        if run_dir is not None:
            cp = Path(run_dir) / 'best_controller.npz'
            if cp.exists():
                try:
                    ctrl_data = np.load(cp)
                except Exception:
                    pass

        run_config = {**best_config, 'lambda': lam, 'budget_alpha': alpha}
        bh = data.get('best_fitness')
        run_best_W = float(np.max(bh)) if bh is not None and len(bh) > 0 else float('nan')
        cmp_str = ''
        if baseline_W is not None and not np.isnan(run_best_W):
            pct = 100.0 * (run_best_W - baseline_W) / (abs(baseline_W) + 1e-12)
            cmp_str = f' ({pct:+.1f}% vs baseline)'

        panel = f'<div class="run-panel">\n<h3>{label} — Best W_net: {run_best_W:.3f}{cmp_str}</h3>\n'

        if ctrl_data is not None:
            try:
                f = fig_controller_strategy(ctrl_data['params'], run_config)
                if f:
                    panel += _img_tag(_fig_to_b64_util(f), 'Controller strategy',
                                      caption='δJ heatmap: aligned (left) and anti-aligned (right) bonds. '
                                              'Blue=strengthen, red=weaken. Ideal engine: strengthen during cold phase, weaken during hot.')
            except Exception as _e:
                print(f'  [warn] strategy fig {label}: {_e}', file=sys.stderr)

            if model_shared is not None and animate:
                try:
                    print(f'  [info] animating {label}...', file=sys.stderr)
                    sf, jf, _, wt, tt, bf = run_anim_frames(
                        model_shared, run_config, 'bond',
                        params_flat=ctrl_data['params'],
                        n_cycles=10, steps_per_cycle=80, frame_skip=4,
                    )
                    gif_b64 = frames_to_gif_b64(sf, jf, fps=8, max_frames=200,
                                                wnet_trace=wt, T_trace=tt,
                                                bud_frames=bf)
                    if gif_b64:
                        panel += _gif_tag(gif_b64, 'Simulation animation',
                                          caption='Spin state (left), mean J (centre), per-site budget (right), '
                                                  'cumulative W_net and bath temperature T(t) below, '
                                                  'over 10 cycles. '
                                                  'J structure emerges as the controller adapts bonds to the oscillating bath.')
                except Exception as _e:
                    print(f'  [warn] animation {label}: {_e}', file=sys.stderr)

        panel += '</div>\n'
        scenario_panels[sid] = panel

    best_sid = f'sc_lam{best_lam:.2f}_alpha{best_alpha:.2f}'.replace('.', 'p')
    if best_sid not in scenario_ids and scenario_ids:
        best_sid = scenario_ids[0]

    selector_html = ''
    if scenario_ids:
        selector_html = scenario_selector_html(scenario_ids, scenario_labels, best_sid,
                                               title='Select Run')
        for sid in scenario_ids:
            display = 'block' if sid == best_sid else 'none'
            selector_html += (f'<div id="{sid}" style="display:{display}" class="card">\n'
                               + scenario_panels.get(sid, '')
                               + '</div>\n')

    # --- Config ---
    cfg = load_config(best_run_dir) if best_run_dir is not None else None
    config_html = config_table_html(cfg, title='Best-Run Configuration') if cfg else ''

    # --- Key results card ---
    baseline_str = f'{baseline_W:.4f}' if baseline_W is not None else 'N/A'
    improvement_str = ''
    if baseline_W is not None and not np.isnan(best_W):
        pct = 100.0 * (best_W - baseline_W) / (abs(baseline_W) + 1e-12)
        improvement_str = f' ({pct:+.1f}% vs baseline)'

    baseline_extra = ''
    if baseline_J0 is not None:
        baseline_extra += f'<li><strong>Baseline optimal J₀:</strong> {baseline_J0:.4f}</li>'
    if baseline_tau is not None:
        baseline_extra += f'<li><strong>Baseline optimal τ:</strong> {baseline_tau:.0f}</li>'

    key_results_html = f"""
<h2>2. Key Results</h2>
<div class="card highlight">
  <h3>Key Results</h3>
  <ul>
    <li><strong>Best run:</strong> λ = {best_lam:.2f}, α = {best_alpha:.2f}</li>
    <li><strong>Best W_net:</strong> {best_W:.4f}{improvement_str}</li>
    <li><strong>Baseline W_net (exp0, W_net_opt):</strong> {baseline_str}</li>
    {baseline_extra}
    <li><strong>Total runs evaluated:</strong> {len(runs)}</li>
  </ul>
</div>
"""

    # --- Summary table ---
    table_rows = []
    for (lam, alpha), data in sorted(runs.items()):
        best_hist = data.get('best_fitness')
        if best_hist is not None and len(best_hist) > 0:
            final_wnet = float(best_hist[-1])
            best_ever  = float(np.max(best_hist))
            n_gens     = len(best_hist)
        else:
            final_wnet = float('nan')
            best_ever  = float('nan')
            n_gens     = 0

        if baseline_W is not None and not np.isnan(best_ever):
            beats     = best_ever > baseline_W
            beats_str = ('<span class="beat-yes">Yes</span>' if beats
                         else '<span class="beat-no">No</span>')
        else:
            beats_str = 'N/A'

        table_rows.append(
            f'<tr>'
            f'<td>{lam:.2f}</td>'
            f'<td>{alpha:.2f}</td>'
            f'<td>{best_ever:.4f}</td>'
            f'<td>{final_wnet:.4f}</td>'
            f'<td>{n_gens}</td>'
            f'<td>{beats_str}</td>'
            f'</tr>'
        )

    table_html = f"""
<h2>8. All Runs Summary</h2>
<table>
  <thead>
    <tr>
      <th>Lambda (λ)</th>
      <th>Alpha (α)</th>
      <th>Best W_net (all gens)</th>
      <th>Final W_net</th>
      <th>Generations</th>
      <th>Beats Baseline?</th>
    </tr>
  </thead>
  <tbody>
    {''.join(table_rows)}
  </tbody>
</table>
"""

    def _img(key, alt=''):
        if key not in figs:
            return ''
        return (f'<img class="fig" src="data:image/png;base64,{figs[key]}" '
                f'alt="{alt}">')

    def _caption(text):
        return f'<div class="caption">{text}</div>'

    # --- Assemble controller strategy section ---
    ctrl_section = ''
    if 'ctrl_strategy' in figs or 'ctrl_budget' in figs:
        ctrl_section = '''
<h2>6. Controller Strategy Analysis</h2>
<div class="card">
<p>
The panels below probe how the best-evolved controller responds to its inputs
by sweeping two key axes while holding the others fixed.  This reveals whether
the controller has learned the optimal heat-engine strategy (raise J when cold,
lower J when hot) or whether it has converged to a degenerate or suboptimal policy.
</p>
</div>
'''
        if 'ctrl_strategy' in figs:
            ctrl_section += _img('ctrl_strategy', 'Controller δJ heatmap')
            ctrl_section += _caption(
                'δJ as a function of normalised temperature (x-axis, cold left → hot right) '
                'and local magnetisation EMA m̄ (y-axis) for aligned bonds (left panel, '
                's<sub>i</sub>·s<sub>j</sub>=+1) and anti-aligned bonds (right, −1), with '
                'budget fixed at 0.5.  '
                'Red = controller proposes to increase J; blue = decrease J.  '
                'The optimal engine strategy shows <em>red in the bottom-left</em> '
                '(cold + aligned → strengthen) and '
                '<em>blue in the top-right</em> (hot + aligned → weaken).  '
                'Deviations indicate the learned strategy and its alignment with theory.'
            )
        if 'ctrl_budget' in figs:
            ctrl_section += _img('ctrl_budget', 'Controller budget sensitivity')
            ctrl_section += _caption(
                'Proposed δJ as a function of the remaining budget (x-axis, 0 = empty, '
                '1 = full) for four conditions: cold vs hot bath × aligned vs anti-aligned '
                'bond (m̄ = 0 fixed).  '
                'A well-trained controller should show larger |δJ| when budget is plentiful '
                'and appropriate sign (+ cold, − hot for aligned bonds).  '
                'Curves that are near-zero everywhere indicate the controller is budget-limited '
                'or has not learned to leverage available credits.'
            )

    # --- Assemble connectivity section ---
    j_section = ''
    if j_sim is not None and ('J_spatial' in figs or 'J_hist' in figs or 'J_portrait' in figs):
        j_section = '''
<h2>7. Connectivity Analysis</h2>
<div class="card">
<p>
After evolving the controller, we run a short simulation (5 cycles) starting
from the uniform initial coupling J<sub>init</sub> and record the final coupling
landscape, its histogram, and the J̄–T phase portrait of the last cycle.
</p>
</div>
'''
        if 'J_spatial' in figs:
            j_section += _img('J_spatial', 'Spatial J map')
            j_section += _caption(
                'Left: mean coupling per lattice site after simulation.  '
                'Red dashed marker on the colorbar indicates the theoretical critical coupling '
                'J<sub>c</sub> = T<sub>mean</sub>/2.269.  Sites with J &gt; J<sub>c</sub> '
                'are biased toward ordered behaviour; sites below it toward disordered.  '
                'Spatial patterns (striping, boundary effects) indicate non-trivial '
                'controller strategies.  '
                'Right: row slices showing heterogeneity across the lattice.'
            )
        if 'J_hist' in figs:
            j_section += _img('J_hist', 'J histogram')
            j_section += _caption(
                'Histogram of all valid bond couplings after simulation.  '
                'A bimodal distribution (peaks below and above J<sub>c</sub>) suggests '
                'the controller is specialising bonds for different roles.  '
                'A shift upward from J<sub>init</sub> (gray dashed) means the controller '
                'is on net strengthening bonds — consistent with a predominantly cold-phase '
                'ordering strategy.'
            )
        if 'J_portrait' in figs:
            j_section += _img('J_portrait', 'J-T phase portrait')
            j_section += _caption(
                'Parametric plot of mean coupling J̄ vs bath temperature T over the last '
                'simulation cycle.  A <em>counter-clockwise</em> loop (J rises as T falls, '
                'J falls as T rises) is the signature of a thermodynamically productive '
                'engine cycle — the controller is correctly pacing its remodelling.  '
                'A clockwise loop indicates lag or the wrong phase relationship.  '
                'A near-circular closed loop with good coverage of the T range is ideal.'
            )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Experiment 1 — BondBudget Controller Report</title>
  <style>{_REPORT_CSS}</style>
</head>
<body>
<h1>Experiment 1: Evolved Local Controller with BondBudget</h1>
<p style="color:#555">
  Generated by <code>experiments/exp1_report.py</code> &mdash;
  results from <code>{results_dir}</code>
</p>

{_EXPLANATION}

{key_results_html}

<h2>3. Learning Curves</h2>
{_img('learning_curves', 'Learning Curves (W_net vs Generation)')}
{_caption(
    'Best W_net per generation for each (λ, α) run, grouped by λ value.  '
    'Each curve colour corresponds to a different α.  '
    'The crimson dashed line marks the exp0 baseline ceiling.  '
    'Look for: (1) runs that cross the baseline (controller beats fixed J), '
    '(2) early plateau (converged quickly) vs late improvement (slow landscape), '
    '(3) whether low λ runs converge faster and higher than high λ runs.'
)}

<h2>3b. Interactive Training Curves</h2>
<div class="card">
  <p>Hover over the chart to inspect values at any generation.
     Click the coloured buttons to toggle individual runs on or off.</p>
  {interactive_curves_html}
</div>

<h2>4. Performance Grid</h2>
<div class="two-col">
  <div>
    {_img('heatmap', 'W_net Heatmap')}
    {_caption(
        'Heatmap of final W_net across the (λ, α) grid.  '
        'Green = high W_net; red = low.  '
        'The navy star marks the single best cell.  '
        'The bottom-left corner (low λ, low α) should be warmest if the '
        'budget constraint is the binding limitation.'
    )}
  </div>
  <div>
    {_img('wnet_vs_lambda', 'W_net vs Lambda')}
    {_caption(
        'W_net as a function of λ for each α value.  '
        'Monotone decrease with λ validates that the basal penalty is '
        'the dominant cost.  Non-monotone behaviour at a particular α '
        'may indicate that moderate cost regularises noisy bond updates '
        'and improves sample efficiency.'
    )}
  </div>
</div>

<h2>5. Convergence</h2>
{_img('sigma', 'CMA-ES Sigma Convergence')}
{_caption(
    'CMA-ES step-size σ on a log scale for all runs.  '
    'Healthy convergence: σ falls monotonically from its initial value and '
    'plateaus at a small value (&lt;10⁻³) where the optimizer has narrowed '
    'to a local optimum.  '
    'Runs where σ stagnates at a large value have not converged; '
    'runs where σ collapses very early may have premature convergence to a '
    'suboptimal solution.'
)}

{ctrl_section}

{j_section}

{table_html}

{config_html}

<h2>9. Per-Run Analysis</h2>
<div class="card">
  <p>Select a run from the dropdown to see its controller strategy heatmap
     and a short simulation animation (spins + J coupling map).</p>
  {selector_html}
</div>

</body>
</html>
"""

    out_path = results_dir / 'report.html'
    out_path.write_text(html, encoding='utf-8')
    print(f'Report written to {out_path}')
    return str(out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate HTML report for Experiment 1 (BondBudget sweep).'
    )
    parser.add_argument(
        '--results-dir', default='results/exp1',
        help='Directory containing per-run subdirs (default: results/exp1)'
    )
    parser.add_argument(
        '--baseline-path', default='results/exp0/sweep.npz',
        help='Path to exp0 baseline .npz (default: results/exp0/sweep.npz)'
    )
    parser.add_argument(
        '--no-animate', action='store_true',
        help='Skip GIF animation generation (faster report, no per-run animations)'
    )
    args = parser.parse_args()

    results_dir   = Path(args.results_dir)
    baseline_path = Path(args.baseline_path) if args.baseline_path else None

    generate_report(results_dir, baseline_path, animate=not args.no_animate)


if __name__ == '__main__':
    main()
