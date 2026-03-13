"""Experiment 2 Report — Neighbourhood Budget.

Loads results/exp2/ and generates a self-contained HTML report.

Usage
-----
cd <repo_root>
python experiments/exp2_report.py
python experiments/exp2_report.py --results-dir results/exp2 --out results/exp2/report.html
"""

import argparse
import os
import sys
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure project root and experiments/ are on the path
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from report_utils import (  # noqa: E402
    REPORT_CSS as _CSS,
    fig_to_b64 as _fig_to_b64,
    load_run as _load_run_util,
    load_config,
    config_table_html,
    run_anim_frames,
    frames_to_gif_b64,
    canvas_chart_html,
    scenario_selector_html,
    gif_tag as _gif_tag,
    PALETTE,
)

_EXPLANATION = """
<h2>1. About This Experiment</h2>
<div class="card">
  <h3>Physical Setup</h3>
  <p>Same oscillating bath as Experiments 0 and 1:
  <p class="formula">T(t) = T<sub>mean</sub> + ΔT · sin(2π t / τ)</p>
  32 × 32 lattice, von Neumann neighbourhood, periodic boundaries. The best
  (λ, α) from Experiment 1 is used throughout.</p>
</div>
<div class="card">
  <h3>NeighbourhoodBudget: What Changes?</h3>
  <p>In Experiment 1, each bond (i, j) had its own private credit accumulating from
  local ordering events. In Experiment 2, budget is <em>shared</em> within each site's
  neighbourhood. When site i accumulates thermodynamic credit, that credit is accessible
  to any bond connecting i to a neighbour.</p>
  <p>The parameter γ controls how widely credit diffuses across the neighbourhood:
  <p class="formula">B<sub>i</sub>(t) = (1 − γ) · B<sub>i,local</sub>(t) + γ · mean<sub>j∈N(i)</sub> B<sub>j</sub>(t)</p>
  At γ = 0 the budget collapses to per-site (no sharing). At γ = 1 all neighbours
  pool completely. Intermediate γ allows coordinated remodelling across correlated
  patches without destroying local signal.</p>
</div>
<div class="card">
  <h3>Why Non-Monotonic W_net vs γ?</h3>
  <p>Too little sharing (γ → 0): bonds at domain walls receive signal only from their
  own local ordering history — sparse and noisy. Too much sharing (γ → 1): budget
  is diluted uniformly across the lattice, losing the spatial information about
  where domain walls actually are. The optimal γ* should match the typical
  domain-wall spacing in the spin configuration, which is set by the correlation
  length ξ at T<sub>mean</sub>.</p>
</div>
<div class="card">
  <h3>Tau Sensitivity</h3>
  <p>A secondary sweep at the best γ* varies τ ∈ {100, 200, 500}. If γ* is truly
  set by the spatial correlation length (a static property), it should not shift
  much with τ. If γ* does shift, it suggests the budget timescale interacts with
  the oscillation period in a non-trivial way.</p>
</div>
"""


def fig_learning_curves(runs_gamma, gamma_values, baseline_W=11.86):
    """One subplot per gamma, W_net (best per gen) vs generation."""
    n = len(gamma_values)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    cmap = matplotlib.colormaps['viridis'].resampled(n)

    for ax, gamma, color in zip(axes, gamma_values, [cmap(i) for i in range(n)]):
        key = f'gamma_{gamma:.2f}'
        log, _ = runs_gamma.get(key, (None, None))
        if log is not None:
            gens = log['generation']
            best = log['best_fitness']
            ax.plot(gens, best, color=color, lw=1.6)
            ax.fill_between(gens, log['mean_fitness'], best, alpha=0.15, color=color)
        ax.axhline(baseline_W, color='tomato', ls='--', lw=1.2, label='Exp0 baseline')
        ax.set_title(f'γ = {gamma:.2f}', fontsize=10)
        ax.set_xlabel('Generation')
        if ax is axes[0]:
            ax.set_ylabel('W_net')
        ax.legend(fontsize=7)

    fig.suptitle('Learning Curves — W_net vs Generation by γ', fontsize=12, y=1.02)
    fig.tight_layout()
    return _fig_to_b64(fig)


def fig_gamma_sweep(gamma_values, best_per_gamma, best_gamma, baseline_W=11.86):
    """Bar chart of best W_net per gamma."""
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ['seagreen' if g == best_gamma else 'steelblue' for g in gamma_values]
    bars = ax.bar([f'{g:.2f}' for g in gamma_values], best_per_gamma, color=colors, alpha=0.85)
    ax.axhline(baseline_W, color='tomato', ls='--', lw=1.4, label=f'Exp0 baseline ({baseline_W:.2f})')
    ax.set_xlabel('γ')
    ax.set_ylabel('Best W_net (final)')
    ax.set_title('W_net vs Neighbourhood Pooling Strength γ')
    ax.legend(fontsize=9)
    for bar, val in zip(bars, best_per_gamma):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    return _fig_to_b64(fig)


def fig_tau_sweep(tau_values, best_per_tau, best_gamma, baseline_W=11.86):
    """Bar chart of best W_net per tau at best gamma."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(t) for t in tau_values], best_per_tau, color='darkorange', alpha=0.85)
    ax.axhline(baseline_W, color='tomato', ls='--', lw=1.4, label=f'Exp0 baseline ({baseline_W:.2f})')
    ax.set_xlabel('τ (cycle period)')
    ax.set_ylabel('Best W_net (final)')
    ax.set_title(f'W_net vs τ  (γ = {best_gamma:.2f})')
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _fig_to_b64(fig)


def fig_controller_strategy(params_flat, config):
    """2-panel heatmap of δJ vs (T_norm, m_bar) for aligned and anti-aligned bonds."""
    try:
        from work_extraction.controller import LocalController
    except ImportError:
        return None

    delta_J_max = config.get('delta_J_max', 0.1)
    hidden_size = config.get('hidden_size', 8)
    controller = LocalController(delta_J_max=delta_J_max, hidden_size=hidden_size)
    controller.set_params(params_flat)

    T_norm_vals = np.linspace(-1, 1, 40)
    m_bar_vals  = np.linspace(-1, 1, 40)
    TT, MM = np.meshgrid(T_norm_vals, m_bar_vals)  # (40, 40)
    n = 40 * 40

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    # Controller inputs: [s_i, s_j, m_bar_i, T_norm, budget_norm, J_norm]
    # J_norm fixed at 0 (J = J_crit) for the visualisation sweep
    for ax, (s_i_val, s_j_val, title) in zip(axes, [
        (+1, +1, 'Aligned  (s_i=+1, s_j=+1)'),
        (+1, -1, 'Anti-aligned  (s_i=+1, s_j=−1)'),
    ]):
        x = np.stack([
            np.full(n, s_i_val, dtype=np.float32),
            np.full(n, s_j_val, dtype=np.float32),
            MM.ravel().astype(np.float32),
            TT.ravel().astype(np.float32),
            np.ones(n, dtype=np.float32),
            np.zeros(n, dtype=np.float32),   # J_norm = 0 → J at J_crit
        ], axis=-1)  # (n, 6)
        dJ = np.asarray(controller.forward(x)).ravel().reshape(40, 40)

        vmax = delta_J_max
        im = ax.imshow(dJ, origin='lower', aspect='auto',
                       extent=[T_norm_vals[0], T_norm_vals[-1], m_bar_vals[0], m_bar_vals[-1]],
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        fig.colorbar(im, ax=ax, label='δJ')
        ax.set_xlabel('T_norm  (−1 = cold, +1 = hot)')
        ax.set_ylabel('m̄_i  (local magnetisation EMA)')
        ax.set_title(title, fontsize=10)
        ax.axvline(0, color='k', lw=0.5, ls='--')
        ax.axhline(0, color='k', lw=0.5, ls='--')
    fig.suptitle('Controller Strategy: δJ vs (T_norm, m̄)', fontsize=12)
    fig.tight_layout()
    return _fig_to_b64(fig)


def fig_J_spatial(best_controller_npz, config):
    """Spatial heatmap of mean J per site from final best_controller params."""
    try:
        from evolving_ising.model import IsingModel
        from work_extraction.controller import LocalController
        import jax, jax.numpy as jnp
    except ImportError:
        return None

    L = config.get('L', 32)
    T_mean = config.get('T_mean', 2.5)
    J_init = config.get('J_init', T_mean / 2.269)
    J_c = T_mean / 2.269
    J_min = config.get('J_min', 0.01)
    J_max = config.get('J_max', 5.0)
    alpha_ema = config.get('budget_alpha', 0.05)

    params_flat = best_controller_npz['params']
    delta_J_max = config.get('delta_J_max', 0.1)
    hidden_size = config.get('hidden_size', 8)
    controller = LocalController(delta_J_max=delta_J_max, hidden_size=hidden_size)
    controller.set_params(params_flat)

    model = IsingModel((L, L), neighborhood=config.get('neighborhood', 'von_neumann'),
                       boundary=config.get('boundary', 'periodic'))
    neighbors = np.asarray(model.neighbors)   # (N, K)
    mask = np.asarray(model.mask, dtype=bool)  # (N, K)
    N, K = neighbors.shape
    J_nk = np.full((N, K), J_init, dtype=np.float32) * mask

    # Flat bond index arrays for vectorized controller calls
    bond_i = np.repeat(np.arange(N), K)   # (N*K,)
    bond_j = neighbors.ravel()             # (N*K,)
    bond_mask = mask.ravel()               # (N*K,)

    key = jax.random.PRNGKey(0)
    spins = model.init_spins(key, 1)       # (1, N)
    m_bar = np.zeros(N, dtype=np.float32)

    n_steps = 200
    T_norm_vals = np.sin(np.linspace(0, 2 * np.pi, n_steps))
    J_sum = np.zeros((N, K))

    for T_n in T_norm_vals:
        spins_np = np.asarray(spins[0], dtype=np.float32)  # (N,)
        m_bar = alpha_ema * spins_np + (1.0 - alpha_ema) * m_bar

        # Build vectorized state: [s_i, s_j, m_bar_i, T_norm, budget_norm, J_norm]
        J_nk_norm = np.tanh(J_nk / J_c - 1.0)
        x = np.stack([
            spins_np[bond_i],
            spins_np[bond_j],
            m_bar[bond_i],
            np.full(N * K, T_n, dtype=np.float32),
            np.ones(N * K, dtype=np.float32),
            J_nk_norm.ravel(),
        ], axis=-1)  # (N*K, 6)

        dJ_all = np.asarray(controller.forward(x)).ravel()  # (N*K,)
        dJ_nk = dJ_all.reshape(N, K)
        J_nk = np.clip(J_nk + dJ_nk * mask, J_min, J_max)
        J_sum += J_nk

        key, sk = jax.random.split(key)
        T_abs = float(T_mean + config.get('delta_T', 1.5) * T_n)
        spins, _ = model.metropolis_checkerboard_sweeps(sk, spins, jnp.array(J_nk), T_abs, 1)

    J_mean_site = (J_sum / n_steps * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
    J_map = J_mean_site.reshape(L, L)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    im = axes[0].imshow(J_map, cmap='viridis', origin='lower',
                        vmin=max(0, J_map.min()), vmax=J_map.max())
    cb = fig.colorbar(im, ax=axes[0], label='Mean J per site')
    cb.ax.axhline(J_c, color='red', lw=1.5, ls='--')
    axes[0].set_title('Spatial J Map (time-averaged)')
    axes[0].text(0.02, 0.02, f'J_c = {J_c:.3f}', color='red',
                 transform=axes[0].transAxes, fontsize=8)

    J_vals = J_nk[mask]
    axes[1].hist(J_vals, bins=50, color='steelblue', alpha=0.8, edgecolor='white')
    axes[1].axvline(J_init, color='gray', ls='--', lw=1.4, label=f'J_init = {J_init:.3f}')
    axes[1].axvline(J_c, color='tomato', ls='--', lw=1.4, label=f'J_c = {J_c:.3f}')
    axes[1].set_xlabel('J')
    axes[1].set_ylabel('Count')
    axes[1].set_title('J Distribution')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return _fig_to_b64(fig)


def generate_report(results_dir='results/exp2', out=None,
                    baseline_W=11.86, exp1_best=82.04, animate=True):
    if not os.path.isdir(results_dir):
        print(f'ERROR: {results_dir} not found. Run exp2_nbhd_budget.py first.')
        return

    gamma_values = [0.0, 0.1, 0.25, 0.5, 1.0]
    tau_values   = [100, 200, 500]

    # Load gamma sweep runs
    runs_gamma = {}
    for gamma in gamma_values:
        key = f'gamma_{gamma:.2f}'
        rd = os.path.join(results_dir, key)
        log, ctrl = _load_run_util(rd)
        runs_gamma[key] = (log, ctrl)

    best_per_gamma = []
    for gamma in gamma_values:
        log, _ = runs_gamma.get(f'gamma_{gamma:.2f}', (None, None))
        if log is not None:
            best_per_gamma.append(float(log['best_fitness'].max()))
        else:
            best_per_gamma.append(float('nan'))

    valid = [(g, v) for g, v in zip(gamma_values, best_per_gamma) if not np.isnan(v)]
    if not valid:
        print('No gamma sweep results found.')
        return
    best_gamma = max(valid, key=lambda x: x[1])[0]
    best_W_gamma = max(v for _, v in valid)

    # Load tau sweep runs
    runs_tau = {}
    best_per_tau = []
    for tau in tau_values:
        key = f'gamma_{best_gamma:.2f}_tau_{tau}'
        rd = os.path.join(results_dir, key)
        log, ctrl = _load_run_util(rd)
        runs_tau[key] = (log, ctrl)
        if log is not None:
            best_per_tau.append(float(log['best_fitness'].max()))
        else:
            best_per_tau.append(float('nan'))

    # Best run overall
    best_run_key = f'gamma_{best_gamma:.2f}'
    best_log, best_ctrl = runs_gamma[best_run_key]

    # Try to load config from best run
    config = {
        'T_mean': 2.5, 'delta_T': 1.5, 'L': 32,
        'J_init': 2.5 / 2.269, 'J_min': 0.01, 'J_max': 5.0,
        'delta_J_max': 0.1, 'hidden_size': 8,
        'neighborhood': 'von_neumann', 'boundary': 'periodic',
        'lambda': 0.01, 'budget_alpha': 0.1, 'gamma': best_gamma,
    }

    figs = {}
    try:
        figs['curves'] = fig_learning_curves(runs_gamma, gamma_values, baseline_W)
    except Exception as e:
        print(f'  Warning: learning curves failed: {e}')
    try:
        figs['gamma_sweep'] = fig_gamma_sweep(gamma_values, best_per_gamma, best_gamma, baseline_W)
    except Exception as e:
        print(f'  Warning: gamma sweep figure failed: {e}')
    if any(not np.isnan(v) for v in best_per_tau):
        try:
            figs['tau_sweep'] = fig_tau_sweep(tau_values, best_per_tau, best_gamma, baseline_W)
        except Exception as e:
            print(f'  Warning: tau sweep figure failed: {e}')
    if best_ctrl is not None:
        try:
            figs['strategy'] = fig_controller_strategy(best_ctrl['params'], config)
        except Exception as e:
            print(f'  Warning: strategy figure failed: {e}')
        try:
            figs['j_spatial'] = fig_J_spatial(best_ctrl, config)
        except Exception as e:
            print(f'  Warning: J spatial figure failed: {e}')

    def img(key, caption=''):
        if key not in figs or figs[key] is None:
            return ''
        parts = [f'<img class="fig" src="data:image/png;base64,{figs[key]}" alt="{key}">']
        if caption:
            parts.append(f'<p class="caption">{caption}</p>')
        return '\n'.join(parts)

    # --- Interactive canvas chart (gamma sweep training curves) ---
    chart_series = []
    for idx, gamma in enumerate(gamma_values):
        key = f'gamma_{gamma:.2f}'
        log, _ = runs_gamma.get(key, (None, None))
        if log is None:
            continue
        bh = log['best_fitness']
        chart_series.append({
            'label': f'γ={gamma:.2f}',
            'x': list(range(len(bh))),
            'y': [float(v) for v in bh],
            'color': PALETTE[idx % len(PALETTE)],
        })
    interactive_curves_html = canvas_chart_html(
        chart_series, 'exp2_curves',
        title='Training Curves — W_net vs Generation by γ (hover to inspect)',
        xlabel='Generation', ylabel='W_net',
        baseline=baseline_W,
    )

    # --- Per-gamma scenario panels ---
    model_shared = None
    try:
        from evolving_ising.model import IsingModel
        model_shared = IsingModel((config['L'], config['L']),
                                  neighborhood=config.get('neighborhood', 'von_neumann'),
                                  boundary=config.get('boundary', 'periodic'))
    except Exception as _e:
        print(f'  Warning: could not build model for animations: {_e}')

    scenario_ids, scenario_labels, scenario_panels = [], [], {}
    for idx, gamma in enumerate(gamma_values):
        sid = f'sc_gamma{gamma:.2f}'.replace('.', 'p')
        label = f'γ={gamma:.2f}'
        scenario_ids.append(sid)
        scenario_labels.append(label)

        log, ctrl = runs_gamma.get(f'gamma_{gamma:.2f}', (None, None))
        gamma_config = {**config, 'gamma': gamma, 'tau': 200,
                        'steps_per_cycle': 200}

        bh = log['best_fitness'] if log is not None else None
        run_best_W = float(np.max(bh)) if bh is not None and len(bh) > 0 else float('nan')
        cmp_str = ''
        if not np.isnan(run_best_W):
            pct = 100.0 * (run_best_W - baseline_W) / (abs(baseline_W) + 1e-12)
            cmp_str = f' ({pct:+.1f}% vs Exp0 baseline)'

        panel = f'<div class="run-panel">\n<h3>{label} — Best W_net: {run_best_W:.3f}{cmp_str}</h3>\n'

        if ctrl is not None:
            strat_b64 = None
            try:
                strat_b64 = fig_controller_strategy(ctrl['params'], gamma_config)
            except Exception as _e:
                print(f'  Warning: strategy {label}: {_e}')
            if strat_b64:
                panel += (f'<img class="fig" src="data:image/png;base64,{strat_b64}" alt="strategy">\n'
                          f'<p class="caption">δJ heatmap: aligned (left) and anti-aligned (right) bonds. '
                          f'Blue=strengthen, red=weaken.</p>\n')

            if model_shared is not None and animate:
                try:
                    print(f'  Animating {label}...', flush=True)
                    sf, jf, _, wt, tt, bf = run_anim_frames(
                        model_shared, gamma_config, 'neighbourhood',
                        params_flat=ctrl['params'],
                        n_cycles=10, steps_per_cycle=80, frame_skip=4,
                    )
                    gif_b64 = frames_to_gif_b64(sf, jf, fps=8, max_frames=200,
                                                wnet_trace=wt, T_trace=tt,
                                                bud_frames=bf)
                    if gif_b64:
                        panel += _gif_tag(gif_b64, 'Simulation animation',
                                          caption='Spin state (left), mean J (centre), per-site budget (right), '
                                                  'cumulative W_net and bath temperature T(t) below, '
                                                  f'over 10 cycles with neighbourhood budget (γ={gamma:.2f}).')
                except Exception as _e:
                    print(f'  Warning: animation {label}: {_e}')

        panel += '</div>\n'
        scenario_panels[sid] = panel

    best_sid = f'sc_gamma{best_gamma:.2f}'.replace('.', 'p')
    if best_sid not in scenario_ids and scenario_ids:
        best_sid = scenario_ids[0]

    selector_html = ''
    if scenario_ids:
        selector_html = scenario_selector_html(scenario_ids, scenario_labels, best_sid,
                                               title='Select γ Run')
        for sid in scenario_ids:
            display = 'block' if sid == best_sid else 'none'
            selector_html += (f'<div id="{sid}" style="display:{display}" class="card">\n'
                               + scenario_panels.get(sid, '') + '</div>\n')

    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

    cfg = load_config(os.path.join(results_dir, best_run_key)) or {}
    config_html = config_table_html(cfg, title='Best-Run Configuration')

    # Results table
    rows = ''
    for gamma, bw in zip(gamma_values, best_per_gamma):
        best_mark = ' ★' if gamma == best_gamma else ''
        cls = 'pass' if gamma == best_gamma else ''
        val_str = f'{bw:.2f}' if not np.isnan(bw) else '—'
        rows += (f'<tr><td class="{cls}">{gamma:.2f}{best_mark}</td>'
                 f'<td class="{cls}">{val_str}</td>'
                 f'<td class="{cls}">{"✓" if not np.isnan(bw) and bw > baseline_W else "—"}</td></tr>\n')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Experiment 2 — Neighbourhood Budget</title>
  <style>{_CSS}</style>
</head>
<body>
<h1>Experiment 2 — Neighbourhood Budget</h1>
<p class="meta">Generated: {ts} · Results: {os.path.abspath(results_dir)}</p>

{_EXPLANATION}

<h2>2. Key Results</h2>
<div class="card">
  <div class="highlight">
    <strong>Best γ:</strong> {best_gamma:.2f} →
    W_net = {best_W_gamma:.2f}
    (vs Exp1 best {exp1_best:.2f}, Exp0 baseline {baseline_W:.2f})
  </div>
  <table>
    <tr><th>γ</th><th>Best W_net</th><th>Beats Exp0 baseline?</th></tr>
    {rows}
  </table>
</div>

<h2>3. Learning Curves</h2>
{img('curves', 'W_net (best-ever and generation mean, shaded) vs generation for each γ value. Dashed red line is the Exp0 fixed-J ceiling. Runs that converge above the baseline demonstrate genuine advantage from neighbourhood budget sharing.')}

<h2>3b. Interactive Training Curves</h2>
<div class="card">
  <p>Hover to inspect values at any generation. Click buttons to toggle individual γ runs.</p>
  {interactive_curves_html}
</div>

<h2>4. W_net vs γ</h2>
{img('gamma_sweep', 'Best W_net achieved (across 500 generations) for each γ. The green bar marks the optimal γ*. A non-monotonic pattern (peak at intermediate γ) would confirm the spatial-correlation hypothesis.')}

<h2>5. Tau Sensitivity at γ* = {best_gamma:.2f}</h2>
{img('tau_sweep', f'Best W_net vs oscillation period τ at the best γ* = {best_gamma:.2f}. If the optimal γ is set by the static correlation length, performance should be relatively insensitive to τ.')}

<h2>6. Controller Strategy Analysis</h2>
<div class="card">
  <p>The controller MLP maps local state (T_norm, m̄, s_i·s_j, (s_i·s_j)·m̄, budget_norm) → δJ.
  By sweeping (T_norm, m̄) at fixed budget and bond alignment, we can read off the thermodynamic
  strategy the controller has learned without re-running the simulation.</p>
</div>
{img('strategy', 'Heatmap of proposed δJ as a function of normalised temperature T_norm (x-axis) and local magnetisation EMA m̄ (y-axis). Left: aligned bonds (s_i·s_j = +1). Right: anti-aligned bonds. Blue = strengthen coupling, red = weaken it. A rational strategy would strengthen bonds during the hot phase when domain walls form, and weaken during the cold phase.')}

<h2>7. Connectivity Analysis</h2>
{img('j_spatial', 'Left: spatial map of time-averaged coupling strength J per site. Heterogeneous patterns indicate the controller has learned spatially structured strategies rather than uniform rescaling. Right: distribution of final J values; markers indicate J_init (grey) and J_c = T_mean/2.269 (red).')}

<h2>8. Per-γ Analysis</h2>
<div class="card">
  <p>Select a γ value to see its controller strategy heatmap and a short simulation animation.</p>
  {selector_html}
</div>

<h2>9. Configuration</h2>
{config_html if config_html else '<p class="caption">[config.json not found — re-run exp2 to generate]</p>'}

</body>
</html>"""

    out = out or os.path.join(results_dir, 'report.html')
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Report written to {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results-dir', default='results/exp2')
    p.add_argument('--out', default=None)
    p.add_argument('--baseline-W', type=float, default=11.86,
                   help='Exp0 W_net_opt to use as baseline (default: 11.86)')
    p.add_argument('--exp1-best', type=float, default=82.04,
                   help='Best W_net from Exp1 (default: 82.04)')
    p.add_argument('--no-animate', action='store_true',
                   help='Skip GIF animation generation (faster, no per-γ animations)')
    args = p.parse_args()
    generate_report(args.results_dir, args.out, args.baseline_W, args.exp1_best,
                    animate=not args.no_animate)