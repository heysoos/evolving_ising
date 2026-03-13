"""Generate a self-contained HTML diagnostic report for Experiment 1b.

Usage
-----
python experiments/exp1b_report.py [--results-dir results/exp1b]
                                   [--exp1-dir results/exp1]
                                   [--n-cycles 500]

Reads analysis.npz from each subdirectory of results-dir (produced by
exp1b_long_run.py) and generates results-dir/report.html.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from report_utils import (  # noqa: E402
    REPORT_CSS,
    fig_to_b64,
    canvas_chart_html,
    scenario_selector_html,
    PALETTE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_run_name(name):
    m = re.fullmatch(r'lam_([0-9.eE+\-]+)_alpha_([0-9.eE+\-]+)', name)
    if m is None:
        return None
    return float(m.group(1)), float(m.group(2))


def _load_analysis(run_dir):
    p = Path(run_dir) / 'analysis.npz'
    if not p.exists():
        return None
    d = np.load(p)
    return {k: d[k] for k in d.files}


def _load_config(run_dir):
    p = Path(run_dir) / 'config.json'
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def _img(b64, alt='', caption=''):
    if not b64:
        return ''
    s = f'<img class="fig" src="data:image/png;base64,{b64}" alt="{alt}">'
    if caption:
        s += f'\n<p class="caption">{caption}</p>'
    return s


# ---------------------------------------------------------------------------
# Per-step aggregation helpers
# ---------------------------------------------------------------------------

def _per_cycle(arr, steps_per_cycle):
    """Reshape (total_steps,) -> (n_cycles, steps_per_cycle)."""
    n = (len(arr) // steps_per_cycle) * steps_per_cycle
    return arr[:n].reshape(-1, steps_per_cycle)


def _phase_mask(T_trace, T_mean):
    """Return bool array: True = hot (T > T_mean), False = cold."""
    return T_trace > T_mean


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------

def fig_J_drift(data, config, run_label):
    """J_bar over all steps with phase colouring and J_c reference."""
    J_bar   = data['J_bar']
    T_trace = data['T_trace']
    T_mean  = float(config.get('T_mean', 2.5))
    J_init  = float(config.get('J_init', 0.92))
    J_c     = T_mean / 2.269
    spc     = int(config.get('steps_per_cycle', 200))
    total   = len(J_bar)

    # Subsample for plotting (plot up to 5000 points)
    step = max(1, total // 5000)
    x    = np.arange(total)[::step]
    J_s  = J_bar[::step]
    T_s  = T_trace[::step]

    fig, ax = plt.subplots(figsize=(11, 3.5))
    hot  = T_s > T_mean
    cold = ~hot
    ax.scatter(x[cold], J_s[cold], c='steelblue', s=0.8, alpha=0.4, label='Cold phase')
    ax.scatter(x[hot],  J_s[hot],  c='firebrick', s=0.8, alpha=0.4, label='Hot phase')
    ax.axhline(J_c,    color='crimson', linestyle='--', linewidth=1.2,
               label=f'J_c = {J_c:.3f}')
    ax.axhline(J_init, color='gray',    linestyle=':',  linewidth=1.0,
               label=f'J_init = {J_init:.3f}')

    # Cycle markers every 50 cycles
    for cyc in range(0, total // spc + 1, 50):
        ax.axvline(cyc * spc, color='k', linewidth=0.3, alpha=0.3)

    ax.set_xlabel('Step')
    ax.set_ylabel('Mean J̄')
    ax.set_title(f'J̄ Drift over {total // spc} Cycles — {run_label}')
    ax.legend(fontsize=8, markerscale=6, loc='upper left')
    fig.tight_layout()
    return fig


def fig_budget_dynamics(data, config, run_label):
    """Budget mean and budget_norm_mean over time, phase-coloured."""
    budget_mean      = data['budget_mean']
    budget_norm_mean = data['budget_norm_mean']
    T_trace          = data['T_trace']
    T_mean           = float(config.get('T_mean', 2.5))
    B_scale          = float(config.get('B_scale', 2.0))
    total            = len(budget_mean)

    step = max(1, total // 5000)
    x    = np.arange(total)[::step]
    bm   = budget_mean[::step]
    bnm  = budget_norm_mean[::step]
    hot  = (T_trace > T_mean)[::step]

    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)

    for ax, y, ylabel, title, ref_val, ref_label in [
        (axes[0], bm,  'Mean Budget B', 'Mean budget per bond', B_scale, f'B_scale={B_scale}'),
        (axes[1], bnm, 'budget_norm_mean', 'Mean tanh(B/B_scale) — saturation indicator',
         0.9, 'saturation ≈ 0.9'),
    ]:
        ax.scatter(x[~hot], y[~hot], c='steelblue', s=0.8, alpha=0.4)
        ax.scatter(x[hot],  y[hot],  c='firebrick', s=0.8, alpha=0.4)
        ax.axhline(ref_val, color='orange', linestyle='--', linewidth=1.2,
                   label=ref_label)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)

    axes[1].set_xlabel('Step')
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(0, 1.05)
    fig.suptitle(f'Budget Dynamics — {run_label}', fontsize=11)
    fig.tight_layout()
    return fig


def fig_budget_norm_histogram(data, config, run_label):
    """Histogram of budget_norm_mean values across all steps."""
    bnm = data['budget_norm_mean']

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(bnm, bins=50, color='steelblue', alpha=0.75, edgecolor='white',
            density=True)
    ax.axvline(0.9, color='orange', linestyle='--', linewidth=1.3,
               label='saturation threshold ≈ 0.9')
    frac_sat = float(np.mean(bnm > 0.9))
    ax.set_xlabel('budget_norm_mean = mean tanh(B/B_scale)')
    ax.set_ylabel('Density')
    ax.set_title(
        f'Budget Saturation Distribution — {run_label}\n'
        f'{frac_sat*100:.1f}% of steps with budget_norm_mean > 0.9'
    )
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    return fig


def fig_decision_breakdown(data, config, run_label):
    """Stacked bar chart of decision breakdown by phase."""
    T_trace = data['T_trace']
    T_mean  = float(config.get('T_mean', 2.5))
    n_up    = int(config.get('bond_update_frac', 0.1) *
                  int(config.get('L', 32)) ** 2 * 4)  # approx n_updates
    n_up    = max(n_up, 1)

    hot_mask  = T_trace > T_mean
    cold_mask = ~hot_mask

    def _breakdown(mask):
        pos   = data['n_applied_pos'][mask].mean()
        neg   = data['n_applied_neg'][mask].mean()
        gated = data['n_gated'][mask].mean()
        total = pos + neg + gated
        if total <= 0:
            return 0.0, 0.0, 0.0
        return pos / total, neg / total, gated / total

    hot_pos,  hot_neg,  hot_gated  = _breakdown(hot_mask)
    cold_pos, cold_neg, cold_gated = _breakdown(cold_mask)

    fig, ax = plt.subplots(figsize=(5, 4))
    phases = ['Cold', 'Hot']
    pos_f   = [cold_pos,   hot_pos]
    neg_f   = [cold_neg,   hot_neg]
    gate_f  = [cold_gated, hot_gated]

    x = np.arange(2)
    b0 = ax.bar(x, pos_f,  color='steelblue', label='Applied dJ > 0')
    b1 = ax.bar(x, neg_f,  bottom=pos_f, color='firebrick', label='Applied dJ < 0')
    bottom2 = [p + n for p, n in zip(pos_f, neg_f)]
    b2 = ax.bar(x, gate_f, bottom=bottom2, color='lightgray', label='Gated')

    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.set_ylabel('Fraction of proposed updates')
    ax.set_ylim(0, 1.05)
    ax.set_title(f'Decision Breakdown by Phase\n{run_label}')
    ax.legend(fontsize=8)

    # Annotate fractions
    for xi, (p, n, g) in enumerate(zip(pos_f, neg_f, gate_f)):
        if p > 0.04:
            ax.text(xi, p / 2, f'{p:.2f}', ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')
        if n > 0.04:
            ax.text(xi, p + n / 2, f'{n:.2f}', ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')
        if g > 0.04:
            ax.text(xi, p + n + g / 2, f'{g:.2f}', ha='center', va='center',
                    fontsize=8, color='#333')

    fig.tight_layout()
    return fig


def fig_dJ_signed(data, config, run_label):
    """Mean signed dJ (proposed and applied) by phase across cycles."""
    spc      = int(config.get('steps_per_cycle', 200))
    T_trace  = data['T_trace']
    T_mean   = float(config.get('T_mean', 2.5))
    dJ_mean  = data['dJ_mean']
    dJ_app   = data['dJ_applied_mean']

    n_cycles = len(T_trace) // spc
    cycles   = np.arange(n_cycles)

    # Per-cycle mean by phase
    def _cycle_phase_mean(arr):
        mat = _per_cycle(arr, spc)          # (n_cycles, spc)
        T_mat = _per_cycle(T_trace, spc)
        hot_mean  = np.where(T_mat > T_mean, mat, np.nan)
        cold_mean = np.where(T_mat <= T_mean, mat, np.nan)
        return np.nanmean(hot_mean, axis=1), np.nanmean(cold_mean, axis=1)

    dJ_hot_prop,  dJ_cold_prop  = _cycle_phase_mean(dJ_mean)
    dJ_hot_app,   dJ_cold_app   = _cycle_phase_mean(dJ_app)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
    for ax, (hot, cold), title in [
        (axes[0], (dJ_hot_prop,  dJ_cold_prop),  'Mean Proposed dJ per cycle'),
        (axes[1], (dJ_hot_app,   dJ_cold_app),   'Mean Applied dJ per cycle'),
    ]:
        ax.plot(cycles, cold, color='steelblue', linewidth=1.0, label='Cold phase')
        ax.plot(cycles, hot,  color='firebrick', linewidth=1.0, label='Hot phase')
        ax.axhline(0, color='k', linewidth=0.6, linestyle='--')
        ax.set_xlabel('Cycle')
        ax.set_ylabel('Mean dJ')
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle(f'Signed dJ by Phase — {run_label}', fontsize=11)
    fig.tight_layout()
    return fig


def fig_wnet_per_cycle(data, config, run_label):
    """W_net per cycle derived from Q_out - Q_in - W_remodel."""
    spc      = int(config.get('steps_per_cycle', 200))
    Q_in     = _per_cycle(data['Q_in_step'],     spc).sum(axis=1)
    Q_out    = _per_cycle(data['Q_out_step'],    spc).sum(axis=1)
    W_rem    = _per_cycle(data['W_remodel_step'], spc).sum(axis=1)
    W_net    = Q_out - Q_in - W_rem
    cycles   = np.arange(len(W_net))

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))

    axes[0].plot(cycles, W_net, color='steelblue', linewidth=1.0)
    axes[0].axhline(0, color='k', linewidth=0.7, linestyle='--')
    axes[0].set_xlabel('Cycle')
    axes[0].set_ylabel('W_net')
    axes[0].set_title('W_net per Cycle')

    # Rolling mean (window = 20 cycles)
    window = min(20, len(W_net) // 5)
    if window > 1:
        roll = np.convolve(W_net, np.ones(window) / window, mode='valid')
        axes[0].plot(cycles[window - 1:], roll, color='orange', linewidth=1.5,
                     label=f'Rolling mean ({window})', alpha=0.9)
        axes[0].legend(fontsize=8)

    # Cumulative W_net
    axes[1].plot(cycles, np.cumsum(W_net), color='steelblue', linewidth=1.2)
    axes[1].axhline(0, color='k', linewidth=0.7, linestyle='--')
    axes[1].set_xlabel('Cycle')
    axes[1].set_ylabel('Cumulative W_net')
    axes[1].set_title('Cumulative W_net')

    fig.suptitle(f'Work Extraction — {run_label}', fontsize=11)
    fig.tight_layout()
    return fig


def fig_net_J_per_cycle(data, config, run_label):
    """Net change in J_bar per cycle, split by hot/cold contributions."""
    spc      = int(config.get('steps_per_cycle', 200))
    J_bar    = data['J_bar']
    T_trace  = data['T_trace']
    T_mean   = float(config.get('T_mean', 2.5))

    J_mat = _per_cycle(J_bar,   spc)   # (n_cycles, spc)
    T_mat = _per_cycle(T_trace, spc)

    n_cycles  = J_mat.shape[0]
    delta_net = J_mat[:, -1] - J_mat[:, 0]   # total per cycle

    # Estimate hot/cold contributions: mean signed dJ applied per phase
    dJ_app = data['dJ_applied_mean']
    n_up   = data['n_applied_pos'] + data['n_applied_neg']  # applied count
    dJ_mat = _per_cycle(dJ_app, spc)
    nu_mat = _per_cycle(n_up,   spc)

    hot_contrib  = np.where(T_mat > T_mean, dJ_mat * nu_mat, 0.0).sum(axis=1)
    cold_contrib = np.where(T_mat <= T_mean, dJ_mat * nu_mat, 0.0).sum(axis=1)

    cycles = np.arange(n_cycles)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))

    axes[0].bar(cycles, delta_net, color=np.where(delta_net >= 0, 'steelblue', 'firebrick'),
                width=1.0, alpha=0.7)
    axes[0].axhline(0, color='k', linewidth=0.7)
    axes[0].set_xlabel('Cycle')
    axes[0].set_ylabel('ΔJ̄ per cycle')
    axes[0].set_title('Net J̄ change per cycle')

    axes[1].plot(cycles, cold_contrib, color='steelblue', linewidth=1.0, label='Cold phase')
    axes[1].plot(cycles, hot_contrib,  color='firebrick', linewidth=1.0, label='Hot phase')
    axes[1].axhline(0, color='k', linewidth=0.7, linestyle='--')
    axes[1].set_xlabel('Cycle')
    axes[1].set_ylabel('Σ (dJ_applied × n_applied)')
    axes[1].set_title('J change contributions by phase')
    axes[1].legend(fontsize=8)

    fig.suptitle(f'J̄ Drift Decomposition — {run_label}', fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Summary table across all runs
# ---------------------------------------------------------------------------

def summary_table_html(runs_summary):
    """runs_summary: list of dicts with keys name, lam, alpha, J_final, J_init,
    bud_norm_sat, frac_gated_hot, frac_gated_cold, W_net_early, W_net_late."""
    rows = []
    for r in sorted(runs_summary, key=lambda x: (x['lam'], x['alpha'])):
        j_drift = r['J_final'] - r['J_init']
        drift_cls = 'fail' if j_drift > 0.05 else 'pass'
        sat_cls   = 'fail' if r['bud_norm_sat'] > 0.8 else 'pass'
        rows.append(
            f'<tr>'
            f'<td>{r["lam"]:.2f}</td>'
            f'<td>{r["alpha"]:.2f}</td>'
            f'<td>{r["J_init"]:.3f}</td>'
            f'<td class="{drift_cls}">{r["J_final"]:.3f} ({j_drift:+.3f})</td>'
            f'<td class="{sat_cls}">{r["bud_norm_sat"]*100:.0f}%</td>'
            f'<td>{r["frac_gated_cold"]*100:.0f}%</td>'
            f'<td>{r["frac_gated_hot"]*100:.0f}%</td>'
            f'<td>{r["W_net_early"]:.3f}</td>'
            f'<td>{r["W_net_late"]:.3f}</td>'
            f'</tr>'
        )
    return f"""
<h2>2. Summary Across All Runs</h2>
<table>
  <thead>
    <tr>
      <th>λ</th><th>α</th>
      <th>J_init</th><th>J_bar final (drift)</th>
      <th>Budget saturated (&gt;0.9)</th>
      <th>Gated (cold)</th><th>Gated (hot)</th>
      <th>W_net early</th><th>W_net late</th>
    </tr>
  </thead>
  <tbody>{''.join(rows)}</tbody>
</table>
<p class="caption">
  J_bar final = mean J̄ over last 10 cycles. Drift = final − J_init.
  Budget saturated = fraction of steps with mean budget_norm &gt; 0.9.
  W_net early/late = mean W_net per cycle over first/last 50 cycles.
  <span class="fail">Red</span> = diagnostic warning; <span class="pass">green</span> = healthy.
</p>
"""


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def generate_report(results_dir, exp1_dir=None, n_cycles=500):
    results_dir = Path(results_dir)

    # Collect runs
    runs = {}      # (lam, alpha) -> data dict
    configs = {}   # (lam, alpha) -> config dict
    run_dirs = {}  # (lam, alpha) -> Path

    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue
        parsed = _parse_run_name(subdir.name)
        if parsed is None:
            continue
        data = _load_analysis(subdir)
        if data is None:
            print(f'  [warn] no analysis.npz in {subdir}', file=sys.stderr)
            continue
        cfg = _load_config(subdir)
        runs[parsed]     = data
        configs[parsed]  = cfg
        run_dirs[parsed] = subdir

    if not runs:
        print(f'No valid runs found under {results_dir}', file=sys.stderr)
        sys.exit(1)

    print(f'Loaded {len(runs)} run(s).')

    # --- Summary stats ---
    runs_summary = []
    for key, data in runs.items():
        lam, alpha = key
        cfg  = configs[key]
        spc  = int(cfg.get('steps_per_cycle', 200))
        J_init_val = float(cfg.get('J_init', 0.92))
        T_mean     = float(cfg.get('T_mean', 2.5))

        J_bar = data['J_bar']
        T_tr  = data['T_trace']
        n_cycles_actual = len(J_bar) // spc

        J_final = float(np.mean(J_bar[-10 * spc:]))
        bud_norm_sat = float(np.mean(data['budget_norm_mean'] > 0.9))

        hot_mask  = T_tr > T_mean
        cold_mask = ~hot_mask
        n_gated   = data['n_gated']
        n_total   = data['n_applied_pos'] + data['n_applied_neg'] + n_gated

        def _frac_gated(mask):
            ng = n_gated[mask].sum()
            nt = n_total[mask].sum()
            return float(ng / nt) if nt > 0 else 0.0

        frac_gated_hot  = _frac_gated(hot_mask)
        frac_gated_cold = _frac_gated(cold_mask)

        Q_in  = _per_cycle(data['Q_in_step'],     spc).sum(axis=1)
        Q_out = _per_cycle(data['Q_out_step'],    spc).sum(axis=1)
        W_rem = _per_cycle(data['W_remodel_step'], spc).sum(axis=1)
        W_net_cycles = Q_out - Q_in - W_rem

        n_early = min(50, n_cycles_actual // 4)
        n_late  = min(50, n_cycles_actual // 4)
        W_net_early = float(W_net_cycles[:n_early].mean()) if n_early > 0 else float('nan')
        W_net_late  = float(W_net_cycles[-n_late:].mean()) if n_late  > 0 else float('nan')

        runs_summary.append(dict(
            name=f'λ={lam:.2f} α={alpha:.2f}',
            lam=lam, alpha=alpha,
            J_init=J_init_val, J_final=J_final,
            bud_norm_sat=bud_norm_sat,
            frac_gated_hot=frac_gated_hot, frac_gated_cold=frac_gated_cold,
            W_net_early=W_net_early, W_net_late=W_net_late,
        ))

    table_html = summary_table_html(runs_summary)

    # --- Interactive J_bar overview chart ---
    j_series = []
    for idx, (key, data) in enumerate(sorted(runs.items())):
        lam, alpha = key
        spc = int(configs[key].get('steps_per_cycle', 200))
        J_bar = data['J_bar']
        n_cycles_actual = len(J_bar) // spc
        # Per-cycle mean J_bar
        J_cycle = _per_cycle(J_bar, spc).mean(axis=1)
        j_series.append({
            'label': f'λ={lam:.2f} α={alpha:.2f}',
            'x': list(range(n_cycles_actual)),
            'y': [float(v) for v in J_cycle],
            'color': PALETTE[idx % len(PALETTE)],
        })

    j_chart_html = canvas_chart_html(
        j_series, 'exp1b_jbar',
        title='Mean J̄ per Cycle — all runs (hover to inspect)',
        xlabel='Cycle', ylabel='Mean J̄',
    )

    # --- W_net overview chart ---
    wnet_series = []
    for idx, (key, data) in enumerate(sorted(runs.items())):
        lam, alpha = key
        spc = int(configs[key].get('steps_per_cycle', 200))
        Q_in  = _per_cycle(data['Q_in_step'],     spc).sum(axis=1)
        Q_out = _per_cycle(data['Q_out_step'],    spc).sum(axis=1)
        W_rem = _per_cycle(data['W_remodel_step'], spc).sum(axis=1)
        W_net_cycles = Q_out - Q_in - W_rem
        wnet_series.append({
            'label': f'λ={lam:.2f} α={alpha:.2f}',
            'x': list(range(len(W_net_cycles))),
            'y': [float(v) for v in W_net_cycles],
            'color': PALETTE[idx % len(PALETTE)],
        })

    wnet_chart_html = canvas_chart_html(
        wnet_series, 'exp1b_wnet',
        title='W_net per Cycle — all runs (hover to inspect)',
        xlabel='Cycle', ylabel='W_net',
        baseline=0.0,
    )

    # --- Per-run scenario panels ---
    scenario_ids    = []
    scenario_labels = []
    scenario_panels = {}

    for idx, (key, data) in enumerate(sorted(runs.items())):
        lam, alpha = key
        cfg    = configs[key]
        label  = f'λ={lam:.2f}, α={alpha:.2f}'
        sid    = f'sc_lam{lam:.2f}_alpha{alpha:.2f}'.replace('.', 'p')
        scenario_ids.append(sid)
        scenario_labels.append(label)

        print(f'  Generating figures for {label}...')
        panel = f'<div class="run-panel">\n<h3>{label}</h3>\n'

        for fig_fn, alt, caption in [
            (lambda d=data, c=cfg, l=label: fig_J_drift(d, c, l),
             'J drift', 'Mean J̄ over all steps coloured by phase (blue=cold, red=hot). '
             'Horizontal lines mark J_c (critical coupling) and J_init. '
             'Upward drift means the controller is systematically strengthening bonds.'),

            (lambda d=data, c=cfg, l=label: fig_budget_dynamics(d, c, l),
             'Budget dynamics', 'Top: mean budget per bond over time. '
             'Bottom: mean tanh(B/B_scale) — values near 1.0 indicate saturation '
             'where the budget constraint is no longer active.'),

            (lambda d=data, c=cfg, l=label: fig_budget_norm_histogram(d, c, l),
             'Budget saturation', 'Distribution of budget_norm_mean across all steps. '
             'A peak near 1.0 confirms that the budget is saturated throughout the run, '
             'making the gating condition trivially satisfied.'),

            (lambda d=data, c=cfg, l=label: fig_decision_breakdown(d, c, l),
             'Decision breakdown', 'Fraction of proposed bond updates that are applied '
             '(positive dJ, negative dJ) or gated (budget < cost), split by phase. '
             'A healthy controller should show more negative dJ during the hot phase '
             'and more positive dJ during the cold phase.'),

            (lambda d=data, c=cfg, l=label: fig_dJ_signed(d, c, l),
             'Signed dJ', 'Per-cycle mean of proposed (left) and applied (right) dJ '
             'in hot vs cold phases. Divergence between hot and cold indicates '
             'phase-aware adaptation. Gap between proposed and applied shows gating effects.'),

            (lambda d=data, c=cfg, l=label: fig_net_J_per_cycle(d, c, l),
             'J drift decomposition', 'Left: net J̄ change per cycle (blue=positive, red=negative). '
             'Right: hot vs cold phase contributions to J changes. '
             'If cold dominates, the controller is strengthening more than weakening.'),

            (lambda d=data, c=cfg, l=label: fig_wnet_per_cycle(d, c, l),
             'W_net per cycle', 'W_net = Q_out - Q_in - W_remodel per cycle. '
             'Left: per-cycle trace with rolling average. Right: cumulative W_net. '
             'Degradation over cycles indicates the J drift is hurting performance.'),
        ]:
            try:
                f = fig_fn()
                panel += _img(fig_to_b64(f), alt, caption) + '\n'
            except Exception as e:
                print(f'    [warn] {alt}: {e}', file=sys.stderr)

        panel += '</div>\n'
        scenario_panels[sid] = panel

    best_sid = scenario_ids[0] if scenario_ids else ''
    selector_html = ''
    if scenario_ids:
        selector_html = scenario_selector_html(
            scenario_ids, scenario_labels, best_sid, title='Select Run'
        )
        for sid in scenario_ids:
            display = 'block' if sid == best_sid else 'none'
            selector_html += (
                f'<div id="{sid}" style="display:{display}" class="card">\n'
                + scenario_panels.get(sid, '')
                + '</div>\n'
            )

    # --- Assemble HTML ---
    spc_display = configs[next(iter(configs))].get('steps_per_cycle', 200) if configs else 200
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Experiment 1b — Long-Run Diagnostics</title>
  <style>{REPORT_CSS}</style>
</head>
<body>
<h1>Experiment 1b: Long-Run Controller Diagnostics</h1>
<p style="color:#555">
  Generated by <code>experiments/exp1b_report.py</code> &mdash;
  results from <code>{results_dir}</code><br>
  Each controller run for {n_cycles} cycles × {spc_display} steps.
</p>

<h2>1. What This Report Shows</h2>
<div class="card">
<p>
This report runs each exp1 controller for <strong>{n_cycles} temperature
cycles</strong> — far beyond the 10 cycles used during training fitness
evaluation — to reveal long-term behaviour that the training loop could
not observe.
</p>
<p>We diagnose two hypothesised failure modes:</p>
<ul>
  <li><strong>Budget saturation:</strong> because budget only grows from ordering
      events and has no decay, it may accumulate until
      <code>budget_norm = tanh(B/B_scale) ≈ 1.0</code> at all times.
      Once saturated, the budget provides no informative input to the controller
      and the gating condition <code>B ≥ cost</code> is trivially satisfied —
      the budget constraint is disabled.</li>
  <li><strong>J upward drift:</strong> budget is earned preferentially during
      the cold phase (ordering events), giving the controller more credits
      exactly when it wants to raise J. Hot-phase J decreases may be
      systematically smaller or more gated, producing a net upward drift in J̄.</li>
</ul>
<p>
The key diagnostic signatures to look for:
</p>
<ul>
  <li><code>budget_norm_mean &gt; 0.9</code> in the vast majority of steps → saturation confirmed.</li>
  <li>J̄ trending upward across cycles, approaching J_max → drift confirmed.</li>
  <li>Cold phase: high fraction of applied dJ &gt; 0; hot phase: expected high fraction
      of applied dJ &lt; 0, but if gating is trivial the fractions may be similar.</li>
  <li>W_net per cycle decreasing over time as J moves away from its optimal value.</li>
</ul>
</div>

{table_html}

<h2>3. Overview: J̄ per Cycle (all runs)</h2>
<div class="card">
  <p>Per-cycle mean J̄ for all controllers. An upward trend confirms the drift hypothesis.</p>
  {j_chart_html}
</div>

<h2>4. Overview: W_net per Cycle (all runs)</h2>
<div class="card">
  <p>Per-cycle W_net = Q_out − Q_in − W_remodel. A downward trend over cycles would
     indicate that J drift is degrading performance over time.</p>
  {wnet_chart_html}
</div>

<h2>5. Per-Run Detailed Analysis</h2>
<div class="card">
  <p>Select a run to see all diagnostic figures for that (λ, α) combination.</p>
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
        description='Generate HTML report for Experiment 1b (long-run diagnostics).'
    )
    parser.add_argument('--results-dir', default='results/exp1b',
                        help='exp1b output directory (default: results/exp1b)')
    parser.add_argument('--exp1-dir', default='results/exp1',
                        help='exp1 results directory for context (default: results/exp1)')
    parser.add_argument('--n-cycles', type=int, default=500,
                        help='Number of cycles used (for display only, default: 500)')
    args = parser.parse_args()
    generate_report(args.results_dir, args.exp1_dir, n_cycles=args.n_cycles)


if __name__ == '__main__':
    main()
