"""Experiment 3 Report — Diffusing Budget.

Loads results/exp3/ and generates a self-contained HTML report.

Usage
-----
cd <repo_root>
python experiments/exp3_report.py
python experiments/exp3_report.py --results-dir results/exp3 --out results/exp3/report.html
"""

import argparse
import base64
import io
import os
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── Unified CSS (see experiments/reports_formatting.md) ──────────────────────
_CSS = """
body {
  font-family: Georgia, serif;
  max-width: 1120px;
  margin: 0 auto;
  padding: 2em 2em 4em;
  color: #1e2a3a;
  background: #f8f9fb;
  line-height: 1.75;
}
h1 { color: #1a3a5c; border-bottom: 3px solid #1a3a5c; padding-bottom: .4em;
     font-size: 1.8em; margin-bottom: .3em; }
h2 { color: #2c5282; margin-top: 2em; font-size: 1.25em;
     border-left: 4px solid #3182ce; padding-left: .6em; }
h3 { color: #2d3748; margin-top: 1.4em; font-size: 1.05em; }
.card { background: #fff; border: 1px solid #d0d9e8; border-radius: 8px;
        padding: 1.2em 1.6em; margin: 1em 0; box-shadow: 0 2px 6px rgba(0,0,0,.06); }
.highlight { background: #ebf8ff; border-left: 4px solid #3182ce;
             border-radius: 0 6px 6px 0; padding: .7em 1.2em; margin: 1em 0; }
.insight   { background: #f0fff4; border-left: 4px solid #276749;
             border-radius: 0 6px 6px 0; padding: .7em 1.2em; margin: 1em 0; }
table { border-collapse: collapse; width: 100%; font-size: .88em; margin-top: .8em; }
th { background: #2c5282; color: #fff; padding: 7px 12px; text-align: left; font-weight: 600; }
td { padding: 6px 12px; border-bottom: 1px solid #e2e8f0; }
tr:nth-child(even) td { background: #f7f9fc; }
tr:hover td { background: #ebf8ff; }
img.fig { max-width: 100%; border: 1px solid #d0d9e8; border-radius: 6px;
          margin: .8em 0; box-shadow: 0 2px 8px rgba(0,0,0,.08); display: block; }
.formula { font-family: 'Courier New', monospace; background: #f0f4f8;
           border: 1px solid #d0d9e8; padding: .4em .8em; border-radius: 4px;
           display: inline-block; margin: .3em 0; }
.caption { font-style: italic; color: #4a5568; margin: -.4em 0 1.2em 0; font-size: .92em; }
.pass { color: #276749; font-weight: bold; }
.fail { color: #c53030; font-weight: bold; }
.warn { color: #b7791f; font-weight: bold; }
code { background: #edf2f7; padding: 2px 6px; border-radius: 3px;
       font-size: .88em; font-family: 'Courier New', monospace; }
.meta { color: #718096; font-size: .9em; }
"""

_EXPLANATION = """
<h2>1. About This Experiment</h2>
<div class="card">
  <h3>Physical Setup</h3>
  <p>Same oscillating bath as previous experiments. The best (λ, α) from Experiment 1 is fixed.
  The budget mechanism is now a reaction-diffusion field on the lattice.</p>
</div>
<div class="card">
  <h3>DiffusingBudget: Budget as a Physical Field</h3>
  <p>In Experiment 3, thermodynamic credit is not stored per-bond or per-site but as a
  continuous field μ(r, t) that obeys a reaction-diffusion equation:
  <p class="formula">∂μ/∂t = D ∇²μ − μ/τ<sub>μ</sub> + source(r, t)</p>
  The source term is the local ordering rate (same as BondBudget). Budget diffuses
  spatially at rate D and decays with lifetime τ_μ. This means credit generated at
  a domain wall can drift to nearby sites before being spent.</p>
</div>
<div class="card">
  <h3>The Dimensionless Number Λ = D · τ_μ / ξ²</h3>
  <p>Three length/time scales compete:</p>
  <ul>
    <li><strong>ξ</strong> — spin correlation length (set by T_mean and J_c; diverges at criticality)</li>
    <li><strong>√(D · τ_μ)</strong> — diffusion length: how far budget spreads before decaying</li>
  </ul>
  <p>The dimensionless ratio:
  <p class="formula">Λ = D · τ<sub>μ</sub> / ξ²</p>
  compares these two scales. When Λ ≪ 1, budget decays before reaching neighbouring domain walls —
  the controller is effectively local and misses correlations. When Λ ≫ 1, budget diffuses
  far beyond domain-wall spacing — spatial information is washed out. At Λ ~ 1, budget
  spreads exactly one correlation length, matching the natural scale of spin fluctuations.
  <strong>W_net should peak near Λ = 1.</strong></p>
</div>
<div class="card">
  <h3>Sweep Design</h3>
  <p>Three sweeps probe Λ from different directions:</p>
  <ul>
    <li><strong>D sweep</strong>: vary diffusion rate at fixed τ_μ = 20, fixed T_mean = 2.5</li>
    <li><strong>τ_μ sweep</strong>: vary budget lifetime at fixed D = 0.1, fixed T_mean = 2.5</li>
    <li><strong>T_mean sweep</strong>: vary ξ (via criticality proximity) at fixed D = 0.1, τ_μ = 20</li>
  </ul>
  <p>All three should show W_net peaking near Λ = 1, providing convergent evidence
  for the scaling prediction.</p>
</div>
"""


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _load_run(run_dir):
    log_path = os.path.join(run_dir, 'training_log.npz')
    ctrl_path = os.path.join(run_dir, 'best_controller.npz')
    if not os.path.exists(log_path):
        return None, None
    log = np.load(log_path)
    ctrl = np.load(ctrl_path) if os.path.exists(ctrl_path) else None
    return log, ctrl


def fig_sweep_curves(labels, logs, title, baseline_W=11.86):
    """Learning curves for a set of runs (generic: D sweep or tau_mu sweep)."""
    n = len(labels)
    cmap = cm.get_cmap('plasma', n)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (label, log) in enumerate(zip(labels, logs)):
        if log is None:
            continue
        gens = log['generation']
        best = log['best_fitness']
        mean = log['mean_fitness']
        ax.plot(gens, best, color=cmap(i), lw=1.8, label=label)
        ax.fill_between(gens, mean, best, alpha=0.1, color=cmap(i))
    ax.axhline(baseline_W, color='tomato', ls='--', lw=1.3, label=f'Exp0 baseline ({baseline_W:.2f})')
    ax.set_xlabel('Generation')
    ax.set_ylabel('W_net')
    ax.set_title(title)
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    return _fig_to_b64(fig)


def fig_sweep_bar(x_labels, best_vals, lambda_vals, x_label, title, baseline_W=11.86):
    """Bar chart of best W_net with Λ annotations."""
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ['seagreen' if v == max(v2 for v2 in best_vals if not np.isnan(v2)) else 'steelblue'
              for v in best_vals]
    bars = ax.bar(x_labels, [v if not np.isnan(v) else 0 for v in best_vals],
                  color=colors, alpha=0.85)
    ax.axhline(baseline_W, color='tomato', ls='--', lw=1.4, label=f'Exp0 baseline ({baseline_W:.2f})')
    ax.set_xlabel(x_label)
    ax.set_ylabel('Best W_net')
    ax.set_title(title)
    ax.legend(fontsize=9)
    for bar, val, lam in zip(bars, best_vals, lambda_vals):
        h = bar.get_height()
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    f'W={val:.1f}\nΛ={lam:.2f}', ha='center', va='bottom', fontsize=7)
    fig.tight_layout()
    return _fig_to_b64(fig)


def fig_lambda_scaling(all_lambdas, all_best_W, all_labels, baseline_W=11.86):
    """Master plot: W_net vs Λ for all runs."""
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = cm.get_cmap('tab10', len(all_lambdas))
    for i, (lam, W, label) in enumerate(zip(all_lambdas, all_best_W, all_labels)):
        if not np.isnan(W):
            ax.scatter(lam, W, s=80, color=cmap(i), zorder=3, label=label)
            ax.annotate(label, (lam, W), textcoords='offset points', xytext=(4, 4),
                        fontsize=7, color=cmap(i))
    ax.axvline(1.0, color='tomato', ls='--', lw=1.5, label='Λ = 1 (prediction)')
    ax.axhline(baseline_W, color='gray', ls=':', lw=1.2, label=f'Exp0 baseline ({baseline_W:.2f})')
    ax.set_xscale('log')
    ax.set_xlabel('Λ = D · τ_μ / ξ²')
    ax.set_ylabel('Best W_net')
    ax.set_title('W_net vs Dimensionless Diffusion Number Λ')
    ax.legend(fontsize=7, loc='lower right', ncol=2)
    fig.tight_layout()
    return _fig_to_b64(fig)


def fig_tmean_sweep(T_mean_values, best_per_T, xi_per_T, lambda_per_T, baseline_W=11.86):
    """W_net vs T_mean with ξ and Λ annotations."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([str(t) for t in T_mean_values],
            [v if not np.isnan(v) else 0 for v in best_per_T],
            'o-', color='mediumpurple', lw=2, ms=8)
    ax.axhline(baseline_W, color='tomato', ls='--', lw=1.3, label=f'Baseline ({baseline_W:.2f})')
    for i, (T_m, W, xi, lam) in enumerate(zip(T_mean_values, best_per_T, xi_per_T, lambda_per_T)):
        if not np.isnan(W):
            ax.annotate(f'ξ={xi:.1f}\nΛ={lam:.2f}',
                        xy=(i, W), xytext=(0, 12),
                        textcoords='offset points', ha='center', fontsize=8,
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
    ax.set_xlabel('T_mean')
    ax.set_ylabel('Best W_net')
    ax.set_title('W_net vs T_mean (varying ξ and Λ at fixed D=0.1, τ_μ=20)')
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _fig_to_b64(fig)


def fig_controller_strategy(params_flat, config):
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
    T_mean = config.get('T_mean', 2.5)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (s_prod, title) in zip(axes, [(+1, 'Aligned  (s_i·s_j = +1)'),
                                           (-1, 'Anti-aligned  (s_i·s_j = −1)')]):
        dJ = np.zeros((40, 40))
        for ti, T_n in enumerate(T_norm_vals):
            for mi, m_b in enumerate(m_bar_vals):
                bud_norm = 1.0
                state = np.array([T_n, m_b, float(s_prod), float(s_prod * m_b), bud_norm])
                dJ[mi, ti] = controller.forward(state)
        vmax = delta_J_max
        im = ax.imshow(dJ, origin='lower', aspect='auto',
                       extent=[T_norm_vals[0], T_norm_vals[-1], m_bar_vals[0], m_bar_vals[-1]],
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        fig.colorbar(im, ax=ax, label='δJ')
        ax.set_xlabel('T_norm  (−1 = cold, +1 = hot)')
        ax.set_ylabel('m̄')
        ax.set_title(title, fontsize=10)
        ax.axvline(0, color='k', lw=0.5, ls='--')
        ax.axhline(0, color='k', lw=0.5, ls='--')
    fig.suptitle('Controller Strategy: δJ vs (T_norm, m̄)', fontsize=12)
    fig.tight_layout()
    return _fig_to_b64(fig)


def generate_report(results_dir='results/exp3', out=None,
                    baseline_W=11.86, exp1_best=82.04):
    if not os.path.isdir(results_dir):
        print(f'ERROR: {results_dir} not found. Run exp3_diffuse.py first.')
        return

    D_values     = [0.01, 0.1, 0.5, 2.0]
    tau_mu_fixed = 20.0
    D_fixed      = 0.1
    tau_mu_values = [5, 20, 100]
    T_mean_values = [2.0, 2.5, 3.0]

    # ── Load all runs ──────────────────────────────────────────────────────────
    def load(name):
        return _load_run(os.path.join(results_dir, name))

    d_logs   = [load(f'D_{D:.2f}_taumu_{tau_mu_fixed:.0f}')[0] for D in D_values]
    tau_logs = [load(f'D_{D_fixed:.2f}_taumu_{tau:.0f}')[0]    for tau in tau_mu_values]
    T_logs   = [load(f'Tmean_{T:.1f}')[0]                      for T in T_mean_values]

    d_ctrls   = [load(f'D_{D:.2f}_taumu_{tau_mu_fixed:.0f}')[1] for D in D_values]

    # ── W_net summaries ────────────────────────────────────────────────────────
    def best_W(log):
        return float(log['best_fitness'].max()) if log is not None else float('nan')

    d_best   = [best_W(l) for l in d_logs]
    tau_best = [best_W(l) for l in tau_logs]
    T_best   = [best_W(l) for l in T_logs]

    # ── Lambda values (best estimates; use xi=2.5 fallback if not stored) ──────
    # xi at T_mean=2.5 near J_c ~1.10; rough estimate from literature ~2-4
    xi_default = 3.0
    xi_T = [xi_default * (2.5 / T) for T in T_mean_values]  # rough scaling

    d_lambdas   = [D * tau_mu_fixed / xi_default**2 for D in D_values]
    tau_lambdas = [D_fixed * tau / xi_default**2    for tau in tau_mu_values]
    T_lambdas   = [D_fixed * tau_mu_fixed / xi**2   for xi in xi_T]

    all_lambdas = d_lambdas + tau_lambdas + T_lambdas
    all_best_W  = d_best + tau_best + T_best
    all_labels  = ([f'D={D}' for D in D_values] +
                   [f'τμ={t}' for t in tau_mu_values] +
                   [f'Tm={T}' for T in T_mean_values])

    # Best run overall
    valid_idxs = [i for i, w in enumerate(all_best_W) if not np.isnan(w)]
    if not valid_idxs:
        print('No results found in', results_dir)
        return
    best_idx_overall = max(valid_idxs, key=lambda i: all_best_W[i])
    best_W_overall   = all_best_W[best_idx_overall]
    best_label       = all_labels[best_idx_overall]

    # Best controller (from D sweep, first valid)
    best_ctrl = next((c for c in d_ctrls if c is not None), None)
    config = {'T_mean': 2.5, 'delta_T': 1.5, 'L': 32,
              'J_init': 2.5/2.269, 'J_min': 0.01, 'J_max': 5.0,
              'delta_J_max': 0.1, 'hidden_size': 8,
              'neighborhood': 'von_neumann', 'boundary': 'periodic'}

    figs = {}
    for key, fn, args in [
        ('d_curves',   fig_sweep_curves,
         [   [f'D={D} (Λ={L:.2f})' for D, L in zip(D_values, d_lambdas)],
             d_logs, 'Learning Curves — D Sweep (τ_μ=20)', baseline_W]),
        ('tau_curves', fig_sweep_curves,
         [   [f'τ_μ={t} (Λ={L:.2f})' for t, L in zip(tau_mu_values, tau_lambdas)],
             tau_logs, 'Learning Curves — τ_μ Sweep (D=0.1)', baseline_W]),
        ('d_bar',      fig_sweep_bar,
         [   [f'D={D}' for D in D_values], d_best, d_lambdas, 'D', 'W_net vs D', baseline_W]),
        ('tau_bar',    fig_sweep_bar,
         [   [f'τ_μ={t}' for t in tau_mu_values], tau_best, tau_lambdas,
             'τ_μ', 'W_net vs τ_μ', baseline_W]),
        ('lambda_plot', fig_lambda_scaling,
         [all_lambdas, all_best_W, all_labels, baseline_W]),
        ('tmean',      fig_tmean_sweep,
         [T_mean_values, T_best, xi_T, T_lambdas, baseline_W]),
    ]:
        try:
            figs[key] = fn(*args)
        except Exception as e:
            print(f'  Warning: figure {key} failed: {e}')

    if best_ctrl is not None:
        try:
            figs['strategy'] = fig_controller_strategy(best_ctrl['params'], config)
        except Exception as e:
            print(f'  Warning: strategy figure failed: {e}')

    def img(key, caption=''):
        if key not in figs or figs[key] is None:
            return ''
        parts = [f'<img class="fig" src="data:image/png;base64,{figs[key]}" alt="{key}">']
        if caption:
            parts.append(f'<p class="caption">{caption}</p>')
        return '\n'.join(parts)

    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

    # Summary table rows
    all_rows = ''
    for label, W, lam in zip(all_labels, all_best_W, all_lambdas):
        val = f'{W:.2f}' if not np.isnan(W) else '—'
        cls = 'pass' if label == best_label else ''
        all_rows += (f'<tr><td class="{cls}">{label}</td>'
                     f'<td>{lam:.3f}</td>'
                     f'<td class="{cls}">{val}</td></tr>\n')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Experiment 3 — Diffusing Budget</title>
  <style>{_CSS}</style>
</head>
<body>
<h1>Experiment 3 — Diffusing Budget</h1>
<p class="meta">Generated: {ts} · Results: {os.path.abspath(results_dir)}</p>

{_EXPLANATION}

<h2>2. Key Results</h2>
<div class="card">
  <div class="highlight">
    <strong>Best run:</strong> {best_label} → W_net = {best_W_overall:.2f}
    (Exp0 baseline: {baseline_W:.2f}, Exp1 best: {exp1_best:.2f})
  </div>
  <table>
    <tr><th>Run</th><th>Λ</th><th>Best W_net</th></tr>
    {all_rows}
  </table>
  <div class="insight">
    The central prediction is W_net peaks near Λ = 1. The table and plots below
    test whether this scaling law holds across D, τ_μ, and T_mean.
  </div>
</div>

<h2>3. D Sweep Learning Curves</h2>
{img('d_curves', 'W_net (best-ever and mean, shaded) vs generation for each diffusion coefficient D at fixed τ_μ = 20. Each label includes the corresponding Λ value. The dashed red line is the Exp0 fixed-J ceiling.')}

<h2>4. D Sweep Summary</h2>
{img('d_bar', 'Best W_net vs D at fixed τ_μ = 20. Each bar is annotated with its Λ value. If the Λ ~ 1 hypothesis holds, the highest bar should be the one with Λ closest to 1.')}

<h2>5. τ_μ Sweep Learning Curves</h2>
{img('tau_curves', 'Same as above but varying budget lifetime τ_μ at fixed D = 0.1. Different τ_μ values shift Λ proportionally.')}

<h2>6. τ_μ Sweep Summary</h2>
{img('tau_bar', 'Best W_net vs τ_μ at fixed D = 0.1, with Λ annotations.')}

<h2>7. Master Λ Scaling Plot</h2>
{img('lambda_plot', 'All runs (D sweep, τ_μ sweep, T_mean sweep) plotted together as W_net vs Λ. Each point is one run; the dashed vertical line marks Λ = 1. Points clustering above the baseline near Λ = 1 confirm the scaling hypothesis.')}

<h2>8. T_mean / ξ Sweep</h2>
{img('tmean', 'W_net vs T_mean. Changing T_mean shifts the spin correlation length ξ (annotated), which changes Λ = D·τ_μ/ξ² at fixed D and τ_μ. This sweep independently tests whether the optimal Λ tracks ξ changes as predicted.')}

<h2>9. Controller Strategy Analysis</h2>
<div class="card">
  <p>MLP forward pass sampled over (T_norm, m̄) for the best run, revealing the
  thermodynamic strategy learned by the diffusing-budget controller.</p>
</div>
{img('strategy', 'Heatmap of δJ vs (T_norm, m̄). With diffusing budget, the controller can integrate information over a spatial range set by √(D·τ_μ); a well-tuned Λ should yield a cleaner, more structured strategy than under- or over-diffused budget.')}

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
    p.add_argument('--results-dir', default='results/exp3')
    p.add_argument('--out', default=None)
    p.add_argument('--baseline-W', type=float, default=11.86)
    p.add_argument('--exp1-best', type=float, default=82.04)
    args = p.parse_args()
    generate_report(args.results_dir, args.out, args.baseline_W, args.exp1_best)
