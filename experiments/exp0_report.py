"""Experiment 0 Report — Baseline Fixed-Coupling Sweep.

Loads results/exp0/sweep.npz and generates a self-contained HTML report.

Usage
-----
cd <repo_root>
python experiments/exp0_report.py                        # writes results/exp0/report.html
python experiments/exp0_report.py --results-dir results/exp0 --out results/exp0/report.html
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
  <p>The 2D Ising model sits in a spatially uniform, time-oscillating heat bath:
  <p class="formula">T(t) = T<sub>mean</sub> + ΔT · sin(2π t / τ)</p>
  with T<sub>mean</sub> = 2.5, ΔT = 1.5, giving T ∈ [1.0, 4.0]. The lattice is
  32 × 32 (1 024 spins) with von Neumann neighbourhood and periodic boundaries.
  Spin dynamics follow Metropolis-Hastings.</p>
</div>
<div class="card">
  <h3>What is Fixed-J?</h3>
  <p>Every coupling is a single constant <em>J₀</em>:
  <p class="formula">J<sub>nk</sub> = J₀ · mask<sub>nk</sub></p>
  No adaptation occurs. This establishes the ceiling that an adaptive controller
  must beat.</p>
</div>
<div class="card">
  <h3>Why does J₀ matter?</h3>
  <p>The 2D Ising model has a phase transition at J<sub>c</sub> = T/(2 ln(1+√2)) ≈ T/2.269.
  At T<sub>mean</sub> = 2.5, J<sub>c</sub> ≈ 1.10. Couplings near J<sub>c</sub> produce
  domain walls that are most responsive to temperature oscillations, maximising the
  heat that can be rectified into net work. Too low → disordered (no structure to exploit);
  too high → frozen ferromagnet (no dynamics).</p>
</div>
<div class="card">
  <h3>What is W_net?</h3>
  <p>Over one temperature cycle the system absorbs heat Q<sub>in</sub> from the hot phase
  and rejects Q<sub>out</sub> in the cold phase. The net extracted work is:
  <p class="formula">W<sub>net</sub> = Q<sub>out</sub> − Q<sub>in</sub></p>
  W<sub>net</sub> &gt; 0 means the system acts as a heat engine (second law is never violated:
  entropy production Σ ≥ 0 always).</p>
  <p>τ sets the oscillation period. Very fast oscillations (small τ) do not allow the
  spin system to equilibrate; very slow ones dissipate more entropy. The optimum lies
  somewhere in between.</p>
</div>
"""


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def fig_heatmap(results):
    W = results['W_net_grid']
    S = results['sigma_grid']
    J0 = results['J0_values']
    tau = results['tau_values']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, data, label, cmap in zip(
        axes,
        [W, S],
        ['W_net', 'Σ_cycle (entropy production)'],
        ['viridis', 'plasma'],
    ):
        im = ax.imshow(
            data,
            aspect='auto', origin='lower',
            extent=[np.log10(tau[0]), np.log10(tau[-1]), J0[0], J0[-1]],
            cmap=cmap,
        )
        plt.colorbar(im, ax=ax, label=label)
        ax.set_xlabel('log₁₀(τ)')
        ax.set_ylabel('J₀')
        ax.set_title(label)

        # Mark optimum
        best = np.unravel_index(data.argmax(), data.shape)
        ax.plot(np.log10(tau[best[1]]), J0[best[0]], 'r*', markersize=14,
                label=f'optimum J₀={J0[best[0]]:.2f}')
        ax.axhline(2.5 / 2.269, color='white', ls='--', lw=1.4,
                   label='J_c = T_mean/2.269')
        ax.legend(fontsize=8)

    fig.tight_layout()
    return _fig_to_b64(fig)


def fig_slices(results):
    W = results['W_net_grid']
    J0 = results['J0_values']
    tau = results['tau_values']
    T_mean = 2.5
    J_c = T_mean / 2.269

    best_idx = np.unravel_index(W.argmax(), W.shape)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(J0, W[:, best_idx[1]], 'o-', color='steelblue', lw=1.8, ms=4)
    ax.axvline(J_c, color='tomato', ls='--', lw=1.5, label=f'J_c = {J_c:.2f}')
    ax.axvline(results['J0_opt'], color='seagreen', ls=':', lw=1.8,
               label=f"J0_opt = {results['J0_opt']:.2f}")
    ax.set_xlabel('J₀')
    ax.set_ylabel('W_net')
    ax.set_title(f'W_net vs J₀  (τ = {results["tau_opt"]})')
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.semilogx(tau, W[best_idx[0], :], 's-', color='darkorange', lw=1.8, ms=5)
    ax.set_xlabel('τ')
    ax.set_ylabel('W_net')
    ax.set_title(f'W_net vs τ  (J₀ = {results["J0_opt"]:.2f})')
    ax.axvline(results['tau_opt'], color='tomato', ls='--', lw=1.5,
               label=f'τ_opt = {int(results["tau_opt"])}')
    ax.legend(fontsize=8)

    fig.tight_layout()
    return _fig_to_b64(fig)


def fig_scatter(results):
    W = results['W_net_grid'].ravel()
    S = results['sigma_grid'].ravel()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(S, W, s=18, alpha=0.6, color='mediumpurple', edgecolors='none')
    ax.set_xlabel('Σ_cycle (entropy production per cycle)')
    ax.set_ylabel('W_net')
    ax.set_title('W_net vs Entropy Production')
    ax.axhline(0, color='k', lw=0.8, ls='--')
    fig.tight_layout()
    return _fig_to_b64(fig)


def generate_report(results_dir='results/exp0', out=None):
    npz_path = os.path.join(results_dir, 'sweep.npz')
    if not os.path.exists(npz_path):
        print(f'ERROR: {npz_path} not found. Run exp0_baseline.py first.')
        return

    data = np.load(npz_path)
    results = {k: data[k] for k in data}

    T_mean = 2.5
    J_c = T_mean / 2.269
    J0_opt = float(results['J0_opt'])
    tau_opt = int(results['tau_opt'])
    W_opt = float(results['W_net_opt'])
    err_pct = abs(J0_opt - J_c) / J_c * 100
    pass_cls = 'pass' if err_pct < 25 else ('warn' if err_pct < 50 else 'fail')

    figs = {}
    for key, fn, args in [
        ('heatmap', fig_heatmap, (results,)),
        ('slices',  fig_slices,  (results,)),
        ('scatter', fig_scatter, (results,)),
    ]:
        try:
            figs[key] = fn(*args)
        except Exception as e:
            print(f'  Warning: figure {key} failed: {e}')

    def img(key, caption=''):
        if key not in figs:
            return ''
        parts = [f'<img class="fig" src="data:image/png;base64,{figs[key]}" alt="{key}">']
        if caption:
            parts.append(f'<p class="caption">{caption}</p>')
        return '\n'.join(parts)

    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Experiment 0 — Baseline: Fixed-Coupling Sweep</title>
  <style>{_CSS}</style>
</head>
<body>
<h1>Experiment 0 — Baseline: Fixed-Coupling Sweep</h1>
<p class="meta">Generated: {ts} · Results: {os.path.abspath(results_dir)}</p>

{_EXPLANATION}

<h2>2. Key Results</h2>
<div class="card">
  <div class="highlight">
    <strong>Optimum found:</strong>
    J₀_opt = {J0_opt:.4f},  τ_opt = {tau_opt},  W_net_opt = {W_opt:.4f}
  </div>
  <table>
    <tr><th>Quantity</th><th>Value</th><th>Target</th><th>Status</th></tr>
    <tr><td>J₀_opt</td><td>{J0_opt:.4f}</td><td>J_c = {J_c:.4f}</td>
        <td class="{pass_cls}">{err_pct:.1f}% error {'✓' if err_pct < 25 else '⚠'}</td></tr>
    <tr><td>W_net_opt</td><td>{W_opt:.4f}</td><td>&gt; 0</td>
        <td class="{'pass' if W_opt > 0 else 'fail'}">{'✓' if W_opt > 0 else '✗'}</td></tr>
    <tr><td>τ_opt</td><td>{tau_opt}</td><td>—</td><td>—</td></tr>
  </table>
  <div class="insight">
    <strong>Baseline ceiling:</strong> W_net = {W_opt:.4f}. An adaptive controller
    must exceed this to demonstrate genuine thermodynamic advantage.
  </div>
</div>

<h2>3. W_net and Entropy Heatmaps</h2>
{img('heatmap', 'Left: net extracted work W_net across the J₀ × τ parameter grid. Right: entropy production Σ_cycle. The white dashed line marks J_c = T_mean/2.269; the red star marks the optimum.')}

<h2>4. Slices at the Optimum</h2>
{img('slices', 'Left: W_net vs J₀ at the optimal τ — the peak near J_c is the critical-point enhancement. Right: W_net vs τ at the optimal J₀ — intermediate τ values balance equilibration time against dissipation.')}

<h2>5. W_net vs Entropy Production</h2>
{img('scatter', 'Each point is one (J₀, τ) pair. High-work configurations tend to produce moderate entropy — extreme work extraction is always accompanied by some irreversibility.')}

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
    p.add_argument('--results-dir', default='results/exp0')
    p.add_argument('--out', default=None)
    args = p.parse_args()
    generate_report(args.results_dir, args.out)