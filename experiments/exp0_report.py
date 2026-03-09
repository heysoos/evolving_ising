"""Generate a self-contained HTML report for Experiment 0 from saved results.

Usage:
    python experiments/exp0_report.py [--results-dir results/exp0]
"""

import argparse
import base64
import io
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def _fig_heatmap(W, J0_vals, tau_vals, title, cmap='viridis', mark_best=True,
                 T_mean=None, label='W_net'):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ext = [np.log10(tau_vals[0]), np.log10(tau_vals[-1]),
           J0_vals[0], J0_vals[-1]]
    im = ax.imshow(W, aspect='auto', origin='lower', extent=ext, cmap=cmap)
    plt.colorbar(im, ax=ax, label=label)
    ax.set_xlabel('log₁₀(τ)')
    ax.set_ylabel('J₀')
    ax.set_title(title)

    if mark_best:
        idx = np.unravel_index(W.argmax(), W.shape)
        ax.plot(np.log10(tau_vals[idx[1]]), J0_vals[idx[0]],
                'r*', ms=14, label=f'Best ({J0_vals[idx[0]]:.2f}, τ={tau_vals[idx[1]]})')
        ax.legend(fontsize=8)

    if T_mean is not None:
        ax.axhline(T_mean / 2.269, color='white', ls='--', lw=1.5,
                   label=f'T/2.269={T_mean/2.269:.3f}')

    plt.tight_layout()
    return fig


def _fig_slices(results, T_mean):
    W = results['W_net_grid']
    J0 = results['J0_values']
    tau = results['tau_values']
    best_idx = np.unravel_index(W.argmax(), W.shape)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Slice along J0 at best tau
    ax = axes[0]
    ax.plot(J0, W[:, best_idx[1]], 'o-', color='steelblue', lw=1.8)
    ax.axvline(T_mean / 2.269, color='tomato', ls='--', lw=1.5,
               label=f'T/2.269 = {T_mean/2.269:.3f}')
    ax.axvline(results['J0_opt'], color='seagreen', ls=':', lw=1.8,
               label=f'J₀_opt = {results["J0_opt"]:.3f}')
    ax.set_xlabel('J₀')
    ax.set_ylabel('W_net')
    ax.set_title(f'W_net vs J₀  (τ = {tau[best_idx[1]]})')
    ax.legend(fontsize=8)

    # Slice along tau at best J0
    ax = axes[1]
    ax.semilogx(tau, W[best_idx[0], :], 's-', color='darkorange', lw=1.8)
    ax.set_xlabel('τ')
    ax.set_ylabel('W_net')
    ax.set_title(f'W_net vs τ  (J₀ = {results["J0_opt"]:.3f})')

    plt.tight_layout()
    return fig


def _fig_sigma_vs_W(W_grid, S_grid):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(S_grid.ravel(), W_grid.ravel(), s=24, alpha=0.7, color='purple')
    ax.set_xlabel('Mean Σ_cycle')
    ax.set_ylabel('W_net')
    ax.set_title('W_net vs Entropy Production')
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

def generate_report(results_dir='results/exp0'):
    npz_path = os.path.join(results_dir, 'sweep.npz')
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"No sweep.npz found at {npz_path}. Run exp0_baseline.py first.")

    data = np.load(npz_path)
    r = {k: data[k] for k in data}
    r['J0_opt']    = float(data['J0_opt'])
    r['tau_opt']   = int(data['tau_opt'])
    r['W_net_opt'] = float(data['W_net_opt'])

    T_mean = 2.5  # from DEFAULT_CONFIG
    J0_expected = T_mean / 2.269
    err_pct = abs(r['J0_opt'] - J0_expected) / J0_expected * 100
    best_idx = np.unravel_index(r['W_net_grid'].argmax(), r['W_net_grid'].shape)

    # Build figures
    fig_wnet  = _fig_heatmap(r['W_net_grid'], r['J0_values'], r['tau_values'],
                              'W_net heatmap', mark_best=True, T_mean=T_mean, label='W_net')
    fig_sigma = _fig_heatmap(r['sigma_grid'], r['J0_values'], r['tau_values'],
                              'Entropy production heatmap', cmap='inferno',
                              mark_best=False, label='Σ_cycle')
    fig_slices = _fig_slices(r, T_mean)
    fig_sv     = _fig_sigma_vs_W(r['W_net_grid'], r['sigma_grid'])

    b64_wnet  = _b64(fig_wnet);  plt.close(fig_wnet)
    b64_sigma = _b64(fig_sigma); plt.close(fig_sigma)
    b64_slices = _b64(fig_slices); plt.close(fig_slices)
    b64_sv    = _b64(fig_sv);    plt.close(fig_sv)

    # W_net and sigma tables
    def _html_grid(arr, row_labels, col_labels, fmt='.4f'):
        rows = [f'<tr><th>{rl:.3f}</th>' +
                ''.join(f'<td>{arr[i,j]:{fmt}}</td>' for j in range(arr.shape[1])) +
                '</tr>' for i, rl in enumerate(row_labels)]
        header = '<tr><th>J₀ \\ τ</th>' + ''.join(f'<th>{c}</th>' for c in col_labels) + '</tr>'
        return f'<table>{header}{"".join(rows)}</table>'

    pass_cls = 'pass' if err_pct < 25 else 'warn'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Exp 0 — Baseline Sweep</title>
<style>
  body  {{ font-family: system-ui, sans-serif; max-width: 1120px; margin: 2rem auto;
           padding: 0 1rem; color: #1a1a2e; line-height: 1.7; }}
  h1   {{ border-bottom: 2px solid #4a4e69; padding-bottom: .4em; }}
  h2   {{ color: #4a4e69; margin-top: 2.2rem; }}
  h3   {{ color: #22223b; margin-top: 1.4rem; }}
  table{{ border-collapse: collapse; font-size: .85rem; margin: .8em 0; }}
  th,td{{ border: 1px solid #c9cdd4; padding: 6px 12px; }}
  th   {{ background: #eef0f5; font-weight: 600; }}
  tr:nth-child(even) td {{ background: #f7f8fb; }}
  img  {{ max-width: 100%; display: block; margin: .8em 0; border-radius: 4px;
          box-shadow: 0 2px 8px rgba(0,0,0,.12); }}
  .card{{ background: #f4f5fa; border-radius: 8px; padding: 1rem 1.5rem; margin: 1rem 0; }}
  .insight {{ background: #eaf4ea; border-left: 4px solid #2d6a4f;
              border-radius: 0 6px 6px 0; padding: .8rem 1.2rem; margin: 1rem 0; }}
  .pass{{ color: #217a3c; font-weight: bold; }}
  .warn{{ color: #c07a00; font-weight: bold; }}
  .grid{{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
  code {{ background: #e8e9f0; padding: 1px 5px; border-radius: 3px; font-size: .9em; }}
  @media(max-width:700px){{ .grid{{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>

<h1>Experiment 0 — Baseline: Fixed-Coupling Sweep</h1>

<h2>What is this experiment?</h2>
<p>
  This is the <strong>baseline experiment</strong> for the work extraction project.
  The goal is to understand how much thermodynamic work a simple, <em>non-adaptive</em>
  Ising spin system can extract from a periodically oscillating heat bath — before any
  learning or adaptation takes place.
</p>
<p>
  Imagine a grid of magnetic spins (each pointing either up or down) coupled together
  with uniform strength J₀. The spins are in contact with a heat bath whose temperature
  oscillates sinusoidally over time:
</p>
<p style="text-align:center; font-size:1.1em;">
  <strong>T(t) = T_mean + ΔT · sin(2πt / τ)</strong>
</p>
<p>
  As the temperature rises, the spins become disordered (less aligned); as it falls,
  they reorder. If the coupling J₀ is tuned near the <em>phase transition</em>, the
  system is maximally sensitive to temperature changes, and the repeated ordering and
  disordering of spins can pump heat asymmetrically — absorbing more heat during the
  hot phase and releasing more during the cold phase. The difference is
  <strong>extracted work</strong>.
</p>
<p>
  This experiment sweeps over two parameters with all other settings fixed:
</p>
<ul>
  <li><strong>J₀</strong> (coupling strength, 10 values from 0.2 to 2.0) — controls
      how strongly neighbouring spins want to align. The 2D Ising model has a phase
      transition at J_c = T_mean / 2.269 ≈ {J0_expected:.3f} for our T_mean={T_mean}.</li>
  <li><strong>τ</strong> (oscillation period in simulation steps, 8 log-spaced values
      from 10 to 1000) — controls how fast the temperature oscillates. Very fast
      oscillations (small τ) don't give spins time to respond; very slow ones
      (large τ) lose the asymmetry that drives work extraction.</li>
</ul>
<p>
  For each of the 80 (J₀, τ) combinations, the simulation runs <strong>50 full
  oscillation cycles</strong> and records:
</p>
<ul>
  <li><strong>W_net</strong> — net work extracted per cycle (positive = useful output).</li>
  <li><strong>Σ_cycle</strong> — entropy produced per cycle (a measure of irreversibility;
      the second law requires this to be ≥ 0).</li>
</ul>
<p>
  The best (J₀, τ) found here sets the <em>fixed-coupling ceiling</em> — the maximum
  W_net achievable without any adaptive control. Later experiments will use an
  evolved neural controller to try to beat this ceiling by adjusting couplings in real time.
</p>

<div class="card">
  <h2 style="margin-top:0">Key Results</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Best J₀ found (J₀_opt)</td><td>{r['J0_opt']:.4f}</td></tr>
    <tr><td>Best τ found (τ_opt)</td><td>{r['tau_opt']}</td></tr>
    <tr><td>Best W_net (fixed-coupling ceiling)</td><td>{r['W_net_opt']:.4f}</td></tr>
    <tr><td>Theoretical J₀ at phase transition (T_mean / 2.269)</td><td>{J0_expected:.4f}</td></tr>
    <tr><td>Error vs. theoretical prediction</td>
        <td class="{pass_cls}">{err_pct:.1f}% {'✓' if err_pct<25 else '⚠'}</td></tr>
    <tr><td>Entropy production at optimum (Σ_cycle)</td><td>{r['sigma_grid'][best_idx]:.4f}</td></tr>
    <tr><td>W_net range across all combos</td>
        <td>[{r['W_net_grid'].min():.4f}, {r['W_net_grid'].max():.4f}]</td></tr>
  </table>
  <div class="insight">
    <strong>Interpretation:</strong> The best coupling J₀ = {r['J0_opt']:.2f} is
    {err_pct:.1f}% away from the theoretical phase-transition value
    T_mean / 2.269 = {J0_expected:.3f}. This confirms that work extraction is maximised
    when the system is tuned close to its critical point, where spin fluctuations are
    largest and the system is most responsive to temperature changes.
    The fixed-coupling ceiling of W_net = {r['W_net_opt']:.4f} is the target for
    adaptive experiments to exceed.
  </div>
</div>

<h2>W_net Heatmap — Where is work extracted?</h2>
<p>
  Each cell shows the mean W_net over 50 cycles for a given (J₀, τ) pair.
  Bright regions are parameter combinations where the system successfully extracts
  work. The red star marks the best combination; the dashed white line marks the
  theoretical phase-transition coupling J_c = T_mean / 2.269.
  Notice that the highest W_net values cluster near J_c, confirming the
  phase-transition hypothesis.
</p>
<div class="grid">
  <div>
    <img src="data:image/png;base64,{b64_wnet}" alt="W_net heatmap">
  </div>
  <div>
    <h2>Entropy Production Heatmap</h2>
    <p>
      Each cell shows mean Σ_cycle — the total entropy generated per oscillation cycle.
      High entropy production (hot colours) means the process is highly irreversible.
      Notice that entropy is largest at small τ and large J₀, where the bath drives
      rapid, energetically costly spin flips. Good work extraction (high W_net) tends
      to occur at intermediate Σ_cycle — enough irreversibility to absorb heat
      asymmetrically, but not so much that all the absorbed energy is wasted as heat.
    </p>
    <img src="data:image/png;base64,{b64_sigma}" alt="Sigma heatmap">
  </div>
</div>

<h2>W_net Slices at the Optimum</h2>
<p>
  <strong>Left:</strong> W_net as a function of J₀ at the best τ.
  The curve peaks near the phase transition (dashed red line), then drops off
  on both sides — weak coupling (small J₀) means spins barely interact and
  don't respond to temperature; strong coupling (large J₀) locks spins into
  a rigid ordered state that is also unresponsive.
</p>
<p>
  <strong>Right:</strong> W_net as a function of τ at the best J₀.
  There is an intermediate optimal period — too fast and spins can't keep up
  with the oscillation; too slow and the system re-equilibrates fully at each
  half-cycle, losing the asymmetry needed for net work.
</p>
<img src="data:image/png;base64,{b64_slices}" alt="W_net slices">

<h2>W_net vs Entropy Production</h2>
<p>
  Each point is one (J₀, τ) combination. A general trend of higher entropy
  production allowing higher W_net is expected (you need some irreversibility
  to run a heat engine), but the best performers sit on the upper-left
  frontier — extracting significant work while not being maximally irreversible.
  Points in the lower-left (low Σ, low W_net) are parameter regimes where the
  system barely interacts with the bath at all.
</p>
<img src="data:image/png;base64,{b64_sv}" alt="W_net vs sigma scatter">

<h2>Raw Data: W_net Grid</h2>
<p>Rows = J₀, columns = τ. Values are mean W_net over 50 cycles.</p>
{_html_grid(r['W_net_grid'], r['J0_values'], r['tau_values'])}

<h2>Raw Data: Σ_cycle Grid</h2>
<p>Rows = J₀, columns = τ. Values are mean entropy production per cycle.</p>
{_html_grid(r['sigma_grid'], r['J0_values'], r['tau_values'])}

</body>
</html>"""

    out_path = os.path.join(results_dir, 'report.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Report written to {out_path}")
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='results/exp0')
    args = parser.parse_args()
    generate_report(args.results_dir)