"""
evolving_ising.viz
==================
All visualisation and report-generation code.

Design constraints
------------------
- **No JAX dependency.**  All inputs are plain numpy arrays.  This means
  this module can be imported in an analysis notebook without triggering
  GPU initialisation or JAX compilation.
- Functions that produce images for the HTML report return base64-encoded
  PNG strings (via `fig_to_base64`) so the report is fully self-contained
  and portable (no external image files needed).
- `save_fitness_plot` writes a richer version of the fitness curve directly
  to disk alongside each experiment's other output files.
- `generate_report` assembles the full HTML document from
  `List[ExperimentResult]` and a run-configuration dict.
"""
from __future__ import annotations

import base64
import io
import os
from typing import Dict, List, TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; safe for headless/WSL use
import matplotlib.pyplot as plt
import numpy as np

# TYPE_CHECKING guard avoids a circular import at runtime:
# experiment.py imports from viz.py, so we cannot import ExperimentResult
# at module level here without creating a cycle.  The import only runs
# during static type analysis (e.g. mypy), not at runtime.
if TYPE_CHECKING:
    from .experiment import ExperimentResult


# ── Low-level helper ─────────────────────────────────────────────────────────

def fig_to_base64(fig) -> str:
    """
    Render a matplotlib Figure to a base64-encoded PNG string and close it.

    Used to embed images inline in the HTML report as data URIs:
        <img src="data:image/png;base64,{fig_to_base64(fig)}">

    The figure is closed after encoding to free memory.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ── Fitness plots ─────────────────────────────────────────────────────────────

def plot_fitness_curve(best_hist: List[float], best_so_far: List[float], title: str) -> str:
    """
    Compact fitness curve for embedding in the HTML report.

    Shows two series:
    - "best so far": monotonically non-decreasing global best — the true
      learning curve showing optimisation progress.
    - "best (iter)": best individual in each generation's population — noisier
      because a lucky sample can temporarily exceed the running best.

    Returns
    -------
    str  base64-encoded PNG
    """
    fig, ax = plt.subplots(figsize=(5, 3))
    iters = np.arange(1, len(best_hist) + 1)
    ax.plot(iters, best_so_far, linewidth=1.5, label="best so far")
    ax.plot(iters, best_hist,   linewidth=0.8, alpha=0.5, label="best (iter)")
    ax.set_xlabel("CMA-ES Iteration")
    ax.set_ylabel("Fitness")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig_to_base64(fig)


def save_fitness_plot(
    best_hist:   List[float],
    best_so_far: List[float],
    mean_hist:   List[float],
    std_hist:    List[float],
    title:       str,
    path:        str,
) -> None:
    """
    Save a detailed fitness statistics plot to disk as a PNG.

    Richer than `plot_fitness_curve` — adds the population mean and
    mean ± std shading so you can see whether the whole distribution is
    improving or just getting lucky outliers.

    Parameters
    ----------
    best_hist   : per-iteration best (noisy)
    best_so_far : running global best (monotone)
    mean_hist   : per-iteration population mean
    std_hist    : per-iteration population standard deviation
    title       : plot title
    path        : full file path to write (e.g. ".../fitness_stats.png")
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    iters    = np.arange(1, len(best_hist) + 1)
    mean_arr = np.array(mean_hist)
    std_arr  = np.array(std_hist)

    ax.plot(iters, best_so_far, linewidth=1.5, label="best so far")
    ax.plot(iters, best_hist,   linewidth=0.8, alpha=0.6, label="best (iter)")
    ax.fill_between(iters, mean_arr - std_arr, mean_arr + std_arr, alpha=0.2, label="mean ± std")
    ax.plot(iters, mean_arr, linewidth=0.8, linestyle="--", label="mean")

    ax.set_xlabel("CMA-ES Iteration")
    ax.set_ylabel("Fitness")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ── Per-experiment figures ────────────────────────────────────────────────────

def plot_temperature(T_final: np.ndarray, H: int, W: int, title: str, hot_t: float, cold_t: float) -> str:
    """
    Heatmap of the final temperature field.

    The colormap is fixed to [cold_t, hot_t] across all experiments so
    that temperatures are visually comparable between objectives.

    Returns
    -------
    str  base64-encoded PNG
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(T_final.reshape(H, W), cmap="magma", vmin=cold_t, vmax=hot_t)
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_spins(S_final: np.ndarray, H: int, W: int, title: str) -> str:
    """
    Grayscale image of the final spin configuration.

    Black = spin -1, white = spin +1.  Ordered (ferromagnetic) regions
    appear as uniform patches; disordered (paramagnetic) regions are speckled.

    Returns
    -------
    str  base64-encoded PNG
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(S_final.reshape(H, W), cmap="gray", vmin=-1, vmax=1)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_connectivity(
    J_nk:      np.ndarray,   # (N, K) float32
    mask:      np.ndarray,   # (N, K) bool
    neighbors: np.ndarray,   # (N, K) int32
    H: int,
    W: int,
    title: str,
) -> str:
    """
    Three-panel visualisation of the learned coupling structure.

    Panels
    ------
    In-degree  (Reds):   total incoming conductance at each site
                         Σ_{j: j→i} |J[j, k(j→i)]|
    Out-degree (Greens): total outgoing conductance from each site
                         Σ_k |J[i,k]| (valid slots only)
    Composite  (RGB):    R=normalised in-degree, G=normalised out-degree, B=0
                         Bright red sites are heat sinks; bright green are sources.

    The in-degree is computed with `np.add.at` (unbuffered scatter-add),
    which correctly accumulates when multiple sources share the same target.
    Invalid bond slots have W_plot[i,k]=0 (masked), so they contribute 0 even
    though their neighbor index may be 0.

    Returns
    -------
    str  base64-encoded PNG
    """
    N = H * W
    # Zero out invalid neighbour slots before computing degrees
    W_plot  = np.abs(J_nk) * mask.astype(np.float32)   # (N, K)
    out_deg = np.sum(W_plot, axis=1)                     # (N,)

    # Scatter-accumulate incoming weights: for each directed edge (i→j)
    # with weight w, add w to in_deg[j].
    dest    = neighbors.reshape(-1)       # flat target indices
    weights = W_plot.reshape(-1)          # corresponding weights
    in_deg  = np.zeros(N, dtype=np.float32)
    np.add.at(in_deg, dest, weights)     # unbuffered accumulation (handles duplicates)

    in_norm  = in_deg  / (np.max(in_deg)  + 1e-8)
    out_norm = out_deg / (np.max(out_deg) + 1e-8)
    RGB = np.stack([in_norm, out_norm, np.zeros(N)], axis=-1).reshape(H, W, 3)

    fig, axs = plt.subplots(1, 3, figsize=(10, 3.5))
    axs[0].imshow(in_deg.reshape(H, W),  cmap="Reds");   axs[0].set_title("In-degree");   axs[0].axis("off")
    axs[1].imshow(out_deg.reshape(H, W), cmap="Greens"); axs[1].set_title("Out-degree");  axs[1].axis("off")
    axs[2].imshow(RGB);                                   axs[2].set_title("R=in, G=out"); axs[2].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_comparison(
    all_T_finals: Dict[str, np.ndarray],
    H: int,
    W: int,
    hot_t: float,
    cold_t: float,
) -> str:
    """
    Side-by-side temperature heatmaps for all experiments on one row.

    All panels share the same colormap limits [cold_t, hot_t] so the
    differences in heat transport between objectives are directly visible.

    Returns
    -------
    str  base64-encoded PNG
    """
    n = len(all_T_finals)
    fig, axs = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axs = [axs]   # keep iterable even for single experiment
    for ax, (name, T_final) in zip(axs, all_T_finals.items()):
        ax.imshow(T_final.reshape(H, W), cmap="magma", vmin=cold_t, vmax=hot_t)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle("Final Temperature Comparison", fontsize=13)
    fig.tight_layout()
    return fig_to_base64(fig)


# ── HTML report ───────────────────────────────────────────────────────────────

def generate_report(
    results:    List[ExperimentResult],
    run_dir:    str,
    H:          int,
    W:          int,
    hot_t:      float,
    cold_t:     float,
    run_config: dict,
) -> None:
    """
    Generate a self-contained HTML report for the full experiment run.

    The report embeds all figures as base64 PNGs (no external files needed)
    and includes a summary table for quick cross-experiment comparison.

    The EXPERIMENTS import is deferred to avoid a circular import:
    experiment.py → objectives.py (for EXPERIMENTS), and viz.py is
    imported by experiment.py.  Importing at the top of this module would
    create a cycle.  Since generate_report is only called at the end of a
    run (after all modules are already loaded), the deferred import is safe.

    Parameters
    ----------
    results    : list of ExperimentResult from run_experiment()
    run_dir    : directory where report.html is written
    H, W       : grid dimensions (for reshaping flat arrays)
    hot_t      : maximum temperature (colormap upper bound)
    cold_t     : minimum temperature (colormap lower bound)
    run_config : dict of config key→value pairs shown in the report header
    """
    from .experiment import EXPERIMENTS  # deferred to break circular import

    experiment_sections = []
    all_T_finals = {}

    for r in results:
        info = EXPERIMENTS[r.name]

        # Generate the four per-experiment figures
        fig_fitness = plot_fitness_curve(r.best_hist, r.best_so_far, f"Fitness — {info['title']}")
        fig_temp    = plot_temperature(r.T_final, H, W, f"Temperature — {r.name}", hot_t, cold_t)
        fig_spins   = plot_spins(r.S_final, H, W, f"Spins — {r.name}")
        fig_conn    = plot_connectivity(r.J_best, r.mask, r.neighbors, H, W, f"Connectivity — {r.name}")
        all_T_finals[r.name] = r.T_final

        experiment_sections.append(f"""
        <div class="experiment-card">
            <h2>{info['title']}</h2>
            <p class="desc">{info['description']}</p>
            <p class="formula"><code>{info['formula']}</code></p>
            <div class="metrics">
                <span>Best fitness: <strong>{r.best_fitness:.6f}</strong></span>
                <span>Top-row temp: <strong>{r.top_row_temp:.4f}</strong></span>
                <span>Mean temp: <strong>{r.mean_temp:.4f}</strong></span>
                <span>Runtime: <strong>{r.elapsed:.1f}s</strong></span>
            </div>
            <div class="grid2x2">
                <div><img src="data:image/png;base64,{fig_fitness}" alt="Fitness"></div>
                <div><img src="data:image/png;base64,{fig_temp}" alt="Temperature"></div>
                <div><img src="data:image/png;base64,{fig_spins}" alt="Spins"></div>
                <div><img src="data:image/png;base64,{fig_conn}" alt="Connectivity"></div>
            </div>
        </div>""")

    fig_comparison = plot_comparison(all_T_finals, H, W, hot_t, cold_t)

    table_rows = "".join(
        f"<tr><td>{r.name}</td><td>{r.best_fitness:.6f}</td>"
        f"<td>{r.top_row_temp:.4f}</td><td>{r.mean_temp:.4f}</td><td>{r.elapsed:.1f}s</td></tr>"
        for r in results
    )

    config_items = " | ".join(f"<code>{k}</code>: <code>{v}</code>" for k, v in run_config.items())

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Evolving Ising — Loss Function Comparison</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           background: #0d1117; color: #c9d1d9; padding: 2rem; line-height: 1.6; }}
    h1 {{ color: #58a6ff; margin-bottom: 0.5rem; font-size: 1.8rem; }}
    h2 {{ color: #58a6ff; margin-bottom: 0.5rem; font-size: 1.3rem; }}
    .subtitle {{ color: #8b949e; margin-bottom: 2rem; }}
    .experiment-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                        padding: 1.5rem; margin-bottom: 2rem; }}
    .desc    {{ color: #8b949e; margin-bottom: 0.3rem; }}
    .formula {{ color: #d2a8ff; margin-bottom: 0.8rem; }}
    .formula code {{ background: #1c2028; padding: 2px 6px; border-radius: 4px; }}
    .metrics {{ display: flex; gap: 2rem; flex-wrap: wrap; margin-bottom: 1rem;
                font-size: 0.9rem; color: #8b949e; }}
    .metrics strong {{ color: #c9d1d9; }}
    .grid2x2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; }}
    .grid2x2 img {{ width: 100%; border-radius: 4px; }}
    .panel {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
              padding: 1.5rem; margin-bottom: 2rem; }}
    .panel img {{ width: 100%; border-radius: 4px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
    th, td {{ padding: 0.6rem 1rem; text-align: left; border-bottom: 1px solid #30363d; }}
    th {{ color: #58a6ff; font-weight: 600; }}
    .config {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
               padding: 1.5rem; margin-bottom: 2rem; font-size: 0.85rem; color: #8b949e; }}
    .config code {{ color: #d2a8ff; }}
</style>
</head>
<body>
<h1>Evolving Ising — Loss Function Comparison</h1>
<p class="subtitle">CMA-ES optimization with different fitness objectives on a {H}×{W} Ising lattice</p>

<div class="config"><strong>Configuration:</strong> {config_items}</div>

{"".join(experiment_sections)}

<div class="panel">
    <h2>Side-by-Side Temperature Comparison</h2>
    <img src="data:image/png;base64,{fig_comparison}" alt="Comparison">
</div>

<div class="panel">
    <h2>Summary Table</h2>
    <table>
        <tr><th>Experiment</th><th>Best Fitness</th><th>Top-Row Temp</th><th>Mean Temp</th><th>Runtime</th></tr>
        {table_rows}
    </table>
</div>

</body>
</html>"""

    path = os.path.join(run_dir, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nReport saved to {path}")