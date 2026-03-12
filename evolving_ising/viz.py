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
- `make_rollout_animation` returns a base64-encoded GIF string embedding a
  combined temperature+spin animation.  Requires Pillow; returns "" otherwise.
- `save_fitness_plot` writes a richer version of the fitness curve directly
  to disk alongside each experiment's other output files.
- `generate_report` assembles the full HTML document from
  `List[ExperimentResult]` and a run-configuration dict.
"""
from __future__ import annotations

import base64
import io
import os
import tempfile
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; safe for headless/WSL use
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

if TYPE_CHECKING:
    from .experiment import ExperimentResult


# ── Low-level helpers ─────────────────────────────────────────────────────────

def fig_to_base64(fig) -> str:
    """
    Render a matplotlib Figure to a base64-encoded PNG string and close it.

    Used to embed images inline in the HTML report as data URIs:
        <img src="data:image/png;base64,{fig_to_base64(fig)}">
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _composite_frame(
    T_flat: np.ndarray,
    S_flat: np.ndarray,
    H: int,
    W: int,
    hot_t: float,
    cold_t: float,
) -> np.ndarray:
    """
    Combine a temperature field and spin configuration into a single RGB image
    using the HSV colourspace so the two channels are unambiguous:

      Hue        — temperature  (cold → blue at 0.65, hot → red at 0.0)
      Saturation — fixed at 0.90
      Value      — spin state   (+1 → bright 1.0,  −1 → dim 0.35)

    Cold + spin=+1  : bright blue
    Hot  + spin=+1  : bright red
    Cold + spin=−1  : dark blue
    Hot  + spin=−1  : dark red

    Parameters
    ----------
    T_flat : (N,) float32  temperature values
    S_flat : (N,) int8     spin values {-1, +1}
    H, W   : grid height / width
    hot_t, cold_t : temperature range for normalisation

    Returns
    -------
    np.ndarray (H, W, 3) float32 in [0, 1]
    """
    T = T_flat.reshape(H, W).astype(np.float32)
    S = S_flat.reshape(H, W).astype(np.float32)

    T_norm = np.clip((T - cold_t) / max(hot_t - cold_t, 1e-8), 0.0, 1.0)

    # Hue: 0.65 (blue) → 0.0 (red) as temperature rises
    hue = 0.65 * (1.0 - T_norm)
    sat = np.full((H, W), 0.90, dtype=np.float32)
    # Value: spin=+1 → 1.0, spin=-1 → 0.35
    val = 0.35 + 0.65 * (S + 1.0) / 2.0

    hsv = np.stack([hue, sat, val], axis=-1)
    return mcolors.hsv_to_rgb(hsv).astype(np.float32)


# ── Fitness plots ─────────────────────────────────────────────────────────────

def plot_fitness_curve(best_hist: List[float], best_so_far: List[float], title: str) -> str:
    """
    Compact fitness curve for embedding in the HTML report.

    Shows "best so far" (monotone global best) and "best (iter)" (noisy per-generation best).

    Returns
    -------
    str  base64-encoded PNG
    """
    fig, ax = plt.subplots(figsize=(6, 3.5))
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

    Includes the population mean ± std band in addition to the best curves.
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

    The colormap is fixed to [cold_t, hot_t] so temperatures are visually
    comparable across experiments.

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

    Black = spin −1 (ordered down), white = spin +1 (ordered up).
    Uniform patches = ferromagnetic domains, speckled = paramagnetic.

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


def plot_temp_and_spins(
    T_final: np.ndarray,
    S_final: np.ndarray,
    H: int,
    W: int,
    title: str,
    hot_t: float,
    cold_t: float,
) -> str:
    """
    Temperature and spin side-by-side in one figure.

    Both panels are the same (H, W) size so they align naturally.
    The temperature colormap uses a fixed [cold_t, hot_t] scale.

    Returns
    -------
    str  base64-encoded PNG
    """
    fig, axs = plt.subplots(1, 2, figsize=(6, 3.5))

    im = axs[0].imshow(T_final.reshape(H, W), cmap="magma", vmin=cold_t, vmax=hot_t)
    axs[0].set_title("Temperature")
    axs[0].axis("off")
    plt.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)

    axs[1].imshow(S_final.reshape(H, W), cmap="gray", vmin=-1, vmax=1)
    axs[1].set_title("Spins")
    axs[1].axis("off")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_connectivity(
    J_nk:      np.ndarray,   # (N, K) float32
    mask:      np.ndarray,   # (N, K) bool
    neighbors: np.ndarray,   # (N, K) int32
    H: int,
    W: int,
    title: str,
    sigma_ref: float = 0.5,
    j_scale:   float = 1.0,
) -> str:
    """
    Five-panel visualisation of the learned coupling structure.

    Top row (maps, same as before)
    -----
    In-degree  (Reds):   total incoming conductance at each site
    Out-degree (Greens): total outgoing conductance from each site
    Composite  (RGB):    R=normalised in-degree, G=normalised out-degree, B=0

    Bottom row (histograms)
    ----------
    Overlapping histograms comparing the J distribution at the start of
    optimisation (reference, sampled from softplus(N(0, sigma_ref)) with a
    fixed seed) versus the learned J at the end.  The overlap reveals how
    much the CMA-ES has shifted and spread the coupling distribution.

    Parameters
    ----------
    J_nk, mask, neighbors : arrays from IsingModel / ExperimentResult
    H, W                  : grid dimensions
    title                 : figure suptitle
    sigma_ref             : std-dev of the reference theta distribution
                            (should match sigma_init used during optimisation)
    j_scale               : j_scale multiplier used during optimisation

    Returns
    -------
    str  base64-encoded PNG
    """
    N = H * W

    # ── Spatial degree maps ───────────────────────────────────────────────────
    W_plot  = np.abs(J_nk) * mask.astype(np.float32)   # (N, K)
    out_deg = np.sum(W_plot, axis=1)                     # (N,)

    dest   = neighbors.reshape(-1)
    weights= W_plot.reshape(-1)
    in_deg = np.zeros(N, dtype=np.float32)
    np.add.at(in_deg, dest, weights)

    in_norm  = in_deg  / (np.max(in_deg)  + 1e-8)
    out_norm = out_deg / (np.max(out_deg) + 1e-8)
    RGB = np.stack([in_norm, out_norm, np.zeros(N)], axis=-1).reshape(H, W, 3)

    # ── Before / after J distribution ────────────────────────────────────────
    D = int(np.sum(mask))
    rng       = np.random.default_rng(0)
    theta_ref = rng.standard_normal(D) * sigma_ref
    # softplus = log(1 + exp(x))
    J_ref     = np.log1p(np.exp(theta_ref)) * j_scale
    J_learned = np.abs(J_nk[mask])

    # ── Layout: 2 rows (maps | histogram) ────────────────────────────────────
    fig = plt.figure(figsize=(11, 5))
    gs  = fig.add_gridspec(2, 3, height_ratios=[1.1, 0.85], hspace=0.4, wspace=0.3)

    ax_in  = fig.add_subplot(gs[0, 0])
    ax_out = fig.add_subplot(gs[0, 1])
    ax_rgb = fig.add_subplot(gs[0, 2])
    ax_hist= fig.add_subplot(gs[1, :])   # spans all 3 columns

    ax_in .imshow(in_deg.reshape(H, W),  cmap="Reds")
    ax_in .set_title("In-degree")
    ax_in .axis("off")

    ax_out.imshow(out_deg.reshape(H, W), cmap="Greens")
    ax_out.set_title("Out-degree")
    ax_out.axis("off")

    ax_rgb.imshow(RGB)
    ax_rgb.set_title("R=in,  G=out")
    ax_rgb.axis("off")

    bins = np.linspace(0.0, max(np.max(J_learned), np.max(J_ref)) * 1.05, 100)
    ax_hist.hist(J_ref,     bins=bins, alpha=0.5, color="steelblue",
                 label=f"Initial reference  (softplus(N(0,{sigma_ref}))·{j_scale})")
    ax_hist.hist(J_learned, bins=bins, alpha=0.5, color="tomato",
                 label="Learned  (|J_best| at valid bonds)")
    ax_hist.set_xlabel("|J| coupling strength")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Coupling distribution: before vs after optimisation")
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, alpha=0.25)

    fig.suptitle(title, fontsize=11)
    return fig_to_base64(fig)


def plot_directional_flows(
    J_nk:      np.ndarray,   # (N, K) float32
    mask:      np.ndarray,   # (N, K) bool
    neighbors: np.ndarray,   # (N, K) int32
    T_final:   np.ndarray,   # (N,) float32
    S_final:   np.ndarray,   # (N,) int8 or float32
    H: int,
    W: int,
    hot_t: float,
    cold_t: float,
    title: str,
) -> str:
    """
    Three-panel directional flow visualisation of the evolved coupling structure.

    Panel A — Heat Flux
        Per-site heat flux vector q_i = Σ_k W_norm[i,k] · ΔT[i,k] · d̂_k,
        overlaid on the final temperature field.  Arrows show *where* heat
        is actively flowing and in which direction; coloured by flux magnitude.

    Panel B — Conductance Asymmetry (Bond Diodes)
        ΔW[i,k] = |J[i,k]| − |J[j, rev_k]|.  A positive value means the
        bond is stronger in the forward direction — a thermal rectifier.
        Background: net asymmetry scalar per site (RdBu_r, centred at 0).
        Arrows: net diode direction per site, coloured by asymmetry magnitude.

    Panel C — Spin Alignment Current
        f[i,k] = J[i,k] · s_i · s_j.  Positive = aligned spins sharing a
        ferromagnetic bond (order being exported in that direction).
        Background: spin configuration (gray).
        Arrows: net direction of spin-order propagation per site.

    Returns
    -------
    str  base64-encoded PNG
    """
    N, K = J_nk.shape

    # ── Bond direction vectors ────────────────────────────────────────────────
    i_row = (np.arange(N) // W)[:, None].astype(np.float32)  # (N, 1)
    i_col = (np.arange(N) % W)[:, None].astype(np.float32)   # (N, 1)
    j_row = (neighbors // W).astype(np.float32)               # (N, K)
    j_col = (neighbors % W).astype(np.float32)                # (N, K)

    d_row = j_row - i_row   # positive = downward
    d_col = j_col - i_col   # positive = rightward

    # Periodic wrap correction
    d_row = np.where(d_row >  H / 2, d_row - H, d_row)
    d_row = np.where(d_row < -H / 2, d_row + H, d_row)
    d_col = np.where(d_col >  W / 2, d_col - W, d_col)
    d_col = np.where(d_col < -W / 2, d_col + W, d_col)

    mask_f = mask.astype(np.float32)
    d_row *= mask_f
    d_col *= mask_f

    # ── Reverse-slot table ────────────────────────────────────────────────────
    # rev_slot[i, k] = kp  such that  neighbors[neighbors[i,k], kp] == i
    rev_slot = np.full((N, K), -1, dtype=np.int32)
    i_idx = np.arange(N, dtype=np.int32)
    for kp in range(K):
        for k in range(K):
            j_vals = neighbors[:, k]
            match = (neighbors[j_vals, kp] == i_idx) & mask[:, k]
            rev_slot[:, k] = np.where(match & (rev_slot[:, k] < 0),
                                      kp, rev_slot[:, k])

    # ── Panel A: Heat flux ────────────────────────────────────────────────────
    T = T_final.astype(np.float32)
    W_abs  = np.abs(J_nk) * mask_f
    W_norm = W_abs / (W_abs.sum(axis=1, keepdims=True) + 1e-8)
    dT     = (T[neighbors] - T[:, None]) * mask_f
    q_col  = (W_norm * dT * d_col).sum(axis=1).reshape(H, W)
    q_row  = (W_norm * dT * d_row).sum(axis=1).reshape(H, W)

    # ── Panel B: Conductance asymmetry ────────────────────────────────────────
    J_abs = np.abs(J_nk)
    J_rev = np.zeros_like(J_abs)
    for k in range(K):
        for kp in range(K):
            sel = (rev_slot[:, k] == kp) & mask[:, k]
            if sel.any():
                J_rev[sel, k] = J_abs[neighbors[sel, k], kp]
    Delta_W     = (J_abs - J_rev) * mask_f
    asym_scalar = Delta_W.sum(axis=1).reshape(H, W)
    asym_col    = (Delta_W * d_col).sum(axis=1).reshape(H, W)
    asym_row    = (Delta_W * d_row).sum(axis=1).reshape(H, W)

    # ── Panel C: Spin alignment current ───────────────────────────────────────
    S        = S_final.astype(np.float32)
    f_ik     = J_nk * S[:, None] * S[neighbors] * mask_f
    spin_col = (f_ik * d_col).sum(axis=1).reshape(H, W)
    spin_row = (f_ik * d_row).sum(axis=1).reshape(H, W)

    # ── Quiver helper ─────────────────────────────────────────────────────────
    stride = max(1, H // 16)
    ys = np.arange(0, H, stride)
    xs = np.arange(0, W, stride)
    XX, YY = np.meshgrid(xs, ys)

    def _panel(ax, bg, bg_cmap, bg_vmin, bg_vmax,
               U_full, V_full, q_cmap, panel_title):
        ax.imshow(bg, cmap=bg_cmap, vmin=bg_vmin, vmax=bg_vmax, origin="upper")
        U   = U_full[::stride, ::stride]
        V   = V_full[::stride, ::stride]
        mag = np.hypot(U, V)
        peak = float(np.percentile(mag, 95)) if mag.max() > 0 else 1.0
        if peak > 0:
            U = U / (peak + 1e-8)
            V = V / (peak + 1e-8)
        ax.quiver(XX, YY, U, V, mag,
                  cmap=q_cmap, alpha=0.85, clim=(0, peak),
                  angles="xy", scale_units="xy",
                  scale=1.0 / (stride * 0.45))
        ax.set_title(panel_title, fontsize=10)
        ax.axis("off")

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    _panel(axs[0], T.reshape(H, W), "magma", cold_t, hot_t,
           q_col, q_row, "cool", "Heat Flux")

    abs_max = float(np.abs(asym_scalar).max()) or 1.0
    _panel(axs[1], asym_scalar, "RdBu_r", -abs_max, abs_max,
           asym_col, asym_row, "PuOr", "Conductance Asymmetry")

    _panel(axs[2], S.reshape(H, W), "gray", -1, 1,
           spin_col, spin_row, "plasma", "Spin Alignment Current")

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

    All panels share the same colormap limits so differences in heat transport
    are directly visible across objectives.

    Returns
    -------
    str  base64-encoded PNG
    """
    n = len(all_T_finals)
    per_w = max(1.5, min(2.5, 12.0 / max(n, 1)))
    fig, axs = plt.subplots(1, n, figsize=(per_w * n, 3.5))
    if n == 1:
        axs = [axs]
    for ax, (name, T_final) in zip(axs, all_T_finals.items()):
        ax.imshow(T_final.reshape(H, W), cmap="magma", vmin=cold_t, vmax=hot_t)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle("Final Temperature Comparison", fontsize=13)
    fig.tight_layout()
    return fig_to_base64(fig)


# ── Animation ─────────────────────────────────────────────────────────────────

def make_rollout_animation(
    T_frames: Sequence[np.ndarray],
    S_frames: Sequence[np.ndarray],
    H: int,
    W: int,
    hot_t: float,
    cold_t: float,
    fps: int = 15,
) -> str:
    """
    Generate an animated GIF from a sequence of temperature and spin frames,
    encoding both channels into a single HSV image per frame (see
    `_composite_frame` for the encoding).

    The animation is encoded as base64 GIF so it can be embedded directly
    in the HTML report as a data URI.  Returns an empty string if Pillow is
    not available (the report will simply omit the animation section).

    Parameters
    ----------
    T_frames : sequence of (N,) float32 arrays, one per timestep
    S_frames : sequence of (N,) int8 arrays, one per timestep
    H, W     : grid dimensions
    hot_t, cold_t : temperature range for normalisation
    fps      : frames per second for the GIF

    Returns
    -------
    str  base64-encoded GIF, or "" if Pillow is unavailable
    """
    try:
        import PIL  # noqa: F401  just check availability
    except ImportError:
        return ""

    frames_rgb = [
        _composite_frame(T_frames[i], S_frames[i], H, W, hot_t, cold_t)
        for i in range(len(T_frames))
    ]

    fig, ax = plt.subplots(figsize=(4, 4))
    # Fill figure completely so the square axes matches the square frame exactly
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    im = ax.imshow(frames_rgb[0], vmin=0.0, vmax=1.0)
    ax.axis("off")

    def _update(i):
        im.set_data(frames_rgb[i])
        return (im,)

    ani = FuncAnimation(fig, _update, frames=len(frames_rgb),
                        interval=1000 // fps, blit=True)

    # Save to a temp file then read back (more portable than BytesIO with pillow writer)
    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    tmp.close()
    try:
        ani.save(tmp.name, writer="pillow", fps=fps, dpi=72)
        with open(tmp.name, "rb") as f:
            gif_bytes = f.read()
    finally:
        plt.close(fig)
        os.unlink(tmp.name)

    return base64.b64encode(gif_bytes).decode("ascii")


# ── HTML report ───────────────────────────────────────────────────────────────

def generate_report(
    results:      "List[ExperimentResult]",
    run_dir:      str,
    H:            int,
    W:            int,
    hot_t:        float,
    cold_t:       float,
    run_config:   dict,
    rollout_data: Optional[Dict[str, tuple]] = None,
) -> None:
    """
    Generate a self-contained HTML report for the full experiment run.

    Layout per experiment card
    --------------------------
    Row 1 (3 columns): fitness curve | temperature | spins
                        (temperature and spins are adjacent — same dimensions)
    Row 2 (full width): connectivity maps + before/after J histograms
    Row 3 (full width): temperature+spin animation (if rollout_data provided)

    Parameters
    ----------
    results      : list of ExperimentResult from run_experiment()
    run_dir      : directory where report.html is written
    H, W         : grid dimensions (for reshaping flat arrays)
    hot_t        : maximum temperature (colormap upper bound)
    cold_t       : minimum temperature (colormap lower bound)
    run_config   : dict of config key→value pairs shown in the report header
    rollout_data : optional dict mapping experiment name →
                   (T_frames, S_frames) where each is a list of (N,) arrays.
                   When provided, an animation is embedded in each card.
    """
    from .experiment import EXPERIMENTS  # deferred to break circular import

    rollout_data = rollout_data or {}

    experiment_sections = []
    all_T_finals = {}

    for r in results:
        info = EXPERIMENTS[r.name]

        fig_fitness  = plot_fitness_curve(r.best_hist, r.best_so_far,
                                          f"Fitness — {info['title']}")
        fig_ts       = plot_temp_and_spins(r.T_final, r.S_final, H, W,
                                           f"{r.name}", hot_t, cold_t)
        fig_conn     = plot_connectivity(r.J_best, r.mask, r.neighbors, H, W,
                                         f"Connectivity — {r.name}")
        fig_flows    = plot_directional_flows(r.J_best, r.mask, r.neighbors,
                                              r.T_final, r.S_final, H, W,
                                              hot_t, cold_t,
                                              f"Directional Flows — {r.name}")
        all_T_finals[r.name] = r.T_final

        # Optional animation section
        anim_html = ""
        if r.name in rollout_data:
            T_frames, S_frames = rollout_data[r.name]
            gif_b64 = make_rollout_animation(T_frames, S_frames, H, W, hot_t, cold_t)
            if gif_b64:
                anim_html = f"""
            <div class="wide-panel">
                <div class="panel-label">Dynamics rollout — combined temperature &amp; spin animation</div>
                <div class="anim-wrap">
                    <img src="data:image/gif;base64,{gif_b64}" alt="Animation" class="anim-img">
                    <div class="anim-meta">
                        <div class="anim-title">Encoding</div>
                        <div class="anim-legend">
                            <span class="cold">■</span> blue hue → cold temperature<br>
                            <span class="hot">■</span> red hue → hot temperature<br>
                            <span class="sp-up">bright</span> → spin +1<br>
                            <span class="sp-dn">dark</span> → spin −1
                        </div>
                    </div>
                </div>
            </div>"""

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
            <div class="grid-top">
                <div><img src="data:image/png;base64,{fig_fitness}" alt="Fitness"></div>
                <div><img src="data:image/png;base64,{fig_ts}" alt="Temp+Spins"></div>
            </div>
            <div class="wide-panel">
                <img src="data:image/png;base64,{fig_conn}" alt="Connectivity">
            </div>
            <div class="wide-panel">
                <img src="data:image/png;base64,{fig_flows}" alt="Directional Flows">
            </div>{anim_html}
        </div>""")

    fig_comparison = plot_comparison(all_T_finals, H, W, hot_t, cold_t)

    table_rows = "".join(
        f"<tr><td>{r.name}</td><td>{r.best_fitness:.6f}</td>"
        f"<td>{r.top_row_temp:.4f}</td><td>{r.mean_temp:.4f}</td><td>{r.elapsed:.1f}s</td></tr>"
        for r in results
    )

    config_items = " | ".join(
        f"<code>{k}</code>: <code>{v}</code>" for k, v in run_config.items()
    )

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
    /* Row 1: two equal columns — fitness and temp+spins share the same figsize
       so they render at identical height when displayed at equal widths. */
    .grid-top {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem;
                 margin-bottom: 0.6rem; align-items: start; }}
    .grid-top img {{ width: 100%; display: block; border-radius: 4px; }}
    /* Full-width panels (connectivity, comparison, animation) */
    .wide-panel {{ margin-top: 0.6rem; }}
    .wide-panel > img {{ width: 100%; display: block; border-radius: 4px; }}
    .panel-label {{ font-size: 0.78rem; color: #8b949e; margin-bottom: 0.3rem;
                    font-style: italic; }}
    /* Animation: GIF left-aligned, legend to the right */
    .anim-wrap {{ display: flex; flex-direction: row; align-items: flex-start; gap: 1.2rem; }}
    .anim-img {{ height: 260px; width: auto; display: block; border-radius: 4px; }}
    .anim-meta {{ display: flex; flex-direction: column; gap: 0.4rem; padding-top: 0.2rem; }}
    .anim-title {{ font-size: 0.85rem; color: #c9d1d9; font-weight: 600; }}
    .anim-legend {{ font-size: 0.8rem; color: #8b949e; line-height: 2.0; }}
    .cold {{ color: #4488ff; }}
    .hot  {{ color: #ff6644; }}
    .sp-up {{ color: #ddd; }}
    .sp-dn {{ color: #666; }}
    /* Global panels */
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
<p class="subtitle">CMA-ES optimisation with different fitness objectives on a {H}×{W} Ising lattice</p>

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
