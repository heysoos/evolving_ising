"""Shared utilities for HTML report generation.

Used by exp1_report.py, exp2_report.py, and exp3_report.py.
Provides: unified CSS, figure helpers, simulation loop for animations,
GIF generation, and an interactive canvas-based training curve chart.
"""

import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import PIL  # noqa: F401
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

REPORT_CSS = """
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
img.anim { max-width: 100%; border: 1px solid #d0d9e8; border-radius: 6px;
           margin: .8em 0; display: block; }
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
.scenario-bar { background: #e8f0f8; border: 1px solid #b0c4de; border-radius: 6px;
                padding: 0.7em 1.4em; margin: 1.2em 0; display: flex;
                align-items: center; gap: 1em; }
.scenario-bar label { font-weight: bold; color: #2c5282; }
.scenario-bar select { padding: 5px 10px; border-radius: 4px; border: 1px solid #b0c4de;
                       font-size: 1em; cursor: pointer; background: #fff; }
.run-panel { padding: 0.5em 0; }
.beat-yes { color: #276749; font-weight: bold; }
.beat-no  { color: #c53030; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5em; margin: 1em 0; }
@media (max-width: 700px) { .two-col { grid-template-columns: 1fr; } }
details { margin: 1.5em 0; }
details > summary {
    color: #2c5282; font-size: 1.25em; font-weight: 600;
    border-left: 4px solid #3182ce; padding: .4em .6em;
    cursor: pointer; list-style: none;
    display: flex; align-items: center; gap: .5em;
}
details > summary::before { content: '▶'; font-size: .75em; transition: transform .15s; }
details[open] > summary::before { transform: rotate(90deg); }
details > summary::-webkit-details-marker { display: none; }
"""


# ---------------------------------------------------------------------------
# Collapsible section helper
# ---------------------------------------------------------------------------

def collapsible_section(title, content_html, open=True):
    """Wrap content in a collapsible <details> block with an <h2>-styled summary."""
    open_attr = ' open' if open else ''
    return f'<details{open_attr}>\n<summary>{title}</summary>\n{content_html}\n</details>\n'


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def fig_to_b64(fig, dpi=120):
    """Render a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')


def img_tag(b64, alt='', cls='fig', caption=''):
    """Return an <img> tag for a base64 PNG, with optional caption."""
    if not b64:
        return ''
    parts = [f'<img class="{cls}" src="data:image/png;base64,{b64}" alt="{alt}">']
    if caption:
        parts.append(f'<p class="caption">{caption}</p>')
    return '\n'.join(parts)


def gif_tag(b64, alt='', caption=''):
    """Return an <img> tag for a base64 GIF animation."""
    if not b64:
        return ''
    parts = [f'<img class="anim" src="data:image/gif;base64,{b64}" alt="{alt}">']
    if caption:
        parts.append(f'<p class="caption">{caption}</p>')
    return '\n'.join(parts)


def load_config(run_dir):
    """Load config.json from a run directory.  Returns dict or None."""
    import json as _json
    from pathlib import Path
    p = Path(run_dir) / 'config.json'
    if not p.exists():
        return None
    with open(p) as f:
        return _json.load(f)


def config_table_html(config, title='Configuration'):
    """Render a config dict as an HTML table card."""
    if not config:
        return ''
    rows = ''.join(
        f'<tr><td><code>{k}</code></td><td>{v}</td></tr>'
        for k, v in sorted(config.items())
    )
    return (
        f'<div class="card">\n<h3>{title}</h3>\n'
        f'<table><thead><tr><th>Parameter</th><th>Value</th></tr></thead>'
        f'<tbody>{rows}</tbody></table>\n</div>\n'
    )


def load_run(run_dir):
    """Load training_log.npz and best_controller.npz from a run directory.

    Returns
    -------
    (log_dict, ctrl_dict) — either may be None if the file is absent.
    """
    from pathlib import Path
    run_dir = Path(run_dir)
    log, ctrl = None, None
    lp = run_dir / 'training_log.npz'
    cp = run_dir / 'best_controller.npz'
    if lp.exists():
        d = np.load(lp)
        log = {k: d[k] for k in d.files}
    if cp.exists():
        d = np.load(cp)
        ctrl = {k: d[k] for k in d.files}
    return log, ctrl


# ---------------------------------------------------------------------------
# Simulation loop for animation frames
# ---------------------------------------------------------------------------


def run_anim_frames(model, config, budget_type='none', params_flat=None,
                    n_cycles=3, steps_per_cycle=None, frame_skip=2,
                    warmup_sweeps=200):
    """Run simulation and capture spin + J frames for animation.

    Mirrors make_jax_eval_fn in optimiser.py: the ENTIRE simulation — including
    controller MLP, budget updates, and J remodelling — runs inside a single
    jax.jit with nested lax.scan calls.  No Python loops in the hot path.

    Outer scan: over n_frames = (n_cycles * spc) // frame_skip frames.
    Inner scan: over frame_skip physics steps per frame.
    Per-frame outputs (spins, J_mean, cumulative W_net, T) are returned by the
    outer scan and converted to NumPy lists.

    Parameters
    ----------
    model : IsingModel
    config : dict
    budget_type : str
    params_flat : ndarray or None
        If provided, the controller MLP weights to use.
    n_cycles : int
    steps_per_cycle : int or None  (defaults to config['steps_per_cycle'])
    frame_skip : int  — physics steps per frame
    warmup_sweeps : int

    Returns
    -------
    spin_frames : list of (L, L) int8 arrays
    J_mean_frames : list of (L, L) float32 arrays  (mean J strength per site)
    W_net_cycles : list  (empty — callers discard with _)
    wnet_trace : list of floats  (cumulative W_net at each captured frame)
    T_frames : list of floats  (bath temperature at each captured frame)
    """
    import jax
    import jax.numpy as jnp
    from work_extraction.controller import _mlp_forward, make_layer_specs
    from work_extraction.budgets import make_budget as _make_budget_fn

    # ------------------------------------------------------------------ config
    L = config.get('L', 32)
    T_mean = float(config['T_mean'])
    delta_T = float(config['delta_T'])
    tau = float(config['tau'])
    J_init_val = float(config['J_init'])
    J_min = float(config.get('J_min', 0.01))
    J_max = float(config.get('J_max', 5.0))
    lam = float(config.get('lambda', 0.05))
    delta_J_max = float(config.get('delta_J_max', 0.1))
    hidden_size = int(config.get('hidden_size', 8))
    mag_alpha = float(config.get('mag_ema_alpha', 0.05))
    bond_update_frac = float(config.get('bond_update_frac', 0.1))
    B_scale = float(config.get('B_scale', 2.0))
    num_sweeps = int(config.get('num_sweeps', 1))
    spc = int(config.get('steps_per_cycle', 100)) if steps_per_cycle is None else steps_per_cycle
    T_norm_denom = delta_T if delta_T > 0 else 1.0
    J_crit = T_mean / 2.269          # critical coupling (scalar)

    # ---------------------------------------------------- precomputed JAX arrays
    N = model.n
    K = model.K
    neighbors_np = np.asarray(model.neighbors)
    mask_np = np.asarray(model.mask, dtype=bool)
    mask_f = jnp.asarray(mask_np, dtype=jnp.float32)

    valid_i_np, valid_k_np = np.where(mask_np)
    valid_j_np = neighbors_np[valid_i_np, valid_k_np]
    n_bonds_total = len(valid_i_np)
    n_updates = max(1, int(n_bonds_total * bond_update_frac))
    valid_i_jax = jnp.asarray(valid_i_np, dtype=jnp.int32)
    valid_k_jax = jnp.asarray(valid_k_np, dtype=jnp.int32)
    valid_j_jax = jnp.asarray(valid_j_np, dtype=jnp.int32)

    valid_count_jax = jnp.asarray(mask_np.sum(axis=1), dtype=jnp.float32)
    J_init_jax = jnp.full((N, K), J_init_val, dtype=jnp.float32) * mask_f

    # Temperature schedule reshaped to (n_frames, frame_skip) for outer scan
    total_steps = n_cycles * spc
    n_frames = total_steps // frame_skip
    t_all = np.arange(total_steps, dtype=np.float32)
    T_sched = (T_mean + delta_T * np.sin(2.0 * np.pi * t_all / tau)).astype(np.float32)
    T_chunks_jax = jnp.array(T_sched[:n_frames * frame_skip].reshape(n_frames, frame_skip))

    # Controller params as a JAX constant (None → controller code excluded at trace time)
    params_jax = jnp.asarray(params_flat, dtype=jnp.float32) if params_flat is not None else None
    layer_specs = make_layer_specs(hidden_size)

    # ----------------------------------------- budget object (pure-function interface)
    # Delegates to budgets.py so budget dynamics are identical to make_jax_eval_fn.
    budget = _make_budget_fn(budget_type, neighbors_np, mask_np, config)

    # ----------------------------------------- inner step function (lax.scan body)
    # Python `if params_jax is not None` is evaluated at trace time, so the
    # controller block is compiled in only when a controller is active.
    def _step_fn(carry, T_t):
        spins, key, J, bud, mag_ema, running_wnet, E_prev = carry
        s_bef_f = spins[0].astype(jnp.float32)
        key, sub_m, sub_b = jax.random.split(key, 3)
        spins, _ = model.metropolis_checkerboard_sweeps(sub_m, spins, J, T_t, num_sweeps)
        s_aft_f = spins[0].astype(jnp.float32)
        E_after = jnp.mean(model.energy(J, spins))
        dE = E_after - E_prev
        running_wnet = running_wnet + jnp.maximum(-dE, 0.0) - jnp.maximum(dE, 0.0)
        bud = budget.update_pure(bud, s_bef_f, s_aft_f)
        mag_ema = mag_alpha * s_aft_f + (1.0 - mag_alpha) * mag_ema

        if params_jax is not None:
            perm = jax.random.permutation(sub_b, n_bonds_total)[:n_updates]
            si = valid_i_jax[perm]
            sk = valid_k_jax[perm]
            sj = valid_j_jax[perm]
            T_norm = (T_t - T_mean) / T_norm_denom
            bud_vals = budget.get_pure(bud, si, sk, sj)
            bud_norm = jnp.tanh(bud_vals / B_scale)
            J_norm_arr = jnp.tanh(J[si, sk] / J_crit - 1.0)
            x = jnp.stack([
                s_aft_f[si], s_aft_f[sj], mag_ema[si],
                jnp.full(n_updates, T_norm, dtype=jnp.float32), bud_norm,
                J_norm_arr,
            ], axis=-1)
            dJ = _mlp_forward(params_jax, x, layer_specs, delta_J_max).ravel()
            costs = jnp.abs(s_aft_f[si] * s_aft_f[sj] * dJ) + lam * jnp.abs(dJ)
            can_apply = bud_vals >= costs
            J = jnp.clip(J.at[si, sk].add(jnp.where(can_apply, dJ, 0.0)), J_min, J_max) * mask_f
            bud = budget.spend_pure(bud, si, sk, sj, costs, can_apply)
            E_after = jnp.mean(model.energy(J, spins))  # re-evaluate after J update

        return (spins, key, J, bud, mag_ema, running_wnet, E_after), None

    # ----------------------------------------- outer frame function (lax.scan body)
    def _frame_fn(carry, T_chunk):
        spins, key, J, bud, mag_ema, running_wnet = carry
        E_init = jnp.mean(model.energy(J, spins))
        step_carry = (spins, key, J, bud, mag_ema, running_wnet, E_init)
        (spins, key, J, bud, mag_ema, running_wnet, _), _ = jax.lax.scan(
            _step_fn, step_carry, T_chunk
        )
        J_mean = jnp.where(
            valid_count_jax > 0,
            (J * mask_f).sum(axis=1) / jnp.maximum(valid_count_jax, 1),
            J_init_val,
        )
        # Per-site budget mean (for visualisation)
        if budget_type == 'bond':
            bud_mean = (bud * mask_f).sum(axis=1) / jnp.maximum(valid_count_jax, 1)
        elif budget_type == 'none':
            bud_mean = jnp.zeros(N, dtype=jnp.float32)
        else:  # neighbourhood, diffusing — bud is already (N,)
            bud_mean = bud
        return (spins, key, J, bud, mag_ema, running_wnet), (
            spins[0],       # (N,) int8  — spin state at end of frame
            J_mean,         # (N,) float32
            running_wnet,   # scalar — cumulative W_net through this frame
            T_chunk[0],     # scalar — T at start of frame
            bud_mean,       # (N,) float32 — per-site budget mean
        )

    # ----------------------------------------- single JIT call for all frames
    @jax.jit
    def _run_all(spins, key):
        init = (
            spins, key, J_init_jax, budget.init(),
            jnp.zeros(N, dtype=jnp.float32),  # mag_ema
            jnp.float32(0.0),                  # running_wnet
        )
        _, (spins_all, J_mean_all, wnet_all, T_all, bud_all) = jax.lax.scan(
            _frame_fn, init, T_chunks_jax
        )
        return spins_all, J_mean_all, wnet_all, T_all, bud_all

    # Initialise and warmup (outside the main scan — one-time cost)
    key = jax.random.PRNGKey(42)
    key, ik, wk = jax.random.split(key, 3)
    spins = model.init_spins(ik, 1)
    spins, _ = model.metropolis_checkerboard_sweeps(wk, spins, J_init_jax, T_mean, warmup_sweeps)

    # Run — single compiled call
    spins_all, J_mean_all, wnet_all, T_all, bud_all = _run_all(spins, key)

    # Convert stacked JAX outputs → Python lists expected by callers
    spins_np  = np.asarray(spins_all)   # (n_frames, N) int8
    J_mean_np = np.asarray(J_mean_all)  # (n_frames, N) float32
    wnet_np   = np.asarray(wnet_all)    # (n_frames,)
    T_np      = np.asarray(T_all)       # (n_frames,)

    spin_frames   = [spins_np[fi].reshape(L, L) for fi in range(n_frames)]
    J_mean_frames = [J_mean_np[fi].reshape(L, L) for fi in range(n_frames)]

    if budget_type == 'none':
        bud_frames = None
    else:
        bud_np = np.asarray(bud_all)    # (n_frames, N) float32
        bud_frames = [bud_np[fi].reshape(L, L) for fi in range(n_frames)]

    return spin_frames, J_mean_frames, [], list(wnet_np), list(T_np), bud_frames


# ---------------------------------------------------------------------------
# GIF generation
# ---------------------------------------------------------------------------

def frames_to_gif_b64(spin_frames, J_mean_frames, fps=8, max_frames=200, scale=5,
                      wnet_trace=None, T_trace=None, bud_frames=None):
    """Render spin + J [+ budget] frames to an animated GIF; return base64 string or None.

    Layout (top to bottom):
      Title bar : "Spins" | "Mean J" [| "Budget"]
      Main row  : spin state | mean J [| per-site budget]   (2 or 3 panels wide)
      W_net strip (optional): spans full width
      T strip (optional)    : spans full width
      X-label bar           : "time →"

    Parameters
    ----------
    wnet_trace : list of floats or None
    T_trace    : list of floats or None
    bud_frames : list of (L, L) float32 arrays or None
        Per-site budget mean at each frame.  When provided a third panel is
        added and all strips below extend to the wider canvas.
    """
    if not HAS_PIL:
        print('  Pillow not installed; skipping GIF (pip install pillow)')
        return None
    if not spin_frames:
        return None

    from PIL import Image as _PILImage
    from PIL import ImageDraw as _ImageDraw

    has_budget = bud_frames is not None and len(bud_frames) > 0

    step = max(1, len(spin_frames) // max_frames)
    sf  = spin_frames[::step]
    jf  = J_mean_frames[::step]
    bf  = bud_frames[::step] if has_budget else None
    wt_sub = wnet_trace[::step] if wnet_trace is not None else None
    tt_sub = T_trace[::step]   if T_trace   is not None else None

    L = sf[0].shape[0]
    n_panels = 3 if has_budget else 2
    strip_h = L // 2  # strip height in pre-scale pixels

    # J colourscale
    J_arr = np.stack(jf)
    j_vmin, j_vmax = float(J_arr.min()), float(J_arr.max())
    if j_vmax <= j_vmin:
        j_vmax = j_vmin + 0.01

    # Budget colourscale
    if has_budget:
        b_arr = np.stack(bf)
        b_vmin, b_vmax = float(b_arr.min()), float(b_arr.max())
        if b_vmax <= b_vmin:
            b_vmax = b_vmin + 0.01

    cmap_j = matplotlib.colormaps['viridis']
    cmap_b = matplotlib.colormaps['plasma']

    # W_net trace bounds
    wt_arr = None
    if wt_sub is not None and len(wt_sub) > 1:
        wt_arr = np.array(wt_sub, dtype=np.float64)
        w_vmin = float(min(wt_arr.min(), 0.0))
        w_vmax = float(max(wt_arr.max(), w_vmin + 1.0))

    # T trace bounds
    tt_arr = None
    if tt_sub is not None and len(tt_sub) > 1:
        tt_arr = np.array(tt_sub, dtype=np.float64)
        t_vmin = float(tt_arr.min())
        t_vmax = float(tt_arr.max())
        if t_vmax <= t_vmin:
            t_vmax = t_vmin + 0.1

    def _render_trace_strip(values, vmin, vmax, colour, frame_idx, n_total, strip_width):
        """Generic growing-trace strip with a red cursor."""
        strip = np.full((strip_h, strip_width, 3), 28, dtype=np.uint8)
        # Reference line at midpoint
        y_mid = int((1.0 - (0.5 * (vmin + vmax) - vmin) / (vmax - vmin)) * (strip_h - 1))
        strip[max(0, min(y_mid, strip_h - 1)), :] = [80, 80, 80]
        for i in range(frame_idx + 1):
            x = min(int(i / n_total * strip_width), strip_width - 1)
            y = int((1.0 - (values[i] - vmin) / (vmax - vmin)) * (strip_h - 1))
            strip[max(0, min(y, strip_h - 1)), x] = colour
        x_cur = min(int(frame_idx / n_total * strip_width), strip_width - 1)
        strip[:, x_cur] = [200, 80, 80]
        return strip

    def _render_wnet_strip(fi, n_total, w):
        # Zero-reference line instead of midpoint
        strip = _render_trace_strip(wt_arr, w_vmin, w_vmax, [80, 200, 120], fi, n_total, w)
        y_zero = int((1.0 - (0.0 - w_vmin) / (w_vmax - w_vmin)) * (strip_h - 1))
        strip[max(0, min(y_zero, strip_h - 1)), :] = [80, 80, 80]
        # Re-draw trace on top of zero line
        for i in range(fi + 1):
            x = min(int(i / n_total * w), w - 1)
            y = int((1.0 - (wt_arr[i] - w_vmin) / (w_vmax - w_vmin)) * (strip_h - 1))
            strip[max(0, min(y, strip_h - 1)), x] = [80, 200, 120]
        x_cur = min(int(fi / n_total * w), w - 1)
        strip[:, x_cur] = [200, 80, 80]
        return strip

    def _render_T_strip(fi, n_total, w):
        return _render_trace_strip(tt_arr, t_vmin, t_vmax, [255, 140, 0], fi, n_total, w)

    # Layout constants (pre-scale pixels)
    TITLE_H = 6
    BOT_H   = 5

    n_frames = len(sf)
    pil_frames = []
    frame_iter = zip(sf, jf, bf) if has_budget else ((s, j, None) for s, j in zip(sf, jf))

    for frame_idx, (s, j, b) in enumerate(frame_iter):
        spin_rgb = np.where(s[:, :, np.newaxis] > 0, 220, 30).astype(np.uint8)
        spin_rgb = np.broadcast_to(spin_rgb, (*s.shape, 3)).copy()
        j_norm = np.clip((j - j_vmin) / (j_vmax - j_vmin), 0.0, 1.0)
        j_rgb  = (cmap_j(j_norm)[:, :, :3] * 255).astype(np.uint8)

        W_px = n_panels * L
        title_bar = np.full((TITLE_H, W_px, 3), 18, dtype=np.uint8)
        main_panels = [spin_rgb, j_rgb]
        if has_budget:
            b_norm = np.clip((b - b_vmin) / (b_vmax - b_vmin), 0.0, 1.0)
            b_rgb  = (cmap_b(b_norm)[:, :, :3] * 255).astype(np.uint8)
            main_panels.append(b_rgb)
        main = np.concatenate(main_panels, axis=1)  # (L, n_panels*L, 3)

        bands = [title_bar, main]
        if wt_arr is not None:
            bands.append(_render_wnet_strip(frame_idx, n_frames, W_px))
        if tt_arr is not None:
            bands.append(_render_T_strip(frame_idx, n_frames, W_px))
            bands.append(np.full((BOT_H, W_px, 3), 18, dtype=np.uint8))

        combined = np.concatenate(bands, axis=0)
        img = _PILImage.fromarray(combined, mode='RGB')
        H_raw, W_raw = combined.shape[:2]
        img = img.resize((W_raw * scale, H_raw * scale), _PILImage.NEAREST)

        # ---- text labels on scaled image ----
        draw = _ImageDraw.Draw(img)
        W_s  = W_raw * scale
        L_s  = L * scale
        th_s = TITLE_H * scale
        sh_s = strip_h * scale

        def _txt(xy, text, fill, stroke=(12, 12, 12)):
            x, y = xy
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                draw.text((x + dx, y + dy), text, fill=stroke)
            draw.text(xy, text, fill=fill)

        # Panel titles (dark title bar — no outline needed)
        draw.text((4,           2), "Spins",  fill=(255, 255, 180))
        draw.text((L_s + 4,     2), "Mean J", fill=(255, 255, 180))
        if has_budget:
            draw.text((2 * L_s + 4, 2), "Budget", fill=(255, 200, 255))

        # Strip annotations on the right edge (outlined)
        y0_wnet = th_s + L_s
        if wt_arr is not None:
            _txt((W_s - 46, y0_wnet + 2),        "W_net",         fill=(150, 255, 180))
            _txt((W_s - 46, y0_wnet + 12),       f"{w_vmax:.1f}", fill=(200, 200, 200))
            _txt((W_s - 46, y0_wnet + sh_s - 9), f"{w_vmin:.1f}", fill=(200, 200, 200))
            y0_T = y0_wnet + sh_s
        else:
            y0_T = y0_wnet

        if tt_arr is not None:
            _txt((W_s - 46, y0_T + 2),        "T(t)",          fill=(255, 200, 100))
            _txt((W_s - 46, y0_T + 12),       f"{t_vmax:.1f}", fill=(200, 200, 200))
            _txt((W_s - 46, y0_T + sh_s - 9), f"{t_vmin:.1f}", fill=(200, 200, 200))
            y0_bot = y0_T + sh_s
            _txt((W_s // 2 - 18, y0_bot + 2), "time \u2192",   fill=(200, 200, 200))

        pil_frames.append(img)

    buf = io.BytesIO()
    try:
        pil_frames[0].save(
            buf, format='GIF', save_all=True,
            append_images=pil_frames[1:],
            duration=1000 // fps, loop=0, optimize=False,
        )
    except Exception as e:
        print(f'  GIF save failed: {e}')
        return None
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')


# ---------------------------------------------------------------------------
# Colour palette for multi-series charts
# ---------------------------------------------------------------------------

# 12-color palette for multi-series charts
PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78',
]


# ---------------------------------------------------------------------------
# Scenario selector widget
# ---------------------------------------------------------------------------

def scenario_selector_html(scenario_ids, labels, default_id, title='Select Run'):
    # kept for backward compatibility — use plotly_training_curves + fig_to_plotly_div for new reports
    """Return a dropdown widget that shows/hides scenario panel divs.

    The content divs must be written separately by the caller, with
    id="{scenario_id}" and initial style='display:block/none'.
    """
    options = '\n'.join(
        f'    <option value="{sid}" {"selected" if sid == default_id else ""}>{lbl}</option>'
        for sid, lbl in zip(scenario_ids, labels)
    )
    hide_all = ';'.join(
        f"document.getElementById('{s}').style.display='none'"
        for s in scenario_ids
    )
    return (
        f'<div class="scenario-bar">\n'
        f'  <label for="sc_sel_{scenario_ids[0]}">{title}:</label>\n'
        f'  <select id="sc_sel_{scenario_ids[0]}" '
        f'onchange="{hide_all};document.getElementById(this.value).style.display=\'block\';window.dispatchEvent(new Event(\'resize\'))">\n'
        f'{options}\n'
        f'  </select>\n'
        f'</div>\n'
    )


# ---------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------

try:
    import plotly.graph_objects as _go
    HAS_PLOTLY = True
except ImportError:
    _go = None
    HAS_PLOTLY = False


def fig_to_plotly_div(fig, include_plotlyjs='cdn'):
    """Convert a Plotly Figure to an HTML div string for embedding.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    include_plotlyjs : str or bool
        'cdn'  — adds a CDN <script> tag (~10 KB stub)
        True   — embeds full plotly.js (~3 MB, fully offline)
        False  — omit (assume plotly.js already included earlier on the page)
    """
    if not HAS_PLOTLY or fig is None:
        return ''
    return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs,
                       config={'responsive': True})


def plotly_training_curves(series_data, baseline=None, title='',
                           xlabel='Generation', ylabel='W_net', log_x=False):
    """Build a Plotly training-curve figure.

    Parameters
    ----------
    series_data : list of dicts
        Each dict has keys 'label', 'x', 'y', and optionally 'color'.
    baseline : float or None
        Horizontal reference line.
    log_x : bool
        If True, use a log scale on the x-axis.

    Returns
    -------
    plotly.graph_objects.Figure or None
    """
    if not HAS_PLOTLY or not series_data:
        return None
    fig = _go.Figure()
    for s in series_data:
        kw = dict(
            name=s['label'], x=s['x'], y=s['y'], mode='lines',
            hovertemplate='gen %{x}<br>' + s['label'] + ': %{y:.3f}<extra></extra>',
        )
        if 'color' in s:
            kw['line'] = dict(color=s['color'])
        fig.add_trace(_go.Scatter(**kw))
    if baseline is not None:
        fig.add_hline(
            y=baseline, line_dash='dash', line_color='crimson',
            annotation_text=f'baseline {baseline:.3f}',
            annotation_position='bottom right',
        )
    layout_kw = dict(
        title=title, xaxis_title=xlabel, yaxis_title=ylabel,
        legend=dict(
            orientation='v',
            yanchor='top', y=1,
            xanchor='left', x=1.02,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#cbd5e0', borderwidth=1,
        ),
        margin=dict(l=60, r=160, t=80, b=40),
        height=350,
        template='plotly_white',
    )
    if len(series_data) > 1:
        layout_kw['updatemenus'] = [{
            'type': 'buttons', 'direction': 'left',
            'x': 1.02, 'y': 1.08, 'xanchor': 'left', 'yanchor': 'bottom',
            'buttons': [
                {'label': 'All On',  'method': 'restyle', 'args': [{'visible': True}]},
                {'label': 'All Off', 'method': 'restyle', 'args': [{'visible': 'legendonly'}]},
            ],
            'bgcolor': '#edf2f7', 'bordercolor': '#cbd5e0', 'font': {'size': 11},
        }]
    fig.update_layout(**layout_kw)
    if log_x:
        fig.update_xaxes(type='log')
    else:
        fig.update_xaxes(rangemode='tozero')
    return fig


def plotly_heatmap(grid_data, x_labels, y_labels, title='',
                   x_title='', y_title='', colorscale='RdYlGn'):
    """Build a Plotly heatmap with hover labels.

    Parameters
    ----------
    grid_data : 2D array (rows = y_labels, cols = x_labels)
    x_labels, y_labels : list of str
    colorscale : str   Plotly colorscale name

    Returns
    -------
    plotly.graph_objects.Figure or None
    """
    if not HAS_PLOTLY or grid_data is None:
        return None
    arr = np.array(grid_data, dtype=float)
    text = [[f'{arr[r, c]:.3f}' if not np.isnan(arr[r, c]) else 'N/A'
             for c in range(arr.shape[1])]
            for r in range(arr.shape[0])]
    fig = _go.Figure(_go.Heatmap(
        z=arr.tolist(),
        x=[str(xl) for xl in x_labels],
        y=[str(yl) for yl in y_labels],
        colorscale=colorscale,
        text=text, texttemplate='%{text}',
        hovertemplate='x=%{x}<br>y=%{y}<br>value=%{z:.3f}<extra></extra>',
        showscale=True,
    ))
    fig.update_layout(
        title=title, xaxis_title=x_title, yaxis_title=y_title,
        margin=dict(l=60, r=20, t=60, b=40),
        height=350,
        template='plotly_white',
    )
    return fig


def plotly_sigma(series_data, title='CMA-ES σ Convergence'):
    """Build a Plotly sigma-convergence figure (log y-axis).

    Parameters
    ----------
    series_data : list of dicts with 'label', 'x', 'y', optional 'color'

    Returns
    -------
    plotly.graph_objects.Figure or None
    """
    if not HAS_PLOTLY or not series_data:
        return None
    fig = _go.Figure()
    for s in series_data:
        kw = dict(
            name=s['label'], x=s['x'], y=s['y'], mode='lines',
            hovertemplate='gen %{x}<br>σ=%{y:.4f}<extra></extra>',
        )
        if 'color' in s:
            kw['line'] = dict(color=s['color'])
        fig.add_trace(_go.Scatter(**kw))
    fig.update_layout(
        title=title, xaxis_title='Generation', yaxis_title='σ (log scale)',
        yaxis_type='log',
        legend=dict(
            orientation='v',
            yanchor='top', y=1,
            xanchor='left', x=1.02,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#cbd5e0', borderwidth=1,
        ),
        margin=dict(l=60, r=160, t=80, b=40),
        height=350,
        template='plotly_white',
    )
    return fig


def _write_run_training_report(run_dir, log):
    """Write a per-run training_report.html with Plotly curves.

    Parameters
    ----------
    run_dir : str or Path
    log : dict with keys 'generation', 'best_fitness', 'mean_fitness', 'sigma'
    """
    if not HAS_PLOTLY:
        return
    import datetime
    from pathlib import Path as _Path
    run_dir = _Path(run_dir)
    name = run_dir.name

    gens = [int(g) for g in log.get('generation', [])]
    best = [float(v) for v in log.get('best_fitness', [])]
    mean = [float(v) for v in log.get('mean_fitness', [])]
    sigma_vals = [float(v) for v in log.get('sigma', [])]

    series_wnet = [
        {'label': 'Best W_net', 'x': gens, 'y': best, 'color': '#1f77b4'},
        {'label': 'Mean W_net', 'x': gens, 'y': mean, 'color': '#ff7f0e'},
    ]
    series_sigma = [{'label': 'σ', 'x': gens, 'y': sigma_vals, 'color': '#2ca02c'}]

    fig_curves = plotly_training_curves(series_wnet, title=f'Training — {name}',
                                        xlabel='Generation', ylabel='W_net')
    fig_sig = plotly_sigma(series_sigma, title=f'σ Convergence — {name}')

    n_gens = len(gens)
    best_so_far = max(best) if best else float('nan')
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Training Report — {name}</title>
  <style>{REPORT_CSS}</style>
</head>
<body>
<h1>Training Report — {name}</h1>
<p class="meta">Generated: {ts} · Generations completed: {n_gens} · Best W_net: {best_so_far:.4f}</p>
<h2>Training Curves</h2>
<div class="card">{fig_to_plotly_div(fig_curves, include_plotlyjs='cdn')}</div>
<h2>σ Convergence</h2>
<div class="card">{fig_to_plotly_div(fig_sig, include_plotlyjs=False)}</div>
</body>
</html>"""

    (run_dir / 'training_report.html').write_text(html, encoding='utf-8')


def generate_training_report(sweep_dir, output_path=None, title=''):
    """Scan sweep_dir for subdirs with training_log.npz; write training_report.html.

    Parameters
    ----------
    sweep_dir : str or Path
    output_path : str or Path or None
        Defaults to sweep_dir/training_report.html.
    title : str
    """
    if not HAS_PLOTLY:
        return
    import datetime
    from pathlib import Path as _Path
    sweep_dir = _Path(sweep_dir)
    if not sweep_dir.is_dir():
        return

    series_data = []
    for sub in sorted(sweep_dir.iterdir()):
        if not sub.is_dir():
            continue
        lp = sub / 'training_log.npz'
        if not lp.exists():
            continue
        try:
            d = np.load(lp)
            gens = list(d['generation'].astype(int))
            best = list(d['best_fitness'].astype(float))
            color = PALETTE[len(series_data) % len(PALETTE)]
            series_data.append({'label': sub.name, 'x': gens, 'y': best, 'color': color})
        except Exception:
            continue

    if not series_data:
        return

    report_title = title or f'Training Progress — {sweep_dir.name}'
    fig = plotly_training_curves(series_data, title=report_title,
                                 xlabel='Generation', ylabel='Best W_net')
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{report_title}</title>
  <style>{REPORT_CSS}</style>
</head>
<body>
<h1>{report_title}</h1>
<p class="meta">Generated: {ts} · Runs: {len(series_data)}</p>
<div class="card">{fig_to_plotly_div(fig, include_plotlyjs='cdn')}</div>
</body>
</html>"""

    out = _Path(output_path) if output_path else sweep_dir / 'training_report.html'
    out.write_text(html, encoding='utf-8')