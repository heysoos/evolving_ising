"""Shared utilities for HTML report generation.

Used by exp1_report.py, exp2_report.py, and exp3_report.py.
Provides: unified CSS, figure helpers, simulation loop for animations,
GIF generation, and an interactive canvas-based training curve chart.
"""

import io
import base64
import json
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
"""


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

def _make_budget(budget_type, neighbors_np, mask_np, config):
    """Instantiate the appropriate budget class from config."""
    from work_extraction.budgets import NoBudget, BondBudget, NeighbourhoodBudget, DiffusingBudget
    alpha = float(config.get('budget_alpha', config.get('mag_ema_alpha', 0.05)))
    if budget_type == 'none':
        return NoBudget()
    elif budget_type == 'bond':
        return BondBudget(neighbors_np, mask_np, alpha=alpha)
    elif budget_type == 'neighbourhood':
        gamma = float(config.get('gamma', 0.25))
        return NeighbourhoodBudget(neighbors_np, mask_np, alpha=alpha, gamma=gamma)
    elif budget_type == 'diffusing':
        D = float(config.get('D', 0.1))
        tau_mu = float(config.get('tau_mu', 20.0))
        return DiffusingBudget(neighbors_np, mask_np, alpha=alpha, D=D, tau_mu=tau_mu)
    else:
        raise ValueError(f'Unknown budget_type: {budget_type!r}')


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
    budget_alpha = float(config.get('budget_alpha', config.get('mag_ema_alpha', 0.05)))
    gamma = float(config.get('gamma', 0.25))
    D_diff = float(config.get('D', 0.1))
    tau_mu = float(config.get('tau_mu', 20.0))
    spc = int(config.get('steps_per_cycle', 100)) if steps_per_cycle is None else steps_per_cycle
    T_norm_denom = delta_T if delta_T > 0 else 1.0
    J_crit = T_mean / 2.269          # critical coupling (scalar)

    # ---------------------------------------------------- precomputed JAX arrays
    N = model.n
    K = model.K
    neighbors_np = np.asarray(model.neighbors)
    mask_np = np.asarray(model.mask, dtype=bool)
    mask_f = jnp.asarray(mask_np, dtype=jnp.float32)
    neighbors_jax = jnp.asarray(neighbors_np, dtype=jnp.int32)
    K_eff_jax = jnp.sum(mask_f, axis=1)

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

    # ----------------------------------------- pure-JAX budget functions
    # Mirrors make_jax_eval_fn exactly so the budget dynamics are identical.
    if budget_type == 'none':
        def bud_init(): return jnp.zeros(1, dtype=jnp.float32)
        def bud_update(bud, s_bef, s_aft): return bud
        def bud_get(bud, si, sk, sj): return jnp.full(n_updates, jnp.inf, dtype=jnp.float32)
        def bud_spend(bud, si, sk, sj, costs, mask): return bud

    elif budget_type == 'bond':
        def bud_init(): return jnp.zeros((N, K), dtype=jnp.float32)
        def bud_update(bud, s_bef, s_aft):
            ordering = jnp.maximum(
                0.0, s_aft[:, None] * s_aft[neighbors_jax]
                   - s_bef[:, None] * s_bef[neighbors_jax]
            ) * mask_f
            return bud + budget_alpha * ordering
        def bud_get(bud, si, sk, sj): return jnp.maximum(0.0, bud[si, sk])
        def bud_spend(bud, si, sk, sj, costs, mask):
            return jnp.maximum(0.0, bud.at[si, sk].add(-jnp.where(mask, costs, 0.0)))

    elif budget_type == 'neighbourhood':
        def bud_init(): return jnp.zeros(N, dtype=jnp.float32)
        def bud_update(bud, s_bef, s_aft):
            ordering = jnp.maximum(
                0.0, s_aft[:, None] * s_aft[neighbors_jax]
                   - s_bef[:, None] * s_bef[neighbors_jax]
            ) * mask_f
            return bud + budget_alpha * ordering.sum(axis=1)
        def bud_get(bud, si, sk, sj):
            nbhd = bud + gamma * (bud[neighbors_jax] * mask_f).sum(axis=1)
            return jnp.minimum(nbhd[si], nbhd[sj])
        def bud_spend(bud, si, sk, sj, costs, mask):
            half = jnp.where(mask, costs / 2.0, 0.0)
            return jnp.maximum(0.0, bud.at[si].add(-half).at[sj].add(-half))

    elif budget_type == 'diffusing':
        def bud_init(): return jnp.zeros(N, dtype=jnp.float32)
        def bud_update(bud, s_bef, s_aft):
            ordering = jnp.maximum(
                0.0, s_aft[:, None] * s_aft[neighbors_jax]
                   - s_bef[:, None] * s_bef[neighbors_jax]
            ) * mask_f
            eta = budget_alpha * ordering.sum(axis=1)
            laplacian = (bud[neighbors_jax] * mask_f).sum(axis=1) - K_eff_jax * bud
            return jnp.maximum(0.0, bud + D_diff * laplacian + eta - bud / tau_mu)
        def bud_get(bud, si, sk, sj):
            return jnp.minimum(jnp.maximum(0.0, bud[si]), jnp.maximum(0.0, bud[sj]))
        def bud_spend(bud, si, sk, sj, costs, mask):
            half = jnp.where(mask, costs / 2.0, 0.0)
            return jnp.maximum(0.0, bud.at[si].add(-half).at[sj].add(-half))

    else:
        raise ValueError(f'Unknown budget_type: {budget_type!r}')

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
        bud = bud_update(bud, s_bef_f, s_aft_f)
        mag_ema = mag_alpha * s_aft_f + (1.0 - mag_alpha) * mag_ema

        if params_jax is not None:
            perm = jax.random.permutation(sub_b, n_bonds_total)[:n_updates]
            si = valid_i_jax[perm]
            sk = valid_k_jax[perm]
            sj = valid_j_jax[perm]
            T_norm = (T_t - T_mean) / T_norm_denom
            bud_vals = bud_get(bud, si, sk, sj)
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
            bud = bud_spend(bud, si, sk, sj, costs, can_apply)
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
            spins, key, J_init_jax, bud_init(),
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
# Interactive canvas chart (pure JS — no external dependencies)
# ---------------------------------------------------------------------------

# 12-color palette for multi-series charts
PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78',
]

_CANVAS_CHART_JS = r"""
(function() {
  const cid = '{CID}';
  const canvas = document.getElementById(cid);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const P = {t:32, r:20, b:44, l:58};
  const series = {SERIES};
  const baseline = {BASELINE};
  const title = '{TITLE}';
  const xlabel = '{XLABEL}';
  const ylabel = '{YLABEL}';
  let hidden = {}, hoverMx = null;

  const allX = series.flatMap(s=>s.x);
  const allY = series.flatMap(s=>s.y);
  const xMin=Math.min(...allX), xMax=Math.max(...allX);
  const rawYMin=Math.min(...allY, baseline!==null?baseline:Infinity);
  const rawYMax=Math.max(...allY, baseline!==null?baseline:-Infinity);
  const yPad=(rawYMax-rawYMin)*0.05||1;
  const yMin=rawYMin-yPad, yMax=rawYMax+yPad;

  function tx(x){ return P.l+(x-xMin)/(xMax-xMin||1)*(W-P.l-P.r); }
  function ty(y){ return H-P.b-(y-yMin)/(yMax-yMin||1)*(H-P.t-P.b); }
  function fromMx(mx){ return xMin+(mx-P.l)/(W-P.l-P.r)*(xMax-xMin); }

  function draw(){
    ctx.clearRect(0,0,W,H);
    // grid
    ctx.strokeStyle='#e4e8ee'; ctx.lineWidth=0.7;
    for(let i=0;i<=5;i++){
      const y=yMin+(yMax-yMin)*i/5;
      const cy=ty(y);
      ctx.beginPath(); ctx.moveTo(P.l,cy); ctx.lineTo(W-P.r,cy); ctx.stroke();
      ctx.fillStyle='#555'; ctx.font='10px sans-serif'; ctx.textAlign='right';
      ctx.fillText(y.toFixed(1),P.l-5,cy+3.5);
    }
    // x ticks
    const nxt=Math.ceil(xMax/5/50)*50||100;
    ctx.fillStyle='#555'; ctx.textAlign='center';
    for(let x=0;x<=xMax;x+=nxt){
      ctx.fillText(x,tx(x),H-P.b+13);
    }
    // axis labels
    ctx.fillStyle='#333'; ctx.font='11px sans-serif'; ctx.textAlign='center';
    ctx.fillText(xlabel, (P.l+W-P.r)/2, H-2);
    ctx.save(); ctx.translate(11,(P.t+H-P.b)/2); ctx.rotate(-Math.PI/2);
    ctx.fillText(ylabel,0,0); ctx.restore();
    // title
    ctx.font='bold 12px sans-serif';
    ctx.fillText(title,(P.l+W-P.r)/2,16);
    // baseline
    if(baseline!==null){
      const by=ty(baseline);
      ctx.strokeStyle='#c0392b'; ctx.lineWidth=1.5; ctx.setLineDash([6,4]);
      ctx.beginPath(); ctx.moveTo(P.l,by); ctx.lineTo(W-P.r,by); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle='#c0392b'; ctx.font='10px sans-serif'; ctx.textAlign='right';
      ctx.fillText('baseline',W-P.r-2,by-4);
    }
    // series
    series.forEach((s,idx)=>{
      if(hidden[idx]) return;
      ctx.strokeStyle=s.color; ctx.lineWidth=1.5; ctx.setLineDash([]);
      ctx.beginPath();
      s.x.forEach((xi,i)=>{ i===0?ctx.moveTo(tx(xi),ty(s.y[i])):ctx.lineTo(tx(xi),ty(s.y[i])); });
      ctx.stroke();
    });
    // hover
    if(hoverMx!==null){
      const hx=hoverMx;
      if(hx>=P.l&&hx<=W-P.r){
        ctx.strokeStyle='rgba(0,0,0,0.18)'; ctx.lineWidth=1; ctx.setLineDash([]);
        ctx.beginPath(); ctx.moveTo(hx,P.t); ctx.lineTo(hx,H-P.b); ctx.stroke();
        const wx=fromMx(hx);
        const lines=[];
        series.forEach((s,idx)=>{
          if(hidden[idx]) return;
          let best=null, bd=Infinity;
          s.x.forEach((xi,i)=>{ const d=Math.abs(xi-wx); if(d<bd){bd=d;best=i;} });
          if(best!==null) lines.push({label:s.label,val:s.y[best],color:s.color,gen:s.x[best]});
        });
        if(lines.length){
          const bw=150, bh=lines.length*15+20;
          let bx=hx+8, by=P.t+4;
          if(bx+bw>W-P.r) bx=hx-bw-8;
          ctx.fillStyle='rgba(255,255,255,0.94)'; ctx.strokeStyle='#bbb'; ctx.lineWidth=1;
          ctx.beginPath(); ctx.roundRect(bx,by,bw,bh,4); ctx.fill(); ctx.stroke();
          ctx.font='10px monospace'; ctx.textAlign='left';
          ctx.fillStyle='#333';
          ctx.fillText('gen '+Math.round(lines[0].gen),bx+6,by+13);
          lines.forEach((l,i)=>{
            ctx.fillStyle=l.color;
            ctx.fillText(l.label+': '+l.val.toFixed(2),bx+6,by+14+14*(i+1));
          });
        }
      }
    }
  }

  canvas.addEventListener('mousemove',e=>{
    const r=canvas.getBoundingClientRect();
    hoverMx=e.clientX-r.left; draw();
  });
  canvas.addEventListener('mouseleave',()=>{ hoverMx=null; draw(); });

  // legend buttons
  const leg=document.getElementById(cid+'_legend');
  if(leg){ series.forEach((s,idx)=>{
    const b=document.createElement('button');
    b.style.cssText='background:'+s.color+';border:none;border-radius:3px;color:#fff;'+
      'padding:3px 9px;margin:2px 3px;font-size:11px;cursor:pointer;';
    b.textContent=s.label;
    b.title='Click to toggle';
    b.onclick=()=>{ hidden[idx]=!hidden[idx]; b.style.opacity=hidden[idx]?'0.35':'1'; draw(); };
    leg.appendChild(b);
  }); }

  draw();
})();
"""


def canvas_chart_html(series_data, canvas_id, title='', xlabel='Generation', ylabel='W_net',
                      width=720, height=290, baseline=None):
    """Return a self-contained HTML+JS canvas chart.

    Parameters
    ----------
    series_data : list of dicts with keys 'label', 'x' (list), 'y' (list), 'color' (str)
    canvas_id : str  — unique HTML id for the canvas element
    baseline : float or None — horizontal reference line
    """
    series_json = json.dumps(series_data)
    baseline_js = 'null' if baseline is None else str(float(baseline))
    js = (_CANVAS_CHART_JS
          .replace('{CID}', canvas_id)
          .replace('{SERIES}', series_json)
          .replace('{BASELINE}', baseline_js)
          .replace('{TITLE}', title.replace("'", "\\'"))
          .replace('{XLABEL}', xlabel.replace("'", "\\'"))
          .replace('{YLABEL}', ylabel.replace("'", "\\'")))
    return (
        f'<canvas id="{canvas_id}" width="{width}" height="{height}" '
        f'style="max-width:100%;border:1px solid #d0d9e8;border-radius:6px;'
        f'box-shadow:0 2px 8px rgba(0,0,0,.07);"></canvas>\n'
        f'<div id="{canvas_id}_legend" style="margin:.4em 0 1em;"></div>\n'
        f'<script>{js}</script>\n'
    )


# ---------------------------------------------------------------------------
# Scenario selector widget
# ---------------------------------------------------------------------------

def scenario_selector_html(scenario_ids, labels, default_id, title='Select Run'):
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
        f'onchange="{hide_all};document.getElementById(this.value).style.display=\'block\'">\n'
        f'{options}\n'
        f'  </select>\n'
        f'</div>\n'
    )