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

    Parameters
    ----------
    model : IsingModel
    config : dict
    budget_type : str
    params_flat : ndarray or None
        If provided, the controller MLP weights to use.
    n_cycles : int
    steps_per_cycle : int or None  (defaults to config['steps_per_cycle'])
    frame_skip : int  — save a frame every this many steps
    warmup_sweeps : int

    Returns
    -------
    spin_frames : list of (L, L) int8 arrays
    J_mean_frames : list of (L, L) float32 arrays  (mean J strength per site)
    W_net_cycles : list of floats
    wnet_trace : list of floats  (cumulative W_net at each captured frame)
    """
    import jax
    import jax.numpy as jnp
    from work_extraction.controller import LocalController

    L = config.get('L', 32)
    T_mean = float(config['T_mean'])
    delta_T = float(config['delta_T'])
    tau = float(config['tau'])
    J_init = float(config['J_init'])
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

    neighbors_np = np.asarray(model.neighbors)
    mask_np = np.asarray(model.mask, dtype=bool)
    N, K = neighbors_np.shape
    valid_count_per_site = mask_np.sum(axis=1)

    valid_i, valid_k = np.where(mask_np)
    valid_j = neighbors_np[valid_i, valid_k]
    n_bonds = len(valid_i)
    n_updates = max(1, int(n_bonds * bond_update_frac))

    budget = _make_budget(budget_type, neighbors_np, mask_np, config)

    controller = None
    if params_flat is not None:
        controller = LocalController(delta_J_max=delta_J_max, hidden_size=hidden_size)
        controller.set_params(params_flat)

    key = jax.random.PRNGKey(42)
    key, ik, wk = jax.random.split(key, 3)
    spins = model.init_spins(ik, 1)
    J_nk = np.full((N, K), J_init, dtype=np.float32) * mask_np
    J_jax = jnp.asarray(J_nk)

    spins, _ = model.metropolis_checkerboard_sweeps(wk, spins, J_jax, T_mean, warmup_sweeps)

    mag_ema = np.zeros(N, dtype=np.float32)
    spin_frames = []
    J_mean_frames = []
    W_net_cycles = []
    wnet_trace = []
    running_wnet = 0.0
    rng = np.random.RandomState(0)

    t_global = 0
    for _cycle in range(n_cycles):
        Q_in = Q_out = 0.0
        E_prev = float(jnp.mean(model.energy(J_jax, spins)))

        for t in range(spc):
            T_t = T_mean + delta_T * np.sin(2.0 * np.pi * t_global / tau)

            spins_bef_f = np.asarray(spins[0], dtype=np.float32)
            key, sk = jax.random.split(key)
            spins, _ = model.metropolis_checkerboard_sweeps(sk, spins, J_jax, float(T_t), num_sweeps)
            spins_aft_f = np.asarray(spins[0], dtype=np.float32)

            E_now = float(jnp.mean(model.energy(J_jax, spins)))
            dE = E_now - E_prev
            Q_in += max(0.0, dE)
            Q_out += max(0.0, -dE)
            running_wnet += max(0.0, -dE) - max(0.0, dE)
            E_prev = E_now

            budget.update(spins_bef_f, spins_aft_f, J_nk, T_t)
            mag_ema = mag_alpha * spins_aft_f + (1.0 - mag_alpha) * mag_ema

            if controller is not None:
                perm = rng.permutation(n_bonds)[:n_updates]
                si = valid_i[perm]
                sk_arr = valid_k[perm]
                sj = valid_j[perm]

                T_norm = (T_t - T_mean) / (delta_T if delta_T > 0 else 1.0)
                bud_vals = np.asarray(budget.get_all_budgets_for_bonds_nk(si, sk_arr, sj),
                                      dtype=np.float32)
                bud_norm = np.tanh(bud_vals / B_scale)

                x = np.stack([
                    spins_aft_f[si], spins_aft_f[sj], mag_ema[si],
                    np.full(n_updates, T_norm, dtype=np.float32), bud_norm,
                ], axis=-1)

                dJ = np.asarray(controller.forward(x)).ravel()
                costs = np.abs(spins_aft_f[si] * spins_aft_f[sj] * dJ) + lam * np.abs(dJ)
                can_apply = bud_vals >= costs

                dJ_gated = np.where(can_apply, dJ, 0.0)
                J_nk[si, sk_arr] = np.clip(J_nk[si, sk_arr] + dJ_gated, J_min, J_max)
                J_nk *= mask_np
                J_jax = jnp.asarray(J_nk)
                budget.spend_all_nk(si, sk_arr, sj, costs, can_apply)

            if t % frame_skip == 0:
                spin_frames.append(np.asarray(spins[0]).reshape(L, L).copy())
                J_mean = np.where(
                    valid_count_per_site > 0,
                    (J_nk * mask_np).sum(axis=1) / np.maximum(valid_count_per_site, 1),
                    J_init,
                )
                J_mean_frames.append(J_mean.reshape(L, L).astype(np.float32).copy())
                wnet_trace.append(running_wnet)

            t_global += 1

        W_net_cycles.append(Q_out - Q_in)

    return spin_frames, J_mean_frames, W_net_cycles, wnet_trace


# ---------------------------------------------------------------------------
# GIF generation
# ---------------------------------------------------------------------------

def frames_to_gif_b64(spin_frames, J_mean_frames, fps=8, max_frames=200, scale=5,
                      wnet_trace=None):
    """Render spin + J frames to an animated GIF using PIL; return base64 string or None.

    Each GIF frame has two rows:
      Top: spin state (left) | mean J per site (right)
      Bottom (optional): cumulative W_net trace with a moving cursor

    Parameters
    ----------
    wnet_trace : list of floats or None
        Cumulative W_net at each frame. If provided, a trace strip is added.
    """
    if not HAS_PIL:
        print('  Pillow not installed; skipping GIF (pip install pillow)')
        return None
    if not spin_frames:
        return None

    from PIL import Image as _PILImage

    step = max(1, len(spin_frames) // max_frames)
    sf = spin_frames[::step]
    jf = J_mean_frames[::step]
    wt = wnet_trace[::step] if wnet_trace is not None else None

    J_arr = np.stack(jf)
    j_vmin, j_vmax = float(J_arr.min()), float(J_arr.max())
    if j_vmax <= j_vmin:
        j_vmax = j_vmin + 0.01

    cmap = matplotlib.colormaps['viridis']
    L = sf[0].shape[0]
    strip_h = L // 2  # height of W_net strip (pre-scale)

    # Pre-compute W_net trace bounds for consistent scaling across all frames
    if wt is not None and len(wt) > 1:
        wt_arr = np.array(wt, dtype=np.float64)
        w_vmin = float(min(wt_arr.min(), 0.0))
        w_vmax = float(max(wt_arr.max(), w_vmin + 1.0))
    else:
        wt = None  # disable trace panel if too short

    def _render_wnet_strip(frame_idx, n_total, strip_width):
        """Render cumulative W_net trace as an RGB numpy strip."""
        strip = np.full((strip_h, strip_width, 3), 28, dtype=np.uint8)  # dark bg
        if wt is None:
            return strip
        # Draw zero line
        y_zero = int((1.0 - (0.0 - w_vmin) / (w_vmax - w_vmin)) * (strip_h - 1))
        y_zero = max(0, min(y_zero, strip_h - 1))
        strip[y_zero, :] = [80, 80, 80]
        # Draw trace up to current frame
        for i in range(frame_idx + 1):
            x = int(i / n_total * strip_width)
            x = min(x, strip_width - 1)
            y = int((1.0 - (wt_arr[i] - w_vmin) / (w_vmax - w_vmin)) * (strip_h - 1))
            y = max(0, min(y, strip_h - 1))
            strip[y, x] = [80, 200, 120]  # green trace
        # Draw vertical cursor at current frame
        x_cur = int(frame_idx / n_total * strip_width)
        x_cur = min(x_cur, strip_width - 1)
        strip[:, x_cur] = [200, 80, 80]  # red cursor
        return strip

    n_frames = len(sf)
    pil_frames = []
    for frame_idx, (s, j) in enumerate(zip(sf, jf)):
        # Top row: spin (left) | J map (right)
        spin_rgb = np.where(s[:, :, np.newaxis] > 0, 220, 30).astype(np.uint8)
        spin_rgb = np.broadcast_to(spin_rgb, (*s.shape, 3)).copy()

        j_norm = np.clip((j - j_vmin) / (j_vmax - j_vmin), 0.0, 1.0)
        j_rgb = (cmap(j_norm)[:, :, :3] * 255).astype(np.uint8)

        top = np.concatenate([spin_rgb, j_rgb], axis=1)  # (L, 2L, 3)

        if wt is not None:
            bottom = _render_wnet_strip(frame_idx, n_frames, top.shape[1])
            combined = np.concatenate([top, bottom], axis=0)  # (L + strip_h, 2L, 3)
        else:
            combined = top

        img = _PILImage.fromarray(combined, mode='RGB')
        H, W = combined.shape[:2]
        img = img.resize((W * scale, H * scale), _PILImage.NEAREST)
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