"""
Loss-function comparison experiments for Evolving Ising.
Runs CMA-ES with 5 different objectives on the same physical setup,
then generates a self-contained HTML report at experiments/report.html.
"""

from __future__ import annotations
import base64
import io
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core import IsingModel, TemperatureDiffuser, SeparableCMAES

# ── Experiment parameters ───────────────────────────────────────────
H, W = 64, 64
N = H * W
HOT_T = 3.0
COLD_T = 0.0
ALPHA = 0.35
STEPS_PER_ITER = 2
SWEEPS_PER_ITER = 2
ITERS_EVAL = 80
CHAINS_PER_EVAL = 8
POP_SIZE = 32
CMA_ITERS = 200
J_SCALE = 1.0
SIGMA_INIT = 0.5
SEED = 0

# ── Shared physical setup ──────────────────────────────────────────
ising = IsingModel((H, W), neighborhood="von_neumann", boundary="periodic_lr")
K = ising.K

cols = jnp.arange(W, dtype=jnp.int32)
top_idx = cols  # row 0
bot_idx = (H - 1) * W + cols

pin_mask = jnp.zeros((N,), dtype=bool).at[bot_idx].set(True)
pin_vals = jnp.zeros((N,), dtype=jnp.float32).at[bot_idx].set(HOT_T)

row_vals = jnp.linspace(COLD_T, HOT_T, H, dtype=jnp.float32)
T0_vec = jnp.repeat(row_vals, W)

diffuser = TemperatureDiffuser(alpha=ALPHA, conductance_mode="abs", normalize_mode="row")

flat_idx = jnp.where(ising.mask.reshape(-1))[0]
D = int(flat_idx.shape[0])


def _theta_to_Jpos(theta):
    return jax.nn.softplus(theta) * J_SCALE


def vec_to_Jnk(theta):
    J = jnp.zeros((N * K,), dtype=jnp.float32)
    J = J.at[flat_idx].set(_theta_to_Jpos(theta.astype(jnp.float32)))
    return J.reshape(N, K)


# ── Loss function factory ──────────────────────────────────────────
EXPERIMENTS = {
    "max_top_temp": {
        "title": "Maximize Top-Row Temperature",
        "description": "Maximize mean temperature on the top row. The baseline heat-channeling objective.",
        "formula": "fitness = mean(T_top_row)",
    },
    "max_mean_temp": {
        "title": "Maximize Mean Temperature",
        "description": "Maximize the average temperature across the entire grid.",
        "formula": "fitness = mean(T_all)",
    },
    "min_temp_variance": {
        "title": "Minimize Temperature Variance",
        "description": "Drive the temperature field toward uniformity by minimizing its standard deviation.",
        "formula": "fitness = -std(T_all)",
    },
    "max_neg_energy": {
        "title": "Minimize Ising Energy",
        "description": "Minimize the Ising energy, favoring aligned neighboring spins.",
        "formula": "fitness = -mean(E) / N",
    },
    "max_top_temp_low_energy": {
        "title": "Top-Row Temp + Low Energy (Multi-Objective)",
        "description": "Combine heat transport with spin alignment: maximize top-row temperature while also minimizing energy.",
        "formula": "fitness = mean(T_top) + 0.1 * (-mean(E) / N)",
    },
}


def make_eval_fn(objective: str):
    """Return a JIT-compiled eval function for the given objective."""

    def _dynamics(key, theta):
        """Shared dynamics: returns (spins_f, T_f, J_nk)."""
        J_nk = vec_to_Jnk(theta)
        B = CHAINS_PER_EVAL
        key_s, key_loop = jax.random.split(key)
        spins = ising.init_spins(key_s, B)
        T = jnp.broadcast_to(T0_vec[None, :], (B, N))

        def body(carry, _):
            spins_c, T_c, key_c = carry
            T_c = diffuser.diffuse(
                ising.neighbors, J_nk, ising.mask, T_c,
                steps=STEPS_PER_ITER, pin_mask=pin_mask, pin_values=pin_vals,
            )
            key_c, sub = jax.random.split(key_c)
            spins_c, _ = ising.metropolis_checkerboard_sweeps(
                sub, spins_c, J_nk, T_c, num_sweeps=SWEEPS_PER_ITER,
            )
            return (spins_c, T_c, key_c), None

        (spins_f, T_f, _), _ = jax.lax.scan(
            body, (spins, T, key_loop), None, length=ITERS_EVAL,
        )
        return spins_f, T_f, J_nk

    if objective == "max_top_temp":
        @jax.jit
        def eval_single(key, theta):
            _, T_f, _ = _dynamics(key, theta)
            return jnp.mean(T_f[:, top_idx])
    elif objective == "max_mean_temp":
        @jax.jit
        def eval_single(key, theta):
            _, T_f, _ = _dynamics(key, theta)
            return jnp.mean(T_f)
    elif objective == "min_temp_variance":
        @jax.jit
        def eval_single(key, theta):
            _, T_f, _ = _dynamics(key, theta)
            return -jnp.std(T_f)
    elif objective == "max_neg_energy":
        @jax.jit
        def eval_single(key, theta):
            spins_f, _, J_nk = _dynamics(key, theta)
            E = ising.energy(J_nk, spins_f)
            return -jnp.mean(E) / N
    elif objective == "max_top_temp_low_energy":
        @jax.jit
        def eval_single(key, theta):
            spins_f, T_f, J_nk = _dynamics(key, theta)
            E = ising.energy(J_nk, spins_f)
            return jnp.mean(T_f[:, top_idx]) + 0.1 * (-jnp.mean(E) / N)
    else:
        raise ValueError(f"Unknown objective: {objective}")

    @jax.jit
    def eval_population(key, thetas):
        keys = jax.random.split(key, thetas.shape[0])
        return jax.vmap(eval_single, in_axes=(0, 0))(keys, thetas)

    return eval_single, eval_population


# ── Figure helpers ─────────────────────────────────────────────────
def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def plot_fitness_curve(hist: List[float], title: str) -> str:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(hist, linewidth=1.5)
    ax.set_xlabel("CMA-ES Iteration")
    ax.set_ylabel("Best Fitness")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_temperature(T_final, title: str) -> str:
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(T_final.reshape(H, W), cmap="magma", vmin=COLD_T, vmax=HOT_T)
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_spins(S_final, title: str) -> str:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(S_final.reshape(H, W), cmap="gray", vmin=-1, vmax=1)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_connectivity(J_nk, title: str) -> str:
    W_plot = jnp.abs(J_nk) * ising.mask.astype(jnp.float32)
    out_deg = jnp.sum(W_plot, axis=1)
    dest = ising.neighbors.reshape(-1)
    weights = W_plot.reshape(-1)
    in_deg = jnp.zeros((N,), dtype=W_plot.dtype).at[dest].add(weights)
    in_norm = in_deg / (jnp.max(in_deg) + 1e-8)
    out_norm = out_deg / (jnp.max(out_deg) + 1e-8)
    RGB = np.array(jnp.stack([in_norm, out_norm, jnp.zeros_like(in_norm)], axis=-1).reshape(H, W, 3))

    fig, axs = plt.subplots(1, 3, figsize=(10, 3.5))
    axs[0].imshow(np.array(in_deg.reshape(H, W)), cmap="Reds")
    axs[0].set_title("In-degree")
    axs[0].axis("off")
    axs[1].imshow(np.array(out_deg.reshape(H, W)), cmap="Greens")
    axs[1].set_title("Out-degree")
    axs[1].axis("off")
    axs[2].imshow(RGB)
    axs[2].set_title("Composite (R=in, G=out)")
    axs[2].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_comparison(all_T_finals: Dict[str, np.ndarray]) -> str:
    n = len(all_T_finals)
    fig, axs = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axs = [axs]
    for ax, (name, T_final) in zip(axs, all_T_finals.items()):
        im = ax.imshow(T_final.reshape(H, W), cmap="magma", vmin=COLD_T, vmax=HOT_T)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle("Final Temperature Comparison", fontsize=13)
    fig.tight_layout()
    return fig_to_base64(fig)


# ── Run one experiment ─────────────────────────────────────────────
@dataclass
class ExperimentResult:
    name: str
    best_hist: List[float]
    best_fitness: float
    top_row_temp: float
    mean_temp: float
    T_final: np.ndarray
    S_final: np.ndarray
    J_best: np.ndarray
    elapsed: float


def run_experiment(name: str) -> ExperimentResult:
    print(f"\n{'='*60}")
    print(f"  Experiment: {name}")
    print(f"  {EXPERIMENTS[name]['title']}")
    print(f"{'='*60}")
    t0 = time.time()

    eval_single, eval_population = make_eval_fn(name)

    key = jax.random.PRNGKey(SEED)
    es = SeparableCMAES(dim=D, pop_size=POP_SIZE, sigma_init=SIGMA_INIT, seed=SEED)

    best_hist = []
    best_theta = None
    best_fit = -jnp.inf

    for t in range(CMA_ITERS):
        X = es.ask()
        key, sub = jax.random.split(key)
        fitness = eval_population(sub, X)
        es.tell(X, fitness)
        idx = int(jnp.argmax(fitness))
        val = float(fitness[idx])
        if val > float(best_fit) or best_theta is None:
            best_fit = fitness[idx]
            best_theta = X[idx]
        best_hist.append(val)
        if (t + 1) % 50 == 0:
            print(f"  Iter {t+1:4d}/{CMA_ITERS}  best fitness: {val:.6f}")

    J_best = vec_to_Jnk(best_theta)

    # Visualization rollout
    key_s, key_loop = jax.random.split(key)
    spins = ising.init_spins(key_s, 1)
    T = jnp.broadcast_to(T0_vec[None, :], (1, N))
    for _ in range(ITERS_EVAL):
        T = diffuser.diffuse(
            ising.neighbors, J_best, ising.mask, T,
            steps=STEPS_PER_ITER, pin_mask=pin_mask, pin_values=pin_vals,
        )
        key_loop, sub = jax.random.split(key_loop)
        spins, _ = ising.metropolis_checkerboard_sweeps(
            sub, spins, J_best, T, num_sweeps=SWEEPS_PER_ITER,
        )

    T_final = np.array(T[0])
    S_final = np.array(spins[0])
    top_row_temp = float(jnp.mean(T[0, top_idx]))
    mean_temp = float(jnp.mean(T[0]))

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s | top-row T: {top_row_temp:.4f} | mean T: {mean_temp:.4f}")

    return ExperimentResult(
        name=name,
        best_hist=best_hist,
        best_fitness=float(best_fit),
        top_row_temp=top_row_temp,
        mean_temp=mean_temp,
        T_final=T_final,
        S_final=S_final,
        J_best=np.array(J_best),
        elapsed=elapsed,
    )


# ── HTML report ────────────────────────────────────────────────────
def generate_report(results: List[ExperimentResult]):
    os.makedirs("experiments", exist_ok=True)

    # Generate per-experiment figures
    experiment_sections = []
    all_T_finals = {}

    for r in results:
        info = EXPERIMENTS[r.name]
        fig_fitness = plot_fitness_curve(r.best_hist, f"Fitness Curve — {info['title']}")
        fig_temp = plot_temperature(r.T_final, f"Final Temperature — {r.name}")
        fig_spins = plot_spins(r.S_final, f"Final Spins — {r.name}")
        fig_conn = plot_connectivity(jnp.array(r.J_best), f"Connectivity — {r.name}")
        all_T_finals[r.name] = r.T_final

        experiment_sections.append(f"""
        <div class="experiment-card">
            <h2>{info['title']}</h2>
            <p class="desc">{info['description']}</p>
            <p class="formula"><code>{info['formula']}</code></p>
            <div class="metrics">
                <span>Final fitness: <strong>{r.best_fitness:.6f}</strong></span>
                <span>Top-row temp: <strong>{r.top_row_temp:.4f}</strong></span>
                <span>Mean temp: <strong>{r.mean_temp:.4f}</strong></span>
                <span>Runtime: <strong>{r.elapsed:.1f}s</strong></span>
            </div>
            <div class="grid2x2">
                <div><img src="data:image/png;base64,{fig_fitness}" alt="Fitness curve"></div>
                <div><img src="data:image/png;base64,{fig_temp}" alt="Temperature"></div>
                <div><img src="data:image/png;base64,{fig_spins}" alt="Spins"></div>
                <div><img src="data:image/png;base64,{fig_conn}" alt="Connectivity"></div>
            </div>
        </div>
        """)

    # Comparison figure
    fig_comparison = plot_comparison(all_T_finals)

    # Summary table rows
    table_rows = "".join(
        f"<tr><td>{r.name}</td><td>{r.best_fitness:.6f}</td>"
        f"<td>{r.top_row_temp:.4f}</td><td>{r.mean_temp:.4f}</td>"
        f"<td>{r.elapsed:.1f}s</td></tr>"
        for r in results
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
    .experiment-card {{
        background: #161b22; border: 1px solid #30363d; border-radius: 8px;
        padding: 1.5rem; margin-bottom: 2rem;
    }}
    .desc {{ color: #8b949e; margin-bottom: 0.3rem; }}
    .formula {{ color: #d2a8ff; margin-bottom: 0.8rem; }}
    .formula code {{ background: #1c2028; padding: 2px 6px; border-radius: 4px; }}
    .metrics {{
        display: flex; gap: 2rem; flex-wrap: wrap;
        margin-bottom: 1rem; font-size: 0.9rem; color: #8b949e;
    }}
    .metrics strong {{ color: #c9d1d9; }}
    .grid2x2 {{
        display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;
    }}
    .grid2x2 img {{ width: 100%; border-radius: 4px; }}
    .comparison {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                   padding: 1.5rem; margin-bottom: 2rem; }}
    .comparison img {{ width: 100%; border-radius: 4px; }}
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
<p class="subtitle">CMA-ES optimization with different fitness objectives on a {H}x{W} Ising lattice</p>

<div class="config">
    <strong>Experiment Configuration:</strong>
    Grid: <code>{H}x{W}</code> |
    Neighborhood: <code>von_neumann</code> |
    Boundary: <code>periodic_lr</code> |
    Hot boundary: <code>T={HOT_T}</code> (bottom row) |
    Pop size: <code>{POP_SIZE}</code> |
    CMA-ES iters: <code>{CMA_ITERS}</code> |
    Eval iters: <code>{ITERS_EVAL}</code> |
    Chains: <code>{CHAINS_PER_EVAL}</code> |
    Alpha: <code>{ALPHA}</code>
</div>

{"".join(experiment_sections)}

<div class="comparison">
    <h2>Side-by-Side Temperature Comparison</h2>
    <img src="data:image/png;base64,{fig_comparison}" alt="Comparison">
</div>

<div class="comparison">
    <h2>Summary Table</h2>
    <table>
        <tr><th>Experiment</th><th>Best Fitness</th><th>Top-Row Temp</th><th>Mean Temp</th><th>Runtime</th></tr>
        {table_rows}
    </table>
</div>

</body>
</html>"""

    path = os.path.join("experiments", "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nReport saved to {path}")


# ── Main ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = []
    for name in EXPERIMENTS:
        results.append(run_experiment(name))
    generate_report(results)
    print("\nAll experiments complete!")
