"""
generate_report_from_files.py
==============================
Re-generate (or update) the HTML report for a saved experiment run without
re-running the optimisation.

Loads each experiment's ``best_final.npz``, runs a short physics rollout to
produce the temperature+spin animation frames, then calls ``generate_report``
with the full rollout data.

Usage
-----
# Use the most-recently created run in results/
python generate_report_from_files.py

# Point at a specific run directory
python generate_report_from_files.py results/run_20260307_024531

# Control the rollout length and fps
python generate_report_from_files.py --rollout-steps 120 --fps 20

# Skip animations (fast, no Pillow dependency)
python generate_report_from_files.py --no-animation
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", nargs="?", default=None,
                   help="Path to a run directory (e.g. results/run_YYYYMMDD_HHMMSS). "
                        "Defaults to the most recently created run in results/.")
    p.add_argument("--rollout-steps", type=int, default=100,
                   help="Number of coupled diffusion+Metropolis steps for the animation rollout. "
                        "Default: 100.")
    p.add_argument("--fps", type=int, default=15,
                   help="Frames per second for the embedded GIF animation. Default: 15.")
    p.add_argument("--no-animation", action="store_true",
                   help="Skip the rollout and animation entirely (faster).")
    p.add_argument("--experiments-dir", default="results",
                   help="Root directory that contains run subdirectories. Default: results/")
    return p.parse_args()


# ── Run-directory discovery ───────────────────────────────────────────────────

def _find_latest_run(experiments_dir: str) -> str:
    """Return the most-recently modified run_* subdirectory."""
    runs = [
        os.path.join(experiments_dir, d)
        for d in os.listdir(experiments_dir)
        if d.startswith("run_") and
           os.path.isdir(os.path.join(experiments_dir, d))
    ]
    if not runs:
        sys.exit(f"No run_* directories found in {experiments_dir!r}")
    return max(runs, key=os.path.getmtime)


def _find_experiment_dirs(run_dir: str) -> List[str]:
    """Return all subdirectories of run_dir that contain best_final.npz."""
    out = []
    for name in sorted(os.listdir(run_dir)):
        exp_dir = os.path.join(run_dir, name)
        if os.path.isdir(exp_dir) and os.path.isfile(
                os.path.join(exp_dir, "best_final.npz")):
            out.append(exp_dir)
    return out


# ── Infer grid parameters from npz ───────────────────────────────────────────

def _infer_grid(J_best: np.ndarray) -> Tuple[int, int, str, str]:
    """
    Guess grid geometry from J_best.shape = (N, K).

    Returns (H, W, neighborhood, boundary).
    Currently assumes square grid and periodic_lr boundary (matching
    run_experiments.py defaults).  Edit here if you use different settings.
    """
    N, K = J_best.shape
    side = int(round(math.sqrt(N)))
    if side * side != N:
        sys.exit(f"Cannot infer square grid from N={N} (sqrt={math.sqrt(N):.2f}). "
                 "Edit _infer_grid() for non-square grids.")
    H = W = side
    neighborhood = "von_neumann" if K == 4 else "moore"
    boundary     = "periodic_lr"   # matches run_experiments.py
    return H, W, neighborhood, boundary


def _infer_hot_cold(T_final: np.ndarray, W: int) -> Tuple[float, float]:
    """
    Infer HOT_T and COLD_T from the final temperature field.

    The hot boundary is the bottom row (which is pinned), so its mean is HOT_T.
    COLD_T is taken as 0.0 (the default).
    """
    hot_t  = float(np.mean(T_final[-W:]))   # bottom row
    cold_t = 0.0
    return round(hot_t, 3), cold_t


# ── Load ExperimentResult from npz ───────────────────────────────────────────

def _load_result(exp_dir: str, name: str) -> "ExperimentResult":
    """Reconstruct an ExperimentResult from best_final.npz + the IsingModel."""
    from evolving_ising.experiment import ExperimentResult

    npz = np.load(os.path.join(exp_dir, "best_final.npz"), allow_pickle=False)
    J_best = npz["J_best"]
    H, W, neighborhood, boundary = _infer_grid(J_best)

    from evolving_ising import IsingModel
    ising = IsingModel((H, W), neighborhood=neighborhood, boundary=boundary)

    # elapsed not saved — estimate from fitness_stats.png mtime vs start, fallback 0
    elapsed = 0.0

    return ExperimentResult(
        name        = name,
        best_hist   = npz["best_hist"].tolist(),
        best_so_far = npz["best_so_far"].tolist(),
        mean_hist   = npz["mean_hist"].tolist(),
        std_hist    = npz["std_hist"].tolist(),
        best_fitness= float(npz["best_fitness"]),
        top_row_temp= float(npz["top_row_temp"]),
        mean_temp   = float(npz["mean_temp"]),
        T_final     = npz["T_final"],
        S_final     = npz["S_final"],
        J_best      = J_best,
        neighbors   = np.array(ising.neighbors),
        mask        = np.array(ising.mask),
        elapsed     = elapsed,
    )


# ── Physics rollout ───────────────────────────────────────────────────────────

def _run_rollout(
    npz_path:      str,
    rollout_steps: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load J_best from npz, reconstruct the physics setup, and run a rollout
    collecting one (T, S) frame per step.

    Parameters
    ----------
    npz_path      : path to best_final.npz
    rollout_steps : number of coupled diffusion+Metropolis steps

    Returns
    -------
    (T_frames, S_frames) — each a list of (N,) numpy arrays
    """
    import jax
    import jax.numpy as jnp
    from evolving_ising import IsingModel, TemperatureDiffuser

    npz    = np.load(npz_path, allow_pickle=False)
    J_best = jnp.array(npz["J_best"])
    N, K   = J_best.shape
    H, W, neighborhood, boundary = _infer_grid(np.array(J_best))

    T_final = npz["T_final"]
    hot_t, cold_t = _infer_hot_cold(T_final, W)

    ising    = IsingModel((H, W), neighborhood=neighborhood, boundary=boundary)
    diffuser = TemperatureDiffuser(alpha=0.35, conductance_mode="abs", normalize_mode="row")

    # Reconstruct pin mask: bottom row pinned at hot_t
    cols       = jnp.arange(W, dtype=jnp.int32)
    bot_idx    = (H - 1) * W + cols
    pin_mask   = jnp.zeros(N, dtype=bool).at[bot_idx].set(True)
    pin_values = jnp.zeros(N, dtype=jnp.float32).at[bot_idx].set(hot_t)

    # Initial temperature: linear gradient cold (top) → hot (bottom)
    T0 = jnp.repeat(jnp.linspace(cold_t, hot_t, H, dtype=jnp.float32), W)

    key = jax.random.PRNGKey(7)
    key, sub = jax.random.split(key)
    spins = ising.init_spins(sub, batch_size=1)
    T     = jnp.broadcast_to(T0[None], (1, N))

    T_frames: List[np.ndarray] = []
    S_frames: List[np.ndarray] = []

    for _ in range(rollout_steps):
        T = diffuser.diffuse(ising.neighbors, J_best, ising.mask, T,
                             steps=2, pin_mask=pin_mask, pin_values=pin_values)
        key, sub = jax.random.split(key)
        spins, _ = ising.metropolis_checkerboard_sweeps(sub, spins, J_best, T, num_sweeps=2)
        T_frames.append(np.array(T[0]))
        S_frames.append(np.array(spins[0]))

    return T_frames, S_frames


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # ── Resolve run directory ─────────────────────────────────────────────────
    if args.run_dir:
        run_dir = args.run_dir
    else:
        run_dir = _find_latest_run(args.experiments_dir)
    run_dir = os.path.abspath(run_dir)

    if not os.path.isdir(run_dir):
        sys.exit(f"Run directory not found: {run_dir!r}")

    print(f"Run directory : {run_dir}")

    # ── Discover experiments ──────────────────────────────────────────────────
    exp_dirs = _find_experiment_dirs(run_dir)
    if not exp_dirs:
        sys.exit(f"No experiment subdirectories with best_final.npz found in {run_dir!r}")

    print(f"Experiments   : {[os.path.basename(d) for d in exp_dirs]}")

    # ── Import after path resolution so the package is always findable ────────
    from evolving_ising.experiment import ExperimentResult
    from evolving_ising.objectives import EXPERIMENTS
    from evolving_ising.viz import generate_report

    # ── Load results ──────────────────────────────────────────────────────────
    results: List[ExperimentResult] = []
    for exp_dir in exp_dirs:
        name = os.path.basename(exp_dir)
        if name not in EXPERIMENTS:
            print(f"  [skip] {name!r} — not in EXPERIMENTS registry")
            continue
        print(f"  Loading {name} ...", end=" ", flush=True)
        results.append(_load_result(exp_dir, name))
        print("ok")

    if not results:
        sys.exit("No valid experiment results found.")

    # Infer H, W, hot_t, cold_t from the first result
    first_npz = np.load(
        os.path.join(run_dir, results[0].name, "best_final.npz"), allow_pickle=False
    )
    H, W, _, _ = _infer_grid(first_npz["J_best"])
    hot_t, cold_t = _infer_hot_cold(first_npz["T_final"], W)

    # ── Optional rollout for animations ──────────────────────────────────────
    rollout_data: Dict[str, tuple] = {}
    if not args.no_animation:
        print(f"\nRunning {args.rollout_steps}-step rollouts for animations ...")
        # Trigger JAX initialisation once before the loop so timing is cleaner
        import jax; jax.devices()
        for r in results:
            npz_path = os.path.join(run_dir, r.name, "best_final.npz")
            print(f"  Rollout: {r.name} ...", end=" ", flush=True)
            try:
                T_frames, S_frames = _run_rollout(npz_path, args.rollout_steps)
                rollout_data[r.name] = (T_frames, S_frames)
                print(f"ok ({len(T_frames)} frames)")
            except Exception as exc:
                print(f"FAILED ({exc})")

    # ── Build run_config for the report header ────────────────────────────────
    run_config = {
        "run_dir" : os.path.basename(run_dir),
        "grid"    : f"{H}×{W}",
        "hot_t"   : hot_t,
        "cold_t"  : cold_t,
        "cma_iters": len(results[0].best_hist) if results else "?",
        "experiments": len(results),
        "rollout_steps": args.rollout_steps if not args.no_animation else "—",
    }

    # ── Generate report ───────────────────────────────────────────────────────
    print("\nGenerating report ...")
    generate_report(
        results      = results,
        run_dir      = run_dir,
        H            = H,
        W            = W,
        hot_t        = hot_t,
        cold_t       = cold_t,
        run_config   = run_config,
        rollout_data = rollout_data,
    )
    print("Done.")


if __name__ == "__main__":
    main()
