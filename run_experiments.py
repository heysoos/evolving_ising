"""
run_experiments.py — entry point for loss-function comparison experiments.

Runs CMA-ES with each objective defined in evolving_ising.objectives.EXPERIMENTS,
saves per-experiment checkpoints/logs, and generates an HTML report.
"""
from __future__ import annotations

import datetime
import os

import jax.numpy as jnp

from evolving_ising import IsingModel, TemperatureDiffuser
from evolving_ising.experiment import run_experiment
from evolving_ising.objectives import EXPERIMENTS, PhysicsSetup
from evolving_ising.viz import generate_report

# ── Experiment parameters ────────────────────────────────────────────
H, W            = 64, 64
HOT_T           = 3.0
COLD_T          = 0.0
ALPHA           = 0.35
ITERS_EVAL      = 80
STEPS_PER_ITER  = 2
SWEEPS_PER_ITER = 2
CHAINS_PER_EVAL = 8
POP_SIZE        = 32
CMA_ITERS       = 1000
J_SCALE         = 1.0
SIGMA_INIT      = 0.5
SEED            = 0
CHECKPOINT_EVERY = 100


def build_setup() -> PhysicsSetup:
    """Construct the shared physical setup from the constants above."""
    N = H * W
    ising = IsingModel((H, W), neighborhood="von_neumann", boundary="periodic_lr")

    cols    = jnp.arange(W, dtype=jnp.int32)
    top_idx = cols                          # row 0 — cold measurement row
    bot_idx = (H - 1) * W + cols           # row H-1 — hot boundary

    pin_mask   = jnp.zeros(N, dtype=bool).at[bot_idx].set(True)
    pin_values = jnp.zeros(N, dtype=jnp.float32).at[bot_idx].set(HOT_T)

    T0 = jnp.repeat(jnp.linspace(COLD_T, HOT_T, H, dtype=jnp.float32), W)

    diffuser = TemperatureDiffuser(alpha=ALPHA, conductance_mode="abs", normalize_mode="row")
    flat_idx = jnp.where(ising.mask.reshape(-1))[0]

    return PhysicsSetup(
        ising=ising,
        diffuser=diffuser,
        T0=T0,
        pin_mask=pin_mask,
        pin_values=pin_values,
        top_idx=top_idx,
        flat_idx=flat_idx,
        iters_eval=ITERS_EVAL,
        steps_per_iter=STEPS_PER_ITER,
        sweeps_per_iter=SWEEPS_PER_ITER,
        chains_per_eval=CHAINS_PER_EVAL,
        j_scale=J_SCALE,
    )


if __name__ == "__main__":
    run_id  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("experiments", f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")

    setup = build_setup()

    run_config = {
        "grid": f"{H}×{W}",
        "neighborhood": "von_neumann",
        "boundary": "periodic_lr",
        "hot_t": HOT_T,
        "pop_size": POP_SIZE,
        "cma_iters": CMA_ITERS,
        "iters_eval": ITERS_EVAL,
        "chains": CHAINS_PER_EVAL,
        "alpha": ALPHA,
    }

    results = [
        run_experiment(
            name, setup, run_dir,
            cma_iters=CMA_ITERS,
            pop_size=POP_SIZE,
            sigma_init=SIGMA_INIT,
            seed=SEED,
            checkpoint_every=CHECKPOINT_EVERY,
        )
        for name in EXPERIMENTS
    ]

    generate_report(results, run_dir, H, W, HOT_T, COLD_T, run_config)
    print("\nAll experiments complete!")