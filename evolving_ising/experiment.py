"""
evolving_ising.experiment
=========================
Orchestrates a single CMA-ES experiment: runs the optimisation loop,
saves checkpoints and logs, performs a final visualisation rollout, and
returns all results in an `ExperimentResult` dataclass.

Responsibilities
----------------
- `ExperimentResult`: dataclass carrying everything needed for analysis
  and report generation after an experiment finishes.
- `Checkpointer`: context manager that owns the CSV log file and periodic
  .npz checkpoint saves.  Using `__enter__`/`__exit__` guarantees the log
  file is closed even if an exception interrupts the training loop.
- `_visualization_rollout`: single-chain rollout (using `jax.lax.scan`)
  to produce the final temperature and spin snapshots for plotting.
- `run_experiment`: the main function — sets up the optimizer and
  Checkpointer, runs the CMA-ES loop with a tqdm progress bar, then
  calls the rollout and saves all outputs.

Output files per experiment (inside run_dir/<name>/)
-----------------------------------------------------
  evolution_log.csv          — per-iteration stats (best, mean, std, best_so_far)
  fitness_stats.png          — fitness curve plot saved to disk
  checkpoints/
    checkpoint_NNNN.npz      — periodic snapshots of the best theta/J found so far
  best_final.npz             — final best theta, J_nk, T_final, S_final, all histories
"""
from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from .objectives import EXPERIMENTS, PhysicsSetup, make_eval_fn, vec_to_Jnk
from .optim import SeparableCMAES
from .viz import save_fitness_plot

Array = jnp.ndarray


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ExperimentResult:
    """
    All outputs from a completed experiment, ready for analysis or reporting.

    Fields
    ------
    name : str
        Experiment key (matches a key in EXPERIMENTS).
    best_hist : List[float]
        Best fitness value found in each generation's population.
        This is *noisy* — a lucky individual can temporarily score high
        without reflecting the distribution's true progress.
    best_so_far : List[float]
        Running global maximum of best_hist — monotonically non-decreasing.
        This is the canonical learning curve to plot.
    mean_hist : List[float]
        Population mean fitness per generation.
    std_hist : List[float]
        Population fitness standard deviation per generation.
    best_fitness : float
        Final best_so_far value — the overall best fitness achieved.
    top_row_temp : float
        Mean temperature on the top row in the visualisation rollout.
        Primary metric for heat-transport objectives.
    mean_temp : float
        Mean temperature across the whole grid in the rollout.
    T_final : np.ndarray (N,) float32
        Final temperature field from the visualisation rollout.
    S_final : np.ndarray (N,) int8
        Final spin configuration from the visualisation rollout.
    J_best : np.ndarray (N, K) float32
        Best coupling matrix found during optimisation.
    neighbors : np.ndarray (N, K) int32
        Neighbour index table (from IsingModel) — stored here so
        connectivity plots can be generated without access to the model.
    mask : np.ndarray (N, K) bool
        Validity mask (from IsingModel) — same reason as neighbors.
    elapsed : float
        Wall-clock seconds for the full experiment (optimisation + rollout).
    """
    name: str
    best_hist:    List[float]
    best_so_far:  List[float]
    mean_hist:    List[float]
    std_hist:     List[float]
    best_fitness: float
    top_row_temp: float
    mean_temp:    float
    T_final:   np.ndarray   # (N,) float32
    S_final:   np.ndarray   # (N,) int8
    J_best:    np.ndarray   # (N, K) float32
    neighbors: np.ndarray   # (N, K) int32
    mask:      np.ndarray   # (N, K) bool
    elapsed:   float


# ── Checkpointer ──────────────────────────────────────────────────────────────

class Checkpointer:
    """
    Context manager that handles incremental saving during the CMA-ES loop.

    Two outputs are managed:
    1. A CSV file (`evolution_log.csv`) written row-by-row each iteration,
       recording per-generation statistics.
    2. Periodic compressed .npz checkpoints of the best theta/J found so far,
       saved every `checkpoint_every` iterations and always at the final iter.

    Using `with Checkpointer(...) as ckpt:` ensures the CSV file is properly
    closed even if the training loop raises an exception.

    Parameters
    ----------
    exp_dir          : directory for this experiment's outputs
    checkpoint_every : save a .npz every this many iterations
    vec_to_Jnk_fn   : callable (theta,) -> J_nk (N, K); used to materialise
                       J_nk from the best theta at checkpoint time
    """

    def __init__(self, exp_dir: str, checkpoint_every: int, vec_to_Jnk_fn):
        self.ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self._log_path = os.path.join(exp_dir, "evolution_log.csv")
        self.checkpoint_every = checkpoint_every
        self._vec_to_Jnk = vec_to_Jnk_fn
        self._log_file = None
        self._writer = None

    def __enter__(self) -> Checkpointer:
        self._log_file = open(self._log_path, "w", newline="")
        self._writer = csv.writer(self._log_file)
        # Header row
        self._writer.writerow(["iter", "best_iter", "mean_iter", "std_iter", "best_so_far"])
        return self

    def __exit__(self, *_) -> None:
        """Always close the log file, even on exception."""
        if self._log_file:
            self._log_file.close()

    def log(self, t: int, val: float, fmean: float, fstd: float, best_fit: float) -> None:
        """Append one row to the CSV (called every iteration)."""
        self._writer.writerow([t + 1, val, fmean, fstd, best_fit])

    def maybe_checkpoint(
        self,
        t:           int,
        cma_iters:   int,
        best_theta:  Array,
        best_fit:    float,
        best_hist:   List[float],
        best_so_far: List[float],
        mean_hist:   List[float],
        std_hist:    List[float],
    ) -> None:
        """
        Save a checkpoint if this is a checkpoint iteration.

        Checkpoints at: t+1 divisible by checkpoint_every, AND at the final
        iteration (t == cma_iters - 1), so there is always a checkpoint
        capturing the very end of optimisation.

        The .npz contains both the raw theta vector and the materialised J_nk,
        plus all fitness history arrays so the checkpoint is self-contained
        for analysis.
        """
        if (t + 1) % self.checkpoint_every == 0 or t == cma_iters - 1:
            path = os.path.join(self.ckpt_dir, f"checkpoint_{t+1:04d}.npz")
            np.savez_compressed(
                path,
                best_theta  = np.array(best_theta),
                J_best      = np.array(self._vec_to_Jnk(best_theta)),
                best_fitness= float(best_fit),
                best_hist   = np.array(best_hist),
                best_so_far = np.array(best_so_far),
                mean_hist   = np.array(mean_hist),
                std_hist    = np.array(std_hist),
                iter        = t + 1,
            )


# ── Visualization rollout ─────────────────────────────────────────────────────

def _visualization_rollout(setup: PhysicsSetup, J_best: Array, key: Array):
    """
    Run a single-chain visualisation rollout using the best J found.

    Unlike the multi-chain evaluation used during training, this uses
    B=1 chain and runs for `setup.iters_eval` steps via `jax.lax.scan`,
    producing a single snapshot of the final temperature and spin state.

    Using `lax.scan` (rather than a Python for-loop) compiles the entire
    rollout into one XLA computation, which is far faster than dispatching
    80 separate JIT calls.

    Parameters
    ----------
    setup  : PhysicsSetup  containing the model, diffuser, and initial conditions
    J_best : (N, K) float32  best coupling matrix from optimisation
    key    : PRNGKey

    Returns
    -------
    spins : (N,) int8     final spin configuration
    T     : (N,) float32  final temperature field
    """
    key_s, key_loop = jax.random.split(key)
    spins = setup.ising.init_spins(key_s, 1)              # (1, N)
    T = jnp.broadcast_to(setup.T0[None], (1, setup.N))   # (1, N)

    def body(carry, _):
        spins_c, T_c, key_c = carry
        T_c = setup.diffuser.diffuse(
            setup.ising.neighbors, J_best, setup.ising.mask, T_c,
            steps=setup.steps_per_iter,
            pin_mask=setup.pin_mask,
            pin_values=setup.pin_values,
        )
        key_c, sub = jax.random.split(key_c)
        spins_c, _ = setup.ising.metropolis_checkerboard_sweeps(
            sub, spins_c, J_best, T_c, num_sweeps=setup.sweeps_per_iter,
        )
        return (spins_c, T_c, key_c), None

    (spins_f, T_f, _), _ = jax.lax.scan(body, (spins, T, key_loop), None, length=setup.iters_eval)
    return spins_f[0], T_f[0]   # strip the batch dimension (B=1)


# ── Main experiment runner ────────────────────────────────────────────────────

def run_experiment(
    name:             str,
    setup:            PhysicsSetup,
    run_dir:          str,
    cma_iters:        int   = 200,
    pop_size:         int   = 32,
    sigma_init:       float = 0.5,
    seed:             int   = 0,
    checkpoint_every: int   = 25,
) -> ExperimentResult:
    """
    Run one complete CMA-ES experiment and save all outputs.

    Phases
    ------
    1. Setup: create output directory, compile eval functions, initialise ES.
    2. Optimisation loop (inside Checkpointer context):
       - ask() → evaluate population → tell()
       - Track best_hist, best_so_far, mean_hist, std_hist
       - Log to CSV each iteration; checkpoint .npz every N iterations
       - Update tqdm progress bar with current best and mean fitness
    3. Post-processing:
       - Save fitness stats plot to disk
       - Run visualisation rollout (lax.scan, single chain)
       - Save best_final.npz
    4. Return ExperimentResult for report generation

    Parameters
    ----------
    name             : experiment key (must be in EXPERIMENTS)
    setup            : shared physical configuration (PhysicsSetup)
    run_dir          : root output directory for this run (timestamped)
    cma_iters        : number of CMA-ES generations
    pop_size         : population size λ
    sigma_init       : initial step-size σ₀
    seed             : PRNG seed for ES + Metropolis chains
    checkpoint_every : save a .npz checkpoint every this many iterations

    Returns
    -------
    ExperimentResult
    """
    print(f"\n{'='*60}\n  Experiment: {name}\n  {EXPERIMENTS[name]['title']}\n{'='*60}")
    t0 = time.time()

    exp_dir = os.path.join(run_dir, name)
    os.makedirs(exp_dir, exist_ok=True)

    # Compile evaluation functions for this objective (once, before the loop)
    _, eval_population = make_eval_fn(name, setup)

    key = jax.random.PRNGKey(seed)
    es  = SeparableCMAES(dim=setup.D, pop_size=pop_size, sigma_init=sigma_init, seed=seed)

    best_hist, best_so_far, mean_hist, std_hist = [], [], [], []
    best_theta = None
    best_fit   = -jnp.inf

    # Bind setup into vec_to_Jnk for use in Checkpointer (avoids passing setup everywhere)
    _jnk_fn = lambda theta: vec_to_Jnk(theta, setup)

    with Checkpointer(exp_dir, checkpoint_every, _jnk_fn) as ckpt:
        pbar = tqdm(range(cma_iters), desc=name, unit="iter", ncols=80)
        for t in pbar:
            # ── CMA-ES step ──────────────────────────────────────────────
            X = es.ask()                           # sample population (pop_size, D)
            key, sub = jax.random.split(key)
            fitness = eval_population(sub, X)      # evaluate all candidates
            es.tell(X, fitness)                    # update distribution

            # ── Statistics ───────────────────────────────────────────────
            idx   = int(jnp.argmax(fitness))
            val   = float(fitness[idx])            # best in this generation
            fmean = float(jnp.mean(fitness))
            fstd  = float(jnp.std(fitness))

            # Update global best (elitist: keep best ever seen)
            if best_theta is None or val > float(best_fit):
                best_fit   = fitness[idx]
                best_theta = X[idx]

            best_hist.append(val)
            best_so_far.append(float(best_fit))   # monotone global best
            mean_hist.append(fmean)
            std_hist.append(fstd)

            # ── Logging and checkpointing ─────────────────────────────────
            ckpt.log(t, val, fmean, fstd, float(best_fit))
            ckpt.maybe_checkpoint(t, cma_iters, best_theta, float(best_fit),
                                  best_hist, best_so_far, mean_hist, std_hist)
            pbar.set_postfix(best=f"{float(best_fit):.5f}", mean=f"{fmean:.5f}")

    # ── Post-processing ───────────────────────────────────────────────────────

    # Detailed fitness plot (best_so_far + mean±std) saved as PNG
    save_fitness_plot(
        best_hist, best_so_far, mean_hist, std_hist,
        title=f"Optimization Progress — {EXPERIMENTS[name]['title']}",
        path=os.path.join(exp_dir, "fitness_stats.png"),
    )

    # Visualisation rollout using lax.scan (one compiled XLA call)
    J_best = vec_to_Jnk(best_theta, setup)
    S_final_jax, T_final_jax = _visualization_rollout(setup, J_best, key)
    T_final = np.array(T_final_jax)   # move to CPU numpy for storage/plotting
    S_final = np.array(S_final_jax)

    top_idx_np   = np.array(setup.top_idx)
    top_row_temp = float(np.mean(T_final[top_idx_np]))
    mean_temp    = float(np.mean(T_final))

    # Comprehensive final save: everything needed to reproduce or analyse the result
    np.savez_compressed(
        os.path.join(exp_dir, "best_final.npz"),
        best_theta  = np.array(best_theta),
        J_best      = np.array(J_best),
        T_final     = T_final,
        S_final     = S_final,
        best_fitness= float(best_fit),
        top_row_temp= top_row_temp,
        mean_temp   = mean_temp,
        best_hist   = np.array(best_hist),
        best_so_far = np.array(best_so_far),
        mean_hist   = np.array(mean_hist),
        std_hist    = np.array(std_hist),
    )

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s | top-row T: {top_row_temp:.4f} | mean T: {mean_temp:.4f}")

    return ExperimentResult(
        name        = name,
        best_hist   = best_hist,
        best_so_far = best_so_far,
        mean_hist   = mean_hist,
        std_hist    = std_hist,
        best_fitness= float(best_fit),
        top_row_temp= top_row_temp,
        mean_temp   = mean_temp,
        T_final     = T_final,
        S_final     = S_final,
        J_best      = np.array(J_best),
        neighbors   = np.array(setup.ising.neighbors),
        mask        = np.array(setup.ising.mask),
        elapsed     = elapsed,
    )