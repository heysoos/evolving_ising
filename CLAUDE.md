# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project simulates and evolves Thermally Diffusing Ising models using genetic algorithms (CMA-ES). The goal is to optimize spin connectivity weights `J_nk` to maximize heat transport across a temperature gradient on a 2D grid.

## Shell Environment

**Claude Code runs from a Windows Git Bash shell, NOT inside WSL.** All Python work must be done inside WSL using the virtualenv at `/home/heysoos/.virtualenvs/Evolving_Ising/`.

**Critical: Git Bash auto-converts POSIX paths to Windows paths.** To prevent this, always run Python/pip commands via `wsl bash -c "..."` and use the `source ... && python` pattern (not bare WSL paths as the first token). Never invoke WSL paths directly as the executable — wrap them in `source ... &&` or use single-quoted paths inside the wsl command string.

To run any Python commands:
```bash
wsl bash -c "source /home/heysoos/.virtualenvs/Evolving_Ising/bin/activate && python <script.py>"
```

To install packages:
```bash
wsl bash -c "source /home/heysoos/.virtualenvs/Evolving_Ising/bin/activate && pip install <package>"
```

The project files are in `/mnt/c/Users/Heysoos/Documents/Pycharm Projects/Evolving_Ising` inside WSL. Always `cd` to that path inside the wsl command:
```bash
wsl bash -c "source /home/heysoos/.virtualenvs/Evolving_Ising/bin/activate && cd '/mnt/c/Users/Heysoos/Documents/Pycharm Projects/Evolving_Ising' && python run_experiments.py"
```

## Running Experiments

There is no build system or test runner. The main entry point is `run_experiments.py`:

```bash
wsl bash -c "source /home/heysoos/.virtualenvs/Evolving_Ising/bin/activate && cd '/mnt/c/Users/Heysoos/Documents/Pycharm Projects/Evolving_Ising' && python run_experiments.py"
```

This runs all objectives from `EXPERIMENTS`, saves per-experiment checkpoints/logs under `experiments/run_<timestamp>/`, and generates an HTML report.

For interactive exploration, use the Jupyter notebook:
```bash
jupyter notebook evolving_ising.ipynb
```

## Key Dependencies

- `jax` / `jax.numpy` — all numerical computation (GPU-accelerated)
- `evosax` — optional CMA-ES backend (unused; built-in `SeparableCMAES` is the default)
- `matplotlib` — visualization and animation
- `tqdm` — progress bars in `run_experiments.py`

## Package Architecture (`evolving_ising/`)

The codebase is a proper Python package split into focused submodules:

```
evolving_ising/
  __init__.py    re-exports IsingModel, TemperatureDiffuser, CMAState, SeparableCMAES, EvoConfig, EvoRunner
  model.py       IsingModel
  diffusion.py   TemperatureDiffuser
  optim.py       CMAState, SeparableCMAES
  runner.py      EvoConfig, EvoRunner
  objectives.py  PhysicsSetup, EXPERIMENTS, make_eval_fn, vec_to_Jnk
  experiment.py  ExperimentResult, Checkpointer, run_experiment
  viz.py         plot_*, save_fitness_plot, generate_report
```

### `IsingModel` (`model.py`)
2D grid of spins `s ∈ {-1, +1}` with configurable `neighborhood` (`moore`/`von_neumann`) and `boundary` (`open`/`periodic`/`periodic_lr`/`periodic_tb`). Precomputes `neighbors (N,K)`, `mask (N,K)`, `rev_slot (N,K)`, and color masks for parallel Metropolis updates.

Key methods:
- `metropolis_checkerboard_sweeps(key, spins, J_nk, T, num_sweeps)` — parallel grouped (checkerboard/4-color) Metropolis-Hastings; returns `(spins, energies)`
- `metropolis_checkerboard_sweeps_with_history(...)` — same but also returns `(num_sweeps, B, N)` history via `lax.scan`
- `metropolis_sweeps` / `metropolis_sweeps_with_history` — backwards-compatible aliases
- `energy(J_nk, spins)` — total energy per batch; `local_energy` returns per-site energies
- `wrap_flags()` → `(wrap_h, wrap_v)` bools
- `vertical_edge_masks()` → `(has_up, has_down)` site masks
- All heavy methods are `@jax.jit` with `static_argnums=(0,)`

### `TemperatureDiffuser` (`diffusion.py`)
Stateless diffuser — graph structure (`neighbors`, `mask`) is passed at call time, not stored. Conductance is derived from `J_nk` via `conductance_mode` (`abs`/`relu`/`softplus`/`square`/`sigmoid`). Uses row normalization by default.

- `step(neighbors, J_nk, mask, T, pin_mask, pin_values)` — one diffusion step
- `diffuse(neighbors, J_nk, mask, T0, steps, pin_mask, pin_values)` — `steps` steps via `jax.lax.scan`

### `SeparableCMAES` (`optim.py`)
Built-in separable (diagonal covariance) CMA-ES optimizer. `CMAState` dataclass holds `(mean, sigma, diagC, pc, ps, rng)`. Call `ask()` → sample `(P, D)`, evaluate fitness, call `tell(X, fitness)`. Fitness is maximized.

### `EvoRunner` / `EvoConfig` (`runner.py`)
High-level wrapper combining `IsingModel` + `TemperatureDiffuser` + `SeparableCMAES`. `EvoConfig` is a dataclass with all hyperparameters. `EvoRunner.run()` returns `(J_best, best_fit)`. Fitness = negative mean Ising energy (minimises E).

### `PhysicsSetup` (`objectives.py`)
Dataclass bundling the shared physical configuration for an experiment: `ising`, `diffuser`, `T0`, `pin_mask`, `pin_values`, `top_idx`, `flat_idx`, plus simulation hyperparameters (`iters_eval`, `steps_per_iter`, `sweeps_per_iter`, `chains_per_eval`, `j_scale`). Properties `N`, `K`, `D` derived from fields.

### `EXPERIMENTS` / `make_eval_fn` (`objectives.py`)
`EXPERIMENTS` dict maps string keys to `{title, description, formula}` metadata. Current objectives: `max_top_temp`, `max_mean_temp`, `min_temp_variance`, `max_neg_energy`, `max_top_temp_low_energy`.

`make_eval_fn(objective, setup)` returns `(eval_single, eval_population)` — JIT-compiled functions. All objectives share the same `_dynamics` inner loop (diffusion + Metropolis via `lax.scan`).

`vec_to_Jnk(theta, setup)` maps `(D,) → (N, K)` via `softplus(theta) * j_scale`.

### `ExperimentResult` / `Checkpointer` / `run_experiment` (`experiment.py`)
`run_experiment(name, setup, run_dir, ...)` runs the full CMA-ES loop with tqdm, saves:
- `evolution_log.csv` — per-iteration stats
- `checkpoints/checkpoint_NNNN.npz` — periodic snapshots
- `fitness_stats.png` — fitness curve plot
- `best_final.npz` — final best theta, J_nk, T_final, S_final, all histories

`Checkpointer` is a context manager ensuring the CSV is always closed.

### `viz.py`
No JAX dependency — all inputs are plain numpy. Key functions:
- `plot_fitness_curve`, `save_fitness_plot` — fitness history plots
- `plot_temperature`, `plot_spins`, `plot_connectivity`, `plot_comparison` — return base64 PNG strings
- `generate_report(results, run_dir, H, W, hot_t, cold_t, run_config)` — writes self-contained `report.html`

## Data Flow

```
theta (D,) -> vec_to_Jnk -> J_nk (N,K)
    |
    +-> TemperatureDiffuser.diffuse(neighbors, J_nk, mask, T0, steps) -> T (B,N)
    +-> IsingModel.metropolis_checkerboard_sweeps(key, spins, J_nk, T) -> spins (B,N)
    |
    fitness <- objective(T, spins)   [e.g. mean top-row temperature]
        |
    SeparableCMAES.tell(X, fitness)
```

## Important Conventions

- **Spins**: `int8` arrays of shape `(B, N)` where `B` is batch/chain count and `N = H*W`
- **J_nk**: `float32` of shape `(N, K)`, invalid slots must be zero (enforced by `mask`)
- **Temperature**: broadcasts: scalar → `(B,N)`, `(N,)` → `(B,N)`, or explicit `(B,N)`
- **Pinned boundaries**: `pin_mask (N,)` bool + `pin_values (N,)` float; pinned sites are restored after each diffusion step
- **TemperatureDiffuser is stateless**: pass `neighbors` and `mask` from `IsingModel` at each call
- `evosax` is not used — the built-in `SeparableCMAES` is the optimizer throughout