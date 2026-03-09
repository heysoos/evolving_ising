# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project simulates and evolves Thermally Diffusing Ising models using genetic algorithms (CMA-ES). The original goal was to optimize spin connectivity weights `J_nk` to maximize heat transport; the extended goal (work extraction) is to evolve a local controller that adapts `J` in response to spin state to extract thermodynamic work from an oscillating bath.

## Shell Environment

**Claude Code runs from a Windows Git Bash shell, NOT inside WSL.** All Python work must be done inside WSL using the virtualenv at `/home/heysoos/.virtualenvs/Evolving_Ising/`.

**Critical: Git Bash auto-converts POSIX paths to Windows paths.** Always run Python/pip commands via `wsl bash -c "..."` using the `source ... && python` pattern. Never invoke WSL paths directly as the executable.

```bash
wsl bash -c "source /home/heysoos/.virtualenvs/Evolving_Ising/bin/activate && cd '/mnt/c/Users/Heysoos/Documents/Pycharm Projects/Evolving_Ising' && python <script.py>"
```

To install packages:
```bash
wsl bash -c "source /home/heysoos/.virtualenvs/Evolving_Ising/bin/activate && pip install <package>"
```

## Running Experiments

No build system. Entry points:

```bash
# Original heat-transport evolution
python run_experiments.py

# Work extraction experiments
python experiments/exp0_baseline.py
python experiments/exp1_bond_budget.py
python experiments/exp2_nbhd_budget.py
python experiments/exp3_diffuse.py

# Tests
pytest tests/
```

(Wrap all commands in `wsl bash -c "source ... && cd '...' && ..."` per above.)

`run_experiments.py` saves to `experiments/run_<timestamp>/` and generates `report.html`. Work extraction experiments save `.npz` results to `results/<experiment_name>/` and figures to `figures/`.

## Key Dependencies

- `jax` / `jax.numpy` — all numerical computation (GPU-accelerated)
- `matplotlib` — visualization
- `tqdm` — progress bars
- `pytest` — tests for `work_extraction/` modules

## JAX / GPU Conventions

**Always use JAX for performance-critical code:**
- Use `jnp` arrays, `jax.lax.scan` for loops, `@jax.jit` for compiled functions
- Mark static (non-array) arguments with `static_argnums` in `jax.jit`
- Use `jax.vmap` to batch over populations/chains instead of Python loops
- Convert to numpy only at boundaries (logging, plain-numpy modules): `np.asarray(x)` or `float(x)`
- The `work_extraction/` modules (controller, budgets, thermodynamics) use **plain numpy** — they ingest JAX outputs after conversion

## Package Architecture

### `evolving_ising/` — Core physics (do not modify)
```
model.py       IsingModel — 2D Ising grid, Metropolis-Hastings sweeps
diffusion.py   TemperatureDiffuser — stateless heat diffusion (not used in work extraction)
optim.py       CMAState, SeparableCMAES — diagonal CMA-ES; fitness is maximized
runner.py      EvoConfig, EvoRunner — high-level evolution wrapper
objectives.py  PhysicsSetup, EXPERIMENTS, make_eval_fn, vec_to_Jnk
experiment.py  ExperimentResult, Checkpointer, run_experiment
viz.py         plot_*, generate_report — pure numpy/matplotlib
```

### `work_extraction/` — Thermodynamic work extraction
```
thermodynamics.py  CycleAccumulator — heat/work/entropy accounting (plain numpy)
controller.py      LocalController (MLP, plain numpy), LocalMagnetisationTracker (EMA)
budgets.py         NoBudget, BondBudget, NeighbourhoodBudget, DiffusingBudget
optimiser.py       WorkExtractionES — thin wrapper around SeparableCMAES
train.py           run_experiment(config) -> ExperimentResult
analysis.py        plotting/analysis functions; runs as script to regenerate all figures
```

### `experiments/` — Experiment scripts
```
exp0_baseline.py    Sweep J0/tau at fixed J; establish W_net ceiling
exp1_bond_budget.py Evolve controller with BondBudget; sweep lambda, alpha
exp2_nbhd_budget.py NeighbourhoodBudget; sweep gamma and tau
exp3_diffuse.py     DiffusingBudget; sweep D, tau_mu; compute Lambda = D*tau_mu/xi^2
```

### `tests/` — pytest tests
```
test_thermodynamics.py  First/second law, Carnot bound
test_controller.py      Round-trip params, output bounds
test_budgets.py         Non-negativity, NoBudget, DiffusingBudget limiting case
test_optimiser.py       Convergence on convex toy function
```

## Key Classes

### `IsingModel` (`evolving_ising/model.py`)
2D grid, spins `int8 (B,N)`, `N=H*W`. All heavy methods `@jax.jit`:
- `metropolis_checkerboard_sweeps(key, spins, J_nk, T, num_sweeps)` → `(spins, energies)`
- `metropolis_checkerboard_sweeps_with_history(...)` → also returns `(sweeps,B,N)` history
- `energy(J_nk, spins)` → total energy per batch; `local_energy` → per-site
- `neighbors (N,K)`, `mask (N,K)` — precomputed graph structure

### `SeparableCMAES` (`evolving_ising/optim.py`)
`ask()` → `(P, D)` samples; `tell(X, fitness)` updates. Fitness is **maximized**.

### Work Extraction Data Flow
```
T(t) = T_mean + ΔT·sin(2πt/τ)   # uniform oscillating bath (scalar)
    |
IsingModel.metropolis_checkerboard_sweeps → spins (B,N)
    |
CycleAccumulator ← (delta_E, T)   # heat/work/entropy (plain numpy)
LocalMagnetisationTracker → m_ema  # EMA of spins (plain numpy)
    |
LocalController(state) → delta_J   # MLP proposal (plain numpy)
budget.spend(i,j,cost)             # gate updates
    |
J_nk updated → WorkExtractionES.tell(params, W_net)
```

## Important Conventions

- **Spins**: `int8 (B, N)`, `B` = batch/chain count, `N = H*W`
- **J_nk**: `float32 (N, K)`, invalid slots zeroed by `mask`; clamped to `[0.01, 5.0]` in work extraction
- **Temperature**: scalar or `(B,N)` — broadcasts automatically in Metropolis
- **Do not modify `evolving_ising/`** — all new code in `work_extraction/` or `experiments/`
- `evosax` is not used — `SeparableCMAES` from `optim.py` is the optimizer throughout
- Save all results as `.npz` under `results/`; never hardcode paths — use a `config` dict