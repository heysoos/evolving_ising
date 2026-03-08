# Agent Implementation Plan

## Evolving Ising Work Extraction

Read `evolving_ising_work_extraction.md` in full before writing any code. If anything is ambigious read
`evolving_ising_work_extraction_verbose.md`. All equations, experiment parameters, and expected behaviours referenced
below come
from that document.

---

## Existing Repository Notes

The repo already contains a working `evolving_ising/` package (model, diffusion, optim, runner, objectives,
experiment, viz). All new modules live **outside** this package at the top level (or in
`work_extraction/` if you prefer a subpackage). The new work uses a **uniform oscillating bath**
`T(t) = T_mean + ΔT·sin(2πt/τ)` passed as a scalar to `IsingModel`; the existing `TemperatureDiffuser`
from `diffusion.py` is **not** used in these experiments. Spin dynamics use
`IsingModel.metropolis_checkerboard_sweeps` (Metropolis-Hastings, not Glauber).

---

## Ground Rules

- Do not modify anything inside `evolving_ising/`.
- Every new module must have a corresponding test that can be run with `pytest`.
- After each phase, run the verification checks listed in the spec before
  proceeding. Do not move to the next phase if any check fails.
- Save all experiment results as `.npz` files under `results/`. Never hardcode
  paths — use a `config` dict passed through every function.
- Commit working code at the end of each phase with a descriptive message.

---

## Phase 1 — Thermodynamic Accounting

**File**: `thermodynamics.py`

Implement tracking of heat (eq. 6), work (eq. 5), and entropy production (eq. 7).

**Interface note**: `IsingModel.metropolis_checkerboard_sweeps(key, spins, J_nk, T, num_sweeps)`
returns `(spins, energies)` where `energies` is a scalar total energy — it does NOT return
individual flip events. Use energy differences between calls to track thermodynamic quantities:

- **Heat per step**: `δQ = E_after_sweeps − E_before_sweeps` (at fixed J). All energy change
  during Metropolis is heat. Split into `Q_in` (positive ΔE, heat absorbed) and
  `Q_out` (negative ΔE, heat rejected) by accumulating separately.
- **Entropy per step**: `σ = −δQ / T(t)` where T is the bath temperature at that step.
- **Work from remodelling**: Use `IsingModel.energy(J_nk, spins)` before and after each J
  update at fixed spins; `δW_remodel = E_after − E_before` (energy injected by the change).
  Also accumulate the basal cost `λ·|δJ|` per bond changed.
- **W_extracted**: equals `−(Q_in − Q_out)` sign-convention check — verify against ΔU.

Implement a `CycleAccumulator` that ingests per-step `(delta_E, T, remodel_cost)` tuples and
exposes `W_extracted`, `Q_in`, `Q_out`, `Sigma_cycle`, `W_remodel`, and `W_net` (eq. 10) at
the end of each cycle. All accumulation is plain numpy (convert JAX arrays with `float(x)` or
`np.asarray(x)` before accumulating).

**Verification** (must all pass before Phase 2):

- First law: `|ΔU - (Q_in - Q_out - W_net)| < 1e-9` per cycle.
- Second law: `Sigma_cycle >= 0` for every cycle over a 100-cycle fixed-J run.
- Carnot bound: `η <= 1 - T_cold/T_hot` always.

---

## Phase 2 — Controller

**File**: `controller.py`

Implement `LocalController`: a small MLP (input size 5, two hidden layers of 8
with tanh, scalar output scaled to `[-delta_J_max, delta_J_max]`). Implement in
plain numpy so that the flat parameter vector is directly compatible with
`SeparableCMAES` from `evolving_ising/optim.py`. Weights must be serialisable to
a flat numpy array and back (needed by the optimiser).

Implement `LocalMagnetisationTracker` as an exponential moving average over the
spin field with configurable decay `alpha`. This operates on numpy arrays extracted
from JAX (use `np.asarray(spins)`).

**Verification**:

- Round-trip test: `set_params(get_params())` leaves all weights identical.
- Output is always in `[-delta_J_max, delta_J_max]` for 10 000 random inputs.

---

## Phase 3 — Budget Classes

**File**: `budgets.py`

Implement the three budget classes with a shared `BaseBudget` interface exposing
`update(spins_before, spins_after, J_nk, T)`, `get_budget(i, j) -> float`, and
`spend(i, j, cost)`.

- Ordering events for `BondBudget` accumulation are inferred from spin-state changes
  between successive Metropolis calls: a bond `(i,j)` has an ordering event when
  `(s_i * s_j)` increases (becomes more aligned) between `spins_before` and
  `spins_after`. Use `IsingModel.neighbors` and `IsingModel.mask` to enumerate bonds.
  All inputs should be converted to numpy before processing.

- `BondBudget` — per-bond accumulation from ordering events (eq. 11).
- `NeighbourhoodBudget` — site budgets pooled over a neighbourhood (eq. 12).
- `DiffusingBudget` — reaction-diffusion field on the lattice (eq. 13).
- `NoBudget` — always returns `inf`; used for Experiment 0.

**Verification**:

- Budget non-negativity: `get_budget(i,j) >= 0` after 1 000 random `spend`
  calls for all three classes.
- `NoBudget` never blocks a remodel.
- `DiffusingBudget` with `D=0`, `tau_mu=inf` reduces to `BondBudget` behaviour
  (no diffusion, no decay).

---

## Phase 4 — Evolutionary Optimiser

**File**: `optimiser.py`

Use the existing `SeparableCMAES` from `evolving_ising/optim.py` as the optimizer.
Do **not** re-implement an ES from scratch. Provide a thin wrapper `WorkExtractionES`
that:
- Creates a `SeparableCMAES(dim=n_params, pop_size=config['pop_size'], sigma_init=config['sigma'])`
- Exposes `ask() -> list[params]` and `tell(params_list, fitnesses)` delegating to the
  underlying CMA-ES.
- Stores per-generation `(mean_fitness, best_fitness, sigma)` in a history list.

`evaluate_fitness(controller, budget, model, config) -> float` runs `n_eval_cycles` full
cycles using `IsingModel.metropolis_checkerboard_sweeps` and returns mean `W_net`. Bond
update proposals should be applied to a random fraction `bond_update_frac` of bonds per
step to keep runtime tractable. `model` here is an `IsingModel` instance from
`evolving_ising.model`.

**Verification**:

- On a convex toy function (e.g. negative L2 norm), the optimiser reaches within
  1% of the optimum within 200 generations.

---

## Phase 5 — Training Loop

**File**: `train.py`

Implement `run_experiment(config) -> ExperimentResult` that wires together the
`IsingModel`, controller, budget, and `WorkExtractionES`. Log per-generation metrics to
`results/<experiment_name>/training_log.npz`. Log the best controller params and
final `J` field at the end of training.

Temperature schedule must follow eq. (1). `J` values must be clamped to
`[J_min, J_max]` after every update (suggest `[0.01, 5.0]`). Construct `IsingModel`
with `boundary='periodic'` and `neighborhood='von_neumann'` unless the config overrides.

---

## Phase 6 — Experiments

Run experiments in order. For each, produce the outputs listed and confirm the
expected behaviours before proceeding to the next.

### Experiment 0 — Baseline (`experiments/exp0_baseline.py`)

Sweep `J0` over `[0.2, 2.0]` (10 values) and `tau` over `[10, 1000]`
(8 log-spaced values). For each combination run 50 cycles and record `W_net`,
`Sigma_cycle`, and `m(t)`.

**Expected**: `W_net` peaks near `J0 ≈ T_mean / 2.269`. Record the
`(J0_opt, tau_opt, W_net_opt)` triple — this is the baseline ceiling.

**Outputs**: `results/exp0/sweep.npz`, `figures/exp0_heatmap.png`.

---

### Experiment 1 — Bond Budget (`experiments/exp1_bond_budget.py`)

Use `J0 = T_mean / 2.269` as the initial uniform coupling. Sweep `lambda` over
`{0.0, 0.01, 0.1, 0.5}` and `alpha` over `{0.05, 0.1, 0.3}`. For each
combination run 500 generations of evolutionary optimisation.

**Expected**: `W_net` of best individual exceeds `W_net_opt` from Experiment 0
at low `lambda`. Increasing `lambda` degrades performance monotonically.

**Outputs**: `results/exp1/`, one subdirectory per `(lambda, alpha)` pair,
each containing `training_log.npz`, `best_controller.npz`, `final_J.npz`.

---

### Experiment 2 — Neighbourhood Budget (`experiments/exp2_nbhd_budget.py`)

Same setup as Experiment 1 but use `NeighbourhoodBudget`. Sweep `gamma` over
`{0.0, 0.1, 0.25, 0.5, 1.0}` at the best `(lambda, alpha)` found in
Experiment 1. Additionally, for the best `gamma`, sweep `tau` over
`{100, 200, 500}` to test whether optimal `gamma` depends on cycle period.

**Expected**: Non-monotonic `W_net` vs. `gamma` with a peak at some `gamma*`.

**Outputs**: `results/exp2/`, same structure as Experiment 1.

---

### Experiment 3 — Diffusing Budget (`experiments/exp3_diffuse.py`)

Use `DiffusingBudget`. Sweep `D` over `{0.01, 0.1, 0.5, 2.0}` at fixed
`tau_mu=20`, and `tau_mu` over `{5, 20, 100}` at fixed `D=0.1`. For each run,
compute the correlation length `xi` from the connected spin-spin correlation
function and compute `Lambda = D * tau_mu / xi**2` (eq. 14).

Additionally run a `T_mean` sweep over `{2.0, 2.5, 3.0}` (which varies `xi`)
to test the `Lambda ~ 1` prediction directly.

**Expected**: `W_net` peaks near `Lambda = 1`.

**Outputs**: `results/exp3/`, same structure. Include `xi` and `Lambda` in each
`training_log.npz`.

---

## Phase 7 — Analysis

**File**: `analysis.py`

Implement the following functions. Each saves a figure to `figures/`.

| Function                              | Description                                          | Key assertion                           |
|---------------------------------------|------------------------------------------------------|-----------------------------------------|
| `plot_learning_curves(exp)`           | `W_net` vs. generation for all runs in an experiment | Best individual exceeds fixed-J ceiling |
| `plot_J_phase_portrait(result)`       | `J_bar(t)` vs. `T(t)` over one cycle                 | Visible phase lag                       |
| `compute_phase_lag(result) -> float`  | Cross-correlation of `T(t)` and `J_bar(t)`           | Converges toward `pi/2` during training |
| `plot_J_spatial_map(result)`          | Heatmap of time-averaged `J_ij`                      | Heterogeneity index > baseline          |
| `plot_budget_vs_domain_walls(result)` | Mean budget at wall vs. interior sites               | Eq. (16) holds                          |
| `plot_entropy_production_map(result)` | Per-site `sigma_i` averaged over cycles              | Colocalised with domain walls           |
| `plot_lambda_sweep(exp3_results)`     | `W_net` vs. `Lambda`                                 | Peak near `Lambda = 1`                  |
| `plot_efficiency_vs_sigma(result)`    | `eta` vs. `Sigma_cycle` across generations           | Linear tradeoff per eq. (8)             |

Run `analysis.py` as a script to regenerate all figures from saved results.

---

## Suggested Default Config

```python
DEFAULT_CONFIG = {
    'L': 32,
    'T_mean': 2.5,
    'delta_T': 1.5,
    'tau': 200,
    'J_init': 1.1,
    'J_min': 0.01,
    'J_max': 5.0,
    'n_generations': 500,
    'n_eval_cycles': 10,
    'steps_per_cycle': 200,
    'bond_update_frac': 0.1,
    'delta_J_max': 0.1,
    'B_scale': 2.0,
    'lambda': 0.05,
    'pop_size': 20,
    'sigma': 0.02,
    'elite_frac': 0.2,
    'sigma_decay': 0.995,
    'hidden_size': 8,
    'mag_ema_alpha': 0.05,
    'log_interval': 10,
    # IsingModel construction defaults
    'neighborhood': 'von_neumann',
    'boundary': 'periodic',
    'num_sweeps': 1,
}
```

Override only the keys that differ per experiment.

---

## Definition of Done

The project is complete when:

1. All five verification checks pass on a fresh run.
2. Experiment 0 reproduces the expected `J0_opt ≈ T_mean / 2.269` peak.
3. At least one adaptive experiment exceeds the fixed-J ceiling in `W_net`.
4. The phase lag plot shows convergence toward `pi/2`.
5. The `Lambda` sweep (Experiment 3) shows a peak in `W_net` near `Lambda = 1`.
6. All figures are saved and `analysis.py` runs without error from saved results.