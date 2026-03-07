# Experiment Plan: Loss Function Comparison

## Goal
Run CMA-ES optimization with 5 different loss functions on the same physical setup (hot bottom row, cold top), then generate a self-contained HTML report with visualizations.

## Files to Create

```
run_experiments.py          # Main experiment script
experiments/report.html     # Generated HTML report (self-contained, base64-embedded images)
```

## Loss Functions

All share the same physical setup (64x64 grid, von Neumann, periodic_lr, bottom pinned at T=3.0):

1. **max_top_temp** — `mean(T_top_row)` — current baseline, heat channeling upward
2. **max_mean_temp** — `mean(T_all)` — maximize overall grid temperature
3. **min_temp_variance** — `-std(T_all)` — drive temperature toward uniformity
4. **max_neg_energy** — `-mean(E) / N` — minimize Ising energy (favor aligned spins)
5. **max_top_temp_low_energy** — `mean(T_top) + 0.1 * (-mean(E) / N)` — multi-objective: heat transport + spin alignment

## Scale (to keep runtime tractable)

| Parameter | Value |
|---|---|
| Grid | 64x64 |
| Neighborhood | von_neumann |
| Boundary | periodic_lr |
| pop_size | 32 |
| CMA-ES iters | 200 |
| sigma_init | 0.5 |
| chains_per_eval | 8 |
| iters_eval | 80 |
| diffusion steps/sweep | 2 |
| MCMC sweeps/iter | 2 |
| alpha | 0.35 |

## Script Architecture (`run_experiments.py`)

### 1. Shared Setup
- Create `IsingModel`, pin masks, initial temperature gradient, `TemperatureDiffuser`
- Compute `flat_idx`, `D` (parameter dimension)
- Define `vec_to_Jnk()` and `_theta_to_Jpos()` (shared across experiments)

### 2. Loss Function Definitions
Each loss function is a **separate `@jax.jit`-compiled function** that takes `(key, theta)` and returns a scalar fitness. They share a common inner loop (diffuse + MCMC sweep) but differ in the final reduction:

```python
def make_eval_fn(objective="max_top_temp"):
    @jax.jit
    def eval_single(key, theta):
        # ... shared dynamics loop ...
        # Switch on objective for final scalar:
        if objective == "max_top_temp":
            return jnp.mean(T_f[:, top_idx])
        elif objective == "max_mean_temp":
            return jnp.mean(T_f)
        elif objective == "min_temp_variance":
            return -jnp.std(T_f)
        elif objective == "max_neg_energy":
            E = ising.energy(J_nk, spins_f)
            return -jnp.mean(E) / N
        elif objective == "max_top_temp_low_energy":
            E = ising.energy(J_nk, spins_f)
            return jnp.mean(T_f[:, top_idx]) + 0.1 * (-jnp.mean(E) / N)
    return eval_single
```

Since JAX traces at JIT time with Python control flow, each `make_eval_fn()` call produces a distinct traced function — no runtime branching issues.

### 3. Experiment Loop
For each loss function:
1. Create eval function via `make_eval_fn(objective)`
2. Initialize fresh `SeparableCMAES`
3. Run 200 CMA-ES iterations, record `best_hist` (fitness per iteration)
4. Extract `best_theta` → `J_best`
5. Run visualization rollout (single chain, `iters_eval` steps) → collect `T_frames`, `S_frames`
6. Generate 4 figures per experiment:
   - Fitness curve (line plot)
   - Final temperature heatmap (magma, vmin=0, vmax=3)
   - Final spin configuration (gray, vmin=-1, vmax=1)
   - Connectivity composite (R=in-degree, G=out-degree)
7. Store figures as base64-encoded PNGs in a dict

### 4. Comparison Section
- Side-by-side final temperature maps (1 row, 5 columns)
- Summary table: experiment name, final fitness, final top-row temp (computed for all regardless of loss)

### 5. HTML Report Generation
- Build HTML string with inline CSS styling
- Embed all figures as `<img src="data:image/png;base64,...">`
- Structure: title → experiment cards (each with 4 plots + description) → comparison section → summary table
- Write to `experiments/report.html`

## Visualization Details

Each experiment gets a card with:

```
┌─────────────────────────────────────────────┐
│ Experiment: max_top_temp                    │
│ Objective: Maximize mean top-row temp       │
│ Final fitness: 0.4821                       │
├─────────────┬───────────────────────────────┤
│ Fitness     │ Final Temperature             │
│ Curve       │ Heatmap                       │
├─────────────┼───────────────────────────────┤
│ Final Spins │ Connectivity                  │
│             │ (R=in, G=out)                 │
└─────────────┴───────────────────────────────┘
```

Comparison section at the bottom: all 5 final temperature maps side by side.

## Runtime Estimate
- Each experiment: ~200 CMA-ES iters x 32 population x 80 eval iters = ~512k forward passes
- 5 experiments total
- On CPU: ~5-15 min per experiment depending on hardware
- On GPU: ~1-3 min per experiment
