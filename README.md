# Evolving Ising

GPU-accelerated evolution of 2D Ising model couplings to maximize heat diffusion through learned spin connectivity. Combines statistical mechanics simulation with evolutionary optimization using [JAX](https://github.com/jax-ml/jax).

## Overview

This project evolves the coupling strengths (J) of a 2D Ising spin lattice so that heat flows efficiently from a hot boundary to a cold one. A thermal gradient is imposed on the grid, and [CMA-ES](https://en.wikipedia.org/wiki/CMA-ES) searches for coupling configurations that maximize temperature propagation across the lattice.

**Physical setup:**
- A grid of spins (e.g. 64x64) with nearest-neighbor interactions
- Bottom row pinned at high temperature (heat source)
- Top row starts cold (heat sink)
- CMA-ES evolves per-edge coupling strengths to maximize average top-row temperature

The system alternates between MCMC spin dynamics (Metropolis-Hastings) and temperature diffusion at each step, coupling the thermodynamic and transport processes.

## Project Structure

```
core.py                 # All model code: Ising, diffusion, CMA-ES, evolution runner
evolving_ising.ipynb    # Main experiment notebook with visualization
test.ipynb              # Testing notebook
```

## Key Components

### `IsingModel`
GPU-friendly 2D Ising lattice with configurable topology.

- **Neighborhoods:** Moore (8-neighbor) or Von Neumann (4-neighbor)
- **Boundaries:** open, periodic, periodic_lr (left-right wrap), periodic_tb (top-bottom wrap)
- **Sampling:** Parallel checkerboard Metropolis-Hastings using graph coloring for GPU efficiency
- **Batched:** All operations vectorized over a batch dimension for running many chains in parallel

### `TemperatureDiffuser`
Simulates heat flow over the coupling graph:

```
T_{t+1} = (1 - alpha) * T_t + alpha * (W_norm @ T_neighbors)
```

Conductance W is derived from coupling strengths via configurable activations (abs, relu, softplus, square, sigmoid). Supports pinned boundary conditions for heat reservoirs.

### `SeparableCMAES`
Minimal diagonal-covariance CMA-ES implementation. Reduces memory from O(D^2) to O(D) while maintaining effective optimization for the high-dimensional coupling space.

### `EvoRunner`
Orchestrates the full optimization loop:
1. CMA-ES samples a population of candidate coupling vectors
2. Each candidate is evaluated by running warmup MCMC sweeps interleaved with diffusion, then measuring energy over additional sweeps
3. Fitness is reported back to CMA-ES to update the search distribution
4. Repeats for a configured number of iterations

## Installation

Requires Python 3.9+.

```bash
# CPU
pip install jax jaxlib matplotlib jupyter

# GPU (CUDA 12)
pip install "jax[cuda12_cudnn]" matplotlib jupyter

# Optional: alternative CMA-ES backend
pip install evosax
```

## Usage

### Notebook (recommended)

```bash
jupyter notebook evolving_ising.ipynb
```

The notebook walks through setup, optimization, and visualization of evolved coupling patterns and temperature dynamics.

### Programmatic

```python
from core import IsingModel, EvoConfig, EvoRunner
import jax.numpy as jnp

ising = IsingModel((64, 64), neighborhood="von_neumann", boundary="periodic_lr")

# Pin bottom row as heat source
N = ising.n
pin_mask = jnp.zeros(N, dtype=bool).at[N - 64 :].set(True)
pin_values = jnp.zeros(N, dtype=jnp.float32).at[N - 64 :].set(3.0)

cfg = EvoConfig(pop_size=64, iters=400, chains_per_eval=8)
runner = EvoRunner(ising, pin_mask, pin_values, cfg)
J_best, best_fitness = runner.run()
```

## Configuration

`EvoConfig` controls all hyperparameters:

| Parameter | Default | Description |
|---|---|---|
| `pop_size` | 32 | CMA-ES population size |
| `iters` | 200 | Number of CMA-ES generations |
| `sigma_init` | 0.3 | Initial search standard deviation |
| `j_scale` | 1.0 | Coupling strength multiplier |
| `j_transform` | `"softplus"` | Parameter-to-coupling map (softplus/relu/sigmoid/tanh01) |
| `warmup_sweeps` | 10 | MCMC sweeps before measurement |
| `measure_sweeps` | 10 | MCMC sweeps during measurement |
| `diffusion_steps_per_sweep` | 2 | Diffusion iterations per MCMC sweep |
| `alpha_diffusion` | 0.5 | Diffusion mixing rate |
| `conductance_mode` | `"abs"` | J-to-conductance mapping |
| `chains_per_eval` | 64 | Parallel MCMC chains per fitness evaluation |

## Algorithms

- **Ising model** &mdash; Classical lattice model from statistical mechanics with energy `E = -0.5 * sum_i s_i * h_i`
- **Metropolis-Hastings MCMC** &mdash; Samples spin configurations from the Boltzmann distribution. Uses graph coloring (2-color checkerboard for Von Neumann, 4-color for Moore) to update non-adjacent sites in parallel on GPU
- **Temperature diffusion** &mdash; Discrete heat equation on the coupling graph with pinned boundary conditions
- **Separable CMA-ES** &mdash; Gradient-free evolutionary strategy that adapts a diagonal Gaussian to maximize fitness

## Future Directions

- Maximizing entropy production
- Minimizing energy while maximizing diffusion
- Engine-like thermodynamic behavior
- Emergent insulation and mixing phenomena
