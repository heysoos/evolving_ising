# Evolving Ising Work Extraction
## A Simulation Study of Adaptive Complexity Under Thermodynamic Selection Pressure

---

## Table of Contents

1. [Introduction](#introduction)
2. [Theoretical Background](#theoretical-background)
3. [Experiment Design Overview](#experiment-design-overview)
4. [Methods](#methods)
   - [Ising Model and Dynamics](#ising-model-and-dynamics)
   - [Thermodynamic Accounting](#thermodynamic-accounting)
   - [The Neural Network Controller](#the-neural-network-controller)
   - [Budget Mechanisms](#budget-mechanisms)
   - [Evolutionary Optimisation](#evolutionary-optimisation)
5. [Experiments](#experiments)
   - [Experiment 0: Baseline — Fixed J](#experiment-0-baseline--fixed-j)
   - [Experiment 1: Bond Budget](#experiment-1-bond-budget)
   - [Experiment 2: Neighbourhood Budget](#experiment-2-neighbourhood-budget)
   - [Experiment 3: Diffusing Neighbourhood Budget](#experiment-3-diffusing-neighbourhood-budget)
6. [Measurements and Analysis](#measurements-and-analysis)
7. [Implementation Notes for Agent](#implementation-notes-for-agent)

---

## Introduction

One of the most profound open questions in science is why life exists at all — and more specifically, why living systems tend to become more complex over time. The second law of thermodynamics tells us that isolated systems move toward disorder. Yet biological systems do the opposite: they build, maintain, and elaborate internal structure across evolutionary time.

The resolution of this apparent paradox lies in the word *isolated*. Living systems are not isolated. They are perpetually driven by environmental free energy gradients — chemical, thermal, and radiative — that keep them far from equilibrium. It has been argued, most forcefully by Schrödinger in *What is Life?* and formalised more recently in the frameworks of stochastic thermodynamics and the dissipation-driven adaptation hypothesis, that complexity is not in spite of thermodynamics but *because* of it: systems that develop internal structure are able to extract more work from their environment, and that selective advantage drives the ratchet of increasing complexity.

The canonical example of an environmental free energy gradient available to early life is the **day/night cycle** — a periodic oscillation in temperature (and light) with a period of approximately 24 hours. Any system capable of developing an internal structure that anticipates and exploits this cycle gains a thermodynamic advantage over one that does not. This is arguably the first and most persistent selective pressure in Earth's history.

This project investigates this hypothesis using the **Ising model** as a minimal substrate. The Ising model is the simplest system that undergoes a thermodynamic phase transition, making it the ideal test bed for studying how internal coupling structure can be tuned to exploit cyclically varying temperature baths. We couple the Ising lattice to a **small neural network controller** that learns to modulate the spin coupling $J(t)$ in response to the current system state and available thermodynamic budget. The budget itself is defined **locally** — different regions of the lattice maintain their own energy accounts, fed by local work extraction events and depleted by local remodelling costs.

The central question is:

> *What coupling structures $J(t)$ does thermodynamic selection pressure spontaneously discover, how do these structures relate to the statistics of the driving cycle, and how does the spatial complexity of the evolved solution scale with the locality and magnitude of the available metabolic budget?*

---

## Theoretical Background

### Work and Heat in the Ising Model

The Hamiltonian of the Ising model is:

$$\mathcal{H} = -\sum_{\langle i,j \rangle} J_{ij} \, s_i s_j - h \sum_i s_i \tag{1}$$

where $s_i \in \{-1, +1\}$, $J_{ij}$ is the coupling on bond $\langle ij \rangle$, and $h$ is an external field. In this project $h = 0$ throughout; all driving comes from temperature and coupling modulation.

The total differential of the energy splits cleanly into **work** (caused by parameter changes) and **heat** (caused by spin flips):

$$d\mathcal{H} = \delta W + \delta Q \tag{2}$$

When a coupling $J_{ij}$ changes at fixed spin configuration:

$$\delta W_{ij} = -s_i s_j \cdot \delta J_{ij} \tag{3}$$

When spin $i$ flips at fixed parameters:

$$\delta Q_i = 2s_i \left( \sum_{j \sim i} J_{ij} s_j \right) \tag{4}$$

Equation (3) is the microscopic work, local to bond $\langle ij \rangle$. Equation (4) is the microscopic heat, local to site $i$. Both are defined event-by-event, not just on average.

### The First Law and Budget Equation

For a local energy budget $\mathcal{B}_{ij}$ on bond $\langle ij \rangle$, the balance equation is:

$$\frac{d\mathcal{B}_{ij}}{dt} = \dot{W}_{ij}^{\text{extracted}} - C(\delta J_{ij}) \tag{5}$$

where $C(\delta J_{ij})$ is the metabolic cost of remodelling:

$$C(\delta J_{ij}) = |s_i s_j \cdot \delta J_{ij}| + \lambda |\delta J_{ij}| \tag{6}$$

The first term is the physical work required to change the bond against the current spin state. The second term $\lambda |\delta J|$ is a basal metabolic cost — it always costs something to remodel a bond, modelling the overhead of any physical change process. $\lambda$ is a hyperparameter.

### Entropy Production

The entropy produced by a single spin flip at site $i$ is:

$$\sigma_i = \ln \frac{W(s_i \to -s_i)}{W(-s_i \to s_i)} = -\frac{\delta Q_i}{T} \tag{7}$$

Total entropy production per cycle:

$$\Sigma_{\text{cycle}} = \sum_{\text{all flips}} \sigma_i \tag{8}$$

The work extraction efficiency is bounded by:

$$W_{\text{extracted}} = -\Delta F - T \cdot \Sigma \tag{9}$$

where $\Delta F = F(T_{\text{cold}}) - F(T_{\text{hot}})$ is the free energy difference between the two equilibrium states the system is cycling between. Equation (9) is the central accounting identity: $\Sigma$ is the tax on irreversibility, and minimising it is equivalent to maximising extracted work.

### The Driving Cycle

The temperature bath follows:

$$T(t) = T_{\text{mean}} + \Delta T \cdot \sin\left(\frac{2\pi t}{\tau}\right) \tag{10}$$

with $T_{\text{mean}}$, $\Delta T$, and cycle period $\tau$ as experimental parameters. The critical temperature for the 2D square lattice Ising model is:

$$T_c = \frac{2J}{k_B \ln(1 + \sqrt{2})} \approx 2.269 \, J \tag{11}$$

An optimally tuned system should place $T_c$ near $T_{\text{mean}}$ so that the cycle straddles the phase transition, maximising $\Delta F$ per cycle.

---

## Experiment Design Overview

All experiments share the following structure:

- A 2D square lattice Ising model of size $L \times L$ (recommended $L = 32$)
- Glauber dynamics at temperature $T(t)$ given by equation (10)
- A small neural network controller that proposes $\delta J_{ij}$ updates each timestep
- A budget mechanism (varying by experiment) that gates those updates
- An evolutionary optimisation loop that selects network weights based on net work extracted per cycle

The three main experiments differ only in **how the local budget is defined and propagated**:

| Experiment | Budget Type | Key Parameter |
|---|---|---|
| 0 | None (fixed $J$) | Baseline |
| 1 | Bond budget $\mathcal{B}_{ij}$ | $\lambda$ (basal cost) |
| 2 | Neighbourhood budget $\mathcal{B}_i^{\text{nbhd}}$ | $\gamma$ (sharing radius) |
| 3 | Diffusing budget $\mu_i(t)$ | $D$, $\tau_\mu$ (diffusion, decay) |

---

## Methods

### Ising Model and Dynamics

The project assumes a working Ising model implementation. The following interface is expected:

```python
class IsingLattice:
    def __init__(self, L: int, J: np.ndarray, T: float):
        """
        L     : lattice size (L x L)
        J     : (L, L, 4) array of bond couplings, one per direction
                directions: [right, up, left, down]
        T     : current temperature
        """

    def step(self) -> list[FlipEvent]:
        """
        Performs one Glauber sweep (L*L spin flip attempts).
        Returns list of FlipEvent(site_i, site_j, delta_H, accepted).
        """

    def magnetisation(self) -> float:
        """Returns mean magnetisation m = (1/N) sum_i s_i."""

    def energy(self) -> float:
        """Returns current total energy H."""

    def set_temperature(self, T: float):
        """Updates bath temperature."""

    def set_J(self, J: np.ndarray):
        """Updates coupling array."""
```

**FlipEvent** is a named tuple or dataclass:

```python
@dataclass
class FlipEvent:
    site: tuple[int, int]      # (row, col) of flipped spin
    delta_H: float             # energy change from flip
    accepted: bool             # whether flip was accepted
    neighbours: list[tuple]    # neighbouring sites
```

**Critical note for agent**: Do not modify the core Ising implementation. All new code wraps around it.

---

### Thermodynamic Accounting

At each timestep, after `lattice.step()` returns the list of flip events, compute the following quantities. These functions should be implemented in `thermodynamics.py`.

**Heat per flip event** (equation 4):

```python
def heat_from_flip(event: FlipEvent) -> float:
    """
    Heat dumped into bath by a single accepted spin flip.
    Positive = energy leaves system into bath (ordering event).
    """
    if not event.accepted:
        return 0.0
    return event.delta_H
```

**Work from bond change** (equation 3):

```python
def work_from_remodel(s_i: int, s_j: int, delta_J: float) -> float:
    """
    Work done on system when bond J_ij changes by delta_J
    at fixed spin configuration.
    Negative = system does work on external agent (desirable).
    """
    return -s_i * s_j * delta_J
```

**Entropy production per flip** (equation 7):

```python
def entropy_production_flip(event: FlipEvent, T: float) -> float:
    """
    Entropy produced by a single flip event.
    Always >= 0 on average; can be negative for individual events.
    """
    if not event.accepted:
        return 0.0
    return -event.delta_H / T
```

**Per-cycle totals** should be accumulated in a `CycleAccumulator` class:

```python
class CycleAccumulator:
    def __init__(self):
        self.W_extracted = 0.0      # net work from J changes
        self.Q_in = 0.0             # heat absorbed from hot bath
        self.Q_out = 0.0            # heat dumped to cold bath
        self.entropy_production = 0.0
        self.remodel_cost = 0.0

    def record_flip(self, event: FlipEvent, T: float):
        q = heat_from_flip(event)
        if q > 0:
            self.Q_out += q
        else:
            self.Q_in += abs(q)
        self.entropy_production += entropy_production_flip(event, T)

    def record_remodel(self, s_i, s_j, delta_J, cost):
        w = work_from_remodel(s_i, s_j, delta_J)
        self.W_extracted += -w   # negative work on system = positive extracted
        self.remodel_cost += cost

    @property
    def W_net(self) -> float:
        return self.W_extracted - self.remodel_cost
```

---

### The Neural Network Controller

The controller is a small MLP implemented in `controller.py`. It is **shared across all bonds** — the same weights apply everywhere, so the network learns a universal local rule rather than a site-specific one. This is the key biological honesty constraint.

**Architecture:**

```
Input (5 nodes)
  ├── s_i                    : current spin at site i
  ├── s_j                    : current spin at site j  
  ├── local_m_i              : short-window average magnetisation around i
  ├── T_normalised           : (T(t) - T_mean) / delta_T  ∈ [-1, 1]
  └── budget_normalised      : tanh(B_ij / B_scale)       ∈ (-1, 1)

Hidden layer: 8 nodes, tanh activation
Hidden layer: 8 nodes, tanh activation

Output (1 node)
  └── delta_J_proposed       : proposed change, scaled to [-delta_J_max, delta_J_max]
```

```python
import numpy as np

class LocalController:
    def __init__(self, hidden_size=8, delta_J_max=0.1, seed=None):
        self.delta_J_max = delta_J_max
        rng = np.random.default_rng(seed)
        # Xavier initialisation
        self.W1 = rng.normal(0, np.sqrt(2/5),  (hidden_size, 5))
        self.b1 = np.zeros(hidden_size)
        self.W2 = rng.normal(0, np.sqrt(2/hidden_size), (hidden_size, hidden_size))
        self.b2 = np.zeros(hidden_size)
        self.W3 = rng.normal(0, np.sqrt(2/hidden_size), (1, hidden_size))
        self.b3 = np.zeros(1)

    def forward(self, s_i, s_j, local_m, T_norm, budget_norm) -> float:
        x = np.array([s_i, s_j, local_m, T_norm, budget_norm])
        h1 = np.tanh(self.W1 @ x + self.b1)
        h2 = np.tanh(self.W2 @ h1 + self.b2)
        out = np.tanh(self.W3 @ h2 + self.b3)
        return float(out[0]) * self.delta_J_max

    def get_params(self) -> np.ndarray:
        return np.concatenate([
            self.W1.ravel(), self.b1,
            self.W2.ravel(), self.b2,
            self.W3.ravel(), self.b3
        ])

    def set_params(self, params: np.ndarray):
        idx = 0
        for attr, shape in [('W1',(8,5)),('b1',(8,)),
                             ('W2',(8,8)),('b2',(8,)),
                             ('W3',(1,8)),('b3',(1,))]:
            n = int(np.prod(shape))
            setattr(self, attr, params[idx:idx+n].reshape(shape))
            idx += n
```

**Local magnetisation** around site $i$ is a short-time exponential moving average:

```python
class LocalMagnetisationTracker:
    def __init__(self, L, alpha=0.05):
        """alpha: EMA decay, smaller = longer memory"""
        self.m = np.zeros((L, L))
        self.alpha = alpha

    def update(self, spins: np.ndarray):
        self.m = (1 - self.alpha) * self.m + self.alpha * spins
```

---

### Budget Mechanisms

Each experiment uses a different budget. All are implemented in `budgets.py` with a common interface:

```python
class BaseBudget:
    def update(self, flip_events: list[FlipEvent], T: float, spins: np.ndarray):
        """Called after each Glauber sweep to update budget values."""
        raise NotImplementedError

    def get_budget(self, i: int, j: int) -> float:
        """Returns the budget available for bond (i,j) to remodel."""
        raise NotImplementedError

    def spend(self, i: int, j: int, cost: float):
        """Deduct cost from budget after a remodel event."""
        raise NotImplementedError
```

#### Experiment 1: Bond Budget

Each bond $\langle ij \rangle$ maintains an independent scalar budget $\mathcal{B}_{ij}$. The budget accumulates when the bond's spin correlation changes in a productive direction (ordering event that releases energy to cold bath) and depletes on remodelling:

$$\frac{d\mathcal{B}_{ij}}{dt} = \alpha \cdot \max(0, \, J_{ij} \cdot \Delta(s_i s_j)) - C(\delta J_{ij}) \tag{12}$$

```python
class BondBudget(BaseBudget):
    def __init__(self, L: int, alpha: float = 0.1, B_max: float = 10.0):
        self.L = L
        self.alpha = alpha
        self.B_max = B_max
        # Shape (L, L, 4): one budget per bond per direction
        self.B = np.zeros((L, L, 4))
        self._prev_corr = None   # previous s_i * s_j for each bond

    def update(self, flip_events, T, spins):
        # Compute current bond correlations
        corr = self._bond_correlations(spins)
        if self._prev_corr is not None:
            delta_corr = corr - self._prev_corr
            # Productive: correlation increasing (system ordering)
            productive = np.maximum(0, delta_corr)
            self.B += self.alpha * productive
            self.B = np.clip(self.B, 0, self.B_max)
        self._prev_corr = corr.copy()

    def _bond_correlations(self, spins) -> np.ndarray:
        """Returns (L, L, 4) array of s_i * s_j for each bond."""
        corr = np.zeros((self.L, self.L, 4))
        corr[:,:,0] = spins * np.roll(spins, -1, axis=1)  # right
        corr[:,:,1] = spins * np.roll(spins, -1, axis=0)  # up
        corr[:,:,2] = spins * np.roll(spins,  1, axis=1)  # left
        corr[:,:,3] = spins * np.roll(spins,  1, axis=0)  # down
        return corr

    def get_budget(self, i, j) -> float:
        # Return minimum budget of both endpoints (bond is shared)
        # Use direction 0 (right) as canonical; agent handles symmetry
        return float(self.B[i, j, 0])

    def spend(self, i, j, cost):
        self.B[i, j, 0] = max(0, self.B[i, j, 0] - cost)
        self.B[i, j, 2] = max(0, self.B[i, j, 2] - cost)  # symmetric bond
```

#### Experiment 2: Neighbourhood Budget

Each site $i$ maintains a budget fed by all thermodynamic events in its local neighbourhood. The neighbourhood budget is a weighted sum of local site budgets:

$$\mathcal{B}_i^{\text{nbhd}} = \mathcal{B}_i + \gamma \sum_{j \sim i} \mathcal{B}_j \tag{13}$$

where $\gamma \in [0, 1]$ controls local sharing. Bond $\langle ij \rangle$ draws from the minimum of $\mathcal{B}_i^{\text{nbhd}}$ and $\mathcal{B}_j^{\text{nbhd}}$.

```python
class NeighbourhoodBudget(BaseBudget):
    def __init__(self, L: int, gamma: float = 0.25, alpha: float = 0.1,
                 B_max: float = 10.0):
        self.L = L
        self.gamma = gamma
        self.alpha = alpha
        self.B_max = B_max
        self.B_site = np.zeros((L, L))   # per-site raw budget

    def update(self, flip_events, T, spins):
        for event in flip_events:
            if not event.accepted:
                continue
            i, j = event.site
            q = event.delta_H
            # Ordering events (q > 0, energy leaving system) fill budget
            if q > 0:
                self.B_site[i, j] += self.alpha * q
        self.B_site = np.clip(self.B_site, 0, self.B_max)

    def _neighbourhood_budget(self) -> np.ndarray:
        """Compute B_nbhd for all sites."""
        neighbour_sum = (
            np.roll(self.B_site, -1, axis=1) +
            np.roll(self.B_site,  1, axis=1) +
            np.roll(self.B_site, -1, axis=0) +
            np.roll(self.B_site,  1, axis=0)
        )
        return self.B_site + self.gamma * neighbour_sum

    def get_budget(self, i, j) -> float:
        B_nbhd = self._neighbourhood_budget()
        # Bond budget = min of the two endpoint neighbourhood budgets
        ni, nj = (i, (j+1) % self.L)   # right neighbour (example)
        return float(min(B_nbhd[i, j], B_nbhd[ni, nj]))

    def spend(self, i, j, cost):
        self.B_site[i, j] = max(0, self.B_site[i, j] - cost * 0.5)
        ni, nj = (i, (j+1) % self.L)
        self.B_site[ni, nj] = max(0, self.B_site[ni, nj] - cost * 0.5)
```

#### Experiment 3: Diffusing Neighbourhood Budget

The most physically natural formulation. A continuous chemical potential field $\mu_i(t)$ diffuses across the lattice, decays spontaneously, is sourced by local work extraction, and is consumed by remodelling:

$$\frac{\partial \mu_i}{\partial t} = D \nabla^2 \mu_i + \eta_i(t) - \frac{\mu_i}{\tau_\mu} - R_i(\mu_i) \tag{14}$$

The discrete Laplacian on the lattice is:

$$\nabla^2 \mu_i = \sum_{j \sim i} \mu_j - 4\mu_i \tag{15}$$

The key dimensionless parameter governing the system's behaviour is:

$$\Lambda = \frac{D \cdot \tau_\mu}{\xi^2} \tag{16}$$

where $\xi$ is the Ising correlation length. When $\Lambda \sim 1$, energy diffusion and spin correlations are matched — this is the most interesting regime.

```python
class DiffusingBudget(BaseBudget):
    def __init__(self, L: int, D: float = 0.1, tau_mu: float = 20.0,
                 alpha: float = 0.5, mu_max: float = 10.0):
        """
        D       : diffusion coefficient
        tau_mu  : decay timescale (steps)
        alpha   : source strength per ordering event
        mu_max  : saturation cap
        """
        self.L = L
        self.D = D
        self.tau_mu = tau_mu
        self.alpha = alpha
        self.mu_max = mu_max
        self.mu = np.zeros((L, L))

    def _laplacian(self) -> np.ndarray:
        return (
            np.roll(self.mu, -1, axis=1) +
            np.roll(self.mu,  1, axis=1) +
            np.roll(self.mu, -1, axis=0) +
            np.roll(self.mu,  1, axis=0)
            - 4 * self.mu
        )

    def update(self, flip_events, T, spins):
        # Source term: ordering flips inject mu locally
        source = np.zeros((self.L, self.L))
        for event in flip_events:
            if not event.accepted:
                continue
            i, j = event.site
            if event.delta_H > 0:   # ordering event
                source[i, j] += self.alpha * event.delta_H

        # Euler step for reaction-diffusion equation (14)
        self.mu += (
            self.D * self._laplacian()
            + source
            - self.mu / self.tau_mu
        )
        self.mu = np.clip(self.mu, 0, self.mu_max)

    def get_budget(self, i, j) -> float:
        # Bond budget = geometric mean of endpoint potentials
        ni, nj = (i, (j+1) % self.L)
        return float(np.sqrt(self.mu[i, j] * self.mu[ni, nj] + 1e-8))

    def spend(self, i, j, cost):
        ni, nj = (i, (j+1) % self.L)
        half = cost / 2.0
        self.mu[i, j]   = max(0, self.mu[i, j]   - half)
        self.mu[ni, nj] = max(0, self.mu[ni, nj]  - half)
```

---

### Evolutionary Optimisation

The network weights $\theta$ are optimised using a simple **evolutionary strategy (ES)** — specifically the $(1 + \lambda)$-ES, which is robust, requires no gradient, and preserves the full stochastic Ising physics.

The fitness function is net work per cycle averaged over $N_{\text{eval}}$ cycles:

$$\mathcal{F}(\theta) = \frac{1}{N_{\text{eval}}} \sum_{k=1}^{N_{\text{eval}}} W_{\text{net}}^{(k)}(\theta) \tag{17}$$

```python
class EvolutionaryOptimiser:
    def __init__(self, controller: LocalController,
                 pop_size: int = 20,
                 sigma: float = 0.02,
                 elite_frac: float = 0.2,
                 n_eval_cycles: int = 10):
        self.controller = controller
        self.pop_size = pop_size
        self.sigma = sigma
        self.elite_frac = elite_frac
        self.n_eval_cycles = n_eval_cycles
        self.best_params = controller.get_params().copy()
        self.best_fitness = -np.inf
        self.history = []   # (generation, mean_fitness, best_fitness)

    def ask(self) -> list[np.ndarray]:
        """Generate population of perturbed parameter vectors."""
        return [
            self.best_params + self.sigma * np.random.randn(*self.best_params.shape)
            for _ in range(self.pop_size)
        ]

    def tell(self, params_list: list[np.ndarray], fitnesses: list[float]):
        """Update best params based on evaluated fitnesses."""
        n_elite = max(1, int(self.pop_size * self.elite_frac))
        ranked = sorted(zip(fitnesses, params_list), reverse=True)
        elite_params = [p for _, p in ranked[:n_elite]]
        self.best_params = np.mean(elite_params, axis=0)
        self.best_fitness = ranked[0][0]
        self.history.append((len(self.history), np.mean(fitnesses), self.best_fitness))

    def sigma_decay(self, decay=0.995):
        """Optionally decay mutation rate over generations."""
        self.sigma *= decay
```

**The main training loop** in `train.py` should follow this structure:

```python
def run_experiment(config: dict) -> ExperimentResult:
    lattice    = IsingLattice(L=config['L'], J=config['J_init'], T=config['T_mean'])
    controller = LocalController(**config['controller_kwargs'])
    budget     = config['budget_class'](**config['budget_kwargs'])
    optimiser  = EvolutionaryOptimiser(controller, **config['es_kwargs'])
    accum      = CycleAccumulator()
    mag_tracker = LocalMagnetisationTracker(config['L'])

    for generation in range(config['n_generations']):
        param_candidates = optimiser.ask()
        fitnesses = []

        for params in param_candidates:
            controller.set_params(params)
            fitness = evaluate_fitness(
                lattice, controller, budget, accum,
                mag_tracker, config
            )
            fitnesses.append(fitness)

        optimiser.tell(param_candidates, fitnesses)

        if generation % config['log_interval'] == 0:
            log_generation(generation, optimiser, lattice, config)

    return ExperimentResult(optimiser=optimiser, config=config)


def evaluate_fitness(lattice, controller, budget, accum, mag_tracker, config):
    """Run N_eval cycles and return mean net work."""
    accum_total = 0.0
    for cycle in range(config['n_eval_cycles']):
        accum.reset()
        for step in range(config['steps_per_cycle']):
            T = temperature_schedule(step, config)
            lattice.set_temperature(T)
            events = lattice.step()
            mag_tracker.update(lattice.spins)
            budget.update(events, T, lattice.spins)

            # Attempt J update for every bond
            propose_and_apply_updates(
                lattice, controller, budget, accum, mag_tracker, T, config
            )
        accum_total += accum.W_net
    return accum_total / config['n_eval_cycles']
```

**Bond update logic** (called each step for a random subset of bonds to keep cost manageable):

```python
def propose_and_apply_updates(lattice, controller, budget, accum,
                               mag_tracker, T, config):
    L = config['L']
    T_norm = (T - config['T_mean']) / config['delta_T']
    B_scale = config['B_scale']
    lambda_ = config['lambda']

    # Sample a fraction of bonds each step
    n_bonds = int(L * L * config['bond_update_frac'])
    sites = np.random.randint(0, L, size=(n_bonds, 2))

    for (i, j) in sites:
        ni, nj = i, (j + 1) % L   # right neighbour
        s_i = lattice.spins[i, j]
        s_j = lattice.spins[ni, nj]
        local_m = float(mag_tracker.m[i, j])
        B = budget.get_budget(i, j)
        B_norm = np.tanh(B / B_scale)

        delta_J = controller.forward(s_i, s_j, local_m, T_norm, B_norm)
        cost = abs(s_i * s_j * delta_J) + lambda_ * abs(delta_J)

        if B >= cost:
            # Apply the remodel
            lattice.J[i, j, 0] += delta_J
            lattice.J[ni, nj, 2] += delta_J   # symmetric bond
            # Clamp J to physical range
            lattice.J[i, j, 0]   = np.clip(lattice.J[i, j, 0],   0.01, 5.0)
            lattice.J[ni, nj, 2] = np.clip(lattice.J[ni, nj, 2], 0.01, 5.0)
            budget.spend(i, j, cost)
            accum.record_remodel(s_i, s_j, delta_J, cost)
```

---

## Experiments

### Experiment 0: Baseline — Fixed J

**Purpose**: Establish the performance ceiling of a non-adaptive system and verify thermodynamic accounting.

**Setup**:
- $J_{ij} = J_0$ constant everywhere, no controller, no budget
- Sweep $J_0 \in [0.2, 2.0]$ to find the optimal fixed coupling
- Sweep cycle period $\tau \in [10, 1000]$ steps

**What to measure**: $W_{\text{net}}$, $\Sigma_{\text{cycle}}$, $m(t)$

**Expected result**: A clear peak in $W_{\text{net}}$ near $J_0 \approx T_{\text{mean}} / 2.269$, and an optimal $\tau$ where the system's relaxation timescale matches the half-period. This establishes the baseline the adaptive experiments must beat.

**Config**:
```python
baseline_config = {
    'L': 32,
    'T_mean': 2.5,
    'delta_T': 1.5,
    'tau': 200,           # sweep this
    'J_init': 1.1,        # sweep this
    'n_eval_cycles': 50,
    'steps_per_cycle': 200,
    'budget_class': NoBudget,
    'bond_update_frac': 0.0,    # no updates
}
```

---

### Experiment 1: Bond Budget

**Purpose**: Test whether local bond-level budget constraints drive the emergence of temporally structured $J(t)$.

**Setup**:
- Each bond has independent budget $\mathcal{B}_{ij}$ per equation (12)
- Sweep basal cost $\lambda \in \{0.0, 0.01, 0.1, 0.5\}$
- Sweep budget accumulation rate $\alpha \in \{0.05, 0.1, 0.3\}$
- Run evolutionary optimisation for 500 generations

**Key questions**:
- Does the optimised $J(t)$ show the predicted quarter-cycle phase lag?
- Does increasing $\lambda$ (higher remodelling cost) reduce the complexity of the learned profile?
- Do spatially heterogeneous $J$ patterns emerge, or does the lattice remain uniform?

**Config**:
```python
bond_budget_config = {
    'L': 32,
    'T_mean': 2.5,
    'delta_T': 1.5,
    'tau': 200,
    'J_init': np.ones((32, 32, 4)) * 1.1,
    'budget_class': BondBudget,
    'budget_kwargs': {'alpha': 0.1, 'B_max': 10.0},
    'lambda': 0.05,           # sweep this
    'B_scale': 2.0,
    'bond_update_frac': 0.1,
    'n_generations': 500,
    'n_eval_cycles': 10,
    'steps_per_cycle': 200,
    'controller_kwargs': {'hidden_size': 8, 'delta_J_max': 0.1},
    'es_kwargs': {'pop_size': 20, 'sigma': 0.02, 'elite_frac': 0.2},
    'log_interval': 10,
}
```

---

### Experiment 2: Neighbourhood Budget

**Purpose**: Test how local energy sharing (controlled by $\gamma$) changes the spatial structure of evolved solutions.

**Setup**:
- Budget is neighbourhood-averaged per equation (13)
- Sweep $\gamma \in \{0.0, 0.1, 0.25, 0.5, 1.0\}$
  - $\gamma = 0$: equivalent to pure site budget (most local)
  - $\gamma = 1$: strong neighbour sharing (most diffuse)

**Key questions**:
- Does increasing $\gamma$ lead to more spatially coherent $J$ patterns?
- Is there an optimal $\gamma$ that maximises $W_{\text{net}}$?
- Does $\gamma$ interact with the cycle period $\tau$ (faster cycles may benefit from more local, faster-responding budgets)?

**Config**:
```python
neighbourhood_budget_config = {
    **bond_budget_config,          # inherit baseline settings
    'budget_class': NeighbourhoodBudget,
    'budget_kwargs': {'gamma': 0.25, 'alpha': 0.1, 'B_max': 10.0},  # sweep gamma
}
```

---

### Experiment 3: Diffusing Neighbourhood Budget

**Purpose**: Test the full reaction-diffusion model and the predictions around the dimensionless parameter $\Lambda = D \tau_\mu / \xi^2$.

**Setup**:
- Budget follows equation (14) with explicit diffusion and decay
- Sweep $D \in \{0.01, 0.1, 0.5, 2.0\}$ at fixed $\tau_\mu = 20$
- Sweep $\tau_\mu \in \{5, 20, 100\}$ at fixed $D = 0.1$
- Also sweep temperature (which changes $\xi$) to test the $\Lambda \sim 1$ prediction

**Key questions**:
- Is work extraction maximised when $\Lambda \sim 1$?
- Does the spatial pattern of $\mu_i$ correlate with domain wall locations?
- Do fast-remodelling (high $\mu$) regions spatially coincide with the domain boundaries, as predicted by the spontaneous regionalisation argument?

**Config**:
```python
diffusing_budget_config = {
    **bond_budget_config,
    'budget_class': DiffusingBudget,
    'budget_kwargs': {
        'D': 0.1,          # sweep this
        'tau_mu': 20.0,    # sweep this
        'alpha': 0.5,
        'mu_max': 10.0,
    },
}
```

For the $\Lambda$ sweep, compute the Ising correlation length from the connected spin-spin correlation function:

```python
def correlation_length(spins: np.ndarray) -> float:
    """
    Estimate correlation length xi from the spin-spin
    correlation function C(r) = <s_i s_{i+r}> - <s_i>^2.
    Fit C(r) ~ exp(-r/xi) to extract xi.
    """
    L = spins.shape[0]
    m = spins.mean()
    C = np.zeros(L // 2)
    for r in range(L // 2):
        shifted = np.roll(spins, -r, axis=1)
        C[r] = (spins * shifted).mean() - m**2
    C = np.maximum(C, 1e-10)
    # Fit log(C) = -r/xi + const
    rs = np.arange(1, L // 2)
    log_C = np.log(C[1:])
    xi = -1.0 / np.polyfit(rs, log_C, 1)[0]
    return xi
```

---

## Measurements and Analysis

All measurements should be saved per-generation and per-cycle to enable post-hoc analysis. Implement a `ResultsLogger` class that saves to HDF5 or numpy `.npz` files.

### Primary Performance Metrics

**1. Net work per cycle $W_{\text{net}}$**

The headline metric. Plot against generation number to show learning curve. Compare across experiments to show the advantage of adaptive over fixed $J$.

```python
# Expected output: learning curve
plot(generation, W_net_mean, label='mean over population')
plot(generation, W_net_best, label='best individual')
hline(W_net_fixed_J_optimal, label='fixed J ceiling')
```

**2. Work extraction efficiency $\eta$**

$$\eta = \frac{W_{\text{net}}}{W_{\text{Carnot}}} = \frac{W_{\text{net}}}{(1 - T_{\text{cold}}/T_{\text{hot}}) \cdot Q_{\text{in}}} \tag{18}$$

This normalises performance against the thermodynamic upper bound. An efficiency above the fixed-$J$ baseline confirms adaptive $J$ is genuinely doing something useful, not just extracting more from a larger free energy reservoir.

**3. Entropy production per cycle $\Sigma_{\text{cycle}}$**

From equation (8). Track over generations — an optimising system should show *decreasing* entropy production as it becomes more reversible. Plot $\Sigma$ vs. $W_{\text{net}}$ to verify the tradeoff described by equation (9).

---

### Temporal Structure Measurements

**4. Mean $J(t)$ profile**

After optimisation, run the best controller for 20 cycles and record the spatially averaged coupling $\bar{J}(t)$ as a function of cycle phase. Plot as a phase portrait:

```python
# Expected: oscillating J(t) with phase lag relative to T(t)
plot_phase(T_cycle, J_cycle_mean)
compute_phase_lag(T_cycle, J_cycle_mean)  # should be ~pi/2 for optimal
```

**5. Phase lag $\phi$**

Compute the cross-correlation between $T(t)$ and $\bar{J}(t)$ and extract the lag at peak correlation. This is a single number that summarises the temporal intelligence of the system. Plot $\phi$ vs. generation to show it converging toward $\pi/2$.

**6. Magnetisation oscillation**

Record $m(t)$ over a cycle. Verify it oscillates at the driving frequency with its own phase lag. The phase relationship between $m(t)$, $J(t)$, and $T(t)$ tells you whether the system is surfing the cycle productively.

---

### Spatial Structure Measurements

**7. Spatial map of $\bar{J}_{ij}$**

After optimisation, plot the time-averaged coupling strength on each bond as a 2D heatmap. Uniform maps indicate no spatial differentiation. Heterogeneous maps indicate the system has developed functional regions.

**8. Spatial correlation of budget and domain walls**

Compute the correlation between the budget field ($\mathcal{B}_{ij}$ or $\mu_i$) and domain wall locations (sites where $s_i \neq s_j$ for at least one neighbour). The spontaneous regionalisation prediction is:

$$\langle \mu_i \rangle_{\text{domain wall}} > \langle \mu_i \rangle_{\text{domain interior}} \tag{19}$$

Test this explicitly. If confirmed, it means the system has spontaneously concentrated adaptive resources at the thermodynamically active boundaries.

**9. Spatial entropy production map**

Compute the per-site entropy production rate $\dot{\sigma}_i$ averaged over many cycles. This should be highest at domain walls and lowest in ordered interiors. Overlay with the budget map to show spatial coincidence of resource production and consumption.

---

### Budget Sensitivity Measurements

**10. $W_{\text{net}}$ vs. $\lambda$ (Experiment 1)**

Sweep the basal remodelling cost $\lambda$ and plot $W_{\text{net}}$ of the converged solution. Expect a monotonically decreasing curve: higher cost → less remodelling → less adaptive $J$ → lower work. The slope of this curve is the *marginal value of metabolic flexibility*.

**11. $W_{\text{net}}$ vs. $\gamma$ (Experiment 2)**

Sweep neighbourhood sharing $\gamma$. Expect a non-monotonic curve with a peak at some optimal $\gamma^*$. Too little sharing and the budget is too noisy to be useful; too much sharing and locality is lost.

**12. $W_{\text{net}}$ and $\Lambda$ (Experiment 3)**

For each $(D, \tau_\mu)$ pair, compute $\Lambda$ from equation (16) using the measured $\xi$. Plot $W_{\text{net}}$ vs. $\Lambda$. The prediction is a peak near $\Lambda \sim 1$, confirming that matched diffusion and correlation lengthscales optimise work extraction.

---

### Verification Checks

These checks verify the implementation is physically correct before interpreting results.

**V1. First law conservation**

Per cycle: $\Delta U \approx Q_{\text{in}} - Q_{\text{out}} - W_{\text{net}}$. Should hold to numerical precision.

**V2. Second law compliance**

$\Sigma_{\text{cycle}} \geq 0$ at all times for all individuals. Any violation indicates a bug in entropy accounting.

**V3. Carnot bound**

$\eta \leq 1 - T_{\text{cold}} / T_{\text{hot}}$ always. Violations indicate a bug in work accounting.

**V4. Fixed-$J$ recovery**

With $\lambda \to \infty$ (infinite remodelling cost), the adaptive system should converge to the optimal fixed-$J$ solution from Experiment 0. This verifies that the budget constraint correctly suppresses adaptation when cost is prohibitive.

**V5. Budget non-negativity**

$\mathcal{B}_{ij}(t) \geq 0$ and $\mu_i(t) \geq 0$ at all times. Violations indicate a bug in the `spend` method.

---

## Implementation Notes for Agent

### Project Structure

```
project/
├── ising/
│   └── lattice.py          # EXISTS — do not modify
├── thermodynamics.py        # NEW: heat, work, entropy accounting
├── controller.py            # NEW: LocalController, LocalMagnetisationTracker
├── budgets.py               # NEW: BondBudget, NeighbourhoodBudget, DiffusingBudget
├── optimiser.py             # NEW: EvolutionaryOptimiser
├── train.py                 # NEW: main training loop, evaluate_fitness
├── analysis.py              # NEW: all measurement and plotting functions
├── experiments/
│   ├── exp0_baseline.py     # NEW
│   ├── exp1_bond_budget.py  # NEW
│   ├── exp2_nbhd_budget.py  # NEW
│   └── exp3_diffuse.py      # NEW
└── results/                 # auto-created, stores .npz output files
```

### Implementation Order

Implement in this order to allow incremental testing:

1. `thermodynamics.py` — verify with V1, V2, V3 checks on a fixed-$J$ run
2. `controller.py` — verify forward pass shapes and parameter get/set roundtrip
3. `budgets.py` — verify V4 and V5 checks; test each budget class independently
4. `optimiser.py` — verify fitness improves on a toy problem before connecting to Ising
5. `train.py` — connect all components; run Experiment 0 first to establish baseline
6. `experiments/` — run Experiments 1–3 in order, each building on the last
7. `analysis.py` — implement all measurements and generate figures

### Dependencies

```
numpy
scipy          # for curve fitting in correlation_length()
matplotlib     # for all plots
h5py           # for results logging (optional, npz is fine)
tqdm           # for progress bars
```

### Performance Notes

- The inner loop (Glauber step + budget update + bond proposals) must be fast. Vectorise bond correlation computation using `np.roll` as shown above — avoid Python loops over bonds.
- The evolutionary optimiser evaluates `pop_size × n_eval_cycles` full simulations per generation. With `pop_size=20` and `n_eval_cycles=10`, this is 200 full cycle runs per generation. On a 32×32 lattice with 200 steps/cycle, this is approximately $200 \times 200 \times 1024 = 40M$ spin flip attempts per generation. This should run in seconds with numpy vectorisation.
- If runtime is an issue, reduce $L$ to 16 for sweeps and use $L=32$ only for final figures.
- Consider using `multiprocessing.Pool` to parallelise the fitness evaluation across the population.
