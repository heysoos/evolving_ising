# Evolving Ising Work Extraction
## Adaptive Complexity Under Thermodynamic Selection Pressure

---

## Introduction

One of the deepest questions in biology is why living systems tend to become more complex over time. The second law of thermodynamics tells us isolated systems move toward disorder — yet biological systems do the opposite. The resolution lies in the fact that living systems are not isolated. They are driven by environmental free energy gradients that keep them perpetually out of equilibrium.

The hypothesis explored here is that **complexity is thermodynamically selected for**: systems that develop richer internal structure are able to extract more work from their environment, and that advantage drives a ratchet of increasing sophistication. The day/night cycle — a periodic oscillation in temperature with a ~24-hour period — represents arguably the oldest and most persistent such gradient on Earth. Any system capable of developing internal structure that anticipates and exploits this cycle gains a thermodynamic advantage over one that cannot.

This project investigates that hypothesis using the **2D Ising model** as a minimal substrate. The Ising model is the simplest system that undergoes a thermodynamic phase transition, making it an ideal test bed for studying how internal coupling structure can be tuned to exploit a cyclically varying temperature bath. A small **neural network controller** learns to modulate the spin coupling $J(t)$ in response to local system state and available thermodynamic budget. Crucially, that budget is **locally defined** — different regions of the lattice maintain their own energy accounts, fed by local work extraction and depleted by local remodelling costs. No central coordination exists.

The central question is:

> *What coupling structures does thermodynamic selection pressure spontaneously discover, how do they relate to the statistics of the driving cycle, and how does spatial complexity scale with the locality and magnitude of the available metabolic budget?*

---

## Theoretical Background

### The Driving Cycle

The system is coupled to a heat bath with periodically oscillating temperature:

$$T(t) = T_{\text{mean}} + \Delta T \cdot \sin\!\left(\frac{2\pi t}{\tau}\right) \tag{1}$$

The critical temperature of the 2D square lattice Ising model is:

$$T_c \approx 2.269 \, J \tag{2}$$

An optimally tuned system should straddle the phase transition — $T_{\text{cold}} < T_c < T_{\text{hot}}$ — since this is where the free energy difference $\Delta F$ between ordered and disordered phases, and therefore the maximum extractable work, is largest.

### Work and Heat

The Hamiltonian is:

$$\mathcal{H} = -\sum_{\langle i,j \rangle} J_{ij} \, s_i s_j \tag{3}$$

(external field $h = 0$ throughout). Energy changes split unambiguously into work and heat depending on their cause:

$$d\mathcal{H} = \delta W + \delta Q \tag{4}$$

**Work** is energy change due to a parameter change at fixed spin configuration. When bond $J_{ij}$ changes by $\delta J_{ij}$:

$$\delta W_{ij} = -s_i s_j \cdot \delta J_{ij} \tag{5}$$

**Heat** is energy change due to a spin flip at fixed parameters. When spin $i$ flips:

$$\delta Q_i = 2s_i \sum_{j \sim i} J_{ij} s_j \tag{6}$$

Both quantities are defined event-by-event, locally in space, and accumulated over a full cycle to give $W_{\text{extracted}}$ and $Q_{\text{in}}, Q_{\text{out}}$.

### Entropy Production and the Cost of Irreversibility

The entropy produced by a single spin flip is:

$$\sigma_i = -\frac{\delta Q_i}{T} \tag{7}$$

Summing over all flip events in a cycle gives total entropy production $\Sigma_{\text{cycle}}$. The central accounting identity is:

$$W_{\text{extracted}} = -\Delta F - T \cdot \Sigma_{\text{cycle}} \tag{8}$$

where $\Delta F = F(T_{\text{cold}}) - F(T_{\text{hot}})$ is the free energy difference between the two equilibrium states the system cycles between. Equation (8) says that every unit of entropy produced costs exactly $T$ joules of extractable work. **Minimising entropy production and maximising work extraction are the same objective.**

### The Remodelling Cost

Changing a bond is not free. The metabolic cost of changing $J_{ij}$ by $\delta J_{ij}$ is:

$$C(\delta J_{ij}) = |s_i s_j \cdot \delta J_{ij}| + \lambda |\delta J_{ij}| \tag{9}$$

The first term is the physical work required to change the bond against the current spin state. The second term $\lambda |\delta J|$ is a basal overhead — it always costs something to remodel, regardless of spin state. $\lambda$ is a key experimental hyperparameter. The net work per cycle is then:

$$W_{\text{net}} = W_{\text{extracted}} - W_{\text{remodel}} \tag{10}$$

This is the fitness function. A system is thermodynamically viable only if $W_{\text{net}} > 0$.

---

## System Architecture

The simulation has three coupled components:

**1. Ising Lattice** — a 2D $L \times L$ square lattice with periodic boundary conditions, evolved under Glauber dynamics at temperature $T(t)$. The coupling array $J_{ij}$ is spatially heterogeneous and time-varying.

**2. Neural Network Controller** — a small MLP with shared weights applied identically at every bond. It takes as input the local spin states $s_i, s_j$, a short-window average of local magnetisation $\bar{m}_i$, the normalised current temperature $(T - T_{\text{mean}})/\Delta T$, and the normalised local budget $\tanh(\mathcal{B}/\mathcal{B}_{\text{scale}})$. It outputs a proposed coupling change $\delta J_{ij}$. Shared weights enforce that the system learns a universal local rule — the same biochemistry at every location — which is both biologically honest and computationally tractable.

**3. Local Budget** — gates whether a proposed $\delta J_{ij}$ is actually applied. If the local budget exceeds the cost $C(\delta J_{ij})$ from equation (9), the change is applied and the cost is deducted. Otherwise nothing changes. The budget accumulates from local thermodynamic events. The three experiments differ only in how this budget is defined.

---

## Evolutionary Optimisation

The network weights $\theta$ are optimised using a simple evolutionary strategy. A population of $N_{\text{pop}}$ parameter vectors is evaluated in parallel; each candidate runs the full Ising simulation for $N_{\text{eval}}$ cycles and its fitness is the mean $W_{\text{net}}$ per cycle from equation (10). The elite fraction of the population is selected, their parameters are averaged to form the new best estimate, and a new population is generated by adding Gaussian perturbations. The mutation magnitude $\sigma$ may be decayed over generations. No gradient is computed through the Ising dynamics — the full stochastic physics is preserved.

---

## Experiments

All experiments use $L = 32$, $T_{\text{mean}} = 2.5$, $\Delta T = 1.5$, $\tau = 200$ steps/cycle as defaults unless stated otherwise.

### Experiment 0: Baseline — Fixed J

A non-adaptive system with constant $J_{ij} = J_0$ everywhere and no controller. Sweep $J_0 \in [0.2, 2.0]$ and cycle period $\tau \in [10, 1000]$ to find the optimal fixed-coupling performance. This establishes the ceiling that adaptive experiments must beat and verifies thermodynamic accounting.

**Expected**: a peak in $W_{\text{net}}$ near $J_0 \approx T_{\text{mean}}/2.269$, and an optimal $\tau$ where the system's relaxation timescale $\tau_{\text{relax}} \sim e^{J/T}$ matches the half-period.

---

### Experiment 1: Bond Budget

Each bond $\langle ij \rangle$ maintains an independent scalar budget $\mathcal{B}_{ij}$. The budget accumulates when the bond's spin correlation increases (an ordering event — productive thermodynamic work) and depletes on remodelling:

$$\frac{d\mathcal{B}_{ij}}{dt} = \alpha \cdot \max\!\left(0,\, \Delta(s_i s_j)\right) - C(\delta J_{ij}) \tag{11}$$

Bonds that actively participate in the cycle accumulate faster; dormant bonds do not. This creates a **rich-get-richer dynamic** — well-tuned bonds earn the resources to become better tuned.

**Sweeps**: basal cost $\lambda \in \{0.0, 0.01, 0.1, 0.5\}$; accumulation rate $\alpha \in \{0.05, 0.1, 0.3\}$.

**Key questions**: Does the learned $J(t)$ show a quarter-cycle phase lag relative to $T(t)$? Does spatial heterogeneity in $J_{ij}$ emerge spontaneously? Does higher $\lambda$ suppress the complexity of the learned profile?

---

### Experiment 2: Neighbourhood Budget

Each site $i$ maintains a budget fed by local thermodynamic events, and bonds draw from a neighbourhood-averaged pool:

$$\mathcal{B}_i^{\text{nbhd}} = \mathcal{B}_i + \gamma \sum_{j \sim i} \mathcal{B}_j \tag{12}$$

The budget available to bond $\langle ij \rangle$ is the minimum of the neighbourhood budgets at its two endpoints. The parameter $\gamma \in [0,1]$ controls local energy sharing: $\gamma = 0$ is fully isolated (pure site budget), $\gamma = 1$ is strong neighbour sharing.

**Sweeps**: $\gamma \in \{0.0, 0.1, 0.25, 0.5, 1.0\}$.

**Key questions**: Does intermediate $\gamma$ outperform both extremes? Does the optimal $\gamma^*$ depend on cycle period $\tau$ — do faster cycles benefit from more local budgets?

---

### Experiment 3: Diffusing Neighbourhood Budget

The most physically natural formulation. A continuous chemical potential field $\mu_i(t)$ diffuses across the lattice, decays spontaneously, is sourced by local ordering events, and is consumed by remodelling:

$$\frac{\partial \mu_i}{\partial t} = D \nabla^2 \mu_i + \eta_i(t) - \frac{\mu_i}{\tau_\mu} - R_i \tag{13}$$

where $D$ is the diffusion coefficient, $\tau_\mu$ is the decay timescale, $\eta_i$ is the local source from ordering flip events, and $R_i$ is consumption from remodelling. The discrete Laplacian is $\nabla^2 \mu_i = \sum_{j\sim i} \mu_j - 4\mu_i$.

The key dimensionless parameter is:

$$\Lambda = \frac{D \cdot \tau_\mu}{\xi^2} \tag{14}$$

where $\xi$ is the Ising spin-spin correlation length, estimated from the exponential decay of the connected correlation function $C(r) = \langle s_i s_{i+r} \rangle - \langle s_i \rangle^2$. This is the ratio of the energy diffusion length $\sqrt{D\tau_\mu}$ to the spin correlation length. Work extraction is predicted to be maximised when $\Lambda \sim 1$ — when the energy currency can travel as far as spins are correlated, but no further.

**Sweeps**: $D \in \{0.01, 0.1, 0.5, 2.0\}$ at fixed $\tau_\mu = 20$; $\tau_\mu \in \{5, 20, 100\}$ at fixed $D = 0.1$. Also vary $T_{\text{mean}}$ to move $\xi$ and test the $\Lambda \sim 1$ prediction directly.

**Key questions**: Is $W_{\text{net}}$ peaked near $\Lambda = 1$? Do high-$\mu$ regions spontaneously colocalise with domain walls?

---

## Measurements and Analysis

### Primary Performance Metrics

**Net work per cycle $W_{\text{net}}$** — the headline fitness metric. Plot as a learning curve vs. generation, and compare all adaptive experiments against the fixed-$J$ ceiling from Experiment 0.

**Carnot efficiency**:

$$\eta = \frac{W_{\text{net}}}{Q_{\text{in}}} \cdot \frac{1}{1 - T_{\text{cold}}/T_{\text{hot}}} \tag{15}$$

Normalises performance against the thermodynamic upper bound and tracks how close the system gets to reversible operation.

**Total entropy production per cycle $\Sigma_{\text{cycle}}$** — should decrease as optimisation proceeds, consistent with equation (8). Plot $\Sigma$ vs. $W_{\text{net}}$ across generations to verify the predicted linear tradeoff.

### Temporal Structure

**Mean $J(t)$ phase portrait** — after optimisation, record the spatially averaged coupling $\bar{J}(t)$ over many cycles as a function of cycle phase. The predicted optimal profile is a sinusoidal oscillation with a ~quarter-cycle lag behind $T(t)$.

**Phase lag $\phi$** — extracted from the cross-correlation of $T(t)$ and $\bar{J}(t)$. Should converge toward $\pi/2$ during optimisation. A single interpretable number summarising the system's temporal intelligence.

**Magnetisation oscillation $m(t)$** — verify it oscillates at the driving frequency with a consistent phase relationship to both $J(t)$ and $T(t)$.

### Spatial Structure

**Spatial map of time-averaged $\bar{J}_{ij}$** — heatmap of coupling strength across the lattice after optimisation. Uniform maps indicate no differentiation; heterogeneous maps indicate emergent functional regions.

**Budget–domain wall colocalisation** — compute the mean budget (or $\mu_i$) at domain wall sites vs. domain interior sites. The spontaneous regionalisation prediction is:

$$\langle \mu_i \rangle_{\text{domain wall}} > \langle \mu_i \rangle_{\text{interior}} \tag{16}$$

If confirmed, adaptive resources have spontaneously concentrated at thermodynamically active boundaries — a minimal model of functional spatial differentiation.

**Per-site entropy production map** — average $\dot{\sigma}_i$ over many cycles and overlay with the budget map to verify spatial coincidence of resource production and consumption.

### Budget Sensitivity

**$W_{\text{net}}$ vs. $\lambda$** (Experiment 1) — the marginal value of metabolic flexibility. Expect monotonically decreasing $W_{\text{net}}$ with increasing $\lambda$.

**$W_{\text{net}}$ vs. $\gamma$** (Experiment 2) — expect a non-monotonic curve with a peak at some optimal $\gamma^*$. Too little sharing: budget too noisy. Too much: locality lost.

**$W_{\text{net}}$ vs. $\Lambda$** (Experiment 3) — the central prediction of the diffusing budget model. Expect a peak near $\Lambda = 1$.

### Verification Checks

Before interpreting any results, confirm:

- **First law**: $\Delta U = Q_{\text{in}} - Q_{\text{out}} - W_{\text{net}}$ holds per cycle to numerical precision.
- **Second law**: $\Sigma_{\text{cycle}} \geq 0$ for all individuals at all times.
- **Carnot bound**: $\eta \leq 1 - T_{\text{cold}}/T_{\text{hot}}$ always.
- **Fixed-$J$ recovery**: at $\lambda \to \infty$, the adaptive system should recover the optimal fixed-$J$ solution from Experiment 0.
- **Budget non-negativity**: $\mathcal{B}_{ij} \geq 0$ and $\mu_i \geq 0$ at all times.

---

## Project Structure

```
project/
├── ising/
│   └── lattice.py            # EXISTS — do not modify
├── thermodynamics.py          # heat, work, entropy accounting per equations (5–8)
├── controller.py              # LocalController MLP + magnetisation tracker
├── budgets.py                 # BondBudget, NeighbourhoodBudget, DiffusingBudget
├── optimiser.py               # evolutionary strategy, fitness evaluation
├── train.py                   # main training loop
├── analysis.py                # all measurements and plots
└── experiments/
    ├── exp0_baseline.py
    ├── exp1_bond_budget.py
    ├── exp2_nbhd_budget.py
    └── exp3_diffuse.py
```

Implement in the order listed: `thermodynamics.py` first (verify with the five checks above), then `controller.py`, `budgets.py`, `optimiser.py`, `train.py`, `analysis.py`, and finally the experiment scripts. Run Experiment 0 before any adaptive experiment to establish the baseline.
