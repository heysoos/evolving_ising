# Brainstorm: Budget Mechanism Issues and Fixes

## 1. Diagnosed Problems

### 1.1 The Budget Grows Without Bound

The current `BondBudget` update rule is:

```
B_ij(t+1) = B_ij(t) + alpha * max(0, delta(s_i * s_j))
```

Budget is only ever *added to* from ordering events. The only drain is
`bud_spend`, which is called when the controller actually applies a dJ update.
There is no natural decay or ceiling.

**Why this breaks the mechanism:** Once the budget is large, `budget_norm =
tanh(bud / B_scale)` saturates to ≈ 1.0. At B_scale=2.0, this happens after
roughly 6 units of accumulated budget — which occurs quickly for a 32×32
lattice with ~4000 bonds each accumulating ordering credits at every cold-phase
step. After saturation, the 5th controller input (`b_norm`) is always ≈ 1.0,
providing no information about whether budget is truly available. Simultaneously,
the gating condition `bud_vals >= costs` becomes trivially satisfied, so the
budget never actually constrains the controller. **The budget mechanism is
effectively disabled after the first few cycles.** This explains why even
λ=0.5 does not strongly suppress performance — the gating is toothless once
the budget has saturated.

### 1.2 J Only Increases Over Time

This is the most important mechanistic finding. Looking at the budget accumulation
timing relative to the temperature cycle:

**Cold phase (T < T_mean):** Metropolis sweeps favour spin alignment (ordering).
`delta(s_i * s_j) > 0` events are frequent. Budget grows rapidly.
Controller sees high `budget_norm ≈ 1.0` and high absolute budget.
Gating is permissive → the controller can act freely.
The thermodynamically correct action during cold phase is to *raise J* on aligned
bonds, and the controller has learned this. So J increases frequently.

**Hot phase (T > T_mean):** Metropolis sweeps favour disordering.
`delta(s_i * s_j) < 0` events are frequent — but since the update is
`max(0, ...)`, no new budget is earned during disordering. Meanwhile the
controller is trying to *decrease J* (the correct hot-phase strategy), which
requires spending budget. Budget only slowly depletes (or has already saturated
from the cold phase). Gating might still be permissive if budget is large.

**However, there is a subtler asymmetry:** Budget is earned from ordering events,
which are the same events that *motivate cold-phase J increases*. The controller
acts exactly when it has just earned credit — typically on freshly-ordered bonds
in the cold phase. The hot-phase J decreases require spending budget accumulated
from earlier cold-phase ordering, and the controller strategy heatmap shows the
controller does propose dJ < 0 during hot phase — but there is a second effect:

The controller sees `budget_norm` as input. When `budget_norm` is large it may
be inclined to propose larger |dJ| updates. During cold phase, `budget_norm` is
high (recently earned). During hot phase, budget may have been partially depleted
OR still be saturated. Either way, the asymmetry in *earning* leads to a slow
drift: cold-phase J increases reliably exceed hot-phase J decreases.

**Physical interpretation:** The budget mechanism is supposed to model an
energetic cost account funded by released heat. It tracks heat released during
ordering but not heat absorbed during disordering. This one-sided accounting
systematically favours J-strengthening operations and creates the upward J drift
observed in the phase portrait.

**The "controller shows dJ < 0 in the strategy plot but J still rises"
paradox:** The strategy heatmap samples the controller at fixed budget_norm=0.5
with a uniform sweep over T_norm. It correctly shows the controller proposes
dJ < 0 when T_norm > 0. But this tells us about the *proposed* dJ, not the
*applied* dJ. During actual simulation, the budget level during hot phase may
be different from 0.5 — likely higher (due to saturation), or if depleted, then
gating blocks the updates. In both limiting cases, the intended hot-phase
weakening either does not occur (gated out) or is less effective than cold-phase
strengthening.

### 1.3 Why λ=0.5 Can Win

High λ makes every bond update more expensive: `cost = |s_i * s_j * dJ| + λ|dJ|`.
This actually introduces a few counterintuitive effects:

1. **Implicit step-size regularisation:** Only small dJ updates clear the budget
   gate, which means the controller must propose more conservative changes.
   This prevents the J distribution from becoming too extreme in few steps.

2. **Budget depletion is slower:** Fewer updates are approved per step, so
   budget does not drain as fast during hot phase. This keeps budget levels
   more uniform across the cycle, reducing the phase-dependent asymmetry.

3. **Selection pressure toward small, frequent updates:** The CMA-ES landscape
   favours a controller that proposes small |dJ| consistently over one that
   proposes large |dJ| occasionally. With high λ, this is the only viable
   strategy anyway.

The net effect may be that high λ inadvertently *helps* the controller maintain
a more consistent intervention pattern, compensating for the budget saturation
problem by making the budget constraint more binding even when budget is large.
This is an accidental fix, not a designed one.

---

## 2. Budget Modification Proposals

### Proposal A: Exponential Decay (Recommended)

Add a decay term to the budget update:

```
B_ij(t+1) = (1 - decay_rate) * B_ij(t) + alpha * max(0, delta(s_i * s_j))
```

or equivalently:

```
B_ij(t+1) = B_ij(t) + alpha * max(0, delta(s_i * s_j)) - decay_rate * B_ij(t)
```

This creates a natural steady-state level `B* = alpha * <ordering_rate> /
decay_rate`, where the budget stabilises around a value that reflects current
ordering activity. The steady-state is proportional to ordering rate and
inversely proportional to decay.

**Advantages:**
- Budget reflects *current* ordering activity, not cumulative history
- budget_norm stays in a meaningful range if decay_rate and B_scale are set
  appropriately (B* ≈ B_scale for informative input)
- During the hot phase, budget decays toward zero (since no ordering events),
  naturally making the gating constraint tighter when the controller wants to
  decrease J — this is thermodynamically appropriate
- Simple one-parameter addition; compatible with JAX lax.scan

**Suggested sweep:** decay_rate ∈ {0.001, 0.005, 0.01, 0.05}

This is equivalent to what `DiffusingBudget` already does with the `tau_mu`
term (`-mu / tau_mu`) but applied directly to BondBudget without diffusion.

---

### Proposal B: Symmetric Budget (Physically Motivated)

Allow budget to decrease from disordering events:

```
B_ij(t+1) = max(0, B_ij(t) + alpha * delta(s_i * s_j))
```

(Note: no `max(0, ...)` on the delta term — disordering events reduce budget.)

**Physical interpretation:** Budget represents the net ordering work deposited
in bond (i,j). Disordering events withdraw from this account. Budget tracks
the *net* thermodynamic work done on the bond, not just the credits.

**Advantages:**
- Budget is naturally low during hot phase (disordering depletes it), tightly
  gating J decreases — the controller must "earn the right" to weaken bonds by
  first depositing ordering work
- Breaks the cold-phase bias by making disordering as relevant as ordering

**Disadvantages:**
- Budget starts at 0 and could oscillate around 0, making the tanh normalisation
  tricky (need to handle near-zero budgets gracefully)
- Might make the budget so volatile that the gating is too noisy

---

### Proposal C: Hard Cap (Simplest Fix)

Cap the budget at some maximum value:

```
B_ij = min(B_max, B_ij + alpha * max(0, delta(s_i * s_j)))
```

With `B_max = C * B_scale` (e.g. B_max = 4.0 with B_scale=2.0).

**Advantages:** Simple, prevents saturation of budget_norm.
**Disadvantages:** Doesn't help with the phase asymmetry; still biased toward
cold-phase accumulation. Doesn't make budget_norm informative about recent
ordering activity.

---

### Proposal D: Cycle-Normalised Reset

At the start of each temperature cycle, rescale (not zero) the budget:

```
B_ij_start_of_cycle = gamma_reset * B_ij_end_of_previous_cycle
```

with gamma_reset ∈ [0.5, 0.9]. This is a soft reset that gives the budget
a "memory" of the recent past while preventing unbounded accumulation.

**Advantages:** Keeps budget in a reasonable range long-term; simple to implement.
**Disadvantages:** Requires cycle-boundary detection in the lax.scan loop (doable
but adds complexity).

---

### Proposal E: Bidirectional Budget with Floor

Combine proposals A and B: budget decays AND responds bidirectionally:

```
B_ij(t+1) = max(0, B_ij(t) * (1 - decay_rate) + alpha * delta(s_i * s_j))
```

This gives a steady-state around 0 when the system is at equilibrium (equal
ordering and disordering), budget is positive when net ordering activity is
positive (cold phase), and near zero or briefly negative (clipped to 0) during
hot phase.

---

## 3. Exp1b: Long-Run Controller Behaviour Analysis

### Motivation

The standard exp1 training uses 10 evaluation cycles per fitness call. The
J-T phase portrait in the report runs only 5 cycles. We do not know what
happens to J over hundreds of cycles:
- Does J keep drifting upward until it saturates at J_max?
- Does the controller find a stable operating point?
- How often does the controller actually apply dJ > 0 vs dJ < 0 vs gated?
- Is the budget actually depleted during hot phase, or is it always saturated?

### Experiment Design

**Load:** The best-performing controllers from exp1 results (all 12 runs, but
focus on top 3 by W_net).

**Run:** 500 temperature cycles per controller. Each cycle = steps_per_cycle=200
steps. Use the same config as training (L=32, T_mean=2.5, delta_T=1.5, tau=200).

**Record at every step:**
- `J_bar`: mean coupling over all valid bonds
- `T(t)`: current bath temperature
- `budget_mean`, `budget_max`, `budget_p10`, `budget_p90`: budget distribution
- `budget_norm_mean`: mean tanh(B/B_scale), how often is it saturated
- `n_proposed`: number of bonds the controller proposes to update
- `n_positive`: proposed dJ > 0 (threshold 1e-4)
- `n_negative`: proposed dJ < 0
- `n_gated`: proposed but gated (budget < cost)
- `n_applied_pos`: applied dJ > 0
- `n_applied_neg`: applied dJ < 0
- `dJ_mean_pos`: mean magnitude of applied positive dJ
- `dJ_mean_neg`: mean magnitude of applied negative dJ
- `phase`: 1 if T > T_mean (hot), 0 if T < T_mean (cold)

**Key analyses:**

1. **J drift plot:** J_bar vs cycle number. Does it plateau or keep rising?
   Overlay with T_mean / 2.269 (critical coupling J_c). Show for each controller.

2. **Decision breakdown:** For hot vs cold phases, compute:
   - Fraction of updates that are positive / negative / gated
   - Mean applied |dJ| in each phase
   - Expected: cold phase → more positive, hot phase → more negative but
     potentially gated

3. **Budget trajectory:** mean budget level vs step, coloured by T(t). This
   directly shows whether budget saturates and whether it phase-tracks at all.

4. **Budget_norm saturation:** Histogram of budget_norm values over all steps
   and bonds. If heavily concentrated near 1.0, confirms saturation problem.

5. **J distribution evolution:** Histogram of J values at cycles 1, 10, 50,
   100, 500. Shows whether J distribution broadens, shifts, or saturates.

6. **Net J change per cycle:** `J_bar(end_of_cycle) - J_bar(start_of_cycle)`.
   If always positive, confirms the upward drift. Separate by hot and cold
   half-cycles to show which contributes more.

7. **W_net vs cycle:** Does W_net degrade as J drifts away from an optimal
   value? This would confirm that the J drift is harmful in the long run.

### Expected Findings (pre-experiment)

- Budget_norm will be heavily saturated (near 1.0) after cycle 5-10, confirming
  that the budget constraint is toothless.
- Net J change per cycle will be systematically positive.
- The hot-phase dJ < 0 proposals will be applied at similar or *lower* rates
  than cold-phase dJ > 0 proposals, despite the controller "knowing" the correct
  strategy.
- W_net will be highest in early cycles (before J has drifted far from optimal)
  and may degrade for 100+ cycles as J moves into a regime where the effective
  Tc is no longer matched to the bath temperature oscillation.

### Observed Findings (post-experiment)

The exp1b run confirmed and exceeded the expected findings with a striking
additional result:

- **J diverges to very large values** — J̄ does not plateau but keeps drifting
  toward J_max, eventually freezing the system in a highly ordered ferromagnetic
  state.
- **W_net approaches zero or goes strongly negative** — once J is large, the
  system is permanently ordered at all temperatures. There is no longer a
  hot/cold asymmetry to exploit; both phases see a fully magnetised lattice.
  W_net becomes dominated by W_remodel costs with no heat-engine benefit.
- **Most dJ proposals are positive — including in the hot phase.** This is the
  most surprising result. The strategy analysis heatmap showed the controller
  proposing dJ < 0 during hot conditions at T_norm > 0. But in the actual long
  run, positive proposals dominate even when T > T_mean.
- **Budget saturation confirmed** — budget_norm_mean is ≈ 1.0 throughout,
  consistent with the gating being inactive.

#### Why are hot-phase proposals mostly positive?

This requires explaining the training/evaluation mismatch in detail.

During training, each fitness evaluation starts fresh from J_init = 0.92 and
runs 10 cycles. The controller is therefore only ever evaluated in the regime
J ≈ [0.92, 0.92 + small_drift]. In this regime (J < J_c ≈ 1.10, close to
critical), both hot and cold phases produce meaningful dynamics — the system
can both order and disorder. The controller discovers that raising J during the
cold phase increases Q_out, and this is rewarded by the fitness signal.

**But the controller is never penalised for the long-run consequence of always
raising J.** After 10 cycles, the fitness evaluation resets to J_init = 0.92,
so the compounding effect never enters the fitness landscape. The controller
optimizes for a 10-cycle window and the optimal policy in that window is
approximately "always raise J slightly on aligned bonds."

At long times (after hundreds of cycles), J has drifted far above J_c (e.g.
J̄ ≈ 3–4). In this regime the system is deeply ordered at ALL temperatures,
including T_hot. Metropolis sweeps at T_hot cannot disorder the system
significantly when J is very large. Consequently:

1. `s_i * s_j ≈ +1` always (deep ferromagnet).
2. `m_bar ≈ +1` always (EMA tracks the persistent order).
3. `T_norm` is the only remaining signal that differentiates hot from cold.
4. But the controller was trained in a regime where T_norm < 0 (cold) reliably
   coincided with high dJ > 0 outcomes. In the new regime (J >> J_c), the
   T_norm > 0 case was essentially never encountered during training at these
   spin configurations — the controller has no well-trained response.
5. The controller's hot-phase behaviour extrapolates poorly outside the training
   distribution. Its default in the untrained regime appears to be: keep raising J.

This is a **distributional shift / out-of-distribution generalisation** failure.
The controller was trained exclusively in the neighbourhood of J_init = 0.92
and extrapolates arbitrarily (and incorrectly) when deployed in the large-J
regime.

---

## 4. Detailed Action Plan

### Phase 1: Exp1b — Diagnosis ✅ IMPLEMENTED

**Files created:**
- `experiments/exp1b_long_run.py` — simulation script
- `experiments/exp1b_report.py` — HTML report generator

**What was built:**

`run_long_sim(run_dir, n_cycles=500)` builds a single `@jax.jit` + `jax.lax.scan`
loop over `n_cycles × steps_per_cycle` steps with no Python loops in the hot path.
The batch runner `run_all()` scans `results/exp1/` and saves one `analysis.npz`
per run to `results/exp1b/<run_name>/`.

Per-step outputs recorded (all shape `(total_steps,) float32`):

| Array | Purpose |
|-------|---------|
| `J_bar`, `T_trace` | Long-term J drift and phase |
| `budget_mean`, `budget_norm_mean` | Budget level and tanh saturation |
| `n_applied_pos`, `n_applied_neg`, `n_gated` | Decision counts per step |
| `dJ_mean`, `dJ_applied_mean` | Signed dJ proposed vs applied |
| `Q_in_step`, `Q_out_step`, `W_remodel_step` | Full thermodynamic accounting |

The report generates 7 diagnostic figures per run in a scenario-selector layout,
plus interactive J̄ and W_net overview charts across all 12 runs:
1. J drift trajectory (coloured by hot/cold phase, J_c reference line)
2. Budget dynamics (budget_mean + budget_norm_mean over time)
3. Budget saturation histogram (% of steps with budget_norm > 0.9)
4. Decision breakdown bar chart (hot vs cold: pos/neg/gated fractions)
5. Signed dJ by phase (proposed vs applied per cycle)
6. J drift decomposition (hot vs cold contributions per cycle)
7. W_net per cycle (with rolling mean and cumulative)

**To run:**
```bash
python experiments/exp1b_long_run.py --n-cycles 500
python experiments/exp1b_report.py
```

**Deviations from original plan:**
- Recorded `dJ_mean` and `dJ_applied_mean` (mean signed dJ) instead of
  `dJ_mean_pos` / `dJ_mean_neg` separately — more compact and equally informative.
- `budget_max`, `budget_p10`, `budget_p90` were dropped in favour of
  `budget_norm_mean` (more directly relevant to the saturation hypothesis) and
  `budget_mean` (raw level). Can be added if needed.
- Full thermodynamic accounting (`Q_in`, `Q_out`, `W_remodel`) was included,
  enabling per-cycle W_net computation in the report without a second pass.

### Phase 1b Addendum: J_init Mismatch — Problem Analysis and Fix Proposals

#### The Core Problem

Training always reinitialises J to the same fixed J_init = 0.92 at the start of
every fitness evaluation. This means:

- The controller is only ever trained in the distribution D_train = {J ≈ 0.92
  after 0–10 cycles of drift}.
- At test time (long-run deployment), the system operates in a completely different
  distribution D_test = {J ≫ J_c, deeply ordered regime}.
- The gap D_train → D_test widens monotonically as J keeps drifting upward,
  until the controller is entirely out-of-distribution.

#### Why Identical J_init Is Problematic

A fixed J_init = 0.92 tells the controller implicitly: "you will always start
from this value, so you only need to be correct near this value." The CMA-ES
fitness landscape rewards strategies that extract work in the first 10 cycles
from J_init = 0.92, which is a completely different objective from "extract work
stably at whatever J the long-run system settles to."

The problem is not that J_init = 0.92 is a bad value per se — it is close to
J_c and thermodynamically reasonable. The problem is that it is *identical across
all evaluations*, so the controller learns a point strategy rather than a
distribution strategy.

#### Why Perfectly Inheriting J from Past Runs Is Also Problematic

An obvious fix is to carry J forward from the previous evaluation (i.e., start
each fitness call with whatever J the last call left behind). This has its own
serious issues:

1. **Non-stationary fitness landscape.** CMA-ES assumes the fitness function is
   fixed. If J_init drifts upward evaluation by evaluation, the fitness landscape
   changes across the training run. The CMA-ES mean and covariance are tuned to
   a moving target. Early in training, J is small and W_net is the dominant term;
   later, J is large and W_remodel or negative W_net dominates. Gradients
   (in an informal sense) point in different directions at different stages.

2. **Population members see different J_init values.** In CMA-ES, all 20
   population members are evaluated in the same generation. If J is inherited
   from the *mean* of the last generation's rollouts, different members will
   have accumulated different J trajectories during their individual rollouts —
   yet all start from the same inherited J, which is inconsistent.

3. **Early-generation catastrophe.** If a bad individual early in training drives
   J to a very large value, all subsequent evaluations start from that corrupted
   state. The training run has no recovery mechanism.

4. **J_init becomes part of the hidden state.** The controller's optimal policy
   depends on the current J, but J is not an input to the controller — it can only
   be inferred indirectly from spin state and history. A controller that cannot
   observe J cannot correctly modulate its behaviour in response to inherited J.

#### Fix Proposals

---

**Fix A: Randomise J_init per Evaluation (Recommended primary fix)**

Draw J_init ~ Uniform(J_lo, J_hi) independently at the start of each fitness
evaluation, where J_lo and J_hi bracket the expected long-run operating range.

Concretely: J_lo = J_c * 0.5 ≈ 0.55, J_hi = J_c * 2.0 ≈ 2.20, with
J_c = T_mean / 2.269 ≈ 1.10. Optionally use a log-uniform distribution since J
spans an order of magnitude.

**What this fixes:**
- The controller must learn to operate correctly at any J in [J_lo, J_hi], not
  just near 0.92. It cannot exploit a point-solution.
- Crucially, it must learn to *lower* J when J is already high and T is hot,
  because fitness evaluations starting at J_hi with a "keep raising J" policy
  will immediately saturate J_max and produce poor W_net.
- The controller is forced to learn the correct sign of dJ as a function of both
  J (current coupling level) and T_norm (phase). But note: J is NOT currently
  an input to the controller, so this fix alone is insufficient — the controller
  can only infer J from spin state and m_bar. At J >> J_c, spins are always
  aligned (s_i = s_j = +1, m_bar ≈ 1) regardless of temperature; this spin
  state is a proxy for high J. So the controller *can* distinguish high-J
  conditions via spin inputs, but must learn to do so.

**What this does NOT fix:**
- It does not prevent the controller from drifting J upward during a single
  evaluation starting at J_hi — it just means the *fitness penalty* for doing so
  is immediately felt (since that evaluation starts in the bad regime).
- Without budget decay, budget still saturates. Random J_init only fixes the
  training distribution problem; the budget saturation is a separate issue
  (Phase 2 fix).

**Practical concern — training difficulty:** Random J_init makes each evaluation
harder and more variable. The fitness signal becomes noisier. CMA-ES may need
more generations to converge. Mitigate by using a fixed seed per generation (not
per evaluation) for J_init, so all population members see the same J_init and
sigma/covariance updates are consistent.

---

**Fix B: J_init from a Prescribed Distribution Matching Long-Run Behaviour**

Use exp1b results to characterise the steady-state J distribution (e.g. the
histogram of J_bar values after cycle 200 onward). Set J_init ~ the observed
steady-state distribution. This is a data-driven version of Fix A that targets
the actual operating regime.

**Advantage:** More targeted — the training distribution matches the test
distribution rather than some arbitrary bracket.
**Disadvantage:** Requires running exp1b first to get the steady-state
distribution, then re-running exp1 with the updated J_init. Also, after the
budget fix, the steady-state J will change, so the J_init distribution would
need to be re-estimated iteratively.

---

**Fix C: Increase n_eval_cycles During Training**

Raise n_eval_cycles from 10 to, say, 100 or 200. The controller then pays the
price of J drift within a single fitness evaluation. A policy that keeps raising
J will see W_net collapse within that single long evaluation.

**Advantage:** Directly addresses the problem at its source — short training
horizon. No changes to the controller or J_init logic.
**Disadvantage:** Expensive — 10× to 20× slower per generation. With 500
generations × 20 population members × 200 cycles = 2,000,000 cycle evaluations.
Even with JAX vmap over the population, this is significant.

Can be combined with Fix A: use random J_init AND more cycles. A moderate
increase (e.g. 50 cycles) with random J_init is likely cheaper than 200 cycles
with fixed J_init.

---

**Fix D: Curriculum — Progressively Expand J_init Range**

Start training with J_init = 0.92 (original), then progressively widen the
J_init distribution as training proceeds. Early generations train in the familiar
regime; later generations train on generalisation.

**Advantage:** Warm-starts training from a known-good region, avoiding the
difficult initialisation problem of random J_init from the start.
**Disadvantage:** Requires a curriculum schedule, which is an additional
hyperparameter. If the transition is too abrupt, the controller "forgets" the
near-J_c regime.

---

**Fix E: Add J̄ as a Controller Input**

Give the controller access to the current mean coupling J̄ (or normalised
J_norm = J̄ / J_c) as a 6th input feature. This way the controller can
*explicitly* condition on whether it is in the high-J or low-J regime and
apply the correct sign of dJ.

**Advantage:** The controller has all the information it needs to avoid the
distributional shift problem. It can learn a simple rule: "if J_norm > 1 and
T_norm > 0, output dJ < 0."
**Disadvantage:** Changes the controller architecture (185 → 6→8→8→1 params).
Requires retraining from scratch. Does not by itself change the training
distribution (J_init is still fixed), so the controller may still not encounter
the high-J regime during training. Should be combined with Fix A.

---

#### Recommended Combined Fix — Phase 1 Scope

The exp1b findings reveal two independent root causes. Both are addressed here
(still Phase 1) before touching the budget mechanism (Phase 2):

| Component | Root Cause | Fix |
|-----------|-----------|-----|
| Training distribution mismatch | Fixed J_init = 0.92 | Fix A: random J_init per chain |
| Controller blind to current J | No J input feature | Fix E: per-bond J_norm as 6th input |

**Fix A — Random J_init per chain (✅ IMPLEMENTED):**
- In `make_jax_eval_fn`, reads `J_init_lo` and `J_init_hi` from config (defaults to
  `J_init` on both sides for backward compatibility).
- Inside `_eval_fn(params_flat, key)`, splits one key to sample
  `j_val ~ Uniform(J_init_lo, J_init_hi)` and builds `J_init_local` from it.
- Each chain (in the `n_eval_chains` vmap) receives a distinct key and therefore
  a distinct J_init — guaranteed by the existing chain-key splitting in `train.py`.
- Suggested range in config: `J_init_lo = 0.5` (below J_c ≈ 1.10), `J_init_hi = 3.0`
  (well above J_c). Log-uniform could be used later; uniform suffices for now.

**Fix E — Per-bond J_norm as 6th controller input (✅ IMPLEMENTED):**
- Controller MLP changed from 5→8→8→1 to 6→8→8→1 (129 → 137 params).
- New feature: `J_norm_ij = tanh(J_ij / J_crit - 1)` where `J_ij = J[si, sk]` is
  the current coupling for the bond being evaluated, and `J_crit = T_mean / 2.269`.
  - J = 0 → ≈ −0.76; J = J_crit → 0; J = 2*J_crit → ≈ +0.76; J = J_max → ≈ 1.
- Gives the controller direct knowledge of whether the current bond is above
  or below the critical coupling, enabling it to learn: "if J_norm > 0 and T_norm > 0
  (hot phase, overcoupled bond), output dJ < 0."
- **Incompatibility note:** existing exp1 saved controllers (129 params) cannot be
  loaded into the new 137-param architecture. Exp1 must be rerun (exp1c).

**Architecture centralisation (✅ IMPLEMENTED):**
- Added `make_layer_specs(hidden_size, input_size=6, output_size=1)` to
  `work_extraction/controller.py` as the single source of truth for MLP layer shapes.
- All files that previously hardcoded the `layer_specs` tuple now call this function:
  `work_extraction/optimiser.py`, `experiments/report_utils.py`,
  `experiments/exp1_report.py`, `experiments/exp1b_long_run.py`.

**Files modified:**
- `work_extraction/controller.py` — `input_size = 5 → 6`, `make_layer_specs()` added, `propose_updates` updated
- `work_extraction/optimiser.py` — random J_init in `_eval_fn`, J_norm in `_step_fn`, `make_layer_specs` call
- `experiments/report_utils.py` — J_norm in `run_anim_frames` step fn, `make_layer_specs` call
- `experiments/exp1_report.py` — J_norm in `_simulate_final_J` and strategy figs, `make_layer_specs` call
- `experiments/exp2_report.py` — J_norm in `fig_controller_strategy`, `fig_J_spatial`
- `experiments/exp1b_long_run.py` — J_norm in step fn, `make_layer_specs` call

---

---

## ⚠️ STOP — Phases 2–5 below are deferred until Phase 1 fixes are run and reviewed.

---

### Phase 2: Budget Fix — Exponential Decay (Proposal A)

After Phase 1 fixes (random J_init + J_norm input) are evaluated, implement
exponential decay for BondBudget:

1. **In `work_extraction/budgets.py`:** Add optional `decay_rate` parameter to
   `BondBudget.__init__` (default 0.0 for backwards compatibility). Modify
   `update()` to apply `self._budget *= (1 - self.decay_rate)` before adding
   ordering increments.

2. **In `work_extraction/optimiser.py` `make_jax_eval_fn`:** Add `decay_rate`
   to the `bud_update` closure for budget_type='bond':
   ```python
   def bud_update(bud, s_bef, s_aft):
       ordering = jnp.maximum(0.0, corr_aft - corr_bef) * mask_f
       return bud * (1.0 - bud_decay) + budget_alpha * ordering
   ```
   Read `bud_decay = float(config.get('bud_decay', 0.0))` from config.

3. **In `experiments/exp2_bond_budget.py`** (note: this would become exp1c or
   a config variant): add `bud_decay` to the sweep grid.

### Phase 3: Exp1c — Budget Decay Sweep

**File:** `experiments/exp1c_budget_decay.py`

Sweep:
- `bud_decay` ∈ {0.001, 0.005, 0.01, 0.05}
- Fix λ=0.0 and λ=0.1 (the two most informative λ values from exp1)
- Fix α=0.1 (moderate; best from exp1)
- 500 generations

Compare W_net to exp1 baseline. Check that:
- Budget_norm no longer saturates at 1.0
- J drift is reduced or eliminated
- W_net is maintained or improved

### Phase 4: Long-Run Test of Fixed Controllers

Repeat exp1b analysis on the best controllers from exp1c to confirm:
- Budget levels are now phase-dependent (high during cold, lower during hot)
- J drift is eliminated or much reduced
- Decision breakdown is now symmetric between hot and cold phases as intended
- W_net is stable over 500 cycles

### Phase 5: Report Integration

Update `exp1_report.py` and create `exp1c_report.py` to include:
- Budget saturation diagnostic (histogram of budget_norm over full run)
- Long-run J drift plot (from exp1b/exp1c analyses)
- Decision breakdown bar chart (hot vs cold, positive vs negative vs gated)

---

## 5. Summary

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| Budget grows unbounded | No decay in BondBudget update | Add exponential decay: `bud * (1 - decay_rate)` |
| Budget_norm always ≈ 1.0 | Saturation of tanh due to unbounded budget | Decay keeps budget at steady-state near B_scale |
| J drifts upward | Budget earned only during cold-phase ordering, enabling asymmetric J increases | Decay makes budget phase-dependent; hot-phase budget lower, J decreases more freely |
| λ=0.5 unexpectedly best | High λ accidentally limits spending rate, partially compensating for saturation | Explicit decay removes need for this crutch |
| Controller proposes dJ<0 but J still rises | Budget constraint is inactive (saturated); applies = proposed | Budget with decay makes constraint active during hot phase |
