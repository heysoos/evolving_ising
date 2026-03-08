"""Evolutionary optimiser wrapper for work extraction (Phase 4).

Thin wrapper around SeparableCMAES from evolving_ising.optim,
plus fitness evaluation using the Ising model and thermodynamic accounting.

The fitness evaluation loop minimizes host↔device transfers and uses
vectorized numpy for bond updates (no per-bond Python loops).
"""

import numpy as np
import jax
import jax.numpy as jnp

from evolving_ising.optim import SeparableCMAES

from .thermodynamics import CycleAccumulator, temperature_schedule
from .controller import LocalController, LocalMagnetisationTracker
from .budgets import BaseBudget, NoBudget


class WorkExtractionES:
    """CMA-ES wrapper for work extraction optimisation."""

    def __init__(self, n_params, pop_size=20, sigma=0.02, seed=0):
        self.cma = SeparableCMAES(
            dim=n_params,
            pop_size=pop_size,
            sigma_init=sigma,
            seed=seed,
        )
        self.history = []

    def ask(self):
        """Sample population of parameter vectors."""
        X = np.asarray(self.cma.ask())
        return [X[i] for i in range(X.shape[0])]

    def tell(self, params_list, fitnesses):
        """Update CMA-ES with evaluated fitnesses (higher = better)."""
        X = jnp.array(np.stack(params_list))
        F = jnp.array(fitnesses)
        self.cma.tell(X, F)

        f_arr = np.array(fitnesses)
        self.history.append({
            'mean_fitness': float(f_arr.mean()),
            'best_fitness': float(f_arr.max()),
            'sigma': float(np.asarray(self.cma.state.sigma)),
        })

    @property
    def best_params(self):
        return np.asarray(self.cma.state.mean)


def evaluate_fitness(params, controller, budget, model, config):
    """Evaluate fitness of a controller parameterisation.

    Runs n_eval_cycles full temperature cycles and returns mean W_net.

    The step loop is kept in Python (required for stateful budget updates)
    but bond updates are fully vectorized and host↔device transfers are
    minimized to one round-trip per step.
    """
    controller.set_params(params)

    T_mean = config['T_mean']
    delta_T = config['delta_T']
    tau = config['tau']
    steps_per_cycle = config['steps_per_cycle']
    num_sweeps = config.get('num_sweeps', 1)
    n_eval_cycles = config['n_eval_cycles']
    J_min = config['J_min']
    J_max = config['J_max']
    bond_update_frac = config['bond_update_frac']
    lam = config['lambda']
    B_scale = config['B_scale']

    N = model.n
    K = model.K
    neighbors = np.asarray(model.neighbors)
    mask_np = np.asarray(model.mask, dtype=bool)

    # Precompute valid bond indices (constant for the lattice)
    valid_i, valid_k = np.where(mask_np)
    valid_j = neighbors[valid_i, valid_k]
    n_bonds_total = len(valid_i)
    n_updates = max(1, int(n_bonds_total * bond_update_frac))

    # Precompute temperature schedule for one cycle (numpy, reused)
    t_steps = np.arange(steps_per_cycle, dtype=np.float32)
    T_schedule = T_mean + delta_T * np.sin(2.0 * np.pi * t_steps / tau)

    # Initialise J, spins
    J_nk = np.full((N, K), config['J_init'], dtype=np.float32) * mask_np
    J_jax = jnp.array(J_nk)

    key = jax.random.PRNGKey(int(np.abs(params[:4]).sum() * 1000) % (2**31))
    key, init_key = jax.random.split(key)
    spins = model.init_spins(init_key, batch_size=1)

    mag_tracker = LocalMagnetisationTracker(
        N, alpha=config.get('mag_ema_alpha', 0.05)
    )

    rng = np.random.default_rng(
        int(np.abs(params[:4]).sum() * 7919) % (2**31)
    )

    W_nets = []

    for cycle in range(n_eval_cycles):
        acc = CycleAccumulator()
        E_init = float(model.energy(J_jax, spins).mean())
        acc.set_initial_energy(E_init)

        for t in range(steps_per_cycle):
            T_t = float(T_schedule[t])

            # --- Metropolis sweep (GPU) ---
            # Get energy before and run sweeps in one GPU round-trip
            E_before = model.energy(J_jax, spins)
            key, subkey = jax.random.split(key)
            spins, energies = model.metropolis_checkerboard_sweeps(
                subkey, spins, J_jax, T_t, num_sweeps
            )

            # Single transfer: pull both energies to host
            E_before_f = float(E_before.mean())
            E_after_f = float(energies.mean())
            delta_Q = E_after_f - E_before_f
            acc.step(delta_Q, T_t)

            # --- Controller + budget update (CPU, vectorized) ---
            # Pull spins to CPU once per step
            spins_np = np.asarray(spins[0])  # (N,) single chain
            mag_tracker.update(spins_np)

            # Select random bonds to update
            bond_idx = rng.choice(n_bonds_total, size=n_updates, replace=False)
            sel_i = valid_i[bond_idx]
            sel_k = valid_k[bond_idx]
            sel_j = valid_j[bond_idx]

            # Build controller inputs — keep on GPU where possible
            s_i_vals = spins_np[sel_i]
            s_j_vals = spins_np[sel_j]
            m_bar = mag_tracker.get()[sel_i]
            T_norm = (T_t - T_mean) / delta_T

            budgets_arr = budget.get_all_budgets_for_bonds(sel_i, sel_j)
            budgets_np = np.asarray(budgets_arr)
            budget_norm = np.tanh(budgets_np / B_scale).astype(np.float32)

            # Controller forward pass (GPU via JAX JIT)
            delta_J = np.asarray(controller.propose_updates(
                s_i_vals, s_j_vals, m_bar, T_norm, budget_norm
            ))

            # Vectorized cost computation and budget check
            costs = np.abs(s_i_vals * s_j_vals * delta_J) + lam * np.abs(delta_J)
            can_apply = budgets_np >= costs

            if can_apply.any():
                # Vectorized bond update via scatter on JAX array
                dJ_apply = np.where(can_apply, delta_J, 0.0)
                J_updated = np.asarray(J_jax)
                J_updated[sel_i[can_apply], sel_k[can_apply]] = np.clip(
                    J_updated[sel_i[can_apply], sel_k[can_apply]]
                    + dJ_apply[can_apply],
                    J_min, J_max
                )

                # Vectorized budget spending
                budget.spend_all(
                    sel_i[can_apply], sel_j[can_apply], costs[can_apply]
                )

                # Energy change from remodelling (single GPU call)
                J_new_jax = jnp.array(J_updated)
                E_remodel = float(
                    (model.energy(J_new_jax, spins) -
                     model.energy(J_jax, spins)).mean()
                )
                acc.step_remodel(E_remodel, 0.0)

                J_jax = J_new_jax

        acc.set_final_energy(float(model.energy(J_jax, spins).mean()))
        W_nets.append(acc.W_net)

    return float(np.mean(W_nets))
