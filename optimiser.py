"""Evolutionary optimiser wrapper for work extraction (Phase 4).

Thin wrapper around SeparableCMAES from evolving_ising.optim,
plus fitness evaluation using the Ising model and thermodynamic accounting.
"""

import numpy as np
import jax
import jax.numpy as jnp

from evolving_ising.model import IsingModel
from evolving_ising.optim import SeparableCMAES

from thermodynamics import (
    CycleAccumulator,
    temperature_schedule,
)
from controller import LocalController, LocalMagnetisationTracker
from budgets import BaseBudget, NoBudget


class WorkExtractionES:
    """CMA-ES wrapper for work extraction optimisation.

    Parameters
    ----------
    n_params : int
        Controller parameter dimensionality.
    pop_size : int
        Population size.
    sigma : float
        Initial step size.
    seed : int
        Random seed.
    """

    def __init__(self, n_params, pop_size=20, sigma=0.02, seed=0):
        self.cma = SeparableCMAES(
            dim=n_params,
            pop_size=pop_size,
            sigma_init=sigma,
            seed=seed,
        )
        self.history = []

    def ask(self):
        """Sample population of parameter vectors.

        Returns
        -------
        params_list : list of numpy arrays, each (n_params,)
        """
        X = np.asarray(self.cma.ask())  # (pop_size, n_params)
        return [X[i] for i in range(X.shape[0])]

    def tell(self, params_list, fitnesses):
        """Update CMA-ES with evaluated fitnesses.

        Parameters
        ----------
        params_list : list of arrays
            Parameter vectors (same order as ask()).
        fitnesses : list or array of floats
            Fitness values (higher is better — maximisation).
        """
        X = jnp.array(np.stack(params_list))
        F = jnp.array(fitnesses)
        self.cma.tell(X, F)

        # Record history
        f_arr = np.array(fitnesses)
        self.history.append({
            'mean_fitness': float(f_arr.mean()),
            'best_fitness': float(f_arr.max()),
            'sigma': float(np.asarray(self.cma.state.sigma)),
        })

    @property
    def best_params(self):
        """Current best estimate (CMA-ES mean)."""
        return np.asarray(self.cma.state.mean)


def evaluate_fitness(params, controller, budget, model, config):
    """Evaluate fitness of a controller parameterisation.

    Runs n_eval_cycles full temperature cycles and returns mean W_net.

    Parameters
    ----------
    params : array (n_params,)
        Controller parameters.
    controller : LocalController
        Controller instance (will be modified in place with params).
    budget : BaseBudget
        Budget instance.
    model : IsingModel
        Ising model.
    config : dict
        Must contain: T_mean, delta_T, tau, steps_per_cycle, num_sweeps,
        n_eval_cycles, J_init, J_min, J_max, bond_update_frac, lambda,
        B_scale.

    Returns
    -------
    float : mean W_net over n_eval_cycles.
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
    mask_np = np.asarray(model.mask)

    # Initialise J, spins
    J_nk = np.full((N, K), config['J_init'], dtype=np.float32) * mask_np

    key = jax.random.PRNGKey(int(np.abs(params[:4]).sum() * 1000) % (2**31))
    key, init_key = jax.random.split(key)
    spins = model.init_spins(init_key, batch_size=1)

    mag_tracker = LocalMagnetisationTracker(N, alpha=config.get('mag_ema_alpha', 0.05))

    W_nets = []

    for cycle in range(n_eval_cycles):
        acc = CycleAccumulator()
        J_jax = jnp.array(J_nk)

        E_init = float(model.energy(J_jax, spins).mean())
        acc.set_initial_energy(E_init)

        for t in range(steps_per_cycle):
            T = temperature_schedule(t, T_mean, delta_T, tau)
            T_float = float(T)

            # Energy before sweeps
            E_before = float(model.energy(J_jax, spins).mean())

            # Metropolis sweeps
            key, subkey = jax.random.split(key)
            spins, energies = model.metropolis_checkerboard_sweeps(
                subkey, spins, J_jax, T_float, num_sweeps
            )

            E_after = float(energies.mean())
            delta_Q = E_after - E_before
            acc.step(delta_Q, T_float)

            # Update magnetisation tracker
            spins_np = np.asarray(spins)
            mag_tracker.update(spins_np)

            # Update budget
            # (For simplicity, use the spins before/after for ordering events)
            # budget.update needs spins_before — we approximate with the
            # energy-based tracking already done

            # Bond updates: propose changes for a random fraction of bonds
            key, subkey = jax.random.split(key)
            n_bonds_total = int(mask_np.sum())
            n_updates = max(1, int(n_bonds_total * bond_update_frac))

            # Select random bonds to update
            valid_bonds_i, valid_bonds_k = np.where(mask_np)
            rng_np = np.random.default_rng(int(jax.random.bits(subkey)) % (2**31))
            bond_idx = rng_np.choice(len(valid_bonds_i), size=n_updates, replace=False)

            sel_i = valid_bonds_i[bond_idx]
            sel_k = valid_bonds_k[bond_idx]
            sel_j = neighbors[sel_i, sel_k]

            # Controller inputs
            s_flat = spins_np.ravel() if spins_np.ndim == 1 else spins_np[0]
            s_i_vals = s_flat[sel_i].astype(np.float32)
            s_j_vals = s_flat[sel_j].astype(np.float32)
            m_bar = mag_tracker.get()[sel_i]
            T_norm = float((T_float - T_mean) / delta_T)

            budgets_arr = budget.get_all_budgets_for_bonds(sel_i, sel_j)
            budget_norm = np.tanh(budgets_arr / B_scale).astype(np.float32)

            # Propose updates
            delta_J = controller.propose_updates(
                s_i_vals, s_j_vals, m_bar, T_norm, budget_norm
            )

            # Apply updates where budget allows
            J_new = J_nk.copy()
            total_remodel_cost = 0.0

            for idx in range(n_updates):
                ii, kk = sel_i[idx], sel_k[idx]
                jj = sel_j[idx]
                dj = delta_J[idx]

                # Cost: |s_i * s_j * dJ| + lambda * |dJ|
                cost = abs(s_i_vals[idx] * s_j_vals[idx] * dj) + lam * abs(dj)

                if budget.get_budget(ii, jj) >= cost:
                    J_new[ii, kk] = np.clip(J_nk[ii, kk] + dj, J_min, J_max)
                    budget.spend(ii, jj, cost)
                    total_remodel_cost += cost

            # Energy change from remodelling
            J_new_jax = jnp.array(J_new)
            E_before_remodel = float(model.energy(J_jax, spins).mean())
            E_after_remodel = float(model.energy(J_new_jax, spins).mean())
            delta_E_remodel = E_after_remodel - E_before_remodel

            acc.step_remodel(delta_E_remodel, 0.0)
            # basal cost already included in total_remodel_cost via spend

            J_nk = J_new
            J_jax = J_new_jax

        acc.set_final_energy(float(model.energy(J_jax, spins).mean()))
        W_nets.append(acc.W_net)

    return float(np.mean(W_nets))