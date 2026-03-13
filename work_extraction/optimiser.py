"""Evolutionary optimiser wrapper for work extraction (Phase 4).

Thin wrapper around SeparableCMAES from evolving_ising.optim,
plus fitness evaluation using the Ising model and thermodynamic accounting.

The fitness evaluation loop minimizes host↔device transfers and uses
vectorized numpy for bond updates (no per-bond Python loops).
"""

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

from evolving_ising.optim import SeparableCMAES

from .thermodynamics import CycleAccumulator, temperature_schedule
from .controller import LocalController, LocalMagnetisationTracker, _mlp_forward, make_layer_specs
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


def make_jax_eval_fn(model, config, budget_type='none'):
    """Build a JAX-native evaluation function with lax.scan inner loop.

    Returns a @jax.jit function:
        eval_fn(params_flat, key) -> scalar W_net (float32)

    The entire simulation loop (warmup, cycles, steps) runs inside
    jax.lax.scan with no Python loops and no host<->device transfers.
    Safe to vmap over a population of parameter vectors.

    Parameters
    ----------
    model : IsingModel
    config : dict
    budget_type : str
        One of 'none', 'bond', 'neighbourhood', 'diffusing'.

    Returns
    -------
    eval_fn : callable  (JIT-compiled)
    """
    # --- Static config extraction ---
    T_mean = float(config['T_mean'])
    delta_T = float(config['delta_T'])
    tau = float(config['tau'])
    steps_per_cycle = int(config['steps_per_cycle'])
    n_eval_cycles = int(config['n_eval_cycles'])
    num_sweeps = int(config.get('num_sweeps', 1))
    warmup_sweeps = int(config.get('warmup_sweeps', 500))
    J_crit = T_mean / 2.269          # critical coupling (scalar constant)
    J_min = float(config['J_min'])
    J_max = float(config['J_max'])
    bond_update_frac = float(config['bond_update_frac'])
    lam = float(config['lambda'])
    B_scale = float(config['B_scale'])
    budget_alpha = float(config.get('budget_alpha', config.get('mag_ema_alpha', 0.05)))
    mag_alpha = float(config.get('mag_ema_alpha', 0.05))
    delta_J_max = float(config['delta_J_max'])
    hidden_size = int(config['hidden_size'])
    gamma = float(config.get('gamma', 0.25))
    D_diff = float(config.get('D', 0.1))
    tau_mu = float(config.get('tau_mu', 20.0))

    # --- Precomputed static arrays ---
    N = model.n
    K = model.K
    neighbors_np = np.asarray(model.neighbors)   # (N, K) int
    mask_np = np.asarray(model.mask, dtype=bool)  # (N, K) bool

    neighbors_jax = jnp.asarray(neighbors_np)    # (N, K) int32
    mask_f = jnp.asarray(mask_np, dtype=jnp.float32)  # (N, K)

    valid_i_np, valid_k_np = np.where(mask_np)
    valid_j_np = neighbors_np[valid_i_np, valid_k_np]
    n_bonds_total = len(valid_i_np)
    n_updates = max(1, int(n_bonds_total * bond_update_frac))

    valid_i_jax = jnp.asarray(valid_i_np, dtype=jnp.int32)
    valid_k_jax = jnp.asarray(valid_k_np, dtype=jnp.int32)
    valid_j_jax = jnp.asarray(valid_j_np, dtype=jnp.int32)

    # Effective neighbor count for diffusing budget Laplacian
    K_eff_jax = jnp.sum(mask_f, axis=1)  # (N,)

    # MLP layer specs (static Python tuple — used as static arg)
    layer_specs = make_layer_specs(hidden_size)

    # --- Budget pure functions dispatched by budget_type ---
    # Each returns updated budget state; all are pure JAX functions.
    # Budget state shapes:
    #   'none':          (1,)   zeros, never used
    #   'bond':          (N, K)
    #   'neighbourhood': (N,)
    #   'diffusing':     (N,)

    if budget_type == 'none':
        def bud_init():
            return jnp.zeros(1, dtype=jnp.float32)

        def bud_update(bud, s_bef, s_aft):
            return bud

        def bud_get(bud, si, sk, sj):
            return jnp.full(n_updates, jnp.inf, dtype=jnp.float32)

        def bud_spend(bud, si, sk, sj, costs, can_apply):
            return bud

    elif budget_type == 'bond':
        def bud_init():
            return jnp.zeros((N, K), dtype=jnp.float32)

        def bud_update(bud, s_bef, s_aft):
            # s_bef, s_aft: (N,) float32
            nbr_bef = s_bef[neighbors_jax]    # (N, K)
            nbr_aft = s_aft[neighbors_jax]
            corr_bef = s_bef[:, None] * nbr_bef
            corr_aft = s_aft[:, None] * nbr_aft
            ordering = jnp.maximum(0.0, corr_aft - corr_bef) * mask_f
            return bud + budget_alpha * ordering

        def bud_get(bud, si, sk, sj):
            return jnp.maximum(0.0, bud[si, sk])

        def bud_spend(bud, si, sk, sj, costs, can_apply):
            spend_amounts = jnp.where(can_apply, costs, 0.0)
            bud = bud.at[si, sk].add(-spend_amounts)
            return jnp.maximum(0.0, bud)

    elif budget_type == 'neighbourhood':
        def bud_init():
            return jnp.zeros(N, dtype=jnp.float32)

        def bud_update(bud, s_bef, s_aft):
            nbr_bef = s_bef[neighbors_jax]
            nbr_aft = s_aft[neighbors_jax]
            corr_bef = s_bef[:, None] * nbr_bef
            corr_aft = s_aft[:, None] * nbr_aft
            ordering = jnp.maximum(0.0, corr_aft - corr_bef) * mask_f
            return bud + budget_alpha * ordering.sum(axis=1)

        def bud_get(bud, si, sk, sj):
            nbr_budgets = bud[neighbors_jax] * mask_f           # (N, K)
            nbhd = bud + gamma * nbr_budgets.sum(axis=1)         # (N,)
            return jnp.minimum(nbhd[si], nbhd[sj])

        def bud_spend(bud, si, sk, sj, costs, can_apply):
            half = jnp.where(can_apply, costs / 2.0, 0.0)
            bud = bud.at[si].add(-half)
            bud = bud.at[sj].add(-half)
            return jnp.maximum(0.0, bud)

    elif budget_type == 'diffusing':
        def bud_init():
            return jnp.zeros(N, dtype=jnp.float32)

        def bud_update(bud, s_bef, s_aft):
            nbr_bef = s_bef[neighbors_jax]
            nbr_aft = s_aft[neighbors_jax]
            corr_bef = s_bef[:, None] * nbr_bef
            corr_aft = s_aft[:, None] * nbr_aft
            ordering = jnp.maximum(0.0, corr_aft - corr_bef) * mask_f
            eta = budget_alpha * ordering.sum(axis=1)
            nbr_mu = bud[neighbors_jax] * mask_f
            laplacian = nbr_mu.sum(axis=1) - K_eff_jax * bud
            bud = bud + D_diff * laplacian + eta - bud / tau_mu
            return jnp.maximum(0.0, bud)

        def bud_get(bud, si, sk, sj):
            return jnp.minimum(
                jnp.maximum(0.0, bud[si]),
                jnp.maximum(0.0, bud[sj])
            )

        def bud_spend(bud, si, sk, sj, costs, can_apply):
            half = jnp.where(can_apply, costs / 2.0, 0.0)
            bud = bud.at[si].add(-half)
            bud = bud.at[sj].add(-half)
            return jnp.maximum(0.0, bud)

    else:
        raise ValueError(f"Unknown budget_type: {budget_type!r}")

    # --- eval_fn: JIT-compiled, vmappable ---
    # _step_fn and _cycle_fn are defined inside _eval_fn so they close over
    # params_flat as a JAX traced value (safe for vmap/jit).

    def _eval_fn(params_flat, key, j_val):
        # j_val is sampled at generation level (in train.py) and shared across
        # all population members at a given chain index, so within-generation
        # fitness differences reflect controller quality not J_init luck.
        J_init_local = jnp.full((N, K), j_val, dtype=jnp.float32) * mask_f

        # Split remaining keys
        key, init_key, warmup_key = jax.random.split(key, 3)

        # Init spins
        spins = model.init_spins(init_key, batch_size=1)

        # Warmup at T_mean
        spins, _ = model.metropolis_checkerboard_sweeps(
            warmup_key, spins, J_init_local, T_mean, warmup_sweeps
        )

        # Init budget and mag EMA
        bud_state = bud_init()
        mag_ema = jnp.zeros(N, dtype=jnp.float32)

        # Outer cycle scan
        cycle_carry_0 = (spins, key, J_init_local, bud_state, mag_ema)

        # _step_fn and _cycle_fn are defined inside _eval_fn so they close over
        # params_flat as a JAX traced value (safe for vmap/jit).

        def _step_fn(carry, t):
            spins_c, key_c, J_c, bud_c, Q_in, Q_out, sigma, mag_c, E_carry = carry
            T_t = T_mean + delta_T * jnp.sin(2.0 * jnp.pi * t / tau)
            spins_bef = spins_c
            key_c, sub_m, sub_b = jax.random.split(key_c, 3)
            spins_c, _ = model.metropolis_checkerboard_sweeps(
                sub_m, spins_c, J_c, T_t, num_sweeps
            )
            E_after = jnp.mean(model.energy(J_c, spins_c))
            dE = E_after - E_carry
            Q_in = Q_in + jnp.maximum(0.0, dE)
            Q_out = Q_out + jnp.maximum(0.0, -dE)
            sigma = sigma + jnp.abs(dE) / T_t

            s_bef_f = jnp.mean(spins_bef.astype(jnp.float32), axis=0)
            s_aft_f = jnp.mean(spins_c.astype(jnp.float32), axis=0)
            bud_c = bud_update(bud_c, s_bef_f, s_aft_f)

            perm = jax.random.permutation(sub_b, n_bonds_total)[:n_updates]
            si = valid_i_jax[perm]
            sk = valid_k_jax[perm]
            sj = valid_j_jax[perm]

            s_i_vals = s_aft_f[si]
            s_j_vals = s_aft_f[sj]
            m_bar = mag_c[si]
            T_norm = (T_t - T_mean) / delta_T
            bud_vals = bud_get(bud_c, si, sk, sj)
            bud_norm = jnp.tanh(bud_vals / B_scale)
            T_norm_arr = jnp.full(n_updates, T_norm, dtype=jnp.float32)
            J_norm_arr = jnp.tanh(J_c[si, sk] / J_crit - 1.0)
            x = jnp.stack([s_i_vals, s_j_vals, m_bar, T_norm_arr, bud_norm,
                            J_norm_arr], axis=-1)

            dJ = _mlp_forward(params_flat, x, layer_specs, delta_J_max).ravel()
            costs = jnp.abs(s_i_vals * s_j_vals * dJ) + lam * jnp.abs(dJ)
            can_apply = bud_vals >= costs

            dJ_gated = jnp.where(can_apply, dJ, 0.0)
            J_c = J_c.at[si, sk].add(dJ_gated)
            J_c = jnp.clip(J_c, J_min, J_max) * mask_f
            bud_c = bud_spend(bud_c, si, sk, sj, costs, can_apply)

            E_post = jnp.mean(model.energy(J_c, spins_c))
            W_remodel_step = jnp.abs(E_post - E_after)
            mag_c = mag_alpha * s_aft_f + (1.0 - mag_alpha) * mag_c

            new_carry = (spins_c, key_c, J_c, bud_c, Q_in, Q_out, sigma, mag_c, E_post)
            return new_carry, W_remodel_step

        def _cycle_fn(carry, _):
            spins_c, key_c, J_c, bud_c, mag_c = carry
            E_init = jnp.mean(model.energy(J_c, spins_c))
            inner_carry_0 = (
                spins_c, key_c, J_c, bud_c,
                jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0),
                mag_c, E_init,
            )
            t_arr = jnp.arange(steps_per_cycle, dtype=jnp.int32)
            final_inner, W_remodels = jax.lax.scan(_step_fn, inner_carry_0, t_arr)
            spins_c, key_c, J_c, bud_c, Q_in, Q_out, sigma, mag_c, _ = final_inner
            W_net = (Q_out - Q_in) - jnp.sum(W_remodels)
            return (spins_c, key_c, J_c, bud_c, mag_c), W_net

        _, W_nets = jax.lax.scan(_cycle_fn, cycle_carry_0, None, length=n_eval_cycles)
        return jnp.mean(W_nets)

    return jax.jit(_eval_fn)
