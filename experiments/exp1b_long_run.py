"""Experiment 1b: Long-run diagnostics for best exp1 controllers.

Loads every saved controller from results/exp1/*/best_controller.npz,
runs each for n_cycles=500 temperature cycles, and records per-step
decision statistics to reveal long-term J drift and budget saturation.

Usage
-----
python experiments/exp1b_long_run.py [--results-dir results/exp1]
                                     [--out-dir results/exp1b]
                                     [--n-cycles 500]
                                     [--seed 0]

Output per run: results/exp1b/<run_name>/analysis.npz
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from evolving_ising.model import IsingModel          # noqa: E402
from work_extraction.controller import _mlp_forward, make_layer_specs  # noqa: E402


# ---------------------------------------------------------------------------
# Default config (matches exp1 training)
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG = {
    'L': 32, 'T_mean': 2.5, 'delta_T': 1.5, 'tau': 200,
    'J_init': 0.92, 'J_min': 0.01, 'J_max': 5.0,
    'steps_per_cycle': 200, 'bond_update_frac': 0.1,
    'delta_J_max': 0.1, 'hidden_size': 8, 'mag_ema_alpha': 0.05,
    'B_scale': 2.0, 'lambda': 0.0, 'budget_alpha': 0.1,
    'neighborhood': 'von_neumann', 'boundary': 'periodic',
    'num_sweeps': 1, 'warmup_sweeps': 500,
}


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def run_long_sim(run_dir, n_cycles=500, seed=0):
    """Run a saved controller for n_cycles cycles and record per-step traces.

    The entire simulation (after warmup) runs inside a single @jax.jit +
    jax.lax.scan call — no Python loops in the hot path.

    Parameters
    ----------
    run_dir : str or Path
    n_cycles : int
    seed : int

    Returns
    -------
    dict with arrays of shape (total_steps,) and 'config' key, or None on failure.

    Saved arrays
    ------------
    J_bar           float32  Mean coupling over all valid bonds
    T_trace         float32  Bath temperature T(t)
    budget_mean     float32  Mean budget over all valid bonds
    budget_norm_mean float32  Mean tanh(B/B_scale) over all valid bonds
    n_applied_pos   float32  Count of applied dJ > 0 per step
    n_applied_neg   float32  Count of applied dJ < 0 per step
    n_gated         float32  Count of gated (budget < cost) proposals per step
    dJ_mean         float32  Mean proposed dJ (signed, all n_updates bonds)
    dJ_applied_mean float32  Mean applied dJ (signed, applied bonds only)
    Q_in_step       float32  Heat absorbed from bath per step
    Q_out_step      float32  Heat released to bath per step
    W_remodel_step  float32  Work cost of J remodelling per step
    """
    run_dir = Path(run_dir)
    ctrl_path = run_dir / 'best_controller.npz'
    cfg_path = run_dir / 'config.json'

    if not ctrl_path.exists():
        print(f'  [skip] no best_controller.npz in {run_dir}', file=sys.stderr)
        return None

    params_flat = np.load(ctrl_path)['params']

    config = dict(_DEFAULT_CONFIG)
    if cfg_path.exists():
        with open(cfg_path) as f:
            config.update(json.load(f))

    # --- Unpack config ---
    L               = int(config['L'])
    T_mean          = float(config['T_mean'])
    delta_T         = float(config['delta_T'])
    tau             = float(config['tau'])
    steps_per_cycle = int(config['steps_per_cycle'])
    num_sweeps      = int(config.get('num_sweeps', 1))
    warmup_sweeps   = int(config.get('warmup_sweeps', 500))
    J_init_val      = float(config['J_init'])
    J_min           = float(config['J_min'])
    J_max           = float(config['J_max'])
    bond_update_frac = float(config['bond_update_frac'])
    delta_J_max     = float(config['delta_J_max'])
    hidden_size     = int(config['hidden_size'])
    mag_alpha       = float(config['mag_ema_alpha'])
    B_scale         = float(config['B_scale'])
    lam             = float(config['lambda'])
    budget_alpha    = float(config['budget_alpha'])

    total_steps = n_cycles * steps_per_cycle

    # --- Build model and precompute static arrays ---
    model = IsingModel(
        (L, L),
        neighborhood=config.get('neighborhood', 'von_neumann'),
        boundary=config.get('boundary', 'periodic'),
    )
    N = model.n
    K = model.K

    neighbors_np = np.asarray(model.neighbors)
    mask_np      = np.asarray(model.mask, dtype=bool)
    neighbors_jax = jnp.asarray(neighbors_np)
    mask_f        = jnp.asarray(mask_np, dtype=jnp.float32)
    valid_count_jax = jnp.float32(mask_np.sum())

    valid_i_np, valid_k_np = np.where(mask_np)
    valid_j_np  = neighbors_np[valid_i_np, valid_k_np]
    n_bonds_total = len(valid_i_np)
    n_updates     = max(1, int(n_bonds_total * bond_update_frac))

    valid_i_jax = jnp.asarray(valid_i_np, dtype=jnp.int32)
    valid_k_jax = jnp.asarray(valid_k_np, dtype=jnp.int32)
    valid_j_jax = jnp.asarray(valid_j_np, dtype=jnp.int32)

    J_init_jax  = jnp.full((N, K), J_init_val, dtype=jnp.float32) * mask_f
    params_jax  = jnp.asarray(params_flat)

    J_crit = T_mean / 2.269   # critical coupling

    layer_specs = make_layer_specs(hidden_size)

    # --- Step function (lax.scan body) ---
    def _step_fn(carry, t):
        spins_c, key_c, J_c, bud_c, mag_c, E_prev = carry

        T_t = T_mean + delta_T * jnp.sin(2.0 * jnp.pi * t / tau)

        # MC sweep
        s_bef_f = jnp.mean(spins_c.astype(jnp.float32), axis=0)
        key_c, sub_m, sub_b = jax.random.split(key_c, 3)
        spins_c, _ = model.metropolis_checkerboard_sweeps(
            sub_m, spins_c, J_c, T_t, num_sweeps
        )
        s_aft_f = jnp.mean(spins_c.astype(jnp.float32), axis=0)

        # Heat accounting (before J update)
        E_after = jnp.mean(model.energy(J_c, spins_c))
        dE = E_after - E_prev
        Q_in_step  = jnp.maximum(0.0,  dE)
        Q_out_step = jnp.maximum(0.0, -dE)

        # Budget update
        ordering = jnp.maximum(
            0.0,
            s_aft_f[:, None] * s_aft_f[neighbors_jax]
            - s_bef_f[:, None] * s_bef_f[neighbors_jax],
        ) * mask_f
        bud_c = bud_c + budget_alpha * ordering

        # Magnetisation EMA
        mag_c = mag_alpha * s_aft_f + (1.0 - mag_alpha) * mag_c

        # Sample bonds for controller
        perm = jax.random.permutation(sub_b, n_bonds_total)[:n_updates]
        si = valid_i_jax[perm]
        sk = valid_k_jax[perm]
        sj = valid_j_jax[perm]

        T_norm    = (T_t - T_mean) / delta_T
        bud_vals  = jnp.maximum(0.0, bud_c[si, sk])
        bud_norm   = jnp.tanh(bud_vals / B_scale)
        J_norm_arr = jnp.tanh(J_c[si, sk] / J_crit - 1.0)

        x = jnp.stack([
            s_aft_f[si], s_aft_f[sj], mag_c[si],
            jnp.full(n_updates, T_norm, dtype=jnp.float32),
            bud_norm, J_norm_arr,
        ], axis=-1)

        dJ    = _mlp_forward(params_jax, x, layer_specs, delta_J_max).ravel()
        costs = jnp.abs(s_aft_f[si] * s_aft_f[sj] * dJ) + lam * jnp.abs(dJ)
        can_apply = bud_vals >= costs

        dJ_gated = jnp.where(can_apply, dJ, 0.0)
        J_c = jnp.clip(
            J_c.at[si, sk].add(dJ_gated),
            J_min, J_max,
        ) * mask_f
        bud_c = jnp.maximum(
            0.0,
            bud_c.at[si, sk].add(-jnp.where(can_apply, costs, 0.0)),
        )

        # Work of remodelling (after J update)
        E_post     = jnp.mean(model.energy(J_c, spins_c))
        W_remodel  = jnp.abs(E_post - E_after)

        # --- Diagnostics ---
        J_bar        = jnp.sum(J_c * mask_f) / jnp.maximum(valid_count_jax, 1.0)
        budget_mean  = jnp.sum(bud_c * mask_f) / jnp.maximum(valid_count_jax, 1.0)
        bud_all_vals = jnp.maximum(0.0, bud_c[valid_i_jax, valid_k_jax])
        budget_norm_mean = jnp.mean(jnp.tanh(bud_all_vals / B_scale))

        n_applied_pos = jnp.sum(jnp.where(can_apply & (dJ > 0.0), 1.0, 0.0))
        n_applied_neg = jnp.sum(jnp.where(can_apply & (dJ < 0.0), 1.0, 0.0))
        n_gated       = jnp.sum(jnp.where(~can_apply, 1.0, 0.0))
        dJ_mean       = jnp.mean(dJ)
        n_applied_f   = jnp.sum(can_apply.astype(jnp.float32))
        dJ_applied_mean = jnp.sum(dJ_gated) / jnp.maximum(n_applied_f, 1.0)

        out = (
            J_bar, T_t, budget_mean, budget_norm_mean,
            n_applied_pos, n_applied_neg, n_gated,
            dJ_mean, dJ_applied_mean,
            Q_in_step, Q_out_step, W_remodel,
        )
        return (spins_c, key_c, J_c, bud_c, mag_c, E_post), out

    # --- JIT-compiled simulation ---
    @jax.jit
    def _run_sim(spins, key):
        E_init = jnp.mean(model.energy(J_init_jax, spins))
        init_carry = (
            spins, key, J_init_jax,
            jnp.zeros((N, K), dtype=jnp.float32),   # budget
            jnp.zeros(N, dtype=jnp.float32),         # mag_ema
            E_init,
        )
        t_arr = jnp.arange(total_steps, dtype=jnp.float32)
        _, outputs = jax.lax.scan(_step_fn, init_carry, t_arr)
        return outputs

    # --- Initialise and warmup ---
    key = jax.random.PRNGKey(seed)
    key, ik, wk, rk = jax.random.split(key, 4)
    spins = model.init_spins(ik, batch_size=1)
    spins, _ = model.metropolis_checkerboard_sweeps(
        wk, spins, J_init_jax, T_mean, warmup_sweeps
    )

    print(f'  Running {n_cycles} cycles ({total_steps} steps)...', end=' ', flush=True)
    outputs = _run_sim(spins, rk)
    print('done.')

    keys_out = [
        'J_bar', 'T_trace', 'budget_mean', 'budget_norm_mean',
        'n_applied_pos', 'n_applied_neg', 'n_gated',
        'dJ_mean', 'dJ_applied_mean',
        'Q_in_step', 'Q_out_step', 'W_remodel_step',
    ]
    return {k: np.asarray(v) for k, v in zip(keys_out, outputs)} | {'config': config}


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_all(exp1_dir, out_dir, n_cycles=500, seed=0):
    exp1_dir = Path(exp1_dir)
    out_dir  = Path(out_dir)

    run_dirs = sorted(
        d for d in exp1_dir.iterdir()
        if d.is_dir() and (d / 'best_controller.npz').exists()
    )
    if not run_dirs:
        print(f'No run directories found under {exp1_dir}', file=sys.stderr)
        return

    print(f'Found {len(run_dirs)} run(s) in {exp1_dir}')
    for run_dir in run_dirs:
        name = run_dir.name
        print(f'\n[{name}]')
        result = run_long_sim(run_dir, n_cycles=n_cycles, seed=seed)
        if result is None:
            continue
        save_dir = out_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)
        config = result.pop('config')
        np.savez(save_dir / 'analysis.npz', **result)
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print(f'  Saved to {save_dir / "analysis.npz"}')
    print(f'\nAll done. Results in {out_dir}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Run long-run diagnostics for exp1 controllers (Exp 1b).'
    )
    parser.add_argument('--results-dir', default='results/exp1',
                        help='exp1 results directory (default: results/exp1)')
    parser.add_argument('--out-dir', default='results/exp1b',
                        help='Output directory (default: results/exp1b)')
    parser.add_argument('--n-cycles', type=int, default=500,
                        help='Number of cycles per controller (default: 500)')
    parser.add_argument('--seed', type=int, default=0,
                        help='JAX PRNG seed (default: 0)')
    args = parser.parse_args()
    run_all(args.results_dir, args.out_dir, n_cycles=args.n_cycles, seed=args.seed)


if __name__ == '__main__':
    main()
