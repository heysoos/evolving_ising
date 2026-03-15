"""Training loop for work extraction experiments (Phase 5).

Wires together IsingModel, controller, budget, and WorkExtractionES.
"""

import os
import sys
import json
import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm

from evolving_ising.model import IsingModel

from .controller import LocalController
from .budgets import make_budget as _make_budget_raw
from .optimiser import WorkExtractionES, make_jax_eval_fn


DEFAULT_CONFIG = {
    # --- Grid ---
    'L': 32,                  # lattice side length; grid is L×L spins
    # --- Temperature bath ---
    'T_mean': 2.5,            # mean bath temperature T̄
    'delta_T': 1.5,           # half-amplitude of sinusoidal oscillation: T(t) = T̄ ± ΔT
    'tau': 200,               # bath oscillation period in Metropolis steps
    # --- Coupling initialisation ---
    'J_init': 0.92,           # default scalar J used when J_init_lo == J_init_hi
    'J_init_lo': 0.01,        # lower bound for per-chain J_init uniform sampling
    'J_init_hi': 5.0,         # upper bound for per-chain J_init uniform sampling
    'J_min': 0.01,            # hard clamp lower bound on J during evolution
    'J_max': 5.0,             # hard clamp upper bound on J during evolution
    # --- Training schedule ---
    'n_generations': 500,     # number of CMA-ES generations
    'n_eval_cycles': 10,      # bath cycles per fitness evaluation
    'n_eval_chains': 5,       # independent MC chains averaged for variance reduction
    'steps_per_cycle': 200,   # Metropolis steps per bath cycle
    # --- Controller / budget ---
    'bond_update_frac': 0.1,  # fraction of bonds the controller may update per step
    'delta_J_max': 0.1,       # max |ΔJ| the controller can apply per update
    'B_scale': 2.0,           # budget scale factor (budget type-dependent)
    'lambda': 0.05,           # remodelling cost coefficient in W_net = Q_out - Q_in - λ·W_rem
    # --- CMA-ES ---
    'pop_size': 20,           # population size per generation
    'sigma': 0.02,            # initial CMA-ES step size
    'elite_frac': 0.2,        # fraction of population used to update the CMA mean
    'sigma_decay': 0.995,     # multiplicative sigma decay applied each generation
    # --- MLP controller ---
    'hidden_size': 8,         # hidden layer width (architecture: 6→H→H→1)
    'mag_ema_alpha': 0.05,    # EMA smoothing coefficient for local magnetisation tracker
    # --- Logging ---
    'log_interval': 10,       # print a status line every N generations (non-TTY runs)
    # --- Physics ---
    'neighborhood': 'von_neumann',  # spin neighbourhood type ('von_neumann' or 'moore')
    'boundary': 'periodic',         # lattice boundary condition
    'num_sweeps': 1,          # Metropolis checkerboard sweeps per step
    'warmup_sweeps': 500,     # sweeps to thermalise spins before evaluation begins
    # --- J pool ---
    'j_pool_size': 50,        # max J matrices stored in pool (0 = disabled)
    'j_random_frac': 0.2,     # fraction of chains using fresh random J_init each gen
    'j_pool_elite_frac': 0.3, # fraction of population whose J_final is added to pool
}


class JPool:
    """Rolling pool of high-fitness J matrices for warm-starting evaluations."""

    def __init__(self, max_size, N, K):
        self.max_size = max_size
        self.N, self.K = N, K
        self._fitnesses = []
        self._matrices = []

    def __len__(self):
        return len(self._fitnesses)

    def sample_chain_jinits(self, n_chains, random_frac, rng, J_init_lo, J_init_hi, mask_np):
        """Return (n_chains, N, K) array of J_init matrices — one per chain.

        random_frac of chains get a scalar j_val sampled from [J_init_lo, J_init_hi]
        (broadcast to (N,K)), matching the original pre-pool generation-level sampling.
        Remaining chains draw from pool. If pool is empty, all chains use random j_val.
        Within each chain, all population members share the same J_init, so
        fitness differences reflect controller quality, not J_init luck.
        """
        N, K = self.N, self.K
        pool_disabled = self.max_size == 0
        n_from_pool = 0 if (pool_disabled or len(self) == 0) else n_chains - max(1, round(n_chains * random_frac))
        n_random = n_chains - n_from_pool

        out = np.empty((n_chains, N, K), dtype=np.float32)

        # Random chains: sample a scalar j_val per chain from [J_init_lo, J_init_hi]
        j_vals = rng.uniform(J_init_lo, J_init_hi, size=n_random)
        for i, j in enumerate(j_vals):
            out[i] = np.full((N, K), j, dtype=np.float32) * mask_np

        if n_from_pool > 0:
            idxs = rng.choice(len(self), size=n_from_pool, replace=len(self) < n_from_pool)
            for slot, idx in enumerate(idxs):
                out[n_random + slot] = self._matrices[idx]

        return out

    def update(self, fitnesses, J_finals, elite_frac):
        """Add top elite_frac of population's J_finals to pool.

        fitnesses : 1D array-like of length pop_size
        J_finals  : np.ndarray (pop_size, N, K)
        """
        if self.max_size == 0:
            return
        fitnesses = np.asarray(fitnesses)
        n_add = max(1, round(len(fitnesses) * elite_frac))
        top_idxs = np.argsort(fitnesses)[-n_add:]

        for idx in top_idxs:
            self._fitnesses.append(float(fitnesses[idx]))
            self._matrices.append(np.asarray(J_finals[idx]).copy())

        if len(self._fitnesses) > self.max_size:
            order = np.argsort(self._fitnesses)[-self.max_size:]
            self._fitnesses = [self._fitnesses[i] for i in order]
            self._matrices = [self._matrices[i] for i in order]


@dataclass
class ExperimentResult:
    """Container for experiment results."""
    name: str
    config: dict
    training_log: dict  # keys: generation, mean_fitness, best_fitness, sigma
    best_params: np.ndarray
    final_J: Optional[np.ndarray] = None
    extra: dict = field(default_factory=dict)


def make_budget(budget_type, model, config):
    """Create a budget instance from config.

    Parameters
    ----------
    budget_type : str
        One of 'none', 'bond', 'neighbourhood', 'diffusing'.
    model : IsingModel
    config : dict

    Returns
    -------
    BaseBudget instance.
    """
    neighbors = np.asarray(model.neighbors)
    mask = np.asarray(model.mask)
    return _make_budget_raw(budget_type, neighbors, mask, config)


def run_experiment(config, budget_type='none', name='experiment',
                   results_dir='results', verbose=True):
    """Run a full work extraction experiment.

    Parameters
    ----------
    config : dict
        Experiment configuration (merged with DEFAULT_CONFIG).
    budget_type : str
        Budget strategy: 'none', 'bond', 'neighbourhood', 'diffusing'.
    name : str
        Experiment name for logging.
    results_dir : str
        Directory for saving results.
    verbose : bool
        Print progress.

    Returns
    -------
    ExperimentResult
    """
    # Merge with defaults
    cfg = {**DEFAULT_CONFIG, **config}

    L = cfg['L']
    model = IsingModel(
        (L, L),
        neighborhood=cfg['neighborhood'],
        boundary=cfg['boundary'],
    )

    # Build JAX eval function.
    # j_val is sampled once per chain per generation and shared across all
    # population members at that chain index, so within-generation fitness
    # differences reflect controller quality rather than J_init luck.
    n_eval_chains = int(cfg.get('n_eval_chains', 1))
    J_init_lo = float(cfg.get('J_init_lo', cfg['J_init']))
    J_init_hi = float(cfg.get('J_init_hi', cfg['J_init']))
    # When J_init_lo == J_init_hi (default), each chain gets the same scalar value.
    # When a range is configured, each random chain draws independently from [lo, hi].

    eval_fn_base = make_jax_eval_fn(model, cfg, budget_type)
    # eval_fn_base: (params_flat, key, J_init_arr) -> (scalar, (N,K))
    # vmap over population; J_init_arr is broadcast (all members share the same J_init per chain)
    _eval_pop = jax.vmap(eval_fn_base, in_axes=(0, 0, None))

    if n_eval_chains > 1:
        def _eval_batch(params_batch, keys_by_chain, J_inits_by_chain):
            # J_inits_by_chain: (n_chains, N, K) — each chain has its own J_init
            def _one_chain(keys_c, J_init_c):
                return _eval_pop(params_batch, keys_c, J_init_c)
            w_nets_all, J_finals_all = jax.vmap(_one_chain)(keys_by_chain, J_inits_by_chain)
            # w_nets_all: (n_chains, pop_size); J_finals_all: (n_chains, pop_size, N, K)
            return jnp.mean(w_nets_all, axis=0), J_finals_all[0]
        eval_batch = jax.jit(_eval_batch)
    else:
        def _eval_batch_single(params_batch, keys, J_init_c):
            # J_init_c: (N, K) — broadcast to all members
            return _eval_pop(params_batch, keys, J_init_c)
        eval_batch = jax.jit(_eval_batch_single)

    # Determine n_params from controller (same MLP architecture)
    controller = LocalController(
        delta_J_max=cfg['delta_J_max'],
        hidden_size=cfg['hidden_size'],
    )

    es = WorkExtractionES(
        n_params=controller.n_params,
        pop_size=cfg['pop_size'],
        sigma=cfg['sigma'],
        seed=0,
    )

    master_key = jax.random.PRNGKey(0)

    # J pool setup
    pool = JPool(max_size=cfg.get('j_pool_size', 50), N=model.n, K=model.K)
    mask_np = np.asarray(model.mask, dtype=np.float32)
    pool_rng = np.random.default_rng(42)
    j_random_frac = float(cfg.get('j_random_frac', 0.2))
    j_pool_elite_frac = float(cfg.get('j_pool_elite_frac', 0.3))

    # Training loop
    generations = []
    mean_fitnesses = []
    best_fitnesses = []
    sigmas = []

    best_ever_fitness = -float('inf')
    best_ever_params = None

    n_gens = cfg['n_generations']

    is_tty = sys.stdout.isatty()
    log_interval = cfg['log_interval']

    pbar = tqdm(range(n_gens), desc=name, unit='gen', disable=not verbose or not is_tty,
                dynamic_ncols=True)
    for gen in pbar:
        params_list = es.ask()

        # Evaluate entire population in parallel via vmap+jit.
        # J_init is sampled once per chain (all members in a chain share the same J_init),
        # so within-generation fitness differences reflect controller quality, not J_init luck.
        params_batch = jnp.array(np.stack(params_list))
        pop_size = cfg['pop_size']

        J_inits_by_chain_np = pool.sample_chain_jinits(
            n_eval_chains, j_random_frac, pool_rng, J_init_lo, J_init_hi, mask_np,
        )

        if n_eval_chains > 1:
            all_keys = jax.random.split(master_key, 1 + n_eval_chains * pop_size)
            master_key = all_keys[0]
            keys_by_chain = all_keys[1:].reshape(n_eval_chains, pop_size, -1)
            J_inits_by_chain = jnp.asarray(J_inits_by_chain_np)
            fitnesses_jax, J_finals_jax = eval_batch(params_batch, keys_by_chain, J_inits_by_chain)
        else:
            all_keys = jax.random.split(master_key, 1 + pop_size)
            master_key = all_keys[0]
            J_init_c = jnp.asarray(J_inits_by_chain_np[0])  # single (N, K)
            fitnesses_jax, J_finals_jax = eval_batch(params_batch, all_keys[1:], J_init_c)
        fitnesses = list(np.asarray(fitnesses_jax))

        # Update pool with top performers' final J (J_finals_jax shape: (pop_size, N, K))
        pool.update(fitnesses, np.asarray(J_finals_jax), j_pool_elite_frac)

        es.tell(params_list, fitnesses)

        gen_best = max(fitnesses)
        gen_mean = float(np.mean(fitnesses))

        if gen_best > best_ever_fitness:
            best_ever_fitness = gen_best
            best_idx = fitnesses.index(gen_best)
            best_ever_params = np.asarray(params_list[best_idx]).copy()

        generations.append(gen)
        mean_fitnesses.append(gen_mean)
        best_fitnesses.append(gen_best)
        sigmas.append(float(np.asarray(es.cma.state.sigma)))

        # Tqdm postfix for interactive runs
        pbar.set_postfix(best=f'{best_ever_fitness:.3f}', mean=f'{gen_mean:.3f}',
                         sigma=f'{sigmas[-1]:.4f}')

        # Plain-text periodic print for file-redirected / non-TTY runs
        if verbose and not is_tty and (gen % log_interval == 0 or gen == n_gens - 1):
            print(f"  [{name}] gen {gen:4d}/{n_gens}  "
                  f"best_ever={best_ever_fitness:8.4f}  "
                  f"gen_best={gen_best:8.4f}  "
                  f"mean={gen_mean:8.4f}  "
                  f"sigma={sigmas[-1]:.4f}", flush=True)

    pbar.close()

    # Save results
    training_log = {
        'generation': np.array(generations),
        'mean_fitness': np.array(mean_fitnesses),
        'best_fitness': np.array(best_fitnesses),
        'sigma': np.array(sigmas),
    }

    result = ExperimentResult(
        name=name,
        config=cfg,
        training_log=training_log,
        best_params=best_ever_params if best_ever_params is not None else es.best_params,
    )

    # Save to disk — auto-rename if directory already exists
    save_dir = os.path.join(results_dir, name)
    if os.path.exists(save_dir):
        suffix = 1
        while os.path.exists(f"{save_dir}_{suffix}"):
            suffix += 1
        save_dir = f"{save_dir}_{suffix}"
        print(f"[train] Directory exists; saving to {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    np.savez(
        os.path.join(save_dir, 'training_log.npz'),
        **training_log,
    )
    np.savez(
        os.path.join(save_dir, 'best_controller.npz'),
        params=result.best_params,
    )

    config_serializable = {
        k: (v.item() if hasattr(v, 'item') else v)
        for k, v in cfg.items()
        if isinstance(v, (int, float, str, bool)) or hasattr(v, 'item')
    }
    with open(os.path.join(save_dir, 'config.json'), 'w') as _f:
        json.dump(config_serializable, _f, indent=2)

    return result