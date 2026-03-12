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

from .thermodynamics import temperature_schedule
from .controller import LocalController
from .budgets import NoBudget, BondBudget, NeighbourhoodBudget, DiffusingBudget
from .optimiser import WorkExtractionES, make_jax_eval_fn


DEFAULT_CONFIG = {
    'L': 32,
    'T_mean': 2.5,
    'delta_T': 1.5,
    'tau': 200,
    'J_init': 1.1,
    'J_min': 0.01,
    'J_max': 5.0,
    'n_generations': 500,
    'n_eval_cycles': 10,
    'n_eval_chains': 5,
    'steps_per_cycle': 200,
    'bond_update_frac': 0.1,
    'delta_J_max': 0.1,
    'B_scale': 2.0,
    'lambda': 0.05,
    'pop_size': 20,
    'sigma': 0.02,
    'elite_frac': 0.2,
    'sigma_decay': 0.995,
    'hidden_size': 8,
    'mag_ema_alpha': 0.05,
    'log_interval': 10,
    'neighborhood': 'von_neumann',
    'boundary': 'periodic',
    'num_sweeps': 1,
    'warmup_sweeps': 500,
}


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
    alpha = config.get('budget_alpha', config.get('mag_ema_alpha', 0.05))

    if budget_type == 'none':
        return NoBudget()
    elif budget_type == 'bond':
        return BondBudget(neighbors, mask, alpha=alpha)
    elif budget_type == 'neighbourhood':
        gamma = config.get('gamma', 0.25)
        return NeighbourhoodBudget(neighbors, mask, alpha=alpha, gamma=gamma)
    elif budget_type == 'diffusing':
        D = config.get('D', 0.1)
        tau_mu = config.get('tau_mu', 20.0)
        return DiffusingBudget(neighbors, mask, alpha=alpha, D=D, tau_mu=tau_mu)
    else:
        raise ValueError(f"Unknown budget type: {budget_type}")


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
    # If n_eval_chains > 1, average fitness over that many independent spin-chain
    # runs per candidate.  The inner vmap over chains and the outer vmap over the
    # population are fused by JAX into a single kernel of pop_size × n_eval_chains
    # concurrent evaluations.
    n_eval_chains = int(cfg.get('n_eval_chains', 1))
    eval_fn_base = make_jax_eval_fn(model, cfg, budget_type)

    if n_eval_chains > 1:
        def eval_fn(params_flat, key):
            chain_keys = jax.random.split(key, n_eval_chains)
            return jnp.mean(jax.vmap(lambda k: eval_fn_base(params_flat, k))(chain_keys))
    else:
        eval_fn = eval_fn_base

    eval_batch = jax.jit(jax.vmap(eval_fn))

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

        # Evaluate entire population in parallel via vmap+jit
        params_batch = jnp.array(np.stack(params_list))
        master_key, *eval_keys = jax.random.split(master_key, cfg['pop_size'] + 1)
        keys_batch = jnp.stack(eval_keys)
        fitnesses_jax = eval_batch(params_batch, keys_batch)
        fitnesses = list(np.asarray(fitnesses_jax))

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

    # Save to disk
    save_dir = os.path.join(results_dir, name)
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