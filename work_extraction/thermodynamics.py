"""Thermodynamic accounting for the Ising work extraction project.

Tracks heat (eq. 6), work (eq. 5), and entropy production (eq. 7) per step,
accumulating over full temperature cycles to yield W_extracted, Q_in, Q_out,
Sigma_cycle, W_remodel, and W_net (eq. 10).

The simulation loop uses jax.lax.scan for GPU-accelerated computation.
Post-processing of thermodynamic quantities is done on the collected traces.
"""

import numpy as np
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Temperature schedule
# ---------------------------------------------------------------------------

def temperature_schedule(t, T_mean, delta_T, tau):
    """Compute bath temperature at time step t (eq. 1).

    Works with both numpy scalars and JAX arrays.

    T(t) = T_mean + delta_T * sin(2*pi*t/tau)
    """
    return T_mean + delta_T * jnp.sin(2.0 * jnp.pi * t / tau)


def temperature_schedule_np(t, T_mean, delta_T, tau):
    """Numpy version of temperature_schedule for post-processing."""
    return T_mean + delta_T * np.sin(2.0 * np.pi * t / tau)


# ---------------------------------------------------------------------------
# JAX-accelerated cycle simulation (fixed J, no remodelling)
# ---------------------------------------------------------------------------

def _run_cycle_scan(model, key, spins, J_nk, T_mean, delta_T, tau,
                    steps_per_cycle, num_sweeps):
    """Run one cycle using lax.scan, returning per-step energy trace.

    Returns
    -------
    spins : array (B, N)
        Final spin configuration.
    energies_trace : array (steps_per_cycle,)
        Mean energy after each Metropolis step.
    key : JAX PRNGKey
    """
    # Pre-compute temperature at each step
    t_steps = jnp.arange(steps_per_cycle, dtype=jnp.float32)
    T_schedule = temperature_schedule(t_steps, T_mean, delta_T, tau)

    # Initial energy
    E_init = jnp.mean(model.energy(J_nk, spins))

    def scan_body(carry, T_t):
        spins_c, key_c = carry
        key_c, subkey = jax.random.split(key_c)

        # Energy before sweeps
        E_before = jnp.mean(model.energy(J_nk, spins_c))

        # Metropolis sweeps at temperature T_t
        spins_c, energies = model.metropolis_checkerboard_sweeps(
            subkey, spins_c, J_nk, T_t, num_sweeps
        )

        # Energy after sweeps
        E_after = jnp.mean(energies)

        # delta_Q = E_after - E_before (heat from bath at fixed J)
        delta_Q = E_after - E_before

        return (spins_c, key_c), (delta_Q, T_t, E_after)

    (spins_f, key_f), (delta_Qs, Ts, Es) = jax.lax.scan(
        scan_body, (spins, key), T_schedule
    )

    return spins_f, delta_Qs, Ts, Es, E_init, key_f


# ---------------------------------------------------------------------------
# CycleAccumulator: post-processes JAX traces into thermodynamic quantities
# ---------------------------------------------------------------------------

class CycleAccumulator:
    """Accumulates thermodynamic quantities from a cycle's energy traces.

    Can be populated either from JAX traces (from _run_cycle_scan) or
    step-by-step. All storage is numpy.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._Q_in = 0.0
        self._Q_out = 0.0
        self._sigma = 0.0
        self._W_remodel = 0.0
        self._E_start = None
        self._E_end = None

    def from_traces(self, delta_Qs, Ts, E_start, E_end):
        """Populate from JAX trace arrays (converted to numpy).

        Parameters
        ----------
        delta_Qs : array (steps,)
            Per-step heat exchange (E_after - E_before at fixed J).
        Ts : array (steps,)
            Per-step bath temperature.
        E_start : float
            Energy at cycle start.
        E_end : float
            Energy at cycle end.
        """
        delta_Qs = np.asarray(delta_Qs, dtype=np.float64)
        Ts = np.asarray(Ts, dtype=np.float64)

        self._E_start = float(E_start)
        self._E_end = float(E_end)

        # Split heat into Q_in (absorbed) and Q_out (rejected)
        self._Q_in = float(np.sum(delta_Qs[delta_Qs > 0]))
        self._Q_out = float(np.sum(np.abs(delta_Qs[delta_Qs < 0])))

        # Entropy production: sigma = sum(-delta_Q / T)
        self._sigma = float(np.sum(-delta_Qs / Ts))

    def add_remodel_cost(self, physical_work, basal_cost):
        """Add remodelling costs (from bond changes)."""
        self._W_remodel += float(physical_work) + float(basal_cost)

    def step(self, delta_E_heat, T, remodel_work=0.0, remodel_basal_cost=0.0):
        """Record one simulation step (for step-by-step accumulation)."""
        delta_E_heat = float(delta_E_heat)
        T = float(T)

        if delta_E_heat > 0:
            self._Q_in += delta_E_heat
        else:
            self._Q_out += abs(delta_E_heat)

        self._sigma += -delta_E_heat / T
        self._W_remodel += float(remodel_work) + float(remodel_basal_cost)

    def step_remodel(self, delta_E_remodel, basal_cost):
        """Record energy change from a bond remodelling event."""
        self._W_remodel += abs(float(delta_E_remodel)) + float(basal_cost)
        if self._E_end is not None:
            self._E_end += float(delta_E_remodel)

    def set_initial_energy(self, E):
        self._E_start = float(E)

    def set_final_energy(self, E):
        self._E_end = float(E)

    @property
    def Q_in(self):
        return self._Q_in

    @property
    def Q_out(self):
        return self._Q_out

    @property
    def Sigma_cycle(self):
        return self._sigma

    @property
    def W_remodel(self):
        return self._W_remodel

    @property
    def W_extracted(self):
        """W_extracted = Q_out - Q_in = -(Q_in - Q_out) per spec."""
        return self._Q_out - self._Q_in

    @property
    def W_net(self):
        """W_net = W_extracted - W_remodel (eq. 10)."""
        return self.W_extracted - self._W_remodel

    @property
    def delta_U(self):
        if self._E_start is None or self._E_end is None:
            return None
        return self._E_end - self._E_start

    @property
    def efficiency(self):
        """Thermodynamic efficiency: W_net / Q_in."""
        if self._Q_in == 0:
            return 0.0
        return self.W_net / self._Q_in

    def carnot_check(self, T_cold, T_hot):
        eta_carnot = 1.0 - T_cold / T_hot
        return self.efficiency <= eta_carnot + 1e-12


# ---------------------------------------------------------------------------
# High-level cycle runner
# ---------------------------------------------------------------------------

def run_cycle_with_accounting(model, key, spins, J_nk, T_mean, delta_T, tau,
                               steps_per_cycle, num_sweeps=1):
    """Run one full temperature cycle with thermodynamic accounting.

    Uses jax.lax.scan for the simulation loop, then post-processes
    the energy traces into thermodynamic quantities.

    Parameters
    ----------
    model : IsingModel
    key : JAX PRNGKey
    spins : array (B, N)
    J_nk : array (N, K)
    T_mean, delta_T, tau : float
    steps_per_cycle : int
    num_sweeps : int

    Returns
    -------
    spins : array (B, N) — final spin configuration
    acc : CycleAccumulator — thermodynamic quantities for the cycle
    key : JAX PRNGKey — updated key
    """
    spins_f, delta_Qs, Ts, Es, E_init, key_f = _run_cycle_scan(
        model, key, spins, J_nk, T_mean, delta_T, tau,
        steps_per_cycle, num_sweeps
    )

    acc = CycleAccumulator()
    acc.from_traces(delta_Qs, Ts, float(E_init), float(Es[-1]))

    return spins_f, acc, key_f


def run_multiple_cycles(model, key, spins, J_nk, T_mean, delta_T, tau,
                         steps_per_cycle, num_cycles, num_sweeps=1):
    """Run multiple cycles, returning per-cycle thermodynamic quantities.

    Returns
    -------
    spins : final spin configuration
    results : dict with arrays of per-cycle quantities
    key : updated key
    """
    W_net_list = []
    W_ext_list = []
    Q_in_list = []
    Q_out_list = []
    sigma_list = []

    for c in range(num_cycles):
        spins, acc, key = run_cycle_with_accounting(
            model, key, spins, J_nk, T_mean, delta_T, tau,
            steps_per_cycle, num_sweeps
        )
        W_net_list.append(acc.W_net)
        W_ext_list.append(acc.W_extracted)
        Q_in_list.append(acc.Q_in)
        Q_out_list.append(acc.Q_out)
        sigma_list.append(acc.Sigma_cycle)

    results = {
        'W_net': np.array(W_net_list),
        'W_extracted': np.array(W_ext_list),
        'Q_in': np.array(Q_in_list),
        'Q_out': np.array(Q_out_list),
        'Sigma_cycle': np.array(sigma_list),
    }
    return spins, results, key


# ---------------------------------------------------------------------------
# Remodelling work computation (JAX-accelerated)
# ---------------------------------------------------------------------------

def compute_remodel_work(model, spins, J_old, J_new, lam=0.0):
    """Compute work and cost from bond remodelling.

    Parameters
    ----------
    model : IsingModel
    spins : array (B, N)
    J_old, J_new : array (N, K)
    lam : float — basal cost coefficient

    Returns
    -------
    delta_E : float — energy change E(J_new, s) - E(J_old, s)
    basal_cost : float — lambda * sum(|J_new - J_old|)
    """
    E_old = jnp.mean(model.energy(J_old, spins))
    E_new = jnp.mean(model.energy(J_new, spins))
    delta_E = float(E_new - E_old)

    delta_J = J_new - J_old
    basal_cost = float(lam * jnp.sum(jnp.abs(delta_J)))

    return delta_E, basal_cost