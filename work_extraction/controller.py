"""Neural network controller for bond modulation (Phase 2).

LocalController: small MLP (5 -> 8 -> 8 -> 1) with tanh activations.
Weights stored as a flat numpy array for CMA-ES compatibility.
Forward pass uses JAX for GPU acceleration.

LocalMagnetisationTracker: exponential moving average of spin field.
"""

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial


def _mlp_forward(params_flat, x, layer_specs, delta_J_max):
    """Pure-function JAX MLP forward pass.

    Parameters
    ----------
    params_flat : array (n_params,)
        Flat parameter vector (JAX array).
    x : array (..., 5)
        Input features.
    layer_specs : tuple of (name, shape) pairs
        Layer specifications for unpacking.
    delta_J_max : float
        Output scaling.

    Returns
    -------
    array (..., 1) in [-delta_J_max, delta_J_max].
    """
    # Unpack params
    offset = 0
    layers = {}
    for name, shape in layer_specs:
        size = 1
        for s in shape:
            size *= s
        layers[name] = jax.lax.dynamic_slice(
            params_flat, (offset,), (size,)
        ).reshape(shape)
        offset += size

    h = jnp.tanh(x @ layers['W1'] + layers['b1'])
    h = jnp.tanh(h @ layers['W2'] + layers['b2'])
    out = jnp.tanh(h @ layers['W3'] + layers['b3'])
    return out * delta_J_max


# JIT-compile the forward pass; layer_specs and delta_J_max are static
@partial(jax.jit, static_argnums=(2, 3))
def _mlp_forward_jit(params_flat, x, layer_specs, delta_J_max):
    return _mlp_forward(params_flat, x, layer_specs, delta_J_max)


def make_layer_specs(hidden_size=8, input_size=6, output_size=1):
    """Return the MLP layer spec tuple for use with _mlp_forward.

    Centralised here so that optimiser.py, report scripts, and experiment
    scripts all derive layer_specs from one source of truth rather than
    reconstructing it manually.
    """
    return (
        ('W1', (input_size, hidden_size)),
        ('b1', (hidden_size,)),
        ('W2', (hidden_size, hidden_size)),
        ('b2', (hidden_size,)),
        ('W3', (hidden_size, output_size)),
        ('b3', (output_size,)),
    )


class LocalController:
    """Small MLP that proposes bond coupling changes.

    Architecture: 5 inputs -> 8 hidden (tanh) -> 8 hidden (tanh) -> 1 output
    Output is scaled to [-delta_J_max, delta_J_max] via tanh.

    Inputs per bond (i, j):
        [s_i, s_j, m_bar_i, T_norm, budget_norm, J_norm]
    where J_norm = tanh(J_ij / J_crit - 1), J_crit = T_mean / 2.269.

    The forward pass runs on GPU via JAX JIT. Parameters are stored as
    numpy for CMA-ES compatibility and converted to JAX on demand.
    """

    def __init__(self, delta_J_max=0.1, hidden_size=8):
        self.delta_J_max = delta_J_max
        self.hidden_size = hidden_size
        self.input_size = 6
        self.output_size = 1

        self._layer_specs = make_layer_specs(hidden_size, self.input_size, self.output_size)

        self.n_params = sum(
            int(np.prod(s)) for _, s in self._layer_specs
        )

        self._params = np.zeros(self.n_params, dtype=np.float32)
        self._params_jax = jnp.zeros(self.n_params, dtype=jnp.float32)

    def get_params(self):
        """Return flat parameter vector (numpy array)."""
        return self._params.copy()

    def set_params(self, params):
        """Set parameters from flat numpy array."""
        params = np.asarray(params, dtype=np.float32).ravel()
        assert len(params) == self.n_params, \
            f"Expected {self.n_params} params, got {len(params)}"
        self._params = params.copy()
        self._params_jax = jnp.array(params)

    def forward(self, x):
        """Forward pass through the MLP (JAX, GPU-accelerated).

        Parameters
        ----------
        x : array (..., 5)
            Input features. Accepts numpy or JAX arrays.

        Returns
        -------
        delta_J : array (..., 1) in [-delta_J_max, delta_J_max].
        """
        x_jax = jnp.asarray(x, dtype=jnp.float32)
        return _mlp_forward_jit(
            self._params_jax, x_jax,
            self._layer_specs, self.delta_J_max,
        )

    def forward_np(self, x):
        """Forward pass returning numpy (for compatibility)."""
        return np.asarray(self.forward(x))

    def propose_updates(self, s_i, s_j, m_bar, T_norm, budget_norm, J_norm):
        """Propose bond coupling changes for a set of bonds.

        All inputs are converted to JAX arrays. Returns JAX array.

        Parameters
        ----------
        s_i, s_j : array (n_bonds,)
        m_bar : array (n_bonds,)
        T_norm : float
        budget_norm : array (n_bonds,)
        J_norm : array (n_bonds,)  tanh(J_ij / J_crit - 1)

        Returns
        -------
        delta_J : JAX array (n_bonds,)
        """
        s_i        = jnp.asarray(s_i,        dtype=jnp.float32)
        s_j        = jnp.asarray(s_j,        dtype=jnp.float32)
        m_bar      = jnp.asarray(m_bar,      dtype=jnp.float32)
        budget_norm = jnp.asarray(budget_norm, dtype=jnp.float32)
        J_norm     = jnp.asarray(J_norm,     dtype=jnp.float32)

        n = s_i.shape[0]
        T_arr = jnp.full(n, T_norm, dtype=jnp.float32)

        x = jnp.stack([s_i, s_j, m_bar, T_arr, budget_norm, J_norm], axis=-1)
        return self.forward(x).ravel()


class LocalMagnetisationTracker:
    """Exponential moving average of local magnetisation.

    Tracks m_bar_i = alpha * s_i + (1 - alpha) * m_bar_i_prev
    """

    def __init__(self, n_sites, alpha=0.05):
        self.alpha = alpha
        self.n_sites = n_sites
        self._m = np.zeros(n_sites, dtype=np.float32)

    def update(self, spins):
        """Update the EMA with new spin configuration.

        Parameters
        ----------
        spins : array (N,) or (B, N)
            Spin configuration. If batched, uses the mean over batch.
        """
        s = np.asarray(spins, dtype=np.float32)
        if s.ndim == 2:
            s = s.mean(axis=0)
        self._m = self.alpha * s + (1.0 - self.alpha) * self._m

    def get(self):
        """Return current magnetisation EMA array (N,)."""
        return self._m.copy()

    def reset(self):
        self._m[:] = 0.0