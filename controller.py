"""Neural network controller for bond modulation (Phase 2).

LocalController: small MLP (5 -> 8 -> 8 -> 1) with tanh activations.
Weights stored as a flat numpy array for CMA-ES compatibility.

LocalMagnetisationTracker: exponential moving average of spin field.
"""

import numpy as np


class LocalController:
    """Small MLP that proposes bond coupling changes.

    Architecture: 5 inputs -> 8 hidden (tanh) -> 8 hidden (tanh) -> 1 output
    Output is scaled to [-delta_J_max, delta_J_max] via tanh.

    Inputs per bond (i, j):
        [s_i, s_j, m_bar_i, T_norm, budget_norm]
    where:
        s_i, s_j: spin values (-1 or +1)
        m_bar_i: local magnetisation EMA at site i
        T_norm: (T - T_mean) / delta_T (normalised temperature)
        budget_norm: tanh(B / B_scale) (normalised budget)

    Parameters
    ----------
    delta_J_max : float
        Maximum coupling change magnitude.
    hidden_size : int
        Hidden layer width (default 8).
    """

    def __init__(self, delta_J_max=0.1, hidden_size=8):
        self.delta_J_max = delta_J_max
        self.hidden_size = hidden_size
        self.input_size = 5
        self.output_size = 1

        # Layer shapes
        self._shapes = [
            ('W1', (self.input_size, hidden_size)),
            ('b1', (hidden_size,)),
            ('W2', (hidden_size, hidden_size)),
            ('b2', (hidden_size,)),
            ('W3', (hidden_size, self.output_size)),
            ('b3', (self.output_size,)),
        ]

        # Total parameter count
        self.n_params = sum(np.prod(s) for _, s in self._shapes)

        # Initialize with small random weights
        self._params = np.zeros(self.n_params, dtype=np.float32)

    def get_params(self):
        """Return flat parameter vector (numpy array)."""
        return self._params.copy()

    def set_params(self, params):
        """Set parameters from flat numpy array."""
        params = np.asarray(params, dtype=np.float32).ravel()
        assert len(params) == self.n_params, \
            f"Expected {self.n_params} params, got {len(params)}"
        self._params = params.copy()

    def _unpack(self):
        """Unpack flat params into weight matrices and biases."""
        layers = {}
        offset = 0
        for name, shape in self._shapes:
            size = int(np.prod(shape))
            layers[name] = self._params[offset:offset + size].reshape(shape)
            offset += size
        return layers

    def forward(self, x):
        """Forward pass through the MLP.

        Parameters
        ----------
        x : array (..., 5)
            Input features. Can be batched.

        Returns
        -------
        delta_J : array (..., 1)
            Proposed coupling changes in [-delta_J_max, delta_J_max].
        """
        layers = self._unpack()

        # Layer 1
        h = np.tanh(x @ layers['W1'] + layers['b1'])
        # Layer 2
        h = np.tanh(h @ layers['W2'] + layers['b2'])
        # Output layer
        out = np.tanh(h @ layers['W3'] + layers['b3'])

        return out * self.delta_J_max

    def propose_updates(self, s_i, s_j, m_bar, T_norm, budget_norm):
        """Propose bond coupling changes for a set of bonds.

        Parameters
        ----------
        s_i, s_j : array (n_bonds,)
            Spin values at bond endpoints.
        m_bar : array (n_bonds,)
            Local magnetisation EMA at site i of each bond.
        T_norm : float
            Normalised temperature (T - T_mean) / delta_T.
        budget_norm : array (n_bonds,)
            Normalised budget tanh(B / B_scale) per bond.

        Returns
        -------
        delta_J : array (n_bonds,)
            Proposed coupling changes.
        """
        s_i = np.asarray(s_i, dtype=np.float32)
        s_j = np.asarray(s_j, dtype=np.float32)
        m_bar = np.asarray(m_bar, dtype=np.float32)
        budget_norm = np.asarray(budget_norm, dtype=np.float32)

        n = len(s_i)
        T_arr = np.full(n, T_norm, dtype=np.float32)

        x = np.stack([s_i, s_j, m_bar, T_arr, budget_norm], axis=-1)
        return self.forward(x).ravel()


class LocalMagnetisationTracker:
    """Exponential moving average of local magnetisation.

    Tracks m_bar_i = alpha * s_i + (1 - alpha) * m_bar_i_prev
    for each site on the lattice.

    Parameters
    ----------
    n_sites : int
        Number of sites on the lattice.
    alpha : float
        EMA decay rate. Higher = faster adaptation.
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
        """Reset all magnetisation values to zero."""
        self._m[:] = 0.0