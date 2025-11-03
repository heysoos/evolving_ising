from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple

# Make evosax optional to allow running Ising tests without it
try:
	from evosax.algorithms import CMA_ES
	import evosax as _ex  # type: ignore
	_HAS_EVO = True
except Exception:  # optional dependency
	CMA_ES = None  # type: ignore
	_ex = None  # type: ignore
	_HAS_EVO = False

import jax
import jax.numpy as jnp
from jax import lax

Array = jnp.ndarray


class IsingModel:
	"""
	GPU-friendly Ising model on a 2D grid with nearest-neighbor connectivity only.
	- Spins s in {-1, +1} of size N = H*W
	- Connectivity stored as per-site per-neighbor weights J_nk with K neighbors (K<=8)
	- Neighborhood uses Moore (8-neighbor) or Von Neumann (4-neighbor) stencil
	- Grouped (checkerboard/colored) Metropolis-Hastings updates for GPU efficiency
	- Supports per-site local temperatures T (vector) or a global scalar T
	"""

	def __init__(self, grid_hw: Tuple[int, int], neighborhood: str = "moore", boundary: str = "open"):
		self.h, self.w = grid_hw
		self.n = self.h * self.w
		assert neighborhood in {"moore", "von_neumann"}, "neighborhood must be 'moore' or 'von_neumann'"
		# Support per-axis wrap modes
		assert boundary in {"open", "periodic", "periodic_lr", "periodic_tb"}, (
			"boundary must be one of 'open', 'periodic', 'periodic_lr', 'periodic_tb'"
		)
		self.neighborhood = neighborhood
		self.boundary = boundary

		# Define neighbor offsets and fixed slot order
		offsets = None
		if neighborhood == "moore":
			# 8-neighborhood (dx, dy) in a fixed order
			offsets = [
				(-1, -1), (-1, 0), (-1, 1),
				(0, -1),            (0, 1),
				(1, -1),  (1, 0),  (1, 1),
			]
		elif neighborhood == "von_neumann":
			# 4-neighborhood
			offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]
		assert offsets is not None
		self.K = len(offsets)
		self._offsets = tuple(offsets)
		# Map offset -> slot index for fast reverse lookup
		offset_to_slot = {off: i for i, off in enumerate(offsets)}
		self._rev_slot_map = {off: offset_to_slot.get((-off[0], -off[1]), -1) for off in offsets}

		# Build neighbor index table and reverse slot table
		def idx(i, j):
			return i * self.w + j

		# Determine wrapping per axis
		wrap_h = self.boundary in ("periodic", "periodic_lr")  # left-right wrap
		wrap_v = self.boundary in ("periodic", "periodic_tb")  # top-bottom wrap
		self._wrap_h = wrap_h
		self._wrap_v = wrap_v

		rows = []
		masks = []
		rev_slots = []
		for i in range(self.h):
			for j in range(self.w):
				row_neigh = []
				row_mask = []
				row_rev = []
				for off in offsets:
					ni = i + off[0]
					nj = j + off[1]
					# Apply wrapping per axis if enabled
					ni_wrapped = (ni % self.h) if wrap_v else ni
					nj_wrapped = (nj % self.w) if wrap_h else nj
					valid = True
					if not wrap_v and (ni < 0 or ni >= self.h):
						valid = False
					if not wrap_h and (nj < 0 or nj >= self.w):
						valid = False
					if valid:
						v = idx(ni_wrapped, nj_wrapped)
						row_neigh.append(v)
						row_mask.append(True)
						rev = self._rev_slot_map[off]
						row_rev.append(rev if rev >= 0 else 0)
					else:
						row_neigh.append(0)
						row_mask.append(False)
						row_rev.append(0)
				rows.append(row_neigh)
				masks.append(row_mask)
				rev_slots.append(row_rev)

		neighbors = jnp.array(rows, dtype=jnp.int32)
		mask = jnp.array(masks, dtype=bool)
		rev_slot = jnp.array(rev_slots, dtype=jnp.int32)

		self.neighbors = neighbors
		self.mask = mask
		self.rev_slot = rev_slot

		# Precompute color masks for parallel updates (JAX arrays to avoid Python indexing in JIT)
		rows_idx = jnp.arange(self.n, dtype=jnp.int32) // self.w
		cols_idx = jnp.arange(self.n, dtype=jnp.int32) % self.w
		if neighborhood == "von_neumann":
			# 2-coloring (checkerboard)
			cid = (rows_idx + cols_idx) & 1  # 0/1
			m0 = (cid == 0)
			m1 = ~m0
			self._color_masks = jnp.stack([m0, m1], axis=0)  # (2, N)
			self._num_colors = 2
		else:
			# 4-coloring for Moore (since diagonals connect same parity)
			cid = (rows_idx & 1) * 2 + (cols_idx & 1)  # 0..3
			masks4 = [cid == g for g in range(4)]
			self._color_masks = jnp.stack(masks4, axis=0)  # (4, N)
			self._num_colors = 4

	def init_spins(self, key: Array, batch_size: int) -> Array:
		s = jax.random.bernoulli(key, 0.5, (batch_size, self.n))
		return s.astype(jnp.int8) * 2 - 1  # {-1, +1}

	@partial(jax.jit, static_argnums=(0,))
	def energy(self, J_nk: Array, spins: Array) -> Array:
		"""
		Compute E = -0.5 * sum_i s_i * h_i, where h_i = sum_k J[i,k] * s[neighbors[i,k]].
		- spins: (B, N) int8 or float32
		- J_nk: (N, K) float32, weights per-site per-neighbor. Invalid neighbors weights must be 0.
		"""
		B, N = spins.shape
		spins_f = spins.astype(jnp.float32)
		# Gather neighbor spins: (B, N, K)
		nbr_spins = spins_f[:, self.neighbors]
		J_masked = J_nk * self.mask.astype(J_nk.dtype)
		h = jnp.sum(nbr_spins * J_masked[None, :, :], axis=2)  # (B, N)
		return -0.5 * jnp.einsum("bi,bi->b", spins_f, h)

	@partial(jax.jit, static_argnums=(0,))
	def _compute_field(self, J_nk: Array, spins: Array) -> Array:
		spins_f = spins.astype(jnp.float32)
		nbr_spins = spins_f[:, self.neighbors]
		J_masked = J_nk * self.mask.astype(J_nk.dtype)
		return jnp.sum(nbr_spins * J_masked[None, :, :], axis=2)  # (B, N)

	# Deprecated single-spin sweep APIs now alias to grouped checkerboard for speed
	@partial(jax.jit, static_argnums=(0,5))
	def metropolis_sweeps(
		self,
		key: Array,
		spins: Array,      # (B, N) int8
		J_nk: Array,       # (N, K) float32
		T: Array,          # scalar or (N,) or (B, N)
		num_sweeps: int = 1,
	) -> Tuple[Array, Array]:
		"""Alias to metropolis_checkerboard_sweeps (grouped updates)."""
		return self.metropolis_checkerboard_sweeps(key, spins, J_nk, T, num_sweeps)

	@partial(jax.jit, static_argnums=(0,5))
	def metropolis_sweeps_with_history(
		self,
		key: Array,
		spins: Array,
		J_nk: Array,
		T: Array,
		num_sweeps: int = 1,
	):
		"""Alias to metropolis_checkerboard_sweeps_with_history (grouped updates)."""
		return self.metropolis_checkerboard_sweeps_with_history(key, spins, J_nk, T, num_sweeps)

	# New: masked, parallel checkerboard updates without Python indexing in JIT
	def _metropolis_update_masked_broadcasted(self, key: Array, spins: Array, J_nk: Array, T_b: Array, site_mask: Array) -> Array:
		"""Vectorized Metropolis update over all sites but only flips where site_mask is True.
		spins: (B,N) int8; J_nk: (N,K); T_b: (B,N); site_mask: (N,) bool
		Returns updated spins with dtype int8.
		"""
		B, N = spins.shape
		spins_f = spins.astype(jnp.float32)
		# Local fields for current configuration
		h = self._compute_field(J_nk, spins)  # (B, N)
		dE = 2.0 * spins_f * h
		T_b = jnp.maximum(T_b, jnp.asarray(1e-8, dtype=T_b.dtype))
		accept_prob = jnp.exp(-jnp.clip(dE / T_b, a_min=-50.0, a_max=50.0))
		u = jax.random.uniform(key, shape=(B, N), dtype=spins_f.dtype)
		accept = (dE <= 0.0) | (u < accept_prob)
		accept = accept & jnp.broadcast_to(site_mask[None, :], (B, N))
		spins_new = jnp.where(accept, -spins_f, spins_f).astype(spins.dtype)
		return spins_new

	@partial(jax.jit, static_argnums=(0,5))
	def metropolis_checkerboard_sweeps(
		self,
		key: Array,
		spins: Array,
		J_nk: Array,
		T: Array,
		num_sweeps: int = 1,
	) -> Tuple[Array, Array]:
		"""Parallel checkerboard Metropolis using color masks stored as JAX arrays.
		Updates all sites of one color in parallel, sequentially across colors per sweep.
		Returns (new_spins, energies_per_chain).
		"""
		B, N = spins.shape
		spins = spins.astype(jnp.int8)
		J_nk = J_nk.astype(jnp.float32)

		# Broadcast temperature to (B, N)
		if T.ndim == 0:
			T_b = jnp.full((B, N), T, dtype=J_nk.dtype)
		elif T.ndim == 1:
			T_b = jnp.broadcast_to(T[None, :], (B, N))
		else:
			T_b = T

		color_masks = self._color_masks  # (G,N)
		G = self._num_colors

		def one_sweep(spins_c, key_sweep):
			# Split per-color keys
			keys_g = jax.random.split(key_sweep, G)

			def apply_color(g, s):
				mask_g = color_masks[g]
				return self._metropolis_update_masked_broadcasted(keys_g[g], s, J_nk, T_b, mask_g)

			spins_c = jax.lax.fori_loop(0, G, apply_color, spins_c)
			return spins_c

		keys_sweeps = jax.random.split(key, num_sweeps)
		spins = jax.lax.fori_loop(0, num_sweeps, lambda t, s: one_sweep(s, keys_sweeps[t]), spins)
		E = self.energy(J_nk, spins)
		return spins, E

	@partial(jax.jit, static_argnums=(0,5))
	def metropolis_checkerboard_sweeps_with_history(
		self,
		key: Array,
		spins: Array,
		J_nk: Array,
		T: Array,
		num_sweeps: int = 1,
	):
		"""Same as metropolis_checkerboard_sweeps but records spins after each sweep.
		Returns (spins, E, history) with history shape (num_sweeps, B, N)."""
		B, N = spins.shape
		spins = spins.astype(jnp.int8)
		J_nk = J_nk.astype(jnp.float32)

		# Broadcast temperature to (B, N)
		if T.ndim == 0:
			T_b = jnp.full((B, N), T, dtype=J_nk.dtype)
		elif T.ndim == 1:
			T_b = jnp.broadcast_to(T[None, :], (B, N))
		else:
			T_b = T

		color_masks = self._color_masks  # (G,N)
		G = self._num_colors

		def one_sweep(carry, key_sweep):
			spins_c = carry
			keys_g = jax.random.split(key_sweep, G)

			def apply_color(g, s):
				mask_g = color_masks[g]
				return self._metropolis_update_masked_broadcasted(keys_g[g], s, J_nk, T_b, mask_g)

			spins_c = jax.lax.fori_loop(0, G, apply_color, spins_c)
			return spins_c, spins_c

		keys_sweeps = jax.random.split(key, num_sweeps)
		spins_f, hist = jax.lax.scan(one_sweep, spins, keys_sweeps)
		E = self.energy(J_nk, spins_f)
		return spins_f, E, hist

	def wrap_flags(self) -> Tuple[bool, bool]:
		"""Returns (wrap_h, wrap_v) for debugging boundary conditions."""
		return self._wrap_h, self._wrap_v

	def vertical_edge_masks(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
		"""Return boolean masks (has_up, has_down) of shape (N,) indicating if each site has a valid vertical neighbor.
		Only meaningful for von_neumann where vertical offsets exist; for moore includes any vertical among 8."""
		up_idx = -1
		down_idx = -1
		for k, off in enumerate(self._offsets):
			if off == (-1, 0):
				up_idx = k
			elif off == (1, 0):
				down_idx = k
		# Fallback for moore: any neighbor with ni != i (vertical component)
		if up_idx >= 0 and down_idx >= 0:
			has_up = self.mask[:, up_idx]
			has_down = self.mask[:, down_idx]
		else:
			# derive by comparing neighbor indices row positions
			rows = jnp.arange(self.n, dtype=jnp.int32) // self.w
			nbr_rows = self.neighbors // self.w
			vert = jnp.any((nbr_rows != rows[:, None]) & self.mask, axis=1)
			# approximate split into up/down by sign (can't distinguish easily here); return same mask
			has_up = vert
			has_down = vert
		return has_up, has_down


class TemperatureDiffuser:
	"""
	Diffuses local temperatures over the sparse neighbor graph induced by (neighbors, J_nk).
	- Conductance W is derived from J_nk via conductance_mode (abs/relu/softplus/square/sigmoid).
	- Normalization can be row-normalization or none.
	- T_{t+1} = (1 - alpha) * T_t + alpha * (W_norm @ T_neighbors)
	- Supports per-batch T and pinned boundary mask.
	"""

	def __init__(self, alpha: float = 0.5, use_abs: Optional[bool] = None, conductance_mode: str = "abs",
				 normalize_mode: str = "row", eps: float = 1e-8):
		# Back-compat: use_abs overrides conductance_mode if provided
		if use_abs is not None:
			conductance_mode = "abs" if use_abs else "relu"
		self.alpha = alpha
		self._c_mode = conductance_mode
		self._n_mode = normalize_mode
		self.eps = eps

	@partial(jax.jit, static_argnums=(0,))
	def _conductance(self, J_nk: jnp.ndarray) -> jnp.ndarray:
		mode = self._c_mode
		if mode == "abs":
			return jnp.abs(J_nk)
		elif mode == "relu":
			return jnp.maximum(J_nk, 0.0)
		elif mode == "softplus":
			return jax.nn.softplus(J_nk)
		elif mode == "square":
			return J_nk * J_nk
		elif mode == "sigmoid":
			return jax.nn.sigmoid(J_nk)
		else:
			# default safe
			return jnp.abs(J_nk)

	@partial(jax.jit, static_argnums=(0,))
	def _normalize(self, W: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
		if self._n_mode == "none":
			return W * mask.astype(W.dtype)
		# default row-norm
		Wm = W * mask.astype(W.dtype)
		deg = jnp.sum(Wm, axis=1, keepdims=True) + self.eps
		return Wm / deg

	@partial(jax.jit, static_argnums=(0,))
	def step(
		self,
		neighbors: jnp.ndarray,  # (N, K)
		J_nk: jnp.ndarray,        # (N, K)
		mask: jnp.ndarray,        # (N, K)
		T: jnp.ndarray,           # (B, N) or (N,)
		pin_mask: Optional[jnp.ndarray] = None,  # (N,), True=pinned
		pin_values: Optional[jnp.ndarray] = None # (N,), values for pinned sites
	) -> jnp.ndarray:
		W = self._conductance(J_nk)
		Wn = self._normalize(W, mask)
		# Broadcast T to (B, N)
		if T.ndim == 1:
			T = T[None, :]
		# Gather neighbor temperatures (B, N, K)
		nbr_T = T[:, neighbors]
		T_flow = jnp.sum(nbr_T * Wn[None, :, :], axis=2)  # (B, N)
		T_new = (1.0 - self.alpha) * T + self.alpha * T_flow
		if pin_mask is not None and pin_values is not None:
			pin_mask_b = jnp.broadcast_to(pin_mask[None, :], T_new.shape)
			pin_vals_b = jnp.broadcast_to(pin_values[None, :], T_new.shape)
			T_new = jnp.where(pin_mask_b, pin_vals_b, T_new)
		return T_new

	@partial(jax.jit, static_argnums=(0,5))
	def diffuse(
		self,
		neighbors: jnp.ndarray,
		J_nk: jnp.ndarray,
		mask: jnp.ndarray,
		T0: jnp.ndarray,
		steps: int,
		pin_mask: Optional[jnp.ndarray] = None,
		pin_values: Optional[jnp.ndarray] = None,
	) -> jnp.ndarray:
		def body(T, _):
			return self.step(neighbors, J_nk, mask, T, pin_mask, pin_values), None
		T, _ = lax.scan(body, T0, None, length=steps)
		return T


@dataclass
class EvoConfig:
	pop_size: int = 32
	iters: int = 200
	sigma_init: float = 0.3
	j_scale: float = 1.0
	j_transform: str = "softplus"        # mapping from theta -> positive J
	warmup_sweeps: int = 10
	measure_sweeps: int = 10
	diffusion_steps_per_sweep: int = 2
	alpha_diffusion: float = 0.5
	conductance_mode: str = "abs"        # abs/relu/softplus/square/sigmoid
	normalize_mode: str = "row"          # row/none
	chains_per_eval: int = 64
	seed: int = 0

# --- Minimal separable CMA-ES, array-only ---
@dataclass
class CMAState:
	mean: Array  # (D,)
	sigma: Array  # ()
	diagC: Array  # (D,)
	pc: Array     # (D,)
	ps: Array     # (D,)
	rng: Array    # PRNGKey

class SeparableCMAES:
	def __init__(self, dim: int, pop_size: int, sigma_init: float, seed: int = 0):
		self.dim = dim
		self.lam = pop_size
		self.mu = max(1, pop_size // 2)
		i = jnp.arange(self.mu)
		w = jnp.log(self.mu + 0.5) - jnp.log(i + 1.0)
		w = w / jnp.sum(w)
		self.w = w
		self.mu_eff = 1.0 / jnp.sum(w**2)
		D = float(dim)
		self.cc = (4 + self.mu_eff / D) / (D + 4 + 2 * self.mu_eff / D)
		self.cs = (self.mu_eff + 2.0) / (D + self.mu_eff + 5.0)
		self.c1 = 2.0 / ((D + 1.3) ** 2 + self.mu_eff)
		self.cmu = jnp.minimum(1.0 - self.c1, 2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / ((D + 2.0) ** 2 + self.mu_eff))
		self.damps = 1.0 + 2.0 * jnp.maximum(0.0, jnp.sqrt((self.mu_eff - 1.0) / (D + 1.0)) - 1.0) + self.cs
		self.E_norm = jnp.sqrt(D) * (1.0 - 1.0 / (4.0 * D) + 1.0 / (21.0 * D * D))
		key = jax.random.PRNGKey(seed)
		self.state = CMAState(
			mean=jnp.zeros((dim,), dtype=jnp.float32),
			sigma=jnp.asarray(sigma_init, dtype=jnp.float32),
			diagC=jnp.ones((dim,), dtype=jnp.float32),
			pc=jnp.zeros((dim,), dtype=jnp.float32),
			ps=jnp.zeros((dim,), dtype=jnp.float32),
			rng=key,
		)

	def ask(self) -> Array:
		key, sub = jax.random.split(self.state.rng)
		z = jax.random.normal(sub, (self.lam, self.dim), dtype=jnp.float32)
		y = z * jnp.sqrt(self.state.diagC)[None, :]
		X = self.state.mean[None, :] + self.state.sigma * y
		self.state = CMAState(mean=self.state.mean, sigma=self.state.sigma, diagC=self.state.diagC, pc=self.state.pc, ps=self.state.ps, rng=key)
		return X

	def tell(self, X: Array, fitness: Array) -> None:
		idx = jnp.argsort(-fitness)
		X_sel = X[idx[: self.mu]]
		w = self.w[:, None]
		mean_old = self.state.mean
		mean_new = jnp.sum(w * X_sel, axis=0)
		y = (X_sel - mean_old[None, :]) / jnp.maximum(self.state.sigma, 1e-8)
		z = y / jnp.sqrt(self.state.diagC)[None, :]
		z_w = jnp.sum(self.w[:, None] * z, axis=0)
		ps = (1 - self.cs) * self.state.ps + jnp.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * z_w
		norm_ps = jnp.linalg.norm(ps)
		h_sigma = (norm_ps / jnp.sqrt(1 - (1 - self.cs) ** 2) < (1.4 + 2.0 / (self.dim + 1.0)) * self.E_norm)
		h_sigma = h_sigma.astype(jnp.float32)
		pc = (1 - self.cc) * self.state.pc + h_sigma * jnp.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * jnp.sum(self.w[:, None] * y, axis=0)
		diagC = (1 - self.c1 - self.cmu) * self.state.diagC + self.c1 * (pc ** 2) + self.cmu * jnp.sum(self.w[:, None] * (z ** 2), axis=0)
		sigma = self.state.sigma * jnp.exp((self.cs / self.damps) * (norm_ps / self.E_norm - 1.0))
		self.state = CMAState(mean=mean_new, sigma=sigma, diagC=diagC, pc=pc, ps=ps, rng=self.state.rng)

# --- EvoRunner using built-in CMA-ES ---
class EvoRunner:
	"""Optimize J_nk by sampling flat parameter arrays and evaluating Ising energy."""
	def __init__(self, ising: 'IsingModel', pin_mask: jnp.ndarray, pin_values: jnp.ndarray, cfg: EvoConfig):
		self.ising = ising
		self.cfg = cfg
		self.pin_mask = pin_mask.astype(bool)
		self.pin_values = pin_values.astype(jnp.float32)
		mask = ising.mask
		self.N = ising.n
		self.K = ising.K
		self.flat_idx = jnp.where(mask.reshape(-1))[0]
		self.D = int(self.flat_idx.shape[0])
		self.es = SeparableCMAES(dim=self.D, pop_size=cfg.pop_size, sigma_init=cfg.sigma_init, seed=cfg.seed)
		self._diffuser = TemperatureDiffuser(alpha=cfg.alpha_diffusion, conductance_mode=cfg.conductance_mode, normalize_mode=cfg.normalize_mode)

	def _theta_to_Jpos(self, theta: jnp.ndarray) -> jnp.ndarray:
		m = self.cfg.j_transform
		if m == "softplus":
			Jvals = jax.nn.softplus(theta)
		elif m == "sigmoid":
			Jvals = jax.nn.sigmoid(theta)
		elif m == "relu":
			Jvals = jnp.maximum(theta, 0.0)
		elif m == "tanh01":
			Jvals = 0.5 * (jnp.tanh(theta) + 1.0)
		else:
			Jvals = jax.nn.softplus(theta)
		return Jvals * self.cfg.j_scale

	def vec_to_Jnk(self, theta: jnp.ndarray) -> jnp.ndarray:
		J = jnp.zeros((self.N * self.K,), dtype=jnp.float32)
		J = J.at[self.flat_idx].set(self._theta_to_Jpos(theta))
		return J.reshape(self.N, self.K)

	def _diffuse(self, J_nk: jnp.ndarray, T_b: jnp.ndarray) -> jnp.ndarray:
		return self._diffuser.diffuse(self.ising.neighbors, J_nk, self.ising.mask, T_b,
									  steps=self.cfg.diffusion_steps_per_sweep,
									  pin_mask=self.pin_mask, pin_values=self.pin_values)

	def _evaluate_single(self, key: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
		J_nk = self.vec_to_Jnk(theta)
		B, N = self.cfg.chains_per_eval, self.N
		key_s, key_perm = jax.random.split(key)
		spins0 = self.ising.init_spins(key_s, B)
		T0 = jnp.full((B, N), jnp.mean(self.pin_values), dtype=jnp.float32)
		T0 = jnp.where(jnp.broadcast_to(self.pin_mask[None, :], T0.shape), jnp.broadcast_to(self.pin_values[None, :], T0.shape), T0)
		def do_sweep(carry, _):
			spins_c, T_c, key_c = carry
			T_c = self._diffuse(J_nk, T_c)
			key_c, sub = jax.random.split(key_c)
			spins_c, _ = self.ising.metropolis_checkerboard_sweeps(sub, spins_c, J_nk, T_c, num_sweeps=1)
			return (spins_c, T_c, key_c), None
		(spins_w, T_w, _), _ = jax.lax.scan(do_sweep, (spins0, T0, key_perm), None, length=self.cfg.warmup_sweeps)
		def measure(carry, _):
			spins_c, T_c, key_c, e_sum = carry
			(spins_c, T_c, key_c), _ = jax.lax.scan(do_sweep, (spins_c, T_c, key_c), None, length=1)
			E = self.ising.energy(J_nk, spins_c)
			e_sum = e_sum + jnp.mean(E)
			return (spins_c, T_c, key_c, e_sum), None
		(_, _, _, e_acc), _ = jax.lax.scan(measure, (spins_w, T_w, key_perm, 0.0), None, length=self.cfg.measure_sweeps)
		return -(e_acc / float(self.cfg.measure_sweeps))

	def _batched_evaluate(self, key: jnp.ndarray, thetas: jnp.ndarray) -> jnp.ndarray:
		keys = jax.random.split(key, thetas.shape[0])
		return jax.jit(jax.vmap(self._evaluate_single, in_axes=(0, 0)))(keys, thetas)

	def run(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
		key = jax.random.PRNGKey(self.cfg.seed)
		best_theta = None
		best_fit = -jnp.inf
		for _ in range(self.cfg.iters):
			X = self.es.ask()
			key, sub = jax.random.split(key)
			fitness = self._batched_evaluate(sub, X)
			self.es.tell(X, fitness)
			idx = int(jnp.argmax(fitness))
			if (best_theta is None) or (fitness[idx] > best_fit):
				best_theta = X[idx]
				best_fit = fitness[idx]
		return self.vec_to_Jnk(best_theta), best_fit
