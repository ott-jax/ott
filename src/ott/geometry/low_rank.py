# Copyright OTT-JAX
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Callable, Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from ott import utils
from ott.geometry import costs, geometry
from ott.math import utils as mu

__all__ = ["LRCGeometry", "LRKGeometry"]


@jax.tree_util.register_pytree_node_class
class LRCGeometry(geometry.Geometry):
  """Geometry whose cost is defined by product of two low-rank matrices.

  Implements geometries that are defined as low rank products, i.e. for which
  there exists two matrices :math:`A` and :math:`B` of :math:`r` columns such
  that the cost of the geometry equals :math:`AB^T`. Apart from being faster to
  apply to a vector, these geometries are characterized by the fact that adding
  two such geometries should be carried out by concatenating factors, i.e.
  if :math:`C = AB^T` and :math:`D = EF^T` then :math:`C + D = [A,E][B,F]^T`

  Args:
    cost_1: Array of shape ``[num_a, r]``.
    cost_2: Array of shape ``[num_b, r]``.
    bias: constant added to entire cost matrix.
    scale: Value used to rescale the factors of the low-rank geometry.
    scale_cost: option to rescale the cost matrix. Implemented scalings are
      'max_bound', 'mean' and 'max_cost'. Alternatively, a float
      factor can be given to rescale the cost such that
      ``cost_matrix /= scale_cost``.
    batch_size: optional size of the batch to compute online (without
      instantiating the matrix) the scale factor ``scale_cost`` of the
      :attr:`cost_matrix` when ``scale_cost = 'max_cost'``. If `None`, the batch
      size is set to `1024` or to the largest number of samples between
      :attr:`cost_1` and :attr:`cost_2` if smaller than `1024`.
    kwargs: keyword arguments for :class:`~ott.geometry.geometry.Geometry`.
  """

  def __init__(
      self,
      cost_1: jnp.ndarray,
      cost_2: jnp.ndarray,
      bias: float = 0.0,
      scale_factor: float = 1.0,
      scale_cost: Union[int, float, Literal["mean", "max_bound",
                                            "max_cost"]] = 1.0,
      batch_size: Optional[int] = None,
      **kwargs: Any,
  ):
    super().__init__(**kwargs)
    self._cost_1 = cost_1
    self._cost_2 = cost_2
    self._bias = bias
    self._scale_factor = scale_factor
    self._scale_cost = scale_cost
    self.batch_size = batch_size

  @property
  def cost_1(self) -> jnp.ndarray:
    """First factor of the :attr:`cost_matrix`."""
    scale_factor = jnp.sqrt(self._scale_factor * self.inv_scale_cost)
    return scale_factor * self._cost_1

  @property
  def cost_2(self) -> jnp.ndarray:
    """Second factor of the :attr:`cost_matrix`."""
    scale_factor = jnp.sqrt(self._scale_factor * self.inv_scale_cost)
    return scale_factor * self._cost_2

  @property
  def bias(self) -> float:
    """Constant offset added to the entire :attr:`cost_matrix`."""
    return self._bias * self.inv_scale_cost

  @property
  def cost_rank(self) -> int:  # noqa: D102
    return self._cost_1.shape[1]

  @property
  def cost_matrix(self) -> jnp.ndarray:
    """Materialize the cost matrix."""
    return jnp.matmul(self.cost_1, self.cost_2.T) + self.bias

  @property
  def shape(self) -> Tuple[int, int]:  # noqa: D102
    return self._cost_1.shape[0], self._cost_2.shape[0]

  @property
  def is_symmetric(self) -> bool:  # noqa: D102
    return (
        self._cost_1.shape[0] == self._cost_2.shape[0] and
        jnp.all(self._cost_1 == self._cost_2)
    )

  @property
  def inv_scale_cost(self) -> float:  # noqa: D102
    if isinstance(self._scale_cost, (int, float, jax.Array)):
      return 1.0 / self._scale_cost
    self = self._masked_geom()
    if self._scale_cost == "max_bound":
      x_norm = self._cost_1[:, 0].max()
      y_norm = self._cost_2[:, 1].max()
      max_bound = x_norm + y_norm + 2 * jnp.sqrt(x_norm * y_norm)
      return 1.0 / (max_bound + self._bias)
    if self._scale_cost == "mean":
      factor1 = jnp.dot(self._n_normed_ones, self._cost_1)
      factor2 = jnp.dot(self._cost_2.T, self._m_normed_ones)
      mean = jnp.dot(factor1, factor2) + self._bias
      return 1.0 / mean
    if self._scale_cost == "max_cost":
      return 1.0 / self.compute_max_cost()
    raise ValueError(f"Scaling {self._scale_cost} not implemented.")

  def apply_square_cost(self, arr: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    """Apply elementwise-square of cost matrix to array (vector or matrix)."""
    (n, m), r = self.shape, self.cost_rank
    # When applying square of a LRCGeometry, one can either elementwise square
    # the cost matrix, or instantiate an augmented (rank^2) LRCGeometry
    # and apply it. First is O(nm), the other is O((n+m)r^2).
    if n * m < (n + m) * r ** 2:  # better use regular apply
      return super().apply_square_cost(arr, axis)

    new_cost_1 = self.cost_1[:, :, None] * self.cost_1[:, None, :]
    new_cost_2 = self.cost_2[:, :, None] * self.cost_2[:, None, :]
    return LRCGeometry(
        cost_1=new_cost_1.reshape((n, r ** 2)),
        cost_2=new_cost_2.reshape((m, r ** 2))
    ).apply_cost(arr, axis)

  def _apply_cost_to_vec(
      self,
      vec: jnp.ndarray,
      axis: int = 0,
      fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
      is_linear: bool = False,
  ) -> jnp.ndarray:
    """Apply [num_a, num_b] fn(cost) (or transpose) to vector.

    Args:
      vec: jnp.ndarray [num_a,] ([num_b,] if axis=1) vector
      axis: axis on which the reduction is done.
      fn: function optionally applied to cost matrix element-wise, before the
        doc product
      is_linear: Whether ``fn`` is a linear function to enable efficient
        implementation. See :func:`ott.geometry.geometry.is_linear`
        for a heuristic to help determine if a function is linear.

    Returns:
      A jnp.ndarray corresponding to cost x vector
    """

    def linear_apply(
        vec: jnp.ndarray, axis: int, fn: Callable[[jnp.ndarray], jnp.ndarray]
    ) -> jnp.ndarray:
      c1 = self.cost_1 if axis == 1 else self.cost_2
      c2 = self.cost_2 if axis == 1 else self.cost_1
      c2 = fn(c2) if fn is not None else c2
      bias = fn(self.bias) if fn is not None else self.bias
      out = jnp.dot(c1, jnp.dot(c2.T, vec))
      return out + bias * jnp.sum(vec) * jnp.ones_like(out)

    if fn is None or is_linear:
      return linear_apply(vec, axis, fn=fn)
    return super()._apply_cost_to_vec(vec, axis, fn=fn)

  def compute_max_cost(self) -> float:
    """Compute the maximum of the :attr:`cost_matrix`.

    Three cases are taken into account:

    - If the number of samples of ``cost_1`` and ``cost_2`` are both smaller
      than 1024 and if ``batch_size`` is `None`, the ``cost_matrix`` is
      computed to obtain its maximum entry.
    - If one of the number of samples of ``cost_1`` or ``cost_2`` is larger
      than 1024 and if ``batch_size`` is `None`, then the maximum of the
      cost matrix is calculated by batch. The batches are created on the
      longest axis of the cost matrix and their size is fixed to 1024.
    - If ``batch_size`` is provided as a float, then the maximum of the cost
      matrix is calculated by batch. The batches are created on the longest
      axis of the cost matrix and their size if fixed by ``batch_size``.

    Returns:
      Maximum of the cost matrix.
    """
    batch_for_y = self.shape[1] > self.shape[0]

    n = self.shape[1] if batch_for_y else self.shape[0]
    p = self._cost_2.shape[1] if batch_for_y else self._cost_1.shape[1]
    carry = ((self._cost_1, self._cost_2) if batch_for_y else
             (self._cost_2, self._cost_1))

    if self.batch_size:
      batch_size = min(self.batch_size, n)
    else:
      batch_size = min(1024, max(self.shape[0], self.shape[1]))
    n_batch = n // batch_size

    def body(carry, slice_idx):
      cost1, cost2 = carry
      cost2_slice = jax.lax.dynamic_slice(
          cost2, (slice_idx * batch_size, 0), (batch_size, p)
      )
      out_slice = jnp.max(jnp.dot(cost2_slice, cost1.T))
      return carry, out_slice

    def finalize(carry):
      cost1, cost2 = carry
      return jnp.dot(cost2[n_batch * batch_size:], cost1.T)

    _, out = jax.lax.scan(body, carry, jnp.arange(n_batch))
    last_slice = finalize(carry)
    max_value = jnp.max(jnp.concatenate((out, last_slice.reshape(-1))))
    return max_value + self._bias

  def to_LRCGeometry(
      self,
      rank: int = 0,
      tol: float = 1e-2,
      rng: Optional[jax.Array] = None,
      scale: float = 1.0,
  ) -> "LRCGeometry":
    """Return self."""
    del rank, tol, rng, scale
    return self

  @property
  def can_LRC(self):  # noqa: D102
    return True

  def subset(  # noqa: D102
      self, src_ixs: Optional[jnp.ndarray], tgt_ixs: Optional[jnp.ndarray],
      **kwargs: Any
  ) -> "LRCGeometry":

    def subset_fn(
        arr: Optional[jnp.ndarray],
        ixs: Optional[jnp.ndarray],
    ) -> jnp.ndarray:
      return arr if arr is None or ixs is None else arr[ixs, ...]

    return self._mask_subset_helper(
        src_ixs, tgt_ixs, fn=subset_fn, propagate_mask=True, **kwargs
    )

  def mask(  # noqa: D102
      self,
      src_mask: Optional[jnp.ndarray],
      tgt_mask: Optional[jnp.ndarray],
      mask_value: float = 0.0,
  ) -> "LRCGeometry":

    def mask_fn(
        arr: Optional[jnp.ndarray],
        mask: Optional[jnp.ndarray],
    ) -> Optional[jnp.ndarray]:
      if arr is None or mask is None:
        return arr
      return jnp.where(mask[:, None], arr, mask_value)

    src_mask = self._normalize_mask(src_mask, self.shape[0])
    tgt_mask = self._normalize_mask(tgt_mask, self.shape[1])
    return self._mask_subset_helper(
        src_mask, tgt_mask, fn=mask_fn, propagate_mask=False
    )

  def _mask_subset_helper(
      self,
      src_ixs: Optional[jnp.ndarray],
      tgt_ixs: Optional[jnp.ndarray],
      *,
      fn: Callable[[Optional[jnp.ndarray], Optional[jnp.ndarray]],
                   Optional[jnp.ndarray]],
      propagate_mask: bool,
      **kwargs: Any,
  ) -> "LRCGeometry":
    (c1, c2, src_mask, tgt_mask, *children), aux_data = self.tree_flatten()
    c1 = fn(c1, src_ixs)
    c2 = fn(c2, tgt_ixs)
    if propagate_mask:
      src_mask = self._normalize_mask(src_mask, self.shape[0])
      tgt_mask = self._normalize_mask(tgt_mask, self.shape[1])
      src_mask = fn(src_mask, src_ixs)
      tgt_mask = fn(tgt_mask, tgt_ixs)

    aux_data = {**aux_data, **kwargs}
    return type(self).tree_unflatten(
        aux_data, [c1, c2, src_mask, tgt_mask] + children
    )

  def __add__(self, other: "LRCGeometry") -> "LRCGeometry":
    if not isinstance(other, LRCGeometry):
      return NotImplemented
    return LRCGeometry(
        cost_1=jnp.concatenate((self.cost_1, other.cost_1), axis=1),
        cost_2=jnp.concatenate((self.cost_2, other.cost_2), axis=1),
        bias=self._bias + other._bias,
        # already included in `cost_{1,2}`
        scale_factor=1.0,
        scale_cost=1.0,
    )

  @property
  def dtype(self) -> jnp.dtype:  # noqa: D102
    return self._cost_1.dtype

  def tree_flatten(self):  # noqa: D102
    return (
        self._cost_1,
        self._cost_2,
        self._src_mask,
        self._tgt_mask,
        self._epsilon_init,
        self._bias,
        self._scale_factor,
    ), {
        "scale_cost": self._scale_cost,
        "batch_size": self.batch_size
    }

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    c1, c2, src_mask, tgt_mask, epsilon, bias, scale_factor = children
    return cls(
        c1,
        c2,
        bias=bias,
        scale_factor=scale_factor,
        epsilon=epsilon,
        src_mask=src_mask,
        tgt_mask=tgt_mask,
        **aux_data
    )


@jax.tree_util.register_pytree_node_class
class LRKGeometry(geometry.Geometry):
  """Low-rank kernel geometry.

  .. note::
    This constructor is not meant to be called by the user,
    please use the :meth:`from_pointcloud` method instead.

  Args:
    k1: Array of shape ``[num_a, r]`` with positive features.
    k2: Array of shape ``[num_b, r]`` with positive features.
    epsilon: Epsilon regularization.
    kwargs: Keyword arguments for :class:`~ott.geometry.geometry.Geometry`.
  """

  def __init__(
      self,
      k1: jnp.ndarray,
      k2: jnp.ndarray,
      epsilon: Optional[float] = None,
      **kwargs: Any
  ):
    super().__init__(epsilon=epsilon, relative_epsilon=False, **kwargs)
    self.k1 = k1
    self.k2 = k2

  @classmethod
  def from_pointcloud(
      cls,
      x: jnp.ndarray,
      y: jnp.ndarray,
      *,
      kernel: Literal["gaussian", "arccos"],
      rank: int = 100,
      std: float = 1.0,
      n: int = 1,
      rng: Optional[jax.Array] = None
  ) -> "LRKGeometry":
    r"""Low-rank kernel approximation :cite:`scetbon:20`.

    Args:
      x: Array of shape ``[n, d]``.
      y: Array of shape ``[m, d]``.
      kernel: Type of the kernel to approximate.
      rank: Rank of the approximation.
      std: Depending on the ``kernel`` approximation:

        - ``'gaussian'`` - scale of the Gibbs kernel.
        - ``'arccos'`` - standard deviation of the random projections.
      n: Order of the arc-cosine kernel, see :cite:`cho:09` for reference.
      rng: Random key used for seeding.

    Returns:
      Low-rank kernel geometry.
    """
    rng = utils.default_prng_key(rng)
    if kernel == "gaussian":
      r = jnp.maximum(
          jnp.linalg.norm(x, axis=-1).max(),
          jnp.linalg.norm(y, axis=-1).max()
      )
      k1 = _gaussian_kernel(rng, x, rank, eps=std, R=r)
      k2 = _gaussian_kernel(rng, y, rank, eps=std, R=r)
      eps = std
    elif kernel == "arccos":
      k1 = _arccos_kernel(rng, x, rank, n=n, std=std)
      k2 = _arccos_kernel(rng, y, rank, n=n, std=std)
      eps = 1.0
    else:
      raise NotImplementedError(kernel)

    return cls(k1, k2, epsilon=eps)

  def apply_kernel(  # noqa: D102
      self,
      scaling: jnp.ndarray,
      eps: Optional[float] = None,
      axis: int = 0,
  ) -> jnp.ndarray:
    if axis == 0:
      return self.k2 @ (self.k1.T @ scaling)
    return self.k1 @ (self.k2.T @ scaling)

  @property
  def kernel_matrix(self) -> jnp.ndarray:  # noqa: D102
    return self.k1 @ self.k2.T

  @property
  def cost_matrix(self) -> jnp.ndarray:  # noqa: D102
    eps = jnp.finfo(self.dtype).tiny
    return -self.epsilon * jnp.log(self.kernel_matrix + eps)

  @property
  def rank(self) -> int:  # noqa: D102
    return self.k1.shape[1]

  @property
  def shape(self) -> Tuple[int, int]:  # noqa: D102
    return self.k1.shape[0], self.k2.shape[0]

  @property
  def dtype(self) -> jnp.dtype:  # noqa: D102
    return self.k1.dtype

  def transport_from_potentials(
      self, f: jnp.ndarray, g: jnp.ndarray
  ) -> jnp.ndarray:
    """Not implemented."""
    raise ValueError("Not implemented.")

  def tree_flatten(self):  # noqa: D102
    return [self.k1, self.k2, self._epsilon_init], {}

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    return cls(*children, **aux_data)


def _gaussian_kernel(
    rng: jax.Array,
    x: jnp.ndarray,
    n_features: int,
    eps: float,
    R: jnp.ndarray,
) -> jnp.ndarray:
  _, d = x.shape
  cost_fn = costs.SqEuclidean()

  y = (R ** 2) / (eps * d)
  q = y / (2.0 * mu.lambertw(y))
  sigma = jnp.sqrt(q * eps * 0.25)

  u = jax.random.normal(rng, shape=(n_features, d)) * sigma
  cost = cost_fn.all_pairs(x, u)
  norm_u = cost_fn.norm(u)

  tmp = -2.0 * (cost / eps) + (norm_u / (eps * q))
  phi = (2 * q) ** (d / 4) * jnp.exp(tmp)

  return (1.0 / jnp.sqrt(n_features)) * phi


def _arccos_kernel(
    rng: jax.Array,
    x: jnp.ndarray,
    n_features: int,
    n: int,
    std: float = 1.0,
    kappa: float = 1e-6,
) -> jnp.ndarray:
  n_points, d = x.shape
  c = jnp.sqrt(2) * (std ** (d / 2))

  u = jax.random.normal(rng, shape=(n_features, d)) * std
  tmp = -(1 / 4) * jnp.sum(u ** 2, axis=-1) * (1.0 - (1.0 / (std ** 2)))
  phi = c * (jnp.maximum(0.0, (x @ u.T)) ** n) * jnp.exp(tmp)

  return jnp.c_[(1.0 / jnp.sqrt(n_features)) * phi,
                jnp.full((n_points,), fill_value=kappa)]
