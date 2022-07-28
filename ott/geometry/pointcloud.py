# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""A geometry defined using 2 point clouds and a cost function between them."""
import math
from typing import Any, Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from typing_extensions import Literal

from ott.geometry import costs, geometry, low_rank, ops


@jax.tree_util.register_pytree_node_class
class PointCloud(geometry.Geometry):
  """Defines geometry for 2 point clouds (possibly 1 vs itself) using CostFn.

  Creates a geometry, specifying a cost function passed as CostFn type object.
  When the number of points is large, setting the `online` flag to `True`
  implies that cost and kernel matrices used to update potentials or scalings
  will be recomputed on the fly, rather than stored in memory. More precisely,
  when setting `online`, the cost function will be partially cached by storing
  norm values for each point in both point clouds, but the pairwise cost
  function evaluations won't be. The sum of norms + the pairwise cost term is
  raised to `power`.

  Args:
    x : n x d array of n d-dimensional vectors
    y : m x d array of m d-dimensional vectors. If `None`, use ``x``.
    cost_fn: a CostFn function between two points in dimension d.
    power: a power to raise (norm(x) + norm(y) + cost(x,y)) **
    batch_size: When ``None``, the cost matrix corresponding to that point cloud
     is computed, stored and later re-used at each application of
     :meth:`apply_lse_kernel`. When ``batch_size`` is a positive integer,
     computations are done in an online fashion, namely the cost matrix is
     recomputed at each call of the :meth:`apply_lse_kernel` step,
     ``batch_size`` lines at a time, used on a vector and discarded.
     The online computation is particularly useful for big point clouds
     whose cost matrix does not fit in memory.
    scale_cost: option to rescale the cost matrix. Implemented scalings are
      'median', 'mean', 'max_cost', 'max_norm' and 'max_bound'.
      Alternatively, a float factor can be given to rescale the cost such
      that ``cost_matrix /= scale_cost``. If `True`, use 'mean'.
    kwargs: other optional parameters to be passed on to superclass
      initializer, notably those related to epsilon regularization.
  """

  def __init__(
      self,
      x: jnp.ndarray,
      y: Optional[jnp.ndarray] = None,
      cost_fn: Optional[costs.CostFn] = None,
      power: float = 2.0,
      batch_size: Optional[int] = None,
      scale_cost: Union[bool, int, float,
                        Literal['mean', 'max_norm', 'max_bound', 'max_cost',
                                'median']] = 1.0,
      **kwargs: Any,
  ):
    super().__init__(**kwargs)
    self.x = x
    self.y = self.x if y is None else y
    self._cost_fn = costs.Euclidean() if cost_fn is None else cost_fn
    self.power = power
    self._axis_norm = 0 if callable(self._cost_fn.norm) else None
    if batch_size is not None:
      assert batch_size > 0, f"`batch_size={batch_size}` must be positive."
    self._batch_size = batch_size
    self._scale_cost = "mean" if scale_cost is True else scale_cost

  @property
  def _norm_x(self) -> Union[float, jnp.ndarray]:
    if self._axis_norm == 0:
      return self._cost_fn.norm(self.x)
    return 0.

  @property
  def _norm_y(self) -> Union[float, jnp.ndarray]:
    if self._axis_norm == 0:
      return self._cost_fn.norm(self.y)
    return 0.

  @property
  def cost_matrix(self) -> Optional[jnp.ndarray]:
    if self.is_online:
      return None
    cost_matrix = self._compute_cost_matrix()
    return cost_matrix * self.inv_scale_cost

  @property
  def kernel_matrix(self) -> Optional[jnp.ndarray]:
    if self.is_online:
      return None
    return jnp.exp(-self.cost_matrix / self.epsilon)

  @property
  def shape(self) -> Tuple[int, int]:
    # in the process of flattening/unflattening in vmap, `__init__`
    # can be called with dummy objects
    # we optionally access `shape` in order to get the batch size
    if self.x is None or self.y is None:
      return 0, 0
    return self.x.shape[0], self.y.shape[0]

  @property
  def is_symmetric(self) -> bool:
    return self.y is None or (
        jnp.all(self.x.shape == self.y.shape) and jnp.all(self.x == self.y)
    )

  @property
  def is_squared_euclidean(self) -> bool:
    return isinstance(self._cost_fn, costs.Euclidean) and self.power == 2.0

  @property
  def is_online(self) -> bool:
    """Whether :attr:`cost_matrix` or :attr:`kernel_matrix` \
      is computed on-the-fly."""
    return self.batch_size is not None

  @property
  def inv_scale_cost(self) -> float:
    if isinstance(self._scale_cost, (int, float)):
      return 1.0 / self._scale_cost
    self = self._masked_geom()
    if self._scale_cost == 'max_cost':
      if self.is_online:
        return 1.0 / self._compute_summary_online(self._scale_cost)
      return 1.0 / jnp.max(self._compute_cost_matrix())
    if self._scale_cost == 'mean':
      if self.is_online:
        return 1.0 / self._compute_summary_online(self._scale_cost)
      if self.shape[0] > 0:
        geom = self._masked_geom(mask_value=jnp.nan)._compute_cost_matrix()
        return 1.0 / jnp.nanmean(geom)
      return 1.0
    if self._scale_cost == 'median':
      if not self.is_online:
        geom = self._masked_geom(mask_value=jnp.nan)
        return 1.0 / jnp.nanmedian(geom._compute_cost_matrix())
      raise NotImplementedError(
          "Using the median as scaling factor for "
          "the cost matrix with the online mode is not implemented."
      )
    if self._scale_cost == 'max_norm':
      if self._cost_fn.norm is not None:
        return 1.0 / jnp.maximum(self._norm_x.max(), self._norm_y.max())
      return 1.0
    if self._scale_cost == 'max_bound':
      if self.is_squared_euclidean:
        x_argmax = jnp.argmax(self._norm_x)
        y_argmax = jnp.argmax(self._norm_y)
        max_bound = (
            self._norm_x[x_argmax] + self._norm_y[y_argmax] +
            2 * jnp.sqrt(self._norm_x[x_argmax] * self._norm_y[y_argmax])
        )
        return 1.0 / max_bound
      raise NotImplementedError(
          "Using max_bound as scaling factor for "
          "the cost matrix when the cost is not squared euclidean "
          "is not implemented."
      )
    raise ValueError(f'Scaling {self._scale_cost} not implemented.')

  def _compute_cost_matrix(self) -> jnp.ndarray:
    cost_matrix = self._cost_fn.all_pairs_pairwise(self.x, self.y)
    if self._axis_norm is not None:
      cost_matrix += self._norm_x[:, jnp.newaxis] + self._norm_y[jnp.newaxis, :]
    return cost_matrix ** (0.5 * self.power)

  def apply_lse_kernel(
      self,
      f: jnp.ndarray,
      g: jnp.ndarray,
      eps: float,
      vec: Optional[jnp.ndarray] = None,
      axis: int = 0
  ) -> jnp.ndarray:

    def body0(carry, i: int):
      f, g, eps, vec = carry
      y = jax.lax.dynamic_slice(
          self.y, (i * self.batch_size, 0), (self.batch_size, self.y.shape[1])
      )
      g_ = jax.lax.dynamic_slice(g, (i * self.batch_size,), (self.batch_size,))
      if self._axis_norm is None:
        norm_y = self._norm_y
      else:
        norm_y = jax.lax.dynamic_slice(
            self._norm_y, (i * self.batch_size,), (self.batch_size,)
        )
      h_res, h_sgn = app(
          self.x, y, self._norm_x, norm_y, f, g_, eps, vec, self._cost_fn,
          self.power, self.inv_scale_cost
      )
      return carry, (h_res, h_sgn)

    def body1(carry, i: int):
      f, g, eps, vec = carry
      x = jax.lax.dynamic_slice(
          self.x, (i * self.batch_size, 0), (self.batch_size, self.x.shape[1])
      )
      f_ = jax.lax.dynamic_slice(f, (i * self.batch_size,), (self.batch_size,))
      if self._axis_norm is None:
        norm_x = self._norm_x
      else:
        norm_x = jax.lax.dynamic_slice(
            self._norm_x, (i * self.batch_size,), (self.batch_size,)
        )
      h_res, h_sgn = app(
          self.y, x, self._norm_y, norm_x, g, f_, eps, vec, self._cost_fn,
          self.power, self.inv_scale_cost
      )
      return carry, (h_res, h_sgn)

    def finalize(i: int):
      if axis == 0:
        norm_y = self._norm_y if self._axis_norm is None else self._norm_y[i:]
        return app(
            self.x, self.y[i:], self._norm_x, norm_y, f, g[i:], eps, vec,
            self._cost_fn, self.power, self.inv_scale_cost
        )
      norm_x = self._norm_x if self._axis_norm is None else self._norm_x[i:]
      return app(
          self.y, self.x[i:], self._norm_y, norm_x, g, f[i:], eps, vec,
          self._cost_fn, self.power, self.inv_scale_cost
      )

    if not self.is_online:
      return super().apply_lse_kernel(f, g, eps, vec, axis)

    app = jax.vmap(
        _apply_lse_kernel_xy,
        in_axes=[
            None, 0, None, self._axis_norm, None, 0, None, None, None, None,
            None
        ]
    )

    if axis == 0:
      fun = body0
      v, n = g, self._y_nsplit
    elif axis == 1:
      fun = body1
      v, n = f, self._x_nsplit
    else:
      raise ValueError(axis)

    _, (h_res, h_sign) = jax.lax.scan(
        fun, init=(f, g, eps, vec), xs=jnp.arange(n)
    )
    h_res, h_sign = jnp.concatenate(h_res), jnp.concatenate(h_sign)
    h_res_rest, h_sign_rest = finalize(n * self.batch_size)
    h_res = jnp.concatenate([h_res, h_res_rest])
    h_sign = jnp.concatenate([h_sign, h_sign_rest])

    return eps * h_res - jnp.where(jnp.isfinite(v), v, 0), h_sign

  def apply_kernel(
      self,
      scaling: jnp.ndarray,
      eps: Optional[float] = None,
      axis: int = 0
  ) -> jnp.ndarray:
    if eps is None:
      eps = self.epsilon

    if not self.is_online:
      return super().apply_kernel(scaling, eps, axis)

    app = jax.vmap(
        _apply_kernel_xy,
        in_axes=[None, 0, None, self._axis_norm, None, None, None, None, None]
    )
    if axis == 0:
      return app(
          self.x, self.y, self._norm_x, self._norm_y, scaling, eps,
          self._cost_fn, self.power, self.inv_scale_cost
      )
    if axis == 1:
      return app(
          self.y, self.x, self._norm_y, self._norm_x, scaling, eps,
          self._cost_fn, self.power, self.inv_scale_cost
      )

  def transport_from_potentials(
      self, f: jnp.ndarray, g: jnp.ndarray
  ) -> jnp.ndarray:
    if not self.is_online:
      return super().transport_from_potentials(f, g)
    transport = jax.vmap(
        _transport_from_potentials_xy,
        in_axes=[
            None, 0, None, self._axis_norm, None, 0, None, None, None, None
        ]
    )
    return transport(
        self.y, self.x, self._norm_y, self._norm_x, g, f, self.epsilon,
        self._cost_fn, self.power, self.inv_scale_cost
    )

  def transport_from_scalings(
      self, u: jnp.ndarray, v: jnp.ndarray
  ) -> jnp.ndarray:
    if not self.is_online:
      return super().transport_from_scalings(u, v)
    transport = jax.vmap(
        _transport_from_scalings_xy,
        in_axes=[
            None, 0, None, self._axis_norm, None, 0, None, None, None, None
        ]
    )
    return transport(
        self.y, self.x, self._norm_y, self._norm_x, v, u, self.epsilon,
        self._cost_fn, self.power, self.inv_scale_cost
    )

  def apply_cost(
      self,
      arr: jnp.ndarray,
      axis: int = 0,
      fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
      is_linear: bool = False,
  ) -> jnp.ndarray:
    """Apply cost matrix to array (vector or matrix).

    This function applies the geometry's cost matrix, to perform either
    output = C arr (if axis=1)
    output = C' arr (if axis=0)
    where C is [num_a, num_b] matrix resulting from the (optional) elementwise
    application of fn to each entry of the :attr:`cost_matrix`.

    Args:
      arr: jnp.ndarray [num_a or num_b, batch], vector that will be multiplied
        by the cost matrix.
      axis: standard cost matrix if axis=1, transpose if 0.
      fn: function optionally applied to cost matrix element-wise, before the
        apply.
      is_linear: Whether ``fn`` is a linear function.
        If true and :attr:`is_squared_euclidean` is ``True``, efficient
        implementation is used. See :func:`ott.geometry.geometry.is_linear`
        for a heuristic to help determine if a function is linear.

    Returns:
      A jnp.ndarray, [num_b, batch] if axis=0 or [num_a, batch] if axis=1
    """
    # switch to efficient computation for the squared euclidean case.
    if self.is_squared_euclidean and (fn is None or is_linear):
      return self.vec_apply_cost(arr, axis, fn=fn)

    return self._apply_cost(arr, axis, fn=fn)

  def _apply_cost(
      self, arr: jnp.ndarray, axis: int = 0, fn=None
  ) -> jnp.ndarray:
    """See :meth:`apply_cost`."""
    if self.is_online:
      app = jax.vmap(
          _apply_cost_xy,
          in_axes=[
              None, 0, None, self._axis_norm, None, None, None, None, None
          ]
      )
      if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
      if axis == 0:
        return app(
            self.x, self.y, self._norm_x, self._norm_y, arr, self._cost_fn,
            self.power, self.inv_scale_cost, fn
        )
      if axis == 1:
        return app(
            self.y, self.x, self._norm_y, self._norm_x, arr, self._cost_fn,
            self.power, self.inv_scale_cost, fn
        )
    else:
      return super().apply_cost(arr, axis, fn)

  def vec_apply_cost(
      self,
      arr: jnp.ndarray,
      axis: int = 0,
      fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
  ) -> jnp.ndarray:
    """Apply the geometry's cost matrix in a vectorised way.

    This performs either:
    output = C arr (if axis=1)
    output = C' arr (if axis=0)
    where C is [num_a, num_b]

    This function can be used when the cost matrix is squared euclidean
    and fn is a linear map.

    Args:
      arr: jnp.ndarray [num_a or num_b, p], vector that will be multiplied
        by the cost matrix.
      axis: standard cost matrix if axis=1, transport if 0
      fn: function optionally applied to cost matrix element-wise, before the
        apply

    Returns:
      A jnp.ndarray, [num_b, p] if axis=0 or [num_a, p] if axis=1
    """
    assert self.is_squared_euclidean, "Cost matrix is not a squared Euclidean."
    rank = arr.ndim
    x, y = (self.x, self.y) if axis == 0 else (self.y, self.x)
    nx, ny = jnp.asarray(self._norm_x), jnp.asarray(self._norm_y)
    nx, ny = (nx, ny) if axis == 0 else (ny, nx)

    applied_cost = jnp.dot(nx, arr).reshape(1, -1)
    applied_cost += ny.reshape(-1, 1) * jnp.sum(arr, axis=0).reshape(1, -1)
    cross_term = -2.0 * jnp.dot(y, jnp.dot(x.T, arr))
    applied_cost += cross_term[:, None] if rank == 1 else cross_term
    if fn is not None:
      applied_cost = fn(applied_cost)
    return self.inv_scale_cost * applied_cost

  def _leading_slice(self, t: jnp.ndarray, i: int) -> jnp.ndarray:
    start_indices = [i * self.batch_size] + (t.ndim - 1) * [0]
    slice_sizes = [self.batch_size] + list(t.shape[1:])
    return jax.lax.dynamic_slice(t, start_indices, slice_sizes)

  def _compute_summary_online(
      self, summary: Literal['mean', 'max_cost']
  ) -> float:
    """Compute mean or max of cost matrix online, i.e. without instantiating it.

    Args:
      summary: can be 'mean' or 'max_cost'.

    Returns:
      summary statistics
    """
    scale_cost = 1.0

    def body0(carry, i: int):
      vec, = carry
      y = self._leading_slice(self.y, i)
      if self._axis_norm is None:
        norm_y = self._norm_y
      else:
        norm_y = self._leading_slice(self._norm_y, i)
      h_res = app(
          self.x, y, self._norm_x, norm_y, vec, self._cost_fn, self.power,
          scale_cost
      )
      return carry, h_res

    def body1(carry, i: int):
      vec, = carry
      x = self._leading_slice(self.x, i)
      if self._axis_norm is None:
        norm_x = self._norm_x
      else:
        norm_x = self._leading_slice(self._norm_x, i)
      h_res = app(
          self.y, x, self._norm_y, norm_x, vec, self._cost_fn, self.power,
          scale_cost
      )
      return carry, h_res

    def finalize(i: int):
      if batch_for_y:
        norm_y = self._norm_y if self._axis_norm is None else self._norm_y[i:]
        return app(
            self.x, self.y[i:], self._norm_x, norm_y, vec, self._cost_fn,
            self.power, scale_cost
        )
      norm_x = self._norm_x if self._axis_norm is None else self._norm_x[i:]
      return app(
          self.y, self.x[i:], self._norm_y, norm_x, vec, self._cost_fn,
          self.power, scale_cost
      )

    if summary == 'mean':
      fn = _apply_cost_xy
    elif summary == 'max_cost':
      fn = _apply_max_xy
    else:
      raise ValueError(
          f'Scaling method {summary} does not exist for online mode.'
      )
    app = jax.vmap(
        fn, in_axes=[None, 0, None, self._axis_norm, None, None, None, None]
    )

    batch_for_y = self.shape[0] < self.shape[1]
    if batch_for_y:
      fun = body0
      n = self._y_nsplit
      vec, other = self._n_normed_ones, self._m_normed_ones
    else:
      fun = body1
      n = self._x_nsplit
      vec, other = self._m_normed_ones, self._n_normed_ones

    _, val = jax.lax.scan(fun, init=(vec,), xs=jnp.arange(n))
    val = jnp.concatenate(val).squeeze()
    val_rest = finalize(n * self.batch_size)
    val_res = jnp.concatenate([val, val_rest])

    if summary == 'mean':
      return jnp.sum(val_res * other)
    if summary == 'max_cost':
      # TODO(michalk8): explain why scaling is not needed
      return jnp.max(val_res)
    raise ValueError(
        f'Scaling method {summary} does not exist for online mode.'
    )

  def barycenter(self, weights: jnp.ndarray) -> jnp.ndarray:
    """Compute barycenter of points in self.x using weights, valid for p=2.0."""
    assert self.power == 2.0, self.power
    return self._cost_fn.barycenter(self.x, weights)

  @classmethod
  def prepare_divergences(
      cls,
      x: jnp.ndarray,
      y: jnp.ndarray,
      static_b: bool = False,
      src_mask: Optional[jnp.ndarray] = None,
      tgt_mask: Optional[jnp.ndarray] = None,
      **kwargs: Any
  ) -> Tuple["PointCloud", ...]:
    """Instantiate the geometries used for a divergence computation."""
    couples = [(x, y), (x, x)]
    masks = [(src_mask, tgt_mask), (src_mask, src_mask)]
    if not static_b:
      couples += [(y, y)]
      masks += [(tgt_mask, tgt_mask)]

    return tuple(
        cls(x, y, src_mask=x_mask, tgt_mask=y_mask, **kwargs)
        for ((x, y), (x_mask, y_mask)) in zip(couples, masks)
    )

  def tree_flatten(self):
    # passing self.power in aux_data to be able to condition on it.
    return ([self.x, self.y, self._src_mask, self._tgt_mask, self._cost_fn], {
        'epsilon': self._epsilon_init,
        'relative_epsilon': self._relative_epsilon,
        'scale_epsilon': self._scale_epsilon,
        'batch_size': self._batch_size,
        'power': self.power,
        'scale_cost': self._scale_cost
    })

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    x, y, src_mask, tgt_mask, cost_fn = children
    return cls(
        x, y, cost_fn=cost_fn, src_mask=src_mask, tgt_mask=tgt_mask, **aux_data
    )

  def to_LRCGeometry(
      self,
      scale: float = 1.0,
      **kwargs: Any,
  ) -> Union[low_rank.LRCGeometry, 'PointCloud']:
    """Convert sqEuc. PointCloud to LRCGeometry if useful, and rescale."""
    if self.is_squared_euclidean:
      (n, m), d = self.shape, self.x.shape[1]
      if n * m > (n + m) * d:  # here apply_cost using LRCGeometry preferable.
        return self._sqeucl_to_lr(scale)
      (x, y, *children), aux_data = self.tree_flatten()
      x = x * jnp.sqrt(scale)
      y = y * jnp.sqrt(scale)
      return type(self).tree_unflatten(aux_data, [x, y] + children)
    return super().to_LRCGeometry(**kwargs)

  def _sqeucl_to_lr(self, scale: float = 1.0) -> low_rank.LRCGeometry:
    assert self.is_squared_euclidean, "Geometry must be squared Euclidean."
    n, m = self.shape
    cost_1 = jnp.concatenate((
        jnp.sum(self.x ** 2, axis=1, keepdims=True), jnp.ones(
            (n, 1)
        ), -jnp.sqrt(2) * self.x
    ),
                             axis=1)
    cost_2 = jnp.concatenate((
        jnp.ones((m, 1)), jnp.sum(self.y ** 2, axis=1,
                                  keepdims=True), jnp.sqrt(2) * self.y
    ),
                             axis=1)
    cost_1 *= jnp.sqrt(scale)
    cost_2 *= jnp.sqrt(scale)

    return low_rank.LRCGeometry(
        cost_1=cost_1,
        cost_2=cost_2,
        epsilon=self._epsilon_init,
        relative_epsilon=self._relative_epsilon,
        scale=self._scale_epsilon,
        scale_cost=self._scale_cost,
        src_mask=self.src_mask,
        tgt_mask=self.tgt_mask,
        **self._kwargs
    )

  def subset(
      self, src_ixs: Optional[jnp.ndarray], tgt_ixs: Optional[jnp.ndarray],
      **kwargs: Any
  ) -> "PointCloud":

    def subset_fn(
        arr: Optional[jnp.ndarray],
        ixs: Optional[jnp.ndarray],
    ) -> jnp.ndarray:
      return arr if arr is None or ixs is None else arr[jnp.atleast_1d(ixs)]

    return self._mask_subset_helper(
        src_ixs, tgt_ixs, fn=subset_fn, propagate_mask=True, **kwargs
    )

  def mask(
      self,
      src_mask: Optional[jnp.ndarray],
      tgt_mask: Optional[jnp.ndarray],
      mask_value: float = 0.,
  ) -> "PointCloud":

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
  ) -> "PointCloud":
    (x, y, src_mask, tgt_mask, *children), aux_data = self.tree_flatten()
    x = fn(x, src_ixs)
    y = fn(y, tgt_ixs)
    if propagate_mask:
      src_mask = self._normalize_mask(src_mask, self.shape[0])
      tgt_mask = self._normalize_mask(tgt_mask, self.shape[1])
      src_mask = fn(src_mask, src_ixs)
      tgt_mask = fn(tgt_mask, tgt_ixs)
    aux_data = {**aux_data, **kwargs}

    return type(self).tree_unflatten(
        aux_data, [x, y, src_mask, tgt_mask] + children
    )

  @property
  def batch_size(self) -> Optional[int]:
    """Batch size for online mode."""
    if self._batch_size is None:
      return None
    n, m = self.shape
    return min(n, m, self._batch_size)

  @property
  def _x_nsplit(self) -> Optional[int]:
    if self.batch_size is None:
      return None
    n, _ = self.shape
    return int(math.floor(n / self.batch_size))

  @property
  def _y_nsplit(self) -> Optional[int]:
    if self.batch_size is None:
      return None
    _, m = self.shape
    return int(math.floor(m / self.batch_size))


def _apply_lse_kernel_xy(
    x, y, norm_x, norm_y, f, g, eps, vec, cost_fn, cost_pow, scale_cost
):
  c = _cost(x, y, norm_x, norm_y, cost_fn, cost_pow, scale_cost)
  return ops.logsumexp((f + g - c) / eps, b=vec, return_sign=True, axis=-1)


def _transport_from_potentials_xy(
    x, y, norm_x, norm_y, f, g, eps, cost_fn, cost_pow, scale_cost
):
  return jnp.exp(
      (f + g - _cost(x, y, norm_x, norm_y, cost_fn, cost_pow, scale_cost)) / eps
  )


def _apply_kernel_xy(
    x, y, norm_x, norm_y, vec, eps, cost_fn, cost_pow, scale_cost
):
  c = _cost(x, y, norm_x, norm_y, cost_fn, cost_pow, scale_cost)
  return jnp.dot(jnp.exp(-c / eps), vec)


def _transport_from_scalings_xy(
    x, y, norm_x, norm_y, u, v, eps, cost_fn, cost_pow, scale_cost
):
  return jnp.exp(
      -_cost(x, y, norm_x, norm_y, cost_fn, cost_pow, scale_cost) * scale_cost /
      eps
  ) * u * v


def _cost(x, y, norm_x, norm_y, cost_fn, cost_pow, scale_cost):
  one_line_pairwise = jax.vmap(cost_fn.pairwise, in_axes=[0, None])
  return ((norm_x + norm_y + one_line_pairwise(x, y)) ** (0.5 * cost_pow) *
          scale_cost)


def _apply_cost_xy(
    x, y, norm_x, norm_y, vec, cost_fn, cost_pow, scale_cost, fn=None
):
  """Apply [num_b, num_a] fn(cost) matrix (or transpose) to vector.

  Applies [num_b, num_a] ([num_a, num_b] if axis=1 from `apply_cost`)
  fn(cost) matrix (or transpose) to vector.

  Args:
    x: jnp.ndarray [num_a, d], first pointcloud
    y: jnp.ndarray [num_b, d], second pointcloud
    norm_x: jnp.ndarray [num_a,], (squared) norm as defined in by cost_fn
    norm_y: jnp.ndarray [num_b,], (squared) norm as defined in by cost_fn
    vec: jnp.ndarray [num_a,] ([num_b,] if axis=1 from `apply_cost`) vector
    cost_fn: a CostFn function between two points in dimension d.
    cost_pow: a power to raise (norm(x) + norm(y) + cost(x,y)) **
    scale_cost: scaling factor of the cost matrix.
    fn: function optionally applied to cost matrix element-wise, before the
      apply.

  Returns:
    A jnp.ndarray corresponding to cost x vector
  """
  c = _cost(x, y, norm_x, norm_y, cost_fn, cost_pow, scale_cost)
  return jnp.dot(c, vec) if fn is None else jnp.dot(fn(c), vec)


def _apply_max_xy(x, y, norm_x, norm_y, vec, cost_fn, cost_pow, scale_cost):
  del vec
  c = _cost(x, y, norm_x, norm_y, cost_fn, cost_pow, scale_cost)
  return jnp.max(jnp.abs(c))
