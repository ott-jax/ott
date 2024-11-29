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
from ott.geometry import costs, geometry, low_rank
from ott.math import utils as mu

__all__ = ["PointCloud"]


@jax.tree_util.register_pytree_node_class
class PointCloud(geometry.Geometry):
  """Defines geometry for 2 point clouds (possibly 1 vs itself).

  Creates a geometry, specifying a cost function passed as CostFn type object.
  When the number of points is large, setting the ``batch_size`` flag implies
  that cost and kernel matrices used to update potentials or scalings
  will be recomputed on the fly, rather than stored in memory. More precisely,
  when setting ``batch_size``, the cost function will be partially cached by
  storing norm values for each point in both point clouds, but the pairwise cost
  function evaluations won't be.

  Args:
    x : n x d array of n d-dimensional vectors
    y : m x d array of m d-dimensional vectors. If `None`, use ``x``.
    cost_fn: a CostFn function between two points in dimension d.
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
      that ``cost_matrix /= scale_cost``.
    kwargs: keyword arguments for :class:`~ott.geometry.geometry.Geometry`.
  """

  def __init__(
      self,
      x: jnp.ndarray,
      y: Optional[jnp.ndarray] = None,
      cost_fn: Optional[costs.CostFn] = None,
      batch_size: Optional[int] = None,
      scale_cost: Union[float, Literal["mean", "max_norm", "max_bound",
                                       "max_cost", "median"]] = 1.0,
      **kwargs: Any,
  ):
    super().__init__(**kwargs)
    self.x = x
    self.y = self.x if y is None else y

    self.cost_fn = costs.SqEuclidean() if cost_fn is None else cost_fn
    if batch_size is not None:
      assert batch_size > 0, f"`batch_size={batch_size}` must be positive."
    self._batch_size = batch_size
    self._scale_cost = scale_cost

  def apply_lse_kernel(  # noqa: D102
      self,
      f: jnp.ndarray,
      g: jnp.ndarray,
      eps: float,
      vec: Optional[jnp.ndarray] = None,
      axis: int = 0
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if not self.is_online:
      return super().apply_lse_kernel(f, g, eps, vec, axis)

    def apply(x: jnp.ndarray, y: jnp.ndarray, f: jnp.ndarray,
              g: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
      x, y = jnp.atleast_2d(x), jnp.atleast_2d(y)
      cost = self.cost_fn.all_pairs(x, y) * inv_scale_cost
      cost = cost.squeeze(1 - axis)
      # axis=-1
      res, sgn = mu.logsumexp((f + g - cost) / eps, b=vec, return_sign=True)
      return eps * res, sgn

    inv_scale_cost = self.inv_scale_cost
    in_axes = (None, 0, None, 0) if axis == 0 else (0, None, 0, None)
    batched_apply = utils.batched_vmap(
        apply,
        batch_size=self.batch_size,
        in_axes=in_axes,
    )
    w_res, w_sgn = batched_apply(self.x, self.y, f, g)
    remove = f if axis == 1 else g
    return w_res - jnp.where(jnp.isfinite(remove), remove, 0), w_sgn

  def apply_kernel(  # noqa: D102
      self,
      vec: jnp.ndarray,
      eps: Optional[float] = None,
      axis: int = 0
  ) -> jnp.ndarray:
    if eps is None:
      eps = self.epsilon
    if not self.is_online:
      return super().apply_kernel(vec, eps, axis)

    def apply(x: jnp.ndarray, y: jnp.ndarray, vec: jnp.ndarray) -> jnp.ndarray:
      x, y = jnp.atleast_2d(x), jnp.atleast_2d(y)
      cost = self.cost_fn.all_pairs(x, y) * inv_scale_cost
      cost = cost.squeeze(1 - axis)
      return jnp.dot(jnp.exp(-cost / eps), vec)

    inv_scale_cost = self.inv_scale_cost
    in_axes = (None, 0, None) if axis == 0 else (0, None, None)
    batched_apply = utils.batched_vmap(
        apply, batch_size=self.batch_size, in_axes=in_axes
    )
    return batched_apply(self.x, self.y, vec)

  def _apply_cost_to_vec(
      self,
      vec: jnp.ndarray,
      axis: int = 0,
      fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
      is_linear: bool = False,
      scale_cost: Optional[float] = None,
  ) -> jnp.ndarray:

    def apply(x: jnp.ndarray, y: jnp.ndarray, arr: jnp.ndarray) -> jnp.ndarray:
      x, y = jnp.atleast_2d(x), jnp.atleast_2d(y)
      cost = self.cost_fn.all_pairs(x, y) * scale_cost
      cost = cost.squeeze(1 - axis)
      if fn is not None:
        cost = fn(cost)
      return jnp.dot(cost, arr)

    # when computing the online properties, this is set to 1.0
    if scale_cost is None:
      scale_cost = self.inv_scale_cost
    # switch to an efficient computation for the squared Euclidean case
    if self.is_squared_euclidean and (fn is None or is_linear):
      return self._apply_sqeucl_cost(
          vec,
          scale_cost,
          axis=axis,
          fn=fn,
      )

    # materialize the cost
    if not self.is_online:
      return super()._apply_cost_to_vec(
          vec, axis=axis, fn=fn, is_linear=is_linear
      )

    in_axes = (None, 0, None) if axis == 0 else (0, None, None)
    batched_apply = utils.batched_vmap(
        apply, batch_size=self.batch_size, in_axes=in_axes
    )
    return batched_apply(self.x, self.y, vec)

  def _apply_sqeucl_cost(
      self,
      vec: jnp.ndarray,
      scale_cost: float,
      axis: int = 0,
      fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
  ) -> jnp.ndarray:
    assert vec.ndim == 1, vec.shape
    assert self.is_squared_euclidean, "Cost matrix is not a squared Euclidean."
    x, y = (self.x, self.y) if axis == 0 else (self.y, self.x)
    nx, ny = self.cost_fn.norm(x), self.cost_fn.norm(y)

    applied_cost = jnp.dot(nx, vec) + ny * jnp.sum(vec, axis=0)
    applied_cost = applied_cost - 2.0 * jnp.dot(y, jnp.dot(x.T, vec))
    if fn is not None:
      applied_cost = fn(applied_cost)
    return scale_cost * applied_cost

  def _compute_summary_online(
      self, summary: Literal["mean", "max_cost"]
  ) -> jnp.ndarray:
    """Compute mean or max of cost matrix online, i.e. without instantiating it.

    Args:
      summary: can be 'mean' or 'max_cost'.

    Returns:
      summary statistics
    """

    def compute_max(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
      x, y = jnp.atleast_2d(x), jnp.atleast_2d(y)
      cost = self.cost_fn.all_pairs(x, y)
      return jnp.max(jnp.abs(cost))

    if summary == "mean":
      a, b = self._n_normed_ones, self._m_normed_ones
      return jnp.sum(self._apply_cost_to_vec(a, scale_cost=1.0) * b)

    if summary == "max_cost":
      fn = utils.batched_vmap(
          compute_max, batch_size=self.batch_size, in_axes=[0, None]
      )
      return jnp.max(fn(self.x, self.y))

    raise ValueError(
        f"Scaling method {summary} does not exist for online mode."
    )

  def barycenter(self, weights: jnp.ndarray) -> jnp.ndarray:
    """Compute barycenter of points in self.x using weights."""
    return self.cost_fn.barycenter(self.x, weights)[0]

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

  def tree_flatten(self):  # noqa: D102
    return (
        self.x,
        self.y,
        self._src_mask,
        self._tgt_mask,
        self._epsilon_init,
        self.cost_fn,
    ), {
        "batch_size": self._batch_size,
        "scale_cost": self._scale_cost,
        "relative_epsilon": self._relative_epsilon,
    }

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    x, y, src_mask, tgt_mask, epsilon, cost_fn = children
    return cls(
        x,
        y,
        cost_fn=cost_fn,
        src_mask=src_mask,
        tgt_mask=tgt_mask,
        epsilon=epsilon,
        **aux_data
    )

  def _cosine_to_sqeucl(self) -> "PointCloud":
    assert isinstance(self.cost_fn, costs.Cosine), type(self.cost_fn)
    (x, y, *args, _), aux_data = self.tree_flatten()
    x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
    y = y / jnp.linalg.norm(y, axis=-1, keepdims=True)
    # TODO(michalk8): find a better way
    aux_data["scale_cost"] = 2.0 / self.inv_scale_cost
    cost_fn = costs.SqEuclidean()
    return type(self).tree_unflatten(aux_data, [x, y] + args + [cost_fn])

  def to_LRCGeometry(
      self,
      scale: float = 1.0,
      **kwargs: Any,
  ) -> Union[low_rank.LRCGeometry, "PointCloud"]:
    r"""Convert point cloud to low-rank geometry.

    Args:
      scale: Value used to rescale the factors of the low-rank geometry.
        Useful when this geometry is used in the linear term of fused GW.
      kwargs: Keyword arguments, such as ``rank``, to
        :meth:`~ott.geometry.geometry.Geometry.to_LRCGeometry` used when
        the point cloud does not have squared Euclidean cost.

    Returns:
      Returns the unmodified point cloud if :math:`n m \ge (n + m) d`, where
      :math:`n, m` is the shape and :math:`d` is the dimension of the point
      cloud with squared Euclidean cost.
      Otherwise, returns the re-scaled low-rank geometry.
    """
    if self.is_squared_euclidean:
      if self._check_LRC_dim:
        return self._sqeucl_to_lr(scale)
      # we don't update the `scale_factor` because in GW, the linear cost
      # is first materialized and then scaled by `fused_penalty` afterwards
      return self
    return super().to_LRCGeometry(scale=scale, **kwargs)

  def _sqeucl_to_lr(self, scale: float = 1.0) -> low_rank.LRCGeometry:
    assert self.is_squared_euclidean, "Geometry must be squared Euclidean."
    n, m = self.shape
    nx = jnp.sum(self.x ** 2, axis=1, keepdims=True)
    ny = jnp.sum(self.y ** 2, axis=1, keepdims=True)
    cost_1 = jnp.concatenate(
        (nx, jnp.ones((n, 1), dtype=self.dtype), -(2.0 ** 0.5) * self.x),
        axis=1,
    )
    cost_2 = jnp.concatenate(
        (jnp.ones((m, 1), dtype=self.dtype), ny, (2.0 ** 0.5) * self.y),
        axis=1,
    )

    return low_rank.LRCGeometry(
        cost_1=cost_1,
        cost_2=cost_2,
        scale_factor=scale,
        epsilon=self._epsilon_init,
        relative_epsilon=self._relative_epsilon,
        scale_cost=self._scale_cost,
        src_mask=self.src_mask,
        tgt_mask=self.tgt_mask,
    )

  def subset(  # noqa: D102
      self, src_ixs: Optional[jnp.ndarray], tgt_ixs: Optional[jnp.ndarray],
      **kwargs: Any
  ) -> "PointCloud":

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
  def cost_matrix(self) -> Optional[jnp.ndarray]:  # noqa: D102
    return self.inv_scale_cost * self._unscaled_cost_matrix

  @property
  def _unscaled_cost_matrix(self) -> jnp.ndarray:
    return self.cost_fn.all_pairs(self.x, self.y)

  @property
  def inv_scale_cost(self) -> float:  # noqa: D102
    if isinstance(self._scale_cost, (int, float, jax.Array)):
      return 1.0 / self._scale_cost
    self = self._masked_geom()
    if self._scale_cost == "max_cost":
      if self.is_online:
        return 1.0 / self._compute_summary_online(self._scale_cost)
      return 1.0 / jnp.max(self._unscaled_cost_matrix)
    if self._scale_cost == "mean":
      if self.is_online:
        return 1.0 / self._compute_summary_online(self._scale_cost)
      geom = self._masked_geom(mask_value=jnp.nan)._unscaled_cost_matrix
      return 1.0 / jnp.nanmean(geom)
    if self._scale_cost == "median":
      if not self.is_online:
        geom = self._masked_geom(mask_value=jnp.nan)
        return 1.0 / jnp.nanmedian(geom._unscaled_cost_matrix)
      raise NotImplementedError(
          "Using the median as scaling factor for "
          "the cost matrix with the online mode is not implemented."
      )
    if not hasattr(self.cost_fn, "norm"):
      raise ValueError("Cost function has no norm method.")
    norm_x = self.cost_fn.norm(self.x)
    norm_y = self.cost_fn.norm(self.y)
    if self._scale_cost == "max_norm":
      return 1.0 / jnp.maximum(norm_x.max(), norm_y.max())
    if self._scale_cost == "max_bound":
      if self.is_squared_euclidean:
        x_argmax = jnp.argmax(norm_x)
        y_argmax = jnp.argmax(norm_y)
        max_bound = (
            norm_x[x_argmax] + norm_y[y_argmax] +
            2 * jnp.sqrt(norm_x[x_argmax] * norm_y[y_argmax])
        )
        return 1.0 / max_bound
      raise NotImplementedError(
          "Using max_bound as scaling factor for "
          "the cost matrix when the cost is not squared euclidean "
          "is not implemented."
      )
    raise ValueError(f"Scaling {self._scale_cost} not implemented.")

  @property
  def kernel_matrix(self) -> Optional[jnp.ndarray]:  # noqa: D102
    return jnp.exp(-self.cost_matrix / self.epsilon)

  @property
  def shape(self) -> Tuple[int, int]:  # noqa: D102
    return self.x.shape[0], self.y.shape[0]

  @property
  def dtype(self) -> jnp.dtype:  # noqa: D102
    return self.x.dtype

  @property
  def is_symmetric(self) -> bool:  # noqa: D102
    return self.y is None or (
        jnp.all(self.x.shape == self.y.shape) and jnp.all(self.x == self.y)
    )

  @property
  def is_squared_euclidean(self) -> bool:  # noqa: D102
    return isinstance(self.cost_fn, costs.SqEuclidean)

  @property
  def can_LRC(self):  # noqa: D102
    return self.is_squared_euclidean and self._check_LRC_dim

  @property
  def _check_LRC_dim(self):
    (n, m), d = self.shape, self.x.shape[1]
    return n * m > (n + m) * d

  @property
  def cost_rank(self) -> int:  # noqa: D102
    return self.x.shape[1]

  @property
  def batch_size(self) -> Optional[int]:
    """Batch size for online mode."""
    if self._batch_size is None:
      return None
    n, m = self.shape
    return min(n, m, self._batch_size)

  @property
  def is_online(self) -> bool:
    """Whether the cost/kernel is computed on-the-fly."""
    return self.batch_size is not None
