import functools
from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

import jax
import jax.scipy as jsp
from jax import numpy as jnp
from typing_extensions import Literal

from ott.core import linear_problems
from ott.geometry import geometry, low_rank, pointcloud

__all__ = [
    "RandomInitializer", "Rank2Initializer", "KMeansInitializer",
    "GeneralizedKMeansInitializer"
]


# TODO(michalk8): move to math utils
def kl(q1: jnp.ndarray, q2: jnp.ndarray) -> float:
  res_1 = -jsp.special.entr(q1)
  res_2 = q1 * safe_log(q2)
  return jnp.sum(res_1 - res_2)


def safe_log(x: jnp.ndarray, *, eps: Optional[float] = None) -> jnp.ndarray:
  if eps is None:
    eps = jnp.finfo(x.dtype).tiny
  return jnp.where(x > 0., jnp.log(x), jnp.log(eps))


@jax.tree_util.register_pytree_node_class
class LRSinkhornInitializer(ABC):
  """Low-rank Sinkhorn initializer.

  Args:
    rank: Rank of the factorization.
  """

  def __init__(self, rank: int):
    self._rank = rank

  @abstractmethod
  def init_q(
      self,
      ot_prob: linear_problems.LinearProblem,
      key: jnp.ndarray,
      **kwargs: Any,
  ) -> jnp.ndarray:
    """Initialize the low-rank factor :math:`Q`.

    Args:
      ot_prob: Linear OT problem.
      key: Random key for seeding.
      kwargs: Additional keyword arguments.

    Returns:
      Array of shape ``[n, rank]``.
    """

  @abstractmethod
  def init_r(
      self,
      ot_prob: linear_problems.LinearProblem,
      key: jnp.ndarray,
      **kwargs: Any,
  ) -> jnp.ndarray:
    """Initialize the low-rank factor :math:`R`.

    Args:
      ot_prob: Linear OT problem.
      key: Random key for seeding.
      kwargs: Additional keyword arguments.

    Returns:
      Array of shape ``[m, rank]``.
    """

  @abstractmethod
  def init_g(
      self,
      ot_prob: linear_problems.LinearProblem,
      key: jnp.ndarray,
      **kwargs: Any,
  ) -> jnp.ndarray:
    """Initialize the low-rank factor :math:`g`.

    Args:
      ot_prob: Linear OT problem.
      key: Random key for seeding.
      kwargs: Additional keyword arguments.

    Returns:
      Array of shape ``[rank,]``.
    """

  def __call__(
      self,
      ot_prob: Optional[linear_problems.LinearProblem],
      q: Optional[jnp.ndarray] = None,
      r: Optional[jnp.ndarray] = None,
      g: Optional[jnp.ndarray] = None,
      *,
      key: Optional[jnp.ndarray] = None,
      **kwargs: Any
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Initialize the factors :math:`Q`, :math:`R` and :math:`g`.

    Args:
      ot_prob: Linear OT problem.
      q: Factor of shape ``[n, rank]``. If not `None`, :meth:`init_q` will be
        used to initialize the factor.
      r: Array of shape ``[m, rank]``. If not `None`, :meth:`init_r` will be
        used to initialize the factor.
      g: Array of shape ``[rank,]``. If not `None`, :meth:`init_g` will be
        used to initialize the factor.
      key: Random key for seeding.
      kwargs: Additional keyword arguments for :meth:`init_q`, :meth:`init_r`
        and :meth:`init_g`.

    Returns:
      The factors :math:`Q`, :math:`R` and :math:`g`, respectively.
    """
    if key is None:
      key = jax.random.PRNGKey(0)
    key1, key2, key3 = jax.random.split(key, 3)

    if g is None:
      g = self.init_g(ot_prob, key1, **kwargs)
    if q is None:
      q = self.init_q(ot_prob, key2, init_g=g, **kwargs)
    if r is None:
      r = self.init_r(ot_prob, key3, init_g=g, **kwargs)

    assert g.shape == (self.rank,)
    assert q.shape == (ot_prob.a.shape[0], self.rank)
    assert r.shape == (ot_prob.b.shape[0], self.rank)

    return q, r, g

  @property
  def rank(self) -> int:
    """Rank of the transport matrix factorization."""
    return self._rank

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    return [self.rank], {}

  @classmethod
  def tree_unflatten(
      cls, aux_data: Dict[str, Any], children: Sequence[Any]
  ) -> "LRSinkhornInitializer":
    return cls(*children, **aux_data)


@jax.tree_util.register_pytree_node_class
class RandomInitializer(LRSinkhornInitializer):
  """Low-rank Sinkhorn factorization using random factors.

  Args:
    rank: Rank of the factorization.
  """

  def init_q(
      self,
      ot_prob: linear_problems.LinearProblem,
      key: jnp.ndarray,
      **kwargs: Any,
  ) -> jnp.ndarray:
    del kwargs
    a = ot_prob.a
    init_q = jnp.abs(jax.random.normal(key, (a.shape[0], self.rank)))
    return a[:, None] * (init_q / jnp.sum(init_q, axis=1, keepdims=True))

  def init_r(
      self,
      ot_prob: linear_problems.LinearProblem,
      key: jnp.ndarray,
      **kwargs: Any,
  ) -> jnp.ndarray:
    del kwargs
    b = ot_prob.b
    init_r = jnp.abs(jax.random.normal(key, (b.shape[0], self.rank)))
    return b[:, None] * (init_r / jnp.sum(init_r, axis=1, keepdims=True))

  def init_g(
      self,
      ot_prob: linear_problems.LinearProblem,
      key: jnp.ndarray,
      **kwargs: Any,
  ) -> jnp.ndarray:
    del kwargs
    init_g = jnp.abs(jax.random.uniform(key, (self.rank,))) + 1.
    return init_g / jnp.sum(init_g)


@jax.tree_util.register_pytree_node_class
class Rank2Initializer(LRSinkhornInitializer):
  """Low-rank Sinkhorn factorization using rank-2 factors :cite:`scetbon:21`.

  Args:
    rank: Rank of the factorization.
  """

  def _compute_factor(
      self,
      ot_prob: linear_problems.LinearProblem,
      init_g: jnp.ndarray,
      *,
      which: Literal["q", "r"],
  ) -> jnp.ndarray:
    a, b = ot_prob.a, ot_prob.b
    marginal = a if which == "q" else b
    n, r = marginal.shape[0], self.rank

    lambda_1 = jnp.min(
        jnp.array([jnp.min(a), jnp.min(init_g),
                   jnp.min(b)])
    ) * .5

    # normalization to 1 can overflow in i32 (e.g., n=128k)
    # using the formula: r * (r + 1) / 2 will raise:
    # OverflowError: Python int 16384128000 too large to convert to int32
    # normalizing by `jnp.sum()` overflows silently
    g1 = 2. * jnp.arange(1, r + 1) / (r ** 2 + r)
    g2 = (init_g - lambda_1 * g1) / (1. - lambda_1)
    x = 2. * jnp.arange(1, n + 1) / (n ** 2 + n)
    y = (marginal - lambda_1 * x) / (1. - lambda_1)

    return ((lambda_1 * x[:, None] @ g1.reshape(1, -1)) +
            ((1 - lambda_1) * y[:, None] @ g2.reshape(1, -1)))

  def init_q(
      self,
      ot_prob: linear_problems.LinearProblem,
      key: jnp.ndarray,
      *,
      init_g: jnp.ndarray,
      **kwargs: Any,
  ) -> jnp.ndarray:
    del key, kwargs
    return self._compute_factor(ot_prob, init_g, which="q")

  def init_r(
      self,
      ot_prob: linear_problems.LinearProblem,
      key: jnp.ndarray,
      *,
      init_g: jnp.ndarray,
      **kwargs: Any,
  ) -> jnp.ndarray:
    del key, kwargs
    return self._compute_factor(ot_prob, init_g, which="r")

  def init_g(
      self,
      ot_prob: linear_problems.LinearProblem,
      key: jnp.ndarray,
      **kwargs: Any,
  ) -> jnp.ndarray:
    del key, kwargs
    return jnp.ones((self.rank,)) / self.rank


@jax.tree_util.register_pytree_node_class
class KMeansInitializer(LRSinkhornInitializer):
  """K-means initializer for low-rank Sinkhorn :cite:`scetbon:22b`.

  Args:
    rank: Rank of the factorization.
    sinkhorn_kwargs: Keyword arguments for :class:`~ott.core.sinkhorn.Sinkhorn`.
    kwargs: Keyword arguments for :func:`~ott.tools.k_means.k_means`.
  """

  def __init__(
      self,
      rank: int,
      sinkhorn_kwargs: Optional[Mapping[str, Any]] = None,
      **kwargs: Any
  ):
    super().__init__(rank)
    self._sinkhorn_kwargs = {} if sinkhorn_kwargs is None else sinkhorn_kwargs
    self._k_means_kwargs = kwargs

  @staticmethod
  def _extract_array(
      geom: Union[pointcloud.PointCloud, low_rank.LRCGeometry], *, first: bool
  ) -> jnp.ndarray:
    if isinstance(geom, pointcloud.PointCloud):
      return geom.x if first else geom.y
    if isinstance(geom, low_rank.LRCGeometry):
      return geom.cost_1 if first else geom.cost_2
    raise TypeError(
        f"k-means initializer not implemented for `{type(geom).__name__}`."
    )

  def _compute_factor(
      self,
      ot_prob: linear_problems.LinearProblem,
      key: jnp.ndarray,
      *,
      init_g: jnp.ndarray,
      which: Literal["q", "r"],
      **kwargs: Any,
  ) -> jnp.ndarray:
    from ott.core import sinkhorn
    from ott.tools import k_means

    del kwargs
    jit = self._sinkhorn_kwargs.get("jit", True)
    fn = functools.partial(k_means.k_means, **self._k_means_kwargs)
    fn = jax.jit(fn, static_argnames="k") if jit else fn

    arr = self._extract_array(ot_prob.geom, first=which == "q")
    marginals = ot_prob.a if which == "q" else ot_prob.b

    centroids = fn(arr, self.rank, key=key).centroids
    geom = pointcloud.PointCloud(
        arr, centroids, epsilon=0.1, scale_cost="max_cost"
    )

    prob = linear_problems.LinearProblem(geom, marginals, init_g)
    solver = sinkhorn.Sinkhorn(**self._sinkhorn_kwargs)
    return solver(prob).matrix

  def init_q(
      self,
      ot_prob: linear_problems.LinearProblem,
      key: jnp.ndarray,
      *,
      init_g: jnp.ndarray,
      **kwargs: Any,
  ) -> jnp.ndarray:
    return self._compute_factor(
        ot_prob, key, init_g=init_g, which="q", **kwargs
    )

  def init_r(
      self,
      ot_prob: linear_problems.LinearProblem,
      key: jnp.ndarray,
      *,
      init_g: jnp.ndarray,
      **kwargs: Any,
  ) -> jnp.ndarray:
    return self._compute_factor(
        ot_prob, key, init_g=init_g, which="r", **kwargs
    )

  def init_g(
      self,
      ot_prob: linear_problems.LinearProblem,
      key: jnp.ndarray,
      **kwargs: Any,
  ) -> jnp.ndarray:
    del key, kwargs
    return jnp.ones((self.rank,)) / self.rank

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    children, aux_data = super().tree_flatten()
    aux_data["sinkhorn_kwargs"] = self._sinkhorn_kwargs
    return children, {**aux_data, **self._k_means_kwargs}


class GeneralizedKMeansInitializer(KMeansInitializer):

  def __init__(
      self,
      rank: int,
      gamma: float,
      min_iterations: int = 10,
      max_iterations: int = 10,
      threshold: float = 1e-6,
      sinkhorn_kwargs: Optional[Mapping[str, Any]] = None,
  ):
    super().__init__(
        rank,
        sinkhorn_kwargs=sinkhorn_kwargs,
        # below argument are stored in `_k_means_kwargs`
        gamma=gamma,
        min_iterations=min_iterations,
        max_iterations=max_iterations,
        threshold=threshold
    )

  class Constants(NamedTuple):
    solver: "sinkhorn.Sinkhorn"  # noqa: F821
    geom: geometry.Geometry  # (n, n)
    marginal: jnp.ndarray  # (n,)
    g: jnp.ndarray  # (r,)
    gamma: float
    threshold: float

  class State(NamedTuple):
    factor: jnp.ndarray
    criterions: jnp.ndarray

  def _compute_factor(
      self,
      ot_prob: linear_problems.LinearProblem,
      key: jnp.ndarray,
      *,
      init_g: jnp.ndarray,
      which: Literal["q", "r"],
      **kwargs: Any,
  ) -> jnp.ndarray:
    from ott.core import fixed_point_loop, linear_problems, sinkhorn

    def init_fn() -> GeneralizedKMeansInitializer.State:
      n = ot_prob.geom.shape[0]
      factor = jnp.abs(jax.random.normal(key, (n, self.rank))) + 1.  # (n, r)
      factor *= consts.marginal[:, None] / jnp.sum(
          factor, axis=1, keepdims=True
      )

      return self.State(factor, criterions=-jnp.ones(max_iter))

    def cond_fn(
        iteration: int,
        consts: GeneralizedKMeansInitializer.Constants,
        state: GeneralizedKMeansInitializer.State,
    ) -> bool:
      # TODO(michalk8): better convergence criterion
      return jnp.logical_or(
          iteration < 1, state.criterions[iteration - 1] > consts.threshold
      )

    def body_fn(
        iteration: int, consts: GeneralizedKMeansInitializer.Constants,
        state: GeneralizedKMeansInitializer.State, compute_error: bool
    ) -> GeneralizedKMeansInitializer.State:
      del compute_error

      grad = consts.geom.apply_cost(state.factor, axis=1)  # (n, r)
      grad += consts.geom.apply_cost(state.factor, axis=0)  # (n ,r)
      grad /= consts.g

      norm = jnp.max(jnp.abs(grad)) ** 2
      gamma = consts.gamma / norm
      eps = 1. / gamma

      geom = grad - eps * safe_log(state.factor)  # (n, r)
      geom = geometry.Geometry(
          cost_matrix=geom, epsilon=eps, scale_cost="max_cost"
      )

      problem = linear_problems.LinearProblem(
          geom, a=consts.marginal, b=consts.g
      )
      out = consts.solver(problem)

      new_factor = out.matrix
      criterion = ((1 / gamma) ** 2) * (
          kl(new_factor, state.factor) + kl(state.factor, new_factor)
      )
      criterions = state.criterions.at[iteration].set(criterion)

      return self.State(factor=new_factor, criterions=criterions)

    del kwargs
    assert ot_prob.geom.shape[0] == ot_prob.geom.shape[1], "TODO: wrong shape"

    min_iter = self._k_means_kwargs["min_iterations"]
    max_iter = self._k_means_kwargs["max_iterations"]
    force_scan = min_iter == max_iter
    fixpoint_fn = (
        fixed_point_loop.fixpoint_iter
        if force_scan else fixed_point_loop.fixpoint_iter_backprop
    )

    consts = self.Constants(
        solver=sinkhorn.Sinkhorn(**self._sinkhorn_kwargs),
        geom=ot_prob.geom._set_scale_cost("max_cost"),
        marginal=ot_prob.a if which == "q" else ot_prob.b,
        g=init_g,
        gamma=self._k_means_kwargs["gamma"],
        threshold=self._k_means_kwargs["threshold"],
    )

    return fixpoint_fn(
        cond_fn,
        body_fn,
        min_iterations=min_iter,
        max_iterations=max_iter,
        inner_iterations=1,
        constants=consts,
        state=init_fn(),
    ).factor
