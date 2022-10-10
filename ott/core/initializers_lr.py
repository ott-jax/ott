import functools
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import jax
import numpy as np
from jax import numpy as jnp
from typing_extensions import Literal

from ott.core import _math_utils as mu
from ott.geometry import geometry, low_rank, pointcloud

__all__ = [
    "RandomInitializer", "Rank2Initializer", "KMeansInitializer",
    "GeneralizedKMeansInitializer"
]

if TYPE_CHECKING:
  from ott.core import (
      gromov_wasserstein,
      linear_problems,
      quad_problems,
      sinkhorn,
      sinkhorn_lr,
  )
  Problem_t = Union[linear_problems.LinearProblem,
                    quad_problems.QuadraticProblem]
else:
  Problem_t = "Union[linear_problems.LinearProblem, " \
              "quad_problems.QuadraticProblem]"


@jax.tree_util.register_pytree_node_class
class LRInitializer(ABC):
  """Low-rank initializer for linear/quadratic problems.

  Args:
    rank: Rank of the factorization.
    kwargs: Additional keyword arguments.
  """

  def __init__(self, rank: int, **kwargs: Any):
    self._rank = rank
    self._kwargs = kwargs

  @abstractmethod
  def init_q(
      self,
      ot_prob: Problem_t,
      key: jnp.ndarray,
      *,
      init_g: jnp.ndarray,
      **kwargs: Any,
  ) -> jnp.ndarray:
    """Initialize the low-rank factor :math:`Q`.

    Args:
      ot_prob: OT problem.
      key: Random key for seeding.
      kwargs: Additional keyword arguments.

    Returns:
      Array of shape ``[n, rank]``.
    """

  @abstractmethod
  def init_r(
      self,
      ot_prob: Problem_t,
      key: jnp.ndarray,
      *,
      init_g: jnp.ndarray,
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
      ot_prob: Problem_t,
      key: jnp.ndarray,
      **kwargs: Any,
  ) -> jnp.ndarray:
    """Initialize the low-rank factor :math:`g`.

    Args:
      ot_prob: OT problem.
      key: Random key for seeding.
      kwargs: Additional keyword arguments.

    Returns:
      Array of shape ``[rank,]``.
    """

  @classmethod
  def from_solver(
      cls,
      solver: Union['sinkhorn_lr.LRSinkhorn',
                    'gromov_wasserstein.GromovWasserstein'],
      *,
      kind: Literal["random", "rank2", "k-means", "generalized-k-means"],
      **kwargs: Any,
  ) -> 'LRInitializer':
    """Create a low-rank initializer from a linear or quadratic solver.

    Args:
      solver: Low-rank linear or quadratic solver.
      kind: Which initializer to instantiate.
      kwargs: Keyword arguments when creating the initializer.

    Returns:
      The low-rank initializer.
    """
    from ott.core import gromov_wasserstein

    if isinstance(solver, gromov_wasserstein.GromovWasserstein):
      assert solver.is_low_rank, "GW solver is not low-rank."
      lin_sol = solver.linear_ot_solver
    else:
      lin_sol = solver

    rank = solver.rank
    sinkhorn_kwargs = {
        "norm_error": lin_sol._norm_error,
        "lse_mode": lin_sol.lse_mode,
        "jit": lin_sol.jit,
        "implicit_diff": lin_sol.implicit_diff,
        "use_danskin": lin_sol.use_danskin
    }

    if kind == "random":
      return RandomInitializer(rank, **kwargs)
    if kind == "rank2":
      return Rank2Initializer(rank, **kwargs)
    if kind == "k-means":
      return KMeansInitializer(rank, sinkhorn_kwargs=sinkhorn_kwargs, **kwargs)
    if kind == "generalized-k-means":
      return GeneralizedKMeansInitializer(
          rank, sinkhorn_kwargs=sinkhorn_kwargs, **kwargs
      )
    raise NotImplementedError(f"Initializer `{kind}` is not implemented.")

  def __call__(
      self,
      ot_prob: Problem_t,
      q: Optional[jnp.ndarray] = None,
      r: Optional[jnp.ndarray] = None,
      g: Optional[jnp.ndarray] = None,
      *,
      key: Optional[jnp.ndarray] = None,
      **kwargs: Any
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Initialize the factors :math:`Q`, :math:`R` and :math:`g`.

    Args:
      ot_prob: OT problem.
      q: Factor of shape ``[n, rank]``. If `None`, it will be initialized
        using :meth:`init_q`.
      r: Factor of shape ``[m, rank]``. If `None`, it will be initialized
        using :meth:`init_r`.
      g: Factor of shape ``[rank,]``. If `None`, it will be initialized
        using :meth:`init_g`.
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
    return [], {**self._kwargs, "rank": self.rank}

  @classmethod
  def tree_unflatten(
      cls, aux_data: Dict[str, Any], children: Sequence[Any]
  ) -> "LRInitializer":
    return cls(*children, **aux_data)


@jax.tree_util.register_pytree_node_class
class RandomInitializer(LRInitializer):
  """Low-rank Sinkhorn factorization using random factors.

  Args:
    rank: Rank of the factorization.
    kwargs: Additional keyword arguments.
  """

  def init_q(
      self,
      ot_prob: Problem_t,
      key: jnp.ndarray,
      *,
      init_g: jnp.ndarray,
      **kwargs: Any,
  ) -> jnp.ndarray:
    del kwargs, init_g
    a = ot_prob.a
    init_q = jnp.abs(jax.random.normal(key, (a.shape[0], self.rank)))
    return a[:, None] * (init_q / jnp.sum(init_q, axis=1, keepdims=True))

  def init_r(
      self,
      ot_prob: Problem_t,
      key: jnp.ndarray,
      *,
      init_g: jnp.ndarray,
      **kwargs: Any,
  ) -> jnp.ndarray:
    del kwargs, init_g
    b = ot_prob.b
    init_r = jnp.abs(jax.random.normal(key, (b.shape[0], self.rank)))
    return b[:, None] * (init_r / jnp.sum(init_r, axis=1, keepdims=True))

  def init_g(
      self,
      ot_prob: Problem_t,
      key: jnp.ndarray,
      **kwargs: Any,
  ) -> jnp.ndarray:
    del kwargs
    init_g = jnp.abs(jax.random.uniform(key, (self.rank,))) + 1.
    return init_g / jnp.sum(init_g)


@jax.tree_util.register_pytree_node_class
class Rank2Initializer(LRInitializer):
  """Low-rank Sinkhorn factorization using rank-2 factors :cite:`scetbon:21`.

  Args:
    rank: Rank of the factorization.
    kwargs: Additional keyword arguments.
  """

  def _compute_factor(
      self,
      ot_prob: Problem_t,
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
      ot_prob: Problem_t,
      key: jnp.ndarray,
      *,
      init_g: jnp.ndarray,
      **kwargs: Any,
  ) -> jnp.ndarray:
    del key, kwargs
    return self._compute_factor(ot_prob, init_g, which="q")

  def init_r(
      self,
      ot_prob: Problem_t,
      key: jnp.ndarray,
      *,
      init_g: jnp.ndarray,
      **kwargs: Any,
  ) -> jnp.ndarray:
    del key, kwargs
    return self._compute_factor(ot_prob, init_g, which="r")

  def init_g(
      self,
      ot_prob: Problem_t,
      key: jnp.ndarray,
      **kwargs: Any,
  ) -> jnp.ndarray:
    del key, kwargs
    return jnp.ones((self.rank,)) / self.rank


@jax.tree_util.register_pytree_node_class
class KMeansInitializer(LRInitializer):
  """K-means initializer for low-rank Sinkhorn :cite:`scetbon:22b`.

  Applicable for :class:`~ott.geometry.pointcloud.PointCloud` and
  :class:`~ott.geometry.low_rank.LRCGeometry`.

  Args:
    rank: Rank of the factorization.
    min_iterations: Minimum number of k-means iterations.
    max_iterations: Maximum number of k-means iterations.
    sinkhorn_kwargs: Keyword arguments for :class:`~ott.core.sinkhorn.Sinkhorn`.
    kwargs: Keyword arguments for :func:`~ott.tools.k_means.k_means`.
  """

  def __init__(
      self,
      rank: int,
      min_iterations: int = 100,
      max_iterations: int = 100,
      sinkhorn_kwargs: Optional[Mapping[str, Any]] = None,
      **kwargs: Any
  ):
    super().__init__(rank, **kwargs)
    self._min_iter = min_iterations
    self._max_iter = max_iterations
    self._sinkhorn_kwargs = {} if sinkhorn_kwargs is None else sinkhorn_kwargs

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
      ot_prob: Problem_t,
      key: jnp.ndarray,
      *,
      init_g: jnp.ndarray,
      which: Literal["q", "r"],
      **kwargs: Any,
  ) -> jnp.ndarray:
    from ott.core import linear_problems, quad_problems, sinkhorn
    from ott.tools import k_means

    del kwargs
    jit = self._sinkhorn_kwargs.get("jit", True)
    fn = functools.partial(
        k_means.k_means,
        min_iterations=self._min_iter,
        max_iterations=self._max_iter,
        **self._kwargs
    )
    fn = jax.jit(fn, static_argnames="k") if jit else fn

    if isinstance(ot_prob, quad_problems.QuadraticProblem):
      geom = ot_prob.geom_xx if which == "q" else ot_prob.geom_yy
    else:
      geom = ot_prob.geom
    arr = self._extract_array(geom, first=which == "q")
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
      ot_prob: Problem_t,
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
      ot_prob: Problem_t,
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
      ot_prob: Problem_t,
      key: jnp.ndarray,
      **kwargs: Any,
  ) -> jnp.ndarray:
    del key, kwargs
    return jnp.ones((self.rank,)) / self.rank

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    children, aux_data = super().tree_flatten()
    aux_data["sinkhorn_kwargs"] = self._sinkhorn_kwargs
    aux_data["min_iterations"] = self._min_iter
    aux_data["max_iterations"] = self._max_iter
    return children, aux_data


class GeneralizedKMeansInitializer(KMeansInitializer):
  """Generalized k-means initializer :cite:`scetbon:22b`.

  Applicable for any :class:`~ott.geometry.geometry.Geometry` with a
  square shape.

  Args:
    rank: Rank of the factorization.
    gamma: The (inverse of) gradient step size used by mirror descent.
    min_iterations: Minimum number of iterations.
    max_iterations: Maximum number of iterations.
    inner_iterations: Number of iterations used by the algorithm before
      re-evaluating progress.
    threshold: Convergence threshold.
    sinkhorn_kwargs: Keyword arguments for :class:`~ott.core.sinkhorn.Sinkhorn`.
  """

  def __init__(
      self,
      rank: int,
      gamma: float = 10.,
      min_iterations: int = 0,
      max_iterations: int = 100,
      inner_iterations: int = 10,
      threshold: float = 1e-6,
      sinkhorn_kwargs: Optional[Mapping[str, Any]] = None,
  ):
    super().__init__(
        rank,
        sinkhorn_kwargs=sinkhorn_kwargs,
        # below argument are stored in `_kwargs`
        gamma=gamma,
        min_iterations=min_iterations,
        max_iterations=max_iterations,
        inner_iterations=inner_iterations,
        threshold=threshold,
    )

  class Constants(NamedTuple):  # noqa: D106
    solver: "sinkhorn.Sinkhorn"
    geom: geometry.Geometry  # (n, n)
    marginal: jnp.ndarray  # (n,)
    g: jnp.ndarray  # (r,)
    gamma: float
    threshold: float

  class State(NamedTuple):  # noqa: D106
    factor: jnp.ndarray
    criterions: jnp.ndarray
    crossed_threshold: bool

  def _compute_factor(
      self,
      ot_prob: Problem_t,
      key: jnp.ndarray,
      *,
      init_g: jnp.ndarray,
      which: Literal["q", "r"],
      **kwargs: Any,
  ) -> jnp.ndarray:
    from ott.core import fixed_point_loop, linear_problems, sinkhorn

    def init_fn() -> GeneralizedKMeansInitializer.State:
      n = geom.shape[0]
      factor = jnp.abs(jax.random.normal(key, (n, self.rank))) + 1.  # (n, r)
      factor *= consts.marginal[:, None] / jnp.sum(
          factor, axis=1, keepdims=True
      )

      return self.State(
          factor,
          criterions=-jnp.ones(outer_iterations),
          crossed_threshold=False
      )

    # see the explanation in `ott.core.sinkhorn_lr`
    def converged(
        state: GeneralizedKMeansInitializer.State,
        consts: GeneralizedKMeansInitializer.Constants, iteration: int
    ) -> bool:

      def conv_crossed(prev_err: float, curr_err: float) -> bool:
        return jnp.logical_and(
            prev_err < consts.threshold, curr_err < consts.threshold
        )

      def conv_not_crossed(prev_err: float, curr_err: float) -> bool:
        return jnp.logical_and(curr_err < prev_err, curr_err < consts.threshold)

      it = iteration // inner_iterations
      return jax.lax.cond(
          state.crossed_threshold, conv_crossed, conv_not_crossed,
          state.criterions[it - 2], state.criterions[it - 1]
      )

    def diverged(
        state: GeneralizedKMeansInitializer.State, iteration: int
    ) -> bool:
      it = iteration // inner_iterations
      return jnp.logical_not(jnp.isfinite(state.criterions[it - 1]))

    def cond_fn(
        iteration: int,
        consts: GeneralizedKMeansInitializer.Constants,
        state: GeneralizedKMeansInitializer.State,
    ) -> bool:
      return jnp.logical_or(
          iteration <= 2,
          jnp.logical_and(
              jnp.logical_not(diverged(state, iteration)),
              jnp.logical_not(converged(state, consts, iteration))
          )
      )

    def body_fn(
        iteration: int, consts: GeneralizedKMeansInitializer.Constants,
        state: GeneralizedKMeansInitializer.State, compute_error: bool
    ) -> GeneralizedKMeansInitializer.State:
      del compute_error
      it = iteration // inner_iterations

      grad = consts.geom.apply_cost(state.factor, axis=1)  # (n, r)
      grad = grad + consts.geom.apply_cost(state.factor, axis=0)  # (n, r)
      grad = grad / consts.g

      norm = jnp.max(jnp.abs(grad)) ** 2
      gamma = consts.gamma / norm
      eps = 1. / gamma

      cost = grad - eps * mu.safe_log(state.factor)  # (n, r)
      cost = geometry.Geometry(
          cost_matrix=cost,
          epsilon=eps,
      )
      problem = linear_problems.LinearProblem(
          cost, a=consts.marginal, b=consts.g
      )

      out = consts.solver(problem)
      new_factor = out.matrix

      criterion = ((1 / gamma) ** 2) * (
          mu.kl(new_factor, state.factor) + mu.kl(state.factor, new_factor)
      )
      crossed_threshold = jnp.logical_or(
          state.crossed_threshold,
          jnp.logical_and(
              state.criterions[it - 1] >= consts.threshold,
              criterion < consts.threshold
          )
      )

      return self.State(
          factor=new_factor,
          criterions=state.criterions.at[it].set(criterion),
          crossed_threshold=crossed_threshold
      )

    del kwargs
    from ott.core import quad_problems

    if isinstance(ot_prob, quad_problems.QuadraticProblem):
      geom = ot_prob.geom_xx if which == "q" else ot_prob.geom_yy
    else:
      geom = ot_prob.geom
    assert geom.shape[0] == geom.shape[
        1], f"Expected the shape to be square, found `{geom.shape}`."

    inner_iterations = self._kwargs["inner_iterations"]
    outer_iterations = np.ceil(self._max_iter / inner_iterations).astype(int)
    force_scan = self._min_iter == self._max_iter
    fixpoint_fn = (
        fixed_point_loop.fixpoint_iter
        if force_scan else fixed_point_loop.fixpoint_iter_backprop
    )

    consts = self.Constants(
        solver=sinkhorn.Sinkhorn(**self._sinkhorn_kwargs),
        geom=geom._set_scale_cost("max_cost"),
        marginal=ot_prob.a if which == "q" else ot_prob.b,
        g=init_g,
        gamma=self._kwargs["gamma"],
        threshold=self._kwargs["threshold"],
    )

    return fixpoint_fn(
        cond_fn,
        body_fn,
        min_iterations=self._min_iter,
        max_iterations=self._max_iter,
        inner_iterations=inner_iterations,
        constants=consts,
        state=init_fn(),
    ).factor
