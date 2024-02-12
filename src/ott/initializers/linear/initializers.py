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
import abc
from typing import Any, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

from ott import utils
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem

__all__ = [
    "DefaultInitializer", "GaussianInitializer", "SortingInitializer",
    "SubsampleInitializer"
]


@jax.tree_util.register_pytree_node_class
class SinkhornInitializer(abc.ABC):
  """Base class for Sinkhorn initializers."""

  @abc.abstractmethod
  def init_dual_a(
      self,
      ot_prob: linear_problem.LinearProblem,
      lse_mode: bool,
      rng: Optional[jax.Array] = None,
  ) -> jnp.ndarray:
    """Initialize Sinkhorn potential/scaling f_u.

    Args:
      ot_prob: Linear OT problem.
      lse_mode: Return potential if ``True``, scaling if ``False``.
      rng: Random number generator for stochastic initializers.

    Returns:
      potential/scaling, array of size ``[n,]``.
    """

  @abc.abstractmethod
  def init_dual_b(
      self,
      ot_prob: linear_problem.LinearProblem,
      lse_mode: bool,
      rng: Optional[jax.Array] = None,
  ) -> jnp.ndarray:
    """Initialize Sinkhorn potential/scaling g_v.

    Args:
      ot_prob: Linear OT problem.
      lse_mode: Return potential if ``True``, scaling if ``False``.
      rng: Random number generator for stochastic initializers.

    Returns:
      potential/scaling, array of size ``[m,]``.
    """

  def __call__(
      self,
      ot_prob: linear_problem.LinearProblem,
      a: Optional[jnp.ndarray],
      b: Optional[jnp.ndarray],
      lse_mode: bool,
      rng: Optional[jax.Array] = None,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Initialize Sinkhorn potentials/scalings f_u and g_v.

    Args:
      ot_prob: Linear OT problem.
      a: Initial potential/scaling f_u.
        If ``None``, it will be initialized using :meth:`init_dual_a`.
      b: Initial potential/scaling g_v.
        If ``None``, it will be initialized using :meth:`init_dual_b`.
      lse_mode: Return potentials if ``True``, scalings if ``False``.
      rng: Random number generator for stochastic initializers.

    Returns:
      The initial potentials/scalings.
    """
    rng = utils.default_prng_key(rng)
    rng_x, rng_y = jax.random.split(rng, 2)
    n, m = ot_prob.geom.shape
    if a is None:
      a = self.init_dual_a(ot_prob, lse_mode=lse_mode, rng=rng_x)
    if b is None:
      b = self.init_dual_b(ot_prob, lse_mode=lse_mode, rng=rng_y)

    assert a.shape == (
        n,
    ), f"Expected `f_u` to have shape `{n,}`, found `{a.shape}`."
    assert b.shape == (
        m,
    ), f"Expected `g_v` to have shape `{m,}`, found `{b.shape}`."

    # cancel dual variables for zero weights
    a = jnp.where(ot_prob.a > 0.0, a, -jnp.inf if lse_mode else 0.0)
    b = jnp.where(ot_prob.b > 0.0, b, -jnp.inf if lse_mode else 0.0)

    return a, b

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:  # noqa: D102
    return [], {}

  @classmethod
  def tree_unflatten(  # noqa: D102
      cls, aux_data: Dict[str, Any], children: Sequence[Any]
  ) -> "SinkhornInitializer":
    return cls(*children, **aux_data)


@jax.tree_util.register_pytree_node_class
class DefaultInitializer(SinkhornInitializer):
  """Default initialization of Sinkhorn dual potentials/primal scalings."""

  def init_dual_a(  # noqa: D102
      self,
      ot_prob: linear_problem.LinearProblem,
      lse_mode: bool,
      rng: Optional[jax.Array] = None,
  ) -> jnp.ndarray:
    del rng
    return jnp.zeros_like(ot_prob.a) if lse_mode else jnp.ones_like(ot_prob.a)

  def init_dual_b(  # noqa: D102
      self,
      ot_prob: linear_problem.LinearProblem,
      lse_mode: bool,
      rng: Optional[jax.Array] = None,
  ) -> jnp.ndarray:
    del rng
    return jnp.zeros_like(ot_prob.b) if lse_mode else jnp.ones_like(ot_prob.b)


@jax.tree_util.register_pytree_node_class
class GaussianInitializer(DefaultInitializer):
  """Gaussian initializer :cite:`thornton2022rethinking:22`.

  Compute Gaussian approximations of each
  :class:`~ott.geometry.pointcloud.PointCloud`, then compute closed from
  Kantorovich potential between Gaussian approximations using Brenier's theorem
  (adapt convex/Brenier potential to Kantorovich). Use this Gaussian potential
  to initialize Sinkhorn potentials/scalings.
  """

  def init_dual_a(  # noqa: D102
      self,
      ot_prob: linear_problem.LinearProblem,
      lse_mode: bool,
      rng: Optional[jax.Array] = None,
  ) -> jnp.ndarray:
    # import Gaussian here due to circular imports
    from ott.tools.gaussian_mixture import gaussian

    del rng
    assert isinstance(
        ot_prob.geom, pointcloud.PointCloud
    ), "Gaussian initializer valid only for pointcloud geoms."

    x, y = ot_prob.geom.x, ot_prob.geom.y
    a, b = ot_prob.a, ot_prob.b

    gaussian_a = gaussian.Gaussian.from_samples(x, weights=a)
    gaussian_b = gaussian.Gaussian.from_samples(y, weights=b)
    # Brenier potential for cost ||x-y||^2/2, multiply by two for ||x-y||^2
    f_potential = 2 * gaussian_a.f_potential(dest=gaussian_b, points=x)
    f_potential = f_potential - jnp.mean(f_potential)
    return f_potential if lse_mode else ot_prob.geom.scaling_from_potential(
        f_potential
    )


@jax.tree_util.register_pytree_node_class
class SortingInitializer(DefaultInitializer):
  """Sorting initializer :cite:`thornton2022rethinking:22`.

  Solve non-regularized OT problem via sorting, then compute potential through
  iterated minimum on C-transform and use this potential to initialize
  regularized potential.

  Args:
    vectorized_update: Whether to use vectorized loop.
    tolerance: DualSort convergence threshold.
    max_iter: Max DualSort steps.
  """

  def __init__(
      self,
      vectorized_update: bool = True,
      tolerance: float = 1e-2,
      max_iter: int = 100
  ):
    super().__init__()
    self.tolerance = tolerance
    self.max_iter = max_iter
    self.vectorized_update = vectorized_update

  def _init_sorting_dual(
      self, modified_cost: jnp.ndarray, init_f: jnp.ndarray
  ) -> jnp.ndarray:
    """Run DualSort algorithm.

    Args:
      modified_cost: cost matrix minus diagonal column-wise.
      init_f: potential f, array of size n. This is the starting potential,
        which is then updated to make the init potential, so an init of an init.

    Returns:
      potential f, array of size n.
    """

    def body_fn(
        state: Tuple[jnp.ndarray, float, int]
    ) -> Tuple[jnp.ndarray, float, int]:
      prev_f, _, it = state
      new_f = fn(prev_f, modified_cost)
      diff = jnp.sum((new_f - prev_f) ** 2)
      it += 1
      return new_f, diff, it

    def cond_fn(state: Tuple[jnp.ndarray, float, int]) -> bool:
      _, diff, it = state
      return jnp.logical_and(diff > self.tolerance, it < self.max_iter)

    fn = _vectorized_update if self.vectorized_update else _coordinate_update
    state = (init_f, jnp.inf, 0)  # init, error, iter
    f_potential, _, _ = jax.lax.while_loop(
        cond_fun=cond_fn, body_fun=body_fn, init_val=state
    )

    return f_potential

  def init_dual_a(
      self,
      ot_prob: linear_problem.LinearProblem,
      lse_mode: bool,
      rng: Optional[jax.Array] = None,
      init_f: Optional[jnp.ndarray] = None,
  ) -> jnp.ndarray:
    """Apply DualSort algorithm.

    Args:
      ot_prob: OT problem between discrete distributions.
      lse_mode: Return potential if ``True``, scaling if ``False``.
      rng: Random number generator for stochastic initializers, unused.
      init_f: potential f, array of size ``[n,]``. This is the starting
        potential, which is then updated to make the init potential,
        so an init of an init.

    Returns:
      potential/scaling f_u, array of size ``[n,]``.
    """
    del rng
    assert not ot_prob.geom.is_online, \
        "Sorting initializer does not work for online geometry."
    # check for sorted x, y requires point cloud and could slow initializer
    cost_matrix = ot_prob.geom.cost_matrix

    assert cost_matrix.shape[0] == cost_matrix.shape[
        1], "Requires square cost matrix."

    modified_cost = cost_matrix - jnp.diag(cost_matrix)[None, :]

    n = cost_matrix.shape[0]
    init_f = jnp.zeros(n) if init_f is None else init_f

    f_potential = self._init_sorting_dual(modified_cost, init_f)
    f_potential = f_potential - jnp.mean(f_potential)

    return f_potential if lse_mode else ot_prob.geom.scaling_from_potential(
        f_potential
    )

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:  # noqa: D102
    return ([], {
        "tolerance": self.tolerance,
        "max_iter": self.max_iter,
        "vectorized_update": self.vectorized_update
    })


@jax.tree_util.register_pytree_node_class
class SubsampleInitializer(DefaultInitializer):
  """Subsample initializer :cite:`thornton2022rethinking:22`.

  Subsample each :class:`~ott.geometry.pointcloud.PointCloud`, then compute
  :class:`Sinkhorn potential <ott.problems.linear.potentials.DualPotentials>`
  from the subsampled approximations and use this potential to initialize
  Sinkhorn potentials/scalings for the original problem.

  Args:
    subsample_n_x: number of points to subsample from the first measure in
      :class:`~ott.geometry.pointcloud.PointCloud`.
    subsample_n_y: number of points to subsample from the second measure in
      :class:`~ott.geometry.pointcloud.PointCloud`.
      If ``None``, use ``subsample_n_x``.
    kwargs: Keyword arguments for
      :class:`~ott.solvers.linear.sinkhorn.Sinkhorn`.
  """

  def __init__(
      self,
      subsample_n_x: int,
      subsample_n_y: Optional[int] = None,
      **kwargs: Any,
  ):
    super().__init__()
    self.subsample_n_x = subsample_n_x
    self.subsample_n_y = subsample_n_y or subsample_n_x
    self.sinkhorn_kwargs = kwargs

  def init_dual_a(  # noqa: D102
      self,
      ot_prob: linear_problem.LinearProblem,
      lse_mode: bool,
      rng: Optional[jax.Array] = None,
  ) -> jnp.ndarray:
    from ott.solvers import linear

    assert isinstance(
        ot_prob.geom, pointcloud.PointCloud
    ), "Subsample initializer valid only for pointcloud geom."
    rng = utils.default_prng_key(rng)
    rng_x, rng_y = jax.random.split(rng, 2)

    x, y = ot_prob.geom.x, ot_prob.geom.y
    a, b = ot_prob.a, ot_prob.b

    # subsample
    sub_x = jax.random.choice(
        rng_x, a=x, shape=(self.subsample_n_x,), replace=True, p=a, axis=0
    )
    sub_y = jax.random.choice(
        rng_y, a=y, shape=(self.subsample_n_y,), replace=True, p=b, axis=0
    )

    # create subsampled point cloud geometry
    sub_geom = pointcloud.PointCloud(
        sub_x,
        sub_y,
        epsilon=ot_prob.geom.epsilon,
        scale_cost=ot_prob.geom._scale_cost,
        cost_fn=ot_prob.geom.cost_fn
    )

    # run sinkhorn
    subsample_sink_out = linear.solve(sub_geom, **self.sinkhorn_kwargs)

    # interpolate potentials
    dual_potentials = subsample_sink_out.to_dual_potentials()
    f_potential = jax.vmap(dual_potentials.f)(x)

    return f_potential if lse_mode else ot_prob.geom.scaling_from_potential(
        f_potential
    )

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:  # noqa: D102
    return ([], {
        "subsample_n_x": self.subsample_n_x,
        "subsample_n_y": self.subsample_n_y,
        **self.sinkhorn_kwargs
    })


def _vectorized_update(
    f: jnp.ndarray, modified_cost: jnp.ndarray
) -> jnp.ndarray:
  """Inner loop DualSort Update.

  Args:
    f: potential f, array of size n.
    modified_cost: cost matrix minus diagonal column-wise.

  Returns:
    updated potential vector, f.
  """
  return jnp.min(modified_cost + f[None, :], axis=1)


def _coordinate_update(
    f: jnp.ndarray, modified_cost: jnp.ndarray
) -> jnp.ndarray:
  """Coordinate-wise updates within inner loop.

  Args:
    f: potential f, array of size n.
    modified_cost: cost matrix minus diagonal column-wise.

  Returns:
    updated potential vector, f.
  """

  def body_fn(i: int, f: jnp.ndarray) -> jnp.ndarray:
    new_f = jnp.min(modified_cost[i, :] + f)
    return f.at[i].set(new_f)

  return jax.lax.fori_loop(0, len(f), body_fn, f)
