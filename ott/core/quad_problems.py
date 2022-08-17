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
"""Classes defining OT problem(s) (objective function + utilities)."""

from typing import Any, Callable, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp

# Because Protocol is not available in Python < 3.8
from typing_extensions import Literal, Protocol

from ott.core import linear_problems, sinkhorn_lr
from ott.geometry import epsilon_scheduler, geometry, low_rank, pointcloud


class Transport(Protocol):
  """Interface for the solution of a transport problem.

  Classes implementing those function do not have to inherit from it, the
  class can however be used in type hints to support duck typing.
  """

  @property
  def matrix(self) -> jnp.ndarray:
    ...

  def apply(self, inputs: jnp.ndarray, axis: int) -> jnp.ndarray:
    ...

  def marginal(self, axis: int = 0) -> jnp.ndarray:
    ...


class Loss(NamedTuple):
  func: Callable[[jnp.ndarray], jnp.ndarray]
  is_linear: bool


class GWLoss(NamedTuple):
  f1: Loss
  f2: Loss
  h1: Loss
  h2: Loss


def make_square_loss() -> GWLoss:
  f1 = Loss(lambda x: x ** 2, is_linear=False)
  f2 = Loss(lambda y: y ** 2, is_linear=False)
  h1 = Loss(lambda x: x, is_linear=True)
  h2 = Loss(lambda y: 2.0 * y, is_linear=True)
  return GWLoss(f1, f2, h1, h2)


def make_kl_loss(clipping_value: float = 1e-8) -> GWLoss:
  f1 = Loss(lambda x: -jax.scipy.special.entr(x) - x, is_linear=False)
  f2 = Loss(lambda y: y, is_linear=True)
  h1 = Loss(lambda x: x, is_linear=True)
  h2 = Loss(lambda y: jnp.log(jnp.clip(y, clipping_value)), is_linear=False)
  return GWLoss(f1, f2, h1, h2)


@jax.tree_util.register_pytree_node_class
class QuadraticProblem:
  """Definition of the quadratic regularized OT problem.

  The quadratic loss of a single OT matrix is assumed to
  have the form given in :cite:`peyre:16`, eq. 4.

  The two geometries below parameterize matrices C and bar{C} in that equation.
  The function L (of two real values) in that equation is assumed
  to match the form given in Eq. 5., with our notations:

  L(x, y) = lin1(x) + lin2(y) - quad1(x) * quad2(y)

  Args:
    geom_xx: the geometry.Geometry object defining the ground geometry / cost
      of the first space.
    geom_yy: the geometry.Geometry object defining the ground geometry / cost
      of the second space.
    geom_xy: the geometry.Geometry object defining the linear penalty term
      for Fused Gromov Wasserstein. If None, the problem reduces to a plain
      Gromov Wasserstein problem.
    fused_penalty: multiplier of the linear term in Fused Gromov Wasserstein,
      i.e. problem = purely quadratic + fused_penalty * linear problem.
      Ignored if ``geom_xy`` is not specified.
    scale_cost: option to rescale the cost matrices:

      - if `True`, use the default for each geometry.
      - if `False`, keep the original scaling in geometries.
      - if :class:`str`, use a specific method available in
        :class:`ott.geometry.geometry.Geometry` or
        :class`ott.geometry.pointcloud.PointCloud`.
      - if `None`, do not scale the cost matrices.

    a: jnp.ndarray[n] representing the probability weights of the samples
      from geom_xx. If None, it will be uniform.
    b: jnp.ndarray[n] representing the probability weights of the samples
      from geom_yy. If None, it will be uniform.
    loss: a 2-tuple of 2-tuples of Callable. The first tuple is the linear
      part of the loss (see in the pydoc of the class lin1, lin2). The second
      one is the quadratic part (quad1, quad2). By default, the loss
      is set as the 4 functions representing the squared Euclidean loss, and
      this property is taken advantage of in subsequent computations. See
      Alternatively, KL loss can be specified in no less optimized way.
    tau_a: if lower that 1.0, defines how much unbalanced the problem is on
      the first marginal.
    tau_b: if lower that 1.0, defines how much unbalanced the problem is on
      the second marginal.
    gw_unbalanced_correction: True (default) if the unbalanced version of
      :cite:`sejourne:21` is used, False if tau_a and tau_b
      only affect the inner Sinkhorn loop.
    ranks: Ranks of the cost matrices, see
      :meth:`~ott.geometry.geometry.Geometry.to_LRCGeometry`. Used when
      geometries are *not* :class:`~ott.geometry.pointcloud.PointCloud` with
      `'sqeucl'` cost function. If `-1`, the geometries will not be converted
      to low-rank. If :class:`tuple`, it specifies the ranks of ``geom_xx``,
      ``geom_yy`` and ``geom_xy``, respectively. If :class:`int`, rank is shared
      across all geometries.
    tolerances: Tolerances used when converting geometries to low-rank. Used when
      geometries are *not* :class:`~ott.geometry.pointcloud.PointCloud` with
      `'sqeucl'` cost. If :class:`float`, it is shared across all geometries.
  """

  def __init__(
      self,
      geom_xx: geometry.Geometry,
      geom_yy: geometry.Geometry,
      geom_xy: Optional[geometry.Geometry] = None,
      fused_penalty: float = 1.0,
      scale_cost: Optional[Union[bool, float, str]] = False,
      a: Optional[jnp.ndarray] = None,
      b: Optional[jnp.ndarray] = None,
      loss: Union[Literal['sqeucl', 'kl'], GWLoss] = 'sqeucl',
      tau_a: Optional[float] = 1.0,
      tau_b: Optional[float] = 1.0,
      gw_unbalanced_correction: bool = True,
      ranks: Union[int, Tuple[int, ...]] = -1,
      tolerances: Union[float, Tuple[float, ...]] = 1e-2,
  ):
    assert fused_penalty > 0, fused_penalty
    self.geom_xx = geom_xx._set_scale_cost(scale_cost)
    self.geom_yy = geom_yy._set_scale_cost(scale_cost)
    self.geom_xy = (
        None if geom_xy is None else geom_xy._set_scale_cost(scale_cost)
    )
    self.fused_penalty = fused_penalty
    self.scale_cost = scale_cost
    self._a = a
    self._b = b
    self.tau_a = tau_a
    self.tau_b = tau_b
    self.gw_unbalanced_correction = gw_unbalanced_correction
    self.ranks = ranks
    self.tolerances = tolerances

    self._loss_name = loss
    if self._loss_name == 'sqeucl':
      self.loss = make_square_loss()
    elif loss == 'kl':
      self.loss = make_kl_loss()
    else:
      self.loss = loss

  @property
  def is_fused(self) -> bool:
    return self.geom_xy is not None

  @property
  def is_low_rank(self) -> bool:
    return (
        isinstance(self.geom_xx, low_rank.LRCGeometry) and
        isinstance(self.geom_yy, low_rank.LRCGeometry) and (
            self.geom_xy is None or
            isinstance(self.geom_xy, low_rank.LRCGeometry)
        )
    )

  @property
  def linear_loss(self) -> Tuple[Loss, Loss]:
    return self.loss.f1, self.loss.f2

  @property
  def quad_loss(self) -> Tuple[Loss, Loss]:
    return self.loss.h1, self.loss.h2

  @property
  def is_balanced(self) -> bool:
    return ((not self.gw_unbalanced_correction) or
            (self.tau_a == 1.0 and self.tau_b == 1.0))

  def tree_flatten(self):
    return ([self.geom_xx, self.geom_yy, self.geom_xy, self._a, self._b], {
        'tau_a': self.tau_a,
        'tau_b': self.tau_b,
        'loss': self._loss_name,
        'fused_penalty': self.fused_penalty,
        'scale_cost': self.scale_cost,
        'gw_unbalanced_correction': self.gw_unbalanced_correction,
        'ranks': self.ranks,
        'tolerances': self.tolerances
    })

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    geoms, (a, b) = children[:3], children[3:]
    return cls(*geoms, a=a, b=b, **aux_data)

  @property
  def a(self) -> jnp.ndarray:
    num_a = self.geom_xx.shape[0]
    return jnp.ones((num_a,)) / num_a if self._a is None else self._a

  @property
  def b(self) -> jnp.ndarray:
    num_b = self.geom_yy.shape[0]
    return jnp.ones((num_b,)) / num_b if self._b is None else self._b

  def marginal_dependent_cost(
      self, marginal_1: jnp.ndarray, marginal_2: jnp.ndarray
  ) -> low_rank.LRCGeometry:
    r"""Initialise cost term that depends on the marginals of the transport.

    Uses the first term in eq. 6, p. 1 of :cite:`peyre:16`.

    Let :math:`p` [num_a,] be the marginal of the transport matrix for samples
    from `geom_xx` and :math:`q` [num_b,] be the marginal of the transport
    matrix for samples from `geom_yy`. `cost_xx` (resp. `cost_yy`) is the
    cost matrix of `geom_xx` (resp. `geom_yy`). The cost term that
    depends on these marginals can be written as:

    `marginal_dep_term` = `lin1`(`cost_xx`) :math:`p \mathbb{1}_{num_b}^T`
                      + (`lin2`(`cost_yy`) :math:`q \mathbb{1}_{num_a}^T)^T`

    Args:
      marginal_1: jnp.ndarray<float>[num_a,], marginal of the transport matrix
       for samples from geom_xx
      marginal_2: jnp.ndarray<float>[num_b,], marginal of the transport matrix
       for samples from geom_yy

    Returns:
      Low-rank geometry.
    """
    if self._loss_name == 'sqeucl':  # quadratic apply, efficient for LR
      tmp1 = self.geom_xx.apply_square_cost(marginal_1, axis=1)
      tmp2 = self.geom_yy.apply_square_cost(marginal_2, axis=1)
    else:
      f1, f2 = self.linear_loss
      tmp1 = apply_cost(self.geom_xx, marginal_1, axis=1, fn=f1)
      tmp2 = apply_cost(self.geom_yy, marginal_2, axis=1, fn=f2)
    x_term = jnp.concatenate((tmp1, jnp.ones_like(tmp1)), axis=1)
    y_term = jnp.concatenate((jnp.ones_like(tmp2), tmp2), axis=1)
    return low_rank.LRCGeometry(cost_1=x_term, cost_2=y_term)

  def cost_unbalanced_correction(
      self,
      transport_matrix: jnp.ndarray,
      marginal_1: jnp.ndarray,
      marginal_2: jnp.ndarray,
      epsilon: float,
      rescale_factor: float,
      delta: float = 1e-9
  ) -> float:
    r"""Calculate cost term from the quadratic divergence when unbalanced.

    In the unbalanced setting (``tau_a < 1.0 or tau_b < 1.0``), the
    introduction of a quadratic divergence :cite:`sejourne:21` adds a term
    to the GW local cost.

    Let :math:`a` [num_a,] be the target weights for samples
    from geom_xx and :math:`b` [num_b,] be the target weights
    for samples from `geom_yy`. Let :math:`P` [num_a, num_b] be the transport
    matrix, :math:`P1` the first marginal and :math:`P^T1` the second marginal.
    The term of the cost matrix coming from the quadratic KL in the
    unbalanced case can be written as:

    `unbalanced_correction_term` =
        :math:`tau_a / (1 - tau_a) * \sum(KL(P1|a))`
        :math:`+ tau_b / (1 - tau_b) * \sum(KL(P^T1|b))`
        :math:`+ epsilon * \sum(KL(P|ab'))`

    Args:
      transport_matrix: jnp.ndarray<float>[num_a, num_b], transport matrix.
      marginal_1: jnp.ndarray<float>[num_a,], marginal of the transport matrix
        for samples from :attr:`geom_xx`.
      marginal_2: jnp.ndarray<float>[num_b,], marginal of the transport matrix
        for samples from :attr:`geom_yy`.
      epsilon: regulariser.
      rescale_factor: scaling factor for the transport matrix.
      delta: small quantity to avoid diverging KLs.

    Returns:
      The cost term.
    """

    def regulariser(tau: float) -> float:
      return tau / (1.0 - tau) if tau != 1.0 else 0

    cost = regulariser(self.tau_a) * jax.scipy.special.xlogy(
        marginal_1, rescale_factor * marginal_1 / jnp.clip(self.a, a_min=delta)
    ).sum()
    cost += regulariser(self.tau_b) * jax.scipy.special.xlogy(
        marginal_2, rescale_factor * marginal_2 / jnp.clip(self.b, a_min=delta)
    ).sum()
    cost += epsilon * jax.scipy.special.xlogy(
        transport_matrix, rescale_factor * transport_matrix /
        jnp.clip(self.a[:, None] * self.b[None, :], a_min=delta)
    ).sum()
    return cost

  def init_transport(self) -> jnp.ndarray:
    """Initialise the transport matrix."""
    # TODO(oliviert, cuturi): consider passing a custom initialization.
    a = jax.lax.stop_gradient(self.a)
    b = jax.lax.stop_gradient(self.b)
    return (
        a[:, None] * b[None, :] if self.is_balanced else a[:, None] *
        b[None, :] / jnp.sqrt(a.sum() * b.sum())
    )

  def init_transport_mass(self) -> float:
    """Initialise the transport mass.

    Returns:
      The sum of the elements of the normalised transport matrix.
    """
    a = jax.lax.stop_gradient(self.a)
    b = jax.lax.stop_gradient(self.b)
    transport_mass = a.sum() * b.sum()
    return (
        transport_mass if self.is_balanced else transport_mass /
        jnp.sqrt(transport_mass)
    )

  def init_linearization(
      self,
      epsilon: Optional[Union[epsilon_scheduler.Epsilon, float]] = None
  ) -> linear_problems.LinearProblem:
    """Initialise a linear problem locally around a naive initializer ab'.

    If the problem is balanced (``tau_a = 1.0 and tau_b = 1.0``),
    the equation of the cost follows eq. 6, p. 1 of :cite:`peyre:16`.

    If the problem is unbalanced (`tau_a<1.0 or tau_b<1.0`), there are two
    possible cases. A first possibility is to introduce a quadratic KL
    divergence on the marginals in the objective as done in :cite:`sejourne:21`
    (``gw_unbalanced_correction = True``), which in turns modifies the
    local cost matrix.

    Alternatively, it could be possible to leave the formulation of the
    local cost unchanged, i.e. follow eq. 6, p. 1 of :cite:`peyre:16`
    (``gw_unbalanced_correction = False``) and include the unbalanced terms
    at the level of the linear problem only.

    Let :math:`P` [num_a, num_b] be the transport matrix, `cost_xx` is the
    cost matrix of `geom_xx` and `cost_yy` is the cost matrix of `geom_yy`.
    `left_x` and `right_y` depend on the loss chosen for GW.
    `gw_unbalanced_correction` is an boolean indicating whether or not the
    unbalanced correction applies.
    The equation of the local cost can be written as:

    `cost_matrix` = `marginal_dep_term`
                + `left_x`(`cost_xx`) :math:`P` `right_y`(`cost_yy`):math:`^T`
                + `unbalanced_correction` * `gw_unbalanced_correction`

    When working with the fused problem, a linear term is added to the cost
    matrix:
    `cost_matrix` += `fused_penalty` * `geom_xy.cost_matrix`

    Args:
      epsilon: An epsilon scheduler or a float passed on to the linearization.

    Returns:
      A linear_problems.LinearProblem, representing local linearization of
      GW problem.
    """
    unbalanced_correction = 0.0
    tmp = self.init_transport()
    marginal_1 = tmp.sum(1)
    marginal_2 = tmp.sum(0)

    # Initialises cost.
    marginal_cost = self.marginal_dependent_cost(marginal_1, marginal_2)

    if not self.is_balanced:
      unbalanced_correction = self.cost_unbalanced_correction(
          tmp, marginal_1, marginal_2, epsilon, 1.0
      )

    h1, h2 = self.quad_loss
    tmp = apply_cost(self.geom_xx, tmp, axis=1, fn=h1)
    tmp = apply_cost(self.geom_yy, tmp.T, axis=1, fn=h2).T
    cost_matrix = (marginal_cost.cost_matrix - tmp + unbalanced_correction)

    # Initialises epsilon for Unbalanced GW according to Sejourne et al (2021).
    if not self.is_balanced:
      transport_mass = marginal_1.sum()
      epsilon = update_epsilon_unbalanced(epsilon, transport_mass)

    cost_matrix += self.fused_penalty * self._fused_cost_matrix

    geom = geometry.Geometry(cost_matrix=cost_matrix, epsilon=epsilon)
    return linear_problems.LinearProblem(
        geom, self.a, self.b, tau_a=self.tau_a, tau_b=self.tau_b
    )

  def init_lr_linearization(
      self, rank: int = 10, **kwargs: Any
  ) -> linear_problems.LinearProblem:
    """Linearizes a Quad problem with a predefined initializer."""
    x_ = self.geom_xx.apply_square_cost(self.a)
    y_ = self.geom_yy.apply_square_cost(self.b)
    geom_ = pointcloud.PointCloud(x_, y_).to_LRCGeometry()
    out = sinkhorn_lr.LRSinkhorn(
        rank=rank, **kwargs
    )(
        linear_problems.LinearProblem(geom_, self.a, self.b)
    )
    return linear_problems.LinearProblem(
        self.update_lr_geom(out),
        self.a,
        self.b,
        tau_a=self.tau_a,
        tau_b=self.tau_b
    )

  def update_lr_geom(
      self, lr_sink: sinkhorn_lr.LRSinkhornOutput
  ) -> geometry.Geometry:
    """Recompute (possibly LRC) linearization using LR Sinkhorn output."""
    marginal_1 = lr_sink.marginal(1)
    marginal_2 = lr_sink.marginal(0)
    marginal_cost = self.marginal_dependent_cost(marginal_1, marginal_2)

    # Extract factors from LR Sinkhorn output
    q, r, inv_sqg = lr_sink.q, lr_sink.r, 1.0 / jnp.sqrt(lr_sink.g)
    # Distribute middle marginal evenly across both factors.
    q, r = q * inv_sqg[None, :], r * inv_sqg[None, :]

    # Handle LRC Geometry case.
    h1, h2 = self.quad_loss
    tmp1 = apply_cost(self.geom_xx, q, axis=1, fn=h1)
    tmp2 = apply_cost(self.geom_yy, r, axis=1, fn=h2)
    if self.is_low_rank:
      geom = low_rank.LRCGeometry(cost_1=tmp1, cost_2=-tmp2)
      geom = low_rank.add_lrc_geom(geom, marginal_cost)
      if self.is_fused:
        geom = low_rank.add_lrc_geom(geom, self.geom_xy)
    else:
      cost_matrix = marginal_cost.cost_matrix - jnp.dot(tmp1, tmp2.T)
      cost_matrix += self.fused_penalty * self._fused_cost_matrix
      geom = geometry.Geometry(cost_matrix=cost_matrix)
    return geom

  def update_linearization(
      self,
      transport: Transport,
      epsilon: Optional[Union[epsilon_scheduler.Epsilon, float]] = None,
      old_transport_mass: float = 1.0
  ) -> linear_problems.LinearProblem:
    """Update linearization of GW problem by updating cost matrix.

    If the problem is balanced (``tau_a = 1.0 and tau_b = 1.0``), the equation
    follows eq. 6, p. 1 of :cite:`peyre:16`.

    If the problem is unbalanced (``tau_a < 1.0 or tau_b < 1.0``), two cases are
    possible, as explained in :meth:`init_linearization` above.
    Finally, it is also possible to consider a Fused Gromov Wasserstein problem.
    Details about the resulting cost matrix are also given in
    :meth:`init_linearization`.

    Args:
      transport: Solution of the linearization of the quadratic problem.
        epsilon: An epsilon scheduler or a float passed on to the linearization.
        old_transport_mass: Sum of the elements of the transport matrix at the
        previous iteration.

    Returns:
      Updated linear OT problem, a new local linearization of GW problem.
    """
    rescale_factor = 1.0
    unbalanced_correction = 0.0

    marginal_1 = transport.marginal(axis=1)
    marginal_2 = transport.marginal(axis=0)
    marginal_cost = self.marginal_dependent_cost(marginal_1, marginal_2)

    if not self.is_balanced:
      # Rescales transport for Unbalanced GW according to Sejourne et al (2021).
      transport_mass = jax.lax.stop_gradient(marginal_1.sum())
      rescale_factor = jnp.sqrt(old_transport_mass / transport_mass)
      unbalanced_correction = self.cost_unbalanced_correction(
          transport.matrix, marginal_1, marginal_2, epsilon, rescale_factor
      )
      # Updates epsilon for Unbalanced GW.
      epsilon = update_epsilon_unbalanced(epsilon, transport_mass)

    h1, h2 = self.quad_loss
    tmp = apply_cost(self.geom_xx, transport.matrix, axis=1, fn=h1)
    tmp = apply_cost(self.geom_yy, tmp.T, axis=1, fn=h2).T

    cost_matrix = marginal_cost.cost_matrix - tmp + unbalanced_correction
    cost_matrix += self.fused_penalty * self._fused_cost_matrix
    cost_matrix *= rescale_factor

    geom = geometry.Geometry(cost_matrix=cost_matrix, epsilon=epsilon)
    return linear_problems.LinearProblem(
        geom, self.a, self.b, tau_a=self.tau_a, tau_b=self.tau_b
    )

  def update_lr_linearization(
      self, lr_sink: sinkhorn_lr.LRSinkhornOutput
  ) -> linear_problems.LinearProblem:
    """Update a Quad problem linearization using a LR Sinkhorn."""
    return linear_problems.LinearProblem(
        self.update_lr_geom(lr_sink),
        self.a,
        self.b,
        tau_a=self.tau_a,
        tau_b=self.tau_b
    )

  @property
  def _fused_cost_matrix(self) -> Union[float, jnp.ndarray]:
    if not self.is_fused:
      return 0.
    if isinstance(
        self.geom_xy, pointcloud.PointCloud
    ) and self.geom_xy.is_online:
      return self.geom_xy._compute_cost_matrix() * self.geom_xy.inv_scale_cost
    return self.geom_xy.cost_matrix

  @property
  def _is_low_rank_convertible(self) -> bool:

    def convertible(geom: geometry.Geometry) -> bool:
      return isinstance(geom, low_rank.LRCGeometry) or (
          isinstance(geom, pointcloud.PointCloud) and geom.is_squared_euclidean
      )

    if self.is_low_rank:
      return True

    geom_xx, geom_yy, geom_xy = self.geom_xx, self.geom_yy, self.geom_xy
    # either explicitly via cost factorization or implicitly (e.g., a PC)
    return self.ranks != 1 or (
        convertible(geom_xx) and convertible(geom_yy) and
        (geom_xy is None or convertible(geom_xy))
    )

  def to_low_rank(self, seed: int = 0) -> "QuadraticProblem":
    """Convert geometries to low-rank.

    Args:
      seed: Random seed.

    Returns:
      Quadratic problem with low-rank geometries.
    """

    def convert(
        vals: Union[int, float, Tuple[Union[int, float], ...]]
    ) -> Tuple[Union[int, float], ...]:
      size = 2 + self.is_fused
      if isinstance(vals, (int, float)):
        return (vals,) * 3
      assert len(vals) == size, vals
      return vals + (None,) * (3 - size)

    if self.is_low_rank:
      return self

    (geom_xx, geom_yy, geom_xy, *children), aux_data = self.tree_flatten()
    (s1, s2, s3) = jax.random.split(jax.random.PRNGKey(seed), 3)[:, 0]
    (r1, r2, r3), (t1, t2, t3) = convert(self.ranks), convert(self.tolerances)

    geom_xx = geom_xx.to_LRCGeometry(rank=r1, tol=t1, seed=s1)
    geom_yy = geom_yy.to_LRCGeometry(rank=r2, tol=t2, seed=s2)
    if self.is_fused:
      if isinstance(
          geom_xy, pointcloud.PointCloud
      ) and geom_xy.is_squared_euclidean:
        geom_xy = geom_xy.to_LRCGeometry(self.fused_penalty)
      else:
        geom_xy = geom_xy.to_LRCGeometry(rank=r3, tol=t3, seed=s3)

    return type(self).tree_unflatten(
        aux_data, [geom_xx, geom_yy, geom_xy] + children
    )


def update_epsilon_unbalanced(epsilon, transport_mass):
  updated_epsilon = epsilon_scheduler.Epsilon.make(epsilon)
  updated_epsilon._scale_epsilon = (
      updated_epsilon._scale_epsilon * transport_mass
  )
  return updated_epsilon


def apply_cost(
    geom: geometry.Geometry, arr: jnp.ndarray, *, axis: int, fn: Loss
) -> jnp.ndarray:
  return geom.apply_cost(arr, axis=axis, fn=fn.func, is_linear=fn.is_linear)
