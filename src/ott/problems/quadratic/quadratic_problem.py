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
from typing import TYPE_CHECKING, Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from ott import utils
from ott.geometry import geometry, low_rank, pointcloud
from ott.problems.linear import linear_problem
from ott.problems.quadratic import quadratic_costs
from ott.types import Transport

if TYPE_CHECKING:
  from ott.solvers.linear import sinkhorn_lr

__all__ = ["QuadraticProblem"]


@jax.tree_util.register_pytree_node_class
class QuadraticProblem:
  r"""Quadratic OT problem.

  The quadratic loss of a single OT matrix is assumed to
  have the form given in :cite:`peyre:16`, eq. 4.

  The two geometries below parameterize matrices :math:`C` and :math:`\bar{C}`
  in that equation. The function :math:`L` (of two real values) in that equation
  is assumed to match the form given in eq. 5., with our notations:

  .. math::
    L(x, y) = f_1(x) + f_2(y) - h_1(x) h_2(y)

  Args:
    geom_xx: Ground geometry of the first space.
    geom_yy: Ground geometry of the second space.
    geom_xy: Geometry defining the linear penalty term for
      fused Gromov-Wasserstein :cite:`vayer:19`. If :obj:`None`, the problem
      reduces to a plain Gromov-Wasserstein problem :cite:`peyre:16`.
    fused_penalty: Multiplier of the linear term in fused Gromov-Wasserstein,
      i.e. ``problem = purely quadratic + fused_penalty * linear problem``.
    scale_cost: How to rescale the cost matrices. If a :class:`str`,
      use specific options available in :class:`~ott.geometry.geometry.Geometry`
      or :class:`~ott.geometry.pointcloud.PointCloud`. If :obj:`None`, keep
      the original scaling.
    a: The first marginal. If :obj:`None`, it will be uniform.
    b: The second marginal. If :obj:`None`, it will be uniform.
    loss: Gromov-Wasserstein loss function, see
      :class:`~ott.problems.quadratic.quadratic_costs.GWLoss` for more
      information.
    tau_a: If :math:`< 1.0`, defines how much unbalanced the problem is on
      the first marginal.
    tau_b: If :math:`< 1.0`, defines how much unbalanced the problem is on
      the second marginal.
    gw_unbalanced_correction: Whether the unbalanced version of
      :cite:`sejourne:21` is used. Otherwise, ``tau_a`` and ``tau_b``
      only affect the inner Sinkhorn loop.
    ranks: Ranks of the cost matrices, see
      :meth:`~ott.geometry.geometry.Geometry.to_LRCGeometry`. Used when
      geometries are *not* :class:`~ott.geometry.pointcloud.PointCloud` with
      `'sqeucl'` cost function. If `-1`, the geometries will not be converted
      to low-rank. If :class:`tuple`, it specifies the ranks of ``geom_xx``,
      ``geom_yy`` and ``geom_xy``, respectively. If :class:`int`, rank is shared
      across all geometries.
    tolerances: Tolerances used when converting geometries to low-rank. Used
      when geometries are not :class:`~ott.geometry.pointcloud.PointCloud` with
      `'sqeucl'` cost. If :class:`float`, it is shared across all geometries.
  """

  def __init__(
      self,
      geom_xx: geometry.Geometry,
      geom_yy: geometry.Geometry,
      geom_xy: Optional[geometry.Geometry] = None,
      fused_penalty: float = 1.0,
      scale_cost: Optional[Union[float, str]] = None,
      a: Optional[jnp.ndarray] = None,
      b: Optional[jnp.ndarray] = None,
      loss: Union[Literal["sqeucl", "kl"], quadratic_costs.GWLoss] = "sqeucl",
      tau_a: float = 1.0,
      tau_b: float = 1.0,
      gw_unbalanced_correction: bool = True,
      ranks: Union[int, Tuple[int, ...]] = -1,
      tolerances: Union[float, Tuple[float, ...]] = 1e-2,
  ):
    if scale_cost is not None:
      geom_xx = geom_xx.set_scale_cost(scale_cost)
      geom_yy = geom_yy.set_scale_cost(scale_cost)
      if geom_xy is not None:
        geom_xy = geom_xy.set_scale_cost(scale_cost)

    self._geom_xx = geom_xx
    self._geom_yy = geom_yy
    self._geom_xy = geom_xy
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
    if self._loss_name == "sqeucl":
      self.loss = quadratic_costs.make_square_loss()
    elif loss == "kl":
      self.loss = quadratic_costs.make_kl_loss()
    else:
      self.loss = loss

  def marginal_dependent_cost(
      self,
      marginal_1: jnp.ndarray,
      marginal_2: jnp.ndarray,
  ) -> low_rank.LRCGeometry:
    r"""Initialize cost term that depends on the marginals of the transport.

    Uses the first term in eq. 6, p. 1 of :cite:`peyre:16`.

    Let :math:`p` be the `[n,]` marginal of the transport matrix for samples
    from :attr:`geom_xx` and :math:`q` the `[m,]` marginal of the
    transport matrix for samples from :attr:`geom_yy`.

    When ``cost_xx`` (resp. ``cost_yy``) is the cost matrix of :attr:`geom_xx`
    (resp. :attr:`geom_yy`), the cost term that depends on these marginals can
    be written as:

    .. math::

      \text{marginal_dep_term} = \text{lin1}(\text{cost_xx}) p \mathbb{1}_{m}^T
                      +  \mathbb{1}_{n}(\text{lin2}(\text{cost_yy}) q)^T

    This helper function instantiates these two low-rank matrices and groups
    them into a single low-rank cost geometry object.

    Args:
      marginal_1: [n,], first marginal of transport matrix.
      marginal_2: [m,], second marginal of transport matrix.

    Returns:
      Low-rank geometry of rank 2, storing normalization constants.
    """
    geom_xx, geom_yy = self.geom_xx, self.geom_yy
    if self._loss_name == "sqeucl":  # quadratic apply, efficient for LR
      tmp1 = geom_xx.apply_square_cost(marginal_1, axis=1)
      tmp2 = geom_yy.apply_square_cost(marginal_2, axis=1)
    else:
      f1, f2 = self.linear_loss
      tmp1 = apply_cost(geom_xx, marginal_1, axis=1, fn=f1)
      tmp2 = apply_cost(geom_yy, marginal_2, axis=1, fn=f2)
    x_term = jnp.concatenate((tmp1, jnp.ones_like(tmp1)), axis=1)
    y_term = jnp.concatenate((jnp.ones_like(tmp2), tmp2), axis=1)
    return low_rank.LRCGeometry(cost_1=x_term, cost_2=y_term)

  def cost_unbalanced_correction(
      self,
      transport_matrix: jnp.ndarray,
      marginal_1: jnp.ndarray,
      marginal_2: jnp.ndarray,
      epsilon: float,
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
      epsilon: entropy regularizer.

    Returns:
      The cost term.
    """

    def regularizer(tau: float) -> float:
      return epsilon * tau / (1.0 - tau)

    marginal_1loga = jsp.special.xlogy(marginal_1, self.a).sum()
    marginal_2logb = jsp.special.xlogy(marginal_2, self.b).sum()

    cost = epsilon * jsp.special.xlogy(transport_matrix, transport_matrix).sum()
    if self.tau_a != 1.0:
      cost += regularizer(
          self.tau_a
      ) * (-jsp.special.entr(marginal_1).sum() - marginal_1loga)
    if self.tau_b != 1.0:
      cost += regularizer(
          self.tau_b
      ) * (-jsp.special.entr(marginal_2).sum() - marginal_2logb)
    return cost

  # TODO(michalk8): highly coupled to the pre-defined initializer, refactor
  def init_transport_mass(self) -> float:
    """Initialize the transport mass.

    Returns:
      The sum of the elements of the normalized transport matrix.
    """
    a = jax.lax.stop_gradient(self.a)
    b = jax.lax.stop_gradient(self.b)
    return a.sum() * b.sum()

  def update_lr_geom(
      self,
      lr_sink: "sinkhorn_lr.LRSinkhornOutput",
      relative_epsilon: Optional[bool] = None,
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
    geom_xx, geom_yy, geom_xy = self.geom_xx, self.geom_yy, self.geom_xy
    tmp1 = apply_cost(geom_xx, q, axis=1, fn=h1)
    tmp2 = apply_cost(geom_yy, r, axis=1, fn=h2)
    if self.is_low_rank:
      geom = low_rank.LRCGeometry(
          cost_1=tmp1, cost_2=-tmp2, relative_epsilon=relative_epsilon
      ) + marginal_cost
      if self.is_fused:
        geom = geom + geom_xy
    else:
      cost_matrix = marginal_cost.cost_matrix - jnp.dot(tmp1, tmp2.T)
      cost_matrix += self.fused_penalty * self._fused_cost_matrix
      geom = geometry.Geometry(
          cost_matrix=cost_matrix, relative_epsilon=relative_epsilon
      )
    return geom  # noqa: RET504

  def update_linearization(
      self,
      transport: Transport,
      epsilon: Optional[float] = None,
      old_transport_mass: float = 1.0,
      relative_epsilon: Optional[bool] = None,
  ) -> linear_problem.LinearProblem:
    """Update linearization of GW problem by updating cost matrix.

    If the problem is balanced (``tau_a = 1.0 and tau_b = 1.0``), the equation
    follows eq. 6, p. 1 of :cite:`peyre:16`.

    If the problem is unbalanced (``tau_a < 1.0 or tau_b < 1.0``), two cases are
    possible, as explained in :meth:`init_linearization` above.
    Finally, it is also possible to consider a Fused Gromov-Wasserstein problem.
    Details about the resulting cost matrix are also given in
    :meth:`init_linearization`.

    Args:
      transport: Solution of the linearization of the quadratic problem.
      epsilon: An epsilon scheduler or a float passed on to the linearization.
      old_transport_mass: Sum of the elements of the transport matrix at the
        previous iteration.
      relative_epsilon: Whether to use relative epsilon in the linearized
        geometry.

    Returns:
      Updated linear OT problem, a new local linearization of GW problem.
    """
    rescale_factor = 1.0
    unbalanced_correction = 0.0

    if not self.is_balanced:
      marginal_1 = transport.marginal(axis=1)
      transport_mass = jax.lax.stop_gradient(marginal_1.sum())
      rescale_factor = jnp.sqrt(old_transport_mass / transport_mass)

    marginal_1 = transport.marginal(axis=1) * rescale_factor
    marginal_2 = transport.marginal(axis=0) * rescale_factor
    marginal_cost = self.marginal_dependent_cost(marginal_1, marginal_2)

    transport_matrix = transport.matrix * rescale_factor

    if not self.is_balanced:
      epsilon *= jax.lax.stop_gradient(marginal_1.sum())
      unbalanced_correction = self.cost_unbalanced_correction(
          transport_matrix, marginal_1, marginal_2, epsilon=epsilon
      )

    h1, h2 = self.quad_loss
    geom_xx, geom_yy = self.geom_xx, self.geom_yy

    tmp = apply_cost(geom_xx, transport_matrix, axis=1, fn=h1)
    tmp = apply_cost(geom_yy, tmp.T, axis=1, fn=h2).T

    cost_matrix = marginal_cost.cost_matrix - tmp + unbalanced_correction
    cost_matrix += self.fused_penalty * rescale_factor * self._fused_cost_matrix

    geom = geometry.Geometry(
        cost_matrix=cost_matrix,
        epsilon=epsilon,
        relative_epsilon=relative_epsilon,
    )

    return linear_problem.LinearProblem(
        geom, self.a, self.b, tau_a=self.tau_a, tau_b=self.tau_b
    )

  def update_lr_linearization(
      self,
      lr_sink: "sinkhorn_lr.LRSinkhornOutput",
      *,
      relative_epsilon: Optional[bool] = None,
  ) -> linear_problem.LinearProblem:
    """Update a Quad problem linearization using a LR Sinkhorn."""
    return linear_problem.LinearProblem(
        self.update_lr_geom(lr_sink, relative_epsilon=relative_epsilon),
        self.a,
        self.b,
        tau_a=self.tau_a,
        tau_b=self.tau_b
    )

  @property
  def _fused_cost_matrix(self) -> Union[float, jnp.ndarray]:
    if not self.is_fused:
      return 0.0
    geom_xy = self.geom_xy

    if isinstance(geom_xy, pointcloud.PointCloud) and geom_xy.is_online:
      return geom_xy._compute_cost_matrix() * geom_xy.inv_scale_cost
    return geom_xy.cost_matrix

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
    return self.ranks != -1 or (
        convertible(geom_xx) and convertible(geom_yy) and
        (geom_xy is None or convertible(geom_xy))
    )

  def to_low_rank(
      self,
      rng: Optional[jax.Array] = None,
  ) -> "QuadraticProblem":
    """Convert geometries to low-rank.

    Args:
      rng: Random key for seeding.

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

    rng = utils.default_prng_key(rng)
    rng1, rng2, rng3 = jax.random.split(rng, 3)
    (geom_xx, geom_yy, geom_xy, *children), aux_data = self.tree_flatten()
    (r1, r2, r3), (t1, t2, t3) = convert(self.ranks), convert(self.tolerances)

    geom_xx = geom_xx.to_LRCGeometry(rank=r1, tol=t1, rng=rng1)
    geom_yy = geom_yy.to_LRCGeometry(rank=r2, tol=t2, rng=rng2)
    if self.is_fused:
      if isinstance(
          geom_xy, pointcloud.PointCloud
      ) and geom_xy.is_squared_euclidean:
        geom_xy = geom_xy.to_LRCGeometry(scale=self.fused_penalty)
      else:
        geom_xy = geom_xy.to_LRCGeometry(
            rank=r3, tol=t3, rng=rng3, scale=self.fused_penalty
        )

    return type(self).tree_unflatten(
        aux_data, [geom_xx, geom_yy, geom_xy] + children
    )

  @property
  def geom_xx(self) -> geometry.Geometry:
    """Geometry of the first space."""
    return self._geom_xx

  @property
  def geom_yy(self) -> geometry.Geometry:
    """Geometry of the second space."""
    return self._geom_yy

  @property
  def geom_xy(self) -> Optional[geometry.Geometry]:
    """Geometry of the joint space."""
    return self._geom_xy

  @property
  def a(self) -> jnp.ndarray:
    """First marginal."""
    num_a = self.geom_xx.shape[0]
    return jnp.ones((num_a,)) / num_a if self._a is None else self._a

  @property
  def b(self) -> jnp.ndarray:
    """Second marginal."""
    num_b = self.geom_yy.shape[0]
    return jnp.ones((num_b,)) / num_b if self._b is None else self._b

  @property
  def is_fused(self) -> bool:
    """Whether the problem is fused."""
    return self.geom_xy is not None

  @property
  def is_low_rank(self) -> bool:
    """Whether all geometries are low-rank."""
    return (
        isinstance(self.geom_xx, low_rank.LRCGeometry) and
        isinstance(self.geom_yy, low_rank.LRCGeometry) and
        (not self.is_fused or isinstance(self.geom_xy, low_rank.LRCGeometry))
    )

  @property
  def linear_loss(self) -> Tuple[quadratic_costs.Loss, quadratic_costs.Loss]:
    """Linear part of the Gromov-Wasserstein loss."""
    return self.loss.f1, self.loss.f2

  @property
  def quad_loss(self) -> Tuple[quadratic_costs.Loss, quadratic_costs.Loss]:
    """Quadratic part of the Gromov-Wasserstein loss."""
    return self.loss.h1, self.loss.h2

  @property
  def is_balanced(self) -> bool:
    """Whether the problem is balanced."""
    return ((not self.gw_unbalanced_correction) or
            (self.tau_a == 1.0 and self.tau_b == 1.0))

  def tree_flatten(self):  # noqa: D102
    return ([self.geom_xx, self.geom_yy, self.geom_xy, self._a, self._b], {
        "tau_a": self.tau_a,
        "tau_b": self.tau_b,
        "loss": self._loss_name,
        "fused_penalty": self.fused_penalty,
        "scale_cost": self.scale_cost,
        "gw_unbalanced_correction": self.gw_unbalanced_correction,
        "ranks": self.ranks,
        "tolerances": self.tolerances
    })

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    geoms, (a, b) = children[:3], children[3:]
    return cls(*geoms, a=a, b=b, **aux_data)


def apply_cost(  # noqa: D103
    geom: geometry.Geometry, arr: jnp.ndarray, *, axis: int,
    fn: quadratic_costs.Loss
) -> jnp.ndarray:
  return geom.apply_cost(arr, axis=axis, fn=fn.func, is_linear=fn.is_linear)
