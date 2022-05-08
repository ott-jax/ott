# coding=utf-8
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

from typing import Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from ott.core import problems
from ott.core import sinkhorn_lr
from ott.geometry import epsilon_scheduler
from ott.geometry import geometry
from ott.geometry import low_rank
from ott.geometry import pointcloud
# Because Protocol is not available in Python < 3.8
from typing_extensions import Protocol


class Transport(Protocol):
  """Defines the interface for the solution of a transport problem.

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


LossTerm = Callable[[jnp.ndarray], jnp.ndarray]
Loss = Tuple[Tuple[LossTerm, LossTerm], Tuple[LossTerm, LossTerm]]


def make_square_loss():
  return ((lambda x: x**2, lambda y: y**2), (lambda x: x,
                                             lambda y: 2.0 * y))


def make_kl_loss(clipping_value: float = 1e-8):

  return (
      (lambda x: -jax.scipy.special.entr(x) - x, lambda y: y),
      (lambda x: x, lambda y: jnp.log(jnp.clip(y, clipping_value)))
  )


@jax.tree_util.register_pytree_node_class
class QuadraticProblem:
  """Holds the definition of the quadratic regularized OT problem.

  The quadratic loss of a single OT matrix is assumed to
  have the form given in Eq. 4 from

  http://proceedings.mlr.press/v48/peyre16.pdf

  The two geometries below parameterize matrices C and bar{C} in that equation.
  The function L (of two real values) in that equation is assumed
  to match the form given in Eq. 5. , with our notations:

  L(x, y) = lin1(x) + lin2(y) - quad1(x) * quad2(y)
  """

  def __init__(self,
               geom_xx: geometry.Geometry,
               geom_yy: geometry.Geometry,
               geom_xy: Optional[geometry.Geometry] = None,
               fused_penalty: Optional[float] = None,
               scale_cost: Optional[Union[bool, float, str]] = False,
               a: Optional[jnp.ndarray] = None,
               b: Optional[jnp.ndarray] = None,
               loss: Optional[Loss] = None,
               tau_a: Optional[float] = 1.0,
               tau_b: Optional[float] = 1.0,
               gw_unbalanced_correction: Optional[bool] = True):
    """Initializes the QuadraticProblem.

    Args:
      geom_xx: the geometry.Geometry object defining the ground geometry / cost
        of the first space.
      geom_yy: the geometry.Geometry object defining the ground geometry / cost
        of the second space.
      geom_xy: the geometry.Geometry object defining the linear penalty term
        for Fused Gromov Wasserstein. If None, the problem reduces to a plain
        Gromov Wasserstein problem.
      fused_penalty: multiplier of the linear term in Fused Gromov Wasserstein,
        i.e. problem = purely quadratic + fused_penalty * linear problem. If
        fused_penalty is None but geom_xy is passed, fused_penalty is set by
        default to 1.0, equal to 0.0 otherwise.
      scale_cost: option to rescale the cost matrices:

        - if `True`, use the default for each geometry.
        - if `False`, keep the original scaling in geometries.
        - if :class:`str`, use a specific method available in
          :meth:`ott.geometry.geometry.Geometry.__init__` or
          :meth:`ott.geometry.pointcloud.PointCloud.__init__`.
        - if `None`, do not scale the cost matrices.

      a: jnp.ndarray[n] representing the probability weights of the samples
        from geom_xx. If None, it will be uniform.
      b: jnp.ndarray[n] representing the probability weights of the samples
        from geom_yy. If None, it will be uniform.
      loss: a 2-tuple of 2-tuples of Callable. The first tuple is the linear
        part of the loss (see in the pydoc of the class lin1, lin2). The second
        one is the quadratic part (quad1, quad2). If None is passed, the loss
        is set as the 4 functions representing the squared euclidean loss, and
        this property is taken advantage of in subsequent computations. See
        make_kl_loss for an alternative, no less optimized way of setting the
        loss.
      tau_a: if lower that 1.0, defines how much unbalanced the problem is on
        the first marginal.
      tau_b: if lower that 1.0, defines how much unbalanced the problem is on
        the second marginal.
      gw_unbalanced_correction: True (default) if the unbalanced version of
        Sejourne et al. (Neurips 2021) is used, False if tau_a and tau_b
        only affect the inner Sinhkorn loop.
    """
    self.geom_xx = geom_xx._set_scale_cost(scale_cost)
    self.geom_yy = geom_yy._set_scale_cost(scale_cost)
    self.geom_xy = (None if geom_xy is None else
                    geom_xy._set_scale_cost(scale_cost))
    if fused_penalty is None:
      fused_penalty = jnp.where(self.geom_xy is None, 0.0, 1.0)
    self.fused_penalty = fused_penalty
    self.scale_cost = scale_cost
    self._a = a
    self._b = b
    self.tau_a = tau_a
    self.tau_b = tau_b
    self.gw_unbalanced_correction = gw_unbalanced_correction
    if loss is None:
      self._sq_euc = True
      self.loss = make_square_loss()
    else:
      self._sq_euc = False
      self.loss = loss

  @property
  def is_fused(self):
    return self.geom_xy is not None and self.fused_penalty > 0.0

  @property
  def is_all_geoms_lr(self):
    return (isinstance(self.geom_xx, low_rank.LRCGeometry) and
            isinstance(self.geom_yy, low_rank.LRCGeometry) and
            (not self.is_fused or isinstance(self.geom_xy, low_rank.LRCGeometry)))

  @property
  def linear_loss(self):
    return self.loss[0]

  @property
  def quad_loss(self):
    return self.loss[1]

  @property
  def is_balanced(self):
    return ((not self.gw_unbalanced_correction)
            or (self.tau_a == 1.0 and self.tau_b == 1.0))

  def tree_flatten(self):
    return ([self.geom_xx, self.geom_yy, self.geom_xy, self._a, self._b],
            {'tau_a': self.tau_a, 'tau_b': self.tau_b, 'loss': self.loss,
             'fused_penalty': self.fused_penalty, 'scale_cost': self.scale_cost,
             'gw_unbalanced_correction': self.gw_unbalanced_correction}
            )

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    geoms, (a, b) = children[:3], children[3:]
    return cls(*geoms, a=a, b=b, **aux_data)

  @property
  def a(self):
    num_a = self.geom_xx.shape[0]
    return jnp.ones((num_a,)) / num_a if self._a is None else self._a

  @property
  def b(self):
    num_b = self.geom_yy.shape[0]
    return jnp.ones((num_b,)) / num_b if self._b is None else self._b

  def marginal_dependent_cost(self, marginal_1, marginal_2):
    r"""Initialises cost term that depends on the marginals of the transport.

    Uses the first term in Equation 6, Proposition 1 of
    http://proceedings.mlr.press/v48/peyre16.pdf.

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
      a LRCGeometry.
    """
    if self._sq_euc:  # quadratic apply
      tmp1 = self.geom_xx.apply_square_cost(marginal_1, axis=1)
      tmp2 = self.geom_yy.apply_square_cost(marginal_2, axis=1)
    else:
      tmp1 = self.geom_xx.apply_cost(marginal_1, axis=1,
                                     fn=self.linear_loss[0])
      tmp2 = self.geom_yy.apply_cost(marginal_2, axis=1,
                                     fn=self.linear_loss[1])
    x_term = jnp.concatenate((tmp1, jnp.ones_like(tmp1)), axis=1)
    y_term = jnp.concatenate((jnp.ones_like(tmp2), tmp2), axis=1)
    return low_rank.LRCGeometry(cost_1=x_term, cost_2=y_term)

  def cost_unbalanced_correction(self, transport_matrix, marginal_1, marginal_2,
                                 epsilon, rescale_factor, delta=1e-9) -> float:
    r"""Calculates cost term from the quadratic divergence when unbalanced.

    In the unbalanced setting (i.e. tau_a<1.0 or tau_b<1.0), the
    introduction of a quadratic divergence (see Sejourne et al. Neurips 2021)
    adds a term to the GW local cost.

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

    Arguments:
      transport_matrix: jnp.ndarray<float>[num_a, num_b], transport matrix.
      marginal_1: jnp.ndarray<float>[num_a,], marginal of the transport matrix
       for samples from geom_xx
      marginal_2: jnp.ndarray<float>[num_b,], marginal of the transport matrix
       for samples from geom_yy
      epsilon: regulariser.
      rescale_factor: float, scaling factor for the transport matrix.
      delta: float, small quantity to avoid diverging KLs.

    Returns:
      float, cost term
    """
    def regulariser(tau):
      return tau / (1.0 - tau) if tau != 1.0 else 0

    cost = regulariser(self.tau_a) * jax.scipy.special.xlogy(
        marginal_1,
        rescale_factor * marginal_1 / jnp.clip(self.a, a_min=delta)).sum()
    cost += regulariser(self.tau_b) * jax.scipy.special.xlogy(
        marginal_2,
        rescale_factor * marginal_2 / jnp.clip(self.b, a_min=delta)).sum()
    cost += epsilon * jax.scipy.special.xlogy(
        transport_matrix,
        rescale_factor * transport_matrix
        / jnp.clip(self.a[:, None] * self.b[None, :], a_min=delta)).sum()
    return cost

  def init_transport(self):
    # TODO(oliviert, cuturi): consider passing a custom initialization.
    a = jax.lax.stop_gradient(self.a)
    b = jax.lax.stop_gradient(self.b)
    return (a[:, None] * b[None, :] if self.is_balanced else a[:, None] *
            b[None, :] / jnp.sqrt(a.sum() * b.sum()))

  def init_transport_mass(self) -> float:
    """Initialises the transport mass.

    Returns:
      A float, sum of the elements of the normalised transport matrix.
    """
    a = jax.lax.stop_gradient(self.a)
    b = jax.lax.stop_gradient(self.b)
    transport_mass = a.sum() * b.sum()
    return (transport_mass if self.is_balanced
            else transport_mass / jnp.sqrt(transport_mass))

  def init_linearization(
      self,
      epsilon: Optional[Union[epsilon_scheduler.Epsilon, float]] = None
  ) -> problems.LinearProblem:
    """Initialises a linear problem locally around a naive initializer ab'.

    If the problem is balanced (`tau_a=1.0 and tau_b=1.0'), the equation of the
    cost follows Equation 6, Proposition 1 of
    http://proceedings.mlr.press/v48/peyre16.pdf.
    If the problem is unbalanced (`tau_a<1.0 or tau_b<1.0`), there are two
    possible cases. A first possibility is to introduce a
    quadratic KL divergence on the marginals in the objective as done
    in Sejourne et al. 2021 (https://arxiv.org/abs/2009.04266)
    (`gw_unbalanced_correction=True`), which in turns modifies the local cost
    matrix.
    Alternatively, it could be possible to leave
    the formulation of the local cost unchanged, i.e. follow Equation 6 of
    Proposition 1 (`gw_unbalanced_correction=False`) and include the
    unbalanced terms at the level of the linear problem only.

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
      A problems.LinearProblem, representing local linearization of GW problem.
    """
    unbalanced_correction = 0.0
    tmp = self.init_transport()
    marginal_1 = tmp.sum(1)
    marginal_2 = tmp.sum(0)

    # Initialises cost.
    marginal_cost = self.marginal_dependent_cost(marginal_1, marginal_2)

    if not self.is_balanced:
      unbalanced_correction = self.cost_unbalanced_correction(
          tmp, marginal_1, marginal_2, epsilon, 1.0)

    tmp = self.geom_xx.apply_cost(tmp, axis=1, fn=self.quad_loss[0])
    tmp = self.geom_yy.apply_cost(tmp.T, axis=1, fn=self.quad_loss[1]).T
    cost_matrix = (marginal_cost.cost_matrix - tmp + unbalanced_correction)

    # Initialises epsilon for Unbalanced GW according to Sejourne et al (2021).
    if not self.is_balanced:
      transport_mass = marginal_1.sum()
      epsilon = update_epsilon_unbalanced(epsilon, transport_mass)

    cost_matrix += self.fused_penalty * jnp.where(
        self.is_fused,
        0.0 if self.geom_xy is None else self.geom_xy.cost_matrix,
        0.0)

    geom = geometry.Geometry(cost_matrix=cost_matrix, epsilon=epsilon)
    return problems.LinearProblem(
        geom, self.a, self.b, tau_a=self.tau_a, tau_b=self.tau_b)

  def init_lr_linearization(
      self,
      rank: int = 10,
      **kwargs
  ) -> problems.LinearProblem:
    """Linearizes a Quad problem with a predefined initializer."""
    x_ = self.geom_xx.apply_square_cost(self.a)
    y_ = self.geom_yy.apply_square_cost(self.b)
    geom_ = pointcloud.PointCloud(x_, y_).to_LRCGeometry()
    out = sinkhorn_lr.LRSinkhorn(rank=rank, **kwargs)(
        problems.LinearProblem(geom_, self.a, self.b))
    return problems.LinearProblem(
        self.update_lr_geom(out),
        self.a, self.b, tau_a=self.tau_a, tau_b=self.tau_b)

  def update_lr_geom(self, lr_sink):
    """Using LR Sinkhorn output, recompute (possibly LRC) linearization."""
    marginal_1 = lr_sink.marginal(1)
    marginal_2 = lr_sink.marginal(0)
    marginal_cost = self.marginal_dependent_cost(marginal_1, marginal_2)

    # Extract factors from LR Sinkhorn output
    q, r, inv_sqg = lr_sink.q, lr_sink.r, 1.0 / jnp.sqrt(lr_sink.g)
    # Distribute middle marginal evenly across both factors.
    q, r = q * inv_sqg[None, :], r * inv_sqg[None, :]

    # Handle LRC Geometry case.
    tmp1 = self.geom_xx.apply_cost(q, axis=1, fn=self.quad_loss[0])
    tmp2 = self.geom_yy.apply_cost(r, axis=1, fn=self.quad_loss[1])
    if self.is_all_geoms_lr:
      geom = low_rank.LRCGeometry(cost_1=tmp1, cost_2=-tmp2)
      geom = low_rank.add_lrc_geom(geom, marginal_cost)
      if self.is_fused:
        geom = low_rank.add_lrc_geom(geom, self.geom_xy)
    else:
      cost_matrix = marginal_cost.cost_matrix - jnp.dot(tmp1, tmp2.T)
      cost_matrix += self.fused_penalty * jnp.where(
          self.is_fused,
          0.0 if self.geom_xy is None else self.geom_xy.cost_matrix, 0.0)
      geom = geometry.Geometry(cost_matrix=cost_matrix)
    return geom

  def update_linearization(
      self,
      transport: Transport,
      epsilon: Optional[Union[epsilon_scheduler.Epsilon, float]] = None,
      old_transport_mass: float = 1.0) -> problems.LinearProblem:
    """Updates linearization of GW problem by updating cost matrix.

    If the problem is balanced (`tau_a=1.0 and tau_b=1.0`), the equation
    follows Equation 6, Proposition 1 of
    http://proceedings.mlr.press/v48/peyre16.pdf. If the problem is unbalanced
    (`tau_a<1.0 or tau_b<1.0`), two cases are possible, as explained in the
    pydoc of `init_linearization` above. Finally, it is also possible to
    consider a Fused Gromov Wasserstein problem. Details about the resulting
    cost matrix are given in the pydoc of `init_linearization`.

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
          transport.matrix, marginal_1, marginal_2, epsilon, rescale_factor)
      # Updates epsilon for Unbalanced GW.
      epsilon = update_epsilon_unbalanced(epsilon, transport_mass)

    tmp = self.geom_xx.apply_cost(
        transport.matrix, axis=1, fn=self.quad_loss[0])
    tmp = self.geom_yy.apply_cost(tmp.T, axis=1, fn=self.quad_loss[1]).T

    cost_matrix = marginal_cost.cost_matrix - tmp + unbalanced_correction

    cost_matrix += self.fused_penalty * jnp.where(
        self.is_fused,
        0.0 if self.geom_xy is None else self.geom_xy.cost_matrix,
        0.0)

    cost_matrix *= rescale_factor

    geom = geometry.Geometry(cost_matrix=cost_matrix, epsilon=epsilon)
    return problems.LinearProblem(
        geom, self.a, self.b, tau_a=self.tau_a, tau_b=self.tau_b)

  def update_lr_linearization(
      self,
      lr_sink: sinkhorn_lr.LRSinkhornOutput) -> problems.LinearProblem:
    """Updates a Quad problem linearization using a LR Sinkhorn."""
    return problems.LinearProblem(
        self.update_lr_geom(lr_sink),
        self.a,
        self.b,
        tau_a=self.tau_a,
        tau_b=self.tau_b)


def update_epsilon_unbalanced(epsilon, transport_mass):
  updated_epsilon = epsilon_scheduler.Epsilon.make(epsilon)
  updated_epsilon._scale_epsilon = (
      updated_epsilon._scale_epsilon * transport_mass)
  return updated_epsilon


def make(*args,
         a: Optional[jnp.ndarray] = None,
         b: Optional[jnp.ndarray] = None,
         tau_a: float = 1.0,
         tau_b: float = 1.0,
         objective: Optional[str] = None,
         gw_unbalanced_correction: Optional[bool] = True,
         fused_penalty: Optional[float] = None,
         scale_cost: Optional[Union[bool, float, str]] = False,
         **kwargs):
  """Makes a problem from arrays, assuming PointCloud geometries."""
  if isinstance(args[0], (jnp.ndarray, np.ndarray)):
    x = args[0]
    y = args[1] if len(args) > 1 else args[0]
    if ((objective == 'linear') or
        (objective is None and x.shape[1] == y.shape[1])):
      geom_xy = pointcloud.PointCloud(x, y, **kwargs)
      return problems.LinearProblem(geom_xy, a=a, b=b, tau_a=tau_a, tau_b=tau_b)
    elif ((objective == 'quadratic') or
          (objective is None and x.shape[1] != y.shape[1])):
      geom_xx = pointcloud.PointCloud(x, x, **kwargs)
      geom_yy = pointcloud.PointCloud(y, y, **kwargs)
      return QuadraticProblem(geom_xx=geom_xx, geom_yy=geom_yy,
                              geom_xy=None,
                              scale_cost=scale_cost,
                              a=a, b=b, tau_a=tau_a, tau_b=tau_b,
                              gw_unbalanced_correction=gw_unbalanced_correction)
    elif objective == 'fused':
      geom_xx = pointcloud.PointCloud(x, x, **kwargs)
      geom_yy = pointcloud.PointCloud(y, y, **kwargs)
      geom_xy = pointcloud.PointCloud(x, y, **kwargs)
      return QuadraticProblem(geom_xx=geom_xx, geom_yy=geom_yy, geom_xy=geom_xy,
                              fused_penalty=fused_penalty,
                              scale_cost=scale_cost,
                              a=a, b=b, tau_a=tau_a, tau_b=tau_b,
                              gw_unbalanced_correction=gw_unbalanced_correction)
    else:
      raise ValueError(f'Unknown transport problem `{objective}`')
  elif isinstance(args[0], geometry.Geometry):
    if len(args) == 1:
      return problems.LinearProblem(*args, a=a, b=b, tau_a=tau_a, tau_b=tau_b)
    return QuadraticProblem(*args, a=a, b=b, tau_a=tau_a, tau_b=tau_b,
                            scale_cost=scale_cost)
  elif isinstance(args[0], (problems.LinearProblem, QuadraticProblem)):
    return args[0]
  else:
    raise ValueError('Cannot instantiate a transport problem.')
