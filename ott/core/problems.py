# coding=utf-8
# Copyright 2021 Google LLC.
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
from ott.geometry import epsilon_scheduler
from ott.geometry import geometry
from ott.geometry import pointcloud
# Because Protocol is not available in Python < 3.8
from typing_extensions import Protocol


LossTerm = Callable[[jnp.ndarray], jnp.ndarray]
Loss = Tuple[Tuple[LossTerm, LossTerm], Tuple[LossTerm, LossTerm]]


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


@jax.tree_util.register_pytree_node_class
class LinearProblem:
  """Holds the definition of a linear regularized OT problem and some tools."""

  def __init__(self,
               geom: geometry.Geometry,
               a: Optional[jnp.ndarray] = None,
               b: Optional[jnp.ndarray] = None,
               tau_a: float = 1.0,
               tau_b: float = 1.0):
    """Initializes the LinearProblem.

    min_P<C, P> - eps H(P), s.t P.1 = a, Pt.1 = b.

    Args:
      geom: the geometry.Geometry object defining the ground geometry / cost of
        the linear problem.
      a: jnp.ndarray[n] representing the first marginal. If None, it will be
        uniform.
      b: jnp.ndarray[n] representing the first marginal. If None, it will be
        uniform.
      tau_a: if lower that 1.0, defines how much unbalanced the problem is on
        the first marginal.
      tau_b: if lower that 1.0, defines how much unbalanced the problem is on
        the second marginal.
    """
    self.geom = geom
    self._a = a
    self._b = b
    self.tau_a = tau_a
    self.tau_b = tau_b

  def tree_flatten(self):
    return ([self.geom, self._a, self._b],
            {'tau_a': self.tau_a, 'tau_b': self.tau_b})

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(*children, **aux_data)

  @property
  def a(self):
    num_a = self.geom.shape[0]
    return jnp.ones((num_a,)) / num_a if self._a is None else self._a

  @property
  def b(self):
    num_b = self.geom.shape[1]
    return jnp.ones((num_b,)) / num_b if self._b is None else self._b

  @property
  def is_balanced(self):
    return self.tau_a == 1.0 and self.tau_b == 1.0

  @property
  def epsilon(self):
    return self.geom.epsilon

  def get_transport_functions(self, lse_mode: bool):
    """Instantiates useful functions for Sinkhorn depending on lse_mode."""
    geom = self.geom
    if lse_mode:
      marginal_a = lambda f, g: geom.marginal_from_potentials(f, g, 1)
      marginal_b = lambda f, g: geom.marginal_from_potentials(f, g, 0)
      app_transport = geom.apply_transport_from_potentials
    else:
      marginal_a = lambda f, g: geom.marginal_from_scalings(
          geom.scaling_from_potential(f), geom.scaling_from_potential(g), 1)
      marginal_b = lambda f, g: geom.marginal_from_scalings(
          geom.scaling_from_potential(f), geom.scaling_from_potential(g), 0)
      app_transport = lambda f, g, z, axis: geom.apply_transport_from_scalings(
          geom.scaling_from_potential(f),
          geom.scaling_from_potential(g), z, axis)
    return marginal_a, marginal_b, app_transport


def make_square_loss():
  return (lambda x: x ** 2, lambda y: y ** 2), (lambda x: x, lambda y: 2.0 * y)


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
               geom_xy: Optional[geometry.Geometry],
               fused_penalty: Optional[float] = 0.0,
               a: Optional[jnp.ndarray] = None,
               b: Optional[jnp.ndarray] = None,
               is_fused: bool = False,
               loss: Optional[Loss] = None,
               tau_a: float = 1.0,
               tau_b: float = 1.0):
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
        i.e. loss = quadratic_loss + fused_penalty * linear_loss. If geom_xy is
        None fused_penalty will be ignored, i.e. fused_penalty = 0
      a: jnp.ndarray[n] representing the first marginal. If None, it will be
        uniform.
      b: jnp.ndarray[n] representing the first marginal. If None, it will be
        uniform.
      is_fused: indicates whether we have a pure Gromov-Wasserstein or a FGW
        problem
      loss: a 2-tuple of 2-tuples of Callable. The first tuple is the linear
        part of the loss (see in the pydoc of the class lin1, lin2). The second
        one is the quadratic part (quad1, quad2). If None is passed, the loss
        is set as the 4 functions representing the squared euclidean loss. See
        make_square_loss and and make_kl_loss for convenient way of setting the
        loss.
      tau_a: if lower that 1.0, defines how much unbalanced the problem is on
        the first marginal.
      tau_b: if lower that 1.0, defines how much unbalanced the problem is on
        the second marginal.
    """
    self.geom_xx = geom_xx
    self.geom_yy = geom_yy
    self.geom_xy = geom_xy
    self.fused_penalty = fused_penalty
    self._a = a
    self._b = b
    self.is_fused = is_fused
    self.tau_a = tau_a
    self.tau_b = tau_b
    self.loss = make_square_loss() if loss is None else loss

  @property
  def linear_loss(self):
    return self.loss[0]

  @property
  def quad_loss(self):
    return self.loss[1]

  @property
  def is_balanced(self):
    return self.tau_a == 1.0 and self.tau_b == 1.0

  def tree_flatten(self):
    return ([
        self.geom_xx, self.geom_yy, self.geom_xy, self.fused_penalty, self._a,
        self._b
    ], {
        'tau_a': self.tau_a,
        'tau_b': self.tau_b,
        'loss': self.loss,
        'is_fused': self.is_fused
    })

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(*children, **aux_data)

  @property
  def a(self):
    num_a = self.geom_xx.shape[0]
    return jnp.ones((num_a,)) / num_a if self._a is None else self._a

  @property
  def b(self):
    num_b = self.geom_yy.shape[0]
    return jnp.ones((num_b,)) / num_b if self._b is None else self._b

  def marginal_dependent_cost(self, marginal_1, marginal_2):
    r"""Calculates part of cost that depends on marginals of transport matrix.

    Uses the first term in Equation 6, Proposition 1 of
    http://proceedings.mlr.press/v48/peyre16.pdf.

    Let :math:`p` [num_a,] be the marginal of the transport matrix for samples
    from geom_x and :math:`q` [num_b,] be the marginal of the transport matrix
    for samples from geom_y. The term in the cost that depends on these
    marginals can be written as:
    marginal_dep_term = fn_x(cost_x) :math:`p \mathbb{1}_{num_b}^T`
                      + (fn_y(cost_y) :math:`q \mathbb{1}_{num_a}^T)^T`

    Args:
      marginal_1: jnp.ndarray<float>[num_a,], marginal of the transport matrix
       for samples from geom_xx
      marginal_2: jnp.ndarray<float>[num_b,], marginal of the transport matrix
       for samples from geom_yy

    Returns:
      jnp.ndarray, [num_a, num_b]
    """
    x_term = jnp.dot(
        self.geom_xx.apply_cost(
            marginal_1[:, None], 1, self.linear_loss[0]).reshape(-1, 1),
        jnp.ones((1, marginal_2.size)))
    y_term = jnp.dot(
        self.geom_yy.apply_cost(
            marginal_2[:, None], 1, self.linear_loss[1]).reshape(-1, 1),
        jnp.ones((1, marginal_1.size))).T
    return x_term + y_term

  def init_linearization(
      self,
      epsilon: Optional[Union[epsilon_scheduler.Epsilon, float]] = None
  ) -> LinearProblem:
    """Initialises a linear problem locally around a naive initializer ab'.

    The equation follows Equation 6, Proposition 1 of
    http://proceedings.mlr.press/v48/peyre16.pdf.

    Args:
      epsilon: An epsilon scheduler or a float passed on to the linearization.

    Returns:
      A LinearProblem, representing local linearization of GW problem.
    """
    a = jax.lax.stop_gradient(self.a)
    b = jax.lax.stop_gradient(self.b)
    # TODO(oliviert, cuturi): consider passing a custom initialization.
    ab = a[:, None] * b[None, :]
    marginal_1 = ab.sum(1)
    marginal_2 = ab.sum(0)
    marginal_term = self.marginal_dependent_cost(marginal_1, marginal_2)
    tmp = self.geom_xx.apply_cost(ab, axis=1, fn=self.quad_loss[0])
    cost_matrix = marginal_term - self.geom_yy.apply_cost(
        tmp.T, axis=1, fn=self.quad_loss[1]).T
    if self.is_fused:
      geom = geometry.Geometry(
          cost_matrix=cost_matrix +
          self.fused_penalty * self.geom_xy.cost_matrix,
          epsilon=epsilon)
    else:
      geom = geometry.Geometry(cost_matrix=cost_matrix, epsilon=epsilon)
    return LinearProblem(geom, self.a, self.b)

  def update_linearization(
      self,
      transport: Transport,
      epsilon: Optional[Union[epsilon_scheduler.Epsilon, float]] = None
  ) -> LinearProblem:
    """Updates linearization of GW problem by updating cost matrix.

    The cost matrix equation follows Equation 6, Proposition 1 of
    http://proceedings.mlr.press/v48/peyre16.pdf.

    Let :math:`p` [num_a,] be the marginal of the transport matrix for samples
    from geom_x and :math:`q` [num_b,] be the marginal of the transport matrix
    for samples from geom_y. Let :math:`T` [num_a, num_b] be the transport
    matrix. The cost matrix equation can be written as:

    cost_matrix = marginal_dep_term
                + left_x(cost_x) :math:`T` right_y(cost_y):math:`^T`

    Args:
      transport: Solution of the linearization of the quadratic problem.
      epsilon: An epsilon scheduler or a float passed on to the linearization.

    Returns:
      Updated linear OT problem, a new local linearization of GW problem.
    """
    # Computes tmp = cost_matrix_x * transport
    # When the transport can be instantiated and a low rank structure
    # of the cost can be taken advantage of, it is preferable to do the product
    # between transport and cost matrix by instantiating first the transport
    # and applying the cost to it on the left.
    # TODO(cuturi,oliviert,lpapaxanthos): handle online & sqEuc geom_xx better
    if not self.geom_xx.is_online or self.geom_xx.is_squared_euclidean:
      tmp = self.geom_xx.apply_cost(
          transport.matrix, axis=1, fn=self.quad_loss[0])
    else:
      # When on the contrary the transport is difficult to instantiate
      # we default back on the application of the transport to the cost matrix.
      tmp = transport.apply(self.quad_loss[0](self.geom_xx.cost_matrix), axis=0)

    marginal_1 = transport.marginal(axis=1)
    marginal_2 = transport.marginal(axis=0)
    marginal_term = self.marginal_dependent_cost(marginal_1, marginal_2)
    # TODO(cuturi,oliviert,lpapaxanthos): handle low rank prod. for geom_yy's.
    cost_matrix = marginal_term - self.geom_yy.apply_cost(
        tmp.T, axis=1, fn=self.quad_loss[1]).T
    if self.is_fused > 0:
      geom = geometry.Geometry(
          cost_matrix=cost_matrix +
          self.fused_penalty * self.geom_xy.cost_matrix,
          epsilon=epsilon)
    else:
      geom = geometry.Geometry(cost_matrix=cost_matrix, epsilon=epsilon)
    return LinearProblem(geom, self.a, self.b)


def make(*args,
         a: Optional[jnp.ndarray] = None,
         b: Optional[jnp.ndarray] = None,
         tau_a: float = 1.0,
         tau_b: float = 1.0,
         objective: Optional[str] = None,
         **kwargs):
  """Makes a problem from arrays, assuming PointCloud geometries."""
  if isinstance(args[0], (jnp.ndarray, np.ndarray)):
    x = args[0]
    y = args[1] if len(args) > 1 else args[0]
    if ((objective == 'linear') or
        (objective is None and x.shape[1] == y.shape[1])):
      geom_xy = pointcloud.PointCloud(x, y, **kwargs)
      return LinearProblem(geom_xy, a=a, b=b, tau_a=tau_a, tau_b=tau_b)
    elif ((objective == 'quadratic') or
          (objective is None and x.shape[1] != y.shape[1])):
      geom_xx = pointcloud.PointCloud(x, x, **kwargs)
      geom_yy = pointcloud.PointCloud(y, y, **kwargs)
      return QuadraticProblem(geom_xx=geom_xx, geom_yy=geom_yy,
                              geom_xy=None,
                              is_fused=False,
                              fused_penalty=0.0,
                              a=a, b=b, tau_a=tau_a, tau_b=tau_b)
    elif objective == 'fused':
      fused_penalty = kwargs.pop('fused_penalty', None)
      is_fused = kwargs.pop('is_fused', True)
      geom_xx = pointcloud.PointCloud(x, x, **kwargs)
      geom_yy = pointcloud.PointCloud(y, y, **kwargs)
      geom_xy = pointcloud.PointCloud(x, y, **kwargs)
      return QuadraticProblem(geom_xx=geom_xx, geom_yy=geom_yy, geom_xy=geom_xy,
                              is_fused=is_fused, fused_penalty=fused_penalty,
                              a=a, b=b, tau_a=tau_a, tau_b=tau_b)
    else:
      raise ValueError(f'Unknown transport problem `{objective}`')
  elif isinstance(args[0], geometry.Geometry):
    cls = LinearProblem if len(args) == 1 else QuadraticProblem
    return cls(*args, a=a, b=b, tau_a=tau_a, tau_b=tau_b)
  elif isinstance(args[0], (LinearProblem, QuadraticProblem)):
    return args[0]
  else:
    raise ValueError('Cannot instantiate a transport problem.')
