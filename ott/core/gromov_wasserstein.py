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

# Lint as: python3
"""A Jax version of Sinkhorn's algorithm."""
import abc
import collections
import functools
from typing import Optional, Union, Tuple, Any, Dict

import jax
import jax.numpy as jnp
from ott.core import sinkhorn
from ott.geometry import costs
from ott.geometry import epsilon_scheduler
from ott.geometry import geometry
from ott.geometry import pointcloud

GromovWassersteinOutput = collections.namedtuple(
    'GromovWassersteinOutput', ['f', 'g', 'transport', 'cost_matrix', 'gw_cost',
                                'reg_gw_cost', 'reg_gw_cost_arr',
                                'errors_sinkhorn', 'converged_sinkhorn'])


@jax.tree_util.register_pytree_node_class
class GWLoss(abc.ABC):
  """A loss function for Gromov-Wasserstein.

  The loss can be written as:
  L(x, y) = fn_x(x) + fn_y(y) - left_x(x) * right_y(y)
  """

  @abc.abstractmethod
  def fn_x(self, x):
    pass

  @abc.abstractmethod
  def fn_y(self, y):
    pass

  @abc.abstractmethod
  def left_x(self, x):
    pass

  @abc.abstractmethod
  def right_y(self, y):
    pass

  def tree_flatten(self):
    return (), None

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(*children)


@jax.tree_util.register_pytree_node_class
class GWSqEuclLoss(GWLoss):
  """Implements the squared Euclidean distance for the Gromov-Wasserstein loss.
  """

  def fn_x(self, x):
    return x ** 2.0

  def fn_y(self, y):
    return y ** 2.0

  def left_x(self, x):
    return x

  def right_y(self, y):
    return 2.0 * y


@jax.tree_util.register_pytree_node_class
class GWKlLoss(GWLoss):
  """Implements the KL divergence for the Gromov-Wasserstein loss."""

  def __init__(self, clipping_value=1e-8):
    self.clipping_value = clipping_value

  def fn_x(self, x):
    return -jax.scipy.special.entr(x) - x

  def fn_y(self, y):
    return y

  def left_x(self, x):
    return x

  def right_y(self, y):
    return jnp.log(jnp.clip(y, self.clipping_value))


_GW_LOSSES = [GWSqEuclLoss, GWKlLoss]
GW_LOSSES = {cls.__name__.strip('GW').strip('Loss').lower(): cls()
             for cls in _GW_LOSSES}


def gromov_wasserstein(
    geom_x: geometry.Geometry,
    geom_y: geometry.Geometry,
    a: Optional[jnp.ndarray] = None,
    b: Optional[jnp.ndarray] = None,
    epsilon: Union[epsilon_scheduler.Epsilon, float] = 1.,
    loss: Union[str, GWLoss] = 'sqeucl',
    max_iterations: int = 20,
    jit: bool = False,
    warm_start: bool = True,
    sinkhorn_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs) -> GromovWassersteinOutput:
  """Fits Gromov Wasserstein.

  Args:
    geom_x: a Geometry object for the first view.
    geom_y: a second Geometry object for the second view.
    a: jnp.ndarray<float>[num_a,] or jnp.ndarray<float>[batch,num_a] weights.
    b: jnp.ndarray<float>[num_b,] or jnp.ndarray<float>[batch,num_b] weights.
    epsilon: a regularization parameter or a epsilon_scheduler.Epsilon object.
    loss: str or GWLoss object, whether to use the square euclidean loss
     ('sqeucl' or GWSqEuclLoss), the KL loss ('kl' or GWKlLoss) or a
     user-defined instance of GWLoss.
    max_iterations: int32, the maximum number of outer iterations for
     Gromov Wasserstein.
    jit: bool, if True, jits the function.
    warm_start: bool, optional initialisation of the potentials/scalings w.r.t.
     first and second marginals between each call to sinkhorn
     (default is True).
    sinkhorn_kwargs: Optionally a dictionary containing the keywords arguments
     for calls to the sinkhorn function.
    **kwargs: additional kwargs for epsilon.

  Returns:
    a GromovWassersteinOutput named tuple.
  """
  loss_fn = GW_LOSSES.get(loss, None) if isinstance(loss, str) else loss
  if loss_fn is None:
    raise ValueError('Unknown loss. Either pass an instance of GWLoss or '
                     f'a string among: [{",".join(GW_LOSSES.keys())}]')
  tau_a = sinkhorn_kwargs.get('tau_a', 1.0)
  tau_b = sinkhorn_kwargs.get('tau_b', 1.0)
  if tau_a != 1.0 or tau_b != 1.0:
    raise ValueError('Unbalanced Gromov-Wasserstein is not supported yet.')

  num_a = geom_x.shape[0]
  num_b = geom_y.shape[0]
  a = jnp.ones((num_a,)) / num_a if a is None else a
  b = jnp.ones((num_b,)) / num_b if b is None else b

  gromov_partial = functools.partial(
      _gw_iterations, epsilon=epsilon, loss=loss_fn,
      max_iterations=max_iterations, warm_start=warm_start,
      sinkhorn_kwargs=sinkhorn_kwargs, **kwargs)
  gromov_fn = jax.jit(gromov_partial) if jit else gromov_partial
  (f, g, geom_gw, reg_gw_cost_arr, errors_sinkhorn,
      converged_sinkhorn) = gromov_fn(geom_x, geom_y, a, b)

  # TODO(lpapaxanthos): remove stop_gradient when using backprop
  geom_gw = _update_geometry_gw(
      jax.lax.stop_gradient(geom_gw),
      geom_x, geom_y,
      jax.lax.stop_gradient(f),
      jax.lax.stop_gradient(g),
      loss_fn, **kwargs)
  transport = geom_gw.transport_from_potentials(f, g)
  cost_matrix = 0.5 * geom_gw.cost_matrix
  gw_cost = jnp.sum(transport * cost_matrix)
  reg_gw_cost = sinkhorn.ent_reg_cost(
      geom_gw, a, b, tau_a, tau_b,
      jax.lax.stop_gradient(f),
      jax.lax.stop_gradient(g),
      lse_mode=True)
  return GromovWassersteinOutput(f, g, transport, cost_matrix, gw_cost,
                                 reg_gw_cost, reg_gw_cost_arr, errors_sinkhorn,
                                 converged_sinkhorn)


def _gw_iterations(
    geom_x: geometry.Geometry,
    geom_y: geometry.Geometry,
    a: jnp.ndarray,
    b: jnp.ndarray,
    epsilon: Union[epsilon_scheduler.Epsilon, float],
    loss: GWLoss,
    max_iterations: int,
    warm_start: bool,
    sinkhorn_kwargs: Optional[Dict[str, Any]],
    **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray, geometry.Geometry, jnp.ndarray,
                       jnp.ndarray, jnp.ndarray]:
  """Fits Gromov Wasserstein.

  Args:
    geom_x: a Geometry object for the first view.
    geom_y: a second Geometry object for the second view.
    a: jnp.ndarray<float>[num_a,] or jnp.ndarray<float>[batch,num_a] weights.
    b: jnp.ndarray<float>[num_b,] or jnp.ndarray<float>[batch,num_b] weights.
    epsilon: a regularization parameter or a epsilon_scheduler.Epsilon object.
    loss: GWLoss object.
    max_iterations: int, the maximum number of outer iterations for
     Gromov Wasserstein.
    warm_start: bool, optional initialisation of the potentials/scalings w.r.t.
     first and second marginals between each call to sinkhorn.
    sinkhorn_kwargs: Optionally a dictionary containing the keywords arguments
     for calls to the sinkhorn function.
    **kwargs: additional kwargs for epsilon.

  Returns:
    f: potential.
    g: potential.
    geom_gw: a Geometry object for Gromov-Wasserstein (GW).
    reg_gw_cost_arr: ndarray of regularised GW costs.
    errors_sinkhorn: ndarray [max_iterations, p], where p depends on
     sinkhorn_kwargs, of errors for the Sinkhorn algorithm for each gromov
     iteration (axis 0) and regularly spaced sinkhorn iterations (axis 1).
    converged_sinkhorn: ndarray [max_iterations,] of flags indicating
    that the sinkhorn algorithm converged.
  """
  lse_mode = sinkhorn_kwargs.get('lse_mode', True)

  geom_gw = _init_geometry_gw(
      geom_x, geom_y, jax.lax.stop_gradient(a), jax.lax.stop_gradient(b),
      epsilon, loss, **kwargs)
  f, g, reg_gw_cost, errors_sinkhorn, converged_sinkhorn = sinkhorn.sinkhorn(
      geom_gw, a, b, **sinkhorn_kwargs)
  carry = geom_gw, f, g
  update_geom_partial = functools.partial(_update_geometry_gw, geom_x=geom_x,
                                          geom_y=geom_y, loss=loss, **kwargs)
  sinkhorn_partial = functools.partial(sinkhorn.sinkhorn, a=a, b=b,
                                       **sinkhorn_kwargs)
  def body_fn(carry=carry, x=None):
    del x
    geom_gw, f, g = carry
    geom_gw = update_geom_partial(geom=geom_gw, f=f, g=g)
    init_dual_a = ((f if lse_mode else geom_gw.scaling_from_potential(f))
                   if warm_start else None)
    init_dual_b = ((g if lse_mode else geom_gw.scaling_from_potential(g))
                   if warm_start else None)
    f, g, reg_gw_cost, errors_sinkhorn, converged_sinkhorn = sinkhorn_partial(
        geom=geom_gw, init_dual_a=init_dual_a, init_dual_b=init_dual_b)
    return (geom_gw, f, g), (reg_gw_cost, errors_sinkhorn, converged_sinkhorn)

  carry, out = jax.lax.scan(f=body_fn, init=carry, xs=None,
                            length=max_iterations - 1)

  geom_gw, f, g = carry
  reg_gw_cost_arr = jnp.concatenate((jnp.array([reg_gw_cost]), out[0]))
  errors_sinkhorn = jnp.concatenate((jnp.array([errors_sinkhorn]), out[1]))
  converged_sinkhorn = jnp.concatenate((jnp.array([converged_sinkhorn]),
                                        out[2]))
  return (f, g, geom_gw, reg_gw_cost_arr, errors_sinkhorn, converged_sinkhorn)


def _init_geometry_gw(
    geom_x: geometry.Geometry,
    geom_y: geometry.Geometry,
    a: jnp.ndarray,
    b: jnp.ndarray,
    epsilon: Union[epsilon_scheduler.Epsilon, float],
    loss: GWLoss,
    **kwargs) -> geometry.Geometry:
  """Initialises the cost matrix for the geometry object for GW.

  The equation follows Equation 6, Proposition 1 of
  http://proceedings.mlr.press/v48/peyre16.pdf.

  Args:
    geom_x: a Geometry object for the first view.
    geom_y: a second Geometry object for the second view.
    a: jnp.ndarray<float>[num_a,], weights.
    b: jnp.ndarray<float>[num_b,], weights.
    epsilon: a regularization parameter or a epsilon_scheduler.Epsilon object.
    loss: a GWLossFn object.
    **kwargs: additional kwargs to epsilon.

  Returns:
    A Geometry object for Gromov-Wasserstein.
  """
  # Initialization of the transport matrix in the balanced case, following
  # http://proceedings.mlr.press/v48/peyre16.pdf
  ab = a[:, None] * b[None, :]
  marginal_x = ab.sum(1)
  marginal_y = ab.sum(0)
  marginal_dep_term = _marginal_dependent_cost(
      marginal_x, marginal_y, geom_x, geom_y, loss)

  tmp = geom_x.apply_cost(ab, axis=1, fn=loss.left_x)
  cost_matrix = marginal_dep_term - geom_y.apply_cost(
      tmp.T, axis=1, fn=loss.right_y).T
  return geometry.Geometry(cost_matrix=cost_matrix,
                           epsilon=epsilon, **kwargs)


def _update_geometry_gw(
    geom: geometry.Geometry,
    geom_x: geometry.Geometry,
    geom_y: geometry.Geometry,
    f: jnp.ndarray,
    g: jnp.ndarray,
    loss: GWLoss,
    **kwargs) -> geometry.Geometry:
  """Updates the geometry object for GW by updating the cost matrix.

  The cost matrix equation follows Equation 6, Proposition 1 of
  http://proceedings.mlr.press/v48/peyre16.pdf.

  Let :math:`p` [num_a,] be the marginal of the transport matrix for samples
  from geom_x and :math:`q` [num_b,] be the marginal of the transport matrix for
  samples from geom_y. Let :math:`T` [num_a, num_b] be the transport matrix.
  The cost matrix equation can be written as:

  cost_matrix = marginal_dep_term
              + left_x(cost_x) :math:`T` right_y(cost_y):math:`^T`

  Args:
    geom: a Geometry object carrying the cost matrix of Gromov Wasserstein.
    geom_x: a Geometry object for the first view.
    geom_y: a second Geometry object for the second view.
    f: jnp.ndarray<float>[num_a,], potentials.
    g: jnp.ndarray<float>[num_b,], potentials.
    loss: a GWLossFn object.
    **kwargs: additional kwargs for epsilon.

  Returns:
    A Geometry object for Gromov-Wasserstein.
  """
  def apply_cost_fn(geom):
    condition = is_sqeuclidean(geom) and isinstance(loss, GWSqEuclLoss)
    return geom.vec_apply_cost if condition else geom.apply_cost

  def is_sqeuclidean(geom):
    return (isinstance(geom, pointcloud.PointCloud)
            and geom.power == 2.0
            and isinstance(geom._cost_fn, costs.Euclidean))

  def is_online(geom):
    return isinstance(geom, pointcloud.PointCloud) and geom._online

  # Computes tmp = cost_matrix_x * transport
  if is_online(geom_x) or is_sqeuclidean(geom_x):
    transport = geom.transport_from_potentials(f, g)
    tmp = apply_cost_fn(geom_x)(transport, axis=1,
                                fn=loss.left_x)
  else:
    tmp = geom.apply_transport_from_potentials(
        f, g, loss.left_x(geom_x.cost_matrix), axis=0)

  # Computes cost_matrix
  marginal_x = geom.marginal_from_potentials(f, g, axis=1)
  marginal_y = geom.marginal_from_potentials(f, g, axis=0)
  marginal_dep_term = _marginal_dependent_cost(
      marginal_x, marginal_y, geom_x, geom_y, loss)
  cost_matrix = marginal_dep_term - apply_cost_fn(geom_y)(
      tmp.T, axis=1, fn=loss.right_y).T
  return geometry.Geometry(
      cost_matrix=cost_matrix, epsilon=geom._epsilon_init, **kwargs)


def _marginal_dependent_cost(marginal_x, marginal_y, geom_x, geom_y, loss):
  r"""Calculates part of cost that depends on marginals of transport matrix.

  Uses the first term in Equation 6, Proposition 1 of
  http://proceedings.mlr.press/v48/peyre16.pdf.

  Let :math:`p` [num_a,] be the marginal of the transport matrix for samples
  from geom_x and :math:`q` [num_b,] be the marginal of the transport matrix for
  samples from geom_y. The term in the cost that depends on these marginals can
  be written as:
  marginal_dep_term = fn_x(cost_x) :math:`p \mathbb{1}_{num_b}^T`
                    + (fn_y(cost_y) :math:`q \mathbb{1}_{num_a}^T)^T`

  Args:
    marginal_x: jnp.ndarray<float>[num_a,], marginal of the transport matrix for
     samples from geom_x
    marginal_y: jnp.ndarray<float>[num_b,], marginal of the transport matrix for
     samples from geom_y
    geom_x: a Geometry object for the first view.
    geom_y: a second Geometry object for the second view.
    loss: a GWLossFn object.

  Returns:
    jnp.ndarray, [num_a, num_b]
  """
  x_term = jnp.dot(
      geom_x.apply_cost(marginal_x, 1, loss.fn_x).reshape(-1, 1),
      jnp.ones((1, marginal_y.size)))
  y_term = jnp.dot(
      geom_y.apply_cost(marginal_y, 1, loss.fn_y).reshape(-1, 1),
      jnp.ones((1, marginal_x.size))).T
  return x_term + y_term
