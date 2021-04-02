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

EPS = 1e-5

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
  def fn_y(self, x):
    pass

  @abc.abstractmethod
  def left_x(self, x):
    pass

  @abc.abstractmethod
  def right_y(self, x):
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

  # TODO(lpapaxanthos): stable implementation for KL
  def fn_x(self, x):
    x = jnp.clip(x, EPS)
    return jax.scipy.special.xlogy(x, x) - x

  def fn_y(self, x):
    return x

  def left_x(self, x):
    return x

  def right_y(self, x):
    return jnp.log(jnp.clip(x, EPS))


_GW_LOSSES = [GWSqEuclLoss, GWKlLoss]
GW_LOSSES = {cls.__name__.strip('GW').strip('Loss').lower(): cls()
             for cls in _GW_LOSSES}


def gromov_wasserstein(
    geom_x: geometry.Geometry,
    geom_y: geometry.Geometry,
    a: Optional[jnp.ndarray] = None,
    b: Optional[jnp.ndarray] = None,
    epsilon: Union[epsilon_scheduler.Epsilon, float] = 1e-2,
    loss: Union[str, GWLoss] = 'sqeucl',
    tau_a: float = 1.0,
    tau_b: float = 1.0,
    max_iterations_gromov: int = 20,
    jit_gromov: bool = False,
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
    tau_a: float, ratio lam/(lam+eps) between KL divergence regularizer to first
     marginal and itself + epsilon regularizer used in the unbalanced
     formulation.
    tau_b: float, ratio lam/(lam+eps) between KL divergence regularizer to
     second marginal and itself + epsilon regularizer used in the unbalanced
     formulation.
    max_iterations_gromov: int32, the maximum number of outer iterations for
     Gromov Wasserstein.
    jit_gromov: bool, if True, jits the function.
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

  num_a = geom_x.shape[0]
  num_b = geom_y.shape[0]
  a = jnp.ones((num_a,)) / num_a if a is None else a
  b = jnp.ones((num_b,)) / num_b if b is None else b

  gromov_partial = functools.partial(
      _gromovw_iterations, epsilon=epsilon, loss=loss_fn,
      tau_a=tau_a, tau_b=tau_b, max_iterations_gromov=max_iterations_gromov,
      sinkhorn_kwargs=sinkhorn_kwargs, **kwargs)
  gromov_fn = jax.jit(gromov_partial) if jit_gromov else gromov_partial
  f, g, geom_gw, reg_gw_cost_arr, errors_sinkhorn, converged_sinkhorn = gromov_fn(
      geom_x, geom_y, a, b)

  transport = geom_gw.transport_from_potentials(f, g)
  cost_matrix = 0.5 * geom_gw.cost_matrix
  gw_cost = jnp.sum(transport * cost_matrix)
  reg_gw_cost = sinkhorn.ent_reg_cost(geom_gw, a, b, tau_a, tau_b, f, g)
  return GromovWassersteinOutput(f, g, transport, cost_matrix, gw_cost,
                                 reg_gw_cost, reg_gw_cost_arr, errors_sinkhorn,
                                 converged_sinkhorn)


def _gromovw_iterations(
    geom_x: geometry.Geometry,
    geom_y: geometry.Geometry,
    a: jnp.ndarray,
    b: jnp.ndarray,
    epsilon: float,
    loss: GWLoss,
    tau_a: float,
    tau_b: float,
    max_iterations_gromov: int,
    sinkhorn_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs) -> Tuple[jnp.ndarray]:
  """Fits Gromov Wasserstein.

  Args:
    geom_x: a Geometry object for the first view.
    geom_y: a second Geometry object for the second view.
    a: jnp.ndarray<float>[num_a,] or jnp.ndarray<float>[batch,num_a] weights.
    b: jnp.ndarray<float>[num_b,] or jnp.ndarray<float>[batch,num_b] weights.
    epsilon: a regularization parameter or a epsilon_scheduler.Epsilon object.
    loss: GWLoss object.
    tau_a: float, ratio lam/(lam+eps) between KL divergence regularizer to first
     marginal and itself + epsilon regularizer used in the unbalanced
     formulation.
    tau_b: float, ratio lam/(lam+eps) between KL divergence regularizer to 
     second marginal and itself + epsilon regularizer used in the unbalanced
     formulation.
    max_iterations_gromov: int32, the maximum number of outer iterations for
    Gromov Wasserstein.
    sinkhorn_kwargs: Optionally a dictionary containing the keywords arguments
     for calls to the sinkhorn function.
    **kwargs: additional kwargs for epsilon.

  Returns:
    f: potential
    g: potential
    geom_gw: a Geometry object for Gromov-Wasserstein (GW)
    reg_gw_cost_arr: ndarray of regularised GW costs
    errors_sinkhorn: ndarray [max_iterations_gromov, p], where p depends on
     sinkhorn_kwargs, of errors for the Sinkhorn algorithm for each gromov
     iteration (axis 0) and regularly spaced sinkhorn iterations (axis 1).
    converged_sinkhorn: ndarray [max_iterations_gromov] of flags indicating that
     the sinkhorn algorithm converged
  """
  # Initialises cost, GW geometry, runs first iteration of sinkhorn
  gw_constant = _init_constant_cost(geom_x, geom_y, a, b, loss)
  geom_gw = _init_geometry_gw(geom_x, geom_y, a, b, epsilon,
                              gw_constant, loss, **kwargs)
  f, g, reg_gw_cost, error_sinkhorn, converged_sinkhorn = sinkhorn.sinkhorn(
      geom_gw, a, b, tau_a, tau_b, **sinkhorn_kwargs)

  carry = geom_gw, f, g
  update_geom_partial = functools.partial(_update_geometry_gw, geom_x=geom_x,
                                          geom_y=geom_y,
                                          gw_constant=gw_constant,
                                          loss=loss, **kwargs)
  sinkhorn_partial = functools.partial(sinkhorn.sinkhorn, a=a, b=b,
                                       tau_a=tau_a, tau_b=tau_b,
                                       **sinkhorn_kwargs)
  def body_fn(carry=carry, x=None):
    del x
    geom_gw, f, g = carry
    geom_gw = update_geom_partial(geom=geom_gw, f=f, g=g)
    f, g, reg_gw_cost, error_sinkhorn, converged_sinkhorn = sinkhorn_partial(
        geom=geom_gw)
    return (geom_gw, f, g), (reg_gw_cost, error_sinkhorn, converged_sinkhorn)

  carry, out = jax.lax.scan(f=body_fn, init=carry, xs=None,
                            length=max_iterations_gromov - 1)
  geom_gw, f, g = carry
  reg_gw_cost_arr = jnp.concatenate((jnp.array([reg_gw_cost]), out[0]))
  errors_sinkhorn = jnp.concatenate((jnp.array([error_sinkhorn]),
                                     out[1]))
  converged_sinkhorn = jnp.concatenate((jnp.array([converged_sinkhorn]),
                                        out[2]))
  return f, g, geom_gw, reg_gw_cost_arr, errors_sinkhorn, converged_sinkhorn


def _init_constant_cost(
    geom_x: geometry.Geometry,
    geom_y: geometry.Geometry,
    a: jnp.ndarray,
    b: jnp.ndarray,
    loss: GWLoss) -> float:
  """Constant term in the GW cost, Equation 6, Proposition 1 of
  http://proceedings.mlr.press/v48/peyre16.pdf.

  Args:
    geom_x: a Geometry object for the first view.
    geom_y: a second Geometry object for the second view.
    a: jnp.ndarray<float>[num_a,] or jnp.ndarray<float>[batch,num_a] weights.
    b: jnp.ndarray<float>[num_b,] or jnp.ndarray<float>[batch,num_b] weights.
    loss: a GWLossFn object.

  Returns:
    float, constant value in cost over GW iterations
  """
  a = a.reshape(-1, 1)
  b = b.reshape(-1, 1)
  constant1 = jnp.dot(geom_x.apply_cost(a, 1, loss.fn_x),
                      jnp.ones((1, b.size)))
  constant2 = jnp.dot(geom_y.apply_cost(b, 1, loss.fn_y),
                      jnp.ones((1, a.size))).T
  return constant1 + constant2


def _init_geometry_gw(
    geom_x: geometry.Geometry,
    geom_y: geometry.Geometry,
    a: jnp.ndarray,
    b: jnp.ndarray,
    epsilon: float,
    gw_constant: float,
    loss: GWLoss,
    **kwargs) -> geometry.Geometry:
  """Initialises the geometry object for GW and updates the cost matrices of
  the geometries for the two views according to the chosen loss.

  Args:
    geom_x: a Geometry object for the first view.
    geom_y: a second Geometry object for the second view.
    a: jnp.ndarray<float>[num_a,] or jnp.ndarray<float>[batch,num_a] weights.
    b: jnp.ndarray<float>[num_b,] or jnp.ndarray<float>[batch,num_b] weights.
    epsilon: a regularization parameter or a epsilon_scheduler.Epsilon object.
    gw_constant: float, constant term in the GW cost
    loss: a GWLossFn object.
    **kwargs: additional kwargs to epsilon

  Returns:
    A Geometry object for Gromov-Wasserstein.
  """
  ab = a[:, None] * b[None, :]

  tmp = geom_x.apply_cost(ab, axis=1, fn=loss.left_x)
  cost_matrix = gw_constant - geom_y.apply_cost(
      tmp.T, axis=1, fn=loss.right_y).T
  return geometry.Geometry(cost_matrix=cost_matrix,
                           epsilon=epsilon, **kwargs)


def _update_geometry_gw(
    geom: geometry.Geometry,
    geom_x: geometry.Geometry,
    geom_y: geometry.Geometry,
    f: jnp.ndarray,
    g: jnp.ndarray,
    gw_constant: float,
    loss: GWLoss,
    **kwargs) -> geometry.Geometry:
  """Updates the geometry object for GW.

  Args:
    geom: a Geometry object carrying the cost matrix of Gromov Wasserstein.
    geom_x: a Geometry object for the first view.
    geom_y: a second Geometry object for the second view.
    f: potentials
    g: potentials
    gw_constant: float, constant term in the GW cost
    loss: a GWLossFn object.
    **kwargs: additional kwargs for epsilon

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

  # Computes cost_matrix = gw_constant - tmp * cost_matrix_y.T
  cost_matrix = gw_constant - apply_cost_fn(geom_y)(
      tmp.T, axis=1, fn=loss.right_y).T
  return geometry.Geometry(cost_matrix=cost_matrix,
                           epsilon=geom._epsilon, **kwargs)
