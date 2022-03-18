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

# Lint as: python 3
"""Pytree for a Gaussian mixture model."""

from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from ott.tools.gaussian_mixture import gaussian
from ott.tools.gaussian_mixture import linalg
from ott.tools.gaussian_mixture import probabilities
from ott.tools.gaussian_mixture import scale_tril


def get_summary_stats_from_points_and_assignment_probs(
    points: jnp.ndarray,
    point_weights: jnp.ndarray,
    assignment_probs: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Get component summary stats from points and component probabilities.

  Args:
    points: array of points, shape (n, n_dim)
    point_weights: array of weights for the points, shape (n,)
    assignment_probs: array of component assignment probabilities for the
      points, shape (n, n_components)

  Returns:
    Tuple containing for each component,
    * the sample mean for each component, shape (n_components, n_dim)
    * the sample covariance for each component,
        shape (n_components, n_dim, n_dim)
    * the weight for each component,
        shape (n_components,)
  """
  def component_from_points(points, point_weights, assignment_probs):
    component_weight = (
        jnp.sum(point_weights * assignment_probs) / jnp.sum(point_weights))
    component_mean, component_cov = linalg.get_mean_and_cov(
        points=points, weights=point_weights * assignment_probs)
    return component_mean, component_cov, component_weight
  components_from_points_fn = jax.vmap(
      component_from_points, in_axes=(None, None, 1), out_axes=0)

  return components_from_points_fn(points, point_weights, assignment_probs)


@jax.tree_util.register_pytree_node_class
class GaussianMixture:
  """Pytree for a Gaussian Mixture model."""

  def __init__(
      self,
      loc: jnp.ndarray,
      scale_params: jnp.ndarray,
      component_weight_ob: probabilities.Probabilities):
    self._loc = loc
    self._scale_params = scale_params
    self._component_weight_ob = component_weight_ob

  @classmethod
  def from_random(
      cls,
      key: jnp.ndarray,
      n_components: int,
      n_dimensions: int,
      stdev: float = 0.1,
      dtype: Optional[jnp.dtype] = None) -> 'GaussianMixture':
    """Construct a random GMM."""
    loc = []
    scale_params = []
    for _ in range(n_components):
      key, subkey = jax.random.split(key)
      component = gaussian.Gaussian.from_random(
          key=subkey,
          n_dimensions=n_dimensions,
          stdev=stdev,
          dtype=dtype)
      loc.append(component.loc)
      scale_params.append(component.scale.params)
    loc = jnp.stack(loc, axis=0)
    scale_params = jnp.stack(scale_params, axis=0)
    weight_ob = probabilities.Probabilities.from_random(
        key=subkey, n_dimensions=n_components, stdev=stdev, dtype=dtype)
    return cls(loc=loc,
               scale_params=scale_params,
               component_weight_ob=weight_ob)

  @classmethod
  def from_mean_cov_component_weights(
      cls,
      mean: jnp.ndarray,
      cov: jnp.ndarray,
      component_weights: jnp.ndarray):
    """Construct a GMM from means, covariances, and component weights."""
    scale_params = []
    for i in range(cov.shape[0]):
      scale_params.append(
          scale_tril.ScaleTriL.from_covariance(cov[i]).params)
    scale_params = jnp.stack(scale_params, axis=0)
    weight_ob = probabilities.Probabilities.from_probs(
        component_weights)
    return cls(loc=mean,
               scale_params=scale_params,
               component_weight_ob=weight_ob)

  @classmethod
  def from_points_and_assignment_probs(
      cls,
      points: jnp.ndarray,
      point_weights: jnp.ndarray,
      assignment_probs: jnp.ndarray,
  )  -> 'GaussianMixture':
    """Estimate a GMM from points and a set of component probabilities."""
    mean, cov, wts = get_summary_stats_from_points_and_assignment_probs(
        points=points,
        point_weights=point_weights,
        assignment_probs=assignment_probs)
    return cls.from_mean_cov_component_weights(
        mean=mean, cov=cov, component_weights=wts)

  @property
  def dtype(self):
    return self.loc.dtype

  @property
  def n_dimensions(self):
    return self._loc.shape[-1]

  @property
  def n_components(self):
    return self._loc.shape[-2]

  @property
  def loc(self) -> jnp.ndarray:
    return self._loc

  @property
  def scale_params(self) -> jnp.ndarray:
    return self._scale_params

  @property
  def cholesky(self) -> jnp.ndarray:
    size = self.n_dimensions
    def _get_cholesky(scale_params):
      return scale_tril.ScaleTriL(params=scale_params, size=size).cholesky()
    return jax.vmap(_get_cholesky, in_axes=0, out_axes=0)(self.scale_params)

  @property
  def covariance(self) -> jnp.ndarray:
    size = self.n_dimensions
    def _get_covariance(scale_params):
      return scale_tril.ScaleTriL(params=scale_params, size=size).covariance()
    return jax.vmap(_get_covariance, in_axes=0, out_axes=0)(self.scale_params)

  @property
  def component_weight_ob(self) -> probabilities.Probabilities:
    return self._component_weight_ob

  @property
  def component_weights(self) -> jnp.ndarray:
    return self._component_weight_ob.probs()

  def log_component_weights(self) -> jnp.ndarray:
    return self._component_weight_ob.log_probs()

  def _get_normal(self,
                  loc: jnp.ndarray,
                  scale_params: jnp.ndarray) -> gaussian.Gaussian:
    size = loc.shape[-1]
    return gaussian.Gaussian(
        loc=loc,
        scale=scale_tril.ScaleTriL(params=scale_params, size=size))

  def get_component(self, index: int) -> gaussian.Gaussian:
    """Get the specified GMM component."""
    return self._get_normal(
        loc=self.loc[index], scale_params=self.scale_params[index])

  def components(self) -> List[gaussian.Gaussian]:
    """Get a list of all GMM components."""
    return [self.get_component(i) for i in range(self.n_components)]

  def sample(self, key: jnp.ndarray, size: int)-> jnp.ndarray:
    """Generate samples from the distribution."""
    subkey0, subkey1 = jax.random.split(key)
    component = self.component_weight_ob.sample(key=subkey0, size=size)
    std_samples = jax.random.normal(
        key=subkey1, shape=(size, self.n_dimensions))

    def _transform_single_component(k, scale, loc):
      def _transform_single_value(single_component, single_x):
        return jax.lax.cond(
            single_component == k,
            lambda x: jnp.matmul(scale, x[:, None])[:, 0] + loc,
            jnp.zeros_like,
            single_x)
      return jax.vmap(_transform_single_value)(
          component, std_samples)
    return jnp.sum(
        jax.vmap(_transform_single_component)(
            jnp.arange(self.n_components), self.cholesky, self.loc),
        axis=0)

  def conditional_log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
    """Compute the component-conditional log probability of x.

    Args:
      x: (n, n_dimensions) array of points

    Returns:
      (n, n_components) array of the log probability of x conditioned on it
      having come from each component.
    """
    def _log_prob_single_component(
        loc: jnp.ndarray,
        scale_params: jnp.ndarray,
        x: jnp.ndarray):
      norm = self._get_normal(loc=loc, scale_params=scale_params)
      return norm.log_prob(x)

    conditional_log_prob_fn = jax.vmap(
        _log_prob_single_component, in_axes=(0, 0, None), out_axes=1)
    return conditional_log_prob_fn(self._loc, self._scale_params, x)

  def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
    """Compute the log probability of the observations x.

    Args:
      x: (n, n_dimensions) array of points

    Returns:
      (n,) array of log probabilities.
    """
    # p(x) = \sum_i p(x|c_i) p(c_i)
    log_prob_conditional = self.conditional_log_prob(x)
    log_component_weight = self.log_component_weights()
    return jax.scipy.special.logsumexp(
        log_prob_conditional + log_component_weight[None, :], axis=-1)

  def get_log_component_posterior(self, x: jnp.ndarray) -> jnp.ndarray:
    """Compute the posterior probability that x came from each component.

    Args:
      x: (n, n_dimensions) array of points

    Returns:
      (n, n_components) array of poster component log probabilities.
    """
    # p(x | c_i) = p(x, c_i) / p(c_i) => p(x, c_i) = p(x | c_i) p(c_i)
    # p(c_i | x) = p(x, c_i) / p(x)
    #            = p(x | c_i) p(c_i) / sum_j(p(x | c_j)p(c_j))
    log_prob_conditional = self.conditional_log_prob(x)
    log_component_weight = self.log_component_weights()
    log_prob_unnorm = log_prob_conditional + log_component_weight[None, :]
    return log_prob_unnorm - jax.scipy.special.logsumexp(
        log_prob_unnorm, axis=-1, keepdims=True)

  def has_nans(self) -> bool:
    for leaf in jax.tree_util.tree_leaves(self):
      if jnp.any(~jnp.isfinite(leaf)):
        return True
    return False

  def tree_flatten(self):
    children = (self.loc, self.scale_params, self.component_weight_ob)
    aux_data = {}
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(*children, **aux_data)

  def __repr__(self):
    class_name = type(self).__name__
    children, aux = self.tree_flatten()
    return '{}({})'.format(
        class_name, ', '.join([repr(c) for c in children] +
                              [f'{k}: {repr(v)}' for k, v in aux.items()]))

  def __hash__(self):
    return jax.tree_util.tree_flatten(self).__hash__()

  def __eq__(self, other):
    return jax.tree_util.tree_flatten(self) == jax.tree_util.tree_flatten(other)
