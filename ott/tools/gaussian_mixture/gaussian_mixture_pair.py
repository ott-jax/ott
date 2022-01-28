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

"""Pytree containing parameters for a pair of coupled Gaussian mixture models.
"""
import jax
import jax.numpy as jnp
from ott.core import sinkhorn
from ott.geometry import costs
from ott.geometry import geometry
from ott.geometry import pointcloud

from ott.tools.gaussian_mixture import gaussian_mixture


@jax.tree_util.register_pytree_node_class
class GaussianMixturePair:
  """Pytree for a coupled pair of Gaussian mixture models.

  Includes methods used in estimating an optimal pairing between GMM components
  using the Wasserstein-like method described in
  https://arxiv.org/abs/1907.05254 as well as generalization that allows for
  the re-weighting of components.

  The Delon & Desolneux paper above proposes fitting a pair of GMMs to a pair
  of point clouds in such a way that the sum of the log likelihood of the
  points minus a weighted penalty involving a Wasserstein-like distance between
  the GMMs. Their proposed algorithm involves using EM in which a balanced
  Sinkhorn algorithm is used to estimate a coupling between the GMMs at each
  step of EM.

  Our generalization of this algorithm allows for a mismatch between the
  marginals of the coupling and the GMM component weights. This mismatch can be
  interpreted as components being re-weighted rather than being transported.
  We penalize reweighting with a generalized KL-divergence penalty, and we give
  the option to use the unbalanced Sinkhorn algorithm rather than the balanced
  to compute the divergence between GMMs.
  """

  def __init__(
      self,
      gmm0: gaussian_mixture.GaussianMixture,
      gmm1: gaussian_mixture.GaussianMixture,
      epsilon: float = 1.e-2,
      tau: float = 1.,
      lock_gmm1: bool = False,
    ):
    """Constructor.

    When fitting a pair of coupled GMMs with *no* reweighting of components
    using the algorithm in https://arxiv.org/abs/1907.05254, set tau = 1. The
    coupling between components will be determined via the balanced Sinkhorn
    algorithm.

    When fitting a pair of coupled GMMs in which reweighting of components is
    allowed, set tau to a value in (0, 1). The resulting coupling will penalize
    the generalized KL divergence between the coupling's marginals and the GMM
    component weights with a weight of rho = epsilon tau / (1 - tau).

    Args:
      gmm0: first GMM in the pair
      gmm1: second GMM in the pair
      epsilon: regularization weight to use for the Sinkhorn algorithm
      tau: encodes the weight, rho, to use for the generalized KL divergence
        between the coupling's marginals and GMM component weights as
        rho = epsilon tau / (1 - tau)
      lock_gmm1: indicates whether the parameters of gmm1 should be modified
        during optimization
    """
    self._gmm0 = gmm0
    self._gmm1 = gmm1
    self._epsilon = epsilon
    self._tau = tau
    self._lock_gmm1 = lock_gmm1

  @property
  def dtype(self):
    return self.gmm0.dtype

  @property
  def gmm0(self):
    return self._gmm0

  @property
  def gmm1(self):
    return self._gmm1

  @property
  def epsilon(self):
    return self._epsilon

  @property
  def tau(self):
    return self._tau

  @property
  def rho(self):
    return self.epsilon * self.tau / (1. - self.tau)

  @property
  def lock_gmm1(self):
    return self._lock_gmm1

  def get_bures_geometry(self) -> pointcloud.PointCloud:
    """Get a Bures Geometry for the two GMMs."""
    mean0 = self.gmm0.loc
    dimension = mean0.shape[-1]
    cov0 = self.gmm0.covariance
    cov0 = cov0.reshape(cov0.shape[:-2] + (dimension * dimension,))
    x = jnp.concatenate([mean0, cov0], axis=-1)
    mean1 = self.gmm1.loc
    cov1 = self.gmm1.covariance
    cov1 = cov1.reshape(cov1.shape[:-2] + (dimension * dimension,))
    y = jnp.concatenate([mean1, cov1], axis=-1)
    return pointcloud.PointCloud(
        x=x, y=y,
        cost_fn=costs.Bures(dimension=dimension),
        epsilon=self.epsilon)

  def get_cost_matrix(self) -> jnp.ndarray:
    """Get matrix of W2^2 costs between all pairs of (gmm0, gmm1) components."""
    return self.get_bures_geometry().cost_matrix

  def get_sinkhorn(self, cost_matrix: jnp.ndarray) -> sinkhorn.SinkhornOutput:
    """Get the output of ott.sinkhorn's method for a given cost matrix."""
    # We use a Geometry here rather than the PointCloud created in
    # in get_bures_geometry to avoid recomputing the cost matrix, since
    # the cost matrix is quite expensive
    geom = geometry.Geometry(cost_matrix=cost_matrix, epsilon=self.epsilon)
    return sinkhorn.sinkhorn(
        geom,
        a=self.gmm0.component_weights,
        b=self.gmm1.component_weights,
        tau_a=self.tau, tau_b=self.tau)

  def get_normalized_sinkhorn_coupling(
      self,
      sinkhorn_output: sinkhorn.SinkhornOutput,
  ) -> jnp.ndarray:
    """Get the normalized coupling matrix for the specified Sinkhorn output.

    Args:
      sinkhorn_output: Sinkhorn algorithm output as returned by get_sinkhorn()

    Returns:
      A coupling matrix that tells how much of the mass of each component of
      gmm0 is mapped to each component of gmm1.
    """
    return sinkhorn_output.matrix / jnp.sum(sinkhorn_output.matrix)

  def tree_flatten(self):
    """Method used by jax.tree_util to flatten a GaussianMixturePair.

    We control the subset of parameters that we will optimize in fit_gmm_pair
    by selectively placing them in either children (the parameters to optimize)
    or aux_data (the parameters to leave alone).

    Returns:
      A tuple of child pytrees and a dict of auxiliary data.
    """
    children = [self.gmm0]
    aux_data = {'epsilon': self.epsilon,
                'tau': self.tau,
                'lock_gmm1': self.lock_gmm1}
    if self.lock_gmm1:
      aux_data['gmm1'] = self.gmm1
    else:
      children.append(self.gmm1)
    return tuple(children), aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    """Method used by jax.tree_util to unflatten a GaussianMixturePair.

    tree_flatten controls which parameters get optimized by placing them in
    either children or aux_data; here we invert the process.

    Args:
      aux_data: auxiliary data that is passed to the constructor as kwargs
      children: child pytrees passed to the constructor as args

    Returns:
      A GaussianMixturePair.
    """
    children = list(children)
    if 'gmm1' in aux_data:
      gmm1 = aux_data.pop('gmm1')
      children.insert(1, gmm1)
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
