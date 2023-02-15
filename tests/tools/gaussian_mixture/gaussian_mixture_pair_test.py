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
"""Tests for gaussian_mixture_pair."""

import pytest

import jax
import jax.numpy as jnp
import numpy as np

from ott.tools.gaussian_mixture import gaussian_mixture, gaussian_mixture_pair


class TestGaussianMixturePair:

  @pytest.fixture(autouse=True)
  def initialize(self, rng: jnp.ndarray):
    self.n_components = 3
    self.n_dimensions = 2
    self.epsilon = 1.e-3
    self.rho = 0.1
    self.tau = self.rho / (self.rho + self.epsilon)
    self.key, subkey0, subkey1 = jax.random.split(rng, num=3)
    self.gmm0 = gaussian_mixture.GaussianMixture.from_random(
        key=subkey0,
        n_components=self.n_components,
        n_dimensions=self.n_dimensions
    )
    self.gmm1 = gaussian_mixture.GaussianMixture.from_random(
        key=subkey1,
        n_components=self.n_components,
        n_dimensions=self.n_dimensions
    )
    self.balanced_pair = gaussian_mixture_pair.GaussianMixturePair(
        gmm0=self.gmm0, gmm1=self.gmm1, epsilon=self.epsilon, tau=1.
    )
    self.unbalanced_pair = gaussian_mixture_pair.GaussianMixturePair(
        gmm0=self.gmm0, gmm1=self.gmm1, epsilon=self.epsilon, tau=self.tau
    )

  def test_get_cost_matrix(self):
    cost_matrix = self.balanced_pair.get_cost_matrix()
    for i0, comp0 in enumerate(self.gmm0.components()):
      for i1, comp1 in enumerate(self.gmm1.components()):
        expected = comp0.w2_dist(comp1)
        actual = cost_matrix[i0, i1]
        np.testing.assert_approx_equal(actual, expected, significant=6)

  def test_get_sinkhorn_to_same_gmm_is_almost_zero(self):
    gmm = self.gmm0
    pair = gaussian_mixture_pair.GaussianMixturePair(
        gmm0=gmm,
        gmm1=gmm,  # use same GMM for both gmm0 and gmm1
        epsilon=self.epsilon,
        tau=1.
    )
    cost_matrix = pair.get_cost_matrix()
    sinkhorn_output = pair.get_sinkhorn(cost_matrix=cost_matrix)
    cost = sinkhorn_output.reg_ot_cost

    np.testing.assert_almost_equal(cost, 0.00, decimal=2)

  @pytest.mark.fast
  def test_get_sinkhorn_to_shifted_is_almost_shift(self):
    loc_shift = jnp.stack([
        2. * jnp.ones(self.n_components),
        jnp.zeros(self.n_components)
    ],
                          axis=-1)
    gmm1 = gaussian_mixture.GaussianMixture(
        loc=self.gmm0.loc + loc_shift,
        scale_params=self.gmm0.scale_params,
        component_weight_ob=self.gmm0.component_weight_ob
    )
    pair = gaussian_mixture_pair.GaussianMixturePair(
        gmm0=self.gmm0, gmm1=gmm1, epsilon=self.epsilon, tau=1.
    )
    cost_matrix = pair.get_cost_matrix()
    sinkhorn_output = pair.get_sinkhorn(cost_matrix=cost_matrix)
    cost = sinkhorn_output.reg_ot_cost

    np.testing.assert_approx_equal(cost, 4.0, significant=2)

  @pytest.mark.fast
  def test_get_coupling_between_same_gmm(self):
    gmm = self.gmm0
    pair = gaussian_mixture_pair.GaussianMixturePair(
        gmm0=gmm,
        gmm1=gmm,  # use same GMM for both gmm0 and gmm1
        epsilon=self.epsilon,
        tau=1.
    )
    cost_matrix = pair.get_cost_matrix()
    sinkhorn_output = pair.get_sinkhorn(cost_matrix=cost_matrix)
    coupling = pair.get_normalized_sinkhorn_coupling(
        sinkhorn_output=sinkhorn_output
    )
    expected = jnp.diag(self.gmm0.component_weights)

    np.testing.assert_allclose(coupling, expected, atol=1e-6)

  def test_get_coupling_to_shifted(self):
    loc_shift = jnp.stack([
        2. * jnp.ones(self.n_components),
        jnp.zeros(self.n_components)
    ],
                          axis=-1)
    gmm1 = gaussian_mixture.GaussianMixture(
        loc=self.gmm0.loc + loc_shift,
        scale_params=self.gmm0.scale_params,
        component_weight_ob=self.gmm0.component_weight_ob
    )
    pair = gaussian_mixture_pair.GaussianMixturePair(
        gmm0=self.gmm0, gmm1=gmm1, epsilon=self.epsilon, tau=1.
    )
    cost_matrix = pair.get_cost_matrix()
    sinkhorn_output = pair.get_sinkhorn(cost_matrix=cost_matrix)
    coupling = pair.get_normalized_sinkhorn_coupling(
        sinkhorn_output=sinkhorn_output
    )
    expected = jnp.diag(self.gmm0.component_weights)
    np.testing.assert_allclose(expected, coupling, atol=5e-3)

  @pytest.mark.fast.with_args(
      "epsilon,tau,lock_gmm1",
      [(1e-2, 1, False), (1e-2, 1, True), (1e-2, 0.1 / (0.1 + 1e-2), False),
       (1e-2, 0.1 / (0.1 + 1e-2), True)],
      only_fast=0,
  )
  def test_flatten_unflatten(self, epsilon, tau, lock_gmm1):
    pair = gaussian_mixture_pair.GaussianMixturePair(
        gmm0=self.gmm0,
        gmm1=self.gmm1,
        epsilon=epsilon,
        tau=tau,
        lock_gmm1=lock_gmm1
    )
    children, aux_data = jax.tree_util.tree_flatten(pair)
    pair_new = jax.tree_util.tree_unflatten(aux_data, children)

    assert pair.gmm0 == pair_new.gmm0
    assert pair.gmm1 == pair_new.gmm1
    assert pair.epsilon == pair_new.epsilon
    assert pair.tau == pair_new.tau
    assert pair.lock_gmm1 == pair_new.lock_gmm1
    assert pair == pair_new

  @pytest.mark.fast.with_args(
      "epsilon,tau,lock_gmm1",
      [(1e-2, 1, False), (1e-2, 1, True), (1e-2, 0.1 / (0.1 + 1e-2), False),
       (1e-2, 0.1 / (0.1 + 1e-2), True)],
      only_fast=0,
  )
  def test_pytree_mapping(self, epsilon, tau, lock_gmm1):
    pair = gaussian_mixture_pair.GaussianMixturePair(
        gmm0=self.gmm0,
        gmm1=self.gmm1,
        epsilon=epsilon,
        tau=tau,
        lock_gmm1=lock_gmm1
    )
    expected_gmm1_loc = 2. * self.gmm1.loc if not lock_gmm1 else self.gmm1.loc

    pair_x_2 = jax.tree_map(lambda x: 2 * x, pair)
    # gmm parameters should be doubled
    np.testing.assert_allclose(2. * pair.gmm0.loc, pair_x_2.gmm0.loc)
    np.testing.assert_allclose(expected_gmm1_loc, pair_x_2.gmm1.loc)
    # epsilon and tau should not
    assert pair.epsilon == pair_x_2.epsilon
    assert pair.tau == pair_x_2.tau
