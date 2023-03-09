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
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ott.geometry import costs, geometry, pointcloud
from ott.solvers.linear import acceleration, sinkhorn
from ott.tools import sinkhorn_divergence
from ott.tools.gaussian_mixture import gaussian_mixture


class TestSinkhornDivergence:

  @pytest.fixture(autouse=True)
  def setUp(self, rng: jax.random.PRNGKeyArray):
    self._dim = 4
    self._num_points = 13, 17
    self.rng, *rngs = jax.random.split(rng, 3)
    a = jax.random.uniform(rngs[0], (self._num_points[0],))
    b = jax.random.uniform(rngs[1], (self._num_points[1],))
    self._a = a / jnp.sum(a)
    self._b = b / jnp.sum(b)

  @pytest.mark.fast.with_args(
      cost_fn=[costs.Euclidean(),
               costs.SqEuclidean(),
               costs.SqPNorm(p=2.1)],
      epsilon=[1e-2, 1e-3],
      only_fast={
          "cost_fn": costs.SqEuclidean(),
          "epsilon": 1e-2
      },
  )
  def test_euclidean_point_cloud(self, cost_fn: costs.CostFn, epsilon: float):
    rngs = jax.random.split(self.rng, 2)
    x = jax.random.uniform(rngs[0], (self._num_points[0], self._dim))
    y = jax.random.uniform(rngs[1], (self._num_points[1], self._dim))

    div = sinkhorn_divergence.sinkhorn_divergence(
        pointcloud.PointCloud,
        x,
        y,
        cost_fn=cost_fn,
        a=self._a,
        b=self._b,
        epsilon=epsilon
    )
    assert div.divergence > 0.0
    assert len(div.potentials) == 3

    geometry_xy = pointcloud.PointCloud(x, y, epsilon=epsilon, cost_fn=cost_fn)
    geometry_xx = pointcloud.PointCloud(x, epsilon=epsilon, cost_fn=cost_fn)
    geometry_yy = pointcloud.PointCloud(y, epsilon=epsilon, cost_fn=cost_fn)

    div2 = sinkhorn.solve(geometry_xy, self._a, self._b).reg_ot_cost
    div2 -= 0.5 * sinkhorn.solve(geometry_xx, self._a, self._a).reg_ot_cost
    div2 -= 0.5 * sinkhorn.solve(geometry_yy, self._b, self._b).reg_ot_cost

    np.testing.assert_allclose(div.divergence, div2, rtol=1e-5, atol=1e-5)

    # Test div of x to itself close to 0.
    div = sinkhorn_divergence.sinkhorn_divergence(
        pointcloud.PointCloud,
        x,
        x,
        cost_fn=cost_fn,
        epsilon=1e-1,
        sinkhorn_kwargs={"inner_iterations": 1},
    )
    np.testing.assert_allclose(div.divergence, 0.0, rtol=1e-5, atol=1e-5)
    iters_xx = jnp.sum(div.errors[0] > 0)
    iters_xx_sym = jnp.sum(div.errors[1] > 0)
    assert iters_xx >= iters_xx_sym

  @pytest.mark.fast()
  def test_euclidean_autoepsilon(self):
    rngs = jax.random.split(self.rng, 2)
    cloud_a = jax.random.uniform(rngs[0], (self._num_points[0], self._dim))
    cloud_b = jax.random.uniform(rngs[1], (self._num_points[1], self._dim))
    div = sinkhorn_divergence.sinkhorn_divergence(
        pointcloud.PointCloud,
        cloud_a,
        cloud_b,
        a=self._a,
        b=self._b,
        sinkhorn_kwargs={"threshold": 1e-2},
    )
    assert div.divergence > 0.0
    assert len(div.potentials) == 3
    assert len(div.geoms) == 3
    np.testing.assert_allclose(div.geoms[0].epsilon, div.geoms[1].epsilon)

  def test_euclidean_autoepsilon_not_share_epsilon(self):
    rngs = jax.random.split(self.rng, 2)
    cloud_a = jax.random.uniform(rngs[0], (self._num_points[0], self._dim))
    cloud_b = jax.random.uniform(rngs[1], (self._num_points[1], self._dim))
    div = sinkhorn_divergence.sinkhorn_divergence(
        pointcloud.PointCloud,
        cloud_a,
        cloud_b,
        a=self._a,
        b=self._b,
        sinkhorn_kwargs={"threshold": 1e-2},
        share_epsilon=False
    )
    assert jnp.abs(div.geoms[0].epsilon - div.geoms[1].epsilon) > 0

  @pytest.mark.parametrize("use_weights", [False, True])
  def test_euclidean_point_cloud_wrapper(self, use_weights: bool):
    rngs = jax.random.split(self.rng, 2)
    cloud_a = jax.random.uniform(rngs[0], (self._num_points[0], self._dim))
    cloud_b = jax.random.uniform(rngs[1], (self._num_points[1], self._dim))
    kwargs = {"a": self._a, "b": self._b} if use_weights else {}
    div = sinkhorn_divergence.sinkhorn_divergence(
        pointcloud.PointCloud,
        cloud_a,
        cloud_b,
        epsilon=0.1,
        sinkhorn_kwargs={"threshold": 1e-2},
        **kwargs
    )
    assert div.divergence > 0.0
    assert len(div.potentials) == 3
    assert len(div.geoms) == 3

  @pytest.mark.fast()
  def test_euclidean_point_cloud_unbalanced_wrapper(self):
    rngs = jax.random.split(self.rng, 2)
    cloud_a = jax.random.uniform(rngs[0], (self._num_points[0], self._dim))
    cloud_b = jax.random.uniform(rngs[1], (self._num_points[1], self._dim))
    div = sinkhorn_divergence.sinkhorn_divergence(
        pointcloud.PointCloud,
        cloud_a,
        cloud_b,
        epsilon=0.1,
        a=self._a + .001,
        b=self._b + .002,
        sinkhorn_kwargs={
            "threshold": 1e-2,
            "tau_a": 0.8,
            "tau_b": 0.9
        }
    )
    assert div.divergence > 0.0
    assert len(div.potentials) == 3
    assert len(div.geoms) == 3

  def test_generic_point_cloud_wrapper(self):
    rngs = jax.random.split(self.rng, 2)
    x = jax.random.uniform(rngs[0], (self._num_points[0], self._dim))
    y = jax.random.uniform(rngs[1], (self._num_points[1], self._dim))

    # Tests with 3 cost matrices passed as args
    cxy = jnp.sum(jnp.abs(x[:, jnp.newaxis] - y[jnp.newaxis, :]) ** 2, axis=2)
    cxx = jnp.sum(jnp.abs(x[:, jnp.newaxis] - x[jnp.newaxis, :]) ** 2, axis=2)
    cyy = jnp.sum(jnp.abs(y[:, jnp.newaxis] - y[jnp.newaxis, :]) ** 2, axis=2)
    div = sinkhorn_divergence.sinkhorn_divergence(
        geometry.Geometry,
        cxy,
        cxx,
        cyy,
        epsilon=0.1,
        a=self._a,
        b=self._b,
        sinkhorn_kwargs={"threshold": 1e-2},
    )
    assert div.divergence > 0.0
    assert len(div.potentials) == 3
    assert len(div.geoms) == 3

    # Tests with 2 cost matrices passed as args
    div = sinkhorn_divergence.sinkhorn_divergence(
        geometry.Geometry,
        cxy,
        cxx,
        epsilon=0.1,
        a=self._a,
        b=self._b,
        sinkhorn_kwargs={"threshold": 1e-2},
    )
    assert div.divergence > 0.0
    assert len(div.potentials) == 3
    assert len(div.geoms) == 3

    # Tests with 3 cost matrices passed as kwargs
    div = sinkhorn_divergence.sinkhorn_divergence(
        geometry.Geometry,
        cost_matrix=(cxy, cxx, cyy),
        epsilon=0.1,
        a=self._a,
        b=self._b,
        sinkhorn_kwargs={"threshold": 1e-2},
    )
    assert div.divergence > 0.0
    assert len(div.potentials) == 3
    assert len(div.geoms) == 3

  @pytest.mark.parametrize("shuffle", [False, True])
  def test_segment_sinkhorn_result(self, shuffle: bool):
    # Test that segmented sinkhorn gives the same results:
    rngs = jax.random.split(self.rng, 4)
    x = jax.random.uniform(rngs[0], (self._num_points[0], self._dim))
    y = jax.random.uniform(rngs[1], (self._num_points[1], self._dim))
    geom_kwargs = {"epsilon": 0.01}
    sinkhorn_kwargs = {"threshold": 1e-2}
    true_divergence = sinkhorn_divergence.sinkhorn_divergence(
        pointcloud.PointCloud,
        x,
        y,
        a=self._a,
        b=self._b,
        sinkhorn_kwargs=sinkhorn_kwargs,
        **geom_kwargs
    ).divergence

    if shuffle:
      # Now, shuffle the order of both arrays, but
      # still maintain the segment assignments:
      idx_x = jax.random.permutation(
          rngs[2], jnp.arange(x.shape[0] * 2), independent=True
      )
      idx_y = jax.random.permutation(
          rngs[3], jnp.arange(y.shape[0] * 2), independent=True
      )
    else:
      idx_x = jnp.arange(x.shape[0] * 2)
      idx_y = jnp.arange(y.shape[0] * 2)

    # Duplicate arrays:
    x_copied = jnp.concatenate((x, x))[idx_x]
    a_copied = jnp.concatenate((self._a, self._a))[idx_x]
    segment_ids_x = jnp.arange(2).repeat(x.shape[0])[idx_x]

    y_copied = jnp.concatenate((y, y))[idx_y]
    b_copied = jnp.concatenate((self._b, self._b))[idx_y]
    segment_ids_y = jnp.arange(2).repeat(y.shape[0])[idx_y]

    segmented_divergences = sinkhorn_divergence.segment_sinkhorn_divergence(
        x_copied,
        y_copied,
        num_segments=2,
        max_measure_size=19,
        segment_ids_x=segment_ids_x,
        segment_ids_y=segment_ids_y,
        indices_are_sorted=False,
        weights_x=a_copied,
        weights_y=b_copied,
        sinkhorn_kwargs=sinkhorn_kwargs,
        **geom_kwargs
    )

    np.testing.assert_allclose(
        true_divergence.repeat(2), segmented_divergences, rtol=1e-6, atol=1e-6
    )

  def test_segment_sinkhorn_different_segment_sizes(self):
    # Test other array sizes
    x1 = jnp.arange(10)[:, None].repeat(2, axis=1)
    y1 = jnp.arange(11)[:, None].repeat(2, axis=1) + 0.1

    # Should have larger divergence since further apart:
    x2 = jnp.arange(12)[:, None].repeat(2, axis=1)
    y2 = 2 * jnp.arange(13)[:, None].repeat(2, axis=1) + 0.1

    sink_div = jax.jit(
        sinkhorn_divergence.segment_sinkhorn_divergence,
        static_argnames=["num_per_segment_x", "num_per_segment_y"],
    )

    segmented_divergences = sink_div(
        jnp.concatenate((x1, x2)),
        jnp.concatenate((y1, y2)),
        # these 2 arguments are not necessary for jitting:
        # num_segments=2,
        # max_measure_size=15,
        num_per_segment_x=(10, 12),
        num_per_segment_y=(11, 13),
        epsilon=0.01
    )

    assert segmented_divergences.shape[0] == 2
    assert segmented_divergences[1] > segmented_divergences[0]

    true_divergences = jnp.array([
        sinkhorn_divergence.sinkhorn_divergence(
            pointcloud.PointCloud, x, y, epsilon=0.01
        ).divergence for x, y in zip((x1, x2), (y1, y2))
    ])
    np.testing.assert_allclose(
        segmented_divergences, true_divergences, rtol=1e-6, atol=1e-6
    )

  def test_sinkhorn_divergence_segment_custom_padding(self, rng):
    rngs = jax.random.split(rng, 4)
    dim = 3
    b_cost = costs.Bures(dim)

    num_segments = 2

    num_per_segment_x = (5, 2)
    num_per_segment_y = (3, 5)
    ns = num_per_segment_x + num_per_segment_y

    means_and_covs_to_x = jax.vmap(
        costs.mean_and_cov_to_x, in_axes=[0, 0, None]
    )

    def g(rng, n):
      out = gaussian_mixture.GaussianMixture.from_random(
          rng, n_components=n, n_dimensions=dim
      )
      return means_and_covs_to_x(out.loc, out.covariance, dim)

    x1, x2, y1, y2 = (g(rngs[i], ns[i]) for i in range(4))

    true_divergences = jnp.array([
        sinkhorn_divergence.sinkhorn_divergence(
            pointcloud.PointCloud,
            x,
            y,
            sinkhorn_kwargs={
                "lse_mode": True
            },
            epsilon=0.1,
            cost_fn=b_cost
        ).divergence for x, y in zip((x1, x2), (y1, y2))
    ])

    x = jnp.vstack((x1, x2))
    y = jnp.vstack((y1, y2))

    segmented_divergences = sinkhorn_divergence.segment_sinkhorn_divergence(
        x,
        y,
        num_segments=num_segments,
        max_measure_size=5,
        num_per_segment_x=num_per_segment_x,
        num_per_segment_y=num_per_segment_y,
        sinkhorn_kwargs={"lse_mode": True},
        epsilon=0.1,
        cost_fn=b_cost
    )

    np.testing.assert_allclose(
        segmented_divergences, true_divergences, rtol=1e-6, atol=1e-6
    )

  # yapf: disable
  @pytest.mark.fast.with_args(
      "sinkhorn_kwargs,epsilon", [
          ({"anderson": acceleration.AndersonAcceleration(memory=3)}, 1e-2),
          ({"anderson": acceleration.AndersonAcceleration(memory=6)}, None),
          ({"momentum": acceleration.Momentum(start=20)}, 1e-3),
          ({"momentum": acceleration.Momentum(start=30)}, None),
          ({"momentum": acceleration.Momentum(value=1.05)}, 1e-3),
          ({"momentum": acceleration.Momentum(value=1.01)}, None),
      ],
      only_fast=[0, -1],
  )
  # yapf: enable
  def test_euclidean_momentum_params(
      self, sinkhorn_kwargs: Dict[str, Any], epsilon: Optional[float]
  ):
    # check if sinkhorn divergence sinkhorn_kwargs parameters used for
    # momentum/Anderson are properly overriden for the symmetric (x,x) and
    # (y,y) parts.
    rngs = jax.random.split(self.rng, 2)
    threshold = 3.2e-3
    cloud_a = jax.random.uniform(rngs[0], (self._num_points[0], self._dim))
    cloud_b = jax.random.uniform(rngs[1], (self._num_points[1], self._dim))
    sinkhorn_kwargs["threshold"] = threshold

    div = sinkhorn_divergence.sinkhorn_divergence(
        pointcloud.PointCloud,
        cloud_a,
        cloud_b,
        epsilon=epsilon,
        a=self._a,
        b=self._b,
        sinkhorn_kwargs=sinkhorn_kwargs,
    )
    assert div.divergence > 0.0
    assert threshold > div.errors[0][-1]
    assert threshold > div.errors[1][-1]
    assert threshold > div.errors[2][-1]


class TestSinkhornDivergenceGrad:

  @pytest.fixture(autouse=True)
  def initialize(self, rng: jax.random.PRNGKeyArray):
    self._dim = 3
    self._num_points = 13, 12
    self.rng, *rngs = jax.random.split(rng, 3)
    a = jax.random.uniform(rngs[0], (self._num_points[0],))
    b = jax.random.uniform(rngs[1], (self._num_points[1],))
    self._a = a / jnp.sum(a)
    self._b = b / jnp.sum(b)

  def test_gradient_generic_point_cloud_wrapper(self):
    rngs = jax.random.split(self.rng, 3)
    x = jax.random.uniform(rngs[0], (self._num_points[0], self._dim))
    y = jax.random.uniform(rngs[1], (self._num_points[1], self._dim))

    def loss_fn(cloud_a: jnp.ndarray, cloud_b: jnp.ndarray) -> float:
      div = sinkhorn_divergence.sinkhorn_divergence(
          pointcloud.PointCloud,
          cloud_a,
          cloud_b,
          epsilon=1.0,
          a=self._a,
          b=self._b,
          sinkhorn_kwargs={"threshold": 0.05},
      )
      return div.divergence

    delta = jax.random.normal(rngs[2], x.shape)
    eps = 1e-3  # perturbation magnitude

    # first calculation of gradient
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    loss_value, grad_loss = loss_and_grad(x, y)
    custom_grad = jnp.sum(delta * grad_loss)

    assert not jnp.isnan(loss_value)
    np.testing.assert_array_equal(grad_loss.shape, x.shape)
    np.testing.assert_array_equal(jnp.isnan(grad_loss), False)

    # second calculation of gradient
    loss_delta_plus = loss_fn(x + eps * delta, y)
    loss_delta_minus = loss_fn(x - eps * delta, y)
    finite_diff_grad = (loss_delta_plus - loss_delta_minus) / (2 * eps)

    np.testing.assert_allclose(
        custom_grad, finite_diff_grad, rtol=1e-02, atol=1e-02
    )
