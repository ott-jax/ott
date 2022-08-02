# Copyright 2022 Apple
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
"""Tests for Continuous barycenters."""
import functools
from typing import Any, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ott.core import bar_problems, continuous_barycenter, gw_barycenter, segment
from ott.geometry import costs, pointcloud
from ott.tools.gaussian_mixture import gaussian_mixture

means_and_covs_to_x = jax.vmap(costs.mean_and_cov_to_x, in_axes=[0, 0, None])


def is_positive_semidefinite(c: jnp.ndarray) -> bool:
  w = jnp.linalg.eigvals(c)
  return jnp.all(w >= 0)


class TestBarycenter:
  DIM = 4
  N_POINTS = 113

  @pytest.mark.fast.with_args(
      rank=[-1, 6],
      epsilon=[1e-1, 1e-2],
      jit=[True, False],
      init_random=[True, False],
      only_fast={
          "rank": -1,
          "epsilon": 1e-1,
          "jit": True,
          "init_random": False
      },
  )
  def test_euclidean_barycenter(
      self, rng: jnp.ndarray, rank: int, epsilon: float, jit: bool,
      init_random: bool
  ):
    rngs = jax.random.split(rng, 20)
    # Sample 2 point clouds, each of size 113, the first around [0,1]^4,
    # Second around [2,3]^4.
    y1 = jax.random.uniform(rngs[0], (self.N_POINTS, self.DIM))
    y2 = jax.random.uniform(rngs[1], (self.N_POINTS, self.DIM)) + 2
    # Merge them
    y = jnp.concatenate((y1, y2))

    # Define segments
    num_per_segment = jnp.array([33, 29, 24, 27, 27, 31, 30, 25])
    # Set weights for each segment that sum to 1.
    b = []
    for i in range(num_per_segment.shape[0]):
      c = jax.random.uniform(rngs[i], (num_per_segment[i],))
      b.append(c / jnp.sum(c))
    b = jnp.concatenate(b, axis=0)
    # Set a barycenter problem with 8 measures, of irregular sizes.
    bar_prob = bar_problems.BarycenterProblem(
        y,
        b,
        epsilon=epsilon,
        num_segments=8,
        max_measure_size=35,
        num_per_segment=num_per_segment
    )
    assert bar_prob.num_measures == 8
    assert bar_prob.max_measure_size == 35
    assert bar_prob.ndim == self.DIM

    # Define solver
    threshold = 1e-3
    solver = continuous_barycenter.WassersteinBarycenter(
        rank=rank, threshold=threshold, jit=jit
    )

    # Set barycenter size to 31.
    bar_size = 31

    # We consider either a random initialization, with points chosen
    # in [0,1]^4, or the default (init_random is False) where the
    # initialization consists in selecting randomly points in the y's.
    if init_random:
      # choose points randomly in area relevant to the problem.
      x_init = 3 * jax.random.uniform(rngs[-1], (bar_size, self.DIM))
      out = solver(bar_prob, bar_size=bar_size, x_init=x_init)
    else:
      out = solver(bar_prob, bar_size=bar_size)

    # Check shape is as expected
    np.testing.assert_array_equal(out.x.shape, (bar_size, self.DIM))

    # Check convergence by looking at cost evolution.
    c = out.costs[out.costs > -1]
    assert jnp.isclose(c[-2], c[-1], rtol=threshold)

    # Check barycenter has all points roughly in [1,2]^4.
    # (this is because sampled points were equally set in either [0,1]^4
    # or [2,3]^4)
    assert jnp.all(out.x.ravel() < 2.3)
    assert jnp.all(out.x.ravel() > .7)

  @pytest.mark.parametrize("segment_before", [False, True])
  def test_barycenter_jit(self, rng: jnp.ndarray, segment_before: bool):

    @functools.partial(jax.jit, static_argnums=(2, 3))
    def barycenter(
        y: jnp.ndarray, b: jnp.ndarray, segment_before: bool,
        num_per_segment: int
    ) -> continuous_barycenter.BarycenterState:
      if segment_before:
        y, b = segment.segment_point_cloud(
            x=y, a=b, num_per_segment=num_per_segment
        )
        bar_prob = bar_problems.BarycenterProblem(y, b, epsilon=1e-1)
      else:
        bar_prob = bar_problems.BarycenterProblem(
            y, b, epsilon=1e-1, num_per_segment=num_per_segment
        )
      solver = continuous_barycenter.WassersteinBarycenter(threshold=threshold)
      return solver(bar_prob)

    rngs = jax.random.split(rng, 20)
    # Sample 2 point clouds, each of size 113, the first around [0,1]^4,
    # Second around [2,3]^4.
    y1 = jax.random.uniform(rngs[0], (self.N_POINTS, self.DIM))
    y2 = jax.random.uniform(rngs[1], (self.N_POINTS, self.DIM)) + 2
    # Merge them
    y = jnp.concatenate((y1, y2))

    # Define segments
    num_per_segment = (33, 29, 24, 27, 27, 31, 30, 25)
    # Set weights for each segment that sum to 1.
    b = []
    for rng, n in zip(rngs, num_per_segment):
      c = jax.random.uniform(rng, (n,))
      b.append(c / jnp.sum(c))
    b = jnp.concatenate(b, axis=0)

    threshold = 1e-3
    out = barycenter(
        y, b, segment_before=segment_before, num_per_segment=num_per_segment
    )
    # Check convergence by looking at cost evolution.
    c = out.costs[out.costs > -1]
    assert jnp.isclose(c[-2], c[-1], rtol=threshold)

    # Check barycenter has all points roughly in [1,2]^4.
    # (this is because sampled points were equally set in either [0,1]^4
    # or [2,3]^4)
    assert jnp.all(out.x.ravel() < 2.3)
    assert jnp.all(out.x.ravel() > .7)

  @pytest.mark.fast.with_args(
      lse_mode=[False, True],
      epsilon=[1e-1, 5e-1],
      jit=[False, True],
      only_fast={
          "lse_mode": True,
          "epsilon": 1e-1,
          "jit": False
      }
  )
  def test_bures_barycenter(
      self, rng: jnp.ndarray, lse_mode: bool, epsilon: float, jit: bool
  ):
    num_measures = 2
    num_components = 2
    dimension = 2
    bar_size = 2
    barycentric_weights = jnp.asarray([0.5, 0.5])
    bures_cost = costs.Bures(dimension=dimension)

    means1 = jnp.array([[-1., 1.], [-1., -1.]])
    means2 = jnp.array([[1., 1.], [1., -1.]])
    sigma = 0.01
    covs1 = sigma * jnp.asarray([
        jnp.eye(dimension) for _ in range(num_components)
    ])
    covs2 = sigma * jnp.asarray([
        jnp.eye(dimension) for _ in range(num_components)
    ])

    y1 = means_and_covs_to_x(means1, covs1, dimension)
    y2 = means_and_covs_to_x(means2, covs2, dimension)

    b1 = b2 = jnp.ones(num_components) / num_components

    y = jnp.concatenate((y1, y2))
    jnp.concatenate((b1, b2))

    gmm_generator = gaussian_mixture.GaussianMixture.from_random(
        rng, n_components=bar_size, n_dimensions=dimension
    )

    x_init_means = gmm_generator.loc
    x_init_covs = gmm_generator.covariance

    x_init = means_and_covs_to_x(x_init_means, x_init_covs, dimension)

    seg_y, seg_b = segment.segment_point_cloud(
        y,
        num_segments=num_measures,
        max_measure_size=num_components,
        num_per_segment=(num_components, num_components),
        padding_vector=bures_cost.padder(y.shape[1]),
    )
    bar_p = bar_problems.BarycenterProblem(
        seg_y,
        seg_b,
        weights=barycentric_weights,
        cost_fn=bures_cost,
        epsilon=epsilon
    )
    assert bar_p.num_measures == seg_y.shape[0]
    assert bar_p.max_measure_size == seg_y.shape[1]
    assert bar_p.ndim == seg_y.shape[2]

    solver = continuous_barycenter.WassersteinBarycenter(
        lse_mode=lse_mode, jit=jit
    )

    out = solver(bar_p, bar_size=bar_size, x_init=x_init)
    barycenter = out.x

    means_bary, covs_bary = costs.x_to_means_and_covs(barycenter, dimension)

    assert jnp.logical_or(
        jnp.allclose(
            means_bary,
            jnp.array([[0., 1.], [0., -1.]]),
            rtol=1e-02,
            atol=1e-02
        ),
        jnp.allclose(
            means_bary,
            jnp.array([[0., -1.], [0., 1.]]),
            rtol=1e-02,
            atol=1e-02
        )
    )

    np.testing.assert_allclose(
        covs_bary,
        jnp.array([sigma * jnp.eye(dimension) for i in range(bar_size)]),
        rtol=1e-05,
        atol=1e-05
    )

  @pytest.mark.fast.with_args(
      alpha=[50., 1.],
      epsilon=[1e-2, 1e-1],
      jit=[False, True],
      dim=[4, 10],
      only_fast={
          "alpha": 50,
          "epsilon": 1e-1,
          "jit": False,
          "dim": 4
      }
  )
  def test_bures_barycenter_different_number_of_components(
      self, rng: jnp.ndarray, dim: int, alpha: float, epsilon: float, jit: bool
  ):
    n_components = jnp.array([3, 4])  # the number of components of the GMMs
    num_measures = n_components.size
    bar_size = 5  # the size of the barycenter

    # Create an instance of the Bures cost class.
    b_cost = costs.Bures(dimension=dim)

    # keys for random number generation
    keys = jax.random.split(rng, num=4)

    # test for non-uniform barycentric weights
    barycentric_weights = jax.random.dirichlet(
        keys[0], alpha=jnp.ones(num_measures) * alpha
    )

    ridges = jnp.array([jnp.ones(dim), 5 * jnp.ones(dim)])
    stdev_means = 0.1 * jnp.mean(ridges, axis=1)
    stdev_covs = jax.random.uniform(
        keys[1], shape=(num_measures,), minval=0., maxval=10.
    )

    seeds = jax.random.randint(
        keys[2], shape=(num_measures,), minval=0, maxval=100
    )

    gmm_generators = [
        gaussian_mixture.GaussianMixture.from_random(
            jax.random.PRNGKey(seeds[i]),
            n_components=n_components[i],
            n_dimensions=dim,
            stdev_cov=stdev_covs[i],
            stdev_mean=stdev_means[i],
            ridge=ridges[i]
        ) for i in range(num_measures)
    ]

    means_covs = [(gmm_generators[i].loc, gmm_generators[i].covariance)
                  for i in range(num_measures)]

    # positions of mass of the measures
    ys = jnp.vstack(
        means_and_covs_to_x(means_covs[i][0], means_covs[i][1], dim)
        for i in range(num_measures)
    )

    # mass distribution of the measures
    weights = [
        gmm_generators[i].component_weight_ob.probs()
        for i in range(num_measures)
    ]
    bs = jnp.hstack(jnp.array(weights[i]) for i in range(num_measures))

    # random initialization of the barycenter
    gmm_generator = gaussian_mixture.GaussianMixture.from_random(
        keys[3], n_components=bar_size, n_dimensions=dim
    )
    x_init_means = gmm_generator.loc
    x_init_covs = gmm_generator.covariance
    x_init = means_and_covs_to_x(x_init_means, x_init_covs, dim)

    # test second interface for segmentation
    seg_ids = jnp.repeat(jnp.arange(num_measures), n_components)
    bar_p = bar_problems.BarycenterProblem(
        y=ys,
        b=bs,
        weights=barycentric_weights,
        cost_fn=b_cost,
        epsilon=epsilon,
        num_segments=num_measures,
        max_measure_size=4,
        segment_ids=seg_ids,
    )
    assert bar_p.max_measure_size == 4
    assert bar_p.num_measures == num_measures
    assert bar_p.ndim == ys.shape[-1]

    solver = continuous_barycenter.WassersteinBarycenter(lse_mode=True, jit=jit)

    # Compute the barycenter.
    out = solver(bar_p, bar_size=bar_size, x_init=x_init)
    barycenter = out.x

    means_bary, covs_bary = costs.x_to_means_and_covs(barycenter, dim)

    # check the values of the means
    # Because of the way the means are generated with gaussian_mixture.
    # GaussianMixture.from_random, for the selected ridges, the elements of the
    # mean of the barycenter will be in the range (0.9, 6.5) almost surely. Due
    # to the fact that the barycentric weights tested are not extreme (as in
    # not [0, 1] or [1, 0]) this test should always pass.

    np.testing.assert_array_equal(means_bary < 6.5, True)
    np.testing.assert_array_equal(means_bary > 0.9, True)

    # check that covariances of barycenter are all psd
    np.testing.assert_array_equal(
        jax.vmap(is_positive_semidefinite, in_axes=0, out_axes=0)(covs_bary),
        True
    )


class TestGWBarycenter:
  NDIM = 3
  NDIM_F = 4

  @staticmethod
  def random_pc(
      n: int,
      d: int,
      rng: jnp.ndarray,
      m: Optional[int] = None,
      **kwargs: Any
  ) -> pointcloud.PointCloud:
    key1, key2 = jax.random.split(rng, 2)
    x = jax.random.normal(key1, (n, d))
    y = x if m is None else jax.random.normal(key2, (m, d))
    return pointcloud.PointCloud(x, y, batch_size=None, **kwargs)

  @staticmethod
  def pad_cost_matrices(
      costs: Sequence[jnp.ndarray],
      shape: Optional[Tuple[int, int]] = None
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if shape is None:
      shape = jnp.asarray([arr.shape for arr in costs]).max()
      shape = (shape, shape)
    else:
      assert shape[0] == shape[1], shape

    cs, weights = [], []
    for cost in costs:
      r, c = cost.shape
      cs.append(jnp.zeros(shape).at[:r, :c].set(cost))
      w = jnp.ones(r) / r
      weights.append(jnp.concatenate([w, jnp.zeros(shape[0] - r)]))
    return jnp.stack(cs), jnp.stack(weights)

  @pytest.mark.parametrize(
      "gw_loss,bar_size,epsilon", [("sqeucl", 17, None), ("kl", 22, 1e-2)]
  )
  def test_gw_barycenter(
      self, rng: jnp.ndarray, gw_loss: str, bar_size: int,
      epsilon: Optional[float]
  ):
    tol = 1e-2 if gw_loss == 'kl' else 1e-3
    num_per_segment = (13, 15, 21)
    rngs = jax.random.split(rng, len(num_per_segment))
    pcs = [
        self.random_pc(n, d=self.NDIM, rng=rng, epsilon=epsilon)
        for n, rng in zip(num_per_segment, rngs)
    ]
    costs = [pc._compute_cost_matrix() for pc, n in zip(pcs, num_per_segment)]
    costs, cbs = self.pad_cost_matrices(costs)
    ys = jnp.concatenate([pc.x for pc in pcs])
    bs = jnp.concatenate([jnp.ones(n) / n for n in num_per_segment])
    kwargs = {
        "gw_loss": gw_loss,
        "num_per_segment": num_per_segment,
        "epsilon": epsilon
    }

    problem_pc = bar_problems.GWBarycenterProblem(y=ys, b=bs, **kwargs)
    problem_cost = bar_problems.GWBarycenterProblem(
        costs=costs,
        b=cbs,
        **kwargs,
    )
    solver = gw_barycenter.GromovWassersteinBarycenter(jit=True)

    out_pc = solver(problem_pc, bar_size=bar_size)
    out_cost = solver(problem_cost, bar_size=bar_size)

    np.testing.assert_allclose(out_pc.cost, out_cost.cost, rtol=tol, atol=tol)
    np.testing.assert_allclose(out_pc.costs, out_cost.costs, rtol=tol, atol=tol)

  @pytest.mark.parametrize("y_as_costs", [False, True])
  def test_fgw_costs(self, y_as_costs: bool):
    pass
