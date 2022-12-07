"""Tests for Fused Gromov-Wasserstein barycenter."""
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ott.geometry import pointcloud
from ott.problems.quadratic import gw_barycenter as gwb
from ott.solvers.quadratic import gw_barycenter as gwb_solver


class FGWBarycenterTest:

  @pytest.mark.fast(
      "jit,fused_penalty,scale_cost", [(False, 1.5, "mean"),
                                       (True, 3.1, "max_cost")],
      only_fast=0
  )
  def test_fgw_barycenter(
      self,
      rng: jnp.ndarray,
      jit: bool,
      fused_penalty: float,
      scale_cost: str,
  ):

    def barycenter(
        y: jnp.ndim, y_fused: jnp.ndarray, num_per_segment: Tuple[int, ...]
    ) -> gwb_solver.GWBarycenterState:
      prob = gwb.GWBarycenterProblem(
          y=y,
          y_fused=y_fused,
          num_per_segment=num_per_segment,
          fused_penalty=fused_penalty,
          scale_cost=scale_cost,
      )
      assert prob.is_fused
      assert prob.fused_penalty == fused_penalty
      assert not prob._y_as_costs
      assert prob.max_measure_size == max(num_per_segment)
      assert prob.num_measures == len(num_per_segment)
      assert prob.ndim == self.ndim
      assert prob.ndim_fused == self.ndim_f

      solver = gwb_solver.GromovWassersteinBarycenter(
          store_inner_errors=True, epsilon=epsilon
      )

      x_init = jax.random.normal(rng, (bar_size, self.ndim_f))
      cost_init = pointcloud.PointCloud(x_init).cost_matrix

      return solver(prob, bar_size=bar_size, bar_init=(cost_init, x_init))

    bar_size, epsilon, = 10, 1e-1
    num_per_segment = (7, 12)

    key1, *rngs = jax.random.split(rng, len(num_per_segment) + 1)
    y = jnp.concatenate([
        self.random_pc(n, d=self.ndim, rng=rng).x
        for n, rng in zip(num_per_segment, rngs)
    ])
    rngs = jax.random.split(key1, len(num_per_segment))
    y_fused = jnp.concatenate([
        self.random_pc(n, d=self.ndim_f, rng=rng).x
        for n, rng in zip(num_per_segment, rngs)
    ])

    fn = jax.jit(barycenter, static_argnums=2) if jit else barycenter
    out = fn(y, y_fused, num_per_segment)

    assert out.cost.shape == (bar_size, bar_size)
    assert out.x.shape == (bar_size, self.ndim_f)
    np.testing.assert_array_equal(jnp.isfinite(out.cost), True)
    np.testing.assert_array_equal(jnp.isfinite(out.x), True)
    np.testing.assert_array_equal(jnp.isfinite(out.costs), True)
    np.testing.assert_array_equal(jnp.isfinite(out.errors), True)
