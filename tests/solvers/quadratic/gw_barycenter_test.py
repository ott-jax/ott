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
from typing import Any, Optional, Sequence, Tuple

import pytest

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ott.geometry import pointcloud
from ott.problems.quadratic import gw_barycenter as gwb
from ott.solvers.linear import sinkhorn
from ott.solvers.quadratic import gromov_wasserstein
from ott.solvers.quadratic import gw_barycenter as gwb_solver


class TestGWBarycenter:
  ndim = 3
  ndim_f = 4

  @staticmethod
  def random_pc(
      n: int,
      d: int,
      rng: jax.Array,
      m: Optional[int] = None,
      **kwargs: Any
  ) -> pointcloud.PointCloud:
    rng1, rng2 = jr.split(rng, 2)
    x = jr.normal(rng1, (n, d))
    y = x if m is None else jr.normal(rng2, (m, d))
    return pointcloud.PointCloud(x, y, **kwargs)

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

  # TODO(cuturi) add back KL test when KL cost GW is fixed.
  @pytest.mark.parametrize(
      ("gw_loss", "bar_size", "epsilon"),
      [("sqeucl", 17, None)]  # , ("kl", 22, 1e-2)]
  )
  def test_gw_barycenter(
      self, rng: jax.Array, gw_loss: str, bar_size: int,
      epsilon: Optional[float]
  ):
    tol = 1e-3 if gw_loss == "sqeucl" else 1e-1
    num_per_segment = (13, 15, 21)
    rngs = jr.split(rng, len(num_per_segment))
    pcs = [
        self.random_pc(n, d=self.ndim, rng=rng)
        for n, rng in zip(num_per_segment, rngs)
    ]
    costs = [pc.cost_matrix for pc, n in zip(pcs, num_per_segment)]
    costs, cbs = self.pad_cost_matrices(costs)
    ys = jnp.concatenate([pc.x for pc in pcs])
    bs = jnp.concatenate([jnp.ones(n) / n for n in num_per_segment])
    kwargs = {
        "gw_loss": gw_loss,
        "num_per_segment": num_per_segment,
        "epsilon": epsilon
    }

    bar_prob_pc = gwb.GWBarycenterProblem(y=ys, b=bs, **kwargs)
    bar_prob_cost = gwb.GWBarycenterProblem(
        costs=costs,
        b=cbs,
        **kwargs,
    )
    for bar_prob in [bar_prob_pc, bar_prob_cost]:
      assert not bar_prob.is_fused
      assert bar_prob.ndim_fused is None
      assert bar_prob.num_measures == len(num_per_segment)
      assert bar_prob.max_measure_size == max(num_per_segment)
      assert bar_prob._loss_name == gw_loss
    assert bar_prob_pc.ndim == self.ndim
    assert bar_prob_cost.ndim is None

    linear_solver = sinkhorn.Sinkhorn()
    quadratic_solver = gromov_wasserstein.GromovWasserstein(linear_solver)
    solver = gwb_solver.GromovWassersteinBarycenter(quadratic_solver)
    solver = jax.jit(solver, static_argnames=["bar_size"])

    out_pc = solver(bar_prob_pc, bar_size=bar_size)
    out_cost = solver(bar_prob_cost, bar_size=bar_size)

    assert out_pc.x is None
    assert out_cost.x is None
    assert out_pc.cost.shape == (bar_size, bar_size)
    np.testing.assert_allclose(out_pc.cost, out_cost.cost, rtol=tol, atol=tol)
    np.testing.assert_allclose(out_pc.costs, out_cost.costs, rtol=tol, atol=tol)

    np.testing.assert_allclose(
        out_pc.costs,
        jnp.sum(out_pc.costs_bary * bar_prob_pc.weights, axis=-1),
        rtol=tol,
        atol=tol
    )

  @pytest.mark.fast(
      "jit,fused_penalty,scale_cost", [(False, 1.5, "mean"),
                                       (True, 3.1, "max_cost")],
      only_fast=0
  )
  def test_fgw_barycenter(
      self,
      rng: jax.Array,
      jit: bool,
      fused_penalty: float,
      scale_cost: str,
  ):

    def barycenter(
        y: jnp.ndim, y_fused: jnp.ndarray, num_per_segment: Tuple[int, ...]
    ) -> gwb_solver.GWBarycenterState:
      bar_prob = gwb.GWBarycenterProblem(
          y=y,
          y_fused=y_fused,
          num_per_segment=num_per_segment,
          fused_penalty=fused_penalty,
          scale_cost=scale_cost,
      )
      assert bar_prob.is_fused
      assert bar_prob.fused_penalty == fused_penalty
      assert not bar_prob._y_as_costs
      assert bar_prob.max_measure_size == max(num_per_segment)
      assert bar_prob.num_measures == len(num_per_segment)
      assert bar_prob.ndim == self.ndim
      assert bar_prob.ndim_fused == self.ndim_f

      linear_solver = sinkhorn.Sinkhorn()
      quadratic_solver = gromov_wasserstein.GromovWasserstein(
          linear_solver, epsilon=epsilon, store_inner_errors=True
      )
      solver = gwb_solver.GromovWassersteinBarycenter(
          quadratic_solver, store_inner_errors=True
      )

      x_init = jr.normal(rng_init, (bar_size, self.ndim_f))
      cost_init = pointcloud.PointCloud(x_init).cost_matrix

      return solver(bar_prob, bar_size=bar_size, bar_init=(cost_init, x_init))

    bar_size, epsilon, = 10, 1e-1
    num_per_segment = (7, 12)

    rng_init, rng1, *rngs = jr.split(rng, len(num_per_segment) + 2)
    y = jnp.concatenate([
        self.random_pc(n, d=self.ndim, rng=rng).x
        for n, rng in zip(num_per_segment, rngs)
    ])
    rngs = jr.split(rng1, len(num_per_segment))
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

    weights = jnp.ones(len(num_per_segment)) / len(num_per_segment)
    np.testing.assert_allclose(
        out.costs,
        jnp.sum(out.costs_bary * weights, axis=-1),
        rtol=1e-6,
        atol=1e-6,
    )
