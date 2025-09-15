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
from typing import Any, Optional

import pytest

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import optax

from ott.geometry import semidiscrete_pointcloud as sdpc
from ott.problems.linear import linear_problem
from ott.problems.linear import semidiscrete_linear_problem as sdlp
from ott.solvers.linear import semidiscrete


def _random_problem(
    rng: jax.Array,
    *,
    m: int,
    d: int,
    dtype: Optional[jnp.dtype] = None,
    **kwargs: Any
) -> sdlp.SemidiscreteLinearProblem:
  rng_b, rng_y = jr.split(rng, 2)
  b = jr.uniform(rng_b, (m,), dtype=dtype)
  b = b.at[np.array([0, 2])].set(0.0)
  b /= b.sum()
  y = jr.normal(rng_y, (m, d), dtype=dtype)
  geom = sdpc.SemidiscretePointCloud(jr.normal, y, **kwargs)
  return sdlp.SemidiscreteLinearProblem(geom, b=b)


class TestSemidiscreteSolver:

  @pytest.mark.parametrize("n", [20, 31])
  @pytest.mark.parametrize("epsilon", [0.0, 1e-3, 1e-2, 1e-1, None])
  def test_c_transform_gradient(
      self, rng: jax.Array, n: int, epsilon: Optional[float]
  ):

    def semidiscrete_loss(
        g: jax.Array, prob: linear_problem.LinearProblem, is_soft: bool
    ) -> jax.Array:
      if is_soft:
        f, _ = semidiscrete._soft_c_transform(g, prob)
      else:
        f, _ = semidiscrete._hard_c_transform(g, prob)
      return -jnp.mean(f) - jnp.dot(g, prob.b)

    rng_prob, rng_potential, rng_sample = jr.split(rng, 3)
    m, d = 17, 5
    prob = _random_problem(rng_prob, m=m, d=d, epsilon=epsilon)

    g = jr.normal(rng_potential, (m,))
    sampled_prob = prob.sample(rng_sample, n)
    is_soft = prob.geom.is_entropy_regularized

    gt_fn = jax.jit(
        jax.value_and_grad(semidiscrete_loss), static_argnames=["is_soft"]
    )
    pred_fn = jax.jit(
        jax.value_and_grad(semidiscrete._semidiscrete_loss),
        static_argnames=["is_soft"]
    )

    gt_val, gt_grad_g = gt_fn(g, sampled_prob, is_soft)
    # for low epsilon, where `b=0`, this can be NaN
    gt_grad_g = jnp.where(jnp.isnan(gt_grad_g), 0.0, gt_grad_g)
    prev_val, pred_grad_g = pred_fn(g, sampled_prob, is_soft)

    np.testing.assert_allclose(prev_val, gt_val, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(pred_grad_g, gt_grad_g, rtol=1e-4, atol=1e-4)

  @pytest.mark.parametrize(("dtype", "epsilon"), [(jnp.float16, 0.0),
                                                  (jnp.bfloat16, 0.5),
                                                  (jnp.float32, None)])
  def test_dtype(
      self, rng: jax.Array, dtype: jnp.dtype, epsilon: Optional[float]
  ):
    m, d = 22, 3
    rng_prob, rng_solver, rng_sample = jr.split(rng, 3)
    prob = _random_problem(rng_prob, m=m, d=d, epsilon=epsilon, dtype=dtype)

    solver = semidiscrete.SemidiscreteSolver(
        min_iterations=1,
        max_iterations=10,
        inner_iterations=5,
        batch_size=32,
        optimizer=optax.sgd(1e-3),
    )
    out = solver(rng_solver, prob)
    sampled_out = out.sample(rng_sample, 17)

    assert out.g.dtype == dtype
    assert out.losses.dtype == dtype
    assert out.errors.dtype == dtype
    assert sampled_out.matrix.dtype == dtype
    assert sampled_out.ot_prob.geom.dtype == dtype
    assert sampled_out.ot_prob.geom.cost_matrix.dtype == dtype

  def test_callback(self):
    pass

  def test_convergence(self):
    pass

  def test_optimizer(self):
    pass

  def test_epsilon(self):
    pass

  def test_soft_output(self):
    pass

  def test_hard_output(self):
    pass
