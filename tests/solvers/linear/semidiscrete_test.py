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
import jax.experimental.sparse as jesp
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import optax

from ott.geometry import costs
from ott.geometry import semidiscrete_pointcloud as sdpc
from ott.problems.linear import linear_problem
from ott.problems.linear import semidiscrete_linear_problem as sdlp
from ott.solvers import linear
from ott.solvers.linear import semidiscrete, sinkhorn


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

  @pytest.mark.fast()
  @pytest.mark.parametrize("n", [20, 31])
  @pytest.mark.parametrize("epsilon", [0.0, 1e-3, 1e-2, 1e-1, None])
  def test_custom_gradient_semidiscrete_loss(
      self, rng: jax.Array, n: int, epsilon: Optional[float]
  ):

    def semidiscrete_loss(
        g: jax.Array, prob: linear_problem.LinearProblem
    ) -> jax.Array:
      f, _ = prob._c_transform(g, axis=1)
      return -jnp.mean(f) - jnp.dot(g, prob.b)

    rng_prob, rng_potential, rng_sample = jr.split(rng, 3)
    m, d = 17, 5
    prob = _random_problem(rng_prob, m=m, d=d, epsilon=epsilon)

    g = jr.normal(rng_potential, (m,))
    sampled_prob = prob.sample(rng_sample, n)

    gt_fn = jax.jit(jax.value_and_grad(semidiscrete_loss))
    # has custom VJP
    pred_fn = jax.jit(jax.value_and_grad(semidiscrete._semidiscrete_loss))

    gt_val, gt_grad_g = gt_fn(g, sampled_prob)
    # for low epsilon, where `b=0`, this can be NaN
    gt_grad_g = jnp.where(jnp.isnan(gt_grad_g), 0.0, gt_grad_g)
    prev_val, pred_grad_g = pred_fn(g, sampled_prob)

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
        num_iterations=10,
        batch_size=32,
        optimizer=optax.sgd(1e-3),
        error_eval_every=5,
        error_num_repeats=3,
    )
    out = jax.jit(solver)(rng_solver, prob)
    sampled_out = out.sample(rng_sample, 17)

    assert out.g.dtype == dtype
    assert out.losses.dtype == dtype
    assert out.errors.dtype == dtype
    assert sampled_out.matrix.dtype == dtype
    assert sampled_out.ot_prob.geom.dtype == dtype
    assert sampled_out.ot_prob.geom.cost_matrix.dtype == dtype

  def test_callback(self, capsys, rng: jax.Array):

    def print_state(state: semidiscrete.SemidiscreteState) -> None:
      print(state.it)  # noqa: T201

    rng_prob, rng_solver = jr.split(rng, 2)
    prob = _random_problem(rng_prob, m=12, d=2)
    num_iters = 10

    solver = semidiscrete.SemidiscreteSolver(
        num_iterations=num_iters,
        batch_size=5,
        error_eval_every=10,
        error_num_repeats=1,
        optimizer=optax.sgd(1e-1),
        callback=print_state,
    )

    _ = jax.jit(solver)(rng_solver, prob)

    expected = "\n".join(str(i) for i in range(1, num_iters + 1)) + "\n"
    actual = capsys.readouterr()
    assert actual.out == expected
    assert actual.err == ""

  @pytest.mark.parametrize("epsilon", [0.0, 1e-2, None])
  def test_epsilon(self, rng: jax.Array, epsilon: Optional[float]):
    rng_prob, rng_solver, rng_sample = jr.split(rng, 3)

    prob = _random_problem(rng_prob, m=15, d=4, epsilon=epsilon)
    if epsilon == 0.0:
      assert not prob.geom.is_entropy_regularized
    else:
      assert prob.geom.is_entropy_regularized

    solver = semidiscrete.SemidiscreteSolver(
        num_iterations=10,
        batch_size=5,
        error_eval_every=5,
        error_num_repeats=3,
        optimizer=optax.sgd(1e-1),
    )

    out = jax.jit(solver)(rng_solver, prob)
    out_sampled = out.sample(rng_sample, 1)

    if out.prob.geom.is_entropy_regularized:
      assert isinstance(out_sampled, sinkhorn.SinkhornOutput)
    else:
      assert isinstance(out_sampled, semidiscrete.HardAssignmentOutput)

  @pytest.mark.parametrize("epsilon", [1e-1, None])
  def test_match_with_finiteOT(self, rng: jax.Array, epsilon: Optional[float]):
    rng_solver, rng_sample, rng_b, rng_y = jr.split(rng, 4)
    m, d = 8, 2
    b = jr.uniform(rng_b, (m,)) + 1.  # balanced distribution helps converge
    b /= b.sum()
    y = jr.normal(rng_y, (m, d))
    geom = sdpc.SemidiscretePointCloud(
        jr.normal, y, epsilon=epsilon, cost_fn=costs.NegDotProduct()
    )
    sd_prob = sdlp.SemidiscreteLinearProblem(geom, b=b)
    num_iterations = 1024

    schedule = optax.linear_schedule(
        init_value=1,
        transition_begin=num_iterations // 4,
        transition_steps=num_iterations // 2,
        end_value=5e-3,
    )

    solver = semidiscrete.SemidiscreteSolver(
        num_iterations=num_iterations,
        batch_size=256,
        optimizer=optax.sgd(learning_rate=schedule),
        potential_ema=.95  # testing
    )

    out_sd = jax.jit(solver)(rng_solver, sd_prob)

    n = 2_048
    finite_geom = geom.sample(
        rng_sample, num_samples=n, epsilon=sd_prob.epsilon
    )
    out_ot = linear.solve(finite_geom)
    assert out_ot.converged
    g_ot = out_ot.g - jnp.mean(out_ot.g)
    g_sd = out_sd.g - jnp.mean(out_sd.g)
    np.testing.assert_allclose(g_ot, g_sd, rtol=1e-2, atol=1e-1)

  @pytest.mark.parametrize("epsilon", [0.0, 1e-2, None])
  def test_initial_potential(self, rng: jax.Array, epsilon: Optional[float]):
    rng_prob, rng_solver = jr.split(rng, 2)
    prob = _random_problem(rng_prob, m=32, d=3, epsilon=epsilon)

    solver = semidiscrete.SemidiscreteSolver(
        num_iterations=10,
        batch_size=64,
        error_eval_every=10,
        error_num_repeats=5,
        optimizer=optax.adam(5e-2, b1=0.5, b2=0.9),
    )

    out = jax.jit(solver)(rng_solver, prob)
    out_init = jax.jit(solver)(rng_solver, prob, out.g)

    np.testing.assert_array_less(out_init.losses, out.losses)

  @pytest.mark.fast()
  def test_solver_wrapper(self, rng: jax.Array):
    rng_prob, rng_solver = jr.split(rng, 2)
    geom = _random_problem(rng_prob, m=32, d=3, epsilon=0.0).geom
    out = linear.solve_semidiscrete(
        geom, num_iterations=5, batch_size=7, optimizer=optax.sgd(1.0), rng=rng
    )
    np.testing.assert_array_equal(jnp.isfinite(out.losses), True)

  @pytest.mark.parametrize(("n", "epsilon"), [(17, 0.0), (20, 1e-3),
                                              (35, None)])
  def test_output(self, rng: jax.Array, n: int, epsilon: Optional[float]):
    m, d = 32, 3
    rng_prob, rng_solver, rng_sample = jr.split(rng, 3)
    prob = _random_problem(rng_prob, m=m, d=d, epsilon=epsilon)

    solver = semidiscrete.SemidiscreteSolver(
        num_iterations=100,
        batch_size=16,
        error_eval_every=10,
        optimizer=optax.adam(0.01, b1=0.5, b2=0.99),
    )

    out = jax.jit(solver)(rng_solver, prob)
    for i in range(10, 15):
      rng_sample, rng_sample_it = jr.split(rng_sample, 2)
      out_sampled = out.sample(rng_sample_it, n + i)

      assert out_sampled.ot_prob.geom.shape == (n + i, m)

      if out.prob.geom.is_entropy_regularized:
        out_sampled = out_sampled.set_cost(
            out_sampled.ot_prob, lse_mode=True, use_danskin=True
        )

        assert isinstance(out_sampled, sinkhorn.SinkhornOutput)
        assert jnp.all(jnp.isfinite(out_sampled.reg_ot_cost))
        assert jnp.isclose(
            out_sampled.transport_mass, 1.0, rtol=1e-4, atol=1e-4
        )
        assert jnp.all(jnp.isfinite(out_sampled.matrix))
      else:
        expected_primal_cost = jnp.sum(
            out_sampled.matrix.todense() * out_sampled.ot_prob.geom.cost_matrix
        )

        assert isinstance(out_sampled, semidiscrete.HardAssignmentOutput)
        assert isinstance(out_sampled.matrix, jesp.BCOO)
        np.testing.assert_allclose(
            out_sampled.primal_cost, expected_primal_cost, rtol=1e-4, atol=1e-4
        )
        assert out_sampled.matrix.nse == n + i
        np.testing.assert_allclose(
            out_sampled.matrix.sum(), 1.0, rtol=1e-5, atol=1e-5
        )

  @pytest.mark.parametrize("epsilon_dp", [None, 0.1])
  @pytest.mark.parametrize("epsilon_prob", [0.0, 1e-2])
  def test_sd_dual_potentials(
      self, rng: jax.Array, epsilon_prob: float, epsilon_dp: Optional[float]
  ):
    rng_prob, rng_solver, rng_sample = jr.split(rng, 3)

    prob = _random_problem(rng_prob, m=15, d=4, epsilon=epsilon_prob)
    x = prob.geom.sample(rng_sample, 6).x
    y = prob.geom.y

    solver = semidiscrete.SemidiscreteSolver(
        num_iterations=10,
        batch_size=5,
        error_eval_every=5,
        error_num_repeats=3,
        optimizer=optax.sgd(1e-1),
    )

    out = jax.jit(solver)(rng_solver, prob)
    # if epsilon_dp=None, epsilon_prob is used
    dp = out.to_dual_potentials(epsilon_dp)

    y_hat = dp.transport(x, forward=True)
    assert y_hat.shape == x.shape
    np.testing.assert_array_equal(jnp.isfinite(y_hat), True)

    with pytest.raises(AssertionError, match=r"The `g` potential is not"):
      _ = dp.transport(y, forward=False)

  @pytest.mark.parametrize("sample_eps", [0.0, 0.1, None])
  @pytest.mark.parametrize("solve_eps", [0.0, 0.1, None])
  def test_sample_different_epsilon(
      self, rng: jax.Array, solve_eps: float, sample_eps: float
  ):
    rng_prob, rng_solver, rng_sample = jr.split(rng, 3)
    n, m = 6, 16

    prob = _random_problem(rng_prob, m=m, d=4, epsilon=solve_eps)

    solver = semidiscrete.SemidiscreteSolver(
        num_iterations=10,
        batch_size=4,
        error_eval_every=5,
        error_num_repeats=3,
        optimizer=optax.sgd(1e-1),
    )
    out = jax.jit(solver)(rng_solver, prob)

    out_sampled = out.sample(rng_sample, n, epsilon=sample_eps)
    is_entreg_solve = (solve_eps is None or solve_eps > 0.0)
    is_entreg_sample = (is_entreg_solve and sample_eps is None
                       ) or (sample_eps is not None and sample_eps > 0.0)

    if is_entreg_sample:
      assert isinstance(out_sampled, sinkhorn.SinkhornOutput)
    else:
      assert isinstance(out_sampled, semidiscrete.HardAssignmentOutput)

    assert out_sampled.ot_prob.geom.shape == (n, m)
    np.testing.assert_allclose(
        out_sampled.matrix.sum(), 1.0, rtol=1e-5, atol=1e-5
    )
