# coding=utf-8
# Lint as: python3
"""Test gradient of Sinkhorn applied to grid w.r.t. location."""
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from ott.core import sinkhorn
from ott.geometry import grid


class SinkhornGradGridTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)

  @parameterized.parameters([True], [False])
  def test_diff_sinkhorn_x_grid_x_perturbation(self, lse_mode):
    """Test gradient w.r.t. probability weights."""
    eps = 1e-3  # perturbation magnitude
    keys = jax.random.split(self.rng, 6)
    x = (jnp.array([.0, 1.0]),
         jnp.array([.3, .4, .7]),
         jnp.array([1.0, 1.3, 2.4, 3.7]))
    grid_size = tuple([xs.shape[0] for xs in x])
    a = jax.random.uniform(keys[0], grid_size) + 1.0
    b = jax.random.uniform(keys[1], grid_size) + 1.0
    a = a.ravel() / jnp.sum(a)
    b = b.ravel() / jnp.sum(b)

    def reg_ot(x):
      geom = grid.Grid(x=x, epsilon=1.0)
      return sinkhorn.sinkhorn(
          geom, a=a, b=b, threshold=0.1, lse_mode=lse_mode).reg_ot_cost

    reg_ot_and_grad = jax.value_and_grad(reg_ot)
    _, grad_reg_ot = reg_ot_and_grad(x)
    delta = [jax.random.uniform(keys[i], (g,)) for i, g in enumerate(grid_size)]

    x_p_delta = [(xs + eps * delt) for xs, delt in zip(x, delta)]
    x_m_delta = [(xs - eps * delt) for xs, delt in zip(x, delta)]

    # center perturbation
    reg_ot_delta_plus = reg_ot(x_p_delta)
    reg_ot_delta_minus = reg_ot(x_m_delta)
    delta_dot_grad = jnp.sum(jnp.array(
        [jnp.sum(delt * gr, axis=None) for delt, gr in zip(delta, grad_reg_ot)]
        ))
    np.testing.assert_allclose(
        delta_dot_grad, (reg_ot_delta_plus - reg_ot_delta_minus) / (2 * eps),
        rtol=1e-03,
        atol=1e-02)


if __name__ == '__main__':
  absltest.main()
