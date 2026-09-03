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

import pytest

import jax

from flax import linen as nn

from ott.initializers.neural import meta_initializer as meta_init
from ott.solvers.linear import sinkhorn
from tests.initializers import _problems


class MetaMLP(nn.Module):
  potential_size: int
  num_hidden_units: int = 512
  num_hidden_layers: int = 3

  @nn.compact
  def __call__(self, z: jax.Array) -> jax.Array:
    for _ in range(self.num_hidden_layers):
      z = nn.relu(nn.Dense(self.num_hidden_units)(z))
    return nn.Dense(self.potential_size)(z)


@pytest.mark.fast()
class TestMetaInitializer:

  @pytest.mark.parametrize("lse_mode", [True, False])
  def test_meta_initializer(self, rng: jax.Array, lse_mode: bool):
    """Tests Meta initializer"""
    n, m = 32, 30
    epsilon = 1e-2

    prob = _problems.create_ot_problem(rng, n, m, epsilon=epsilon, batch_size=3)

    # run sinkhorn
    solver = sinkhorn.Sinkhorn(lse_mode=lse_mode, max_iterations=3000)
    sink_out = jax.jit(solver)(prob)

    # overfit the initializer to the problem.
    meta_model = MetaMLP(n)
    meta_initializer = meta_init.MetaInitializer(prob.geom, meta_model)
    for _ in range(50):
      _, _, meta_initializer.state = meta_initializer.update(
          meta_initializer.state, a=prob.a, b=prob.b
      )

    solver = sinkhorn.Sinkhorn(
        initializer=meta_initializer, lse_mode=lse_mode, max_iterations=3000
    )
    meta_out = jax.jit(solver)(prob)

    # check initializer is better
    if lse_mode:
      assert sink_out.converged
      assert meta_out.converged
      assert sink_out.n_iters > meta_out.n_iters
    else:
      assert sink_out.n_iters >= meta_out.n_iters
