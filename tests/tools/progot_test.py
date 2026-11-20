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
import jax.numpy as jnp
import numpy as np

from ott.problems.linear import linear_problem
from ott.tools import progot
from tests import _utils


class TestProgOT:

  @pytest.mark.parametrize("is_debiased", [False, True])
  def test_is_debiased(self, rng: jax.Array, is_debiased: bool):
    clouds = _utils.random_clouds(rng)
    solver = progot.ProgOT(
        alphas=jnp.array([0.5, 1.0]), is_debiased=is_debiased
    )

    out = solver(clouds.problem())

    assert out.is_debiased == is_debiased

  def test_iterates(self, rng: jax.Array):
    clouds = _utils.random_clouds(rng)
    prob = linear_problem.LinearProblem(clouds.geom())
    alphas = progot.get_alpha_schedule("lin", num_steps=4)

    out = progot.ProgOT(alphas=alphas)(prob)
    xs, _ = out.transport(clouds.x, return_intermediate=True)

    # every step solves the problem between the previous interpolation and `y`
    np.testing.assert_allclose(
        out.outputs.geom.x[1:], xs[:-1], rtol=1e-5, atol=1e-5
    )
