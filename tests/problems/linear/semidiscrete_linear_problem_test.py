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
import jax.random as jr

from ott.geometry import semidiscrete_pointcloud as sdpc
from ott.problems.linear import semidiscrete_linear_problem as sdlp


class TestSemidiscreteLinearProblem:

  @pytest.mark.parametrize("tau_b", [0.5, 0.999, 1.0])
  def unbalanced_not_supported(self, rng: jax.Array, tau_b: float):
    y = jr.normal(rng, (12, 3))
    geom = sdpc.SemidiscretePointCloud(jr.normal, y)

    if tau_b != 1.0:
      with pytest.raises(
          AssertionError,
          match=r"Unbalanced semi-discrete problem is not supported."
      ):
        _ = sdlp.SemidiscreteLinearProblem(geom, tau_b=tau_b)
    else:
      _ = sdlp.SemidiscreteLinearProblem(geom, tau_b=tau_b)

  def test_sample(self):
    pass
