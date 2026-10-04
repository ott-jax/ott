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

from ott.problems.linear import barycenter_problem
from ott.solvers.linear import continuous_barycenter, sinkhorn


def _toy_two_atom_problem():
  """Two identical measures on [-1, +1] with skewed weights [0.9, 0.1]."""
  y = jnp.array([
      [[-1.0], [1.0]],
      [[-1.0], [1.0]],
  ])
  b = jnp.array([
      [0.9, 0.1],
      [0.9, 0.1],
  ])
  return barycenter_problem.FreeBarycenterProblem(y=y, b=b)


class TestLearnBarycenterWeights:

  @pytest.mark.fast()
  def test_default_keeps_uniform_a(self, rng: jax.Array):
    """Without learn_a the barycenter weights stay uniform."""
    bar_prob = _toy_two_atom_problem()
    solver = continuous_barycenter.FreeWassersteinBarycenter(
        sinkhorn.Sinkhorn(),
        max_iterations=8,
    )
    out = solver(bar_prob, bar_size=2, rng=rng)
    np.testing.assert_allclose(out.a, jnp.array([0.5, 0.5]), atol=1e-6)

  @pytest.mark.fast()
  def test_learn_a_balanced_stays_uniform(self, rng: jax.Array):
    """learn_a=True with tau_a=1 (balanced) cannot change weights."""
    bar_prob = _toy_two_atom_problem()
    solver = continuous_barycenter.FreeWassersteinBarycenter(
        sinkhorn.Sinkhorn(),
        learn_a=True,
        tau_a=1.0,
        max_iterations=8,
    )
    out = solver(bar_prob, bar_size=2, rng=rng)
    np.testing.assert_allclose(out.a, jnp.array([0.5, 0.5]), atol=1e-6)

  @pytest.mark.fast()
  def test_learn_a_unbalanced_recovers_skewed_weights(self, rng: jax.Array):
    """Unbalanced + learn_a recovers the skewed input weights."""
    bar_prob = _toy_two_atom_problem()
    solver = continuous_barycenter.FreeWassersteinBarycenter(
        sinkhorn.Sinkhorn(),
        learn_a=True,
        tau_a=0.9,
        max_iterations=25,
    )
    out = solver(bar_prob, bar_size=2, rng=rng)

    got = jnp.sort(out.a)[::-1]
    target = jnp.array([0.9, 0.1])

    np.testing.assert_allclose(got, target, atol=0.15)
    assert jnp.isfinite(out.a).all()
    assert jnp.all(out.a > 0.0)
    np.testing.assert_allclose(jnp.sum(out.a), 1.0, atol=1e-6)
    # Weights should have moved meaningfully away from uniform.
    assert jnp.max(jnp.abs(out.a - 0.5)) > 1e-2

  @pytest.mark.fast()
  def test_invalid_tau_a(self):
    with pytest.raises(ValueError, match="tau_a"):
      continuous_barycenter.FreeWassersteinBarycenter(
          sinkhorn.Sinkhorn(),
          tau_a=0.0,
      )
