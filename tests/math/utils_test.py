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
import jax
import jax.numpy as jnp
import pytest
from ott.math import utils as mu


@pytest.mark.fast()
class TestNorm:

  def test_norm(self, rng: jax.random.PRNGKeyArray):
    d = 13
    f = lambda x: mu.norm(x - x)
    g = lambda x: jnp.linalg.norm(x - x)
    x = jax.random.uniform(rng, (d,))
    assert jnp.all(jax.grad(f)(x) == 0.0)
    assert jnp.all(jnp.isnan(jax.grad(g)(x)))
