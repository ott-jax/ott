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


class TestMongeGap:

  def test_monge_gap(self, rng: jax.random.PRNGKey):
    """Tests convexity of ICNN."""
    n_samples, n_features = 10, 2

    # define icnn model
    # model = models.ICNN(n_features, dim_hidden=dim_hidden)

    # # initialize model
    rng1, rng2, rng3 = jax.random.split(rng, 3)
    # model.init(rng1, jnp.ones(n_features))["params"]

    # check convexity
    jax.random.normal(rng1, (n_samples, n_features)) * 0.1
    jax.random.normal(rng2, (n_samples, n_features))

    raise NotImplementedError("Add a test.")
