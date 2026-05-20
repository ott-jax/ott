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

from flax import nnx

from ott.neural.networks import icnn


@pytest.mark.fast()
class TestICNN:

  def test_icnn_convexity(self, rng: jax.Array):
    """Tests convexity of ICNN."""
    n_samples, n_features = 10, 2
    dim_hidden = (64, 64)

    model = icnn.ICNN(dim_hidden, input_dim=n_features, rngs=nnx.Rngs(0))

    rng1, rng2 = jax.random.split(rng, 2)
    x = jax.random.normal(rng1, (n_samples, n_features)) * 0.1
    y = jax.random.normal(rng2, (n_samples, n_features))

    out_x = model(x)
    out_y = model(y)

    out = []
    for t in jnp.linspace(0, 1):
      out_xy = model(t * x + (1 - t) * y)
      out.append((t * out_x + (1 - t) * out_y) - out_xy)

    np.testing.assert_array_equal(np.array(out) >= 0.0, True)

  def test_icnn_hessian(self, rng: jax.Array):
    """Tests if Hessian of ICNN is positive-semidefinite."""
    n_features = 2
    dim_hidden = (64, 64)
    model = icnn.ICNN(dim_hidden, input_dim=n_features, rngs=nnx.Rngs(0))

    rng1, rng2 = jax.random.split(rng)

    data = jax.random.normal(rng2, (n_features,))

    # Compute Hessian of scalar output
    hessian = jax.hessian(lambda x: model(x[None]).squeeze())(data)

    w = jnp.linalg.eigvalsh((hessian + hessian.T) / 2.0)
    np.testing.assert_array_equal(w >= 0, True)

  def test_icnn_vector_output(self, rng: jax.Array):
    """Tests ICNN with vector output (each component convex)."""
    n_samples, n_features, output_dim = 10, 3, 4
    dim_hidden = (32, 32)

    model = icnn.ICNN(
        dim_hidden,
        input_dim=n_features,
        output_dim=output_dim,
        rngs=nnx.Rngs(0)
    )

    x = jax.random.normal(rng, (n_samples, n_features))
    out = model(x)
    assert out.shape == (n_samples, output_dim)

  def test_icnn_wx_inject_frequency(self, rng: jax.Array):
    """Tests ICNN with wx_inject as frequency."""
    n_features = 4
    dim_hidden = (32, 32, 32, 32)

    model = icnn.ICNN(
        dim_hidden, input_dim=n_features, wx_inject=2, rngs=nnx.Rngs(0)
    )
    x = jax.random.normal(rng, (5, n_features))
    out = model(x)
    assert out.shape == (5,)

  def test_icnn_pos_def_potentials(self, rng: jax.Array):
    """Tests ICNN with PosDefPotentials enabled."""
    n_features = 3
    dim_hidden = (32, 32)

    model = icnn.ICNN(
        dim_hidden, input_dim=n_features, pos_def_rank=2, rngs=nnx.Rngs(0)
    )
    x = jax.random.normal(rng, (5, n_features))
    out = model(x)
    assert out.shape == (5,)

  @pytest.mark.parametrize("mode", ["softmax", "sinkhorn"])
  def test_icnn_stochastic_weights(self, rng: jax.Array, mode: str):
    """Tests ICNN with softmax/sinkhorn weight normalization."""
    n_samples, n_features = 10, 4
    dim_hidden = (32, 32)

    kwargs = {
        "use_softmax": mode == "softmax",
        "use_sinkhorn": mode == "sinkhorn",
    }
    model = icnn.ICNN(
        dim_hidden, input_dim=n_features, rngs=nnx.Rngs(0), **kwargs
    )

    rng1, rng2 = jax.random.split(rng)
    x = jax.random.normal(rng1, (n_samples, n_features)) * 0.1
    y = jax.random.normal(rng2, (n_samples, n_features))

    out_x = model(x)
    out_y = model(y)
    assert out_x.shape == (n_samples,)

    # Verify convexity
    out = []
    for t in jnp.linspace(0, 1):
      out_xy = model(t * x + (1 - t) * y)
      out.append((t * out_x + (1 - t) * out_y) - out_xy)

    np.testing.assert_array_equal(np.array(out) >= -1e-5, True)


@pytest.mark.fast()
class TestKeyNet:

  def test_keynet_output_shape(self, rng: jax.Array):
    """Tests KeyNet output dimensions."""
    n_samples, n_features = 10, 4
    dim_hidden = (32, 32)

    model = icnn.KeyNet(dim_hidden, input_dim=n_features, rngs=nnx.Rngs(0))
    x = jax.random.normal(rng, (n_samples, n_features))

    # gradient returns vectors
    grad_out = model.gradient(x)
    assert grad_out.shape == (n_samples, n_features)

    # __call__ returns scalars (dot product)
    scalar_out = model(x)
    assert scalar_out.shape == (n_samples,)

  def test_keynet_resnet(self, rng: jax.Array):
    """Tests KeyNet in resnet mode."""
    n_features = 4
    dim_hidden = (32, 32)

    model = icnn.KeyNet(
        dim_hidden, input_dim=n_features, resnet=True, rngs=nnx.Rngs(0)
    )
    x = jax.random.normal(rng, (5, n_features))

    grad_out = model.gradient(x)
    assert grad_out.shape == (5, n_features)

  def test_keynet_custom_output_dim(self, rng: jax.Array):
    """Tests KeyNet with explicit output_dim != input_dim."""
    n_features = 4
    output_dim = 8
    dim_hidden = (32, 32)

    model = icnn.KeyNet(
        dim_hidden,
        input_dim=n_features,
        output_dim=output_dim,
        rngs=nnx.Rngs(0)
    )
    x = jax.random.normal(rng, (5, n_features))

    grad_out = model.gradient(x)
    assert grad_out.shape == (5, output_dim)

  def test_keynet_unbatched(self, rng: jax.Array):
    """Tests KeyNet with single (unbatched) input."""
    n_features = 4
    dim_hidden = (32, 32)

    model = icnn.KeyNet(dim_hidden, input_dim=n_features, rngs=nnx.Rngs(0))
    x = jax.random.normal(rng, (n_features,))

    grad_out = model.gradient(x)
    assert grad_out.shape == (n_features,)

    scalar_out = model(x)
    assert scalar_out.shape == ()
