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
from typing import Iterator

import jax
import jax.numpy as jnp
import optax
import pytest

from ott.neural.models.models import NeuralVectorField
from ott.neural.solvers.flows import OffsetUniformSampler, UniformSampler
from ott.neural.solvers.genot import GENOT
from ott.solvers.linear import sinkhorn
from ott.solvers.quadratic import gromov_wasserstein


class TestGENOT:
  #TODO: add tests for unbalancedness

  @pytest.mark.parametrize("k_noise_per_x", [1, 2])
  def test_genot_linear_unconditional(
      self, genot_data_loader_linear: Iterator, k_noise_per_x: int
  ):
    source_lin, source_quad, target_lin, target_quad, condition = next(
        genot_data_loader_linear
    )
    source_dim = source_lin.shape[1]
    target_dim = target_lin.shape[1]
    condition_dim = 0

    neural_vf = NeuralVectorField(
        output_dim=target_dim,
        condition_dim=condition_dim,
        latent_embed_dim=5,
    )
    ot_solver = sinkhorn.Sinkhorn()
    time_sampler = UniformSampler()
    optimizer = optax.adam(learning_rate=1e-3)
    genot = GENOT(
        neural_vf,
        input_dim=source_dim,
        output_dim=target_dim,
        cond_dim=condition_dim,
        iterations=3,
        valid_freq=2,
        ot_solver=ot_solver,
        time_sampler=time_sampler,
        optimizer=optimizer,
        k_samples_per_x=k_noise_per_x,
    )
    genot(genot_data_loader_linear, genot_data_loader_linear)

    source_lin, source_quad, target_lin, target_quad, condition = next(
        genot_data_loader_linear
    )
    result_forward = genot.transport(
        source_lin, condition=condition, forward=True
    )
    assert isinstance(result_forward, jax.Array)
    assert jnp.sum(jnp.isnan(result_forward)) == 0

  @pytest.mark.parametrize("k_noise_per_x", [1, 2])
  def test_genot_quad_unconditional(
      self, genot_data_loader_quad: Iterator, k_noise_per_x: int
  ):
    source_lin, source_quad, target_lin, target_quad, condition = next(
        genot_data_loader_quad
    )
    source_dim = source_quad.shape[1]
    target_dim = target_quad.shape[1]
    condition_dim = 0
    neural_vf = NeuralVectorField(
        output_dim=target_dim,
        condition_dim=condition_dim,
        latent_embed_dim=5,
    )
    ot_solver = gromov_wasserstein.GromovWasserstein(epsilon=1e-2)
    time_sampler = OffsetUniformSampler(1e-3)
    optimizer = optax.adam(learning_rate=1e-3)
    genot = GENOT(
        neural_vf,
        input_dim=source_dim,
        output_dim=target_dim,
        cond_dim=condition_dim,
        epsilon=None,
        iterations=3,
        valid_freq=2,
        ot_solver=ot_solver,
        time_sampler=time_sampler,
        optimizer=optimizer,
        k_samples_per_x=k_noise_per_x,
    )
    genot(genot_data_loader_quad, genot_data_loader_quad)

    result_forward = genot.transport(
        source_quad, condition=condition, forward=True
    )
    assert isinstance(result_forward, jax.Array)
    assert jnp.sum(jnp.isnan(result_forward)) == 0

  @pytest.mark.parametrize("k_noise_per_x", [1, 2])
  def test_genot_fused_unconditional(
      self, genot_data_loader_fused: Iterator, k_noise_per_x: int
  ):
    source_lin, source_quad, target_lin, target_quad, condition = next(
        genot_data_loader_fused
    )
    source_dim = source_lin.shape[1] + source_quad.shape[1]
    target_dim = target_lin.shape[1] + target_quad.shape[1]
    condition_dim = 0
    neural_vf = NeuralVectorField(
        output_dim=target_dim,
        condition_dim=condition_dim,
        latent_embed_dim=5,
    )
    ot_solver = gromov_wasserstein.GromovWasserstein(epsilon=1e-2)
    time_sampler = UniformSampler()
    optimizer = optax.adam(learning_rate=1e-3)
    genot = GENOT(
        neural_vf,
        input_dim=source_dim,
        output_dim=target_dim,
        cond_dim=condition_dim,
        epsilon=None,
        iterations=3,
        valid_freq=2,
        ot_solver=ot_solver,
        time_sampler=time_sampler,
        optimizer=optimizer,
        fused_penalty=0.5,
        k_samples_per_x=k_noise_per_x,
    )
    genot(genot_data_loader_fused, genot_data_loader_fused)

    result_forward = genot.transport(
        source_quad, condition=condition, forward=True
    )
    assert isinstance(result_forward, jax.Array)
    assert jnp.sum(jnp.isnan(result_forward)) == 0

  @pytest.mark.parametrize("k_noise_per_x", [1, 2])
  def test_genot_linear_conditional(
      self, genot_data_loader_linear_conditional: Iterator, k_noise_per_x: int
  ):
    source_lin, source_quad, target_lin, target_quad, condition = next(
        genot_data_loader_linear_conditional
    )
    source_dim = source_lin.shape[1]
    target_dim = target_lin.shape[1]
    condition_dim = condition.shape[1]

    neural_vf = NeuralVectorField(
        output_dim=target_dim,
        condition_dim=source_dim + condition_dim,
        latent_embed_dim=5,
    )
    ot_solver = sinkhorn.Sinkhorn()
    time_sampler = UniformSampler()
    optimizer = optax.adam(learning_rate=1e-3)
    genot = GENOT(
        neural_vf,
        input_dim=source_dim,
        output_dim=target_dim,
        cond_dim=condition_dim,
        iterations=3,
        valid_freq=2,
        ot_solver=ot_solver,
        time_sampler=time_sampler,
        optimizer=optimizer,
        k_samples_per_x=k_noise_per_x,
    )
    genot(
        genot_data_loader_linear_conditional,
        genot_data_loader_linear_conditional
    )

    source_lin, source_quad, target_lin, target_quad, condition = next(
        genot_data_loader_linear_conditional
    )
    result_forward = genot.transport(
        source_lin, condition=condition, forward=True
    )
    assert isinstance(result_forward, jax.Array)
    assert jnp.sum(jnp.isnan(result_forward)) == 0

  @pytest.mark.parametrize("k_noise_per_x", [1, 2])
  def test_genot_quad_conditional(
      self, genot_data_loader_quad: Iterator, k_noise_per_x: int
  ):
    source_lin, source_quad, target_lin, target_quad, condition = next(
        genot_data_loader_quad
    )
    source_dim = source_quad.shape[1]
    target_dim = target_quad.shape[1]
    condition_dim = condition.shape[1]
    neural_vf = NeuralVectorField(
        output_dim=target_dim,
        condition_dim=condition_dim,
        latent_embed_dim=5,
    )
    ot_solver = gromov_wasserstein.GromovWasserstein(epsilon=1e-2)
    time_sampler = UniformSampler()
    optimizer = optax.adam(learning_rate=1e-3)
    genot = GENOT(
        neural_vf,
        input_dim=source_dim,
        output_dim=target_dim,
        cond_dim=condition_dim,
        epsilon=None,
        iterations=3,
        valid_freq=2,
        ot_solver=ot_solver,
        time_sampler=time_sampler,
        optimizer=optimizer,
        k_samples_per_x=k_noise_per_x,
    )
    genot(genot_data_loader_quad, genot_data_loader_quad)

    result_forward = genot.transport(
        source_quad, condition=condition, forward=True
    )
    assert isinstance(result_forward, jax.Array)
    assert jnp.sum(jnp.isnan(result_forward)) == 0

  @pytest.mark.parametrize("k_noise_per_x", [1, 2])
  def test_genot_fused_conditional(
      self, genot_data_loader_fused: Iterator, k_noise_per_x: int
  ):
    source_lin, source_quad, target_lin, target_quad, condition = next(
        genot_data_loader_fused
    )
    source_dim = source_lin.shape[1] + source_quad.shape[1]
    target_dim = target_lin.shape[1] + target_quad.shape[1]
    condition_dim = condition.shape[1]
    neural_vf = NeuralVectorField(
        output_dim=target_dim,
        condition_dim=condition_dim,
        latent_embed_dim=5,
    )
    ot_solver = gromov_wasserstein.GromovWasserstein(epsilon=1e-2)
    time_sampler = UniformSampler()
    optimizer = optax.adam(learning_rate=1e-3)
    genot = GENOT(
        neural_vf,
        input_dim=source_dim,
        output_dim=target_dim,
        cond_dim=condition_dim,
        epsilon=None,
        iterations=3,
        valid_freq=2,
        ot_solver=ot_solver,
        time_sampler=time_sampler,
        optimizer=optimizer,
        fused_penalty=0.5,
        k_samples_per_x=k_noise_per_x,
    )
    genot(genot_data_loader_fused, genot_data_loader_fused)

    result_forward = genot.transport(
        source_quad, condition=condition, forward=True
    )
    assert isinstance(result_forward, jax.Array)
    assert jnp.sum(jnp.isnan(result_forward)) == 0

  @pytest.mark.parametrize("conditional", [False, True])
  def test_genot_linear_learn_rescaling(
      self, conditional: bool, genot_data_loader_linear: Iterator,
      genot_data_loader_linear_conditional: Iterator
  ):
    data_loader = genot_data_loader_linear_conditional if conditional else genot_data_loader_linear

    source_lin, source_quad, target_lin, target_quad, condition = next(
        genot_data_loader_linear
    )
    source_dim = source_lin.shape[1]
    target_dim = target_lin.shape[1]
    condition_dim = condition.shape[1] if conditional else 0

    neural_vf = NeuralVectorField(
        output_dim=target_dim,
        condition_dim=condition_dim,
        latent_embed_dim=5,
    )
    ot_solver = sinkhorn.Sinkhorn()
    time_sampler = UniformSampler()
    optimizer = optax.adam(learning_rate=1e-3)
    genot = GENOT(
        neural_vf,
        input_dim=source_dim,
        output_dim=target_dim,
        cond_dim=condition_dim,
        iterations=3,
        valid_freq=2,
        ot_solver=ot_solver,
        time_sampler=time_sampler,
        optimizer=optimizer,
    )
    genot(data_loader, data_loader)

    source_lin, source_quad, target_lin, target_quad, condition = next(
        genot_data_loader_linear
    )

    result_eta = genot.evaluate_eta(source_lin, condition=condition)
    assert isinstance(result_eta, jax.Array)
    assert jnp.sum(jnp.isnan(result_eta)) == 0

    result_xi = genot.evaluate_xi(target_lin, condition=condition)
    assert isinstance(result_xi, jax.Array)
    assert jnp.sum(jnp.isnan(result_xi)) == 0
