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
import functools
from typing import Iterator, Optional, Union, Literal

import pytest

import jax.numpy as jnp

import optax

from ott.geometry import costs
from ott.neural.flows.genot import GENOT
from ott.neural.flows.models import VelocityField
from ott.neural.flows.samplers import sample_uniformly
from ott.neural.models.models import RescalingMLP
from ott.solvers.linear import sinkhorn
from ott.solvers.quadratic import gromov_wasserstein


class TestGENOT:

  @pytest.mark.parameterize("scale_cost", ["mean", 2.0])
  @pytest.mark.parametrize("k_samples_per_x", [1, 2])
  @pytest.mark.parametrize("solver_latent_to_data", [None, "sinkhorn"])
  def test_genot_linear_unconditional(
      self, genot_data_loader_linear: Iterator, scale_cost: Union[float, Literal["mean"]], k_samples_per_x: int,
      solver_latent_to_data: Optional[str]
  ):
    solver_latent_to_data = (
        None if solver_latent_to_data is None else sinkhorn.Sinkhorn()
    )
    batch = next(genot_data_loader_linear)
    source_lin, target_lin, source_condition = batch["source_lin"], batch[
        "target_lin"], batch["source_conditions"]

    source_dim = source_lin.shape[1]
    target_dim = target_lin.shape[1]
    condition_dim = 0

    neural_vf = VelocityField(
        output_dim=target_dim,
        condition_dim=source_dim + condition_dim,
        latent_embed_dim=5,
    )
    ot_solver = sinkhorn.Sinkhorn()
    time_sampler = sample_uniformly
    optimizer = optax.adam(learning_rate=1e-3)
    genot = GENOT(
        neural_vf,
        input_dim=source_dim,
        output_dim=target_dim,
        cond_dim=condition_dim,
        iterations=3,
        valid_freq=2,
        ot_solver=ot_solver,
        epsilon=0.1,
        cost_fn=costs.SqEuclidean(),
        scale_cost=1.0,
        optimizer=optimizer,
        time_sampler=time_sampler,
        k_samples_per_x=k_samples_per_x,
        solver_latent_to_data=solver_latent_to_data,
    )
    genot(genot_data_loader_linear, genot_data_loader_linear)

    batch = next(genot_data_loader_linear)
    source_lin, target_lin, source_condition = batch["source_lin"], batch[
        "target_lin"], batch["source_conditions"]

    result_forward = genot.transport(
        source_lin, condition=source_condition, forward=True
    )
    assert isinstance(result_forward, jnp.ndarray)
    assert jnp.sum(jnp.isnan(result_forward)) == 0

  @pytest.mark.parametrize("k_samples_per_x", [1, 2])
  @pytest.mark.parametrize("solver_latent_to_data", [None, "sinkhorn"])
  def test_genot_quad_unconditional(
      self, genot_data_loader_quad: Iterator, k_samples_per_x: int,
      solver_latent_to_data: Optional[str]
  ):
    None if solver_latent_to_data is None else sinkhorn.Sinkhorn()
    batch = next(genot_data_loader_quad)
    source_quad, target_quad, source_condition = batch["source_quad"], batch[
        "target_quad"], batch["source_conditions"]

    source_dim = source_quad.shape[1]
    target_dim = target_quad.shape[1]
    condition_dim = 0
    neural_vf = VelocityField(
        output_dim=target_dim,
        condition_dim=source_dim + condition_dim,
        latent_embed_dim=5,
    )
    ot_solver = gromov_wasserstein.GromovWasserstein(epsilon=1e-2)
    time_sampler = functools.partial(sample_uniformly, offset=1e-2)
    optimizer = optax.adam(learning_rate=1e-3)
    genot = GENOT(
        neural_vf,
        input_dim=source_dim,
        output_dim=target_dim,
        cond_dim=condition_dim,
        iterations=3,
        valid_freq=2,
        ot_solver=ot_solver,
        epsilon=None,
        cost_fn=costs.SqEuclidean(),
        scale_cost=1.0,
        optimizer=optimizer,
        time_sampler=time_sampler,
        k_samples_per_x=k_samples_per_x,
    )
    genot(genot_data_loader_quad, genot_data_loader_quad)

    result_forward = genot.transport(
        source_quad, condition=source_condition, forward=True
    )
    assert isinstance(result_forward, jnp.ndarray)
    assert jnp.sum(jnp.isnan(result_forward)) == 0

  @pytest.mark.parametrize("k_samples_per_x", [1, 2])
  @pytest.mark.parametrize("solver_latent_to_data", [None, "sinkhorn"])
  def test_genot_fused_unconditional(
      self, genot_data_loader_fused: Iterator, k_samples_per_x: int,
      solver_latent_to_data: Optional[str]
  ):
    None if solver_latent_to_data is None else sinkhorn.Sinkhorn()
    batch = next(genot_data_loader_fused)
    source_lin, source_quad, target_lin, target_quad, source_condition = batch[
        "source_lin"], batch["source_quad"], batch["target_lin"], batch[
            "target_quad"], batch["source_conditions"]

    source_dim = source_lin.shape[1] + source_quad.shape[1]
    target_dim = target_lin.shape[1] + target_quad.shape[1]
    condition_dim = 0
    neural_vf = VelocityField(
        output_dim=target_dim,
        condition_dim=source_dim + condition_dim,
        latent_embed_dim=5,
    )
    ot_solver = gromov_wasserstein.GromovWasserstein(epsilon=1e-2)
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
        cost_fn=costs.SqEuclidean(),
        scale_cost=1.0,
        optimizer=optimizer,
        fused_penalty=0.5,
        k_samples_per_x=k_samples_per_x,
    )
    genot(genot_data_loader_fused, genot_data_loader_fused)

    result_forward = genot.transport(
        jnp.concatenate((source_lin, source_quad), axis=1),
        condition=source_condition,
        forward=True
    )
    assert isinstance(result_forward, jnp.ndarray)
    assert jnp.sum(jnp.isnan(result_forward)) == 0

  @pytest.mark.parametrize("k_samples_per_x", [1, 2])
  @pytest.mark.parametrize("solver_latent_to_data", [None, "sinkhorn"])
  def test_genot_linear_conditional(
      self, genot_data_loader_linear_conditional: Iterator,
      k_samples_per_x: int, solver_latent_to_data: Optional[str]
  ):
    None if solver_latent_to_data is None else sinkhorn.Sinkhorn()
    batch = next(genot_data_loader_linear_conditional)
    source_lin, target_lin, source_condition = batch["source_lin"], batch[
        "target_lin"], batch["source_conditions"]
    source_dim = source_lin.shape[1]
    target_dim = target_lin.shape[1]
    condition_dim = source_condition.shape[1]

    neural_vf = VelocityField(
        output_dim=target_dim,
        condition_dim=source_dim + condition_dim,
        latent_embed_dim=5,
    )
    ot_solver = sinkhorn.Sinkhorn()
    time_sampler = sample_uniformly
    optimizer = optax.adam(learning_rate=1e-3)
    genot = GENOT(
        neural_vf,
        input_dim=source_dim,
        output_dim=target_dim,
        cond_dim=condition_dim,
        iterations=3,
        valid_freq=2,
        ot_solver=ot_solver,
        epsilon=0.1,
        cost_fn=costs.SqEuclidean(),
        scale_cost=1.0,
        optimizer=optimizer,
        time_sampler=time_sampler,
        k_samples_per_x=k_samples_per_x,
    )
    genot(
        genot_data_loader_linear_conditional,
        genot_data_loader_linear_conditional
    )
    result_forward = genot.transport(
        source_lin, condition=source_condition, forward=True
    )
    assert isinstance(result_forward, jnp.ndarray)
    assert jnp.sum(jnp.isnan(result_forward)) == 0

  @pytest.mark.parametrize("k_samples_per_x", [1, 2])
  @pytest.mark.parametrize("solver_latent_to_data", [None, "sinkhorn"])
  def test_genot_quad_conditional(
      self, genot_data_loader_quad_conditional: Iterator, k_samples_per_x: int,
      solver_latent_to_data: Optional[str]
  ):
    None if solver_latent_to_data is None else sinkhorn.Sinkhorn()
    batch = next(genot_data_loader_quad_conditional)
    source_quad, target_quad, source_condition = batch["source_quad"], batch[
        "target_quad"], batch["source_conditions"]

    source_dim = source_quad.shape[1]
    target_dim = target_quad.shape[1]
    condition_dim = source_condition.shape[1]
    neural_vf = VelocityField(
        output_dim=target_dim,
        condition_dim=source_dim + condition_dim,
        latent_embed_dim=5,
    )
    ot_solver = gromov_wasserstein.GromovWasserstein(epsilon=1e-2)
    time_sampler = sample_uniformly
    optimizer = optax.adam(learning_rate=1e-3)
    genot = GENOT(
        neural_vf,
        input_dim=source_dim,
        output_dim=target_dim,
        cond_dim=condition_dim,
        iterations=3,
        valid_freq=2,
        ot_solver=ot_solver,
        epsilon=None,
        cost_fn=costs.SqEuclidean(),
        scale_cost=1.0,
        optimizer=optimizer,
        time_sampler=time_sampler,
        k_samples_per_x=k_samples_per_x,
    )
    genot(
        genot_data_loader_quad_conditional, genot_data_loader_quad_conditional
    )

    result_forward = genot.transport(
        source_quad, condition=source_condition, forward=True
    )
    assert isinstance(result_forward, jnp.ndarray)
    assert jnp.sum(jnp.isnan(result_forward)) == 0

  @pytest.mark.parametrize("k_samples_per_x", [1, 2])
  @pytest.mark.parametrize("solver_latent_to_data", [None, "sinkhorn"])
  def test_genot_fused_conditional(
      self, genot_data_loader_fused_conditional: Iterator, k_samples_per_x: int,
      solver_latent_to_data: Optional[str]
  ):
    None if solver_latent_to_data is None else sinkhorn.Sinkhorn()
    batch = next(genot_data_loader_fused_conditional)
    source_lin, source_quad, target_lin, target_quad, source_condition = batch[
        "source_lin"], batch["source_quad"], batch["target_lin"], batch[
            "target_quad"], batch["source_conditions"]
    source_dim = source_lin.shape[1] + source_quad.shape[1]
    target_dim = target_lin.shape[1] + target_quad.shape[1]
    condition_dim = source_condition.shape[1]
    neural_vf = VelocityField(
        output_dim=target_dim,
        condition_dim=source_dim + condition_dim,
        latent_embed_dim=5,
    )
    ot_solver = gromov_wasserstein.GromovWasserstein(epsilon=1e-2)
    time_sampler = sample_uniformly
    optimizer = optax.adam(learning_rate=1e-3)
    genot = GENOT(
        neural_vf,
        input_dim=source_dim,
        output_dim=target_dim,
        cond_dim=condition_dim,
        iterations=3,
        valid_freq=2,
        ot_solver=ot_solver,
        epsilon=None,
        cost_fn=costs.SqEuclidean(),
        scale_cost=1.0,
        optimizer=optimizer,
        time_sampler=time_sampler,
        k_samples_per_x=k_samples_per_x,
    )
    genot(
        genot_data_loader_fused_conditional, genot_data_loader_fused_conditional
    )

    result_forward = genot.transport(
        jnp.concatenate((source_lin, source_quad), axis=1),
        condition=source_condition,
        forward=True
    )
    assert isinstance(result_forward, jnp.ndarray)
    assert jnp.sum(jnp.isnan(result_forward)) == 0

  @pytest.mark.parametrize("conditional", [False, True])
  @pytest.mark.parametrize("solver_latent_to_data", [None, "sinkhorn"])
  def test_genot_linear_learn_rescaling(
      self, conditional: bool, genot_data_loader_linear: Iterator,
      solver_latent_to_data: Optional[str],
      genot_data_loader_linear_conditional: Iterator
  ):
    None if solver_latent_to_data is None else sinkhorn.Sinkhorn()
    data_loader = (
        genot_data_loader_linear_conditional
        if conditional else genot_data_loader_linear
    )

    batch = next(data_loader)
    source_lin, target_lin, source_condition = batch["source_lin"], batch[
        "target_lin"], batch["source_conditions"]

    source_dim = source_lin.shape[1]
    target_dim = target_lin.shape[1]
    condition_dim = source_condition.shape[1] if conditional else 0

    neural_vf = VelocityField(
        output_dim=target_dim,
        condition_dim=source_dim + condition_dim,
        latent_embed_dim=5,
    )
    ot_solver = sinkhorn.Sinkhorn()
    time_sampler = sample_uniformly
    optimizer = optax.adam(learning_rate=1e-3)
    tau_a = 0.9
    tau_b = 0.2
    rescaling_a = RescalingMLP(hidden_dim=4, condition_dim=condition_dim)
    rescaling_b = RescalingMLP(hidden_dim=4, condition_dim=condition_dim)
    genot = GENOT(
        neural_vf,
        input_dim=source_dim,
        output_dim=target_dim,
        cond_dim=condition_dim,
        iterations=3,
        valid_freq=2,
        ot_solver=ot_solver,
        epsilon=0.1,
        cost_fn=costs.SqEuclidean(),
        scale_cost=1.0,
        optimizer=optimizer,
        time_sampler=time_sampler,
        tau_a=tau_a,
        tau_b=tau_b,
        rescaling_a=rescaling_a,
        rescaling_b=rescaling_b,
    )

    genot(data_loader, data_loader)

    result_eta = genot.evaluate_eta(source_lin, condition=source_condition)
    assert isinstance(result_eta, jnp.ndarray)
    assert jnp.sum(jnp.isnan(result_eta)) == 0

    result_xi = genot.evaluate_xi(target_lin, condition=source_condition)
    assert isinstance(result_xi, jnp.ndarray)
    assert jnp.sum(jnp.isnan(result_xi)) == 0
