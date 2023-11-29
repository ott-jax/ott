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
from typing import Iterator, Type

import pytest

import jax.numpy as jnp

import optax

from ott.neural.models.models import NeuralVectorField, RescalingMLP
from ott.neural.solvers.flows import (
    BaseFlow,
    BrownianNoiseFlow,
    ConstantNoiseFlow,
    OffsetUniformSampler,
    UniformSampler,
)
from ott.neural.solvers.otfm import OTFlowMatching
from ott.solvers.linear import sinkhorn


class TestOTFlowMatching:

  @pytest.mark.parametrize(
      "flow",
      [ConstantNoiseFlow(0.0),
       ConstantNoiseFlow(1.0),
       BrownianNoiseFlow(0.2)]
  )
  def test_flow_matching(self, data_loader_gaussian, flow: Type[BaseFlow]):
    neural_vf = NeuralVectorField(
        output_dim=2,
        condition_dim=0,
        latent_embed_dim=5,
    )
    ot_solver = sinkhorn.Sinkhorn()
    time_sampler = UniformSampler()
    optimizer = optax.adam(learning_rate=1e-3)
    fm = OTFlowMatching(
        neural_vf,
        input_dim=2,
        cond_dim=0,
        iterations=3,
        valid_freq=2,
        ot_solver=ot_solver,
        flow=flow,
        time_sampler=time_sampler,
        optimizer=optimizer
    )
    fm(data_loader_gaussian, data_loader_gaussian)

    batch = next(data_loader_gaussian)
    result_forward = fm.transport(
        batch["source_lin"], condition=batch["source_conditions"], forward=True
    )
    assert isinstance(result_forward, jnp.ndarray)
    assert jnp.sum(jnp.isnan(result_forward)) == 0

    result_backward = fm.transport(
        batch["target_lin"],
        condition=batch["source_conditions"],
        forward=False
    )
    assert isinstance(result_backward, jnp.ndarray)
    assert jnp.sum(jnp.isnan(result_backward)) == 0

  @pytest.mark.parametrize(
      "flow",
      [ConstantNoiseFlow(0.0),
       ConstantNoiseFlow(1.0),
       BrownianNoiseFlow(0.2)]
  )
  def test_flow_matching_with_conditions(
      self, data_loader_gaussian_with_conditions, flow: Type[BaseFlow]
  ):
    neural_vf = NeuralVectorField(
        output_dim=2,
        condition_dim=1,
        latent_embed_dim=5,
    )
    ot_solver = sinkhorn.Sinkhorn()
    time_sampler = OffsetUniformSampler(1e-6)
    optimizer = optax.adam(learning_rate=1e-3)
    fm = OTFlowMatching(
        neural_vf,
        input_dim=2,
        cond_dim=1,
        iterations=3,
        valid_freq=2,
        ot_solver=ot_solver,
        flow=flow,
        time_sampler=time_sampler,
        optimizer=optimizer
    )
    fm(
        data_loader_gaussian_with_conditions,
        data_loader_gaussian_with_conditions
    )

    batch = next(data_loader_gaussian_with_conditions)
    result_forward = fm.transport(
        batch["source_lin"], condition=batch["source_conditions"], forward=True
    )
    assert isinstance(result_forward, jnp.ndarray)
    assert jnp.sum(jnp.isnan(result_forward)) == 0

    result_backward = fm.transport(
        batch["target_lin"],
        condition=batch["source_conditions"],
        forward=False
    )
    assert isinstance(result_backward, jnp.ndarray)
    assert jnp.sum(jnp.isnan(result_backward)) == 0

  @pytest.mark.parametrize(
      "flow",
      [ConstantNoiseFlow(0.0),
       ConstantNoiseFlow(1.0),
       BrownianNoiseFlow(0.2)]
  )
  def test_flow_matching_conditional(
      self, data_loader_gaussian_conditional, flow: Type[BaseFlow]
  ):
    neural_vf = NeuralVectorField(
        output_dim=2,
        condition_dim=0,
        latent_embed_dim=5,
    )
    ot_solver = sinkhorn.Sinkhorn()
    time_sampler = UniformSampler()
    optimizer = optax.adam(learning_rate=1e-3)
    fm = OTFlowMatching(
        neural_vf,
        input_dim=2,
        cond_dim=0,
        iterations=3,
        valid_freq=2,
        ot_solver=ot_solver,
        flow=flow,
        time_sampler=time_sampler,
        optimizer=optimizer
    )
    fm(data_loader_gaussian_conditional, data_loader_gaussian_conditional)

    batch = next(data_loader_gaussian_conditional)
    result_forward = fm.transport(
        batch["source_lin"], condition=batch["source_conditions"], forward=True
    )
    assert isinstance(result_forward, jnp.ndarray)
    assert jnp.sum(jnp.isnan(result_forward)) == 0

    result_backward = fm.transport(
        batch["target_lin"],
        condition=batch["source_conditions"],
        forward=False
    )
    assert isinstance(result_backward, jnp.ndarray)
    assert jnp.sum(jnp.isnan(result_backward)) == 0

  @pytest.mark.parametrize("conditional", [False, True])
  def test_flow_matching_learn_rescaling(
      self, conditional: bool, data_loader_gaussian: Iterator,
      data_loader_gaussian_conditional: Iterator
  ):
    data_loader = (
        data_loader_gaussian_conditional
        if conditional else data_loader_gaussian
    )
    batch = next(data_loader)
    source_dim = batch["source_lin"].shape[1]
    condition_dim = batch["source_conditions"].shape[1] if conditional else 0
    neural_vf = NeuralVectorField(
        output_dim=2,
        condition_dim=0,
        latent_embed_dim=5,
    )
    ot_solver = sinkhorn.Sinkhorn()
    time_sampler = UniformSampler()
    flow = ConstantNoiseFlow(1.0)
    optimizer = optax.adam(learning_rate=1e-3)

    tau_a = 0.9
    tau_b = 0.2
    mlp_eta = RescalingMLP(hidden_dim=4, condition_dim=condition_dim)
    mlp_xi = RescalingMLP(hidden_dim=4, condition_dim=condition_dim)
    fm = OTFlowMatching(
        neural_vf,
        input_dim=source_dim,
        cond_dim=condition_dim,
        iterations=3,
        valid_freq=2,
        ot_solver=ot_solver,
        flow=flow,
        time_sampler=time_sampler,
        optimizer=optimizer,
        tau_a=tau_a,
        tau_b=tau_b,
        mlp_eta=mlp_eta,
        mlp_xi=mlp_xi,
    )
    fm(data_loader, data_loader)

    result_eta = fm.evaluate_eta(
        batch["source_lin"], condition=batch["source_conditions"]
    )
    assert isinstance(result_eta, jnp.ndarray)
    assert jnp.sum(jnp.isnan(result_eta)) == 0

    result_xi = fm.evaluate_xi(
        batch["target_lin"], condition=batch["source_conditions"]
    )
    assert isinstance(result_xi, jnp.ndarray)
    assert jnp.sum(jnp.isnan(result_xi)) == 0
