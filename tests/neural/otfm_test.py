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
from typing import Iterator, Literal, Type

import pytest

import jax.numpy as jnp
from jax import random

import optax

from ott.neural.flow_models import flows, models, otfm, samplers
from ott.neural.models import base_solver, nets
from ott.solvers.linear import sinkhorn, sinkhorn_lr


class TestOTFlowMatching:

  @pytest.mark.parametrize(
      "flow", [
          flows.ConstantNoiseFlow(0.0),
          flows.ConstantNoiseFlow(1.0),
          flows.BrownianNoiseFlow(0.2)
      ]
  )
  def test_flow_matching_unconditional(
      self, data_loaders_gaussian, flow: Type[flows.BaseFlow]
  ):
    input_dim = 2
    condition_dim = 0
    neural_vf = models.VelocityField(
        output_dim=2,
        condition_dim=0,
        latent_embed_dim=5,
    )
    ot_solver = sinkhorn.Sinkhorn()
    ot_matcher = base_solver.OTMatcherLinear(ot_solver)
    time_sampler = samplers.uniform_sampler
    optimizer = optax.adam(learning_rate=1e-3)
    unbalancedness_handler = base_solver.UnbalancednessHandler(
        random.PRNGKey(0), input_dim, input_dim, condition_dim
    )
    fm = otfm.OTFlowMatching(
        neural_vf,
        input_dim=input_dim,
        cond_dim=condition_dim,
        iterations=3,
        valid_freq=2,
        ot_matcher=ot_matcher,
        flow=flow,
        time_sampler=time_sampler,
        optimizer=optimizer,
        unbalancedness_handler=unbalancedness_handler
    )
    fm(
        data_loaders_gaussian[0], data_loaders_gaussian[1],
        data_loaders_gaussian[0], data_loaders_gaussian[1]
    )

    batch_src = next(iter(data_loaders_gaussian[0]))
    source = jnp.asarray(batch_src["lin"])
    batch_tgt = next(iter(data_loaders_gaussian[1]))
    target = jnp.asarray(batch_tgt["lin"])
    source_conditions = jnp.asarray(
        batch_src["conditions"]
    ) if "conditions" in batch_src else None
    result_forward = fm.transport(
        source, condition=source_conditions, forward=True
    )
    assert isinstance(result_forward, jnp.ndarray)
    assert jnp.sum(jnp.isnan(result_forward)) == 0

    result_backward = fm.transport(
        target, condition=source_conditions, forward=False
    )
    assert isinstance(result_backward, jnp.ndarray)
    assert jnp.sum(jnp.isnan(result_backward)) == 0

  @pytest.mark.parametrize(
      "flow", [
          flows.ConstantNoiseFlow(0.0),
          flows.ConstantNoiseFlow(1.1),
          flows.BrownianNoiseFlow(2.2)
      ]
  )
  def test_flow_matching_with_conditions(
      self, data_loader_gaussian_with_conditions, flow: Type[flows.BaseFlow]
  ):
    input_dim = 2
    condition_dim = 1
    neural_vf = models.VelocityField(
        output_dim=2,
        condition_dim=1,
        latent_embed_dim=5,
    )
    ot_solver = sinkhorn.Sinkhorn()
    ot_matcher = base_solver.OTMatcherLinear(ot_solver)
    time_sampler = functools.partial(samplers.uniform_sampler, offset=1e-5)
    optimizer = optax.adam(learning_rate=1e-3)
    unbalancedness_handler = base_solver.UnbalancednessHandler(
        random.PRNGKey(0), input_dim, input_dim, condition_dim
    )

    fm = otfm.OTFlowMatching(
        neural_vf,
        input_dim=2,
        cond_dim=1,
        iterations=3,
        valid_freq=2,
        ot_matcher=ot_matcher,
        flow=flow,
        time_sampler=time_sampler,
        optimizer=optimizer,
        unbalancedness_handler=unbalancedness_handler
    )
    fm(
        data_loader_gaussian_with_conditions,
        data_loader_gaussian_with_conditions
    )

    batch = next(iter(data_loader_gaussian_with_conditions))
    source = jnp.asarray(batch["source_lin"])
    target = jnp.asarray(batch["target_lin"])
    source_conditions = jnp.asarray(batch["source_conditions"]) if len(
        batch["source_conditions"]
    ) > 0 else None
    result_forward = fm.transport(
        source, condition=source_conditions, forward=True
    )
    assert isinstance(result_forward, jnp.ndarray)
    assert jnp.sum(jnp.isnan(result_forward)) == 0

    result_backward = fm.transport(
        target, condition=source_conditions, forward=False
    )
    assert isinstance(result_backward, jnp.ndarray)
    assert jnp.sum(jnp.isnan(result_backward)) == 0

  @pytest.mark.parametrize(
      "flow",
      [
          flows.ConstantNoiseFlow(0.0),
          flows.ConstantNoiseFlow(13.0),
          flows.BrownianNoiseFlow(0.12)
      ],
  )
  @pytest.mark.parametrize("solver", ["sinkhorn", "lr_sinkhorn"])
  def test_flow_matching_conditional(
      self, data_loader_gaussian_conditional, flow: Type[flows.BaseFlow],
      solver: Literal["sinkhorn", "lr_sinkhorn"]
  ):
    dim = 2
    condition_dim = 0
    neural_vf = models.VelocityField(
        output_dim=dim,
        condition_dim=condition_dim,
        latent_embed_dim=5,
    )
    ot_solver = sinkhorn.Sinkhorn(
    ) if solver == "sinkhorn" else sinkhorn_lr.LRSinkhorn()
    ot_matcher = base_solver.OTMatcherLinear(ot_solver)
    time_sampler = samplers.uniform_sampler
    optimizer = optax.adam(learning_rate=1e-3)
    unbalancedness_handler = base_solver.UnbalancednessHandler(
        random.PRNGKey(0), dim, dim, condition_dim
    )

    fm = otfm.OTFlowMatching(
        neural_vf,
        input_dim=dim,
        cond_dim=condition_dim,
        iterations=3,
        valid_freq=2,
        ot_matcher=ot_matcher,
        flow=flow,
        time_sampler=time_sampler,
        optimizer=optimizer,
        unbalancedness_handler=unbalancedness_handler
    )
    fm(data_loader_gaussian_conditional, data_loader_gaussian_conditional)

    batch = next(iter(data_loader_gaussian_conditional))
    source = jnp.asarray(batch["source_lin"])
    target = jnp.asarray(batch["target_lin"])
    source_conditions = jnp.asarray(batch["source_conditions"]) if len(
        batch["source_conditions"]
    ) > 0 else None
    result_forward = fm.transport(
        source, condition=source_conditions, forward=True
    )
    assert isinstance(result_forward, jnp.ndarray)
    assert jnp.sum(jnp.isnan(result_forward)) == 0

    result_backward = fm.transport(
        target, condition=source_conditions, forward=False
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
    batch = next(iter(data_loader))
    source = jnp.asarray(batch["source_lin"])
    target = jnp.asarray(batch["target_lin"])
    source_conditions = jnp.asarray(batch["source_conditions"]) if len(
        batch["source_conditions"]
    ) > 0 else None

    source_dim = source.shape[1]
    condition_dim = source_conditions.shape[1] if conditional else 0
    neural_vf = models.VelocityField(
        output_dim=2,
        condition_dim=0,
        latent_embed_dim=5,
    )
    ot_solver = sinkhorn.Sinkhorn()
    time_sampler = samplers.uniform_sampler
    flow = flows.ConstantNoiseFlow(1.0)
    optimizer = optax.adam(learning_rate=1e-3)

    tau_a = 0.9
    tau_b = 0.2
    rescaling_a = nets.RescalingMLP(hidden_dim=4, condition_dim=condition_dim)
    rescaling_b = nets.RescalingMLP(hidden_dim=4, condition_dim=condition_dim)
    ot_matcher = base_solver.OTMatcherLinear(
        ot_solver,
        tau_a=tau_a,
        tau_b=tau_b,
    )
    unbalancedness_handler = base_solver.UnbalancednessHandler(
        random.PRNGKey(0),
        source_dim,
        source_dim,
        condition_dim,
        tau_a=tau_a,
        tau_b=tau_b,
        rescaling_a=rescaling_a,
        rescaling_b=rescaling_b
    )

    fm = otfm.OTFlowMatching(
        neural_vf,
        input_dim=source_dim,
        cond_dim=condition_dim,
        iterations=3,
        valid_freq=2,
        ot_matcher=ot_matcher,
        flow=flow,
        time_sampler=time_sampler,
        optimizer=optimizer,
        unbalancedness_handler=unbalancedness_handler,
    )
    fm(data_loader, data_loader)

    result_eta = fm.unbalancedness_handler.evaluate_eta(
        source, condition=source_conditions
    )
    assert isinstance(result_eta, jnp.ndarray)
    assert jnp.sum(jnp.isnan(result_eta)) == 0

    result_xi = fm.unbalancedness_handler.evaluate_xi(
        target, condition=source_conditions
    )
    assert isinstance(result_xi, jnp.ndarray)
    assert jnp.sum(jnp.isnan(result_xi)) == 0
