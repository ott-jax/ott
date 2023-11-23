from typing import Type

import diffrax
import jax
import jax.numpy as jnp
import optax
import pytest

from ott.neural.models.models import NeuralVectorField
from ott.neural.solvers.flow_matching import FlowMatching
from ott.neural.solvers.flows import (
    BaseFlow,
    BrownianNoiseFlow,
    ConstantNoiseFlow,
)
from ott.solvers.linear import sinkhorn


class TestFlowMatching:

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
    optimizer = optax.adam(learning_rate=1e-3)
    fm = FlowMatching(
        neural_vf,
        input_dim=2,
        cond_dim=0,
        iterations=3,
        valid_freq=2,
        ot_solver=ot_solver,
        flow=flow,
        optimizer=optimizer
    )
    fm(data_loader_gaussian, data_loader_gaussian)

    source, target, condition = next(data_loader_gaussian)
    result_forward = fm.transport(source, condition=condition, forward=True)
    assert isinstance(result_forward, diffrax.Solution)
    assert jnp.sum(jnp.isnan(result_forward.y)) == 0

    result_backward = fm.transport(target, condition=condition, forward=False)
    assert isinstance(result_backward, diffrax.Solution)
    assert jnp.sum(jnp.isnan(result_backward.y)) == 0

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
    optimizer = optax.adam(learning_rate=1e-3)
    fm = FlowMatching(
        neural_vf,
        input_dim=2,
        cond_dim=1,
        iterations=3,
        valid_freq=2,
        ot_solver=ot_solver,
        flow=flow,
        optimizer=optimizer
    )
    fm(
        data_loader_gaussian_with_conditions,
        data_loader_gaussian_with_conditions
    )

    source, target, condition = next(data_loader_gaussian_with_conditions)
    result_forward = fm.transport(source, condition=condition, forward=True)
    assert isinstance(result_forward, jax.Array)
    assert jnp.sum(jnp.isnan(result_forward)) == 0

    result_backward = fm.transport(target, condition=condition, forward=False)
    assert isinstance(result_backward, jax.Array)
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
    optimizer = optax.adam(learning_rate=1e-3)
    fm = FlowMatching(
        neural_vf,
        input_dim=2,
        cond_dim=0,
        iterations=3,
        valid_freq=2,
        ot_solver=ot_solver,
        flow=flow,
        optimizer=optimizer
    )
    fm(data_loader_gaussian_conditional, data_loader_gaussian_conditional)

    source, target, condition = next(data_loader_gaussian_conditional)
    result_forward = fm.transport(source, condition=condition, forward=True)
    assert isinstance(result_forward, diffrax.Solution)
    assert jnp.sum(jnp.isnan(result_forward.y)) == 0

    result_backward = fm.transport(target, condition=condition, forward=False)
    assert isinstance(result_backward, diffrax.Solution)
    assert jnp.sum(jnp.isnan(result_backward.y)) == 0
