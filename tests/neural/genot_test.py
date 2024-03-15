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
from typing import Iterator, Literal, Optional, Union

import pytest

import jax.numpy as jnp

import optax

from ott.geometry import costs
from ott.neural.flow_models.models import VelocityField
from ott.neural.flow_models.utils import uniform_sampler
from ott.solvers.linear import sinkhorn, sinkhorn_lr
from ott.solvers.quadratic import gromov_wasserstein, gromov_wasserstein_lr


class TestGENOTLin:

  @pytest.mark.parametrize("scale_cost", ["mean", 2.0])
  @pytest.mark.parametrize("k_samples_per_x", [1, 3])
  @pytest.mark.parametrize("solver_latent_to_data", [None, "sinkhorn"])
  @pytest.mark.parametrize("solver", ["sinkhorn", "lr_sinkhorn"])
  def test_genot_linear_unconditional(
      self,
      genot_data_loader_linear: Iterator,
      scale_cost: Union[float, Literal["mean"]],
      k_samples_per_x: int,
      solver_latent_to_data: Optional[str],
      solver: Literal["sinkhorn", "lr_sinkhorn"],
  ):
    matcher_latent_to_data = (
        None if solver_latent_to_data is None else
        base_solver.OTMatcherLinear(sinkhorn.Sinkhorn())
    )
    batch = next(iter(genot_data_loader_linear))
    source_lin, source_conditions, target_lin = jnp.array(
        batch["source_lin"]
    ), jnp.array(batch["source_conditions"]) if len(batch["source_conditions"]
                                                   ) else None, jnp.array(
                                                       batch["target_lin"]
                                                   )

    source_dim = source_lin.shape[1]
    target_dim = target_lin.shape[1]
    condition_dim = 0

    neural_vf = VelocityField(
        output_dim=target_dim,
        condition_dim=source_dim + condition_dim,
        hidden_dim=5,
    )
    ot_solver = sinkhorn.Sinkhorn(
    ) if solver == "sinkhorn" else sinkhorn_lr.LRSinkhorn(rank=3)
    ot_matcher = base_solver.OTMatcherLinear(
        ot_solver, cost_fn=costs.SqEuclidean(), scale_cost=scale_cost
    )
    time_sampler = uniform_sampler
    optimizer = optax.adam(learning_rate=1e-3)
    genot = GENOTLin(
        neural_vf,
        input_dim=source_dim,
        output_dim=target_dim,
        cond_dim=condition_dim,
        ot_matcher=ot_matcher,
        optimizer=optimizer,
        time_sampler=time_sampler,
        k_samples_per_x=k_samples_per_x,
        matcher_latent_to_data=matcher_latent_to_data,
    )
    genot(
        genot_data_loader_linear,
        genot_data_loader_linear,
        n_iters=2,
        valid_freq=3
    )

    batch = next(iter(genot_data_loader_linear))
    result_forward = genot.transport(
        source_lin, condition=source_conditions, forward=True
    )
    assert isinstance(result_forward, jnp.ndarray)
    assert jnp.sum(jnp.isnan(result_forward)) == 0

  @pytest.mark.parametrize("k_samples_per_x", [1, 2])
  @pytest.mark.parametrize("solver_latent_to_data", [None, "sinkhorn"])
  @pytest.mark.parametrize("solver", ["sinkhorn", "lr_sinkhorn"])
  def test_genot_linear_conditional(
      self, genot_data_loader_linear_conditional: Iterator,
      k_samples_per_x: int, solver_latent_to_data: Optional[str],
      solver: Literal["sinkhorn", "lr_sinkhorn"]
  ):
    matcher_latent_to_data = (
        None if solver_latent_to_data is None else
        base_solver.OTMatcherLinear(sinkhorn.Sinkhorn())
    )

    batch = next(iter(genot_data_loader_linear_conditional))
    source_lin, source_conditions, target_lin = jnp.array(
        batch["source_lin"]
    ), jnp.array(batch["source_conditions"]) if len(batch["source_conditions"]
                                                   ) else None, jnp.array(
                                                       batch["target_lin"]
                                                   )
    source_dim = source_lin.shape[1]
    target_dim = target_lin.shape[1]
    condition_dim = source_conditions.shape[1]

    neural_vf = VelocityField(
        output_dim=target_dim,
        condition_dim=source_dim + condition_dim,
        hidden_dim=5,
    )
    ot_solver = sinkhorn.Sinkhorn(
    ) if solver == "sinkhorn" else sinkhorn_lr.LRSinkhorn(rank=3)
    ot_matcher = base_solver.OTMatcherLinear(
        ot_solver, cost_fn=costs.SqEuclidean()
    )
    time_sampler = uniform_sampler

    optimizer = optax.adam(learning_rate=1e-3)
    genot = GENOTLin(
        neural_vf,
        input_dim=source_dim,
        output_dim=target_dim,
        cond_dim=condition_dim,
        ot_matcher=ot_matcher,
        optimizer=optimizer,
        time_sampler=time_sampler,
        k_samples_per_x=k_samples_per_x,
        matcher_latent_to_data=matcher_latent_to_data,
    )
    genot(
        genot_data_loader_linear_conditional,
        genot_data_loader_linear_conditional,
        n_iters=2,
        valid_freq=3
    )
    result_forward = genot.transport(
        source_lin, condition=source_conditions, forward=True
    )
    assert isinstance(result_forward, jnp.ndarray)
    assert jnp.sum(jnp.isnan(result_forward)) == 0


class TestGENOTQuad:

  @pytest.mark.parametrize("k_samples_per_x", [1, 2])
  @pytest.mark.parametrize("solver_latent_to_data", [None, "sinkhorn"])
  @pytest.mark.parametrize("solver", ["gromov", "gromov_lr"])
  def test_genot_quad_unconditional(
      self, genot_data_loader_quad: Iterator, k_samples_per_x: int,
      solver_latent_to_data: Optional[str], solver: Literal["gromov",
                                                            "gromov_lr"]
  ):
    matcher_latent_to_data = (
        None if solver_latent_to_data is None else
        base_solver.OTMatcherLinear(sinkhorn.Sinkhorn())
    )

    batch = next(iter(genot_data_loader_quad))
    (source_quad, source_conditions, target_quad) = (
        jnp.array(batch["source_quad"]), jnp.array(batch["source_conditions"])
        if len(batch["source_conditions"]) else None,
        jnp.array(batch["target_quad"])
    )
    source_dim = source_quad.shape[1]
    target_dim = target_quad.shape[1]
    condition_dim = 0
    neural_vf = VelocityField(
        output_dim=target_dim,
        condition_dim=source_dim + condition_dim,
        hidden_dim=5,
    )
    ot_solver = gromov_wasserstein.GromovWasserstein(
        epsilon=1e-2
    ) if solver == "gromov" else gromov_wasserstein_lr.LRGromovWasserstein(
        rank=3, epsilon=1e-2
    )
    ot_matcher = base_solver.OTMatcherQuad(
        ot_solver, cost_fn=costs.SqEuclidean()
    )

    time_sampler = functools.partial(uniform_sampler, offset=1e-2)
    optimizer = optax.adam(learning_rate=1e-3)
    genot = GENOTQuad(
        neural_vf,
        input_dim=source_dim,
        output_dim=target_dim,
        cond_dim=condition_dim,
        ot_matcher=ot_matcher,
        optimizer=optimizer,
        time_sampler=time_sampler,
        k_samples_per_x=k_samples_per_x,
        matcher_latent_to_data=matcher_latent_to_data,
    )
    genot(
        genot_data_loader_quad, genot_data_loader_quad, n_iters=2, valid_freq=3
    )

    result_forward = genot.transport(
        source_quad, condition=source_conditions, forward=True
    )
    assert isinstance(result_forward, jnp.ndarray)
    assert jnp.sum(jnp.isnan(result_forward)) == 0

  @pytest.mark.parametrize("k_samples_per_x", [1, 2])
  @pytest.mark.parametrize("solver_latent_to_data", [None, "sinkhorn"])
  @pytest.mark.parametrize("solver", ["gromov", "gromov_lr"])
  def test_genot_fused_unconditional(
      self, genot_data_loader_fused: Iterator, k_samples_per_x: int,
      solver_latent_to_data: Optional[str], solver: Literal["gromov",
                                                            "gromov_lr"]
  ):
    matcher_latent_to_data = (
        None if solver_latent_to_data is None else
        base_solver.OTMatcherLinear(sinkhorn.Sinkhorn())
    )

    batch = next(iter(genot_data_loader_fused))
    (source_lin, source_quad, source_conditions, target_lin, target_quad) = (
        jnp.array(batch["source_lin"]) if len(batch["source_lin"]) else None,
        jnp.array(batch["source_quad"]), jnp.array(batch["source_conditions"])
        if len(batch["source_conditions"]) else None,
        jnp.array(batch["target_lin"]) if len(batch["target_lin"]) else None,
        jnp.array(batch["target_quad"])
    )
    source_dim = source_lin.shape[1] + source_quad.shape[1]
    target_dim = target_lin.shape[1] + target_quad.shape[1]
    condition_dim = 0
    neural_vf = VelocityField(
        output_dim=target_dim,
        condition_dim=source_dim + condition_dim,
        hidden_dim=5,
    )
    ot_solver = gromov_wasserstein.GromovWasserstein(
        epsilon=1e-2
    ) if solver == "gromov" else gromov_wasserstein_lr.LRGromovWasserstein(
        rank=3, epsilon=1e-2
    )
    ot_matcher = base_solver.OTMatcherQuad(
        ot_solver, cost_fn=costs.SqEuclidean(), fused_penalty=0.5
    )

    optimizer = optax.adam(learning_rate=1e-3)
    genot = GENOTQuad(
        neural_vf,
        input_dim=source_dim,
        output_dim=target_dim,
        cond_dim=condition_dim,
        ot_matcher=ot_matcher,
        optimizer=optimizer,
        k_samples_per_x=k_samples_per_x,
        matcher_latent_to_data=matcher_latent_to_data,
    )
    genot(
        genot_data_loader_fused,
        genot_data_loader_fused,
        n_iters=2,
        valid_freq=3
    )

    result_forward = genot.transport(
        jnp.concatenate((source_lin, source_quad), axis=1),
        condition=source_conditions,
        forward=True
    )
    assert isinstance(result_forward, jnp.ndarray)
    assert jnp.sum(jnp.isnan(result_forward)) == 0

  @pytest.mark.parametrize("k_samples_per_x", [1, 2])
  @pytest.mark.parametrize("solver_latent_to_data", [None, "sinkhorn"])
  @pytest.mark.parametrize("solver", ["gromov", "gromov_lr"])
  def test_genot_quad_conditional(
      self, genot_data_loader_quad_conditional: Iterator, k_samples_per_x: int,
      solver_latent_to_data: Optional[str], solver: Literal["gromov",
                                                            "gromov_lr"]
  ):
    matcher_latent_to_data = (
        None if solver_latent_to_data is None else
        base_solver.OTMatcherLinear(sinkhorn.Sinkhorn())
    )

    batch = next(iter(genot_data_loader_quad_conditional))
    (source_quad, source_conditions, target_quad) = (
        jnp.array(batch["source_quad"]), jnp.array(batch["source_conditions"])
        if len(batch["source_conditions"]) else None,
        jnp.array(batch["target_quad"])
    )

    source_dim = source_quad.shape[1]
    target_dim = target_quad.shape[1]
    condition_dim = source_conditions.shape[1]
    neural_vf = VelocityField(
        output_dim=target_dim,
        condition_dim=source_dim + condition_dim,
        hidden_dim=5,
    )
    ot_solver = gromov_wasserstein.GromovWasserstein(
        epsilon=1e-2
    ) if solver == "gromov" else gromov_wasserstein_lr.LRGromovWasserstein(
        rank=3, epsilon=1e-2
    )
    ot_matcher = base_solver.OTMatcherQuad(
        ot_solver, cost_fn=costs.SqEuclidean()
    )
    time_sampler = uniform_sampler

    optimizer = optax.adam(learning_rate=1e-3)
    genot = GENOTQuad(
        neural_vf,
        input_dim=source_dim,
        output_dim=target_dim,
        cond_dim=condition_dim,
        ot_matcher=ot_matcher,
        optimizer=optimizer,
        time_sampler=time_sampler,
        k_samples_per_x=k_samples_per_x,
        matcher_latent_to_data=matcher_latent_to_data,
    )
    genot(
        genot_data_loader_quad_conditional,
        genot_data_loader_quad_conditional,
        n_iters=2,
        valid_freq=3
    )

    result_forward = genot.transport(
        source_quad, condition=source_conditions, forward=True
    )
    assert isinstance(result_forward, jnp.ndarray)
    assert jnp.sum(jnp.isnan(result_forward)) == 0

  @pytest.mark.parametrize("k_samples_per_x", [1, 2])
  @pytest.mark.parametrize("solver_latent_to_data", [None, "sinkhorn"])
  @pytest.mark.parametrize("solver", ["gromov", "gromov_lr"])
  def test_genot_fused_conditional(
      self, genot_data_loader_fused_conditional: Iterator, k_samples_per_x: int,
      solver_latent_to_data: Optional[str], solver: Literal["gromov",
                                                            "gromov_lr"]
  ):
    solver_latent_to_data = (
        None if solver_latent_to_data is None else sinkhorn.Sinkhorn()
    )
    matcher_latent_to_data = (
        None if solver_latent_to_data is None else
        base_solver.OTMatcherLinear(sinkhorn.Sinkhorn())
    )
    batch = next(iter(genot_data_loader_fused_conditional))
    (source_lin, source_quad, source_conditions, target_lin, target_quad) = (
        jnp.array(batch["source_lin"]) if len(batch["source_lin"]) else None,
        jnp.array(batch["source_quad"]), jnp.array(batch["source_conditions"])
        if len(batch["source_conditions"]) else None,
        jnp.array(batch["target_lin"]) if len(batch["target_lin"]) else None,
        jnp.array(batch["target_quad"])
    )
    source_dim = source_lin.shape[1] + source_quad.shape[1]
    target_dim = target_lin.shape[1] + target_quad.shape[1]
    condition_dim = source_conditions.shape[1]
    neural_vf = VelocityField(
        output_dim=target_dim,
        condition_dim=source_dim + condition_dim,
        hidden_dim=5,
    )
    ot_solver = gromov_wasserstein.GromovWasserstein(
        epsilon=1e-2
    ) if solver == "gromov" else gromov_wasserstein_lr.LRGromovWasserstein(
        rank=3, epsilon=1e-2
    )
    ot_matcher = base_solver.OTMatcherQuad(
        ot_solver, cost_fn=costs.SqEuclidean(), fused_penalty=0.5
    )
    time_sampler = uniform_sampler
    optimizer = optax.adam(learning_rate=1e-3)

    genot = GENOTQuad(
        neural_vf,
        input_dim=source_dim,
        output_dim=target_dim,
        cond_dim=condition_dim,
        ot_matcher=ot_matcher,
        optimizer=optimizer,
        time_sampler=time_sampler,
        k_samples_per_x=k_samples_per_x,
        matcher_latent_to_data=matcher_latent_to_data,
    )
    genot(
        genot_data_loader_fused_conditional,
        genot_data_loader_fused_conditional,
        n_iters=2,
        valid_freq=3
    )

    result_forward = genot.transport(
        jnp.concatenate((source_lin, source_quad), axis=1),
        condition=source_conditions,
        forward=True
    )
    assert isinstance(result_forward, jnp.ndarray)
    assert jnp.sum(jnp.isnan(result_forward)) == 0
