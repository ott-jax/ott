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

from ott import datasets
from ott.geometry import costs
from ott.neural.methods import conditional_monge_gap
from ott.neural.methods.monge_gap import monge_gap_from_samples
from ott.neural.networks.conditional_perturbation_network import (
    ConditionalPerturbationNetwork,
)
from ott.tools import sinkhorn_divergence


@pytest.mark.fast()
class TestConditionalMongeGap:

    @pytest.mark.parametrize("n_samples", [10, 30])
    @pytest.mark.parametrize("n_features", [4, 10])
    @pytest.mark.parametrize("num_conditions", [2, 3])
    def test_non_negativity(
        self, rng: jax.Array, n_samples: int, n_features: int,
        num_conditions: int,
    ):
        rng1, rng2 = jax.random.split(rng)
        per_cond = n_samples // num_conditions
        n = per_cond * num_conditions

        source = jax.random.normal(rng1, (n, n_features))
        target = source + 0.5 * jax.random.normal(rng2, (n, n_features))
        condition = jnp.repeat(jnp.arange(num_conditions), per_cond)

        gap = conditional_monge_gap.cmonge_gap_from_samples(
            source, target, condition,
            num_segments=num_conditions,
            max_measure_size=per_cond,
        )
        np.testing.assert_array_equal(gap >= 0, True)

    def test_jit_consistency(self, rng: jax.Array):
        n, d, k = 60, 4, 3
        per_cond = n // k
        rng1, rng2 = jax.random.split(rng)
        source = jax.random.normal(rng1, (n, d))
        target = source + 0.1 * jax.random.normal(rng2, (n, d))
        condition = jnp.repeat(jnp.arange(k), per_cond)

        eager_gap = conditional_monge_gap.cmonge_gap_from_samples(
            source, target, condition,
            num_segments=k, max_measure_size=per_cond,
        )
        jit_gap = jax.jit(
            lambda s, t, c: conditional_monge_gap.cmonge_gap_from_samples(
                s, t, c, num_segments=k, max_measure_size=per_cond,
            )
        )(source, target, condition)

        np.testing.assert_allclose(eager_gap, jit_gap, rtol=1e-3)

    def test_matches_loop_baseline(self, rng: jax.Array):
        """Segment-based result matches manual per-condition loop."""
        n, d, k = 60, 4, 3
        per_cond = n // k
        rng1, rng2 = jax.random.split(rng)
        source = jax.random.normal(rng1, (n, d))
        target = source + 0.1 * jax.random.normal(rng2, (n, d))
        condition = jnp.repeat(jnp.arange(k), per_cond)

        new_gap = conditional_monge_gap.cmonge_gap_from_samples(
            source, target, condition,
            num_segments=k, max_measure_size=per_cond,
        )

        # Manual loop (the old approach)
        manual_gaps = []
        for c in range(k):
            mask = condition == c
            gap = monge_gap_from_samples(source[mask], target[mask])
            manual_gaps.append(float(gap))
        manual_avg = sum(manual_gaps) / len(manual_gaps)

        np.testing.assert_allclose(float(new_gap), manual_avg, atol=1e-5)

    def test_identity_smaller_than_random(self, rng: jax.Array):
        """Identity map should have smaller Monge gap than a random map."""
        n, d, k = 60, 4, 3
        per_cond = n // k
        rng1, rng2 = jax.random.split(rng)
        source = jax.random.normal(rng1, (n, d))
        condition = jnp.repeat(jnp.arange(k), per_cond)

        identity_gap = conditional_monge_gap.cmonge_gap_from_samples(
            source, source, condition,
            num_segments=k, max_measure_size=per_cond,
        )
        random_target = jax.random.normal(rng2, (n, d)) * 3.0
        random_gap = conditional_monge_gap.cmonge_gap_from_samples(
            source, random_target, condition,
            num_segments=k, max_measure_size=per_cond,
        )
        assert identity_gap < random_gap

    @pytest.mark.parametrize("cost_fn", [
        costs.SqEuclidean(),
        costs.PNormP(p=1),
    ], ids=["sqeucl", "pnorm-1"])
    def test_different_costs(self, rng: jax.Array, cost_fn: costs.CostFn):
        n, d, k = 30, 4, 3
        per_cond = n // k
        rng1, rng2 = jax.random.split(rng)
        source = jax.random.normal(rng1, (n, d))
        target = source + jax.random.normal(rng2, (n, d)) * 0.5
        condition = jnp.repeat(jnp.arange(k), per_cond)

        gap = conditional_monge_gap.cmonge_gap_from_samples(
            source, target, condition, cost_fn=cost_fn,
            num_segments=k, max_measure_size=per_cond,
        )
        np.testing.assert_array_equal(jnp.isfinite(gap), True)
        np.testing.assert_array_equal(gap >= 0, True)

    def test_return_output_shape(self, rng: jax.Array):
        n, d, k = 60, 4, 3
        per_cond = n // k
        rng1, rng2 = jax.random.split(rng)
        source = jax.random.normal(rng1, (n, d))
        target = source + 0.1 * jax.random.normal(rng2, (n, d))
        condition = jnp.repeat(jnp.arange(k), per_cond)

        result = conditional_monge_gap.cmonge_gap_from_samples(
            source, target, condition,
            num_segments=k, max_measure_size=per_cond,
            return_output=True,
        )
        assert isinstance(result, tuple)
        avg_gap, per_cond_gaps = result
        assert per_cond_gaps.shape == (k,)
        np.testing.assert_allclose(
            float(avg_gap), float(jnp.mean(per_cond_gaps)), rtol=1e-5,
        )


@pytest.mark.fast()
class TestConditionalMongeGapEstimator:

    def test_estimator_convergence(self):
        """Train a conditional map and verify loss decreases."""
        num_conditions = 3
        dim_data = 2
        dim_cond = num_conditions  # one-hot
        batch_size = 30

        train_ds, valid_ds, _, n_cond, max_ms = (
            datasets.create_conditional_gaussian_mixture_samplers(
                num_conditions=num_conditions,
                dim=dim_data,
                train_batch_size=batch_size,
                valid_batch_size=batch_size,
            )
        )

        def fitting_loss(mapped, target):
            div, _ = sinkhorn_divergence.sinkdiv(x=mapped, y=target)
            return div, None

        def regularizer(source, mapped, labels):
            gap, per_cond = conditional_monge_gap.cmonge_gap_from_samples(
                source, mapped, labels,
                num_segments=n_cond,
                max_measure_size=max_ms,
                return_output=True,
            )
            return gap, None

        model = ConditionalPerturbationNetwork(
            dim_hidden=[16, 8],
            dim_data=dim_data,
            dim_cond=dim_cond,
            dim_cond_map=(16,),
            is_potential=False,
            context_entity_bonds=((0, dim_cond),),
            num_contexts=1,
        )

        solver = conditional_monge_gap.ConditionalMongeGapEstimator(
            dim_data=dim_data,
            fitting_loss=fitting_loss,
            regularizer=regularizer,
            model=model,
            regularizer_strength=1.0,
            num_train_iters=15,
            logging=True,
            valid_freq=5,
        )

        neural_state, logs = solver.train_map_estimator(
            *train_ds, *valid_ds,
        )

        # Loss should decrease
        assert logs["train"]["total_loss"][0] > logs["train"]["total_loss"][-1]

        # Output shape should match input
        source_batch = next(train_ds.source_iter)
        cond_batch = next(train_ds.condition_iter)
        mapped = neural_state.apply_fn(
            {"params": neural_state.params}, source_batch, cond_batch,
        )
        assert mapped.shape == source_batch.shape
        np.testing.assert_array_equal(jnp.all(jnp.isfinite(mapped)), True)

    def test_estimator_no_regularizer(self):
        """Training with regularizer_strength=0 still converges."""
        num_conditions = 2
        dim_data = 2
        dim_cond = num_conditions
        batch_size = 20

        train_ds, valid_ds, _, _, _ = (
            datasets.create_conditional_gaussian_mixture_samplers(
                num_conditions=num_conditions,
                dim=dim_data,
                train_batch_size=batch_size,
                valid_batch_size=batch_size,
            )
        )

        def fitting_loss(mapped, target):
            div, _ = sinkhorn_divergence.sinkdiv(x=mapped, y=target)
            return div, None

        model = ConditionalPerturbationNetwork(
            dim_hidden=[8, 8],
            dim_data=dim_data,
            dim_cond=dim_cond,
            dim_cond_map=(8,),
            is_potential=False,
            context_entity_bonds=((0, dim_cond),),
            num_contexts=1,
        )

        solver = conditional_monge_gap.ConditionalMongeGapEstimator(
            dim_data=dim_data,
            fitting_loss=fitting_loss,
            model=model,
            regularizer_strength=0.0,
            num_train_iters=10,
            logging=True,
            valid_freq=5,
        )

        neural_state, logs = solver.train_map_estimator(
            *train_ds, *valid_ds,
        )

        # Should have run without errors and logged metrics
        assert len(logs["train"]["total_loss"]) > 0
        # Mapped output should be finite
        source_batch = next(train_ds.source_iter)
        cond_batch = next(train_ds.condition_iter)
        mapped = neural_state.apply_fn(
            {"params": neural_state.params}, source_batch, cond_batch,
        )
        np.testing.assert_array_equal(jnp.all(jnp.isfinite(mapped)), True)
