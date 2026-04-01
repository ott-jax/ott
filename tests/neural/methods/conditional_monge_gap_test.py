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

import time

import pytest

import jax
import jax.numpy as jnp
import numpy as np

from ott import datasets
from ott.geometry import costs, regularizers
from ott.neural.methods import conditional_monge_gap
from ott.neural.methods.monge_gap import monge_gap_from_samples
from ott.neural.networks import potentials
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
      self,
      rng: jax.Array,
      n_samples: int,
      n_features: int,
      num_conditions: int,
  ):
    rng1, rng2 = jax.random.split(rng)
    per_cond = n_samples // num_conditions
    n = per_cond * num_conditions

    source = jax.random.normal(rng1, (n, n_features))
    target = source + 0.5 * jax.random.normal(rng2, (n, n_features))
    condition = jnp.repeat(jnp.arange(num_conditions), per_cond)

    gap = conditional_monge_gap.cmonge_gap_from_samples(
        source,
        target,
        condition,
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
        source,
        target,
        condition,
        num_segments=k,
        max_measure_size=per_cond,
    )
    jit_gap = jax.jit(
        lambda s, t, c: conditional_monge_gap.cmonge_gap_from_samples(
            s,
            t,
            c,
            num_segments=k,
            max_measure_size=per_cond,
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
        source,
        target,
        condition,
        num_segments=k,
        max_measure_size=per_cond,
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
        source,
        source,
        condition,
        num_segments=k,
        max_measure_size=per_cond,
    )
    random_target = jax.random.normal(rng2, (n, d)) * 3.0
    random_gap = conditional_monge_gap.cmonge_gap_from_samples(
        source,
        random_target,
        condition,
        num_segments=k,
        max_measure_size=per_cond,
    )
    assert identity_gap < random_gap

  @pytest.mark.parametrize(
      "cost_fn",
      [
          costs.SqEuclidean(),
          costs.PNormP(p=1),
      ],
      ids=["sqeucl", "pnorm-1"],
  )
  def test_different_costs(self, rng: jax.Array, cost_fn: costs.CostFn):
    n, d, k = 30, 4, 3
    per_cond = n // k
    rng1, rng2 = jax.random.split(rng)
    source = jax.random.normal(rng1, (n, d))
    target = source + jax.random.normal(rng2, (n, d)) * 0.5
    condition = jnp.repeat(jnp.arange(k), per_cond)

    gap = conditional_monge_gap.cmonge_gap_from_samples(
        source,
        target,
        condition,
        cost_fn=cost_fn,
        num_segments=k,
        max_measure_size=per_cond,
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
        source,
        target,
        condition,
        num_segments=k,
        max_measure_size=per_cond,
        return_output=True,
    )
    assert isinstance(result, tuple)
    avg_gap, per_cond_gaps = result
    assert per_cond_gaps.shape == (k,)
    np.testing.assert_allclose(
        float(avg_gap),
        float(jnp.mean(per_cond_gaps)),
        rtol=1e-5,
    )

  @pytest.mark.parametrize("n_samples", [10, 30])
  @pytest.mark.parametrize("n_features", [4, 10])
  def test_non_negativity_neural_map(
      self,
      rng: jax.Array,
      n_samples: int,
      n_features: int,
  ):
    """Non-negativity with a learned nonlinear map."""
    k = 2
    per_cond = n_samples // k
    n = per_cond * k
    rng1, rng2 = jax.random.split(rng)

    source = jax.random.normal(rng1, (n, n_features))
    model = potentials.PotentialMLP(dim_hidden=[8, 8], is_potential=False)
    params = model.init(rng2, x=source[0])
    target = model.apply(params, source)
    condition = jnp.repeat(jnp.arange(k), per_cond)

    gap = conditional_monge_gap.cmonge_gap_from_samples(
        source,
        target,
        condition,
        num_segments=k,
        max_measure_size=per_cond,
    )
    np.testing.assert_array_equal(jnp.isfinite(gap), True)
    np.testing.assert_array_equal(gap >= 0, True)

  @pytest.mark.parametrize(
      "cost_fn",
      [
          costs.PNormP(p=1),
          costs.RegTICost(regularizers.L1(), lam=2.0),
          costs.RegTICost(regularizers.STVS(gamma=3.0), lam=1.0),
      ],
      ids=["pnorm-1", "l1-lam2", "stvs-lam1"],
  )
  def test_different_costs_give_different_values(
      self,
      rng: jax.Array,
      cost_fn: costs.CostFn,
  ):
    """Non-Euclidean costs produce different cmonge_gap than Euclidean."""
    n, d, k = 30, 5, 3
    per_cond = n // k
    rng1, rng2 = jax.random.split(rng)
    source = jax.random.normal(rng1, (n, d))
    target = jax.random.normal(rng2, (n, d)) * 0.1 + 3.0
    condition = jnp.repeat(jnp.arange(k), per_cond)

    gap_eucl = conditional_monge_gap.cmonge_gap_from_samples(
        source,
        target,
        condition,
        cost_fn=costs.Euclidean(),
        num_segments=k,
        max_measure_size=per_cond,
    )
    gap_other = conditional_monge_gap.cmonge_gap_from_samples(
        source,
        target,
        condition,
        cost_fn=cost_fn,
        num_segments=k,
        max_measure_size=per_cond,
    )

    with pytest.raises(AssertionError, match=r"tolerance"):
      np.testing.assert_allclose(
          gap_eucl,
          gap_other,
          rtol=1e-1,
          atol=1e-1,
      )
    np.testing.assert_array_equal(jnp.isfinite(gap_eucl), True)
    np.testing.assert_array_equal(jnp.isfinite(gap_other), True)

  def test_uniform_conditions_equals_averaged_monge_gap(
      self,
      rng: jax.Array,
  ):
    """cmonge_gap with equal-size conditions == mean of monge_gap calls."""
    k = 3
    per_cond = 20
    d = 5

    # Different offsets per condition so gaps are distinct
    offsets = jnp.array([0.1, 1.0, 3.0])
    rngs = jax.random.split(rng, 2 * k)
    sources, targets = [], []
    for c in range(k):
      s = jax.random.normal(rngs[2 * c], (per_cond, d))
      t = (
          s + offsets[c] +
          0.05 * jax.random.normal(rngs[2 * c + 1], (per_cond, d))
      )
      sources.append(s)
      targets.append(t)

    source = jnp.concatenate(sources, axis=0)
    target = jnp.concatenate(targets, axis=0)
    condition = jnp.repeat(jnp.arange(k), per_cond)

    # Segmented cmonge_gap (single call, vmapped)
    t0 = time.perf_counter()
    avg_gap, per_cond_gaps = conditional_monge_gap.cmonge_gap_from_samples(
        source,
        target,
        condition,
        num_segments=k,
        max_measure_size=per_cond,
        return_output=True,
    )
    # Force computation to complete before timing
    avg_gap.block_until_ready()
    t_cmonge = time.perf_counter() - t0

    # Manual per-condition monge_gap calls (K sequential calls)
    t0 = time.perf_counter()
    manual_gaps = []
    for c in range(k):
      gap_c = monge_gap_from_samples(sources[c], targets[c])
      manual_gaps.append(float(gap_c))
    manual_avg = sum(manual_gaps) / k
    t_loop = time.perf_counter() - t0

    # Single-condition overhead: cmonge_gap(K=1) vs monge_gap
    t0 = time.perf_counter()
    gap_single_cmonge = conditional_monge_gap.cmonge_gap_from_samples(
        sources[0],
        targets[0],
        jnp.zeros(per_cond, dtype=jnp.int32),
        num_segments=1,
        max_measure_size=per_cond,
    )
    gap_single_cmonge.block_until_ready()
    t_cmonge_1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    gap_single_monge = monge_gap_from_samples(sources[0], targets[0])
    float(gap_single_monge)  # block
    t_monge_1 = time.perf_counter() - t0

    print(  # noqa: T201
        f"\n  K={k}: cmonge_gap: {t_cmonge:.3f}s | "
        f"loop({k}x monge_gap): {t_loop:.3f}s | "
        f"speedup: {t_loop / t_cmonge:.1f}x"
        f"\n  K=1: cmonge_gap: {t_cmonge_1:.3f}s | "
        f"monge_gap: {t_monge_1:.3f}s | "
        f"overhead: {t_cmonge_1 / t_monge_1:.1f}x"
    )

    # Average should match
    np.testing.assert_allclose(float(avg_gap), manual_avg, atol=1e-5)
    # Per-condition gaps should match individual calls
    for c in range(k):
      np.testing.assert_allclose(
          float(per_cond_gaps[c]),
          manual_gaps[c],
          atol=1e-5,
      )

  def test_unequal_conditions_shifts_average(self, rng: jax.Array):
    """With unequal n_k, per-condition gaps change and shift the average.

        The segment interface pads all conditions to max_measure_size, so
        per-condition gaps with padding do NOT exactly match non-padded
        monge_gap_from_samples calls (the geometry differs). We verify
        structural properties instead: gaps are finite, easy < hard,
        average = mean(per_cond_gaps), and the average shifts when n_k changes.
        """
    d = 5
    rng_easy, rng_hard, rng_noise = jax.random.split(rng, 3)

    base_easy = jax.random.normal(rng_easy, (60, d))
    base_hard = jax.random.normal(rng_hard, (60, d))
    noise = 0.01 * jax.random.normal(rng_noise, (60, d))

    target_easy = base_easy + noise
    target_hard = base_hard + 5.0

    # (a) Equal sizes: 30/30
    n_eq = 30
    src_eq = jnp.concatenate([base_easy[:n_eq], base_hard[:n_eq]])
    tgt_eq = jnp.concatenate([target_easy[:n_eq], target_hard[:n_eq]])
    cond_eq = jnp.repeat(jnp.arange(2), n_eq)

    avg_eq, gaps_eq = conditional_monge_gap.cmonge_gap_from_samples(
        src_eq,
        tgt_eq,
        cond_eq,
        num_segments=2,
        max_measure_size=n_eq,
        return_output=True,
    )

    # (b) Unequal sizes: 50 easy / 10 hard
    n_a, n_b = 50, 10
    src_uneq = jnp.concatenate([base_easy[:n_a], base_hard[:n_b]])
    tgt_uneq = jnp.concatenate([target_easy[:n_a], target_hard[:n_b]])
    cond_uneq = jnp.concatenate([
        jnp.zeros(n_a, dtype=jnp.int32),
        jnp.ones(n_b, dtype=jnp.int32),
    ])

    avg_uneq, gaps_uneq = conditional_monge_gap.cmonge_gap_from_samples(
        src_uneq,
        tgt_uneq,
        cond_uneq,
        num_segments=2,
        max_measure_size=n_a,
        return_output=True,
    )

    # All gaps are finite and non-negative
    for gaps in [gaps_eq, gaps_uneq]:
      np.testing.assert_array_equal(jnp.all(jnp.isfinite(gaps)), True)
      np.testing.assert_array_equal(jnp.all(gaps >= 0), True)

    # Easy condition has smaller gap than hard condition
    assert gaps_eq[0] < gaps_eq[1]
    assert gaps_uneq[0] < gaps_uneq[1]

    # Average is the mean of per-condition gaps
    np.testing.assert_allclose(
        float(avg_eq),
        float(jnp.mean(gaps_eq)),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        float(avg_uneq),
        float(jnp.mean(gaps_uneq)),
        rtol=1e-5,
    )

    # Averages differ between equal and unequal splits (n_k affects
    # the padded OT cost estimation, shifting per-condition gaps)
    assert float(avg_eq) != float(avg_uneq)

  def test_per_condition_gaps_reflect_difficulty(self, rng: jax.Array):
    """Per-condition gaps increase with offset magnitude."""
    k = 3
    per_cond = 25
    d = 4
    offsets = jnp.array([0.0, 1.5, 5.0])

    rngs = jax.random.split(rng, 2 * k)
    sources, targets = [], []
    for c in range(k):
      s = jax.random.normal(rngs[2 * c], (per_cond, d))
      t = s + offsets[c]
      sources.append(s)
      targets.append(t)

    source = jnp.concatenate(sources, axis=0)
    target = jnp.concatenate(targets, axis=0)
    condition = jnp.repeat(jnp.arange(k), per_cond)

    _, per_cond_gaps = conditional_monge_gap.cmonge_gap_from_samples(
        source,
        target,
        condition,
        num_segments=k,
        max_measure_size=per_cond,
        return_output=True,
    )

    assert per_cond_gaps[0] < per_cond_gaps[1] < per_cond_gaps[2]


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
          source,
          mapped,
          labels,
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
        *train_ds,
        *valid_ds,
    )

    # Loss should decrease
    assert logs["train"]["total_loss"][0] > logs["train"]["total_loss"][-1]

    # Output shape should match input
    source_batch = next(train_ds.source_iter)
    cond_batch = next(train_ds.condition_iter)
    mapped = neural_state.apply_fn(
        {"params": neural_state.params},
        source_batch,
        cond_batch,
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
        *train_ds,
        *valid_ds,
    )

    # Should have run without errors and logged metrics
    assert len(logs["train"]["total_loss"]) > 0
    # Mapped output should be finite
    source_batch = next(train_ds.source_iter)
    cond_batch = next(train_ds.condition_iter)
    mapped = neural_state.apply_fn(
        {"params": neural_state.params},
        source_batch,
        cond_batch,
    )
    np.testing.assert_array_equal(jnp.all(jnp.isfinite(mapped)), True)
