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
import math
from typing import Callable, Tuple

import pytest

import jax
import jax.numpy as jnp
import numpy as np
from sklearn import datasets, linear_model, model_selection

from ott.tools import conformal


def get_model_and_data(
    *,
    n_samples: int,
    target_dim: int,
    random_state: int = 0,
) -> Tuple[Callable[[jnp.ndarray], jnp.ndarray], Tuple[jnp.ndarray, ...]]:
  x, y = datasets.make_regression(
      n_samples=n_samples,
      n_features=5,
      n_targets=target_dim,
      random_state=random_state
  )
  x_trn, x_calib, y_trn, y_calib = model_selection.train_test_split(
      x, y, train_size=0.7, random_state=random_state
  )
  x_calib, x_test, y_calib, y_test = model_selection.train_test_split(
      x_calib, y_calib, train_size=0.5, random_state=random_state
  )
  model = linear_model.LinearRegression().fit(x_trn, y_trn)

  A = jnp.asarray(model.coef_)
  b = jnp.asarray(model.intercept_[None])
  model_fn = jax.jit(lambda x: x @ A.T + b)
  data = jax.tree.map(
      jnp.asarray, (x_trn, x_calib, x_test, y_trn, y_calib, y_test)
  )
  return model_fn, data


class TestOTCP:

  @pytest.mark.parametrize("shape", [(16, 2), (58, 9), (128, 9)])
  def test_sample_target_measure(self, shape: Tuple[int, int], rng: jax.Array):
    n, d = shape
    n_per_radius = math.ceil(math.sqrt(n))
    n_sphere, n_0s = divmod(n, n_per_radius)
    n_expected = n_sphere * n_per_radius + (n_0s > 0)

    sample_fn = jax.jit(conformal.sobol_ball_sampler, static_argnames="shape")
    points, weights = sample_fn(rng, shape, n_per_radius=None)

    assert weights.shape == (n_expected,)
    assert points.shape == (n_expected, d)
    np.testing.assert_allclose(weights.sum(), 1.0, rtol=1e-4, atol=1e-4)
    if n_0s:
      np.testing.assert_array_equal(points[-1], 0.0)

  @pytest.mark.parametrize(("target_dim", "epsilon", "sampler_fn"),
                           [(3, 1e-1, True), (5, 1e-2, False)])
  def test_otcp(
      self, rng: jax.Array, target_dim: int, epsilon: float, sampler_fn: bool
  ):
    sampler = jax.tree_util.Partial(
        conformal.sobol_ball_sampler, n_per_radius=10
    ) if sampler_fn else None
    n_samples = 32
    n_target_measure = n_samples
    model, data = get_model_and_data(n_samples=n_samples, target_dim=target_dim)
    x_trn, x_calib, x_test, y_trn, y_calib, y_test = data

    otcp = conformal.OTCP(model, sampler=sampler)
    otcp = jax.jit(
        otcp.fit_transport, static_argnames=["n_target"]
    )(x_trn, y_trn, epsilon=epsilon, n_target=n_target_measure, rng=rng)
    otcp = jax.jit(otcp.calibrate)(x_calib, y_calib)
    predict_fn = jax.jit(otcp.predict)

    calib_scores = otcp.calibration_scores
    np.testing.assert_array_equal(
        jax.jit(otcp.get_scores)(x_calib, y_calib), calib_scores
    )

    # predict backward
    preds = predict_fn(x_test[0])
    assert preds.shape == (len(otcp.target_measure), target_dim)
    preds = predict_fn(x_test)  # vectorized
    assert preds.shape == (len(x_test), len(otcp.target_measure), target_dim)

    # predict forward
    preds = predict_fn(x_test[0], y_candidates=y_test)
    assert preds.shape == (len(y_test),)
    assert jnp.isdtype(preds.dtype, jnp.bool)
    preds = predict_fn(x_test, y_candidates=y_test)  # vectorized
    assert preds.shape == (len(x_test), len(y_test))
    assert jnp.isdtype(preds.dtype, jnp.bool)
