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
    n_targets: int,
    random_state: int = 0,
) -> Tuple[Callable[[jnp.ndarray], jnp.ndarray], Tuple[jnp.ndarray, ...]]:
  x, y = datasets.make_regression(
      n_samples=n_samples,
      n_features=5,
      n_targets=n_targets,
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

  @pytest.mark.parametrize(("n_targets", "epsilon"), [(3, 1e-2), (5, 1e-3)])
  def test_otcp(self, rng: jax.Array, n_targets: int, epsilon: float):
    n_samples = 128
    n_target_measure = n_samples
    model, data = get_model_and_data(n_samples=n_samples, n_targets=n_targets)
    x_trn, x_calib, x_test, y_trn, y_calib, y_test = data

    otcp_fn = jax.jit(conformal.otcp, static_argnames=["model", "n_target"])
    otcp_output: conformal.OTCPOutput = otcp_fn(
        model,
        x_trn=x_trn,
        y_trn=y_trn,
        x_calib=x_calib,
        y_calib=y_calib,
        epsilon=epsilon,
        n_target=n_target_measure,
        rng=rng,
    )
    predict_fn = jax.jit(otcp_output.predict)

    calib_scores = otcp_output.calibration_scores
    np.testing.assert_array_equal(
        otcp_output.get_scores(x_calib, y_calib), calib_scores
    )

    # predict backward
    preds = predict_fn(x_test[0])
    assert preds.shape == (len(otcp_output.target_measure), n_targets)
    preds = predict_fn(x_test)  # vectorized
    assert preds.shape == (
        len(x_test), len(otcp_output.target_measure), n_targets
    )

    # predict forward
    preds = predict_fn(x_test[0], y_candidates=y_test)
    assert preds.shape == (len(y_test),)
    assert jnp.isdtype(preds.dtype, jnp.bool)
    preds = predict_fn(x_test, y_candidates=y_test)  # vectorized
    assert preds.shape == (len(x_test), len(y_test))
    assert jnp.isdtype(preds.dtype, jnp.bool)
