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
import dataclasses
import math
from typing import Any, Callable, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import scipy as sp
from scipy.stats import qmc

from ott import utils
from ott.geometry import pointcloud
from ott.solvers import linear
from ott.solvers.linear import sinkhorn

__all__ = ["otcp", "OTCPOutput", "sample_target_measure"]


@jtu.register_dataclass
@dataclasses.dataclass
class OTCPOutput:
  """TODO."""
  model: Callable[[jnp.ndarray],
                  jnp.ndarray] = dataclasses.field(metadata={"static": True})
  is_classifier: bool = dataclasses.field(metadata={"static": True})
  out: sinkhorn.SinkhornOutput
  x_calib: jnp.ndarray
  y_calib: jnp.ndarray
  offset: jnp.ndarray = 0.0
  scale: jnp.ndarray = 1.0

  def predict(self, x: jnp.ndarray, alpha: float = 0.1) -> jnp.ndarray:
    """TODO."""
    y_hat = self.model(jnp.atleast_2d(x))  # [B, D]
    quantile = jnp.quantile(self.calib_scores, q=1 - alpha)
    if self.is_classifier:
      res = self._predict_classification(y_hat, quantile)
    else:
      res = self._predict_regression(y_hat, quantile)
    return res.squeeze(0) if x.ndim == 1 else res

  def _predict_classification(self, y_hat: jnp.ndarray, quantile: float):
    ys = jnp.eye(y_hat.shape[-1])  # candidates
    score_fn = jax.vmap(self._get_scores, in_axes=[0, None])
    score_fn = jax.vmap(score_fn, in_axes=[None, 0])
    scores = score_fn(ys, y_hat)
    return scores <= quantile

  def _predict_regression(self, y_hat: jnp.ndarray, quantile: float):
    candidates = self._transport(
        quantile * self.target, forward=False
    )  # [C, D]
    candidates = self._rescale(candidates, forward=False)
    return y_hat[:, None] + candidates[None]

  @property
  def calib_scores(self) -> jnp.ndarray:
    """Calibration scores."""
    y_hat = self.model(self.x_calib)
    return self._get_scores(self.y_calib, y_hat)

  @property
  def score_function(self) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """TODO."""
    return classification_score if self.is_classifier else regression_score

  @property
  def target(self) -> jnp.ndarray:
    """TODO."""
    return self.out.geom.y

  def _get_scores(self, y: jnp.ndarray, y_hat: jnp.ndarray) -> jnp.ndarray:
    """TODO."""
    residuals = self.score_function(jnp.atleast_2d(y), jnp.atleast_2d(y_hat))
    residuals = self._rescale(residuals, forward=True)
    scores = self._transport(residuals, forward=True)
    scores = jnp.linalg.norm(scores, axis=-1)
    return scores.squeeze(0) if y.ndim == 1 else scores

  def _transport(self, x: jnp.ndarray, *, forward: bool = True) -> jnp.ndarray:
    """TODO."""
    return self.out.to_dual_potentials().transport(x, forward=forward)

  def _rescale(self, x: jnp.ndarray, *, forward: bool) -> jnp.ndarray:
    if forward:
      return (x - self.offset) / self.scale
    return (self.scale * x) + self.offset


def otcp(
    model: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    x_trn: jnp.ndarray,
    y_trn: jnp.ndarray,
    x_calib: jnp.ndarray,
    y_calib: jnp.ndarray,
    is_classifier: bool,
    epsilon: Optional[float] = 1e-1,
    num_target: int = 8192,
    rng: Optional[jax.Array] = None,
    **kwargs: Any
) -> Callable[[jnp.ndarray, float], jnp.ndarray]:
  """TODO."""
  dim = y_trn.shape[-1]
  if is_classifier:
    score_fn, sample_method = classification_score, "random"
  else:
    score_fn, sample_method = regression_score, "sobol"

  y_hat_trn = model(x_trn)
  trn_residuals = score_fn(y_trn, y_hat_trn)

  offset = jnp.mean(trn_residuals, axis=0, keepdims=True)
  scale = jnp.linalg.norm(trn_residuals - offset, axis=-1).max()
  trn_residuals = (trn_residuals - offset) / scale

  target, weights = sample_target_measure((num_target, dim),
                                          method=sample_method,
                                          rng=rng)
  geom = pointcloud.PointCloud(trn_residuals, target, epsilon=epsilon)

  out = linear.solve(geom, b=weights, **kwargs)
  return OTCPOutput(
      model=model,
      is_classifier=is_classifier,
      out=out,
      x_calib=x_calib,
      y_calib=y_calib,
      offset=offset,
      scale=scale,
  )


def sample_target_measure(
    shape: Tuple[int, int],
    method: Literal["random", "sobol"],
    rng: Optional[jax.Array] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """TODO."""
  rng = utils.default_prng_key(rng)

  num_samples, dim = shape
  num_radii = int(math.sqrt(num_samples))
  num_sphere, num_0s = divmod(num_samples, num_radii)

  radii = jnp.linspace(
      1.0 / (num_radii + 1), num_radii / (num_radii + 1), num_radii
  )

  if method == "random":
    sphere = _random_sphere(num_sphere, d=dim, rng=rng)
    sphere = jnp.abs(sphere)
  elif method == "sobol":
    seed = jax.random.randint(rng, shape=(), minval=0, maxval=2 ** 16 - 1)
    out_struct = jax.ShapeDtypeStruct(
        shape=(num_sphere, dim), dtype=radii.dtype
    )
    sphere = jax.pure_callback(_sobol_sphere, out_struct, num_sphere, dim, seed)
  else:
    raise ValueError(method)

  points = sphere[None] * radii[:, None, None]
  points = points.reshape(-1, dim)

  weights = jnp.full(
      points.shape[0] + (num_0s > 0), fill_value=1.0 / num_samples
  )
  if num_0s:
    points = jnp.vstack([points, jnp.zeros([1, dim])])
    weights = weights.at[-1].set(num_0s / num_samples)

  return points, weights


def _sobol_sphere(n: int, d: int, seed: int) -> np.ndarray:
  n, d, seed = int(n), int(d), int(seed)
  sampler = qmc.Sobol(d=d, seed=seed, scramble=True)

  points = sampler.random_base2(m=math.ceil(math.log2(n)))[:n]
  points = sp.special.ndtri(points)
  points /= (
      np.linalg.norm(points, keepdims=True, axis=-1) +
      np.finfo(points.dtype).tiny
  )
  return points


def _random_sphere(n: int, d: int, rng: jax.Array) -> jnp.ndarray:
  points = jax.random.normal(rng, (n, d))
  return points / (
      jnp.linalg.norm(points, keepdims=True, axis=-1) +
      jnp.finfo(points.dtype).tiny
  )


def classification_score(y: jnp.ndarray, y_hat: jnp.ndarray) -> jnp.ndarray:
  """TODO."""
  return jnp.abs(y - y_hat)


def regression_score(y: jnp.ndarray, y_hat: jnp.ndarray) -> jnp.ndarray:
  """TODO."""
  return y - y_hat
