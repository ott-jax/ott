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
import operator
from typing import Any, Callable, Optional, Tuple

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

ScoreFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


@jtu.register_dataclass
@dataclasses.dataclass
class OTCPOutput:
  """Optimal transport conformal prediction output.

  Args:
    model: Fitted model.
    out: Sinkhorn output.
    nonconformity_fn: Multivariate nonconformity score function with a signature
      ``(target, prediction) -> score``.
    x_calib: Calibration features of shape ``[n_calib, dim_x]``.
    y_calib: Calibration targets of shape ``[n_calib, dim_y]``.
    offset: Offset used when re-scaling the data.
    scale: Scale when re-scaling the data.
  """
  model: Callable[[jnp.ndarray],
                  jnp.ndarray] = dataclasses.field(metadata={"static": True})
  out: sinkhorn.SinkhornOutput
  nonconformity_fn: ScoreFn = dataclasses.field(metadata={"static": True})
  x_calib: jnp.ndarray
  y_calib: jnp.ndarray
  offset: jnp.ndarray = 0.0
  scale: jnp.ndarray = 1.0

  def predict(
      self,
      x: jnp.ndarray,
      y_candidates: Optional[jnp.ndarray] = None,
      alpha: float = 0.1
  ) -> jnp.ndarray:
    """Conformalize the model's prediction.

    Args:
      x: Features of shape ``[..., dim_x]``.
      y_candidates: Candidate targets of shape ``[m, dim_y]``.
      alpha: Miscoverage level.

    Returns:
      If ``y_candidates = None``, return an array of shape
      ``[..., n_target, dim_y]``, else a boolean array of shape ``[..., m]``.
    """
    assert x.ndim in (1, 2), x.shape
    y_hat = self.model(jnp.atleast_2d(x))
    quantile = jnp.quantile(self.calib_scores, q=1 - alpha)
    if y_candidates is None:
      res = self._predict_backward(y_hat, quantile=quantile)
    else:
      res = self._predict_forward(y_hat, y_candidates, quantile=quantile)
    return res.squeeze(0) if x.ndim == 1 else res

  def _predict_backward(
      self, y_hat: jnp.ndarray, *, quantile: float
  ) -> jnp.ndarray:
    candidates = self._transport(quantile * self.target, forward=False)
    candidates = self._rescale(candidates, forward=False)
    return y_hat[:, None] + candidates[None]

  def _predict_forward(
      self, y_hat: jnp.ndarray, y_candidates: jnp.ndarray, *, quantile: float
  ) -> jnp.ndarray:
    assert y_candidates.ndim == 2, y_candidates.shape
    score_fn = jax.vmap(
        jax.vmap(self._get_scores, in_axes=[0, None]), in_axes=[None, 0]
    )
    scores = score_fn(y_candidates, y_hat)
    return scores <= quantile

  def get_scores(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Compute nonconformity scores.

    Args:
      x: Features of shape ``[n, dim_x]``.
      y: Targets of shape ``[n, dim_y]``.

    Returns:
      Nonconformity scores of shape ``[n,]``.
    """
    return self._get_scores(y, self.model(x))

  def _get_scores(self, y: jnp.ndarray, y_hat: jnp.ndarray) -> jnp.ndarray:
    scores = self.nonconformity_fn(jnp.atleast_2d(y), jnp.atleast_2d(y_hat))
    scores = self._rescale(scores, forward=True)
    scores = self._transport(scores, forward=True)
    scores = jnp.linalg.norm(scores, axis=-1)
    return scores.squeeze(0) if y.ndim == 1 else scores

  def _transport(self, x: jnp.ndarray, *, forward: bool = True) -> jnp.ndarray:
    return self.out.to_dual_potentials().transport(x, forward=forward)

  def _rescale(self, x: jnp.ndarray, *, forward: bool) -> jnp.ndarray:
    if forward:
      return (x - self.offset) / self.scale
    return (self.scale * x) + self.offset

  @property
  def calibration_scores(self) -> jnp.ndarray:
    """Nonconformity calibration scores of shape ``[n_calib,]``."""
    return self.get_scores(self.x_calib, self.y_calib)

  @property
  def target_measure(self) -> jnp.ndarray:
    """Target measure of shape ``[n_target, dim_y]``."""
    return self.out.geom.y


def otcp(
    model: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    x_trn: jnp.ndarray,
    y_trn: jnp.ndarray,
    x_calib: jnp.ndarray,
    y_calib: jnp.ndarray,
    nonconformity_fn: ScoreFn = operator.sub,
    epsilon: Optional[float] = 1e-1,
    n_target: int = 8192,
    rng: Optional[jax.Array] = None,
    **kwargs: Any,
) -> OTCPOutput:
  """Multivariate optimal transport conformal prediction :cite:`klein:25`.

  Args:
    model: Fitted model.
    x_trn: Features of shape ``[n, dim_x]`` to fit the transport map.
    y_trn: Targets of shape ``[n, dim_y]`` to fit the transport map.
    x_calib: Features of shape ``[n_calib, dim_x]`` to compute
      the calibration scores.
    y_calib: Targets of shape ``[n_calib, dim_y]`` to compute
      the calibration scores.
    nonconformity_fn: Multivariate nonconformity score function with a signature
      ``(target, prediction) -> score``.
    epsilon: Epsilon regularization
    n_target: Number of points when :func:`sampling <sample_target_measure>`
      the target measure.
    rng: Random number generator.
    kwargs: Keyword arguments for :func:`~ott.solvers.linear.solve`.

  Returns:
    Optimal transport conformal prediction output.
  """
  assert y_trn.ndim == 2, y_trn.shape
  dim = y_trn.shape[-1]

  y_hat_trn = model(x_trn)
  scores = nonconformity_fn(y_trn, y_hat_trn)
  offset = jnp.mean(scores, axis=0, keepdims=True)
  scale = jnp.linalg.norm(scores - offset, axis=-1).max()
  scores = (scores - offset) / scale

  target, weights = sample_target_measure((n_target, dim), rng=rng)
  geom = pointcloud.PointCloud(scores, target, epsilon=epsilon)
  out = linear.solve(geom, b=weights, **kwargs)

  return OTCPOutput(
      model=model,
      out=out,
      nonconformity_fn=nonconformity_fn,
      x_calib=x_calib,
      y_calib=y_calib,
      offset=offset,
      scale=scale,
  )


def sample_target_measure(
    shape: Tuple[int, int],
    rng: Optional[jax.Array] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Sample target measure for :func:`otcp`.

  Args:
    shape: Tuple of ``[n_samples, dim]``.
    rng: Random number generator.

  Returns:
    Points of shape ``[n_S * n_R, dim]`` and weights ``[n_S * n_R,]``.
  """
  rng = utils.default_prng_key(rng)

  n_samples, dim = shape
  n_radii = math.ceil(math.sqrt(n_samples))
  n_sphere, n_0s = divmod(n_samples, n_radii)

  radii = jnp.linspace(1.0 / (n_radii + 1), n_radii / (n_radii + 1), n_radii)

  seed = jax.random.randint(rng, shape=(), minval=0, maxval=2 ** 16 - 1)
  out_struct = jax.ShapeDtypeStruct(shape=(n_sphere, dim), dtype=radii.dtype)
  sphere = jax.pure_callback(_sobol_sphere, out_struct, n_sphere, dim, seed)

  points = sphere[None] * radii[:, None, None]
  points = points.reshape(-1, dim)

  weights = jnp.full(points.shape[0] + (n_0s > 0), fill_value=1.0 / n_samples)
  if n_0s:
    points = jnp.vstack([points, jnp.zeros([1, dim])])
    weights = weights.at[-1].set(n_0s / n_samples)
  return points, weights


def _sobol_sphere(n: int, d: int, seed: int) -> np.ndarray:
  # convert because usually called from the `pure_callback`
  n, d, seed = int(n), int(d), int(seed)
  sampler = qmc.Sobol(d=d, seed=seed, scramble=True)
  points = sampler.random_base2(m=math.ceil(math.log2(n)))[:n]
  points = sp.special.ndtri(points)
  return points / (
      np.linalg.norm(points, keepdims=True, axis=-1) +
      np.finfo(points.dtype).tiny
  )
