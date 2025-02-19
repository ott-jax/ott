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
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp
from scipy.stats import qmc

from ott import utils
from ott.geometry import pointcloud
from ott.solvers import linear

__all__ = ["otcp", "sobol_sphere"]


def otcp(
    model: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    x_trn: jnp.ndarray,
    y_trn: jnp.ndarray,
    x_calib: jnp.ndarray,
    y_calib: jnp.ndarray,
    epsilon: Optional[float] = 1e-1,
    num_target: int = 8192,
    rng: Optional[jax.Array] = None,
    **kwargs: Any
) -> Callable[[jnp.ndarray, float], jnp.ndarray]:
  """TODO."""

  def get_residuals(y: jnp.ndarray, y_hat: jnp.ndarray) -> jnp.ndarray:
    return y - y_hat

  def rescale(x: jnp.ndarray, *, forward: bool):
    offset = jnp.mean(resid_trn, axis=0, keepdims=True)
    scale = jnp.linalg.norm(resid_trn - offset, axis=-1).max()
    return ((x - offset) / scale) if forward else ((x * scale) + offset)

  def conformalize(x_pred: jnp.ndarray, alpha: float = 0.1) -> jnp.ndarray:
    """TODO."""
    assert x_pred.ndim == 1, x_pred.shape
    x_pred = jnp.atleast_2d(x_pred)
    y_hat = model(x_pred)

    radius_alpha = jnp.quantile(scores_calib, q=1.0 - alpha)
    candidates = tmap.transport(target * radius_alpha, forward=False)
    candidates = rescale(candidates, forward=False)
    return y_hat - candidates

  resid_trn = get_residuals(y_trn, model(x_trn))
  resid_trn = rescale(resid_trn, forward=True)

  target, weights = sobol_sphere(num_target, dim=resid_trn.shape[-1], rng=rng)

  geom = pointcloud.PointCloud(resid_trn, target, epsilon=epsilon)
  tmap = linear.solve(geom, b=weights, **kwargs).to_dual_potentials()

  resid_calib = get_residuals(y_calib, model(x_calib))
  resid_calib = rescale(resid_calib, forward=True)
  scores_calib = tmap.transport(resid_calib, forward=True)
  scores_calib = jnp.linalg.norm(scores_calib, axis=-1)

  return conformalize


def sobol_sphere(
    num_samples: int,
    dim: int,
    rng: Optional[jax.Array] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """TODO."""

  def sample(n: jnp.ndarray, d: jnp.ndarray, seed: jnp.ndarray) -> np.ndarray:
    n, d, seed = n.item(), d.item(), seed.item()
    sampler = qmc.Sobol(d=d, seed=seed, scramble=True)
    samples = sampler.random_base2(m=math.ceil(math.log2(n)))[:n]
    theta = sp.special.ndtri(samples)
    eps = np.finfo(theta.dtype).tiny
    theta /= (np.linalg.norm(theta, keepdims=True, axis=-1) + eps)
    return theta.astype(radii.dtype)

  rng = utils.default_prng_key(rng)
  seed = jax.random.randint(rng, shape=(), minval=0, maxval=2 ** 16 - 1)

  num_radii = int(math.sqrt(num_samples))
  num_sphere, num_0s = divmod(num_samples, num_radii)

  radii = jnp.linspace(
      1.0 / (num_radii + 1), num_radii / (num_radii + 1), num_radii
  )
  out_struct = jax.ShapeDtypeStruct(shape=(num_sphere, dim), dtype=radii.dtype)
  points_sphere = jax.pure_callback(sample, out_struct, num_sphere, dim, seed)

  points = points_sphere[None] * radii[:, None, None]
  points = points.reshape(-1, dim)

  weights = jnp.full(
      points.shape[0] + (num_0s > 0), fill_value=1.0 / num_samples
  )
  if num_0s:
    points = jnp.vstack([points, jnp.zeros([1, dim])])
    weights = weights.at[-1].set(num_0s / num_samples)

  return points, weights
