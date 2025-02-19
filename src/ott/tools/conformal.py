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

__all__ = ["otcp", "OTCPOutput", "sobol_sphere"]


@jtu.register_dataclass
@dataclasses.dataclass
class OTCPOutput:
  """TODO."""
  model: Callable[[jnp.ndarray],
                  jnp.ndarray] = dataclasses.field(metadata={"static": True})
  out: sinkhorn.SinkhornOutput
  calib_scores: jnp.ndarray
  offset: jnp.ndarray = 0.0
  scale: jnp.ndarray = 1.0

  def predict(self, x: jnp.ndarray, alpha: float = 0.1) -> jnp.ndarray:
    """TODO."""
    target = self.out.geom.y
    y_hat = self.model(jnp.atleast_2d(x))  # [B, D]
    radius = jnp.quantile(self.calib_scores, q=1.0 - alpha)
    candidates = self.transport(radius * target, forward=False)  # [C, D]
    candidates = self._rescale(candidates, forward=False)
    res = y_hat[:, None] - candidates[None]
    return res.squeeze(0) if x.ndim == 1 else res

  def get_scores(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """TODO."""
    residuals = y - self.model(x)
    residuals = self._rescale(residuals, forward=True)
    scores = self.transport(residuals, forward=True)
    return jnp.linalg.norm(scores, axis=-1)

  def transport(self, x: jnp.ndarray, *, forward: bool = True):
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
    epsilon: Optional[float] = 1e-1,
    num_target: int = 8192,
    rng: Optional[jax.Array] = None,
    **kwargs: Any
) -> Callable[[jnp.ndarray, float], jnp.ndarray]:
  """TODO."""
  trn_residuals = y_trn - model(x_trn)
  offset = jnp.mean(trn_residuals, axis=0, keepdims=True)
  scale = jnp.linalg.norm(trn_residuals - offset, axis=-1).max()

  trn_residuals = (trn_residuals - offset) / scale
  target, weights = sobol_sphere(
      num_target, dim=trn_residuals.shape[-1], rng=rng
  )
  geom = pointcloud.PointCloud(trn_residuals, target, epsilon=epsilon)

  sink_out = linear.solve(geom, b=weights, **kwargs)

  otcp_out = OTCPOutput(
      model=model,
      out=sink_out,
      calib_scores=None,
      offset=offset,
      scale=scale,
  )
  calib_scores = otcp_out.get_scores(x_calib, y_calib)
  return dataclasses.replace(otcp_out, calib_scores=calib_scores)


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
