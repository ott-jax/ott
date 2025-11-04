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
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

from ott.math import _lbfgs as lbfgs

__all__ = ["legendre"]


def legendre(
    fun: Callable[[jnp.ndarray], jnp.ndarray],
    **kwargs: Any,
) -> Callable[[jnp.ndarray, Optional[jnp.ndarray], Any], jnp.ndarray]:
  """Legendre (Fenchel) transform of a function.

  The solution is computed numerically using L-BFGS.

  Args:
    fun: A function to be transformed, must be convex for the transform
      to be properly defined.
    kwargs: Keyword arguments for :func:`~ott.math.lbfgs`, e.g. maximal
      iterations ``max_iters``, convergence tolerance ``tol`` or
      :func:`optax.lbfgs` arguments.

  Returns:
    A function that computes numerically the Legendre transform of
    the ``fun`` at a given point.
  """

  def fun_star(
      x: jnp.ndarray,
      x_init: Optional[jnp.ndarray] = None,
  ) -> float:
    """Runs optimization to compute the Legendre transform of ``fun`` at ``x``.

    Args:
      x: Array of shape ``[d,]`` where to evaluate the function.
      x_init: Initialization for optimization, of the same size of ``x``.
        If :obj:`None`, use ``x``.

    Returns:
        The Legendre transform of the ``fun`` evaluated at ``x``.
    """
    x_init = x if x_init is None else x_init

    def mod_fun(z: jnp.ndarray) -> float:
      """Conjugate maximizes <x,z> - fun(z), here minimize fun(z) - <x,z>."""
      return fun(z) - jnp.dot(x, z)

    z, _ = lbfgs.lbfgs(fun=mod_fun, x_init=x_init, **kwargs)
    # Flip sign again to revert to maximization convention, stop gradient.
    return -mod_fun(jax.lax.stop_gradient(z))

  return fun_star
