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

import jax
import jax.numpy as jnp

import optax
import optax.tree

__all__ = ["lbfgs"]

# see https://optax.readthedocs.io/en/stable/_collections/examples/lbfgs.html


def run_opt(
    opt: optax.GradientTransformationExtraArgs, x_init: jnp.array,
    fun: Callable, max_iter: int, tol: float
):
  """Runs an optimization algorithm on a function.

  Args:
    opt: An instance of an optax optimizer.
    x_init: Initial point to start optimization.
    fun: The function to minimize.
    max_iter: Maximum number of iterations.
    tol: Tolerance for convergence, measured as the norm of the gradient.

  Returns:
    Final optimization variable obtained after running the optimization.
  """
  value_and_grad_fun = optax.value_and_grad_from_state(fun)

  def step(carry):
    params, state = carry
    value, grad = value_and_grad_fun(params, state=state)
    updates, state = opt.update(
        grad, state, params, value=value, grad=grad, value_fn=fun
    )
    params = optax.apply_updates(params, updates)
    return params, state

  def continuing_criterion(carry):
    _, state = carry
    iter_num = optax.tree.get(state, "count")
    grad = optax.tree.get(state, "grad")
    err = optax.tree.norm(grad)
    return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

  init_carry = (x_init, opt.init(x_init))
  final_params, _ = jax.lax.while_loop(continuing_criterion, step, init_carry)
  return final_params


def lbfgs(
    fun: Callable,
    x_init: jax.Array,
    max_iter: int = 100,
    tol: float = 1e-4,
    **kwargs
) -> Tuple[jax.Array, optax.OptState]:
  """Runs optax's L-BFGS optimization on function.

  Args:
    fun: The function to minimize.
    x_init: Initial point to start optimization.
    max_iter: Maximum number of iterations.
    tol: Tolerance for convergence.
    **kwargs: Additional arguments for the L-BFGS optimizer class.

  Returns:
    Final optimization variable obtained after running L-BFGS.
  """
  opt = optax.lbfgs(**kwargs)
  return run_opt(opt, x_init, fun, max_iter=max_iter, tol=tol)
