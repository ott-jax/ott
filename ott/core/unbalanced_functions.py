# coding=utf-8
# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Functions useful to define unbalanced OT problems."""

import jax.numpy as jnp


def phi_star(h: jnp.ndarray, rho: float) -> jnp.ndarray:
  """Legendre transform of KL, https://arxiv.org/pdf/1910.12958.pdf p.9."""
  return rho * (jnp.exp(h / rho) - 1)


# TODO(cuturi): use jax.grad directly.
def derivative_phi_star(f: jnp.ndarray, rho: float) -> jnp.ndarray:
  """Derivative of Legendre transform of phi_starKL, see phi_star."""
  return jnp.exp(f / rho)


def grad_of_marginal_fit(c, h, tau, epsilon):
  """Computes grad of terms linked to marginals in objective.

  Computes gradient w.r.t. f ( or g) of terms in
  https://arxiv.org/pdf/1910.12958.pdf, left-hand-side of Eq. 15
  (terms involving phi_star)

  Args:
    c: jnp.ndarray, first target marginal (either a or b in practice)
    h: jnp.ndarray, potential (either f or g in practice)
    tau: float, strength (in ]0,1]) of regularizer w.r.t. marginal
    epsilon: regularization

  Returns:
    a vector of the same size as c or h
  """
  if tau == 1.0:
    return c
  else:
    rho = epsilon * tau / (1 - tau)
    return jnp.where(c > 0, c * derivative_phi_star(-h, rho), 0.0)


def second_derivative_phi_star(f: jnp.ndarray, rho: float) -> jnp.ndarray:
  """Second Derivative of Legendre transform of KL, see phi_star."""
  return jnp.exp(f / rho) / rho


def diag_jacobian_of_marginal_fit(c, h, tau, epsilon, derivative):
  """Computes grad of terms linked to marginals in objective.

  Computes second derivative w.r.t. f ( or g) of terms in
  https://arxiv.org/pdf/1910.12958.pdf, left-hand-side of Eq. 32
  (terms involving phi_star)

  Args:
    c: jnp.ndarray, first target marginal (either a or b in practice)
    h: jnp.ndarray, potential (either f or g in practice)
    tau: float, strength (in ]0,1]) of regularizer w.r.t. marginal
    epsilon: regularization
    derivative: Callable

  Returns:
    a vector of the same size as c or h.
  """
  if tau == 1.0:
    return 0
  else:
    rho = epsilon * tau / (1 - tau)
    # here no minus sign because we are taking derivative w.r.t -h
    return jnp.where(c > 0,
                     c * second_derivative_phi_star(-h, rho) * derivative(
                         c * derivative_phi_star(-h, rho)),
                     0.0)
