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

"""Low level functions used within the scope of Geometric processing."""

import functools
from typing import Sequence

import jax
import jax.numpy as jnp


@functools.partial(jax.custom_jvp, nondiff_argnums=(1, 2, 4))
def logsumexp(mat, axis=None, keepdims=False, b=None, return_sign=False):
  return jax.scipy.special.logsumexp(
      mat, axis=axis, keepdims=keepdims, b=b, return_sign=return_sign)


@logsumexp.defjvp
def logsumexp_jvp(axis, keepdims, return_sign, primals, tangents):
  """Implements custom derivative rule for lse that does not blow up with -inf.

  This logsumexp implementation uses the standard jax one in forward mode but
  implements a custom rule to differentiate. Given the preference of jax for
  jvp over vjp, and the fact that this is a simple linear rule, jvp is used.
  This custom differentiation address issues when the output of lse is
  -inf (which corresponds to the case where all inputs in a slice are -inf,
  which happens typically when ``a`` or ``b`` weight vectors have zeros.)

  Although both exp(lse) and its derivative should be 0, automatic
  differentiation returns a NaN derivative because of a -inf - (-inf) operation
  appearing in the definition of centered_exp below. This is corrected in the
  implementation below.

  Args:
    axis: argument from original logsumexp
    keepdims: argument from original logsumexp
    return_sign: argument from original logsumexp
    primals: mat and b, the two arguments against which we differentiate.
    tangents: of same size as mat and b.

  Returns:
    original primal outputs + their tangent.
  """
  mat, b = primals
  tan_mat, tan_b = tangents
  lse = logsumexp(mat, axis, keepdims, b, return_sign)
  if return_sign:
    lse, sign = lse
  lse = jnp.where(jnp.isfinite(lse), lse, 0.0)
  centered_exp = jnp.exp(mat - jnp.expand_dims(lse, axis=axis))

  if b is None:
    res = jnp.sum(centered_exp * tan_mat, axis=axis, keepdims=keepdims)
  else:
    res = jnp.sum(b * centered_exp * tan_mat, axis=axis, keepdims=keepdims)
    res += jnp.sum(tan_b * centered_exp, axis=axis, keepdims=keepdims)
  if return_sign:
    return (lse, sign), (sign * res, jnp.zeros_like(sign))
  else:
    return lse, res
