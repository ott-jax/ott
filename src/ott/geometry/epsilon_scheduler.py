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
from typing import Any, Optional

import jax
import jax.numpy as jnp

__all__ = ["Epsilon"]


@jax.tree_util.register_pytree_node_class
class Epsilon:
  """Scheduler class for the regularization parameter epsilon.

  An epsilon scheduler outputs a regularization strength, to be used by in a
  Sinkhorn-type algorithm, at any iteration count. That value is either the
  final, targeted regularization, or one that is larger, obtained by
  geometric decay of an initial value that is larger than the intended target.
  Concretely, the value returned by such a scheduler will consider first
  the max between ``target`` and ``init * target * decay ** iteration``.
  If the ``scale_epsilon`` parameter is provided, that value is used to
  multiply the max computed previously by ``scale_epsilon``.

  Args:
    target: the epsilon regularizer that is targeted.
      If ``None``, use :math:`0.05`.
    scale_epsilon: if passed, used to multiply the regularizer, to rescale it.
      If ``None``, use :math:`1`.
    init: initial value when using epsilon scheduling, understood as multiple
      of target value. if passed, ``int * decay ** iteration`` will be used
      to rescale target.
    decay: geometric decay factor, :math:`<1`.
  """

  def __init__(
      self,
      target: Optional[float] = None,
      scale_epsilon: Optional[float] = None,
      init: float = 1.0,
      decay: float = 1.0
  ):
    self._target_init = target
    self._scale_epsilon = scale_epsilon
    self._init = init
    self._decay = decay

  @property
  def target(self) -> float:
    """Return the final regularizer value of scheduler."""
    target = 5e-2 if self._target_init is None else self._target_init
    scale = 1.0 if self._scale_epsilon is None else self._scale_epsilon
    return scale * target

  def at(self, iteration: Optional[int] = 1) -> float:
    """Return (intermediate) regularizer value at a given iteration."""
    if iteration is None:
      return self.target
    # check the decay is smaller than 1.0.
    decay = jnp.minimum(self._decay, 1.0)
    # the multiple is either 1.0 or a larger init value that is decayed.
    multiple = jnp.maximum(self._init * (decay ** iteration), 1.0)
    return multiple * self.target

  def done(self, eps: float) -> bool:
    """Return whether the scheduler is done at a given value."""
    return eps == self.target

  def done_at(self, iteration: Optional[int]) -> bool:
    """Return whether the scheduler is done at a given iteration."""
    return self.done(self.at(iteration))

  def set(self, **kwargs: Any) -> "Epsilon":
    """Return a copy of self, with potential overwrites."""
    kwargs = {
        "target": self._target_init,
        "scale_epsilon": self._scale_epsilon,
        "init": self._init,
        "decay": self._decay,
        **kwargs
    }
    return Epsilon(**kwargs)

  def tree_flatten(self):  # noqa: D102
    return (
        self._target_init, self._scale_epsilon, self._init, self._decay
    ), None

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    del aux_data
    return cls(*children)
