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
from typing import Any, Literal, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.experimental import checkify

from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from ott.tools import sinkhorn_divergence as sd

__all__ = [
    "ProgOT",
    "ProgOTOutput",
    "get_epsilon_schedule",
    "get_alpha_schedule",
]

Output = Union[sinkhorn.SinkhornOutput, sd.SinkhornDivergenceOutput]


class ProgOTState(NamedTuple):
  x: jnp.ndarray
  init_potentials: Optional[Tuple[jnp.ndarray, jnp.ndarray]]


class ProgOTOutput(NamedTuple):
  """Output of the :class:`ProgOT` solver.

  Args:
    prob: Linear problem.
    alphas: Stepsize schedule of shape ``[num_steps,]``.
    epsilons: Entropy regularizations of shape ``[num_steps,]``.
    outputs: OT solver outputs for every step, a struct of arrays.
    xs: Intermediate interpolations of shape ``[num_steps, n, d]``, if present.
  """
  prob: linear_problem.LinearProblem
  alphas: jnp.ndarray
  epsilons: jnp.ndarray
  outputs: Output
  xs: Optional[jnp.ndarray] = None

  def transport(
      self,
      x: jnp.ndarray,
      num_steps: Optional[int] = None,
      return_intermediate: bool = False,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Transport points.

    Args:
      x: Array of shape ``[n, d]`` to transport.
      num_steps: Number of steps. If :obj:`None`, use the full number of steps.
      return_intermediate: Whether to return intermediate values.

    Returns:
      - If ``return_intermediate = True``, return arrays of shape
        ``[num_steps, n, d]`` and ``[num_steps, n, d]`` corresponding to the
        interpolations and push-forwards after each step, respectively.
      - Otherwise, return arrays of shape ``[n, d]`` and ``[n, d]``
        corresponding to the last interpolation and push-forward, respectively.
    """

    def body_fn(
        xy: Tuple[jnp.ndarray, Optional[jnp.ndarray]], it: int
    ) -> Tuple[Tuple[jnp.ndarray, Optional[jnp.ndarray]], Tuple[
        Optional[jnp.ndarray], Optional[jnp.ndarray]]]:
      x, _ = xy
      alpha = self.alphas[it]
      dp = self.get_output(it).to_dual_potentials()

      t_x = dp.transport(x, forward=True)
      next_x = (1.0 - alpha) * x + alpha * t_x

      if return_intermediate:
        return (next_x, None), (next_x, t_x)
      return (next_x, t_x), (None, None)

    if num_steps is None:
      num_steps = self.num_steps
    else:
      assert (
          0 < num_steps <= self.num_steps
      ), f"Maximum number of steps must be in (0, {self.num_steps}], " \
         f"found {num_steps}."

    state = (x, None) if return_intermediate else (x, jnp.empty_like(x))
    xy, xs_ys = jax.lax.scan(body_fn, state, xs=jnp.arange(num_steps))
    return xs_ys if return_intermediate else xy

  def get_output(self, step: int) -> Output:
    r"""Get the OT solver output at a given step.

    Args:
      step: Iteration step in :math:`[0, \text{num_steps})`.

    Returns:
      The OT solver output at a ``step``.
    """
    return jtu.tree_map(lambda x: x[step], self.outputs)

  @property
  def converged(
      self
  ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Convergence at each step.

    - If :attr:`is_debiased`, return an array of shape ``[num_steps, 3]`` with
      values corresponding to the convergence of the  ``(x, y)``, ``(x, x)`` and
      ``(y, y)`` problems.
    - Otherwise, return an array of shape ``[num_steps,]``.
    """
    return jnp.stack(self.outputs.converged, axis=-1)

  @property
  def num_iters(self) -> jnp.ndarray:
    """Number of Sinkhorn iterations within each step.

    - If :attr:`is_debiased`, return an array of shape ``[num_steps, 3]`` with
      values corresponding to the number of iterations for the ``(x, y)``,
      ``(x, x)`` and ``(y, y)`` problems.
    - Otherwise, return an array of shape ``[num_steps,]``.
    """
    return jnp.array([
        self.get_output(it).n_iters for it in range(self.num_steps)
    ])

  @property
  def num_steps(self) -> int:
    """Number of :class:`ProgOT` steps."""
    return len(self.alphas)

  @property
  def is_debiased(self) -> bool:
    """Whether the OT solver is debiased."""
    return isinstance(self.outputs[0], sd.SinkhornDivergenceOutput)


@jtu.register_pytree_node_class
class ProgOT:
  """Progressive Entropic Optimal Transport solver :cite:`kassraie:24`.

  Args:
    alphas: Stepsize schedule of shape ``[num_steps,]``.
    epsilons: Epsilon regularization schedule of shape ``[num_steps,]``.
      If :obj:`None`, use the default epsilon at each step.
    epsilon_scales: Scale for the default epsilon of shape ``[num_steps,]``.
      If :obj:`None`, don't scale the epsilons. Note that only one of
      ``epsilons`` and ``epsilon_scales`` can be passed.
    is_debiased: Whether to use
      :func:`~ott.tools.sinkhorn_divergence.sinkhorn_divergence` or
      :class:`~ott.solvers.linear.sinkhorn.Sinkhorn`.
  """

  def __init__(
      self,
      alphas: jnp.ndarray,
      *,
      epsilons: Optional[jnp.ndarray] = None,
      epsilon_scales: Optional[jnp.ndarray] = None,
      is_debiased: bool = False,
  ):
    if epsilons is not None and epsilon_scales is not None:
      raise ValueError(
          "Please pass either `epsilons` or `epsilon_scales`, not both."
      )
    if epsilons is not None:
      assert len(alphas) == len(
          epsilons
      ), "Epsilons have different length than alphas."
    if epsilon_scales is not None:
      assert len(alphas) == len(
          epsilon_scales
      ), "Epsilon scales have different length than alphas."

    checkify.check(
        jnp.all((alphas >= 0.0) & (alphas <= 1.0)),
        "Alphas must be a sequence with values between zero and one."
    )

    self.alphas = alphas
    self.epsilons = epsilons
    self.epsilon_scales = epsilon_scales
    self.is_debiased = is_debiased

  def __call__(
      self,
      prob: linear_problem.LinearProblem,
      warm_start: bool = False,
      **kwargs: Any,
  ) -> ProgOTOutput:
    """Run the solver.

    Args:
      prob: Linear problem.
      warm_start: Whether to initialize potentials from the previous step.
      kwargs: Keyword arguments for
        :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` or
        :func:`~ott.tools.sinkhorn_divergence.sinkhorn_divergence`, depending
        on :attr:`is_debiased`.

    Returns:
      The solver output.
    """

    def body_fn(state: ProgOTState,
                it: int) -> Tuple[ProgOTState, Tuple[Output, float]]:
      alpha = self.alphas[it]
      eps = None if self.epsilons is None else self.epsilons[it]
      if self.epsilon_scales is not None:
        # use the default epsilon and scale it
        geom = pointcloud.PointCloud(state.x, y, cost_fn=cost_fn)
        eps = self.epsilon_scales[it] * geom.epsilon

      if self.is_debiased:
        assert state.init_potentials is None, \
          "Warm start is not implemented for debiased."
        out = _sinkhorn_divergence(
            state.x, y, cost_fn=cost_fn, eps=eps, **kwargs
        )
        eps = out.geoms[0].epsilon
      else:
        out = _sinkhorn(
            state.x,
            y,
            cost_fn=cost_fn,
            eps=eps,
            init=state.init_potentials,
            **kwargs
        )
        eps = out.geom.epsilon

      t_x = out.to_dual_potentials().transport(state.x, forward=True)
      next_x = (1.0 - alpha) * x + alpha * t_x

      next_init = ((1.0 - alpha) * out.f,
                   (1.0 - alpha) * out.g) if warm_start else None
      next_state = ProgOTState(x=next_x, init_potentials=next_init)

      return next_state, (out, eps)

    lse_mode = kwargs.get("lse_mode", True)
    num_steps = len(self.alphas)
    n, m = prob.geom.shape
    x, y, cost_fn = prob.geom.x, prob.geom.y, prob.geom.cost_fn
    _, d = x.shape

    if warm_start:
      init_potentials = (jnp.zeros(n), jnp.zeros(m)
                        ) if lse_mode else (jnp.ones(n), jnp.ones(m))
    else:
      init_potentials = None

    init_state = ProgOTState(x=x, init_potentials=init_potentials)
    _, (outputs, epsilons) = jax.lax.scan(
        body_fn, init_state, xs=jnp.arange(num_steps)
    )

    return ProgOTOutput(
        prob,
        alphas=self.alphas,
        epsilons=epsilons,
        outputs=outputs,
    )

  def tree_flatten(self):  # noqa: D102
    return (self.alphas, self.epsilons, self.epsilon_scales), {
        "is_debiased": self.is_debiased,
    }

  @classmethod
  def tree_unflatten(  # noqa: D102
      cls, aux_data: dict[str, Any], children: Any
  ) -> "ProgOT":
    alphas, epsilons, epsilon_scales = children
    return cls(
        alphas=alphas,
        epsilons=epsilons,
        epsilon_scales=epsilon_scales,
        **aux_data
    )


def get_epsilon_schedule(
    geom: pointcloud.PointCloud,
    *,
    alphas: jnp.ndarray,
    epsilon_scales: jnp.ndarray,
    y_eval: jnp.ndarray,
    start_epsilon_scale: float = 1.0,
    **kwargs: Any,
) -> jnp.ndarray:
  """Get the epsilon regularization schedule.

  See Algorithm 4 in :cite:`kassraie:24` for more information.

  Args:
    geom: Point cloud geometry.
    alphas: Stepsize schedule of shape ``[num_steps,]``.
    epsilon_scales: Array of shape ``[num_scales,]`` from which to select
      the best scale of the default epsilon in the ``(y, y)`` point cloud.
    y_eval: Array of shape ``[k, d]`` from the target distribution used to
      compute the error.
    start_epsilon_scale: Constant by which to scale the initial epsilon.
    kwargs: Keyword arguments for
      :class:`~ott.solvers.linear.sinkhorn.Sinkhorn`.

  Returns:
    The epsilon regularization schedule of shape ``[num_steps,]``.
  """

  def error(epsilon_scale: float) -> float:
    epsilon = epsilon_scale * geom_end.epsilon

    out = _sinkhorn(y, y, cost_fn=cost_fn, eps=epsilon, **kwargs)
    dp = out.to_dual_potentials()
    y_hat = dp.transport(y_eval, forward=True)

    return jnp.linalg.norm(y_eval - y_hat)

  y, cost_fn = geom.y, geom.cost_fn

  start_eps = start_epsilon_scale * geom.epsilon

  geom_end = pointcloud.PointCloud(y, y, cost_fn=cost_fn)
  errors = jax.vmap(error)(epsilon_scales)
  end_epsilon = epsilon_scales[jnp.argmin(errors)] * geom_end.epsilon

  mod_alpha = jnp.concatenate([jnp.array([0.0]), alphas])
  no_ending_1 = mod_alpha[-1] != 1.0  # e.g. the exp schedule
  # TODO(michalk8): not jittable
  if no_ending_1:
    mod_alpha = jnp.concatenate([mod_alpha, jnp.array([1.0])])

  tk = 1.0 - jnp.cumprod(1.0 - mod_alpha)
  epsilons = end_epsilon * tk + (1.0 - tk) * start_eps

  epsilons = epsilons[:-1]
  if no_ending_1:
    epsilons = epsilons[:-1]

  return epsilons


def get_alpha_schedule(
    kind: Literal["lin", "exp", "quad"], *, num_steps: int
) -> jnp.ndarray:
  """Get the step size schedule.

  Convenience wrapper to get a sequence of ``num_steps`` timestamps between
  0 and 1, distributed according to the ``kind`` option below.
  See Section 4 in :cite:`kassraie:24` for more details.

  Args:
    kind: The schedule to create:

      - ``'lin'`` - constant-speed schedule.
      - ``'exp'`` - decelerating schedule.
      - ``'quad'`` - accelerating schedule.
    num_steps: Total number of steps.

  Returns:
    The stepsize schedule, array of shape ``[num_steps,]``.
  """
  if kind == "lin":
    arr = jnp.arange(2, num_steps + 2)
    arr = 1.0 / (num_steps - arr + 2)
  elif kind == "exp":
    arr = jnp.full(num_steps, fill_value=1.0 / jnp.e)
  elif kind == "quad":
    arr = jnp.arange(2, num_steps + 2)
    arr = (2.0 * arr - 1.0) / ((num_steps + 1) ** 2 - (arr - 1) ** 2)
  else:
    raise ValueError(f"Invalid stepsize schedule `{kind}`.")

  return arr


def _sinkhorn(
    x: jnp.ndarray,
    y: jnp.ndarray,
    cost_fn: costs.TICost,
    eps: Optional[float],
    init: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    **kwargs: Any,
) -> sinkhorn.SinkhornOutput:
  geom = pointcloud.PointCloud(x, y, cost_fn=cost_fn, epsilon=eps)
  prob = linear_problem.LinearProblem(geom)
  solver = sinkhorn.Sinkhorn(**kwargs)
  return solver(prob, init=init)


def _sinkhorn_divergence(
    x: jnp.ndarray,
    y: jnp.ndarray,
    cost_fn: costs.TICost,
    eps: Optional[float],
    **kwargs: Any,
) -> sd.SinkhornDivergenceOutput:
  _, out = sd.sinkhorn_divergence(
      pointcloud.PointCloud,
      x,
      y,
      cost_fn=cost_fn,
      epsilon=eps,
      share_epsilon=False,
      solve_kwargs=kwargs,
  )
  return out
