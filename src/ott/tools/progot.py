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
  init_potentials: Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray]]


class ProgOTOutput(NamedTuple):
  """:class:`ProgOT` solver output.

  Args:
    prob: Linear problem.
    alphas: Stepsize schedule of shape ``[num_steps,]``.
    epsilons: Entropy regularization of shape ``[num_steps,]``.
    outputs: Solver outputs at every step.
    xs: Intermediate interpolations of shape ``[num_steps, n, d]``.
  """
  prob: linear_problem.LinearProblem
  alphas: jnp.ndarray
  epsilons: jnp.ndarray
  outputs: Output
  xs: Optional[jnp.ndarray] = None

  def transport(
      self,
      x: jnp.ndarray,
      max_steps: Optional[int] = None,
      return_intermediate: bool = False,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Transport points.

    Args:
      x: Source points of shape ``[n, d]`` to transport.
      max_steps: Maximum number of steps. If :obj:`None`, use :attr:`num_steps`.
      return_intermediate: Whether to return inte

    Returns:
      If ``return_intermediate = True``, return arrays of shape
      ``[max_steps + 1, n, d]`` and ``[max_steps, n, d]``, containing TODO.
      Otherwise, return arrays of shape ``[n, d]`` and ``[n, d]``.
    """

    def body_fn(x: jnp.ndarray,
                it: int) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
      alpha = self.alphas[it]
      dp = self.get_output(it).to_dual_potentials()

      t_x = dp.transport(x, forward=True)
      next_x = _interpolate(
          x=x, t_x=t_x, alpha=alpha, cost_fn=self.prob.geom.cost_fn
      )

      return next_x, (next_x, t_x)

    if max_steps is None:
      max_steps = self.num_steps
    assert (
        max_steps <= self.num_steps
    ), f"Maximum number of steps <= {self.num_steps}."

    _, (xs, ys) = jax.lax.scan(body_fn, x, xs=jnp.arange(max_steps))
    if return_intermediate:
      # also include the starting point
      return jnp.concatenate([x[None], xs]), ys
    return xs[-1], ys[-1]

  def get_output(self, step: int) -> Output:
    r"""Get the solver output at a specific step.

    Args:
      step: Iteration step in :math:`[0, num_steps)`.

    Returns:
      The output.
    """
    return jtu.tree_map(lambda x: x[step], self.outputs)

  @property
  def converged(
      self
  ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Converged flag at each step.

    If :attr`is_debiased`, return an array of shape ``[num_steps, 3]`` with
    values corresponding to ``(x, y)``, ``(x, x)`` and ``(y, y)`` problems.
    Otherwise, return an array of shape ``[num_steps,]``.
    """
    return jnp.stack(self.outputs.converged, axis=1)

  @property
  def num_iters(self) -> jnp.ndarray:
    """Number of iterations at each steps.

    If :attr`is_debiased`, return an array of shape ``[num_steps, 3]`` with
    values corresponding to ``(x, y)``, ``(x, x)`` and ``(y, y)`` problems.
    Otherwise, return an array of shape ``[num_steps,]``.
    """
    return jnp.array([
        self.get_output(it).n_iters for it in range(self.num_steps)
    ])

  @property
  def num_steps(self) -> int:
    """Number of steps."""
    return len(self.alphas)

  @property
  def is_debiased(self) -> bool:
    """Whether the solver is debiased."""
    return isinstance(self.outputs[0], sd.SinkhornDivergenceOutput)


@jtu.register_pytree_node_class
class ProgOT:
  """Progressive Entropic Optimal Transport :cite:`kassraie:24`.

  Args:
    alphas: Stepsize schedule.
    epsilons: Epsilon regularization schedule. If :obj:`None`, use the default
      epsilon at each step.
    epsilon_scales: TODO.
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
    if epsilons is not None:
      assert len(alphas) == len(
          epsilons
      ), "Epsilons have different length than alphas."
    if epsilon_scales is not None:
      assert len(alphas) == len(
          epsilon_scales
      ), "Epsilon scales have different length than alphas."

    self.alphas = alphas
    self.epsilons = epsilons
    self.epsilon_scales = epsilon_scales
    self.is_debiased = is_debiased

  def __call__(
      self,
      prob: linear_problem.LinearProblem,
      store_intermediate: bool = False,
      warm_start: bool = False,
      **kwargs: Any,
  ) -> ProgOTOutput:
    """Run the estimator.

    Args:
      prob: Linear problem.
      store_intermediate: Whether to also store the intermediate values.
      warm_start: Whether to initialize potentials from the previous step.
      kwargs: Keyword arguments for
        :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` or
        :func:`~ott.tools.sinkhorn_divergence.sinkhorn_divergence`, depending
        on :attr:`debiased`.

    Returns:
      The output.
    """

    def body_fn(
        state: ProgOTState, it: int
    ) -> Tuple[ProgOTState, Tuple[Output, float, Optional[jnp.ndarray]]]:
      alpha = self.alphas[it]
      eps = None if self.epsilons is None else self.epsilons[it]
      if self.epsilon_scales is not None:
        # use the default epsilon and scale it
        geom = pointcloud.PointCloud(state.x, y, cost_fn=cost_fn)
        eps = self.epsilon_scales[it] * geom.epsilon

      if self.is_debiased:
        assert state.init_potentials == (
            None, None
        ), "Warm start is not implemented for debiased."
        out = _sinkhorn_divergence(
            state.x, y, cost_fn=cost_fn, eps=eps, **kwargs
        )
        eps = _sink_out_from_debiased(out, idx=0).geom.epsilon
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
      next_x = _interpolate(x=state.x, t_x=t_x, alpha=alpha, cost_fn=cost_fn)

      next_init = ((1.0 - alpha) * out.f,
                   (1.0 - alpha) * out.g) if warm_start else (None, None)
      next_state = ProgOTState(x=next_x, init_potentials=next_init)

      return next_state, (out, eps, (next_x if store_intermediate else None))

    lse_mode = kwargs.get("lse_mode", True)
    num_steps = len(self.alphas)
    n, m = prob.geom.shape
    x, y, cost_fn = prob.geom.x, prob.geom.y, prob.geom.cost_fn
    _, d = x.shape

    if warm_start:
      init_potentials = (jnp.zeros(n), jnp.zeros(m)
                        ) if lse_mode else (jnp.ones(n), jnp.ones(m))
    else:
      init_potentials = (None, None)

    init_state = ProgOTState(x=x, init_potentials=init_potentials)
    _, (outputs, epsilons, xs) = jax.lax.scan(
        body_fn, init_state, xs=jnp.arange(num_steps)
    )

    if store_intermediate:
      # add the initial `x` for nicer impl. in `ProgOTOutput`
      # also we could do `xs[:-1]`, since it's not needed
      xs = jnp.concatenate([x[None], xs[:-1]], axis=0)

    return ProgOTOutput(
        prob,
        xs=xs,
        alphas=self.alphas,
        epsilons=epsilons,
        outputs=outputs,
    )

  def tree_flatten(self):  # noqa: D102
    return (self.alphas, self.epsilons), {
        "debiased": self.is_debiased,
        "epsilon_scales": self.epsilon_scales,
    }

  @classmethod
  def tree_unflatten(  # noqa: D102
      cls, aux_data: dict[str, Any], children: Any
  ) -> "ProgOT":
    alphas, epsilons = children
    return cls(alphas=alphas, epsilons=epsilons, **aux_data)


def _sinkhorn(
    x: jnp.ndarray,
    y: jnp.ndarray,
    cost_fn: costs.TICost,
    eps: Optional[float],
    init: Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray]] = (None, None),
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
  return sd.sinkhorn_divergence(
      pointcloud.PointCloud,
      x,
      y,
      cost_fn=cost_fn,
      epsilon=eps,
      share_epsilon=False,
      sinkhorn_kwargs=kwargs,
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
  """TODO."""

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
  """Get the stepsize schedule.

  Args:
    kind: Kind of the schedule.
    num_steps: Total number of steps.

  Returns:
    The schedule.
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
    raise ValueError(kind)

  return arr


def _interpolate(
    x: jnp.ndarray, t_x: jnp.ndarray, alpha: float, cost_fn: costs.TICost
) -> jnp.ndarray:
  xx, weights = jnp.stack([x, t_x]), jnp.array([1.0 - alpha, alpha])
  xx, _ = cost_fn.barycenter(weights=weights, xs=xx)
  return xx


def _sink_out_from_debiased(
    out: sd.SinkhornDivergenceOutput,
    *,
    idx: int = 0
) -> sinkhorn.SinkhornOutput:
  geom_xy = out.geoms[idx]
  prob = linear_problem.LinearProblem(geom_xy, a=out.a, b=out.b)

  return sinkhorn.SinkhornOutput(
      potentials=out.potentials[idx],
      errors=out.errors[idx],
      ot_prob=prob,
      # not needed
      reg_ot_cost=None,
      threshold=None,
      inner_iterations=None,
  )
