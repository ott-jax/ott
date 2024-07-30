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
from typing import Any, Callable, Literal, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem, potentials
from ott.solvers.linear import sinkhorn
from ott.tools import sinkhorn_divergence as sd

# TODO(michalk8): simplify
Output = Union[sinkhorn.SinkhornOutput, sd.SinkhornDivergenceOutput]
Initializer = tuple[Optional[jnp.ndarray], Optional[jnp.ndarray]]
SolveFn = Callable[
    [
        jnp.ndarray,
        jnp.ndarray,
        costs.TICost,
        Optional[float],
        Optional[float],
        Initializer,
        Optional[float],
    ],
    Output,
]

__all__ = [
    "ProgOT",
    "ProgOTState",
    "ProgOTOutput",
    "solve_sinkhorn",
    "solve_sinkdiv",
    "get_epsilon_schedule",
    "get_alpha_schedule",
]


class ProgOTState(NamedTuple):
  x: jnp.ndarray  # [n, d]
  xs: jnp.ndarray  # [k, n, d]
  epsilons: jnp.ndarray  # [k,]
  alphas: jnp.ndarray  # [k,]
  init_potentials: Initializer

  def set(self, **kwargs: Any) -> "ProgOTState":
    return self._replace(**kwargs)


# TODO(michalk8): docs
class ProgOTOutput(NamedTuple):
  prob: linear_problem.LinearProblem
  alphas: jnp.ndarray  # [k,]
  epsilons: jnp.ndarray  # [k,]
  outputs: Output  # [k, Output]
  xs: Optional[jnp.ndarray] = None  # [k, n, d]

  def transport(
      self,
      x: jnp.ndarray,
      return_all: bool = False,
      max_steps: Optional[int] = None,
  ) -> tuple[jnp.ndarray, jnp.ndarray]:

    def body_fn(x: jnp.ndarray,
                it: int) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
      alpha = self.alphas[it]
      dp = self.get_entropic_map(it)

      t_x = dp.transport(x, forward=True)
      next_x = _interpolate(x=x, t_x=t_x, alpha=alpha, cost_fn=self.cost_fn)

      return next_x, (next_x, t_x)

    if max_steps is None:
      max_steps = self.num_steps
    assert (
        max_steps <= self.num_steps
    ), f"Maximum number of steps <= {self.num_steps}."

    _, (xs, ys) = jax.lax.scan(body_fn, x, xs=jnp.arange(max_steps))
    if return_all:
      # also include the starting point
      return jnp.concatenate([x[None], xs]), ys
    return xs[-1], ys[-1]

  def get_entropic_map(self, it: int) -> potentials.EntropicPotentials:
    return self.get_output(it).to_dual_potentials()

  def get_output(self, it: int) -> Output:
    return jtu.tree_map(lambda x: x[it], self.outputs)

  def get_plan(self, it: int) -> jnp.ndarray:
    out = self.get_output(it)
    if isinstance(out, sd.SinkhornDivergenceOutput):
      out = _sink_out_from_debiased(out)
    return out.matrix

  @property
  def converged(self) -> jnp.ndarray:
    return self.outputs.converged[0]

  @property
  def num_iters(self) -> jnp.ndarray:
    n_iters = jnp.array([
        self.get_output(it).n_iters for it in range(self.num_steps)
    ])
    # handle sinkdiv output
    return n_iters[:, 0] if n_iters.ndim == 2 else n_iters

  @property
  def num_steps(self) -> int:
    return len(self.alphas)

  @property
  def cost_fn(self) -> costs.TICost:
    return self.prob.geom.cost_fn


@jtu.register_pytree_node_class
class ProgOT:

  def __init__(
      self,
      # TODO(michalk8): pass `debiased` + kwargs?
      solve_fn: SolveFn,
      *,
      alphas: jnp.ndarray,
      # if `None`, all epsilons will be `None`
      epsilons: Optional[jnp.ndarray],
      epsilon_scales: Optional[jnp.ndarray] = None,
  ):
    if epsilons is not None:
      assert len(alphas) == len(
          epsilons
      ), "Epsilon schedule has different length than alpha schedule."
    if epsilon_scales is not None:
      assert len(alphas) == len(
          epsilon_scales
      ), "Epsilon scales has different length than alpha schedule."
    self.solve_fn = solve_fn
    self.alphas = alphas
    self.epsilons = epsilons
    self.epsilon_scales = epsilon_scales

  def __call__(
      self,
      prob: linear_problem.LinearProblem,
      store_intermediate: bool = False,
      warm_start: bool = False,
  ) -> ProgOTOutput:

    def body_fn(state: ProgOTState, it: int) -> tuple[ProgOTState, jnp.ndarray]:
      # TODO(michalk8): check only one is being passed
      eps = None if self.epsilons is None else self.epsilons[it]
      eps_scale = None if self.epsilon_scales is None else self.epsilon_scales[
          it]
      alpha = self.alphas[it]
      thr = 1e-3  # TODO(michalk8)

      out = self.solve_fn(
          state.x, y, cost_fn, eps, eps_scale, state.init_potentials, thr
      )

      # TODO(michalk8): simplify
      if is_debiased:
        # out_xx = _sink_out_from_debiased(out, idx=1)
        out_xy = _sink_out_from_debiased(out, idx=0)
        eps = out_xy.geom.epsilon
      else:
        # out_xx = None
        out_xy, eps = out, out.geom.epsilon

      t_x = out.to_dual_potentials().transport(state.x, forward=True)
      next_x = _interpolate(x=state.x, t_x=t_x, alpha=alpha, cost_fn=cost_fn)

      next_init = ((1.0 - alpha) * out.f,
                   (1.0 - alpha) * out.g) if warm_start else (None, None)

      next_state = state.set(
          x=next_x,
          xs=state.xs.at[it].set(next_x) if store_intermediate else None,
          alphas=state.alphas.at[it].set(alpha),
          epsilons=state.epsilons.at[it].set(eps),
          init_potentials=next_init,
      )

      return next_state, out

    assert isinstance(
        prob.geom, pointcloud.PointCloud
    ), "Geometry must be a point cloud."
    assert isinstance(
        prob.geom.cost_fn, costs.TICost
    ), "Cost must be a TI cost."

    num_steps = len(self.alphas)

    n, m = prob.geom.shape
    x, y, cost_fn = prob.geom.x, prob.geom.y, prob.geom.cost_fn
    _, d = x.shape

    tree = jax.eval_shape(self.solve_fn, x, y, cost_fn, 1.0, None, (None, None))
    is_debiased = isinstance(tree, sd.SinkhornDivergenceOutput)
    if warm_start:
      assert not is_debiased, "Warm-start is only available for Sinkhorn."
      # FIXME(michalk8)
      # assume always LSE mode for now
      init_potentials = (jnp.zeros(n), jnp.zeros(m))
    else:
      init_potentials = (None, None)

    init = ProgOTState(
        x=x,
        xs=jnp.empty((num_steps, n, d)) if store_intermediate else None,
        alphas=jnp.empty((num_steps,)),
        epsilons=jnp.empty((num_steps,)),
        init_potentials=init_potentials,
    )
    state, outputs = jax.lax.scan(body_fn, init, xs=jnp.arange(num_steps))

    return ProgOTOutput(
        prob,
        # add the initial `x` for nicer impl. in `ProgOTOutput`
        # also we could do `xs[:-1]`, since it's not needed
        xs=(
            jnp.concatenate([x[None], state.xs[:-1]], axis=0)
            if store_intermediate else None
        ),
        alphas=state.alphas,
        epsilons=state.epsilons,
        outputs=outputs,
    )

  def tree_flatten(self):
    return [], {
        "solve_fn": self.solve_fn,
        "alphas": self.alphas,
        "epsilons": self.epsilons,
        "epsilon_scales": self.epsilon_scales,
    }

  @classmethod
  def tree_unflatten(cls, aux_data: dict[str, Any], children: Any) -> "ProgOT":
    del children
    return cls(**aux_data)


def solve_sinkhorn(
    x: jnp.ndarray,
    y: jnp.ndarray,
    cost_fn: costs.TICost,
    eps: Optional[float],
    eps_scale: Optional[float],
    init: Initializer = (None, None),
    thr: float = 1e-3,
    **kwargs: Any,
) -> sinkhorn.SinkhornOutput:
  if eps_scale is not None:
    geom = pointcloud.PointCloud(x, y, cost_fn=cost_fn, epsilon=eps)
    eps = eps_scale * geom.epsilon

  geom = pointcloud.PointCloud(x, y, cost_fn=cost_fn, epsilon=eps)
  prob = linear_problem.LinearProblem(geom)
  solver = sinkhorn.Sinkhorn(threshold=thr, **kwargs)
  return solver(prob, init=init)


def solve_sinkdiv(
    x: jnp.ndarray,
    y: jnp.ndarray,
    cost_fn: costs.TICost,
    eps: Optional[float],
    eps_scale: Optional[float],
    init: Initializer = (None, None),
    thr: float = 1e-3,
    sinkhorn_kwargs: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> sd.SinkhornDivergenceOutput:
  del init  # not implemented
  if eps_scale is not None:
    geom = pointcloud.PointCloud(x, y, cost_fn=cost_fn, epsilon=eps)
    eps = eps_scale * geom.epsilon

  if sinkhorn_kwargs is None:
    sinkhorn_kwargs = {}
  sinkhorn_kwargs["threshold"] = thr

  return sd.sinkhorn_divergence(
      pointcloud.PointCloud,
      x,
      y,
      cost_fn=cost_fn,
      epsilon=eps,
      share_epsilon=False,
      sinkhorn_kwargs=sinkhorn_kwargs,
      **kwargs,
  )


def get_epsilon_schedule(
    *,
    x: jnp.ndarray,
    y: jnp.ndarray,
    alphas: jnp.ndarray,
    epsilon_scales: jnp.ndarray,
    # TODO(michalk8): remove this
    solve_fn: SolveFn,
    eps0_multiplier: float = 1.0,
    y_eval: Optional[jnp.ndarray] = None,
    cost_fn: Optional[costs.TICost] = None,
) -> jnp.ndarray:

  def error(es: float) -> float:
    init, epsilon = (None, None), None
    dp = solve_fn(y, y, cost_fn, epsilon, es, init).to_dual_potentials()
    y_hat = dp.transport(y_eval, forward=True)
    return jnp.linalg.norm(y_eval - y_hat)

  if cost_fn is None:
    cost_fn = costs.SqEuclidean()

  geom0 = pointcloud.PointCloud(x, y, cost_fn=cost_fn)
  geom1 = pointcloud.PointCloud(y, y, cost_fn=cost_fn)

  eps0 = eps0_multiplier * geom0.epsilon

  errors = jax.vmap(error)(epsilon_scales)
  eps1_scale = epsilon_scales[jnp.argmin(errors)]
  eps1 = eps1_scale * geom1.epsilon

  mod_alpha = jnp.concatenate([jnp.array([0.0]), alphas])
  no_ending_1 = mod_alpha[-1] != 1.0  # e.g. the exp schedule
  if no_ending_1:
    mod_alpha = jnp.concatenate([mod_alpha, jnp.array([1.0])])

  tk = 1.0 - jnp.cumprod(1.0 - mod_alpha)
  epsilons = eps1 * tk + (1.0 - tk) * eps0

  epsilons = epsilons[:-1]
  if no_ending_1:
    epsilons = epsilons[:-1]

  return epsilons


def get_alpha_schedule(
    kind: Literal["lin", "exp", "quad"], *, num_steps: int
) -> jnp.ndarray:
  if kind == "lin":
    arr = jnp.arange(2, num_steps + 2)
    arr = 1.0 / (num_steps - arr + 2)
  elif kind == "exp":
    arr = jnp.full(num_steps, fill_value=1.0 / jnp.e)
  elif kind == "quad":
    arr = jnp.arange(2, num_steps + 2)
    arr = (2.0 * arr - 1.0) / ((num_steps + 1) ** 2 - (arr - 1) ** 2)
  elif kind == "exp2":
    arr = jnp.full(num_steps, fill_value=0.5)
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
