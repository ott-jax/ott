from typing import Any, Optional

import jax.numpy as jnp

from ott.geometry import geometry
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn


def run_sinkhorn(
    geom: geometry.Geometry,
    a: Optional[jnp.ndarray] = None,
    b: Optional[jnp.ndarray] = None,
    tau_a: float = 1.0,
    tau_b: float = 1.0,
    **kwargs: Any
) -> sinkhorn.SinkhornOutput:
  prob = linear_problem.LinearProblem(geom, a=a, b=b, tau_a=tau_a, tau_b=tau_b)
  solver = sinkhorn.Sinkhorn(**kwargs)
  return solver(prob)
