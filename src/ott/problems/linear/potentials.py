from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Literal,
    Optional,
    Sequence,
    Tuple,
)

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from ott.problems.linear import linear_problem

if TYPE_CHECKING:
  from ott.geometry import costs

__all__ = [
    "DualPotentials", "EntropicPotentials", "plot_ot_map", "plot_potential"
]
Potential_t = Callable[[jnp.ndarray], float]


@jtu.register_pytree_node_class
class DualPotentials:
  r"""The Kantorovich dual potential functions :math:`f` and :math:`g`.

  :math:`f` and :math:`g` are a pair of functions, candidates for the dual
  OT Kantorovich problem, supposedly optimal for a given pair of measures.

  Args:
    f: The first dual potential function.
    The: g second dual potential function.
    cost_fn: The cost function used to solve the OT problem.
    corr: Whether the duals solve the problem in distance form, or correlation
      form (as used for instance for ICNNs, see, e.g., top right of p.3 in
      :cite:`makkuva:20`)
  """

  def __init__(
      self,
      f: Potential_t,
      g: Potential_t,
      *,
      cost_fn: 'costs.CostFn',
      corr: bool = False
  ):
    self._f = f
    self._g = g
    self.cost_fn = cost_fn
    self._corr = corr

  def transport(self, vec: jnp.ndarray, forward: bool = True) -> jnp.ndarray:
    r"""Transport ``vec`` according to Brenier formula :cite:`brenier:91`.

    Uses Theorem 1.17 from :cite:`santambrogio:15` to compute an OT map when
    given the Legendre transform of the dual potentials.

    That OT map can be recovered as :math:`x- (\nabla h)^{-1}\circ \nabla f(x)`
    For the case :math:`h(\cdot) = \|\cdot\|^2, \nabla h(\cdot) = 2 \cdot\,`,
    and as a consequence :math:`h^*(\cdot) = \|.\|^2 / 4`, while one has that
    :math:`\nabla h^*(\cdot) = (\nabla h)^{-1}(\cdot) = 0.5 \cdot\,`.

    When the dual potentials are solved in correlation form (only in the Sq.
    Euclidean distance case), the maps are :math:`\nabla g` for forward,
    :math:`\nabla f` for backward.

    Args:
      vec: Points to transport, array of shape ``[n, d]``.
      forward: Whether to transport the points from source  to the target
        distribution or vice-versa.

    Returns:
      The transported points.
    """
    from ott.geometry import costs

    vec = jnp.atleast_2d(vec)
    if self._corr and isinstance(self.cost_fn, costs.SqEuclidean):
      return self._grad_f(vec) if forward else self._grad_g(vec)
    if forward:
      return vec - self._grad_h_inv(self._grad_f(vec))
    else:
      return vec - self._grad_h_inv(self._grad_g(vec))

  def distance(self, src: jnp.ndarray, tgt: jnp.ndarray) -> float:
    """Evaluate 2-Wasserstein distance between samples using dual potentials.

    Uses Eq. 5 from :cite:`makkuva:20` when given in `corr` form, direct
    estimation by integrating dual function against points when using dual form.

    Args:
      src: Samples from the source distribution, array of shape ``[n, d]``.
      tgt: Samples from the target distribution, array of shape ``[m, d]``.

    Returns:
      Wasserstein distance.
    """
    src, tgt = jnp.atleast_2d(src), jnp.atleast_2d(tgt)
    f = jax.vmap(self.f)

    if self._corr:
      grad_g_y = self._grad_g(tgt)
      term1 = -jnp.mean(f(src))
      term2 = -jnp.mean(jnp.sum(tgt * grad_g_y, axis=-1) - f(grad_g_y))

      C = jnp.mean(jnp.sum(src ** 2, axis=-1))
      C += jnp.mean(jnp.sum(tgt ** 2, axis=-1))
      return 2. * (term1 + term2) + C

    g = jax.vmap(self.g)
    return jnp.mean(f(src)) + jnp.mean(g(tgt))

  @property
  def f(self) -> Potential_t:
    """The first dual potential function."""
    return self._f

  @property
  def g(self) -> Potential_t:
    """The second dual potential function."""
    return self._g

  @property
  def _grad_f(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Vectorized gradient of the potential function :attr:`f`."""
    return jax.vmap(jax.grad(self.f, argnums=0))

  @property
  def _grad_g(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Vectorized gradient of the potential function :attr:`g`."""
    return jax.vmap(jax.grad(self.g, argnums=0))

  @property
  def _grad_h_inv(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
    from ott.geometry import costs

    assert isinstance(self.cost_fn, costs.TICost), (
        "Cost must be a `TICost` and "
        "provide access to Legendre transform of `h`."
    )
    return jax.vmap(jax.grad(self.cost_fn.h_legendre))

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    return [], {
        "f": self._f,
        "g": self._g,
        "cost_fn": self.cost_fn,
        "corr": self._corr
    }

  @classmethod
  def tree_unflatten(
      cls, aux_data: Dict[str, Any], children: Sequence[Any]
  ) -> "DualPotentials":
    return cls(*children, **aux_data)


@jtu.register_pytree_node_class
class EntropicPotentials(DualPotentials):
  """Dual potential functions from finite samples :cite:`pooladian:21`.

  Args:
    f_xy: The first dual potential vector of shape ``[n,]``.
    g_xy: The second dual potential vector of shape ``[m,]``.
    prob: Linear problem with :class:`~ott.geometry.pointcloud.PointCloud`
      geometry that was used to compute the dual potentials using, e.g.,
      :class:`~ott.solvers.linear.sinkhorn.Sinkhorn`.
    f_xx: The first dual potential vector of shape ``[n,]`` used for debiasing
      :cite:`pooladian:22`.
    g_yy: The second dual potential vector of shape ``[m,]`` used for debiasing.
  """

  def __init__(
      self,
      f_xy: jnp.ndarray,
      g_xy: jnp.ndarray,
      prob: linear_problem.LinearProblem,
      f_xx: Optional[jnp.ndarray] = None,
      g_yy: Optional[jnp.ndarray] = None,
  ):
    # we pass directly the arrays and override the properties
    # since only the properties need to be callable
    super().__init__(f_xy, g_xy, cost_fn=prob.geom.cost_fn, corr=False)
    self._prob = prob
    self._f_xx = f_xx
    self._g_yy = g_yy

  @property
  def f(self) -> Potential_t:
    return self._potential_fn(kind="f")

  @property
  def g(self) -> Potential_t:
    return self._potential_fn(kind="g")

  def _potential_fn(self, *, kind: Literal["f", "g"]) -> Potential_t:
    from ott.geometry import pointcloud

    def callback(
        x: jnp.ndarray,
        *,
        potential: jnp.ndarray,
        y: jnp.ndarray,
        weights: jnp.ndarray,
        epsilon: float,
    ) -> float:
      x = jnp.atleast_2d(x)
      assert x.shape[-1] == y.shape[-1], (x.shape, y.shape)
      geom = pointcloud.PointCloud(x, y, cost_fn=self.cost_fn)
      cost = geom.cost_matrix
      z = (potential - cost) / epsilon
      lse = -epsilon * jsp.special.logsumexp(z, b=weights, axis=-1)
      return jnp.squeeze(lse)

    assert isinstance(
        self._prob.geom, pointcloud.PointCloud
    ), f"Expected point cloud geometry, found `{type(self._prob.geom)}`."
    x, y = self._prob.geom.x, self._prob.geom.y
    a, b = self._prob.a, self._prob.b

    if kind == "f":
      # When seeking to evaluate 1st potential function,
      # the 2nd set of potential values and support should be used,
      # see proof of Prop. 2 in https://arxiv.org/pdf/2109.12004.pdf
      potential, arr, weights = self._g, y, b
    else:
      potential, arr, weights = self._f, x, a

    potential_xy = jax.tree_util.Partial(
        callback,
        potential=potential,
        y=arr,
        weights=weights,
        epsilon=self.epsilon,
    )
    if not self.is_debiased:
      return potential_xy

    ep = EntropicPotentials(self._f_xx, self._g_yy, prob=self._prob)
    # switch the order because for `kind='f'` we require `f/x/a` in `other`
    # which is accessed when `kind='g'`
    potential_other = ep._potential_fn(kind="g" if kind == "f" else "f")

    return lambda x: (potential_xy(x) - potential_other(x))

  @property
  def is_debiased(self) -> bool:
    """Whether the entropic map is debiased."""
    return self._f_xx is not None and self._g_yy is not None

  @property
  def epsilon(self) -> float:
    """Entropy regularizer."""
    return self._prob.geom.epsilon

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    return [self._f, self._g, self._prob, self._f_xx, self._g_yy], {}


def plot_ot_map(
    learned_potentials: DualPotentials,
    source: jnp.ndarray,
    target: jnp.ndarray,
    inverse: bool = False,
    ax: Optional[matplotlib.axes.Axes] = None,
    **kwargs: Any,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
  """Plot data and learned optimal transport map.

  Args:
    learned_potentials: the potentials object for the map
    source: samples from the source measure
    target: samples from the target measure
    inverse: if the inverse map from the potentials
      should be used
    ax: axis to add the plot to

  Returns: matplotlib figure and axis with the plots
  """

  def draw_arrows(a: np.ndarray, b: np.ndarray) -> None:
    ax.arrow(
        a[0], a[1], b[0] - a[0], b[1] - a[1], color=[0.5, 0.5, 1], alpha=0.3
    )

  grad_state_s = learned_potentials.transport(source, forward=not inverse)

  if ax is None:
    fig = plt.figure(facecolor="white")
    ax = fig.add_subplot(111)
  else:
    fig = ax.get_figure()

  if not inverse:
    label_src, label_tgt = 'source', 'target'
    label_transport = r"$\nabla f(source)$"
  else:
    label_src, label_tgt = 'target', 'source'
    label_transport = r"$\nabla g(source)$"

  ax.scatter(
      source[:, 0],
      source[:, 1],
      color="#1A254B",
      alpha=0.5,
      label=label_src,
      **kwargs,
  )
  ax.scatter(
      target[:, 0],
      target[:, 1],
      color="#A7BED3",
      alpha=0.5,
      label=label_tgt,
      **kwargs,
  )
  ax.scatter(
      grad_state_s[:, 0],
      grad_state_s[:, 1],
      color="#F2545B",
      alpha=0.5,
      label=label_transport,
      **kwargs,
  )

  fig.legend(ncol=3, loc="upper center")

  for i in range(source.shape[0]):
    draw_arrows(source[i, :], grad_state_s[i, :])

  return fig, ax


def plot_potential(
    learned_potentials: DualPotentials,
    inverse: bool = False,
    alpha: float = 0.05,
    ax: Optional[matplotlib.axes.Axes] = None,
    **kwargs: Any
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
  """Plot the potential.

  Args:
    learned_potentials: the potentials object for the map
    inverse: if the inverse map from the potentials
      should be used
    alpha: Quantile to filter the potentials with
    ax: axis to add the plot to

  Returns: matplotlib figure and axis with the plots.
  """
  if ax is None:
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
  else:
    fig = ax.get_figure()

  x1 = np.linspace(-6, 6)
  x2 = np.linspace(-6, 6)
  X1, X2 = np.meshgrid(x1, x2)
  X1flat = np.ravel(X1)
  X2flat = np.ravel(X2)
  X12flat = np.stack((X1flat, X2flat)).T
  if not inverse:
    Zflat = jax.vmap(learned_potentials.f)(X12flat)
  else:
    Zflat = jax.vmap(learned_potentials.g)(X12flat)
  Zflat = np.asarray(Zflat)
  vmin, vmax = np.quantile(Zflat, [alpha, 1. - alpha])
  Zflat = Zflat.clip(vmin, vmax)
  Z = Zflat.reshape(X1.shape)

  CS = ax.contourf(X1, X2, Z, cmap="Blues", **kwargs)
  fig.colorbar(CS, ax=ax)
  fig.tight_layout()
  if not inverse:
    ax.set_title(r"$f$")
  else:
    ax.set_title(r"$g$")
  return fig, ax
