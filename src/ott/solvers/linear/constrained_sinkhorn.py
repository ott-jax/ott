#Import libraries
from typing import Tuple, Sequence, Callable

import jax
import jax.numpy as jnp
from ott.geometry import geometry          # core class of OTT-JAX

Array = jnp.ndarray

# Define preliminary functions
def make_geom(cost: Array,
              mats_D: Array,
              alpha: Array,
              eta: float) -> geometry.Geometry:
    """
    Build a fresh :class:`ott.geometry.geometry.Geometry` object
    that encodes the current tilted cost

        \\widetilde C = C - \\sum_m alpha_m D_m .

    Parameters
    ----------
    cost
        Ground cost matrix ``C`` (shape *(n, m)*).
    mats_D
        Stack of linear-constraint matrices ``D`` with shape *(M, n, m)*.
    alpha
        Dual prices for the constraints (length *M*).
    eta
        Entropic scale (``eta`` == 1/epsilon).

    Notes
    -----
    ``Geometry`` uses :math:`\\varepsilon = 1/eta`, so we pass
    ``epsilon = 1.0 / eta`` below.
    """
    # contraction over axis 0 of D and alpha  ->  \sum_m \alpha_m D_m, shape (n,n)
    tilt = cost - jnp.tensordot(alpha, mats_D, axes=1)
    return geometry.Geometry(cost_matrix=tilt,
                             epsilon=1.0 / eta,   # Geometry's notation
                             scale_cost=1)      # keep original numeric scale

def sinkhorn_once(geom: geometry.Geometry,
                  u: Array,
                  v: Array,
                  a: Array,
                  b: Array) -> Tuple[Array, Array]:
    """
    One ''IBP-style'' pair of row + column projections.

    Parameters
    ----------
    geom
        Geometry instance produced by :func:`make_geom`.
    u, v
        Current positive scalings (u ~ e^{eta*x}, v ~ e^{\eta*y}).
    a, b
        Target marginals (sum to 1).

    Returns
    -------
    u, v
        Updated scalings.
    """
    # ---- Row projection  --------------------------------------------
    row_sum = geom.marginal_from_scalings(u, v, axis=1)    # P 1
    u = u * (a/row_sum)       # u <- u * (r / row_sum)

    # ---- Column projection  -----------------------------------------
    col_sum = geom.marginal_from_scalings(u, v, axis=0)   # P'*1
    v = v * (b/col_sum)         # v <- v * (c / col_sum)

    return u, v

    def residuals(transport: Array, mats_D: Array,thr= None) -> Array:
        r"""
        Compute residual vector  \rho_k = <D_k, P>  for all constraints.

        Shapes
        ------
        transport :  (n, m)
        mats_D    :  (M, n, m)
        thr : Array of shape (M,), optional
            Thresholds for each constraint. If provided, returns
            res[k] = <D_k, P> - thr[k]
            enforcing  <D_k, P> >= thr[k].
            If None, returns raw[k] = <D_k, P>.
        return    :  (M,)
        """
        if thr is None:
            return jnp.tensordot(mats_D, transport, axes=((1, 2), (0, 1)))
        else:
            return jnp.tensordot(mats_D, transport, axes=((1, 2), (0, 1))) - thr


def diag_inv_hessian(mats_D: Array,
                     transport: Array,
                     damp: float = 1e-8) -> Array:
    """
    Very cheap diagonal approximation of the Hessian
        H_{kl} = <D_k @ D_l , P>
    used as a pre-conditioner for Newton.

    Returns
    -------
    diag_inv : (M,)  element-wise inverse of the diagonal.
    """
    diag = jnp.tensordot(mats_D ** 2, transport, axes=((1, 2), (0, 1)))
    return 1.0 / (diag + damp)

#final function
def constrained_sinkhorn(C: Array,
                   a: Array,
                   b: Array,
                   mats_D: Array,
                   eta_init: float = 1.0,
                   eta_final: float = 128.0,
                   tau: float = 1e-4,
                   max_iters: int = 10_000,
                   hessian: Callable[[Array, Array], Array] = diag_inv_hessian
                   ) -> Tuple[Array, Array]:
    """
    Run Algorithm 1 with *eta-doubling* schedule.

    Parameters
    ----------
    C
        Ground cost matrix :math:`C`.
    a, b
        Source and target probability vectors.
    mats_D
        Stack of linear-constraint matrices (*M, n, m*).
    eta_init, eta_final
        Start and stop values for the entropy scale.
        The loop doubles ``eta`` each outer sweep.
    tau
        Stop once  ||residuals||_1 <= ``tau``.
    max_iter
        Hard cap on (inner) Sinkhorn iterations **per eta**.
    hessian
        Function returning a vector *prec* ~ diag(H)^{-1}.
        Replace with an exact solver for small *M* if desired.

    Returns
    -------
    P
        Entropy-regularised optimal plan (n x m).
    alpha
        Duals for the constraints (length *M*).
    """
    n,m = C.shape
    M = mats_D.shape[0]

    # ------------- Initialise duals & scalings -----------------------
    x = jnp.zeros(n)                       # row duals
    y = jnp.zeros(m)                       # col duals
    alpha = jnp.zeros(M)                   # constraint duals
    u = jnp.exp(+x)                        # = e^{eta*x}  (eta = 1 at start)
    v = jnp.exp(+y)

    eta = eta_init
    while eta <= eta_final:
        # =========================================================== #
        #   Inner loop at fixed epsilon = 1/eta                               #
        # =========================================================== #
        i = 0
        converged = False
        while (i < max_iters) & (~converged):
            geom = make_geom(C, mats_D, alpha, eta)

            # ---- 1. Sinkhorn pair ----------------------------------
            u, v = sinkhorn_once(geom, u, v, a, b)

            # ---- 2. Build P & residuals ----------------------------
            P = geom.transport_from_scalings(u, v)         # diag(u) K diag(v)
            res = residuals(P, mats_D, thr=thr)                     # size M

            # ---- 3. Check stopping criterion ----------------------
            converged = (jnp.linalg.norm(res, ord=1) <= tau)
            if converged:
              break

            # ---- 4. Newton / gradient-descent on alpha ----------------
            prec = hessian(mats_D, P)                      # (M,) or (M,M)
            delta_alpha = -prec * res                      # diagonal pre-cond.
            alpha += delta_alpha

            # ---- 5. Optional centering scalar t -------------------
            # Heuristic: keep average of x near zero for numerical
            # stability.  See App. C of the paper.
            t = -delta_alpha.sum() / n
            x = (1.0 / eta) * jnp.log(u) + t               # update duals
            u = jnp.exp(+eta * x)                          # keep consistency

            i += 1

        # ========== End inner loop; prepare *warm start* ============
        eta *= 2.0                 # halves epsilon, doubles eta
        u = u ** 2                 # because K = e^{-C/epsilon} will square
        v = v ** 2                 #   when epsilon halves

    # ----------------------------------------------------------------
    # done: build final P with last geometry for return
    geom = make_geom(C, mats_D, alpha, eta / 2.0)       # last geom used
    P = geom.transport_from_scalings(u, v)
    return P, alpha