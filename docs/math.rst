ott.math
========
.. currentmodule:: ott.math
.. automodule:: ott.math

The :mod:`ott.math` module holds low level computational primitives that
appear in some more advanced optimal transport problems.
Function :func:`~ott.math.fixed_point_loop.fixpoint_iter` implements a
fixed-point iteration `while` loop that can be automatically differentiated,
and which might be of more general interest to other `JAX` users.
Function :func:`~ott.math.matrix_square_root.sqrtm` contains an implementation
of the matrix square-root using the Newton-Schulz iterations. That
implementation is itself differentiable using either
:term:`implicit differentiation` or :term:`unrolling` of the updates of these
iterations. :mod:`ott.math.utils` contains various low-level routines
re-implemented for their usage in :mod:`jax`. Of particular interest are
the custom jvp/vjp re-implementations for ``logsumexp`` and ``norm`` that have
a behavior that differs, in terms of differentiability, from the standard
:mod:`jax` implementations.


Fixed-point Iteration
---------------------
.. autosummary::
    :toctree: _autosummary

    fixed_point_loop.fixpoint_iter

Matrix Square Root
------------------
.. autosummary::
    :toctree: _autosummary

    matrix_square_root.sqrtm

Miscellaneous
-------------
.. autosummary::
    :toctree: _autosummary

    lbfgs
    legendre
    utils.norm
    utils.logsumexp
    utils.softmin
    utils.lambertw
