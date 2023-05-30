ott.math
========
.. currentmodule:: ott.math
.. automodule:: ott.math

The :mod:`ott.math` modules holds low level comptutational primitives that
appear in some more advanced optimal transport problems.
:mod:`ott.math.fixed_point_loop` implements a fixed-point iteration `while` loop
that can be automatically differentiated, and which might
be of more general interest to other `JAX` users.
:mod:`ott.math.matrix_square_root` contains an implementation of the
matrix square-root using the Newton-Schulz iterations. That implementation is
itself differentiable using either implicit differentiation or unrolling of the
updates of these iterations.


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
