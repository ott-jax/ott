ott.problems
============
.. module:: ott.problems

The :mod:`ott.problems` module describes the low level optimal transport
problems that are solved by :mod:`ott.solvers`. These problems are
loosely divided into two categories, first finite-sample based problems, as in
:mod:`ott.problems.linear` and :mod:`ott.problems.quadratic`, or relying on
iterators. In that latter category, :mod:`ott.problems.nn` contains synthetic
data iterators.

.. toctree::
    :maxdepth: 2

    linear
    quadratic
    nn
