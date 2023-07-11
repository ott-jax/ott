ott.initializers
================
.. module:: ott.initializers

The :mod:`ott.initializers` module implement simple strategies to initialize
solvers. For convex solvers, these initializations can be used to gain
computational efficiency, but only have an impact in that respect.
When used on more advanced and non-convex problems, these initializations
play a far more important role.

Two problems and their solvers fall in the convex category, those are the
:class:`~ott.problems.linear.linear_problem.LinearProblem` solved with a
:class:`~ott.solvers.linear.sinkhorn.Sinkhorn` solver, or the fixed-support
Wasserstein barycenter problems :cite:`cuturi:14` described in
:class:`~ott.problems.linear.barycenter_problem.FixedBarycenterProblem` and
solved with a :class:`~ott.solvers.linear.discrete_barycenter.FixedBarycenter`
solver.

When the problem is *not* convex, which describes pretty much all other pairings
of problems/solvers in ``OTT``, notably quadratic and neural-network based
below, initializers play a more important role: different initializations will
very likely result in different end solutions.

.. toctree::
    :maxdepth: 2

    linear
    quadratic
    nn
