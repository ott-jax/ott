ott.problems.linear
===================
.. module:: ott.problems.linear
.. currentmodule:: ott.problems.linear

The :mod:`ott.problems.linear` describes the simplest family of optimal
transport problems, those that involve computing the Kantorovich problem itself,
also known as the linear optimal transport problem, or, more generally,
objective functions that are sums of such optimal transport costs, which
includes the two variants of Wasserstein barycenter problems.

The module also holds dual potential variables, a class of functions that act
as optimization variables for the dual optimal transport problem.

OT Problems
-----------
.. autosummary::
    :toctree: _autosummary

    linear_problem.LinearProblem
    semidiscrete_problem.SemidiscreteProblem
    barycenter_problem.FixedBarycenterProblem
    barycenter_problem.FreeBarycenterProblem

Dual Potentials
---------------
.. autosummary::
    :toctree: _autosummary

    potentials.DualPotentials
    potentials.EntropicPotentials
