ott.problems.quadratic
======================
.. module:: ott.problems.quadratic
.. currentmodule:: ott.problems.quadratic

The :mod:`ott.problems.quadratic` module describes the quadratic assignment
problem and its generalizations, including notably the fused-problem (including
a linear term) and the more advanced GW barycenter problem.

OT Problems
-----------
.. autosummary::
    :toctree: _autosummary

    quadratic_problem.QuadraticProblem
    gw_barycenter.GWBarycenterProblem

Costs
-----
.. autosummary::
    :toctree: _autosummary

    quadratic_costs.GWLoss
    quadratic_costs.make_square_loss
    quadratic_costs.make_kl_loss
