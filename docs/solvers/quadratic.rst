ott.solvers.quadratic
=====================
.. module:: ott.solvers.quadratic
.. currentmodule:: ott.solvers.quadratic

The :mod:`ott.solvers.quadratic` module holds the important family of
Gromov-Wasserstein (GW) solvers and related GW barycenters. They are designed to
solve :mod:`ott.problems.quadratic` problems.

Gromov-Wasserstein Solvers
--------------------------
.. autosummary::
    :toctree: _autosummary

    solve
    gromov_wasserstein.GromovWasserstein
    gromov_wasserstein.GWOutput
    gromov_wasserstein_lr.LRGromovWasserstein
    gromov_wasserstein_lr.LRGWOutput
    lower_bound.LowerBoundSolver


Barycenter Solvers
------------------
.. autosummary::
    :toctree: _autosummary

    gw_barycenter.GWBarycenterState
    gw_barycenter.GromovWassersteinBarycenter
