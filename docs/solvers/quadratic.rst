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

    gromov_wasserstein.solve
    gromov_wasserstein.GromovWasserstein
    gromov_wasserstein.GWOutput

Barycenter Solvers
------------------
.. autosummary::
    :toctree: _autosummary

    gw_barycenter.GWBarycenterState
    gw_barycenter.GromovWassersteinBarycenter
