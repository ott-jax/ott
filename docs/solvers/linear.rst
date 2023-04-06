ott.solvers.linear
==================
.. module:: ott.solvers.linear
.. currentmodule:: ott.solvers.linear

.. TODO(marcocuturi): maybe add some text here

Sinkhorn Solvers
----------------
.. autosummary::
    :toctree: _autosummary

    sinkhorn.solve
    sinkhorn.Sinkhorn
    sinkhorn.SinkhornState
    sinkhorn.SinkhornOutput
    sinkhorn_lr.LRSinkhorn
    sinkhorn_lr.LRSinkhornState
    sinkhorn_lr.LRSinkhornOutput

Barycenter Solvers
------------------
.. autosummary::
    :toctree: _autosummary

    continuous_barycenter.FreeWassersteinBarycenter
    continuous_barycenter.FreeBarycenterState
    discrete_barycenter.FixedBarycenter
    discrete_barycenter.SinkhornBarycenterOutput

Sinkhorn Acceleration
---------------------
.. autosummary::
    :toctree: _autosummary

    acceleration.Momentum
    acceleration.AndersonAcceleration

Implicit Differentiation
------------------------
.. autosummary::
    :toctree: _autosummary

    implicit_differentiation.ImplicitDiff
