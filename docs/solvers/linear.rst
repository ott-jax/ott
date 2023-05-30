ott.solvers.linear
==================
.. module:: ott.solvers.linear
.. currentmodule:: ott.solvers.linear

Linear solvers are the bread-and-butter of OT solvers, and can be called on
their own (such when using the Sinkhorn or Low-Rank solvers to match two
datasets), or, as subroutines for more advanced solvers in the
:mod:`ott.solvers` module, notably :mod:`ott.solvers.quadratic` or
:mod:`ott.solvers.nn`.

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
