ott.solvers.linear
==================
.. currentmodule:: ott.solvers.linear
.. automodule:: ott.solvers.linear

.. TODO(marcocuturi): maybe add some text here

Sinkhorn Solvers
----------------
.. autosummary::
    :toctree: _autosummary

    sinkhorn.solve
    sinkhorn.Sinkhorn
    sinkhorn.SinkhornOutput
    sinkhorn_lr.LRSinkhorn
    sinkhorn_lr.LRSinkhornOutput

Barycenter Solvers
------------------
.. autosummary::
    :toctree: _autosummary

    continuous_barycenter.WassersteinBarycenter
    continuous_barycenter.BarycenterState
    discrete_barycenter.discrete_barycenter
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
