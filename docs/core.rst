.. _core:

ott.core package
================
.. currentmodule:: ott.core
.. automodule:: ott.core

The core package contains definitions of various OT problems, starting
from the most simple, the linear OT problem, to more advanced problems
such as quadratic, or involving multiple measures, the barycenter problem.
We follow with the classic :class:`~ott.core.sinkhorn.sinkhorn` routine (essentially a
wrapper for the :class:`~ott.core.sinkhorn.Sinkhorn` solver class) :cite:`cuturi:13,sejourne:19`.
We also provide an analogous low-rank Sinkhorn solver :cite:`scetbon:21` to handle very large instances.
Both are used within our Wasserstein barycenter solvers :cite:`benamou:15,janati:20a`, as well as our
Gromov-Wasserstein solver :cite:`memoli:11,scetbon:22`. We also provide an implementation of
input convex neural networks :cite:`amos:17`, a NN that can be used to estimate OT :cite:`makkuva:20`.

OT Problems
-----------
.. autosummary::
    :toctree: _autosummary

    linear_problems.LinearProblem
    quad_problems.QuadraticProblem
    bar_problems.BarycenterProblem
    bar_problems.GWBarycenterProblem

Sinkhorn
--------
.. autosummary::
    :toctree: _autosummary

    sinkhorn.sinkhorn
    sinkhorn.Sinkhorn
    sinkhorn.SinkhornOutput

Low-Rank Sinkhorn
-----------------
.. autosummary::
    :toctree: _autosummary

    sinkhorn_lr.LRSinkhorn
    sinkhorn_lr.LRSinkhornOutput

Barycenters (Entropic and LR)
-----------------------------
.. autosummary::
    :toctree: _autosummary

    discrete_barycenter.discrete_barycenter
    continuous_barycenter.WassersteinBarycenter
    continuous_barycenter.BarycenterState
    gw_barycenter.GromovWassersteinBarycenter
    gw_barycenter.GWBarycenterState

Gromov-Wasserstein (Entropic and LR)
------------------------------------
.. autosummary::
    :toctree: _autosummary

    gromov_wasserstein.gromov_wasserstein
    gromov_wasserstein.GromovWasserstein
    gromov_wasserstein.GWOutput

Neural Potentials
-----------------
.. autosummary::
    :toctree: _autosummary

    icnn.ICNN
    neuraldual.NeuralDualSolver
    neuraldual.NeuralDual

Padding Utilities
-----------------
.. autosummary::
    :toctree: _autosummary

    segment.segment_point_cloud
    segment.pad_along_axis
