ott.initializers.linear
=======================
.. module:: ott.initializers.linear
.. currentmodule:: ott.initializers.linear

Initializers for linear OT problems, focusing on Sinkhorn and low-rank solvers.

Sinkhorn Initializers
---------------------
.. autosummary::
    :toctree: _autosummary

    initializers.DefaultInitializer
    initializers.GaussianInitializer
    initializers.SortingInitializer
    initializers.SubsampleInitializer

Low-rank Sinkhorn Initializers
------------------------------
.. autosummary::
    :toctree: _autosummary

    initializers_lr.LRInitializer
    initializers_lr.RandomInitializer
    initializers_lr.Rank2Initializer
    initializers_lr.KMeansInitializer
    initializers_lr.GeneralizedKMeansInitializer
