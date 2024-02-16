ott.neural.models
=================
.. module:: ott.neural.models
.. currentmodule:: ott.neural.models

This module implements models, network architectures and helper
functions which apply to various neural optimal transport solvers.

Utils
-----
.. autosummary::
    :toctree: _autosummary

    base_solver.BaseOTMatcher
    base_solver.OTMatcherLinear
    base_solver.OTMatcherQuad
    base_solver.UnbalancednessHandler


Neural networks
---------------
.. autosummary::
    :toctree: _autosummary

    layers.MLPBlock
    nets.RescalingMLP
