ott.initializers.nn
===================
.. module:: ott.initializers.nn
.. currentmodule:: ott.initializers.nn

Neural network solvers depend heavily on initialization. This module provides
simple tools to ensure these initialization are meaningful, either by defaulting
to the simplest (something looking like an identity map), or to mimic a linear
OT map between Gaussian approximations of the source and target measures.

Neural Initializers
-------------------
.. autosummary::
    :toctree: _autosummary

    initializers.MetaInitializer
    initializers.MetaMLP
