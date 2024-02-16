ott.neural.duality
==================
.. module:: ott.neural.duality
.. currentmodule:: ott.neural.duality

This module implements various solvers to estimate optimal transport between
two probability measures, through samples, parameterized as neural networks.
These solvers build uponn dual formulation of the optimal transport problem.

Solvers
-------
.. autosummary::
    :toctree: _autosummary

    neuraldual.W2NeuralDual
    neuraldual.BaseW2NeuralDual

Conjugate Solvers
-----------------
.. autosummary::
    :toctree: _autosummary

    conjugate.FenchelConjugateLBFGS
    conjugate.FenchelConjugateSolver
    conjugate.ConjugateResults

Models
------
.. autosummary::
    :toctree: _autosummary

    neuraldual.W2NeuralTrainState
    neuraldual.BaseW2NeuralDual
    neuraldual.W2NeuralDual
    models.ICNN
    models.PotentialMLP
    models.MetaInitializer
