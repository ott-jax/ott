ott.neural.solvers
==================
.. module:: ott.neural.solvers
.. currentmodule:: ott.neural.solvers

This module implements various solvers to estimate optimal transport between
two probability measures, through samples, parameterized as neural networks.
These neural networks are described in :mod:`ott.neural.models`, borrowing
lower-level components from :mod:`ott.neural.layers` using
`flax <https://flax.readthedocs.io/en/latest/>`__.

Solvers
-------
.. autosummary::
    :toctree: _autosummary

    map_estimator.MapEstimator
    neuraldual.W2NeuralDual
    neuraldual.BaseW2NeuralDual
    expectile_neural_dual.ExpectileNeuralDual

Conjugate Solvers
-----------------
.. autosummary::
    :toctree: _autosummary

    conjugate.FenchelConjugateLBFGS
    conjugate.FenchelConjugateSolver
    conjugate.ConjugateResults
