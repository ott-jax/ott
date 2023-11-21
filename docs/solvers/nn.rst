ott.solvers.nn
==============
.. module:: ott.solvers.nn
.. currentmodule:: ott.solvers.nn

This module implements the input-convex neural network :cite:`amos:17` which
can be used to solve OT problems between point-clouds. Other simpler
alternatives, such as MLP are implemented, essentially borrowed from the
`flax <https://flax.readthedocs.io/en/latest/examples.html>`__ library.

Neural Dual
-----------
.. autosummary::
    :toctree: _autosummary

    neuraldual.W2NeuralDual

Models
------
.. autosummary::
    :toctree: _autosummary

    models.BaseW2NeuralDual
    models.ICNN
    models.MLP

Conjugate Solvers
-----------------
.. autosummary::
    :toctree: _autosummary

    conjugate_solvers.ConjugateResults
    conjugate_solvers.FenchelConjugateSolver
    conjugate_solvers.FenchelConjugateLBFGS

Losses
------
.. autosummary::
    :toctree: _autosummary

    losses.monge_gap
    losses.monge_gap_from_samples
