ott.neural
==========
.. module:: ott.neural
.. currentmodule:: ott.neural

In contrast to most methods presented in :mod:`ott.solvers`, which output
vectors or matrices, the goal of the :mod:`ott.neural` module is to parameterize
optimal transport maps and couplings as neural networks. These neural networks
can generalize to new samples, in the sense that they can be conveniently
evaluated outside training samples. This module implements layers, models
and solvers to estimate such neural networks.

.. toctree::
    :maxdepth: 2

    solvers

Models
------
.. autosummary::
    :toctree: _autosummary

    models.ICNN
    models.MLP
    models.MetaInitializer

Losses
------
.. autosummary::
    :toctree: _autosummary

    losses.monge_gap
    losses.monge_gap_from_samples

Layers
------
.. autosummary::
    :toctree: _autosummary

    layers.PositiveDense
    layers.PosDefPotentials
