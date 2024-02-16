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

    data
    duality
    flow_models
    gaps
    models
