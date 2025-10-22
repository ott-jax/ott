ott.neural.networks
===================
.. module:: ott.neural.networks
.. currentmodule:: ott.neural.networks

Networks
--------
.. autosummary::
    :toctree: _autosummary

    icnn.ICNN
    potentials.BasePotential
    potentials.PotentialMLP
    potentials.MLP
    potentials.PotentialTrainState

ott.neural.networks.velocity_fields
===================================
.. module:: ott.neural.networks.velocity_field
.. currentmodule:: ott.neural.networks.velocity_field

MLP
---
.. autosummary::
    :toctree: _autosummary

    mlp.MLP

UNet
----
.. autosummary::
    :toctree: _autosummary

    unet.UNet

EMA
---
.. autosummary::
    :toctree: _autosummary

    ema.EMA
    ema.init_ema
    ema.update_ema

ott.neural.networks.layers
==========================
.. module:: ott.neural.networks.layers
.. currentmodule:: ott.neural.networks.layers

Layers
------
.. autosummary::
    :toctree: _autosummary

    conjugate.FenchelConjugateSolver
    conjugate.FenchelConjugateLBFGS
    conjugate.ConjugateResults
    posdef.PositiveDense
    posdef.PosDefPotentials
