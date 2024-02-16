ott.neural.flow_models
======================
.. module:: ott.neural.flow_models
.. currentmodule:: ott.neural.flow_models

This module implements various solvers building upon flow matching
:cite:`lipman:22` to match distributions.

Flows
-----
.. autosummary::
    :toctree: _autosummary

    flows.BaseFlow
    flows.StraightFlow
    flows.ConstantNoiseFlow
    flows.BrownianNoiseFlow

Optimal Transport Flow Matching
-------------------------------
.. autosummary::
    :toctree: _autosummary

    otfm.OTFlowMatching

GENOT
-----
.. autosummary::
    :toctree: _autosummary

    genot.GENOTBase
    genot.GENOTLin
    genot.GENOTQuad

Models
------
.. autosummary::
    :toctree: _autosummary

    models.VelocityField

Utils
-----
.. autosummary::
    :toctree: _autosummary

    layers.CyclicalTimeEncoder
    samplers.uniform_sampler
