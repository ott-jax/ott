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

OT Flow Matching
----------------
.. autosummary::
    :toctree: _autosummary

    otfm.OTFlowMatching

GENOT
-----
.. autosummary::
    :toctree: _autosummary

    genot.GENOT

Models
------
.. autosummary::
    :toctree: _autosummary

    models.VelocityField

Utils
-----
.. autosummary::
    :toctree: _autosummary

    utils.match_linear
    utils.match_quadratic
    utils.sample_joint
    utils.sample_conditional
    utils.cyclical_time_encoder
    utils.uniform_sampler
