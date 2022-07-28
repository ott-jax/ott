.. _tools:

ott.tools package
=================
.. currentmodule:: ott.tools
.. automodule:: ott.tools

The tools package contains high level functions that build on outputs produced by core functions.
They can be used to compute Sinkhorn divergences :cite:`sejourne:19`, instantiate transport matrices,
provide differentiable approximations to ranks and quantile functions :cite:`cuturi:19`, etc.

Optimal Transport
-----------------
.. autosummary::
    :toctree: _autosummary

    transport.Transport

Segmented Sinkhorn
------------------
.. autosummary::
    :toctree: _autosummary

    segment_sinkhorn.segment_sinkhorn


Sinkhorn Divergence
-------------------
.. autosummary::
    :toctree: _autosummary

    sinkhorn_divergence.sinkhorn_divergence
    sinkhorn_divergence.segment_sinkhorn_divergence

Soft Sorting Algorithms
-----------------------
.. autosummary::
    :toctree: _autosummary

    soft_sort.quantile
    soft_sort.quantile_normalization
    soft_sort.quantize
    soft_sort.ranks
    soft_sort.sort
    soft_sort.sort_with
