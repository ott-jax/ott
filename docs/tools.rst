ott.tools
=========
.. currentmodule:: ott.tools
.. automodule:: ott.tools

The tools package contains high level functions that build on outputs produced
by core functions. They can be used to compute Sinkhorn divergences
:cite:`sejourne:19`, instantiate transport matrices, provide differentiable
approximations to ranks and quantile functions :cite:`cuturi:19`, etc.

Segmented Sinkhorn
------------------
.. autosummary::
    :toctree: _autosummary

    segment_sinkhorn.segment_sinkhorn

Plotting
--------
.. autosummary::
    :toctree: _autosummary

    plot.Plot

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

Clustering
----------
.. autosummary::
    :toctree: _autosummary

    k_means.k_means
    k_means.KMeansOutput

ott.tools.gaussian_mixture package
----------------------------------
.. currentmodule:: ott.tools.gaussian_mixture
.. automodule:: ott.tools.gaussian_mixture

.. TODO(cuturi): add a description

Gaussian Mixtures
^^^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: _autosummary

    gaussian.Gaussian
    gaussian_mixture.GaussianMixture
    gaussian_mixture_pair.GaussianMixturePair
    fit_gmm.initialize
    fit_gmm.fit_model_em
    fit_gmm_pair.get_fit_model_em_fn
