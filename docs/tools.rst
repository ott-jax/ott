ott.tools
=========
.. module:: ott.tools
.. currentmodule:: ott.tools

The :mod:`~ott.tools` package contains high level functions that build on
outputs produced by lower-level components in the toolbox, such as
:mod:`~ott.solvers`.

In particular, we provide user-friendly APIs to compute Sinkhorn divergences
:cite:`genevay:18,sejourne:19`, sliced Wasserstein distances :cite:`rabin:12`,
differentiable approximations to ranks and quantile functions :cite:`cuturi:19`,
and various tools to study Gaussians with the 2-Wasserstein metric
:cite:`gelbrich:90,delon:20`, etc.

Segmented Sinkhorn
------------------
.. autosummary::
    :toctree: _autosummary

    segment_sinkhorn.segment_sinkhorn

Sinkhorn Divergence
-------------------
.. autosummary::
    :toctree: _autosummary

    sinkhorn_divergence.sinkdiv
    sinkhorn_divergence.sinkhorn_divergence
    sinkhorn_divergence.SinkhornDivergenceOutput
    sinkhorn_divergence.segment_sinkhorn_divergence

Sliced Wasserstein Distance
---------------------------
.. autosummary::
    :toctree: _autosummary

    sliced.random_proj_sphere
    sliced.sliced_wasserstein

ProgOT
------
.. autosummary::
    :toctree: _autosummary

    progot.ProgOT
    progot.ProgOTOutput
    progot.get_alpha_schedule
    progot.get_epsilon_schedule


Soft Sorting Algorithms
-----------------------
.. autosummary::
    :toctree: _autosummary

    soft_sort.multivariate_cdf_quantile_maps
    soft_sort.quantile
    soft_sort.quantile_normalization
    soft_sort.quantize
    soft_sort.ranks
    soft_sort.sort
    soft_sort.sort_with
    soft_sort.topk_mask

Clustering
----------
.. autosummary::
    :toctree: _autosummary

    k_means.k_means
    k_means.KMeansOutput

Plotting
--------
.. autosummary::
    :toctree: _autosummary

    plot.Plot

ott.tools.gaussian_mixture package
----------------------------------
.. currentmodule:: ott.tools.gaussian_mixture
.. automodule:: ott.tools.gaussian_mixture

This package implements various tools to manipulate Gaussian mixtures with a
slightly modified Wasserstein geometry: here a Gaussian mixture is no longer
strictly regarded as a density :math:`\mathbb{R}^d`, but instead as a point
cloud in the space of Gaussians in :math:`\mathbb{R}^d`. This viewpoint provides
a new approach to compare, and fit Gaussian mixtures, as described for instance
in :cite:`delon:20` and references therein.

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
