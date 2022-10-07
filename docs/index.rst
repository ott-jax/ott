Optimal Transport Tools (OTT) documentation
===========================================
`Code <https://github.com/ott-jax/ott>`_ hosted on Github.
To install, clone that repo or simply run ``pip install ott-jax``.

Intro
-----
OTT is a `JAX <https://jax.readthedocs.io/en/latest/index.html>`_ package that bundles a few utilities to compute and
differentiate the solution to optimal transport problems. OTT can help you compute Wasserstein distances between
weighted clouds of points (or histograms), using a cost (e.g. a distance) between individual points.

To that end OTT uses various implementation of the Sinkhorn algorithm :cite:`cuturi:13,peyre:19,scetbon:21`.
These implementation take advantage of several JAX features, such as `Just-in-time (JIT) compilation`_,
`auto-vectorization (VMAP)`_ and both `automatic`_ and/or `implicit`_ differentiation.
A few tutorials are provided below, along with different use-cases,
notably for single-cell genomics data :cite:`schiebinger:19`.

Packages
--------
There are currently three packages, ``geometry``, ``core`` and ``tools``, playing the following roles:

- :ref:`geometry` defines classes that describe *two point clouds* paired with a *cost* function (simpler geometries
  are also implemented, such as that defined by points supported on a multi-dimensional grids with a separable
  cost :cite:`solomon:15`). The design choice in OTT is to state that cost functions and algorithms should operate
  independently: if a particular cost function allows for faster computations
  (e.g. squared-Euclidean distance when comparing grids), this should not be taken advantage of at the level of
  optimizers, but at the level of the problems description. Geometry objects are therefore only considered as
  arguments to describe OT problem handled in ``core``, using subroutines provided by geometries;
- :ref:`core` help define first an OT problem (linear, quadratic, barycenters). These problems are then solved using
  Sinkhorn algorithm and its variants, the main workhorse to solve OT in this package, as well as variants that
  can comppute Gromov-Wasserstein distances or barycenters of several measures;
- :ref:`tools` provides an interface to exploit OT solutions, as produced by ``core`` functions. Such tasks include
  instantiating OT matrices, computing approximations to Wasserstein distances :cite:`genevay:18,sejourne:19`,
  or computing differentiable sort and quantile operations :cite:`cuturi:19`.

.. toctree::
    :maxdepth: 1
    :caption: Tutorials:

    notebooks/point_clouds.ipynb
    notebooks/introduction_grid.ipynb

.. toctree::
    :maxdepth: 1
    :caption: Benchmarks:

    notebooks/OTT_&_POT.ipynb
    notebooks/One_Sinkhorn.ipynb
    notebooks/LRSinkhorn.ipynb

.. toctree::
    :maxdepth: 1
    :caption: Advanced Applications:

    notebooks/Sinkhorn_Barycenters.ipynb
    notebooks/gromov_wasserstein.ipynb
    notebooks/GWLRSinkhorn.ipynb
    notebooks/Hessians.ipynb
    notebooks/soft_sort.ipynb
    notebooks/application_biology.ipynb
    notebooks/gromov_wasserstein_multiomics.ipynb
    notebooks/fairness.ipynb
    notebooks/neural_dual.ipynb
    notebooks/icnn_inits.ipynb
    notebooks/wasserstein_barycenters_gmms.ipynb
    notebooks/gmm_pair_demo.ipynb

.. toctree::
    :maxdepth: 1
    :caption: Public API: ott packages

    geometry
    core
    tools

.. toctree::
    :maxdepth: 1
    :caption: References:

    references

.. _Just-in-time (JIT) compilation: https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit
.. _auto-vectorization (VMAP): https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap
.. _automatic: https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation
.. _implicit: https://jax.readthedocs.io/en/latest/_autosummary/jax.custom_jvp.html#jax.custom_jvp
