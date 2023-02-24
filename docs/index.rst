|Downloads| |Tests| |Docs| |Coverage|

Optimal Transport Tools (OTT)
=============================

Introduction
------------
``OTT`` is a `JAX <https://jax.readthedocs.io/en/latest/>`_ package that bundles
a few utilities to compute, and differentiate as needed, the solution to optimal
transport (OT) problems, taken in a fairly wide sense. For instance, ``OTT`` can
of course compute Wasserstein (or Gromov-Wasserstein) distances between weighted
clouds of points (or histograms) in a wide variety of scenarios, but also
estimate Monge maps, Wasserstein barycenters, and help with simpler tasks such
as differentiable approximations to ranking or even clustering.

To achieve this, ``OTT`` rests on two families of tools:

- the first family consists in *discrete* solvers computing transport between
  point clouds, using the Sinkhorn :cite:`cuturi:13` and low-rank Sinkhorn
  :cite:`scetbon:21` algorithms, and moving up towards Gromov-Wasserstein
  :cite:`memoli:11,peyre:16`;
- the second family consists in *continuous* solvers, using suitable neural
  architectures such as an MLP or input-convex neural network
  :cite:`amos:17` coupled with SGD-type estimators
  :cite:`makkuva:20,korotin:21,amos:23`.

Installation
------------
Install ``OTT`` from `PyPI <https://pypi.org/project/ott-jax/>`_ as:

.. code-block:: bash

    pip install ott-jax

or with ``conda`` via `conda-forge`_ as:

.. code-block:: bash

    conda install -c conda-forge ott-jax

Design Choices
--------------
``OTT`` is designed with the following choices:

- Take advantage whenever possible of JAX features, such as
  `Just-in-time (JIT) compilation`_, `auto-vectorization (VMAP)`_ and both
  `automatic`_ but most importantly `implicit`_ differentiation.
- Split geometry from OT solvers in the discrete case: We argue that there
  should be one, and one implementation only, of every major OT algorithm
  (Sinkhorn, Gromov-Wasserstein, barycenters, etc...), regardless of the
  geometric setup that is considered. To give a concrete example, any
  speedups one may benefit from by using a specific cost (e.g. Sinkhorn being
  faster when run on a separable cost on histograms supported on a separable
  grid :cite:`solomon:15`) should not require a separate reimplementation
  of a Sinkhorn routine.
- As a consequence, and to minimize code copy/pasting, use as often as possible
  object hierarchies, and interleave outer solvers (such as quadratic,
  aka Gromov-Wasserstein solvers) with inner solvers (e.g. Low-Rank Sinkhorn).
  This choice ensures that speedups achieved at lower computation levels
  (e.g. low-rank factorization of squared Euclidean distances) propagate
  seamlessly and automatically in higher level calls (e.g. updates in
  Gromov-Wasserstein), without requiring any attention from the user.

.. TODO(marcocuturi): add missing package descriptions below

Packages
--------
- :doc:`geometry` contains classes that instantiate the ground *cost matrix*
  used to specify OT problems. Here cost matrix can be understood in
  a litteral (by actually passing a matrix) or abstract sense (by passing
  information that is sufficient to recreate that matrix, apply all or parts
  of it, or apply its kernel). A typical example in the latter case arises
  when comparing *two point clouds*, paired with a *cost function*. Geometry
  objects are used to describe OT *problems*, solved next by *solvers*.
- :doc:`problems/index` are used to describe linear or quadratic (GW) OT
  problems.
- :doc:`solvers/index` solve linear or quadratic problems with various
  techniques, including some neural approaches.
- :doc:`initializers/index` are used to speed up the resolution of OT
  solvers.
- :doc:`tools` provides an interface to exploit OT solutions, as produced by
  solvers in the solvers. Such tasks include computing approximations
  to Wasserstein distances :cite:`genevay:18,sejourne:19`, approximating OT
  between GMMs, or computing differentiable sort and quantile operations
  :cite:`cuturi:19`.
- :doc:`math` holds low-level mathematical primitives.
- :doc:`utils` provides misc helper functions

.. toctree::
    :maxdepth: 1
    :caption: Examples

    Getting Started <tutorials/notebooks/point_clouds>
    tutorials/index

.. toctree::
    :maxdepth: 1
    :caption: API

    geometry
    problems/index
    solvers/index
    initializers/index
    tools
    math
    utils

.. toctree::
    :maxdepth: 1
    :caption: References

    references

.. |Downloads| image:: https://static.pepy.tech/badge/ott-jax
    :target: https://pypi.org/project/ott-jax/
    :alt: Documentation

.. |Tests| image:: https://img.shields.io/github/actions/workflow/status/ott-jax/ott/tests.yml?branch=main
    :target: https://github.com/ott-jax/ott/actions/workflows/tests.yml
    :alt: Documentation

.. |Docs| image:: https://img.shields.io/readthedocs/ott-jax/latest
    :target: https://ott-jax.readthedocs.io/en/latest/
    :alt: Documentation

.. |Coverage| image:: https://img.shields.io/codecov/c/github/ott-jax/ott/main
    :target: https://app.codecov.io/gh/ott-jax/ott
    :alt: Coverage

.. _Just-in-time (JIT) compilation: https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit
.. _auto-vectorization (VMAP): https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap
.. _automatic: https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation
.. _implicit: https://jax.readthedocs.io/en/latest/_autosummary/jax.custom_jvp.html#jax.custom_jvp
.. _conda-forge: https://anaconda.org/conda-forge/ott-jax
