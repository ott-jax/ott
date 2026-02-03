|Downloads| |Tests| |Docs| |Coverage|

Optimal Transport Tools (OTT)
=============================

Introduction
------------
``OTT`` is a `JAX <https://jax.readthedocs.io/en/latest/>`_ package that bundles
a few utilities to compute, and differentiate as needed, the solution to optimal
transport (OT) problems, taken in a fairly wide sense. For instance, ``OTT`` can
compute the :term:`Wasserstein distance`
(or :term:`Gromov-Wasserstein distance`) between weighted point clouds
(or histograms) in a wide variety of scenarios,
but also estimate a :term:`Monge map`, :term:`Wasserstein barycenter`, or even
help with simpler tasks such as differentiable approximations to ranking or
clustering.

To achieve this, ``OTT`` rests on two families of tools:

- the first family consists in *discrete* solvers computing transport between
  two families of points or histograms using e.g. the :term:`Sinkhorn algorithm`
  :cite:`cuturi:13` or low-rank solvers :cite:`scetbon:21`, with further
  extensions to more advanced scenarios such as the
  :term:`Gromov-Wasserstein problem` :cite:`memoli:11,peyre:16`;
- the second family consists in *continuous* solvers, whose goal is to output,
  given two point cloud samples, a *function* that is an approximate
  :term:`Monge map`, a :term:`transport map` that can map efficiently the first
  measure to the second. Such functions can be recovered using directly tools
  above, notably the family of :term:`entropic map` approximations. Such maps
  can also be parameterized as neural architectures such as an MLP or as
  gradients of :term:`input convex neural network` :cite:`amos:17`, trained with
  advanced SGD approaches :cite:`makkuva:20,korotin:21,amos:23`. Such functions
  can also be parameterized as Neural ODEs, where to an input source point is
  associated the end-result of a path integral with a time-parameterized
  velocity field. Such velocity fields can be parameterized as neural networks
  and learned from data, using the framework of flow matching
  :cite:`lipman:22,albergo:23` augmented with OT couplings for noise/data
  :cite:`pooladian:23,tong:23,zhang:25,mousavi:25`.

Installation
------------
Install ``OTT`` from `PyPI <https://pypi.org/project/ott-jax/>`_ as:

.. code-block:: bash

    pip install ott-jax

or with the :mod:`neural OT <ott.neural>` dependencies:

.. code-block:: bash

    pip install 'ott-jax[neural]'

or using `conda`_ as:

.. code-block:: bash

    conda install -c conda-forge ott-jax

Design Choices
--------------
``OTT`` is designed with the following choices:

- Take advantage whenever possible of JAX features, such as
  `just-in-time (JIT) compilation`_, `auto-vectorization (VMAP)`_ and both
  `automatic`_ and `implicit`_ differentiation.
- Split geometry from OT solvers in the discrete case: you will find one, and
  one implementation only, of every major OT algorithm
  (Sinkhorn, Gromov-Wasserstein, barycenters, etc...), that are all agnostic to
  speedups one may benefit from by using a specific cost (e.g. Sinkhorn being
  the geometric (i.e. the cost function) setup. To give a concrete example, if
  the inner operations in the :term:`Sinkhorn algorithm` can be run more
  efficiently (because e.g. the cost function is low-rank, or the cost is a
  separable function for points supported on on a separable grid
  :cite:`solomon:15`), this should not trigger a separate reimplementation
  of the :term:`Sinkhorn algorithm`.
- As a consequence, and to minimize code copy/pasting, use as often as possible
  object hierarchies, and interleave outer solvers (such as quadratic,
  aka Gromov-Wasserstein solvers) with inner solvers (e.g., low-rank Sinkhorn).
  This choice ensures that speedups achieved at lower computation levels
  (e.g. low-rank factorization of squared Euclidean distances) propagate
  seamlessly and automatically in higher level calls (e.g. updates in
  Gromov-Wasserstein), without requiring any attention from the user.

Packages
--------
.. module:: ott

- :mod:`ott.geometry` contains classes that instantiate the ground *cost matrix*
  used to specify a :term:`Kantorovich problem`. Here cost matrix can be
  both understood in a literal (by instantiating a matrix) or abstract (by
  storing information that is sufficient to recreate that matrix, apply all or
  parts of it, or apply its kernel) sense. An important case is handled by the
  :class:`~ott.geometry.pointcloud.PointCloud` class which specifies
  *two point clouds*, paired with a *cost function* (to be chosen within
  :mod:`ott.geometry.costs`). Geometry objects are used to describe
  OT *problems*, solved next by *solvers*.
- :mod:`ott.problems` are used to describe the interactions between multiple
  measures, to define linear (a.k.a. :term:`Kantorovich problem`), quadratic
  (a.k.a. :term:`Gromov-Wasserstein problem`) or :term:`Wasserstein barycenter`
  problems.
- :mod:`ott.solvers` solve a problem instantiated with :mod:`ott.problems` using
  one among many implemented approaches.
- :mod:`ott.initializers` implement simple strategies to initialize the solvers
  above. When the problems are solved with a convex solver, such as a
  :class:`~ott.problems.linear.linear_problem.LinearProblem` solved with a
  :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` solver, the resolution of OT
  solvers, then this initialization is mostly useful to speed up convergences.
  When the problem is *not* convex, which is the case for most other uses of
  this toolbox, the initialization can play a decisive role to reach a useful
  solution.
- :mod:`ott.experimental` lists tools whose API is not mature yet to be included
  in the main toolbox, with changes expected in the near future, but which might
  still prove useful for users. This includes at the moment the
  :class:`~ott.solvers.linear.mmsinkhorn.MMSinkhorn` solver class to compute an
  optimal :term:`multimarginal coupling`
- :mod:`ott.neural` provides tools to parameterize and compute an optimal
  :term:`transport map` as a neural network. Such networks can be parameterized
  as an :term:`input convex neural network`, and trained to approximate the
  :term:`Brenier potential` between two measures. Alternatively, one can
  parameterize that map as a neural ODE, using a time-dependent velocity field
  trained with :mod:`~ott.neural.methods.flow_matching`
  :cite:`lipman:22,albergo:23` using independent couplings or more advanced OT
  couplings :cite:`pooladian:23,tong:23,zhang:25,mousavi:25`.
- :mod:`ott.tools` provides an interface to exploit OT solutions produced by
  solvers from the :mod:`ott.solvers` module. Such tasks include computing
  approximations to Wasserstein distances :cite:`genevay:18,sejourne:19`,
  approximating OT between GMMs, or computing differentiable sort and quantile
  operations :cite:`cuturi:19`. That module also provides plotting tools to
  display OT solutions.
- :mod:`ott.math` holds low-level miscellaneous mathematical primitives, such as
  an implementation of the matrix square-root, or the
  :term:`Legendre transform`.
- :mod:`ott.utils` provides miscellaneous helper functions.

.. toctree::
    :maxdepth: 1
    :caption: Examples

    Getting Started <tutorials/basic_ot_between_datasets>
    tutorials/index

.. toctree::
    :maxdepth: 1
    :caption: API

    geometry
    problems/index
    solvers/index
    initializers/index
    neural/index
    experimental/index
    tools
    math
    utils

.. toctree::
    :maxdepth: 1
    :caption: References

    glossary
    bibliography
    contributing

.. |Downloads| image:: https://static.pepy.tech/personalized-badge/ott-jax?period=total&units=INTERNATIONAL_SYSTEM&left_color=GREY&right_color=BLUE&left_text=downloads
    :target: https://pypi.org/project/ott-jax/
    :alt: Documentation

.. |Tests| image:: https://img.shields.io/github/actions/workflow/status/ott-jax/ott/tests.yml?branch=main
    :target: https://github.com/ott-jax/ott/actions/workflows/tests.yml
    :alt: Documentation

.. |Docs| image:: https://img.shields.io/readthedocs/ott-jax
    :target: https://ott-jax.readthedocs.io
    :alt: Documentation

.. |Coverage| image:: https://img.shields.io/codecov/c/github/ott-jax/ott/main
    :target: https://app.codecov.io/gh/ott-jax/ott
    :alt: Coverage

.. _Just-in-time (JIT) compilation: https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit
.. _auto-vectorization (VMAP): https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html#jax.vmap
.. _automatic: https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation
.. _implicit: https://jax.readthedocs.io/en/latest/_autosummary/jax.custom_jvp.html#jax.custom_jvp
.. _conda: https://anaconda.org/channels/conda-forge/packages/ott-jax/overview
