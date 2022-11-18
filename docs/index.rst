Optimal Transport Tools (OTT) documentation
===========================================
`Code <https://github.com/ott-jax/ott>`_ on github.
To install, simply run ``pip install ott-jax``.

Intro
-----
`OTT` is a `JAX <https://jax.readthedocs.io/en/latest/index.html>`_ package that bundles a few utilities to compute, and
differentiate as needed, the solution to optimal transport (OT) problems, taken in a fairly wide sense.
For instance, `OTT` can of course compute Wasserstein (or Gromov-Wasserstein) distances between
weighted clouds of points (or histograms) in a wide variety of scenarios,
but also estimate Monge maps, Wasserstein barycenters, and help with simpler tasks
such as differentiable approximations to ranking or even clustering.

To achieve this, `OTT` rests on two families of tools:
The first family consists in *discrete* solvers computing transport between point clouds,
using the Sinkhorn :cite:`cuturi:13` and low-rank Sinkhorn :cite:`scetbon:21` algorithms,
and moving up towards Gromov-Wasserstein :cite:`memoli:11`, :cite:`memoli:11`;
the second family consists in *continuous* solvers, using suitable neural architectures :cite:`amos:17` coupled
with SGD type estimators :cite:`makkuva:20`, :cite:`korotin:21`.

Design Choices
--------------

`OTT` is designed with the following choices:

- Take advantage whenever possible of JAX features, such as `Just-in-time (JIT) compilation`_,
  `auto-vectorization (VMAP)`_ and both `automatic`_ but most importantly `implicit`_ differentiation.
- Split geometry from OT solvers in the discrete case: We argue that there
  should be one, and one implementation only, of every major OT algorithm
  (Sinkhorn, Gromov-Wasserstein, barycenters, etc...), regardless of the
  geometric setup that is considered. To give a concrete example, any
  speedups one may benefit from by using a specific cost
  (e.g. Sinkhorn being faster when run on a separable cost on histograms supported
  on a separable grid :cite:`solomon:15`) should not require a separate
  reimplementation of a Sinkhorn routine.
- As a consequence, and to minimize code copy/pasting, use as often as possible
  object hierarchies, and interleave outer solvers (such as quadratic,
  aka Gromov-Wasserstein solvers) with inner solvers (e.g. Low-Rank Sinkhorn).
  This choice ensures that speedups achieved at lower computation levels
  (e.g. low-rank factorization of squared Euclidean distances) propagate seamlessly and
  automatically in higher level calls (e.g. updates in Gromov-Wasserstein),
  without requiring any attention from the user.

Packages
--------
There are currently three packages, ``geometry``, ``core`` and ``tools``, playing the following roles:

- :ref:`geometry` contains classes to instantiate objects that describe
  *two point clouds* paired with a *cost* function. Geometry objects are used to
  describe OT problems, handled by solvers in ``core``.
- :ref:`core` classes describe OT problems (linear, quadratic, barycenters), and
  solver classes, to instantiate algorithms that will output an OT.
- :ref:`tools` provides an interface to exploit OT solutions, as produced by
  solvers in the ``core`` package. Such tasks include computing approximations
  to Wasserstein distances :cite:`genevay:18,sejourne:19`, approximating OT
  between GMMs, or computing differentiable sort and quantile operations
  :cite:`cuturi:19`.


.. toctree::
    :maxdepth: 1
    :caption: Public API: ott packages

    geometry
    problems/index
    solvers/index

.. toctree::
    :maxdepth: 1
    :caption: References:

    references

.. _Just-in-time (JIT) compilation: https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit
.. _auto-vectorization (VMAP): https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap
.. _automatic: https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation
.. _implicit: https://jax.readthedocs.io/en/latest/_autosummary/jax.custom_jvp.html#jax.custom_jvp
