.. ott documentation master file, created by
   sphinx-quickstart on Mon Feb  1 14:10:11 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Optimal Transport Tools (OTT) documentation
===========================================

`Code <https://github.com/google-research/ott>`_ hosted on Github.

Intro
-----
OTT is a `JAX <https://jax.readthedocs.io/en/latest/index.html>`_ toolbox that bundles a few utilities to compute and differentiate the
solution to optimal transport problems. OTT can help you quantify how different two
weighted clouds of points (or histograms) are, using a cost (e.g. a distance) between individual points.

To that end OTT uses a sturdy and versatile implementation of
the Sinkhorn algorithm [#]_ [#]_. This implementation takes advantage of several
JAX features, such as `Just-in-time compilation <https://jax.readthedocs.io/en/latest/jax.html?highlight=jit#just-in-time-compilation-jit>`_,
`auto-vectorization <https://jax.readthedocs.io/en/latest/jax.html?highlight=jit#jax.vmap>`_, and
both `automatic <https://jax.readthedocs.io/en/latest/jax.html?highlight=jit#automatic-differentiation>`_ 
and/or `implicit <https://jax.readthedocs.io/en/latest/jax.html?highlight=jit#jax.custom_vjp>`_
differentiation. Some simple examples are provided in the tutorial notebooks below, notably to single-cell genomics [#]_.

Packages
--------


There are currently three packages, ``geometry``, ``core`` and ``tools``, playing the following roles:

- ``geometry`` defines classes that describe *two point clouds* paired with a *cost* function (simpler geometries are also implemented, such as that defined by points supported on a multi-dimensional grids with a separable cost [#]_).
  A geometry, along with weight vectors ``a`` and ``b``, describe an OT problem. Geometries provide the subroutines that are needed by ``core`` algorithms to solve OT problems.
- ``core`` contains the Sinkhorn algorithm, the main workhorse used in this package, as well as variants that can be used to compute barycenters;
- ``tools`` builds on top of outputs produced by ``core`` functions, to carry out standard OT tasks
such as instantiating OT matrices, computing OT divergences [#]_ [#]_, or computing soft-sort and soft-quantiles [#]_.

.. toctree::
   :maxdepth: 1
   :caption: Tutorials:

   notebooks/point_clouds.ipynb
   notebooks/introduction_grid.ipynb
   notebooks/application_biology.ipynb


.. toctree::
   :maxdepth: 1
   :caption: Public API: ott packages

   geometry
   core
   tools


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. [#] G. Schiebinger et al., `Optimal-Transport Analysis of Single-Cell Gene Expression Identifies Developmental Trajectories in Reprogramming <https://www.cell.com/cell/pdf/S0092-8674(19)30039-X.pdf>`_, Cell 176, 928--943.
.. [#] M. Cuturi, `Sinkhorn Distances: Lightspeed Computation of Optimal Transport <https://papers.nips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html>`_, NIPS'13.
.. [#] G. Peyré, M. Cuturi, `Computational Optimal Transport <https://www.nowpublishers.com/article/Details/MAL-073>`_, FNT in ML, 2019.
.. [#] J. Solomon et al, `Convolutional Wasserstein distances: efficient optimal transportation on geometric domains <https://dl.acm.org/doi/10.1145/2766963>`_, ACM ToG, SIGGRAPH'15.
.. [#] A. Genevay et al., `Learning Generative Models with Sinkhorn Divergences <http://proceedings.mlr.press/v84/genevay18a.html>`_, AISTATS'18.
.. [#] T. Séjourné et al., `Sinkhorn Divergences for Unbalanced Optimal Transport <https://arxiv.org/abs/1910.12958>`_, arXiv:1910.12958.
.. [#] M. Cuturi et al. `Differentiable Ranking and Sorting using Optimal Transport <https://papers.nips.cc/paper/2019/hash/d8c24ca8f23c562a5600876ca2a550ce-Abstract.html>`_, NeurIPS'19.