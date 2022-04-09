.. ott documentation master file, created by
   sphinx-quickstart on Mon Feb  1 14:10:11 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Optimal Transport Tools (OTT) documentation
===========================================

`Code <https://github.com/ott-jax/ott>`_ hosted on Github. To install,
clone that repo or simply run ``pip install ott-jax``.

Intro
-----
OTT is a `JAX <https://jax.readthedocs.io/en/latest/index.html>`_ package that bundles a few utilities to compute and differentiate the
solution to optimal transport problems. OTT can help you compute Wasserstein distances between
weighted clouds of points (or histograms), using a cost (e.g. a distance) between individual points.

To that end OTT uses various implementation of
the Sinkhorn algorithm [#]_ [#]_ [#]_. These implementation take advantage of several
JAX features, such as `Just-in-time (JIT) compilation <https://jax.readthedocs.io/en/latest/jax.html?highlight=jit#just-in-time-compilation-jit>`_,
`auto-vectorization (VMAP) <https://jax.readthedocs.io/en/latest/jax.html?highlight=jit#jax.vmap>`_, and
both `automatic <https://jax.readthedocs.io/en/latest/jax.html?highlight=jit#automatic-differentiation>`_
and/or `implicit <https://jax.readthedocs.io/en/latest/jax.html?highlight=jit#jax.custom_vjp>`_
differentiation. A few tutorial snippets are provided below, along with different use-cases, notably for single-cell genomics data [#]_.

Packages
--------


There are currently three packages, ``geometry``, ``core`` and ``tools``, playing the following roles:

- ``geometry`` defines classes that describe *two point clouds* paired with a *cost* function (simpler geometries are also implemented, such as that defined by points supported on a multi-dimensional grids with a separable cost [#]_). The design choice in OTT is to state that cost functions and algorithms should operate independently: if a particular cost function allows for faster computations (e.g. squared-Euclidean distance when comparing grids), this should not be taken advantage of at the level of optimizers, but at the level of the problems description. Geometry objects are therefore only considered as arguments to describe OT problem handled in ``core``, using subroutines provided by geometries;
- ``core`` help define first an OT problem (linear, quadratic, barycenters). These problems are then solved using Sinkhorn algorithm and its variants, the main workhorse to solve OT in this package, as well as variants that can comppute Gromov-Wasserstein distances or barycenters of several measures;
- ``tools`` provides an interface to exploit OT solutions, as produced by ``core`` functions. Such tasks include instantiating OT matrices, computing approximations to Wasserstein distances [#]_ [#]_, or computing differentiable sort and quantile operations [#]_.


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
   notebooks/fairness.ipynb
   notebooks/neural_dual.ipynb


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

References
==========

.. [#] M. Cuturi, `Sinkhorn Distances: Lightspeed Computation of Optimal Transport <https://papers.nips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html>`_, NIPS'13.
.. [#] G. Peyré, M. Cuturi, `Computational Optimal Transport <https://www.nowpublishers.com/article/Details/MAL-073>`_, FNT in ML, 2019.
.. [#] M. Scetbon et al., `Low-Rank Sinkhorn Factorization <http://proceedings.mlr.press/v139/scetbon21a/scetbon21a.pdf>`_ , ICML 2021.
.. [#] G. Schiebinger et al., `Optimal-Transport Analysis of Single-Cell Gene Expression Identifies Developmental Trajectories in Reprogramming <https://www.cell.com/cell/pdf/S0092-8674(19)30039-X.pdf>`_, Cell 176, 928--943.
.. [#] J. Solomon et al, `Convolutional Wasserstein distances: efficient optimal transportation on geometric domains <https://dl.acm.org/doi/10.1145/2766963>`_, ACM ToG, SIGGRAPH'15.
.. [#] A. Genevay et al., `Learning Generative Models with Sinkhorn Divergences <http://proceedings.mlr.press/v84/genevay18a.html>`_, AISTATS'18.
.. [#] T. Séjourné et al., `Sinkhorn Divergences for Unbalanced Optimal Transport <https://arxiv.org/abs/1910.12958>`_, arXiv:1910.12958.
.. [#] M. Cuturi et al. `Differentiable Ranking and Sorting using Optimal Transport <https://papers.nips.cc/paper/2019/hash/d8c24ca8f23c562a5600876ca2a550ce-Abstract.html>`_, NeurIPS'19.
