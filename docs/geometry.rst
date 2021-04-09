ott.geometry package
====================

.. currentmodule:: ott.geometry
.. automodule:: ott.geometry

This package implements several ways to define a geometry, arguably the most influential
ingredient of optimal transport problem. A geometry defines source points (input measure), target points (target measure)
as well as the cost of transporting from any of these source points to target points. Alternatively, geometries can also
be exclusively defined through their kernels.

The geometry package proposes a few classic geometries. These can be instantiated in their most basic
form via a cost matrix, or instead by listing points in point clouds and specifying a cost function, or alternatively
by considering points supported on a grid.

We provide a few cost functions that are meaningful in an OT context, notably the (unbalanced, regularized) Bures distances
between Gaussians [#]_. That cost can be used for instance to compute a distance between Gaussian mixtures,
as proposed by [#]_ and revisited by [#]_.

Geometries
----------
.. autosummary::
  :toctree: _autosummary

    geometry.Geometry
    pointcloud.PointCloud
    grid.Grid


Cost Functions
--------------
.. autosummary::
  :toctree: _autosummary

    costs.CostFn
    costs.Euclidean
    costs.Bures
    costs.UnbalancedBures


Regularization Parameter
------------------------
.. autosummary::
  :toctree: _autosummary

    epsilon_scheduler.Epsilon

References
==========

.. [#] H. Janati et al., `Entropic Optimal Transport between Unbalanced Gaussian Measures has a Closed Form <https://proceedings.neurips.cc//paper_files/paper/2020/hash/766e428d1e232bbdd58664b41346196c-Abstract.html>`_ , NeurIPS 2020.
.. [#] Y. Chen et al., `Optimal Transport for Gaussian Mixture Models <https://ieeexplore.ieee.org/document/8590715>`_ , IEEE Access (7)
.. [#] J. Delon and A. Desolneux, `A Wasserstein-Type Distance SIIMS <https://epubs.siam.org/doi/pdf/10.1137/19M1301047>` (13)-2, 936--970
