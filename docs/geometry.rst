ott.geometry package
====================

.. currentmodule:: ott.geometry
.. automodule:: ott.geometry

This package implements several classes to define a geometry, arguably the most influential
ingredient of optimal transport problem. In its full generality, a `Geometry` defines source points
(input measure), target points (target measure) and a ground cost function
(resp. a positive kernel function) that quantifies how expensive
(resp. easy) it is to displace a unit of mass from any of the
input points to the target points.

The geometry package proposes a few simple geometries. The simplest of all would
be that for which input and target points coincide, and the geometry between them
simplifies to a symmetric cost or kernel matrix. In the very particular case
where these points happen to lie on grid (a cartesian product in full generality,
e.g. 2 or 3D grids), the `Grid` geometry will prove useful.

For more general settings where input/target points do not coincide, one can
alternatively instantiate a `Geometry` through a rectangular cost matrix.

However, it is often preferable in applications to define ground costs "symbolically",
by listing instead points in the input/target point clouds, to specify directly
a cost *function* between them. Such functions should follow the `CostFn` class
description. We provide a few standard cost functions that are meaningful in an
OT context, notably the (unbalanced, regularized) Bures distances between
Gaussians [#]_. That cost can be used for instance to compute a distance between
Gaussian mixtures, as proposed in [#]_ and revisited in [#]_.

To be useful with Sinkhorn solvers, `Geometries` typically need to provide an
`epsilon` regularization paramter. We propose either to set that value or
to implement an annealing scheduler.

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
----------

.. [#] H. Janati et al., `Entropic Optimal Transport between Unbalanced Gaussian Measures has a Closed Form <https://proceedings.neurips.cc//paper_files/paper/2020/hash/766e428d1e232bbdd58664b41346196c-Abstract.html>`_ , NeurIPS 2020.
.. [#] Y. Chen et al., `Optimal Transport for Gaussian Mixture Models <https://ieeexplore.ieee.org/document/8590715>`_ , IEEE Access (7)
.. [#] J. Delon and A. Desolneux, `A Wasserstein-Type Distance SIIMS <https://epubs.siam.org/doi/pdf/10.1137/19M1301047>`_ , (13)-2, 936--970
