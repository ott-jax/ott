ott.geometry package
====================
.. currentmodule:: ott.geometry
.. automodule:: ott.geometry

This package implements several classes to define a geometry, arguably the most influential
ingredient of optimal transport problem. In its full generality, a :class:`ott.geometry.geometry.Geometry`
defines source points (input measure), target points (target measure) and a ground cost function
(resp. a positive kernel function) that quantifies how expensive (resp. easy) it is to displace
a unit of mass from any of the input points to the target points.

The geometry package proposes a few simple geometries. The simplest of all would
be that for which input and target points coincide, and the geometry between them
simplifies to a symmetric cost or kernel matrix. In the very particular case
where these points happen to lie on grid (a cartesian product in full generality,
e.g. 2 or 3D grids), the :class:`ott.geometry.grid.Grid` geometry will prove useful.

For more general settings where input/target points do not coincide, one can
alternatively instantiate a :class:`ott.geometry.geometry.Geometry` through a rectangular cost matrix.

However, it is often preferable in applications to define ground costs "symbolically",
by listing instead points in the input/target point clouds, to specify directly
a cost *function* between them. Such functions should follow the :class:`ott.geometry.costs.CostFn`
class description. We provide a few standard cost functions that are meaningful in an
OT context, notably the (unbalanced, regularized) Bures distances between
Gaussians [#]_. That cost can be used for instance to compute a distance between
Gaussian mixtures, as proposed in [#]_ and revisited in [#]_.

To be useful with Sinkhorn solvers, ``Geometries`` typically need to provide an
``epsilon`` regularization parameter. We propose either to set that value once for
all, or implement an annealing scheduler :class:`ott.geometry.epsilon_scheduler.Epsilon`.

Geometries
----------
.. autosummary::
    :toctree: _autosummary

    geometry.Geometry
    pointcloud.PointCloud
    grid.Grid
    low_rank.LRCGeometry
    epsilon_scheduler.Epsilon


Cost Functions
--------------
.. autosummary::
    :toctree: _autosummary

    costs.CostFn
    costs.Euclidean
    costs.Cosine
    costs.Bures
    costs.UnbalancedBures

References
----------
.. [#] H. Janati et al., `Entropic Optimal Transport between Unbalanced Gaussian Measures has a Closed Form <https://proceedings.neurips.cc//paper_files/paper/2020/hash/766e428d1e232bbdd58664b41346196c-Abstract.html>`_ , NeurIPS 2020.
.. [#] Y. Chen et al., `Optimal Transport for Gaussian Mixture Models <https://ieeexplore.ieee.org/document/8590715>`_ , IEEE Access (7)
.. [#] J. Delon and A. Desolneux, `A Wasserstein-Type Distance in the Space of Gaussian Mixture Models <https://epubs.siam.org/doi/pdf/10.1137/19M1301047>`_ , SIIMS  (13)-2, 936--970
