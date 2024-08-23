ott.geometry
============
.. module:: ott.geometry
.. currentmodule:: ott.geometry

This package implements several classes to define a geometry, arguably the most
influential ingredient of optimal transport problem. In its full generality, a
:class:`~ott.geometry.geometry.Geometry` defines source points (input measure),
target points (target measure) and a ground cost function (resp. a positive
kernel function) that quantifies how expensive (resp. easy) it is to displace a
unit of mass from any of the input points to the target points.

The geometry package proposes a few simple geometries. The simplest of all would
be that for which input and target points coincide, and the geometry between
them simplifies to a symmetric cost or kernel matrix. In the very particular
case where these points happen to lie on grid (a Cartesian product in full
generality, e.g., 2- or-3-dimensional grids), the
:class:`~ott.geometry.grid.Grid` geometry will prove useful.

For more general settings where input/target points do not coincide, one can
alternatively instantiate a :class:`~ott.geometry.geometry.Geometry` through a
rectangular cost matrix.

However, it is often preferable in applications to define ground costs
"symbolically", by listing instead points in the input/target point clouds, to
specify directly a cost *function* between them. Such functions should follow
the :class:`~ott.geometry.costs.CostFn` class description. We provide a few
standard cost functions that are meaningful in an OT context, notably the
(unbalanced, regularized) Bures distances between Gaussians :cite:`janati:20`.
That cost can be used for instance to compute a distance between Gaussian
mixtures, as proposed in :cite:`chen:19a` and revisited in :cite:`delon:20`.

To be useful with :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` solvers,
:class:`Geometries <ott.geometry.geometry.Geometry>` typically need to provide
an ``epsilon`` regularization parameter. We propose either to set that value
once for all, or implement an annealing
:class:`~ott.geometry.epsilon_scheduler.Epsilon` scheduler.

Geometries
----------
.. autosummary::
    :toctree: _autosummary

    geometry.Geometry
    pointcloud.PointCloud
    grid.Grid
    graph.Graph
    geodesic.Geodesic
    low_rank.LRCGeometry
    low_rank.LRKGeometry
    epsilon_scheduler.Epsilon

Cost Functions
--------------
.. autosummary::
    :toctree: _autosummary

    costs.CostFn
    costs.TICost
    costs.SqPNorm
    costs.PNormP
    costs.SqEuclidean
    costs.RegTICost
    costs.Euclidean
    costs.Cosine
    costs.Arccos
    costs.Bures
    costs.UnbalancedBures
    costs.SoftDTW
    distrib_costs.UnivariateWasserstein

Regularizers
------------
.. autosummary::
    :toctree: _autosummary

    regularizers.ProximalOperator
    regularizers.PostComposition
    regularizers.Regularization
    regularizers.Orthogonal
    regularizers.Quadratic
    regularizers.L1
    regularizers.L2
    regularizers.STVS
    regularizers.SqKOverlap

Utilities
---------
.. autosummary::
    :toctree: _autosummary

    segment.segment_point_cloud
