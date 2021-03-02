ott.geometry package
====================

.. currentmodule:: ott.geometry
.. automodule:: ott.geometry

This package implements several ways to define a geometry, arguably the most influential
ingredient of optimal transport problem. A geometry defines source points (input measure), target points (target measure)
as well as the cost of transporting from any of these source points to target points.

The geometry package proposes a few classic geometries. These can be instantiated in their most basic
form via a cost matrix, or instead by listing points in point clouds and specifying a cost function, or alternatively 
by considering points supported on a grid.


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


Regularization Parameter
------------------------
.. autosummary::
  :toctree: _autosummary

    epsilon_scheduler.Epsilon
