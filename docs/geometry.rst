ott.geometry package
====================

.. currentmodule:: ott.geometry
.. automodule:: ott.geometry

This package implements several ways to define a geometry. The geometry is key
in Optimal Transport, since it defines how much it costs to transport mass from
a source to a target. The geometry package proposes several implementation of
such geometries: via a cost matrix, by a point cloud and cost function or by a
grid.


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
