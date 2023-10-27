ott.solvers
===========
.. module:: ott.solvers

The :mod:`ott.solvers` module contains the main algorithmic engines of the
``OTT`` package. The biggest component in this module are without a doubt the
linear solvers in :mod:`ott.solvers.linear`, designed to solve linear OT
problems. More advanced solvers, notably quadratic in
:mod:`ott.solvers.quadratic`, rely on calls to linear solvers as subroutines.
That property itself is implemented in the more abstract
:class:`~ott.solvers.was_solver.WassersteinSolver` class, which provides a
lower-level template at the interface between the two. Neural based solvers in
:mod:`ott.solvers.nn` live on a different category of their own, since they
typically solve the Monge formulation of OT.

.. toctree::
    :maxdepth: 2

    linear
    quadratic
    nn

Wasserstein Solver
------------------
.. autosummary::
    :toctree: _autosummary

    was_solver.WassersteinSolver
