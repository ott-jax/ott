ott.core package
================

.. currentmodule:: ott.core
.. automodule:: ott.core

The core package contains the classic Sinkhorn routine, along with a
generalization of that algorithm that can be used to compute barycenters [#]_ [#]_.
We also provide a Gromov-Wasserstein solver, which builds itself on top of Sinkhorn.


Sinkhorn and variants
---------------------
.. autosummary::
  :toctree: _autosummary

    sinkhorn.sinkhorn
    discrete_barycenter.discrete_barycenter

Gromov-Wasserstein Solver
-------------------------
.. autosummary::
  :toctree: _autosummary

    gromov_wasserstein.gromov_wasserstein

References
==========

.. [#] J.D. Benamou et al., `Iterative Bregman Projections for Regularized Transportation Problems <https://epubs.siam.org/doi/abs/10.1137/141000439>`_ , SIAM J. Sci. Comput. 37(2), A1111-A1138.
.. [#] H. Janati et al., `Debiased Sinkhorn Barycenters <http://proceedings.mlr.press/v119/janati20a.html>`_ , ICML 2020.
.. [#] F. Memoli, `Gromov–Wasserstein distances and the metric approach to object matching <https://link.springer.com/article/10.1007/s10208-011-9093-5>`_ , FOCM 2011 