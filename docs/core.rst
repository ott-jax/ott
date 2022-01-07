ott.core package
================

.. currentmodule:: ott.core
.. automodule:: ott.core

The core package contains the classic ``sinkhorn`` routine (essentially a
wrapper for the ``Sinkhorn`` solver class), along with a
generalization of that algorithm that can be used to compute barycenters
[#]_ [#]_. In addition, we provide a low-rank Sinkhorn solver [#]_ to handle
very large instances, as well as a Gromov-Wasserstein solver (which builds
itself on top of ``Sinkhorn``), and an implementation of input convex neural
networks [#]_


Sinkhorn
---------------------
.. autosummary::
  :toctree: _autosummary

    sinkhorn.sinkhorn
    sinkhorn.Sinkhorn
    discrete_barycenter.discrete_barycenter

Low-Rank Sinkhorn Solver
---------------------
.. autosummary::
  :toctree: _autosummary

    sinkhorn_lr.LRSinkhorn

Gromov-Wasserstein Solver
-------------------------
.. autosummary::
  :toctree: _autosummary

    gromov_wasserstein.gromov_wasserstein

Neural Potentials
-------------------------
.. autosummary::
  :toctree: _autosummary

    icnn.ICNN


References
----------
.. [#] M. Cuturi, `Sinkhorn Distances: Lightspeed Computation of Optimal Transport <https://papers.nips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html>`_ , NIPS 2013.
.. [#] T. Séjourné, `Sinkhorn Divergences for Unbalanced Optimal Transport <https://arxiv.org/abs/1910.12958>`_ , NeurIPS 2019.
.. [#] J.D. Benamou et al., `Iterative Bregman Projections for Regularized Transportation Problems <https://epubs.siam.org/doi/abs/10.1137/141000439>`_ , SIAM J. Sci. Comput. 37(2), A1111-A1138.
.. [#] H. Janati et al., `Debiased Sinkhorn Barycenters <http://proceedings.mlr.press/v119/janati20a.html>`_ , ICML 2020.
.. [#] M. Scetbon et al., `Low-Rank Sinkhorn Factorization <http://proceedings.mlr.press/v139/scetbon21a/scetbon21a.pdf>`_ , ICML 2021.
.. [#] F. Memoli, `Gromov–Wasserstein distances and the metric approach to object matching <https://link.springer.com/article/10.1007/s10208-011-9093-5>`_ , FOCM 2011.
.. [#] B. Amos, L. Xu, J. Z. Kolter, `Input Convex Neural Networks <https://proceedings.mlr.press/v70/amos17b/amos17b.pdf>`_, ICML 2017.