ott.core package
================
.. currentmodule:: ott.core
.. automodule:: ott.core

The core package contains definitions of various OT problems, starting
from the most simple, the linear OT problem, to more advanced problems
such as quadratic, or involving multiple measures, the barycenter problem.
We follow with the classic :class:`ott.core.sinkhorn.sinkhorn` routine (essentially a
wrapper for the :class:`ott.core.sinkhorn.Sinkhorn` solver class) [#]_, [#]_. We also provide an analogous
low-rank Sinkhorn solver [#]_ to handle very large instances. Both are used
within our Wasserstein barycenter solvers [#]_, [#]_ as well as our
Gromov-Wasserstein solver [#]_, [#]_. We also provide an implementation of
input convex neural networks [#]_, a NN that can be used to estimate OT [#]_.

OT Problems
-----------
.. autosummary::
    :toctree: _autosummary

    linear_problems.LinearProblem
    quad_problems.QuadraticProblem
    bar_problems.BarycenterProblem
    bar_problems.GWBarycenterProblem

Sinkhorn
--------
.. autosummary::
    :toctree: _autosummary

    sinkhorn.sinkhorn
    sinkhorn.Sinkhorn
    sinkhorn.SinkhornOutput

Low-Rank Sinkhorn
-----------------
.. autosummary::
    :toctree: _autosummary

    sinkhorn_lr.LRSinkhorn
    sinkhorn_lr.LRSinkhornOutput

Barycenters (Entropic and LR)
-----------------------------
.. autosummary::
    :toctree: _autosummary

    discrete_barycenter.discrete_barycenter
    continuous_barycenter.WassersteinBarycenter
    continuous_barycenter.BarycenterState
    gw_barycenter.GromovWassersteinBarycenter
    gw_barycenter.GWBarycenterState

Gromov-Wasserstein (Entropic and LR)
------------------------------------
.. autosummary::
    :toctree: _autosummary

    gromov_wasserstein.gromov_wasserstein
    gromov_wasserstein.GromovWasserstein
    gromov_wasserstein.GWOutput

Neural Potentials
-----------------
.. autosummary::
    :toctree: _autosummary

    icnn.ICNN
    neuraldual.NeuralDualSolver
    neuraldual.NeuralDual

Utilities
---------
.. autosummary::
    :toctree: _autosummary

    segment.segment_point_cloud

References
----------
.. [#] M. Cuturi, `Sinkhorn Distances: Lightspeed Computation of Optimal Transport <https://papers.nips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html>`_ , NIPS 2013.
.. [#] T. Séjourné, `Sinkhorn Divergences for Unbalanced Optimal Transport <https://arxiv.org/abs/1910.12958>`_ , NeurIPS 2019.
.. [#] M. Scetbon et al., `Low-Rank Sinkhorn Factorization <http://proceedings.mlr.press/v139/scetbon21a/scetbon21a.pdf>`_ , ICML 2021.
.. [#] J.D. Benamou et al., `Iterative Bregman Projections for Regularized Transportation Problems <https://epubs.siam.org/doi/abs/10.1137/141000439>`_ , SIAM J. Sci. Comput. 37(2), A1111-A1138.
.. [#] H. Janati et al., `Debiased Sinkhorn Barycenters <http://proceedings.mlr.press/v119/janati20a.html>`_ , ICML 2020.
.. [#] F. Memoli, `Gromov–Wasserstein distances and the metric approach to object matching <https://link.springer.com/article/10.1007/s10208-011-9093-5>`_ , FOCM 2011.
.. [#] M. Scetbon et al., `Linear-Time Gromov Wasserstein Distances using Low Rank Couplings and Costs <https://arxiv.org/abs/2106.01128>`_, Arxiv.
.. [#] B. Amos, L. Xu, J. Z. Kolter, `Input Convex Neural Networks <https://proceedings.mlr.press/v70/amos17b/amos17b.pdf>`_, ICML 2017.
.. [#] Ashok Vardhan Makkuva, Amirhossein Taghvaei, Sewoong Oh, Jason D. Lee,  `Optimal transport mapping via input convex neural networks <https://arxiv.org/abs/1908.10962>`_ , ICML 2020
