ott.tools package
=================

.. currentmodule:: ott.tools
.. automodule:: ott.tools

The tools package contains high level functions that build on outputs produced by core functions.
They can be used to compute Sinkhorn divergences [#]_, instantiate transport matrices,
provide differentiable approximations to ranks and quantile functions [#]_, etc.


Optimal Transport
-----------------
.. autosummary::
  :toctree: _autosummary

    transport.Transport


Sinkhorn Divergence
-------------------
.. autosummary::
  :toctree: _autosummary

    sinkhorn_divergence.sinkhorn_divergence


Soft Sorting algorithms
-----------------------
.. autosummary::
  :toctree: _autosummary

    soft_sort.softsort
    soft_sort.softranks
    soft_sort.softquantile

.. [#] T. Séjourné et al., `Sinkhorn Divergences for Unbalanced Optimal Transport <https://arxiv.org/abs/1910.12958>`_, arXiv:1910.12958.
.. [#] M. Cuturi et al. `Differentiable Ranking and Sorting using Optimal Transport <https://papers.nips.cc/paper/2019/hash/d8c24ca8f23c562a5600876ca2a550ce-Abstract.html>`_, NeurIPS'19.