<img src="https://raw.githubusercontent.com/ott-jax/ott/main/docs/_static/images/logoOTT.png" width="10%" alt="logo">

# Optimal Transport Tools (OTT)
[![Downloads](https://static.pepy.tech/badge/ott-jax)](https://pypi.org/project/ott-jax/)
[![Tests](https://img.shields.io/github/actions/workflow/status/ott-jax/ott/tests.yml?branch=main)](https://github.com/ott-jax/ott/actions/workflows/tests.yml)
[![Docs](https://img.shields.io/readthedocs/ott-jax/latest)](https://ott-jax.readthedocs.io/en/latest/)
[![Coverage](https://img.shields.io/codecov/c/github/ott-jax/ott/main)](https://app.codecov.io/gh/ott-jax/ott)

**See the [full documentation](https://ott-jax.readthedocs.io/en/latest/).**

## What is OTT-JAX?
A ``JAX`` powered library to compute optimal transport at scale and on accelerators, ``OTT-JAX`` includes the fastest
implementation of the Sinkhorn algorithm you will find around. We have implemented all tweaks (scheduling, momentum, acceleration, initializations) and extensions (low-rank, entropic maps). They can be used directly between two datasets, or within more advanced problems
(Gromov-Wasserstein, barycenters). Some of ``JAX`` features, including
[JIT](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html#Using-jit-to-speed-up-functions),
[auto-vectorization](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html#Auto-vectorization-with-vmap) and
[implicit differentiation](https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html)
work towards the goal of having end-to-end differentiable outputs. ``OTT-JAX`` is led by a team of researchers at Apple, with contributions from Google and Meta researchers, as well as many academic partners, including TU München, Oxford, ENSAE/IP Paris, ENS Paris and the Hebrew University.

## Installation
Install ``OTT-JAX`` from [PyPI](https://pypi.org/project/ott-jax/) as:
```bash
pip install ott-jax
```
or with ``conda`` via [conda-forge](https://anaconda.org/conda-forge/ott-jax) as:
```bash
conda install -c conda-forge ott-jax
```

## What is optimal transport?
Optimal transport can be loosely described as the branch of mathematics and optimization that studies
*matching problems*: given two families of points, and a cost function on pairs of points, find a "good" (low cost) way
to associate bijectively to every point in the first family another in the second.

Such problems appear in all areas of science, are easy to describe, yet hard to solve. Indeed, while matching optimally
two sets of $n$ points using a pairwise cost can be solved with the
[Hungarian algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm), solving it costs an order of $O(n^3)$
operations, and lacks flexibility, since one may want to couple families of different sizes.

Optimal transport extends all of this, through faster algorithms (in $n^2$ or even linear in $n$) along with numerous
generalizations that can help it handle weighted sets of different size, partial matchings, and even more evolved
so-called quadratic matching problems.

In the simple toy example below, we compute the optimal coupling matrix between two point clouds sampled randomly
(2D vectors, compared with the squared Euclidean distance):

## Example
```python
import jax
import jax.numpy as jnp

from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

# sample two point clouds and their weights.
rngs = jax.random.split(jax.random.PRNGKey(0), 4)
n, m, d = 12, 14, 2
x = jax.random.normal(rngs[0], (n,d)) + 1
y = jax.random.uniform(rngs[1], (m,d))
a = jax.random.uniform(rngs[2], (n,))
b = jax.random.uniform(rngs[3], (m,))
a, b = a / jnp.sum(a), b / jnp.sum(b)
# Computes the couplings using the Sinkhorn algorithm.
geom = pointcloud.PointCloud(x, y)
prob = linear_problem.LinearProblem(geom, a, b)

solver = sinkhorn.Sinkhorn()
out = solver(prob)
```

The call to `solver(prob)` above works out the optimal transport solution. The `out` object contains a transport matrix
(here of size $12\times 14$) that quantifies the association strength between each point of the first point cloud, to one or
more points from the second, as illustrated in the plot below. We provide more flexibility to define custom cost
functions, objectives, and solvers, as detailed in the [full documentation](https://ott-jax.readthedocs.io/en/latest/).

![obtained coupling](https://raw.githubusercontent.com/ott-jax/ott/main/docs/_static/images/couplings.png)

## Citation
If you have found this work useful, please consider citing this reference:

```
@article{cuturi2022optimal,
  title={Optimal Transport Tools (OTT): A JAX Toolbox for all things Wasserstein},
  author={Cuturi, Marco and Meng-Papaxanthos, Laetitia and Tian, Yingtao and Bunne, Charlotte and
          Davis, Geoff and Teboul, Olivier},
  journal={arXiv preprint arXiv:2201.12324},
  year={2022}
}
```
## See also
The [moscot](https://moscot.readthedocs.io/en/latest/index.html) package for OT analysis of multi-omics data also uses OTT as a backbone.
