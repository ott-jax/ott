<img src="https://raw.githubusercontent.com/ott-jax/ott/main/docs/_static/images/logoOTT.png" width="10%" alt="logo">

# Optimal Transport Tools (OTT)
[![Downloads](https://static.pepy.tech/personalized-badge/ott-jax?period=total&units=INTERNATIONAL_SYSTEM&left_color=GREY&right_color=BLUE&left_text=downloads)](https://pepy.tech/projects/ott-jax)
[![Tests](https://img.shields.io/github/actions/workflow/status/ott-jax/ott/tests.yml?branch=main)](https://github.com/ott-jax/ott/actions/workflows/tests.yml)
[![Docs](https://img.shields.io/readthedocs/ott-jax/latest)](https://ott-jax.readthedocs.io/en/latest/)
[![Coverage](https://img.shields.io/codecov/c/github/ott-jax/ott/main)](https://app.codecov.io/gh/ott-jax/ott)

**See the [full documentation](https://ott-jax.readthedocs.io/en/latest/).**

## What is OTT-JAX?
A ``JAX`` powered library to solve a wide variety of problems leveraging optimal transport theory, at scale and on accelerators.

In particular, ``OTT-JAX`` implements various discrete solvers to match two point clouds, notably the Sinkhorn algorithm implemented to work on various geometric domains and sped up using various tweaks (scheduling, momentum, acceleration, initializations) and extensions (low-rank).

These algorithms power the resolution of more advanced problems
(Gromov-Wasserstein, Wasserstein barycenter) to compare point clouds in versatile settings.

On top of these discrete solvers, we also propose implementations of neural network
approaches. Given an source/target pair of measure, they output a neural net network
that seeks to approximation their optimal transport map.

``OTT-JAX`` is led by a team of researchers at Apple, with past contributions from Google and Meta researchers, as well as academic partners, including TU München, Oxford, ENSAE/IP Paris, ENS Paris and the Hebrew University.

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
from ott.solvers import linear
from ott.tools import plot

# sample two point clouds and their weights.
rngs = jax.random.split(jax.random.key(42), 4)
n, m, d = 6, 11, 2
x = jax.random.uniform(rngs[0], (n,d))
y = jax.random.uniform(rngs[1], (m,d))
a = jax.random.uniform(rngs[2], (n,)) +.2
b = jax.random.uniform(rngs[3], (m,)) +.2
a, b = a / jnp.sum(a), b / jnp.sum(b)
# instantiate geometry object to compare point clouds.
geom = pointcloud.PointCloud(x, y)
# compute coupling using the Sinkhorn algorithm.
out = jax.jit(linear.solve)(geom,a,b)

# plot
plot.Plot()(out)
```

The call to `solve(prob)` above works out the optimal transport solution. The `out` object contains a transport matrix
(here of size $12\times 14$) that quantifies the association strength between each point of the first point cloud, to one or
more points from the second, as illustrated in the plot below. We provide more flexibility to define custom cost
functions, objectives, and solvers, as detailed in the [full documentation](https://ott-jax.readthedocs.io/en/latest/). The last command displays the transport matrix by using a `Plot` object.

![obtained coupling](https://raw.githubusercontent.com/ott-jax/ott/main/docs/_static/images/coupling.png)

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
The [moscot](https://moscot.readthedocs.io/en/latest/index.html) package for OT analysis of multi-omics data uses OTT as a backbone.
