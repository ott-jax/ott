<img src="https://raw.githubusercontent.com/ott-jax/ott/main/docs/_static/images/logoOTT.png" width="10%" alt="logo">

# Optimal Transport Tools (OTT).

[![Tests](https://img.shields.io/github/workflow/status/ott-jax/ott/tests/main)](https://github.com/ott-jax/ott/actions/workflows/tests.yml)
[![Docs](https://img.shields.io/readthedocs/ott-jax/latest)](https://ott-jax.readthedocs.io/en/latest/)
[![Coverage](https://img.shields.io/codecov/c/github/ott-jax/ott/main)](https://app.codecov.io/gh/ott-jax/ott)

**See [full documentation](https://ott-jax.readthedocs.io/en/latest/).**

## What is OTT-JAX?

A JAX powered library to compute optimal transport at scale and on accelerators, OTT-JAX includes the fastest implementation of the Sinkhorn algorithm you will find around. We have implemented all tweaks (scheduling, acceleration, initializations) and extensions (low-rank), that can be used directly, or within more advanced problems (Gromov-Wasserstein, barycenters). Some of JAX features, including [JIT](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html#Using-jit-to-speed-up-functions), [auto-vectorization](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html#Auto-vectorization-with-vmap) and [implicit differentiation](https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html) work towards the goal of having end-to-end differentiable outputs. OTT-JAX is developed by a team of researchers from Apple, Google, Meta and many academic contributors, including TU München, Oxford, ENSAE/IP Paris and the Hebrew University.

## What is optimal transport?

Optimal transport can be loosely described as the branch of mathematics and optimization that studies *matching problems*: given two sets of points, how to find (given some prior information, typically a cost function) a good way to associate bijectively to every point in the first set another in the second.

These problems are easy to describe yet hard to solve. Indeed, while matching optimally two sets of *n* points using a pairwise cost can be solved with the [Hungarian algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm), this requires an effort that scales as $n^3$ and lacks flexibility (one may encouter sets with different size). Optimal transport extends all of this, through faster algorithms (in $n^2$ or even linear in $n$) along with numerous extensions (wighted sets of different size, partial matchings, so-called quadratic matching problems).

To go back to the basics: the crucial ingredient in OT problems will always consists of two measures, encoded most often as point clouds, and a cost function comparing points. In the simple toy example below, these point clouds are composed of vectors, and that these vectors are compared with a squared Euclidean distance:

## Example

```py
import jax
import jax.numpy as jnp
from ott.tools import transport
# Samples two point clouds and their weights.
rngs = jax.random.split(jax.random.PRNGKey(0),4)
n, m, d = 12, 14, 2
x = jax.random.normal(rngs[0], (n,d)) + 1
y = jax.random.uniform(rngs[1], (m,d))
a = jax.random.uniform(rngs[2], (n,))
b = jax.random.uniform(rngs[3], (m,))
a, b = a / jnp.sum(a), b / jnp.sum(b)
# Computes the couplings using the Sinkhorn algorithm.
ot = transport.solve(x, y, a=a, b=b)
P = ot.matrix
```

The call to `solve` above works out an optimal transport solution. The `ot` object contains a transport matrix that links each point from the first point cloud to one or more points from the second, as illustrated below. In this toy example, most choices were arbitrary, and are reflected in the crude `solve` API. We provide far more flexibility to define custom solvers, as detailed in the [full documentation](https://ott-jax.readthedocs.io/en/latest/).

![obtained coupling](https://raw.githubusercontent.com/ott-jax/ott/main/images/couplings.png)
## Citation

If you have found this work useful, please consider citing this reference:

```
@article{cuturi2022optimal,
  title={Optimal Transport Tools (OTT): A JAX Toolbox for all things Wasserstein},
  author={Cuturi, Marco and Meng-Papaxanthos, Laetitia and Tian, Yingtao and Bunne, Charlotte and Davis, Geoff and Teboul, Olivier},
  journal={arXiv preprint arXiv:2201.12324},
  year={2022}
}
```
