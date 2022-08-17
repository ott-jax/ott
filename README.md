<img src="https://raw.githubusercontent.com/ott-jax/ott/main/docs/_static/images/logoOTT.png" width="10%" alt="logo">

# Optimal Transport Tools (OTT).

![Tests](https://img.shields.io/github/workflow/status/ott-jax/ott/tests/main)
![Coverage](https://img.shields.io/codecov/c/github/ott-jax/ott/main)

**See [full documentation](https://ott-jax.readthedocs.io/en/latest/).**

## What is OTT-JAX?
Optimal transport theory can be loosely described as the branch of mathematics and optimization that studies *matching problems*: given two sets of points, how to find (given some prior information, typically a cost function) a good way to associate bijectively every point in the first set with another in the second. A typical matching problem arises, for instance, when sorting numbers (when sorting, one associates to numbers *[3.1, -4.2, -18, 5.4]* the ranks *[3, 2, 1, 4]* that reorder them in increasing fashion) or when matching submitted papers with reviewers at ML conferences!

These problems are easy to describe yet hard to solve. Indeed, while matching optimally two sets of *n* points using a pairwise cost can be solved with the [Hungarian algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm), this requires an effort that scales as $n^3$. Additionally, one may run into various issues. For instance, the two sets might have different sizes and a relevant matching cost function might not be given (it remains an open problem to set a score to qualify how well a reviewer is likely to be to check a paper!). Fortunately, optimal transport theory has made decisive progresses since the Hungarian algorithm was proposed, and can count on many efficient algorithms and extensions that can handle such situations.

OTT-JAX is a toolbox providing sturdy, scalable and efficient solvers for those problems. OTT builds upon the [JAX](https://jax.readthedocs.io/en) framework. Some of JAX features include [JIT](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html#Using-jit-to-speed-up-functions), [auto-vectorization](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html#Auto-vectorization-with-vmap) and [implicit differentiation](https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html).

The typical ingredient in OT problems consists of two discrete measures (an efficient way to encode weighted sets of points), and a cost function comparing points. By default, OTT assumes that the measures are supported on vectors, and that these vectors are compared with a squared Euclidean distance, as given below:

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
# Computes the couplings via Sinkhorn algorithm.
ot = transport.solve(x, y, a=a, b=b)
P = ot.matrix
```

The call to `sinkhorn` above works out the optimal transport solution by storing its output. That transoprt matrix links each point from the first point cloud to one or more points from the second, as illustrated below.

![obtained coupling](https://raw.githubusercontent.com/ott-jax/ott/main/images/couplings.png)

## Overall description of source code

Currently implements the following classes and functions:

-   The [geometry](ott/geometry) folder describes tools that to encapsulate the essential ingredients of OT problems: measures and cost functions.

    -   The `CostFn` class in [costs.py](ott/geometry/costs.py) and its descendants define cost functions between points. A few simple costs are considered, `Euclidean` between vectors, and `Bures`, between a pair of mean vector and covariance (p.d.) matrix.

    -   The `Geometry` class in [geometry.py](ott/geometry/geometry.py) describes a cost structure between two measures. That cost structure is accessed through various member functions, either used when running the Sinkhorn algorithm (typically kernel multiplications, or log-sum-exp row/column-wise application) or after (to apply the OT matrix to a vector).

        -   In its generic `Geometry` implementation, as in [geometry.py](ott/geometry/geometry.py), an object can be initialized with either a `cost_matrix` along with an `epsilon` regularization parameter (or scheduler), or with a `kernel_matrix`.

        -   If one wishes to compute OT between two weighted point clouds $x=\left(x_1, \ldots, x_n\right)$ and  $y=\left(y_1, \ldots, y_m\right)$  endowed with a
            cost function (e.g. Euclidean) $c$, the `PointCloud`
            class in [pointcloud.py](ott/geometry/pointcloud.py) can be used to define the corresponding cost and kernel matrices
            $C_{i j}=c\left(x_{i}, y_{j}\right)$ and $K_{i j}=\exp\left(-c\left(x_{i}, y_{j}\right) / \epsilon\right)$. When $n$ and $m$ are very large, this geometry can be instantiated with a `batch_size` parameter, to avoid storing the cost and/or kernel matrices, to recompute instead these matrices on the fly as needed, `batch_size` lines at a time, at each application.

        -   Simlarly, if all measures to be considered are supported on a
            separable grid (e.g. $\\{1, \ldots, n\\}^{d}$), and the cost is separable
            along all axis, i.e. the cost between two points on that
            grid is equal to the sum of (possibly $d$ different) cost
            functions evaluated on each of the $d$ pairs of coordinates, then
            the application of the kernel is much simplified, both in log space
            or on the histograms themselves. This particular case is exploited in the `Grid` geometry in [grid.py](ott/geometry/grid.py) which can be instantiated as a hypercube using a `grid_size` parameter, or directly through grid locations in `x`.

        -  `LRCGeometry`, low-rank cost geometries, of which a `PointCloud` endowed with a squared-Euclidean distance is a particular example, can efficiently carry apply their cost to another matrix. This is leveraged in particular in the low-rank Sinkhorn (and Gromov-Wasserstein) solvers.


-   In the [core](ott/core) folder,
    -   The `sinkhorn` function in [sinkhorn.py](ott/core/sinkhorn.py) is a wrapper around the `Sinkhorn` solver class, running the Sinkhorn algorithm, with the aim of solving approximately one or various optimal transport problems in parallel. An OT problem is defined by a `Geometry` object, and a pair $\left(a, b\right)$ (or batch thereof) of histograms. The function's outputs are stored in a `SinkhornOutput` named t-uple, containing potentials, regularized OT cost, sequence of errors and a convergence flag. Such outputs (with the exception of errors and convergence flag) can be differentiated w.r.t. any of the three inputs `(Geometry, a, b)` either through backprop or implicit differentiation of the optimality conditions of the optimal potentials `f` and `g`.
    -   A later addition in [sinkhorn_lr.py](ott/core/sinkhorn.py) is focused on the `LRSinkhorn` solver class, which is able to solve OT problems at larger scales using an explicit factorization of couplings as being low-rank.

    -   In [discrete_barycenter.py](ott/core/discrete_barycenter.py): implementation of discrete Wasserstein barycenters : given $N$ histograms all supported on the same `Geometry`, compute a barycenter of theses measures, using an algorithm by [Janati et al. (2020)](https://arxiv.org/abs/2006.02575).

    -   In [continuous_barycenter.py](ott/core/continuous_barycenter.py): implementation of continuous Wasserstein barycenters : given <img src="https://render.githubusercontent.com/render/math?math=%24N%24"> probability measures described as points which can be compared with an arbitrary cost function, compute a barycenter of theses measures, supported at most $k$ points on using an algorithm by [Cuturi and Doucet (2014)](https://proceedings.mlr.press/v32/cuturi14.html).

    -   In [gromov_wasserstein.py](ott/tools/gromov_wasserstein.py): implementation of two Gromov-Wasserstein solvers (both entropy-regularized and low-rank) to compare two measured-metric spaces, here encoded as a pair of `Geometry` objects, `geom_xx`, `geom_xy` along with weights `a` and `b`. Additional options include using a fused term by specifying `geom_xy`.

-   In the [tools](ott/tools) folder,

    -   In [soft_sort.py](ott/tools/soft_sort.py): implementation of
        [soft-sorting](https://papers.nips.cc/paper/2019/hash/d8c24ca8f23c562a5600876ca2a550ce-Abstract.html) operators, notably [soft-quantile transforms](http://proceedings.mlr.press/v119/cuturi20a.html)

    -   The `sinkhorn_divergence` function in [sinkhorn_divergence.py](ott/tools/sinkhorn_divergence.py), implements the [unbalanced](https://arxiv.org/abs/1910.12958) formulation of the [Sinkhorn divergence](http://proceedings.mlr.press/v84/genevay18a.html), a variant of the Wasserstein distance that uses regularization and is computed by centering the output of `sinkhorn` when comparing two measures.

    -   The `Transport` class in [sinkhorn_divergence.py](ott/tools/transport.py), provides a simple wrapper to the `sinkhorn` function defined above when the user is primarily interested in computing and storing an OT matrix.

    -   The [gaussian_mixture](ott/tools/gaussian_mixture) folder provides novel tools to compare and estimate GMMs with an OT perspective.

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
