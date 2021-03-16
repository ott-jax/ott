<div align="center">
<img src="https://github.com/google-research/ott/raw/master/docs/logoOTT.png" alt="logo"  width="150"></img>
</div>

# Optimal Transport Tools (OTT), A toolbox for everything Wasserstein.

**See [full documentation](https://ott-jax.readthedocs.io/en/latest/) for detailed info.**

OTT is a JAX toolbox that bundles a few utilities to solve [optimal transport problems](https://arxiv.org/abs/1803.00567). These tools can help you compare
and match two weighted point clouds (or histograms, measures, etc.), given a cost (e.g. a distance) between single points.

Most of OTT is, for now, supported by a sturdy, versatile and efficient implementation of the Sinkhorn algorithm that takes advantage of JAX features, such as [JIT](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html#Using-jit-to-speed-up-functions), [auto-vectorization](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html#Auto-vectorization-with-vmap) and [implicit differentiation](https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html).

A typical OT problem has two ingredients: a pair of weight vectors `a` and `b` (one for each measure), with a ground cost matrix that is either directly given, or derived as the pairwise evaluation of a cost function on pairs of points taken from two measures. The main design choice in OTT comes from encapsulating the cost in a `Geometry` object, and bundle it with a few useful operations (notably kernel applications). The most common geometry is that of two clouds of vectors compared with the squared Euclidean distance, as illustrated in the example below:

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
ot = transport.Transport(x, y, a=a, b=b)
P = ot.matrix
```

The call to `sinkhorn` above works out the optimal transport solution by storing its output. The transport matrix can be instantiated using those optimal solutions and the `Geometry` again. That transoprt matrix links each point from the first point cloud to one or more points from the second, as illustrated below.

![obtained coupling](./images/couplings.png)

To be more precise, the `sinkhorn` algorithm operates on the `Geometry`,
taking into account weights `a` and `b`, to solve the OT problem, produce a named tuple that contains two optimal dual potentials `f` and `g` (vectors of the same size as `a` and `b`), the objective `reg_ot_cost` and a log of the `errors` of the algorithm as it converges, and a `converged` flag.

## Overall description of source code

Currently implements the following classes and functions:

-   In the [geometry](ott/geometry) folder,

    -   The `CostFn` class in [costs.py](ott/geometry/costs.py) and its descendants define cost functions between points. Two simple costs are currently provided, `Euclidean` between vectors, and `Bures`, between a pair of mean vector and covariance (p.d.) matrix.

    -   The `Geometry` class in [geometry.py](ott/geometry/geometry.py) and its descendants describe a cost structure between two measures. That cost structure is accessed through various member functions, either used when running the Sinkhorn algorithm (typically kernel multiplications, or log-sum-exp row/column-wise application) or after (to apply the OT matrix to a vector).

        -   In its generic `Geometry` implementation, as in [geometry.py](ott/geometry/geometry.py), an object can be initialized with either a `cost_matrix` along with an `epsilon` regularization parameter (or scheduler), or with a `kernel_matrix`.

        -   If one wishes to compute OT between two weighted point clouds
            <img src="https://render.githubusercontent.com/render/math?math=%24x%3D(x_1%2C%20%5Cdots%2C%20x_n)%24"> and <img src="https://render.githubusercontent.com/render/math?math=%24y%3D(y_1%2C%20%5Cdots%2C%20y_m)%24"> endowed with a
            given cost function (e.g. Euclidean) <img src="https://render.githubusercontent.com/render/math?math=%24c%24">, the `PointCloud`
            class in [pointcloud.py](ott/geometry/grid.py) can be used to define the corresponding kernel
            <img src="https://render.githubusercontent.com/render/math?math=%24K_%7Bij%7D%3D%5Cexp(-c(x_i%2Cy_j)%2F%5Cepsilon)%24">. When the number of these points grows very large, this geometry can be instantiated with an `online=True` parameter, to avoid storing the kernel matrix and choose instead to recompute the matrix on the fly at each application.

        -   Simlarly, if all measures to be considered are supported on a
            separable grid (e.g. <img src="https://render.githubusercontent.com/render/math?math=%24%5C%7B1%2C...%2Cn%5C%7D%5Ed%24">), and the cost is separable
            along all axis, i.e. the cost between two points on that
            grid is equal to the sum of (possibly <img src="https://render.githubusercontent.com/render/math?math=%24d%24"> different) cost
            functions evaluated on each of the <img src="https://render.githubusercontent.com/render/math?math=%24d%24"> pairs of coordinates, then
            the application of the kernel is much simplified, both in log space
            or on the histograms themselves. This particular case is exploited in the `Grid` geometry in [grid.py](ott/geometry/grid.py) which can be instantiated as a hypercube using a `grid_size` parameter, or directly through grid locations in `x`.

-   In the [core](ott/core) folder,
    -   The `sinkhorn` function in [sinkhorn.py](ott/core/sinkhorn.py) runs the Sinkhorn algorithm, with the aim of solving approximately one or various optimal transport problems in parallel. An OT problem is defined by a `Geometry` object, and a pair <img src="https://render.githubusercontent.com/render/math?math=%24(a%2C%20b)%24"> (or batch thereof) of histograms. The function's outputs are stored in a `SinkhornOutput` named t-uple, containing potentials, regularized OT cost, sequence of errors and a convergence flag. Such outputs (with the exception of errors and convergence flag) can be differentiated w.r.t. any of the three inputs `(Geometry, a, b)` either through backprop or implicit differentiation of the optimality conditions of the optimal potentials `f` and `g`.

    -   In [discrete_barycenter.py](ott/tools/discrete_barycenter.py): implementation of discrete Wasserstein barycenters : given <img src="https://render.githubusercontent.com/render/math?math=%24N%24"> histograms all supported on the same `Geometry`, compute a barycenter of theses measures, using an algorithm by [Janati et al. (2020)](https://arxiv.org/abs/2006.02575)

-   In the [tools](ott/tools) folder,

    -   In [soft_sort.py](ott/tools/soft_sort.py): implementation of
        [soft-sorting](https://papers.nips.cc/paper/2019/hash/d8c24ca8f23c562a5600876ca2a550ce-Abstract.html)
        operators .

    -   The `sinkhorn_divergence` function in [sinkhorn_divergence.py](ott/tools/sinkhorn_divergence.py), implements the
        [Sinkhorn divergence](http://proceedings.mlr.press/v84/genevay18a.html),
        a variant of the Wasserstein distance that uses regularization and is
        computed by centering the output of `sinkhorn` when comparing two
        measures.

    -   The `Transport` class in [sinkhorn_divergence.py](ott/tools/transport.py), provides a simple wrapper to the `sinkhorn` function defined above when the user is primarily interested in computing and storing an OT matrix.

_Disclaimer: this is not an official Google product._
