# Optimal Transport Tools (OTT)

**Disclaimer: this is not an official Google product.**

OTT is a JAX toolbox that bundles a few utilities to solve numerically
[optimal transport problems](https://arxiv.org/abs/1803.00567), along with
selected applications that rely on solving OT. This first version considers the
computation of OT divergences between distributions/point clouds/histograms,
computation of barycenters of the same, or more advanced estimation problems
that leverage the optimal transport geometry, such as soft-quantiles / soft-sort operators.

In this first version, we have focused our efforts on providing a sturdy and
versatile implementation of the Sinkhorn algorithm. The Sinkhorn algorithm is
the computational workhorse of several approaches building on OT. The Sinkhorn
algorithm is a fixed-point algorithm; each iteration consists in kernel matrix / vector multiplications, followed by elementwise divisions:

<img src="https://render.githubusercontent.com/render/math?math=%24%5Cmathbf%7Bu%7D%20%5Cleftarrow%20%5Cfrac%7B%5Cmathbf%7Ba%7D%7D%7B%5Cmathbf%7BKv%7D%7D%2C%5Cquad%20%5Cmathbf%7Bv%7D%20%5Cleftarrow%20%5Cfrac%7B%5Cmathbf%7Ba%7D%7D%7B%5Cmathbf%7BK%7D%5ET%5Cmathbf%7Bu%7D%7D%24">

Here <img src="https://render.githubusercontent.com/render/math?math=%24%5Cmathbf%7Ba%7D%24"> and <img src="https://render.githubusercontent.com/render/math?math=%24%5Cmathbf%7Bb%7D%24"> are probability vectors, possibly of different sizes <img src="https://render.githubusercontent.com/render/math?math=%24n%24">
and <img src="https://render.githubusercontent.com/render/math?math=%24m%24">, while <img src="https://render.githubusercontent.com/render/math?math=%24%5Cmathbf%7BK%7D%24"> is a linear map from <img src="https://render.githubusercontent.com/render/math?math=%24%5Cmathbb%7BR%7D%5Em%24"> to
<img src="https://render.githubusercontent.com/render/math?math=%24%5Cmathbb%7BR%7D%5En%24">. Although this iteration is very simple, we focus here on a few important details, such as - parallelism of its application on several pairs of measures that may share structure, - backward-mode evaluation for automatic differentiation with respect to relevant parameters that define <img src="https://render.githubusercontent.com/render/math?math=%24%5Cmathbf%7BK%7D%24">, <img src="https://render.githubusercontent.com/render/math?math=%24%5Cmathbf%7Ba%7D%24"> or <img src="https://render.githubusercontent.com/render/math?math=%24%5Cmathbf%7Bb%7D%24">, - speed-ups that can be obtained depending on the specifics of the kernel <img src="https://render.githubusercontent.com/render/math?math=%24%5Cmathbf%7BK%7D%24">, - stability using log-space computations.

In our implementation, we encode such kernels <img src="https://render.githubusercontent.com/render/math?math=%24%5Cmathbf%7BK%7D%24"> in a `Geometry` object which is typically defined using two measures.

## Example

```
from ott import pointcloud
from ott import sinkhorn

# Samples two point clouds and their weights.
rngs = jax.random.split(jax.random.PRNGKey(0),4)
n, m, d = 12, 14, 2
x = jax.random.normal(rngs[0], (n,d)) + 1
y = jax.random.uniform(rngs[1], (m,d))
a = jax.random.uniform(rngs[2], (n,))
b = jax.random.uniform(rngs[3], (m,))
a, b  = a / np.sum(a), b / np.sum(b)

# Computes the couplings via Sinkhorn algorithm.
geom = pointcloud.PointCloudGeometry(x,y)
out = sinkhorn.sinkhorn(geom, a, b)
P = geom.transport_from_potentials(out.f, out.g)
```

One can then plot the transport and obtain something like:

![obtained coupling](./images/couplings.png)



## Overall description of source code

Currently implements the following classes and functions:

-   In the [core](ott/core) folder,

    -   The `Geometry` class in [geometry.py](ott/core/ground_geometry/geometry.py) and its descendants describe a cost structure
        between the supports of a pair of input/output measures. That cost
        structure is accessed through various member functions, mostly used when
        running the Sinkhorn algorithm (typically kernel multiplications, or
        log-sum-exp row/column-wise application) or after (to apply the OT
        transport matrix to a vector).

        -   In its generic `Geometry` implementation, the class is initialized
            with a `cost_matrix` or a `kernel_matrix`, as well as a `epsilon`
            regularization parameter.

        -   If one wishes to compute OT between two weighted point clouds
            <img src="https://render.githubusercontent.com/render/math?math=%24x%3D(x_1%2C%20%5Cdots%2C%20x_n)%24"> and <img src="https://render.githubusercontent.com/render/math?math=%24y%3D(y_1%2C%20%5Cdots%2C%20y_m)%24"> endowed with a
            given cost function (e.g. Euclidean) <img src="https://render.githubusercontent.com/render/math?math=%24c%24">, the `PointCloudGeometry`
            class can be used to define how the corresponding kernel
            <img src="https://render.githubusercontent.com/render/math?math=%24K_%7Bij%7D%3D%5Cexp(-c(x_i%2Cy_j)%2F%5Cepsilon)%24">.

        -   Simlarly, if all measures to be considered are supported on a
            separable grid (e.g. <img src="https://render.githubusercontent.com/render/math?math=%24%5C%7B1%2C...%2Cn%5C%7D%5Ed%24">), and the cost is separable
            along the various axis, i.e. the cost between two points on that
            grid is equal to the sum of (possibly <img src="https://render.githubusercontent.com/render/math?math=%24d%24"> different) cost
            functions evaluated on each of the <img src="https://render.githubusercontent.com/render/math?math=%24d%24"> pairs of coordinates, then
            the application of the kernel is much simplified, both in log space
            or on the histograms themselves.

    -   The `sinkhorn` function in [sinkhorn.py](ott/core/sinkhorn.py) runs the Sinkhorn algorithm, with the aim of
        solving approximately one or various optimal transport problems in
        parallel. An OT problem is defined by a `Geometry` object, and a pair
        <img src="https://render.githubusercontent.com/render/math?math=%24(a%2C%20b)%24"> (or batch thereof) of histograms. The function's outputs are
        stored in a `SinkhornOutput` named t-uple, containing potentials,
        regularized OT cost and an error term quantifying convergence of the
        algorithm.

    -   The `sinkhorn_divergence` function in [sinkhorn_divergence.py](ott/core/sinkhorn_divergence.py), implements the
        [Sinkhorn divergence](http://proceedings.mlr.press/v84/genevay18a.html),
        a variant of the Wasserstein distance that uses regularization and is
        computed by centering the output of `sinkhorn` when comparing two
        measures.

-   In the [tools](ott/tools) folder,

    -   In [discrete_barycenter.py](ott/tools/discrete_barycenter.py): implementation of discrete Wasserstein
        barycenters : given <img src="https://render.githubusercontent.com/render/math?math=%24N%24"> histograms all supported on the same
        `Geometry`, compute a barycenter of theses measures, using an algorithm
        by [Janati et al. (2020)](https://arxiv.org/abs/2006.02575)

    -   In [soft_sort.py](ott/tools/soft_sort.py): implementation of
        [soft-sorting](https://papers.nips.cc/paper/2019/hash/d8c24ca8f23c562a5600876ca2a550ce-Abstract.html)
        operators .
