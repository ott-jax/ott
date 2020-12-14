# Optimal Transport Tools (OTT)

**Disclaimer: this is not an official Google product.**

OTT is a JAX toolbox that bundles a few utilities to solve numerically
[optimal transport problems](https://arxiv.org/abs/1803.00567), along with
selected applications that rely on solving OT. This first version considers the
computation of OT divergences between distributions/point clouds/histograms,
computation of barycenters of the same, or more advanced estimation problems
that leverage the optimal transport geometry, such as soft-quantiles / soft-sort
operators.

In this first version, we have focused our efforts on providing a sturdy and
versatile implementation of the Sinkhorn algorithm. The Sinkhorn algorithm is
the computational workhorse of several approaches building on OT. The Sinkhorn
algorithm is a fixed-point algorithm; each iteration consists in kernel matrix /
vector multiplications, followed by elementwise divisions:

$$
u \leftarrow \frac{a}{Kv},\quad v \leftarrow \frac{a}{K^Tu},
$$

Here $$a$$ and $$b$$ are probability vectors, possibly of different sizes $$n$$
and $$m$$, while $$K$$ is a linear map from $$\mathbf{R}^m$$ to
$$\mathbf{R}^n$$. Although this iteration is very simple, we focus here on a few
important details, such as - parallelism of its application on several pairs of
measures that may share structure, - backward-mode evaluation for automatic
differentiation with respect to relevant parameters that define $$K$$, $$a$$ or
$$b$$, - speed-ups that can be obtained depending on the specifics of the kernel
$$K$$, - stability using log-space computations.

In our implementation, we encode such kernels $$K$$ in a `Geometry` object which
is typically defined using two measures.

## Overall description of source code

Currently implements the following classes and functions:

-   In the `core` folder,

    -   The `Geometry` class and its descendants describe a cost structure
        between the supports of a pair of input/output measures. That cost
        structure is accessed through various member functions, mostly used when
        running the Sinkhorn algorithm (typically kernel multiplications, or
        log-sum-exp row/column-wise application) or after (to apply the OT
        transport matrix to a vector).

        -   In its generic `geometry` implementation, the class is initialized
            with a `cost_matrix` or a `kernel_matrix`, as well as a `epsilon`
            regularization parameter.

        -   If one wishes to compute OT between two weighted point clouds
            $$x=(x_1, \dots, x_n)$$ and $$y=(y_1, \dots, y_m)$$ endowed with a
            given cost function (e.g. Euclidean) $$c$$, the `PointCloudGeometry`
            class can be used to define how the corresponding kernel
            $$K=\exp(-c(x_i,y_j)/\epsilon)$$.

        -   Simlarly, if all measures to be considered are supported on a
            separable grid (e.g. $$\{1,...,n\}^d$$), and the cost is separable
            along the various axis, i.e. the cost between two points on that
            grid is equal to the sum of (possibly $$d$$ different) cost
            functions evaluated on each of the $$d$$ pairs of coordinates, then
            the application of the kernel is much simplified, both in log space
            or on the histograms themselves.

    -   The `sinkhorn` function runs the Sinkhorn algorithm, with the aim of
        solving approximately one or various optimal transport problems in
        parallel. An OT problem is defined by a `Geometry` object, and a pair
        $$(a, b)$$ (or batch thereof) of histograms. The function's outputs are
        stored in a `SinkhornOutput` named t-uple, containing potentials,
        regularized OT cost and an error term quantifying convergence of the
        algorithm.

    -   In `sinkhorn_divergence`, implentation of the
        [Sinkhorn divergence](http://proceedings.mlr.press/v84/genevay18a.html),
        a variant of the Wasserstein distance that uses regularization and is
        computed by centering the output of `sinkhorn` when comparing two
        measures.

-   In the `tools` folder,

    -   In `discrete_barycenter`: implentation of discrete Wasserstein
        barycenters : given $$N$$ histograms all supported on the same
        `Geometry`, compute a barycenter of theses measures, using an algorithm
        by [Janati et al. (2020)](https://arxiv.org/abs/2006.02575)

    -   implementation of
        [soft-sorting](https://papers.nips.cc/paper/2019/hash/d8c24ca8f23c562a5600876ca2a550ce-Abstract.html)
        operators in `soft_sort`.
