Glossary
========

.. glossary::
    :sorted:

    coupling
        A coupling of two probability measures :math:`\mu` and :math:`\nu` is a
        probability measure on the product space of their respective supports.
        When the coupling is balanced, the first and second marginals of that
        probability measure must coincide with :math:`\mu` and
        :math:`\nu` respectively. Equivalently, given two non-negative vectors
        :math:`a\in\mathbb{R}^n` and :math:`b\in\mathbb{R}^m`, a coupling in
        matrix form is a non-negative matrix :math:`P` of size
        :math:`n\times m`. When the coupling is balanced :math:`P` is in their
        :term:`transportation polytope` :math:`U(a,b)`.

    dual Kantorovich potentials
        Real-valued functions or vectors that solve the
        :term:`dual Kantorovich problem`.

    dual Kantorovich problem
        Dual formulation of the :term:`Kantorovich problem`, seeking two
        vectors :math:`f, g` that are :term:`dual Kantorovich potentials` such
        that, given a :term:`ground cost` matrix :math:`C` of size ``[n, m]``
        and two probability vectors :math:`a \in\mathbb{R}^n,b\in\mathbb{R}^m`,
        they belong to the :term:`dual transportation polyhedron` :math:`D(C)`
        and maximize:

        .. math::

          \max_{f,g \,\in D(C)} \langle f,a \rangle + \langle g,b \rangle.

        This problem admits a continuous formulation between two probability
        distributions :math:`\mu,\nu`:

        .. math::

          \max_{f\oplus g\leq c} \int f d\mu + \int g d\nu,

        where :math:`f,g` are in that case real-valued *functions* on the
        supports of :math:`\mu,\nu` and :math:`f\oplus g\leq c` means that for
        any pair :math:`x,y` in the respective supports, one has
        :math:`f(x)+g(y)\leq c(x,y)`.

    dual transportation polyhedron
        Given a :math:`n\times m` cost matrix :math:`C`, denotes the set of
        pairs of vectors

        .. math::

          D(C):= \{f \in\mathbb{R}^n, g \in \mathbb{R}^m
          | f_i + g_j \leq C_{ij}\}.

    dualizing
        Within the context of optimization, the process of simplifying a
        constrained optimization problem into an unconstrained one, by
        transforming constraints into penalty terms in the objective function.

    entropy-regularized optimal transport
        The data of the entropy regularized OT (EOT) problem is parameterized by
        a cost matrix :math:`C` of size ``[n, m]`` and two vectors :math:`a,b`
        of non-negative weights of respective size ``n`` and ``m``.
        The parameters of the EOT problem consist of three numbers
        :math:`\varepsilon, \tau_a, \tau_b>0`.

        The optimization variables are a pair of vectors of sizes ``n`` and
        ``m`` denoted as :math:`f` and :math:`g`, akin to
        :term:`dual Kantorovich potentials` but not constrained to belong to the
        :term:`dual transportation polyhedron`.

        Using the reparameterization for :math:`\rho_a` and
        :math:`\rho_b` using
        :math:`\tau_a=\rho_a /(\varepsilon + \rho_a)` and
        :math:`\tau_b=\rho_b /(\varepsilon + \rho_b)`, the EOT optimization
        problem reads:

        .. math::

          \max_{f, g} - \langle a, \phi_a^{*}(-f) \rangle -  \langle b,
          \phi_b^{*}(-g) \rangle - \varepsilon \left(\langle e^{f/\varepsilon},
          e^{-C/\varepsilon} e^{g/\varepsilon} \rangle -|a||b|\right)

        where :math:`\phi_a(z) = \rho_a z(\log z - 1)` is a scaled entropy, and
        :math:`\phi_a^{*}(z) = \rho_a e^{z/\varepsilon}`, its Legendre transform
        :cite:`sejourne:19`.

        That problem can also be written, instead, using positive scaling
        vectors `u`, `v` of size ``n``, ``m``, and the kernel matrix
        :math:`K := e^{-C/\varepsilon}`, as

        .. math::

          \max_{u, v >0} - \langle a,\phi_a^{*}(-\varepsilon\log u) \rangle
          + \langle b, \phi_b^{*}(-\varepsilon\log v) \rangle -
          \langle u, K v \rangle

        Both of these problems can be written with a *primal* formulation, that
        solves the :term:`unbalanced` optimal transport problem with a variable
        matrix :math:`P` of size ``n`` x ``m`` and positive entries:

        .. math::

          \min_{P>0} \langle P,C \rangle +\varepsilon \text{KL}(P | ab^T)
          + \rho_a \text{KL}(P\mathbf{1}_m | a)
          + \rho_b \text{KL}(P^T \mathbf{1}_n | b)

        where :math:`\text{KL}` is the generalized Kullback-Leibler divergence.

        The very same primal problem can also be written using a kernel
        :math:`K` instead of a cost :math:`C` as well:

        .. math::

          \min_{P>0}\, \varepsilon \text{KL}(P|K)
          + \rho_a \text{KL}(P\mathbf{1}_m | a)
          + \rho_b \text{KL}(P^T \mathbf{1}_n | b)

        The *original* OT problem taught in linear programming courses is
        recovered by using the formulation above relying on the cost :math:`C`,
        and letting :math:`\varepsilon \rightarrow 0`, and
        :math:`\rho_a, \rho_b \rightarrow \infty`.
        In that case the entropy disappears, whereas the :math:`\text{KL}`
        regularization above become constraints on the marginals of :math:`P`:
        This results in a standard min cost flow problem also called the
        :term:`Kantorovich problem`.

        The *balanced* regularized OT problem is recovered for finite
        :math:`\varepsilon > 0` but letting :math:`\rho_a, \rho_b \rightarrow
        \infty`. This problem can be shown to be equivalent to a matrix scaling
        problem, which can be solved using the :term:`Sinkhorn algorithm`.
        To handle the case :math:`\rho_a, \rho_b \rightarrow \infty`, the
        Sinkhorn function uses parameters ``tau_a`` and ``tau_b`` equal
        respectively to :math:`\rho_a /(\varepsilon + \rho_a)` and
        :math:`\rho_b / (\varepsilon + \rho_b)` instead. Setting either of these
        parameters to 1 corresponds to setting the corresponding
        :math:`\rho_a, \rho_b` to :math:`\infty`.

    envelope theorem
        The envelope theorem or Danskin's theorem is a major result about the
        differentiability properties of the value function of a parameterized
        optimization problem. Namely, that for a function :math:`f` defined
        implicitly as an optimal objective parameterized by a vector :math:`x`,

        .. math::
          h(x):=\min_z s(x,z), z^\star(x):=\arg\min_z s(x,z)

        one has

        .. math::
          \nabla h(x)=\nabla_1 s(x,z^\star(x))

        stating in effect that the optimal :math:`z^\star(x)` does not
        need to be differentiated w.r.t. :math:`x` when computing the
        gradient of :math:`h`. Note that this result is not valid for higher
        order differentiation.

    ground cost
        A real-valued function of two variables, :math:`c(x,y)` that describes
        the cost needed to displace a point :math:`x` in a source measure to
        :math:`y` in a target measure. Can also refer to a matrix :math:`C` of
        evaluations of :math:`c` on various pairs of points,
        :math:`C=[c(x_i, y_j)]_{ij}`.

    Hungarian algorithm
        Combinatorial algorithm proposed by Harold Kuhn to solve the
        :term:`optimal matching problem`. See the
        `Wikipedia definition <https://en.wikipedia.org/wiki/Hungarian_algorithm>`__
        .

    implicit differentiation
        Formula used to compute the vector-Jacobian
        product of the minimizer of an optimization procedure that leverages
        the fact that small variations in the input of the optimization problem
        still result in minimizers that verify optimality conditions
        (KKT or first-order conditions). These identities can then help recover
        the vector-Jacobian operator by inverting a linear system.

    input-convex neural networks
        A neural network architecture for vectors with a few distinguishing
        features: some parameters of this NN must be non-negative, the NN's
        output is real-valued and guaranteed to be convex in the input vector.

    Kantorovich problem
        Linear program that is the original formulation of optimal transport
        between two point-clouds, seeking an optimal :term:`coupling` matrix
        :math:`P`. The problem is parameterized by a :term:`ground cost` matrix
        :math:`C` of size ``[n, m]`` and two probability vectors :math:`a,b` of
        non-negative weights of respective sizes ``n`` and ``m``, summing to
        :math:`1`. The :term:`coupling` is in the
        :term:`transportation polytope` :math:`U(a,b)` and must minimize the
        objective

        .. math::

          \min_{P \in U(a,b)} \langle P,C \rangle = \sum_{ij} P_{ij} C_{ij}.

        This linear program can be seen as the primal problem of the
        :term:`dual Kantorovich problem`. Alternatively, this problem admits a
        continuous formulation between two probability distributions
        :math:`\mu,\nu`:

        .. math::

          \min_{\pi \in \Pi(\mu,\nu)} \iint cd\pi.

        where :math:`\pi` is a :term:`coupling` density with first marginal
        :math:`\mu` and second marginal :math:`\nu`.

    low-rank optimal transport
        Variant of the :term:`Kantorovich problem` whereby the search for an
        optimal :term:`coupling` matrix :math:`P` is restricted to lie in a
        subset of matrices of low-rank. Effectively, this is parameterized by
        replacing :math:`P` by a low-rank factorization

        .. math::
          P = Q \text{diag}(g) R^T,

        where :math:`Q,R` are :term:`coupling` matrices of size ``[n,r]`` and
        ``[m,r]`` and :math:`g` is a vector of size ``[r,]``. To be effective,
        one assumes implicitly that rank :math:`r\ll n,m`. To solve this in
        practice, the  :term:`Kantorovich problem` is modified to only seek
        solutions with this factorization, and updates on :math:`Q,R,g` are done
        alternatively. These updates are themselves carried out by solving an
        :term:`entropy-regularized optimal transport` problem.

    matching
        A bijective pairing between two families of points of the same size
        :math:`N`, parameterized using a permutation of size :math:`N`.

    multimarginal coupling
        A multimarginal coupling of :math:`N` probability measures
        :math:`\mu_1, \dots, \mu_N` is a probability measure on the product
        space of their respective supports, such that its marginals coincide,
        in that order, with :math:`\mu_1, \dots, \mu_N`.

    push-forward map
        Given a measurable mapping :math:`T` (e.g. a vector to vector map),
        the push-forward measure of :math:`\mu` by :math:`T` denoted as
        :math:`T\#\mu`, is the measure defined to be such that for any
        measurable set :math:`B`, :math:`T\#\mu(B)=\mu(T^{-1}(B))`. Intuitively,
        it is the measure obtained by applying the map :math:`T` to all points
        described in the support of :math:`\mu`. See also the
        `Wikipedia definition <https://en.wikipedia.org/wiki/push-forward_measure>`__.

    optimal transport
        Theory that characterizes efficient transformations between probability
        measures. Theoretical aspects usually arise when studying such
        transformations between continuous probability measures (e.g. densities)
        whereas computational aspects become relevant when estimating such
        transforms from samples.

    optimal matching problem
        Instance of the :term:`Kantorovich problem` where both marginal weight
        vectors :math:`a,b` are equal, and set both to a uniform weight vector
        of the form :math:`(\tfrac{1}{n},\dots,\tfrac{1}{n})\in\mathbb{R}^n`.

    Sinkhorn algorithm
        Fixed point iteration that solves the
        :term:`entropy-regularized optimal transport` problem (EOT).
        The Sinkhorn algorithm solves the EOT problem by seeking optimal
        :math:`f`, :math:`g` :term:`dual Kantorovich potentials` (or
        alternatively their parameterization as positive scaling vectors
        :math:`u`, :math:`v`), rather than seeking
        a :term:`coupling` :math:`P`. This is mostly for efficiency
        (potentials and scalings have a ``n + m`` memory footprint, rather than
        ``n m`` required to store :math:`P`). Note that an optimal coupling
        :math:`P^{\star}` can be recovered from optimal potentials
        :math:`f^{\star}`, :math:`g^{\star}` or scaling :math:`u^{\star}`,
        :math:`v^{\star}`.

        .. math::

          P^{\star} = \exp\left(\frac{f^{\star}\mathbf{1}_m^T +
          \mathbf{1}_n g^{*T}-C}{\varepsilon}\right) \text{ or } P^{\star}
          = \text{diag}(u^{\star}) K \text{diag}(v^{\star})

        The Sinkhorn algorithm solves this dual problem using block
        coordinate ascent, i.e. devising an update for each :math:`f` and
        :math:`g` (resp. :math:`u` and :math:`v`) that cancels alternatively
        their respective gradients, one at a time.

    Sinkhorn divergence
        Proxy for the :term:`Wasserstein distance` between two samples. Rather
        than use the output of the :term:`Kantorovich problem` to compare two
        families of samples, whose numerical resolution requires running a
        linear program, use instead the objective of
        :term:`entropy-regularized optimal transport` or that of
        :term:`low-rank optimal transport` properly renormalized. This
        normalization is done by considering:

        .. math::

          \text{SD}(\mu, \nu):= \Delta(\mu, \nu)
          - \tfrac12 \left(\Delta(\mu, \mu) + \Delta(\nu, \nu)\right)

        where :math:`Delta` is either the output of either
        :term:`entropy-regularized optimal transport` or
        :term:`low-rank optimal transport`

    transport map
        A function :math:`T` that associates to each point :math:`x` in the
        support of a source distribution :math:`\mu` another point :math:`T(x)`
        in the support of a target distribution :math:`\nu`, which must
        satisfy a :term:`push-forward map` constraint :math:`T\#\mu = \nu`.

    transport plan
        A :term:`coupling` (either in matrix or joint density form),
        quantifying the strength of association between any point :math:`x`` in
        the source distribution :math:`\mu` and target point :math:`y`` in the
        :math:`\nu` distribution.

    transportation polytope
        Given two probability vectors :math:`a,b` of non-negative weights of
        respective size ``n`` and ``m``, summing each to :math:`1`, the
        transportation polytope is the set of matrices

        .. math::

          U(a,b):= \{P \in \mathbb{R}^{n\times m} | ,
          P\mathbf{1}_m = a, P^T\mathbf{1}_n=b \}.

    twist condition
        Given a :term:`ground cost` function :math:`c(x, y)` taking two input
        vectors, the twist condition refers to the requirement that at any given
        point :math:`x`, the map :math:`y \mapsto \nabla_1 c(x, y)` be
        invertible. Although not necessary, this condition is sufficient to
        prove the existence of an optimal :term:`transport map` from a source
        to a target measure with suitable assumptions on the measures
        themselves.

    unbalanced
        A generalization of the :term:`Kantorovich problem` defined to bring
        more flexibility to optimal transport computations. Such a
        generalization arises when :term:`dualizing` the constraint that the
        variable :term:`coupling` in the :term:`Kantorovich problem` has
        marginals that coincide exactly with those of :math:`a` and :math:`b`
        or :math:`\mu` and :math:`\nu` in the continuous formulation. Instead,
        deviations from those marginals appear as penalty terms.

    unrolling
        Automatic differentiation technique to compute the vector-Jacobian
        product of the minimizer of an optimization procedure by treating the
        iterations (used to converge from an initial point) as layers in a
        computational graph, and computing its differential using reverse-order
        differentiation.

    Wasserstein distance
        Distance between two probability functions parameterized by a
        :term:`ground cost` function that is equal to the optimal objective
        reached when solving the :term:`Kantorovich problem`. The Wasserstein
        distance is truly a distance (in the sense that it satisfies all 3
        `metric axioms <https://en.wikipedia.org/wiki/Metric_space#Definition>`__
        ) if the  :term:`ground cost` is itself a distance to a power
        :math:`p\leq 1`, and the :math:`p` root of the objective of the
        :term:`Kantorovich problem` is used.

    semidiscrete optimal transport problem
        TODO(mcuturi)

    c-transform
        TODO(mcuturi)
