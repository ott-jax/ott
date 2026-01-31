Glossary
========

.. glossary::
    :sorted:

    Benamou-Brenier
        A dynamic formulation of optimal transport :cite:`benamou:00` that
        characterizes the optimal transport problem between two probability
        distributions supported on a vector space as the minimum kinetic energy
        needed to move one distribution to the other, using a time-dependent
        velocity field. More precisely, given two probability distributions,
        the problem reads:

        .. math::

          \min_{(\rho_t, v_t)} \int_0^1 \int \|v_t(x)\|^2 d\rho_t(x) dt

        where :math:`\rho_t` is a time-dependent probability distribution
        such that :math:`\rho_0=\mu` and :math:`\rho_1=\nu`, and :math:`v_t`
        is a time-dependent velocity field such that the continuity equation
        :math:`\partial_t \rho_t + \nabla \cdot (\rho_t v_t)=0` is satisfied.

        The solution to that problem results in a velocity field :math:`v_t`
        that is constant along the integration path linking a source point to
        the point it is mapped to in the target distribution. Writing :math:`T`
        for the :term:`Monge map` between :math:`\mu` and :math:`\nu`, one has
        in particular that :math:`T(x) = x + \int_0^1 v_t(x) dt`, but also writing
        :math:`x_t = x + t(T(x)-x)` the barycenter between :math:`x` and :math:`T(x)`,
        one has

        .. math::

          \forall t\in[0,1], v_t(x_t) = T(x) -x

        In other words, the velocity field is constant along the straight line
        connecting a source point to its image in the target.

    Brenier theorem
        Fundamental result in optimal transport theory :cite:`brenier:91`
        stating that when the :term:`ground cost` is the squared Euclidean
        distance, and the source measure is absolutely continuous (e.g. has a
        density), then there exists a unique optimal :term:`transport map`
        between the source and target measures, which is the gradient of a
        convex function (a so-called :term:`Brenier potential`).
        Conversely, any :term:`transport map` :math:`T` that is
        the gradient of a convex function is optimal for the squared Euclidean
        cost when considering the OT problem between any source distribution
        :math:`\mu`. to that same source modified by the
        :term:`push-forward` map :math:`T`, namely :math:`T\#\mu`. The Brenier
        theorem is a special case of the more general
        :term:`Gangbo-McCann theorem` when instantiated with the squared
        Euclidean cost :math:`c(x,y)=\tfrac12\|x-y\|^2`.

    Brenier potential
        Convex potential function whose gradient transports optimally
        (w.r.t. the squared-Euclidean cost) a probability measure in
        :math:`\mathcal{P}(\mathbb{R}^d)` onto another (i.e. whose gradient is
        the :term:`Monge map` between them). More specifically, the
        Brenier potential :math:`\varphi` for the optimal transport problem from
        a measure :math:`\mu` to :math:`\nu` solves a variant of the
        :term:`dual Kantorovich problem` that reads

        .. math::

          \min_{\varphi: \mathbb{R}^d\rightarrow \mathbb{R}} \int \varphi d\mu + \int \varphi^* d\nu\,.

        where :math:`\varphi^*` is the :term:`Legendre transform` of
        :math:`\varphi`. Note that if one picks an arbitrary source measure
        :math:`\mu` and a convex potential :math:`\varphi`, and thereafter sets
        :math:`\nu:=\nabla\varphi \#\mu`, then :math:`\varphi` is necessarily a
        Brenier potential solving the OT problem from :math:`\mu` to
        :math:`\nu`.

    c-transform
        The c-transform of a function :math:`g` with respect to a
        :term:`ground cost` :math:`c(x,y)` is defined as the function :math:`g^c`
        such that for any :math:`x` in the source space, :math:`g^c(x) :=
        \inf_y c(x,y) - g(y)`. Alternatively, the c-transform of a function
        :math:`f` defined on the source space is the function :math:`f^c`
        defined on the target space such that for any :math:`y` in the target
        space, :math:`f^c(y) := \inf_x c(x,y) - f(x)` if :math:`c` is symmetric.
        One always has :math:`f^{ccc} = f^{c}`.

    c-concave function
        A function that can be written as the
        :term:`c-transform` of another function. Optimal
        :term:`dual Kantorovich potentials` can be found within such functions,
        under mild assumptions on the :term:`ground cost` and the measures
        being compared. For c-concave functions, the c-transform is an
        involution, i.e. :math:`f^{cc} = f`.

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
        :math:`f(x)+g(y)\leq c(x,y)`. The
        :term:`semidiscrete problem` studies more specifically
        the mixed setting in which either measure is discrete and the other
        continuous.

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

    entropic map
        Refers to an approximate :term:`transport map` obtained by solving first
        a :term:`entropy-regularized optimal transport` problem in dual form
        (using typically the :term:`Sinkhorn algorithm`) between two point
        clouds, to leverage next the pair of dual potential vectors :math:`f,g`
        returned by that algorithm to form a *continuous* approximation to the
        dual functions arising in the :term:`dual Kantorovich problem`. This
        approximation in then plugged into the :term:`Gangbo-McCann theorem` to
        recover an approximate :term:`Monge map`.

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
        :math:`\phi_a^{*}(z) = \rho_a e^{z/\varepsilon}`, its
        :term:`Legendre transform` :cite:`sejourne:19`.

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

    flow matching
        Family of methods :cite:`lipman:22` designed to fit a time-dependent
        velocity flow model so that it maps progressively a given source to a
        target distribution. The estimation of that velocity flow is done
        by specifying a pre-defined stochastic interpolant :cite:`albergo:23`
        between said distribution, and fitting the velocity field so that it
        agrees with that interpolation.

    Gangbo-McCann theorem
        Fundamental result in optimal transport theory :cite:`gangbo:96`
        stating that when both source and target measures are supported on the
        same Euclidean vector space, source measure is absolutely continuous
        (has a density) and cost is differentiable and satisfies the
        :term:`twist condition`, then there exists a unique optimal
        :term:`transport map` between the source and target measures,

        .. math::

          T(x) = \nabla_1 c(x, \cdot)^{-1}(\nabla f(x))

        where :math:`f` is a :term:`c-concave function` that solves the
        :term:`dual Kantorovich problem`. Conversely, any :term:`transport map`
        :math:`T` that can be written as above for some :term:`c-concave function`
        :math:`f` is optimal for the cost :math:`c` when considering the OT
        problem between any source distribution :math:`\mu` and the target
        distribution :math:`\nu = T\#\mu`.

    Gromov-Wasserstein distance
        A generalization of the :term:`Wasserstein distance` between two point
        clouds living in possibly different spaces and equipped each with
        different cost functions, obtained by solving the
        :term:`Gromov-Wasserstein problem`.

    Gromov-Wasserstein problem
        A generalization of the :term:`Kantorovich problem` in which the
        objective function is no longer a linear function of a coupling matrix
        :math:`P`, but more generally a quadratic function of :math:`P`.
        See :cite:`memoli:11`.

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

    input convex neural network
        A neural network architecture for vectors with a few distinguishing
        features: some parameters of this NN must be non-negative, the NN's
        output is real-valued and guaranteed to be convex in the input vector.
        Abbreviated as ICNN, see :cite:`amos:17` for exact definition.

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

    Legendre transform
        The Legendre transform of a convex function :math:`\phi` is the convex
        function :math:`\phi^{*}` defined as

        .. math::

          \phi^{*}(y) = \sup_x  \langle x, y \rangle - \phi(x)

        one has the identities :math:`\nabla\phi\circ\nabla\phi^{*} = Id` and
        :math:`\nabla\phi^{*}\circ\nabla\phi = Id` when :math:`\phi` is strictly
        convex and differentiable.


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

    Monge map
        A :term:`transport map` :math:`T` that is optimal for the :term:`Kantorovich problem`.
        in the sense that for two measures :math:`\mu` and :math:`\nu`, it solves

        .. math::

          \min_{T : T\#\mu=\nu} \int c(x,T(x)) d\mu(x).

        The constraint :math:`T\#\mu=\nu` means that the :term:`push-forward`
        measure obtained by pushing :math:`\mu` through :math:`T`
        is equal to :math:`\nu`. An optimal solution to the problem above solves
        the :term:`Kantorovich problem` between :math:`\mu` and :math:`\nu` in
        the sense that the coupling :math:`\pi` defined as the push-forward of
        :math:`\mu` through the map :math:`(Id,T)`, where :math:`Id` is the
        identity map, is an optimal coupling between :math:`\mu` and :math:`\nu`.

        When it exists, a Monge map is a deterministic mapping between the
        supports of two measures, as opposed to a :term:`transport plan` that
        can be stochastic. The Monge map can be recovered from the solution to
        the :term:`dual Kantorovich problem` when the :term:`ground cost` is
        differentiable and satisfies the :term:`twist condition`, and the source
        measure is absolutely continuous (e.g. has a density), using the
        :term:`Gangbo-McCann theorem`.

    multimarginal coupling
        A multimarginal coupling of :math:`N` probability measures
        :math:`\mu_1, \dots, \mu_N` is a probability measure on the product
        space of their respective supports, such that its marginals coincide,
        in that order, with :math:`\mu_1, \dots, \mu_N`.

    push-forward
        Given a measurable mapping :math:`T` (e.g. a vector to vector map),
        the push-forward measure of :math:`\mu` by :math:`T` denoted as
        :math:`T\#\mu`, is the measure defined to be such that for any
        measurable set :math:`B`, :math:`T\#\mu(B)=\mu(T^{-1}(B))`. Intuitively,
        it is the measure obtained by applying the map :math:`T` to all points
        described in the support of :math:`\mu`. See also the
        `Wikipedia definition <https://en.wikipedia.org/wiki/push-forward_measure>`__.
        Note that the OTT-JAX logo is a stylized depiction of the push-forward
        operator :math:`\#`.

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

    semidiscrete problem
        Refers to the optimal transport problem where one of the two measures
        is discrete (a weighted sum of Dirac masses) and the other is absolutely
        continuous, which, in the context of this toolbox, means that one can
        get i.i.d. samples from it. When both conditions are met, the
        :term:`dual Kantorovich problem` can be cast as a finite-dimensional
        concave maximization problem :cite:`cuturi:18`. Numerical integration
        methods can be used to approximate the objective and its gradients in
        lower dimension :cite:`merigot:11`, whereas simpler SGD methods
        can be leveraged in higher dimensions :cite:`genevay:16`.

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
        satisfy a :term:`push-forward` constraint :math:`T\#\mu = \nu`.

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

    Wasserstein barycenter
        The notion of a mean of vectors generalized to the Wasserstein space
        of probability distributions. A Wasserstein barycenter is a measure :math:`\mu`
        that summarizes a weighted family of measures :math:`(\nu_1,\dots,\nu_n)` in the sense
        that for a family of :math:`n` probability weights :math:`\lambda_1,\dots,\lambda_n`,

        .. math::

          \mu^\star:=\arg\min_{\mu\in\mathcal{P}(\Omega)} \sum_{i=1}^n \lambda_i W(\mu,\nu_i)

        See for instance :cite:`agueh:11`, :cite:`cuturi:14` and :cite:`benamou:15`.
