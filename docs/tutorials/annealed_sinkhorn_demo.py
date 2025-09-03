# %% [markdown]
# # Annealed Sinkhorn Demo
# 
# This notebook demonstrates the Annealed Sinkhorn algorithm for solving Optimal Transport (OT) problems. The Annealed Sinkhorn algorithm is an enhanced variant of the classic Sinkhorn algorithm that provides improved convergence and stability properties.
#
# ## Background
#
# - **Optimal Transport (OT)**: A mathematical framework for comparing probability distributions by finding the most efficient way to transform one distribution into another.
#
# - **Sinkhorn Algorithm**: A popular method for approximating OT that uses entropic regularization to make the problem computationally tractable. However, it can be sensitive to the choice of regularization parameter.
#
# - **Annealed Sinkhorn**: Improves upon the classic algorithm by gradually increasing the regularization strength (β) during optimization, similar to simulated annealing in optimization.
#
# ## Experiments
#
# This notebook includes three experiments that demonstrate key properties of the algorithm:
#
# 1. **Random Cost Matrix Case**: Explores convergence behavior on synthetic data with random costs, showing how annealing improves optimization.
#
# 2. **Geometric Cost Case**: Demonstrates performance on a structured geometric problem with meaningful distances, using circles and stripes patterns.
#
# 3. **Piecewise Constant Schedules**: Investigates different annealing schedules, comparing smooth vs piecewise constant β trajectories.
#
# The experiments help visualize how annealing and debiasing affect convergence rates and solution quality.

# %% [markdown]
# ## Setup and Common Functions
#
# We use JAX for efficient numerical computing and automatic differentiation. The experiments utilize the OTT-JAX library's geometry and solver components.

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from functools import partial

from ott.geometry import geometry
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

# Common plotting settings
plt.rcParams.update({'font.size': 12})

# %% [markdown]
# ## Core Functions
#
# The implementation consists of three main algorithmic components:
#
# 1. **Project to Marginals**: Ensures transport plans satisfy marginal constraints
# 2. **Exact OT**: Computes ground truth optimal transport cost for evaluation
# 3. **Sinkhorn Track**: Main implementation of the annealed algorithm
#
# ### Mathematical Background
#
# The Sinkhorn algorithm solves the entropy-regularized optimal transport problem:
#
# ```
# min_{π} <C,π> + 1/β H(π)
# s.t. π1 = p, π^T1 = q
# ```
#
# where:
# - C is the cost matrix 
# - β controls regularization strength
# - H(π) is the entropy of π
# - p,q are the source/target distributions
#
# The annealed variant increases β over time according to β(t) = β₀ * t^κ.
# Debiasing helps correct for the entropic bias using a carefully designed correction term.
#
# %%
def project_to_marginals(gamma, p, q):
    """Project transport plan to match marginals exactly.
    
    Args:
        gamma: Transport plan matrix of shape (m, n)
        p: Source distribution of shape (m,)
        q: Target distribution of shape (n,)
        
    Returns:
        Projected transport plan that exactly matches marginals p and q.
    """
    # Project to first marginal
    gamma_sum_1 = jnp.sum(gamma, axis=1)
    a = jnp.minimum(p / gamma_sum_1, 1.0)
    gamma_temp = a[:, None] * gamma
    
    # Project to second marginal
    gamma_sum_2 = jnp.sum(gamma_temp, axis=0)
    b = jnp.minimum(q / gamma_sum_2, 1.0)
    gamma_temp = gamma_temp * b[None, :]
    
    # Handle remaining mass
    delta_p = p - jnp.sum(gamma_temp, axis=1)
    delta_q = q - jnp.sum(gamma_temp, axis=0)
    gamma_remaining = jnp.outer(delta_p, delta_q) / jnp.sum(delta_p)
    
    return gamma_temp + gamma_remaining

def compute_exact_ot(p, q, c):
    """Compute exact OT cost using high regularization."""
    # Create geometry with low regularization for approximate exact solution
    geom = geometry.Geometry(cost_matrix=c, epsilon=1e-3)
    prob = linear_problem.LinearProblem(geom, a=p, b=q)
    solver = sinkhorn.Sinkhorn(threshold=1e-8)
    result = solver(prob)
    return result.matrix, jnp.sum(c * result.matrix)

def sinkhorn_track(p, q, c, niter=100, OT_cost=0.0, kappa=0.0, beta0=1.0, debiased=True, 
                   iter_stop=None, plateau_length=1):
    """Exact implementation of Julia's Sinkhorn_track."""
    if iter_stop is None:
        iter_stop = niter
    
    m, n = len(p), len(q)
    assert c.shape == (m, n)
    
    # Initialize potentials and parameters (exactly as in Julia)
    u = jnp.zeros(m)
    v = jnp.zeros(n)
    lp = jnp.log(p)
    lq = jnp.log(q)
    beta = beta0
    
    plans = []
    errors = []
    
    for t in range(niter):
        # Update u (left potential)
        if debiased and t <= iter_stop:
            # Julia: temp1 = (lq .+ v .- β*c)  .- ((t^κ-(t-1)^κ)/t^κ) .* (u)
            t1 = t + 1  # Convert to 1-based indexing
            scale = (jnp.power(t1, kappa) - jnp.power(t1-1, kappa)) / jnp.power(t1, kappa)
            # Exactly match Julia's broadcasting order
            temp1 = lq.reshape(1, -1) + v.reshape(1, -1) - beta * c  # Column operation
            temp1 = temp1 - scale * u.reshape(-1, 1)  # Row operation
        else:
            # Julia: temp1 = v .+ lq .- β*c
            temp1 = lq.reshape(1, -1) + v.reshape(1, -1) - beta * c
        
        # Stabilized u update (as in Julia)
        stab1 = jnp.max(temp1, axis=1, keepdims=True)
        u = -stab1.reshape(-1) - jnp.log(jnp.sum(jnp.exp(temp1 - stab1), axis=1))
        
        # Update beta (between u and v updates, as in Julia)
        if (t <= iter_stop) and (int(jnp.floor(jnp.sqrt(t+1))) % plateau_length == 0):
            beta = beta0 * jnp.power(t+2, kappa)  # t+2 for 1-based indexing
        
        # Update v (right potential)
        # Julia: temp2 = u .+ lp .- β*c
        temp2 = u.reshape(-1, 1) + lp.reshape(-1, 1) - beta * c  # Row operation first
        stab2 = jnp.max(temp2, axis=0, keepdims=True)
        v = -stab2.reshape(-1) - jnp.log(jnp.sum(jnp.exp(temp2 - stab2), axis=0))
        
        # Compute transport plan (exactly as in Julia)
        # Julia: plans[:,:,t] = exp.(u .+ lp .+ v .+ lq .- β*c)
        logpi = u.reshape(-1, 1) + lp.reshape(-1, 1)  # Row terms
        logpi = logpi + v.reshape(1, -1) + lq.reshape(1, -1)  # Column terms
        logpi = logpi - beta * c  # Cost matrix
        
        # Stabilize and normalize
        stab = jnp.max(logpi)
        logpi = logpi - stab
        pi = jnp.exp(logpi)
        pi = pi / jnp.sum(pi)
        
        # Project and compute error
        pi_proj = project_to_marginals(pi, p, q)
        error = jnp.abs(jnp.sum(c * pi_proj) - OT_cost)  # Take absolute value
        
        plans.append(pi)
        errors.append(error)
    
    return jnp.stack(plans), jnp.array(errors)

# %% [markdown]
# ## Data Generation Functions
#
# We use two types of test problems to evaluate the algorithm:
#
# ### Random Problems
# - Generate random source/target distributions and cost matrices
# - Useful for testing algorithm behavior in general scenarios
# - Distributions drawn from uniform distribution
# - Costs drawn from normal distribution and normalized
#
# ### Geometric Problems
# A carefully designed test case that creates:
# - Source: Two circular distributions of different sizes
#   * Large circle with radius ~0.3-0.5
#   * Small circle with radius ~0.1, offset from center
# - Target: Two stripe patterns
#   * Horizontal stripe near the bottom
#   * Vertical stripe through the middle
#
# This geometric case tests the algorithm's ability to handle:
# - Multiple connected components
# - Different scales and shapes
# - Real 2D transport distances
#
# %%
def create_random_problem(key, size=75):
    """Create random OT problem (exactly as in Julia)."""
    key1, key2, key3 = jax.random.split(key, 3)
    
    # Create random distributions
    p = jax.random.uniform(key1, (size,))
    q = jax.random.uniform(key2, (size,))
    p = p / jnp.sum(p)
    q = q / jnp.sum(q)
    
    # Create random cost matrix (exactly as in Julia)
    c = jax.random.normal(key3, (size, size))
    c = (c - jnp.min(c)) / (jnp.max(c) - jnp.min(c))
    
    return p, q, c

def create_geometric_case():
    """Create geometric test case matching the original paper."""
    rng = jax.random.key(1)  # Same seed as original
    keys = jax.random.split(rng, 8)
    
    # First measure: Two circles
    m = 150
    # First circle
    ang1 = jax.random.uniform(keys[0], (m,)) * jnp.pi
    rad1 = jax.random.uniform(keys[1], (m,)) * 0.2 + 0.3
    X1 = jnp.stack([rad1 * jnp.cos(ang1), rad1 * jnp.sin(ang1)], axis=1)
    
    # Second circle
    ang2 = jax.random.uniform(keys[2], (m,)) * 2*jnp.pi
    rad2 = jax.random.uniform(keys[3], (m,)) * 0.1
    X2 = jnp.stack([
        rad2 * jnp.cos(ang2) - 0.3,
        rad2 * jnp.sin(ang2) + 0.4
    ], axis=1)
    
    X = jnp.concatenate([X1, X2], axis=0)
    
    # Second measure: Vertical stripes
    n = 150
    # First stripe
    Y1 = jnp.stack([
        jax.random.uniform(keys[4], (n,)) - 0.5,
        jax.random.uniform(keys[5], (n,)) * 0.15
    ], axis=1)
    
    # Second stripe
    Y2 = jnp.stack([
        (jax.random.uniform(keys[6], (n,)) - 0.5) * 0.15,
        jax.random.uniform(keys[7], (n,)) * 0.6
    ], axis=1)
    
    Y = jnp.concatenate([Y1, Y2], axis=0)
    
    # Create uniform weights
    m, n = X.shape[0], Y.shape[0]
    p = jnp.ones(m) / m
    q = jnp.ones(n) / n
    
    # Compute cost matrix
    X_exp = X[:, None, :]  # Shape (m, 1, 2)
    Y_exp = Y[None, :, :]  # Shape (1, n, 2)
    c = jnp.sum((X_exp - Y_exp) ** 2, axis=2)
    
    # Normalize cost
    c = c / (jnp.max(c) - jnp.min(c))
    
    return X, Y, p, q, c

# %% [markdown]
# ## Plotting Functions
#
# ### Visualization Types
#
# 1. **Pareto Front Plots**
#    - Show convergence behavior over iterations 
#    - X-axis: Iteration number
#    - Y-axis: Suboptimality (error vs exact OT)
#    - Multiple curves show different β₀ values for standard Sinkhorn
#    - Highlight annealed and debiased variants
#    - Log-log scale emphasizes convergence rates
#
# 2. **Geometric Data Plots**
#    - Visualize 2D source and target distributions
#    - Source points shown as blue stars
#    - Target points shown as orange stars
#    - Demonstrates problem structure (circles and stripes)
#    - Helps interpret transport distances
#
# The plots demonstrate both:
# - Algorithm performance (convergence rates and final accuracy)
# - Problem structure (distribution shapes and transport challenges)
#
# %%
def plot_pareto_front(std_results, beta0s, err1, err2, filename):
    """Plot Pareto front results."""
    plt.figure(figsize=(7, 5))
    
    # Plot standard Sinkhorn results
    for i, errors in enumerate(std_results):
        color = [i/len(beta0s), 0.5, 0.8]
        if i == 0:  # First β₀
            plt.loglog(errors + 1e-16, '--', color=color, alpha=0.8,
                      label=fr"standard Sinkhorn, $\beta={beta0s[0]:.0f}$")
        elif i == len(beta0s)-1:  # Last β₀
            plt.loglog(errors + 1e-16, '--', color=color, alpha=0.8,
                      label=fr"standard Sinkhorn, $\beta={beta0s[-1]:.0f}$")
        else:
            plt.loglog(errors + 1e-16, '--', color=color, alpha=0.5)
    
    # Plot annealed results
    plt.loglog(err1 + 1e-16, 'b-', label=r'Annealed Sinkhorn, $\kappa=1/2$', linewidth=4)
    plt.loglog(err2 + 1e-16, 'r-', label=r'Debiased Annealed Sinkhorn, $\kappa=2/3$', linewidth=4)
    
    plt.legend(fontsize=10, ncol=1)
    plt.xlabel(r'Iteration $t$')
    plt.ylabel('OT suboptimality after projection')
    plt.grid(True, which='both', alpha=0.2)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=200)
    plt.close()

def plot_geometric_data(X, Y):
    """Plot source and target distributions."""
    plt.figure(figsize=(7, 5))
    plt.plot(X[:, 0], X[:, 1], '*', markersize=2, label='Source')
    plt.plot(Y[:, 0], Y[:, 1], '*', markersize=2, label='Target')
    plt.legend(fontsize=10)
    plt.axis('equal')
    plt.axis([-0.55, 0.55, 0, 0.65])
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    # plt.savefig('output/geometric-data.png', bbox_inches='tight', dpi=200)
    plt.show()
    plt.close()

# %% [markdown]
# ## Experiment Functions
# 
# The three experiments test different aspects of the algorithm:
#
# ### 1. Pareto Front Experiment
# - Tests convergence behavior with random cost matrices
# - Compares standard Sinkhorn for multiple β₀ values
# - Shows how annealing and debiasing improve convergence
# - Demonstrates tradeoff between speed and accuracy
#
# ### 2. Geometric Experiment 
# - Tests algorithm on structured 2D transport problem
# - Uses circles and stripes to create challenging transport paths
# - Shows how annealing handles varying transport distances
# - Visualizes source/target distributions directly
#
# ### 3. Piecewise Constant Schedules
# - Investigates effect of different annealing schedules
# - Tests smooth vs piecewise constant β trajectories
# - Compares three different κ values (1/2, 2/3, 3/4)
# - Shows impact of plateau length on convergence
#
# Each experiment demonstrates different strengths of the annealed approach
# and produces plots that help visualize algorithm behavior.
#
# %%
def run_pareto_front_experiment(key=jax.random.PRNGKey(0)):
    """Run Pareto front experiment from paper."""
    print("Running Pareto front experiment...")
    
    # Create random problem
    p, q, c = create_random_problem(key)
    
    # Compute exact OT solution
    _, opt_cost = compute_exact_ot(p, q, c)
    print(f"Exact OT cost: {opt_cost:.4f}")
    
    # Parameters exactly as in Julia
    niter = 1000
    beta0 = 10.0
    
    # Standard Sinkhorn with different β₀ values
    beta0s = jnp.logspace(1, 3, num=21)  # 10 to 1000
    print(f"Running standard Sinkhorn with {len(beta0s)} different β₀...")
    
    std_results = []
    for beta0_i in beta0s:
        # Run without annealing (κ=0) and no debiasing
        _, errors = sinkhorn_track(p, q, c, niter=niter, OT_cost=opt_cost, 
                                 kappa=0.0, beta0=beta0_i, debiased=False)
        std_results.append(errors)
    
    # Run annealed versions (exactly as in Julia)
    print("Running annealed versions...")
    # Standard annealed (κ=1/2)
    _, err1 = sinkhorn_track(p, q, c, niter=niter, OT_cost=opt_cost,
                            kappa=1/2, beta0=10.0, debiased=False)
    
    # Debiased annealed (κ=2/3)
    _, err2 = sinkhorn_track(p, q, c, niter=niter, OT_cost=opt_cost,
                            kappa=2/3, beta0=10.0, debiased=True)
    
    # Plot results
    plot_pareto_front(std_results, beta0s, err1, err2, 'output/random-case.png')

def run_geometric_experiment(key=jax.random.PRNGKey(0)):
    """Run geometric cost experiment from paper."""
    print("Running geometric experiment...")
    
    # Create geometric problem
    X, Y, p, q, c = create_geometric_case()
    
    # Plot source and target distributions
    plot_geometric_data(X, Y)
    
    # Compute exact OT solution
    _, opt_cost = compute_exact_ot(p, q, c)
    print(f"Exact OT cost: {opt_cost:.4f}")
    
    # Parameters exactly as in Julia
    niter = 1000
    beta0 = 10.0
    
    # Standard Sinkhorn with different β₀ values
    beta0s = jnp.logspace(1, 3, num=21)  # 10 to 1000
    print(f"Running standard Sinkhorn with {len(beta0s)} different β₀...")
    
    std_results = []
    for beta0_i in beta0s:
        # Run without annealing (κ=0) and no debiasing
        _, errors = sinkhorn_track(p, q, c, niter=niter, OT_cost=opt_cost, 
                                 kappa=0.0, beta0=beta0_i, debiased=False)
        std_results.append(errors)
    
    # Run annealed versions (exactly as in Julia)
    print("Running annealed versions...")
    # Standard annealed (κ=1/2)
    _, err1 = sinkhorn_track(p, q, c, niter=niter, OT_cost=opt_cost,
                            kappa=1/2, beta0=10.0, debiased=False)
    
    # Debiased annealed (κ=2/3)
    _, err2 = sinkhorn_track(p, q, c, niter=niter, OT_cost=opt_cost,
                            kappa=2/3, beta0=10.0, debiased=True)
    
    # Plot results
    plot_pareto_front(std_results, beta0s, err1, err2, 'output/pareto-front-geo.png')

def run_piecewise_constant_experiment(key=jax.random.PRNGKey(0)):
    """Run piecewise constant schedules experiment."""
    print("Running piecewise constant schedules experiment...")

    # Create random problem
    p, q, c = create_random_problem(key)

    # Compute exact OT solution
    _, opt_cost = compute_exact_ot(p, q, c)
    print(f"Exact OT cost: {opt_cost:.4f}")

    # Experiment parameters
    niter = 2000
    beta0 = 10

    # Run configurations
    print("Running configurations...")
    # a. κ=1/2 with plateau_length=1 and 8
    _, err1 = sinkhorn_track(p, q, c, niter=niter, OT_cost=opt_cost,
                            kappa=1/2, beta0=beta0, debiased=False, plateau_length=1)
    _, err2 = sinkhorn_track(p, q, c, niter=niter, OT_cost=opt_cost,
                            kappa=1/2, beta0=beta0, debiased=False, plateau_length=8)

    # b. κ=3/4 with plateau_length=1 and 8
    _, errb1 = sinkhorn_track(p, q, c, niter=niter, OT_cost=opt_cost,
                             kappa=3/4, beta0=beta0, debiased=False, plateau_length=1)
    _, errb2 = sinkhorn_track(p, q, c, niter=niter, OT_cost=opt_cost,
                             kappa=3/4, beta0=beta0, debiased=False, plateau_length=8)

    # c. κ=2/3 (debiased) with plateau_length=1
    _, errc1 = sinkhorn_track(p, q, c, niter=niter, OT_cost=opt_cost,
                             kappa=2/3, beta0=beta0, debiased=True, plateau_length=1)

    # Plotting
    print("Plotting results...")
    plt.figure(figsize=(7, 5))

    # Three pairs of curves:
    # err1/err2 (κ=1/2) in "C0" color
    plt.loglog(err1 + 1e-16, "C0", label=r"$\kappa=1/2$", lw=3)
    plt.semilogy(err2 + 1e-16, "C0--", label=r"$\kappa=1/2$ (piecewise cst)")

    # errb1/errb2 (κ=3/4) in "C1" color
    plt.semilogy(errb1 + 1e-16, "C1", label=r"$\kappa=3/4$", lw=3)
    plt.semilogy(errb2 + 1e-16, "C1--", label=r"$\kappa=3/4$ (piecewise cst)")

    # errc1 (κ=2/3 debiased) in "C3" color
    plt.semilogy(errc1 + 1e-16, "C3", label=r"$\kappa=2/3$ (debiased)", lw=3)

    plt.xlabel(r"Iteration $t$")
    plt.ylabel("OT suboptimality after projection")
    plt.grid("on")
    plt.legend()
    # plt.savefig("output/piecewise-constant.png", bbox_inches="tight", dpi=200)
    plt.show()
    plt.close()

# %%
if __name__ == "__main__":
    # Create output directory
    import os
    os.makedirs('output', exist_ok=True)
    
    # Set random seed
    key = jax.random.PRNGKey(1)  # Same seed as Julia
    
    # Run experiments
    run_pareto_front_experiment(key)
    run_geometric_experiment(key)
    run_piecewise_constant_experiment(key)
