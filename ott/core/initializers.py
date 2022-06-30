# Copyright 2022 The OTT Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sinkhorn initializers."""
import functools
import jax
from jax import numpy as jnp

from .linear_problems import LinearProblem
from ..tools.gaussian_mixture.gaussian import Gaussian
from ..geometry.pointcloud import PointCloud

@jax.tree_util.register_pytree_node_class
class SinkhornInitializer():

    def apply(self, linear_problem: LinearProblem) -> jnp.ndarray:
        """
        Input:
            linear_problem: OT problem between discrete distributions of size n and m
        
        Return:
            dual potential, array of size m
        """
        pass



class GaussianInitializer(SinkhornInitializer):

    def __init__(self, stop_gradient=True) -> None:
        super().__init__()
        
        self.stop_gradient = stop_gradient

    
    def apply(self, linear_problem: LinearProblem, init_f=None) -> jnp.ndarray:
        

        cost_matrix = linear_problem.geom.cost_matrix
        if self.stop_gradient:
            cost_matrix = jax.lax.stop_gradient(cost_matrix)

        n = cost_matrix.shape[0]
        f_potential = jnp.zeros(n) if init_f is None else init_f

        if not isinstance(linear_problem.geom, PointCloud):
            return f_potential

        else:
            x = linear_problem.geom.x
            y = linear_problem.geom.y
            gaussian_a = Gaussian.from_samples(x, linear_problem.a)
            gaussian_b = Gaussian.from_samples(y, linear_problem.b)
        
        f_potential = gaussian_a.f_potential(dest=gaussian_b, points=x)

        return f_potential

class SortingInit(SinkhornInitializer):

    def __init__(self, vector_min=False, tol=1e-2, max_iter=100, stop_gradient=True) -> None:
        super().__init__()
        
        self.tolerance = tol
        self.stop_gradient = stop_gradient
        self.max_iter = self.max_iter
        self.update_fn = self.vectorized_update if vector_min else self.coordinate_update

    def vectorized_update(self, f, modified_cost):
        f = jnp.min(modified_cost + f[None, :], axis=1)
        return f


    @jax.jit
    def coordinate_update(self, f, modified_cost):
        
        def body_fn(i, f):
            new_f = jnp.min(modified_cost[i, :] + f)
            f = f.at[i].set(new_f)
            return f

        return jax.lax.fori_loop(0, len(f), body_fn, f)

    @functools.partial(jax.jit, static_argnums=(1, 2, 3))  
    def init_sorting_dual(self, modified_cost, f_potential):
        it = 0
        diff = self.tolerance + 1.0

        state = (f_potential, diff, it)
        def body_fn(state):
            prev_f, _, it = state
            f_potential = self.update_fn(prev_f, modified_cost)
            diff = jnp.sum((f_potential - prev_f) ** 2)
            it += 1
            return f_potential, diff, it

        def cond_fn(state):
            _, diff, it = state
            return (diff > self.tolerance) & (it < self.mat_iter)

        f_potential, _, it = jax.lax.while_loop(cond_fun=cond_fn, body_fun=body_fn, init_val=state)

        return f_potential
    
    def apply(self, linear_problem: LinearProblem, init_f=None) -> jnp.ndarray:
        
        cost_matrix = linear_problem.geom.cost_matrix
        if self.stop_gradient:
            cost_matrix = jax.lax.stop_gradient(cost_matrix)

        modified_cost = cost_matrix - jnp.diag(cost_matrix)[None, :]
        
        n = cost_matrix.shape[0]
        f_potential = jnp.zeros(n) if init_f is None else init_f

        f_potential = self.init_sorting_dual(modified_cost, f_potential)

        return f_potential




    






