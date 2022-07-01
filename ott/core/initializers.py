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
from typing import Optional

from ott.core.ot_problems import LinearProblem
from ott.geometry.pointcloud import PointCloud



class SinkhornInitializer():

    def init_dual_a(self, ot_problem: LinearProblem, lse_mode: bool = True) -> jnp.ndarray:
        """
        Input:
            ot_problem: OT problem between discrete distributions of size n and m
        
        Return:
            dual potential, array of size m
        """
        a = ot_problem.a
        init_dual_a = jnp.zeros_like(a) if lse_mode else jnp.ones_like(a)
        return init_dual_a


    def init_dual_b(self, ot_problem: LinearProblem, lse_mode: bool = True) -> jnp.ndarray:
        """
        Input:
            ot_problem: OT problem between discrete distributions of size n and m
        
        Return:
            dual potential, array of size m
        """
        b = ot_problem.b
        init_dual_b = jnp.zeros_like(b) if lse_mode else jnp.ones_like(b)
        return init_dual_b
    
    def remove_null_weight_potentials(self, ot_problem, init_dual_a, init_dual_b, lse_mode: bool=True):
         # Cancel dual variables for zero weights.
        a, b = ot_problem.a, ot_problem.b
        init_dual_a = jnp.where(
            a > 0, init_dual_a, -jnp.inf if lse_mode else 0.0
        )
        init_dual_b = jnp.where(
            b > 0, init_dual_b, -jnp.inf if lse_mode else 0.0
        )
        return init_dual_a, init_dual_b

    def default_dual_a(self, ot_problem, lse_mode):
        a = ot_problem.a
        init_dual_a = jnp.zeros_like(a) if lse_mode else jnp.ones_like(a)
        return init_dual_a

    def default_dual_b(self, ot_problem, lse_mode):
        b = ot_problem.b
        init_dual_b = jnp.zeros_like(b) if lse_mode else jnp.ones_like(b)
        return init_dual_b


class GaussianInitializer(SinkhornInitializer):

    def __init__(self, stop_gradient: Optional[bool] =True) -> None:
        """_summary_

        Args:
            stop_gradient (bool, optional): _description_. Defaults to True.
        """
        super().__init__()
        
        self.stop_gradient = stop_gradient

    
    def init_dual_a(self, ot_problem: LinearProblem, init_f: Optional[jnp.ndarray] =None, lse_mode: bool = True) -> jnp.ndarray:

        """_summary_

        Returns:
            _type_: _description_
        """
        from ott.tools.gaussian_mixture.gaussian import Gaussian

        cost_matrix = ot_problem.geom.cost_matrix
        if self.stop_gradient:
            cost_matrix = jax.lax.stop_gradient(cost_matrix)

        n = cost_matrix.shape[0]
        f_potential = jnp.zeros(n) if init_f is None else init_f

        if not isinstance(ot_problem.geom, PointCloud):
            return f_potential

        else:
            x = ot_problem.geom.x
            y = ot_problem.geom.y
            gaussian_a = Gaussian.from_samples(x, weights=ot_problem.a)
            gaussian_b = Gaussian.from_samples(y, weights=ot_problem.b)
            f_potential = gaussian_a.f_potential(dest=gaussian_b, points=x)

        return f_potential

class SortingInit(SinkhornInitializer):

    def __init__(self, 
                vector_min: Optional[bool] = False, 
                tol: Optional[float] = 1e-2, 
                max_iter: Optional[int] = 100, 
                stop_gradient: Optional[bool] = True) -> None:
        """_summary_

        Args:
            vector_min (Optional[bool], optional): _description_. Defaults to False.
            tol (Optional[float], optional): _description_. Defaults to 1e-2.
            max_iter (Optional[int], optional): _description_. Defaults to 100.
            stop_gradient (Optional[bool], optional): _description_. Defaults to True.
        """
        super().__init__()
        
        self.tolerance = tol
        self.stop_gradient = stop_gradient
        self.max_iter = max_iter
        self.update_fn = self.vectorized_update if vector_min else self.coordinate_update

    def vectorized_update(self, f: jnp.ndarray, modified_cost: jnp.ndarray):
        """_summary_

        Args:
            f (jnp.ndarray): _description_
            modified_cost (jnp.ndarray): _description_

        Returns:
            _type_: _description_
        """
        f = jnp.min(modified_cost + f[None, :], axis=1)
        return f


    @jax.jit
    def coordinate_update(self, f: jnp.ndarray, modified_cost: jnp.ndarray):
        """_summary_

        Args:
            f (jnp.ndarray): _description_
            modified_cost (jnp.ndarray): _description_
        """
        
        def body_fn(i, f):
            new_f = jnp.min(modified_cost[i, :] + f)
            f = f.at[i].set(new_f)
            return f

        return jax.lax.fori_loop(0, len(f), body_fn, f)

    @functools.partial(jax.jit, static_argnums=(1, 2, 3))  
    def init_sorting_dual(self, modified_cost: jnp.ndarray, f_potential: jnp.ndarray):
        """_summary_

        Args:
            modified_cost (jnp.ndarray): _description_
            f_potential (jnp.ndarray): _description_

        Returns:
            _type_: _description_
        """
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
    
    def init_dual_a(self, ot_problem: LinearProblem, init_f: jnp.ndarray = None, lse_mode: bool = True) -> jnp.ndarray:
        """

        Args:
            ot_problem (LinearProblem): _description_
            init_f (jnp.ndarray, optional): _description_. Defaults to None.

        Returns:
            jnp.ndarray: _description_
        """
        cost_matrix = ot_problem.geom.cost_matrix
        if self.stop_gradient:
            cost_matrix = jax.lax.stop_gradient(cost_matrix)

        modified_cost = cost_matrix - jnp.diag(cost_matrix)[None, :]
        
        n = cost_matrix.shape[0]
        f_potential = jnp.zeros(n) if init_f is None else init_f

        f_potential = self.init_sorting_dual(modified_cost, f_potential)


        return f_potential




    






