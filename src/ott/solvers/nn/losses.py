# Copyright OTT-JAX
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp
from ott.geometry import pointcloud
from ott.solvers.linear import sinkhorn
from ott.problems.linear import linear_problem
import ott.geometry.costs as costs
from typing import Any, Mapping, Callable
from types import MappingProxyType

class MongeGap:
    """
    A class to define the Monge gap regularizer.
    For a cost function :`math:`c` and an empirical reference :math:`\rho_x`
    defined by samples :math:`x_i`, the (entropic) Monge gap of a vector field :math:`T` 
    is defined as:
        :math:`\mathcal{M}^c_{\rho_x, \epsilon} = \frac{1}{n} \sum_{i=1}^n c(x_i, T(x_i)) 
        - W_\epsilon(\mu_x, T \# \rho_x)`.
    
    Args:
    geometry_kwargs: Holds the kwargs to instanciate the geometry to compute the ``reg_ot_cost``. 
                     Default cost function (if no cost function is passed in ``geometry_kwargs``)
                     is the quadratic costs ``ott.geometry.costs.SqEuclidean()``.
    sinkhorn_kwargs: Holds the kwargs to instanciate the Sinkhorn solver to compute the ``reg_ot_cost``.
    """

    def __init__(
        self,
        geometry_kwargs: Mapping[str, Any] = MappingProxyType({}),
        sinkhorn_kwargs: Mapping[str, Any] = MappingProxyType({}),
    ) -> None:
        self.geometry_kwargs = geometry_kwargs
        self.sinkhorn_kwargs = sinkhorn_kwargs

    def __call__(
        self, samples: jnp.ndarray, T: Callable[[jnp.ndarray], jnp.ndarray]
    ) -> float:
        """
        Evaluate the Monge gap of vector field ``T``,
        on the empirical reference measure samples defined by ``samples``.
        """
        T_samples = T(samples)
        geom = pointcloud.PointCloud(
            x=samples, y=T_samples,
            **self.geometry_kwargs
        )
        id_displacement = jnp.mean(
            jax.vmap(self.cost_fn)(samples, T_samples)
        )
        opt_displacement = sinkhorn.Sinkhorn(
            **self.sinkhorn_kwargs
        )(
            linear_problem.LinearProblem(geom)
        ).reg_ot_cost
        opt_displacement = jnp.add(
            opt_displacement,
            - 2 * geom.epsilon * jnp.log(len(samples))
        )  # use Shannon entropy instead of relative entropy as entropic regularizer
           # to ensure Monge gap positivity
        
        return id_displacement - opt_displacement
           
    @property
    def cost_fn(self) -> costs.CostFn:
        """"
        Set cost function on which Monge gap is instanciated.
        Default is squared euclidean.
        """
        if "cost_fn" in self.geometry_kwargs:
            return self.geometry_kwargs["cost_fn"]
        else:
            return costs.SqEuclidean()