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
from . import fixed_point_loop, matrix_square_root, unbalanced_functions, utils
from ._lbfgs import lbfgs
from ._legendre import legendre
from ._velocity_from_brenier_potential import velocity_from_brenier_potential

__all__ = [
    "fixed_point_loop",
    "matrix_square_root",
    "unbalanced_functions",
    "utils",
    "lbfgs",
    "legendre",
    "velocity_from_brenier_potential",
]
