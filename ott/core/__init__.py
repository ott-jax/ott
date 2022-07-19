# Copyright 2022 Google LLC.
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
"""OTT core libraries: the engine behind most computations happening in OTT."""

# pytype: disable=import-error  # kwargs-checking
from . import (
    anderson,
    bar_problems,
    continuous_barycenter,
    dataclasses,
    discrete_barycenter,
    gromov_wasserstein,
    gw_barycenter,
    implicit_differentiation,
    linear_problems,
    momentum,
    quad_problems,
    sinkhorn,
    sinkhorn_lr,
)

# from . import neuraldual
from .implicit_differentiation import ImplicitDiff
from .linear_problems import LinearProblem
from .sinkhorn import Sinkhorn
from .sinkhorn_lr import LRSinkhorn

# pytype: enable=import-error  # kwargs-checking
