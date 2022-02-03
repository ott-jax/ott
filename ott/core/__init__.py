# coding=utf-8
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

"""OTT core libraries: the engines behind most computations happening in OTT."""

# pytype: disable=import-error  # kwargs-checking
from . import anderson
from . import dataclasses
from . import discrete_barycenter
from . import gromov_wasserstein
from . import icnn
from . import implicit_differentiation
from . import momentum
from . import problems
from . import sinkhorn
from . import sinkhorn_lr
from .implicit_differentiation import ImplicitDiff
from .problems import LinearProblem
from .sinkhorn import Sinkhorn
from .sinkhorn_lr import LRSinkhorn

# pytype: enable=import-error  # kwargs-checking
