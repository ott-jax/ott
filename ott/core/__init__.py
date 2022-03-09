# coding=utf-8
"""OTT core libraries: the engines behind most computations happening in OTT."""

# pytype: disable=import-error  # kwargs-checking
from . import anderson
from . import dataclasses
from . import discrete_barycenter
from . import gromov_wasserstein
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
