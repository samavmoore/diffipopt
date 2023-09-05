
import jax
import jax.numpy as np
from jax.tree_util import Partial
import scipy.sparse as sparse
from collections import namedtuple
from api import BoundingBox
from common import _filter_lbs, _filter_ubs
jax.config.update("jax_enable_x64", True)

class Standard():
    def __init__(self, Problem, Parameters):
        self.problem = Problem
        self.params = Parameters

    def objective(self, x):
        return self.problem.cost.fun(x, self.params.p)