
import jax
import jax.numpy as np
from jax.tree_util import Partial
import scipy.sparse as sparse
from collections import namedtuple
from api import BoundingBox
from common import _filter_lbs, _filter_ubs, _get_B, _get_A, bound_box_func
jax.config.update("jax_enable_x64", True)

class HermiteSimpson():
    def __init__(self, Problem, Parameters):
        self.problem = Problem
        self.params = Parameters

    def objective(self, x):
        return self.problem.cost.fun(x, self.params.p)