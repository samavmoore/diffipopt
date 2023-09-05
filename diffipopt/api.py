from collections import namedtuple
import jax
import jax.numpy as np
import numpy as onp
import cyipopt
from jax import core
from jax.tree_util import Partial
from functools import partial
import inspect
import scipy.sparse as sparse
import time
jax.config.update("jax_enable_x64", True)

# naming the fields of the problem and control problem tuples
fields = ['cost', 'constraints', 'bounds' 'options']
Problem = namedtuple("Problem", fields, defaults=(None for i in range(len(fields))))
"""
Problem namedtuple representing a conventional optimization problem.

Attributes:
    cost (Callable[[NamedTuple, NamedTuple], Union[jax.numpy.ndarray, float]]):
        Objective function with signature f(x, p) where:
        x: Decision variables.
        p: Parameters (user-defined namedtuples).
        
    constraints (NamedTuple):
        Constraint functions with the signature g(x, p) <= 0 passed in as an instace of the namedtuple Constraint with the field g.
        
    bounds (NamedTuple):
        The bounds on the decision variables, an instance of BoundingBox namedtuples.
        
    options (Optional[Dict[str, Any]):
        Additional optional configurations or settings for the optimization problem.
"""
control_fields = ['integration_type', 'final_cost', 'path_cost', 'initial_state', 'path_state', 
        'final_state', 'input', 'initial_g', 'path_g', 'final_g', 
        'final_time', 'state_tup', 'input_tup', 'param_tup', 'dynamics', 'grid_pts', 'options']

ControlProblem = namedtuple("ControlProblem", control_fields, defaults=(None for i in range(len(control_fields))))

Constraint  = namedtuple('Constraint',['g'])

BoundingBox = namedtuple('BoundingBox', ['lb', 'ub'])



    

    






