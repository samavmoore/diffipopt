from collections import namedtuple
import jax
import jax.numpy as np
import cyipopt
from trapezoidal import Trapezoidal
from hermite_simpson import HermiteSimpson
from standard import Standard
from common import initial_guess_traj_opt, initialize_problem
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



    
def solve(problem_instance, parameters_instance):
    '''Solve an optimization problem instance'''

    # initialize the problem instance
    print("Initializing problem")
    problem_instance = initialize_problem(problem_instance)

    if hasattr(problem_instance, 'integration_type'):
        if problem_instance.integration_type == 'trapezoidal':
            problem_cls = Trapezoidal(problem_instance, parameters_instance)
        elif problem_instance.integration_type == 'hermite-simpson':
            problem_cls = HermiteSimpson(problem_instance, parameters_instance)
        else:
            raise ValueError('integration_type must be either "trapezoidal" or "hermite-simpson"')
        
    elif isinstance(problem_instance, Problem):
        problem_cls = Standard(problem_instance, parameters_instance)

    xL, xU = problem_cls.xL, problem_cls.xU
    gL, gU = problem_cls.gL, problem_cls.gU
    x0 = initial_guess_traj_opt(problem_instance, parameters_instance)
    nlp = cyipopt.Problem(n=len(x0), m=len(gL), problem_obj=problem_cls, lb=xL, ub=xU, cl=gL, cu=gU)
    nlp.add_option('tol', 1e-4)
    nlp.add_option('max_iter', int(1e5))
    nlp.add_option('nlp_scaling_method', 'gradient-based')

    print("Solving problem")
    x, info = nlp.solve(x0)
    tf = x[0]
    states_inputs = x[1:].reshape((problem_cls.grid_pts, problem_cls.n_states + problem_cls.n_inputs))
    states = states_inputs[:, :problem_cls.n_states]
    inputs = states_inputs[:, problem_cls.n_states:]
    states = problem_instance.state_tup(*states.T)
    inputs = problem_instance.input_tup(*inputs.T)

    return tf, states, inputs
    






