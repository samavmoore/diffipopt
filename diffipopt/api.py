from collections import namedtuple
import jax
import jax.numpy as np
import cyipopt
from trapezoidal import Trapezoidal
from hermite_simpson import HermiteSimpson
from standard import Standard
from jax.tree_util import Partial
from common import initial_guess_traj_opt
import inspect
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


def solve(parameters_instance, problem_instance):
    '''Solve an optimization problem instance'''

    # initialize the problem instance
    print("Initializing problem")
    prblm = _initialize_problem(problem_instance)
    parameters_instance = jax.lax.stop_gradient(parameters_instance)

    def _solve(params):

        if hasattr(prblm, 'integration_type'):
            if prblm.integration_type == 'trapezoidal':
                problem_cls = Trapezoidal(prblm, params)
            elif prblm.integration_type == 'hermite-simpson':
                problem_cls = HermiteSimpson(prblm, params)
            else:
                raise ValueError('integration_type must be either "trapezoidal" or "hermite-simpson"')

        elif isinstance(prblm, Problem):
            problem_cls = Standard(prblm, params)
        
        xL, xU = problem_cls.xL, problem_cls.xU
        gL, gU = problem_cls.gL, problem_cls.gU
        x0 = initial_guess_traj_opt(prblm, params)
        nlp = cyipopt.Problem(n=len(x0), m=len(gL), problem_obj=problem_cls, lb=xL, ub=xU, cl=gL, cu=gU)
        nlp.add_option('tol', 1e-4)
        nlp.add_option('max_iter', int(1e5))
        nlp.add_option('nlp_scaling_method', 'gradient-based')

        print("Solving problem")
        x, info = nlp.solve(x0)

        return x.astype(np.float64).reshape((-1, problem_cls.n_vars))
    
    n_states = len(prblm.state_tup._fields)
    n_inputs = len(prblm.input_tup._fields)
    grid_pts = prblm.grid_pts
    n_vars = (n_states + n_inputs)*grid_pts + 1

    result_shape_dtype = jax.ShapeDtypeStruct(shape=(1, n_vars), dtype=np.float64)

    x = jax.pure_callback(_solve, result_shape_dtype, parameters_instance, vectorized=False)

    tf = x[0, 0]
    states_inputs = x[0, 1:].reshape((grid_pts, n_states + n_inputs))
    states = states_inputs[:, :n_states]
    inputs = states_inputs[:, n_states:]
    states = problem_instance.state_tup(*states.T)
    inputs = problem_instance.input_tup(*inputs.T)
            
    return tf, states, inputs
    

solve = jax.custom_jvp(solve)

@solve.defjvp
def _solve_jvp(primals, tangents):

    return primals, tangents

def del_KKT_del_p(problem_cls, parameters_instance, solution):
    prob = problem_cls
    params = parameters_instance
    x, lam_g, lam_xL, lam_xU = solution
    lams = np.concatenate([lam_g, lam_xL, lam_xU])

    top = np.array([prob.obj_hess_px(x, params) + prob.all_constraints_hess_px_lam(x, params, lams)])

    jac_g_p = prob.all_constraints_jac_p(x, params)

    bottom = np.vstack([lams[i]*jac_g_p[i, :] for i in range(len(lams))])

    return np.vstack([top, bottom])

def del_KKT_del_x_lambda(problem_instance, parameters_instance, solution):
    prob = problem_instance
    params = parameters_instance
    x, lam_g, lam_xL, lam_xU = solution
    lams = np.concatenate([lam_g, lam_xL, lam_xU])

    block_00 = prob.obj_hess_xx(x, params) + prob.all_constraints_hess_xx_lam(x, params, lams)
    block_01 = prob.all_constraints_jac_x(x, params)
    block_10 = prob.all_constraints_jac_x(x, params).T @ np.diag(lams)
    block_11 = np.diag(prob.all_constraints(x, params))

    return np.array([[block_00, block_01], [block_10, block_11]])


def _check_control_problem(problem_instance):
    '''Check if the fields of a ControlProblem namedtuple are valid and wrap callable fields/subfields with Partial'''
     # Wrap functions in ControlProblem or Problem with Partial and perform error-checking
    wrapped_fields = {}
    bounding_box = {}
    constraints = {}

    for field in problem_instance._fields:
        attr = getattr(problem_instance, field)
        
        # Check if final_cost and path_cost are functions with correct number of arguments
        if field in ['final_cost', 'path_cost']:
            if attr is None:
                continue
            if not callable(attr):
                raise ValueError(f"{field} should be a callable function")
            if len(inspect.signature(attr).parameters) != 3:
                raise ValueError(f"{field} should accept exactly 3 arguments (states namedtuple, inputs namedtuple, parameters namedtuple)")
        
        # Check if the state constraint fields are of type BoundingBox and have correct number of arguments convert them to Partial
        elif field in ['initial_state', 'path_state', 'final_state', 'input', 'final_time']: 
            if attr is None:
                raise ValueError(f"{field} should be not be None")
            if not isinstance(attr, BoundingBox):
                raise ValueError(f"{field} should be of type BoundingBox")
            for subfield in attr._fields:
                subattr = getattr(attr, subfield)
                if subfield in ['lb', 'ub']:
                    if not callable(subattr):
                        raise ValueError(f"{field}.{subfield} should be a callable function")
                    if len(inspect.signature(subattr).parameters) != 1:
                        raise ValueError(f"{field}.{subfield} should accept exactly 1 argument (parameter namedtuple)")
                    
                    if not isinstance(subattr, Partial):
                        bounding_box[subfield] = subattr #Partial(subattr)
                    else:
                        bounding_box[subfield] = subattr
                else:
                    raise ValueError(f"{field} should only have 'lb' and 'ub' fields")
                
            wrapped_fields[field] = BoundingBox(**bounding_box)
            continue
            
        # Check if the 'g' fields are of type Constraint and have correct number of arguments convert them to Partial
        elif field in ['initial_g', 'path_g', 'final_g']:
            if attr is None:
                continue 
            if not isinstance(attr, Constraint):
                raise ValueError(f"{field} should be of type Constraint")
            for subfield in attr._fields:
                subattr = getattr(attr, subfield)
                if subfield == 'g':
                    if not callable(subattr):
                        raise ValueError(f"{field}.{subfield} should be a callable function")
                    
                    if len(inspect.signature(subattr).parameters) != 3:
                            raise ValueError(f"{field}.{subfield} should accept exactly 3 arguments (states namedtuple, inputs namedtuple, parameters namedtuple)")
                        
                    if not isinstance(subattr, Partial):
                        constraints[subfield] = subattr #Partial(subattr)
                    else:
                        constraints[subfield] = subattr

                else:
                    raise ValueError(f"{field} should only have 'g' field")
            
            wrapped_fields[field] = Constraint(**constraints)
            continue

        # check if state_tup and input_tup are namedtuples
        elif field in ['state_tup', 'input_tup', 'param_tup']:
            if not issubclass(attr, tuple) or not hasattr(attr, '_fields'):
                raise ValueError(f"{field} should be of type namedtuple")
            wrapped_fields[field] = attr
            continue

        
        elif field == 'dynamics':
            if not callable(attr):
                raise ValueError(f"{field} should be a callable function")
            if len(inspect.signature(attr).parameters) != 3:
                raise ValueError(f"{field} should accept exactly 3 arguments (states namedtuple, inputs namedtuple, parameters namedtuple)")
            
        elif field == 'grid_pts':
            if not isinstance(attr, int):
                raise ValueError(f"{field} should be of type int")
            if attr < 2:
                raise ValueError(f"{field} should be greater than 1")
        
        elif field == 'options':
            if attr is None:
                continue
            if not isinstance(attr, dict):
                raise ValueError(f"{field} should be of type dict")

        # Wrap callable fields with Partial
        if callable(attr) and not isinstance(attr, Partial):
            wrapped_fields[field] = attr #Partial(attr)
        else:
            wrapped_fields[field] = attr

    return ControlProblem(**wrapped_fields)

def _check_problem(problem_instance):
    '''Check if the fields of a Problem namedtuple are valid and wrap callable field/subfields with Partial'''
    wrapped_fields = {}
    bounding_box = {}
    constraints = {}

    for field in problem_instance._fields:
        attr = getattr(problem_instance, field)

        if field == 'cost':
            if not callable(attr):
                raise ValueError(f"{field} should be a callable function")
            if len(inspect.signature(attr).parameters) != 2:
                raise ValueError(f"{field} should accept exactly 2 arguments (vars namedtule,  parameters namedtuple)")
        
        elif field == 'constraints':
            if attr is None:
                continue
            if not isinstance(attr, Constraint):
                raise ValueError(f"{field} should be of type Constraint")
            for subfield in attr._fields:
                subattr = getattr(attr, subfield)
                if subfield == 'g':
                    if not callable(subattr):
                        raise ValueError(f"{field}.{subfield} should be a callable function")
                    
                    if len(inspect.signature(subattr).parameters) != 2:
                            raise ValueError(f"{field}.{subfield} should accept exactly 2 arguments (vars namedtuple, parameters namedtuple)")
                        
                    if not isinstance(subattr, Partial):
                        constraints[subfield] = Partial(subattr)
                    else:
                        constraints[subfield] = subattr

                else:
                    raise ValueError(f"{field} should only have 'g' field")
            
            wrapped_fields[field] = Constraint(**constraints)
            continue

        elif field == 'bounds':
            if attr is None:
                continue
            if not isinstance(attr, BoundingBox):
                raise ValueError(f"{field} should be of type BoundingBox")
            for subfield in attr._fields:
                subattr = getattr(attr, subfield)
                if subfield in ['lb', 'ub']:
                    if not callable(subattr):
                        raise ValueError(f"{field}.{subfield} should be a callable function")
                    if len(inspect.signature(subattr).parameters) != 1:
                        raise ValueError(f"{field}.{subfield} should accept exactly 1 argument (parameter namedtuple)")
                    
                    if not isinstance(subattr, Partial):
                        bounding_box[subfield] = Partial(subattr)
                    else:
                        bounding_box[subfield] = subattr
                else:
                    raise ValueError(f"{field} should only have 'lb' and 'ub' fields")
            
            wrapped_fields[field] = BoundingBox(**bounding_box)
            continue

        elif field == 'options':
            if attr is None:
                continue
            if not isinstance(attr, dict):
                raise ValueError(f"{field} should be of type dict")

    return Problem(**wrapped_fields)


def _initialize_problem(problem_instance):
    '''
        Initializes a optimization problem instance from a ControlProblem or Problem namedtuple by performing error-checking 
        and making sure that functions are compatable with JAX transformations by wrapping them with jax.tree_util.Partial if not done so already.
    '''

    # Check if the instance is either a ControlProblem or a Problem
    if not isinstance(problem_instance, (ControlProblem, Problem)):
        raise TypeError('problem_instance must be of type ControlProblem or Problem (namedtuples)')
    
    # Check if the instance is a ControlProblem and wrap functions with Partial and perform error-checking
    if isinstance(problem_instance, ControlProblem):
        problem_instance = _check_control_problem(problem_instance)
    else:
        problem_instance = _check_problem(problem_instance)
    
    return problem_instance





