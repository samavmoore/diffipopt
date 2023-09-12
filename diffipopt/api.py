from collections import namedtuple
import jax
import jax.numpy as np
import cyipopt
from trapezoidal import Trapezoidal, Trapezoidal_Meta_Data
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
    params = jax.lax.stop_gradient(parameters_instance)

    meta_dat_shape_dtype = jax.ShapeDtypeStruct(shape=(1, 3), dtype=np.int64)
    dat_shapes = jax.pure_callback(define_meta_getter(prblm), meta_dat_shape_dtype, params, vectorized=False)
    n_vars, n_constraints, n_params = dat_shapes[0, 0], dat_shapes[0, 1], dat_shapes[0, 2]
    n_constraints = n_constraints + 2*n_vars # for the bounds on the decision variables including all of the states and inputs and the final time

    if isinstance(params, jax.core.Tracer):
        # seeing if the computation is being traced (i.e., during vectorized evaluation)
        n_vars = n_vars.val[0]
        n_constraints = n_constraints.val[0]
        n_params = n_params.val[0]

    _solve = _define_solve(prblm)

    x_shape_dtype = jax.ShapeDtypeStruct(shape=(1, n_vars), dtype=np.float64)
    lam_shape_dtype = jax.ShapeDtypeStruct(shape=(1, n_constraints), dtype=np.float64)

    x, lam = jax.pure_callback(_solve, (x_shape_dtype, lam_shape_dtype), params, vectorized=False)

    #tf = x[0, 0]
    #states_inputs = x[0, 1:].reshape((grid_pts, n_states + n_inputs))
    #states = states_inputs[:, :n_states]
    #inputs = states_inputs[:, n_states:]
    #states = problem_instance.state_tup(*states.T)
    #inputs = problem_instance.input_tup(*inputs.T)
            
    return x, lam

def define_meta_getter(prblm):
    
    def meta_getter(params):

        if hasattr(prblm, 'integration_type'):
            if prblm.integration_type == 'trapezoidal':
                problem_cls = Trapezoidal_Meta_Data(prblm, params)
            #elif prblm.integration_type == 'hermite-simpson':
            #    problem_cls = HermiteSimpson_Meta_Data(prblm, params)
            #else:
            #    raise ValueError('integration_type must be either "trapezoidal" or "hermite-simpson"')

        #elif isinstance(prblm, Problem):
        #    problem_cls = Standard(prblm, params)

        n_vars = problem_cls.n_vars
        n_constraints = problem_cls.n_constraints
        n_params = problem_cls.n_params
        
        return np.array([n_vars, n_constraints, n_params]).reshape((1, 3)).astype(np.int64)
    
    return meta_getter


def _define_solve(prblm):
    
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
        
        lam_g = info['mult_g']
        lam_xL = info['mult_x_L']
        lam_xU = info['mult_x_U']
        lam = np.concatenate([lam_g, lam_xL, lam_xU])

        return x.astype(np.float64).reshape((-1, problem_cls.n_vars)), lam.astype(np.float64).reshape((-1, problem_cls.n_constraints))

    return _solve
        


solve = jax.custom_jvp(solve, nondiff_argnums=(1,))

def _define_KKT_solve(prblm):

    def _KKT_solve(params):
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
        
        lam_g = info['mult_g']
        lam_xL = info['mult_x_L']
        lam_xU = info['mult_x_U']
        lam = np.concatenate([lam_g, lam_xL, lam_xU])

        dKKT_dp = del_KKT_del_p(problem_cls, params, (x, lam_g, lam_xL, lam_xU))
        dKKT_dx_lambda = del_KKT_del_x_lambda(problem_cls, params, (x, lam_g, lam_xL, lam_xU))
        dKKT_dx_lambda = dKKT_dx_lambda + np.eye(dKKT_dx_lambda.shape[0]) * 1e-10
        dx_lam_dp = -np.linalg.solve(dKKT_dx_lambda, dKKT_dp)

        x = x.astype(np.float64).reshape((-1, problem_cls.n_vars))
        lam = lam.astype(np.float64).reshape((-1, problem_cls.n_constraints))

        dx_lam_dp = dx_lam_dp.astype(np.float64).reshape((problem_cls.n_vars + problem_cls.n_constraints, problem_cls.n_params))

        return x, lam, dx_lam_dp
    
    return _KKT_solve


@solve.defjvp
def _solve_jvp(prblm, primals, tangents):
    (parameters_instance, ) = primals
    (parameters_instance_dot, ) = tangents

    # initialize the problem instance
    print("Initializing problem")
    prblm = _initialize_problem(prblm)
    params = jax.lax.stop_gradient(parameters_instance)

    meta_dat_shape_dtype = jax.ShapeDtypeStruct(shape=(1, 3), dtype=np.int64)
    dat_shapes = jax.pure_callback(define_meta_getter(prblm), meta_dat_shape_dtype, params, vectorized=False)
    n_vars, n_constraints, n_params = dat_shapes[0, 0], dat_shapes[0, 1], dat_shapes[0, 2]
    n_constraints = n_constraints + 2*n_vars # for the bounds on the decision variables including all of the states and inputs and the final time

    if isinstance(params, jax.core.Tracer):
        # seeing if the computation is being traced (i.e., during vectorized evaluation)
        n_vars = n_vars.val[0]
        n_constraints = n_constraints.val[0]
        n_params = n_params.val[0]

    _KKT_solve = _define_KKT_solve(prblm)

    x_shape_dtype = jax.ShapeDtypeStruct(shape=(1, n_vars), dtype=np.float64)
    lam_shape_dtype = jax.ShapeDtypeStruct(shape=(1, n_constraints), dtype=np.float64)
    dx_lam_dp_shape_dtype = jax.ShapeDtypeStruct(shape=(n_vars + n_constraints, n_params), dtype=np.float64)


    x, lam, dx_lam_dp  = jax.pure_callback(_KKT_solve, (x_shape_dtype, lam_shape_dtype, dx_lam_dp_shape_dtype), params, vectorized=False)

    tanget_out = dx_lam_dp @ np.array([*parameters_instance_dot])

    tanget_out_x = tanget_out[:n_vars].reshape((1, n_vars))
    tanget_out_lam = tanget_out[n_vars:].reshape((1, n_constraints))

    return (x, lam), (tanget_out_x, tanget_out_lam)



def del_KKT_del_p(problem_cls, parameters_instance, solution):
    prob = problem_cls
    params = parameters_instance
    x, lam_g, lam_xL, lam_xU = solution
    lams = np.concatenate([lam_g, lam_xL, lam_xU])

    obj_hess_px = np.array([f(x, params) for f in prob.obj_hess_px]).squeeze(0).T
    con_hess_px_dot_lam = np.vstack([*prob.all_constraints_hess_px_lam(x, params, lams)]).T
    top_block = obj_hess_px + con_hess_px_dot_lam

    jac_g_p = [g(x, params) for g in prob.all_constraints_jac_p]

    jac_g_p = np.vstack([np.vstack([*jac]).T for jac in jac_g_p])

    bottom_block = jac_g_p * lams[:, np.newaxis]

    return np.vstack([top_block, bottom_block])

def del_KKT_del_x_lambda(problem_cls, parameters_instance, solution):
    prob = problem_cls
    params = parameters_instance
    x, lam_g, lam_xL, lam_xU = solution
    lams = np.concatenate([lam_g, lam_xL, lam_xU])

    obj_hess_xx = np.array([f(x, params) for f in prob.obj_hess_xx]).squeeze(0)
    con_hess_xx_dot_lam = prob.all_constraints_hess_xx_lam(x, params, lams)
    block_00 =  obj_hess_xx + con_hess_xx_dot_lam
    block_01 = np.vstack([g(x, params) for g in prob.all_constraints_jac_x]).T
    block_10 = np.diag(lams) @ block_01.T
    block_11 = np.diag(np.concatenate([g(x, params) for g in prob.all_constraints]))

    top = np.hstack([block_00, block_01])
    bottom = np.hstack([block_10, block_11])

    return np.vstack([top, bottom])


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





