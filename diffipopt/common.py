import jax
import jax.numpy as np
from collections import namedtuple
from jax.tree_util import Partial
import inspect
import time
from api import Problem, ControlProblem, Constraint, BoundingBox
jax.config.update("jax_enable_x64", True)

bound_box_func = namedtuple(
            "bound_box_func", ["name", "start", "end", "bound_func1", "bound_func2", "filter_func"]
        )


def initial_guess_traj_opt(problem_instance, parameters_instance):
    '''Generate an initial guess for the optimization problem instance'''
    
    if hasattr(problem_instance, 'integration_type'):
        n_states = len(problem_instance.state_tup._fields)
        n_inputs = len(problem_instance.input_tup._fields)
        n_grid_pts = problem_instance.grid_pts

        # Determine the number of decision variables based on the integration type
        if problem_instance.integration_type == 'trapezoidal':
            n_vars = (n_states + n_inputs) * n_grid_pts + 1  # +1 for final time
        elif problem_instance.integration_type == 'hermite-simpson':
            n_vars = (n_states + n_inputs) * (2 * n_grid_pts - 1) + 1

        # Interpolate between the initial and final states with np.interp
        if problem_instance.initial_state is not None:
            initial_state = np.nan_to_num(np.asarray(problem_instance.initial_state.lb(parameters_instance)), nan=0).squeeze()
            final_state = np.nan_to_num(np.asarray(problem_instance.final_state.lb(parameters_instance)), nan=0).squeeze()

            def interp(s0, sf, tau):
                return s0 + (sf - s0) * tau
            
            state_vars = jax.vmap(Partial(interp, initial_state, final_state))(np.linspace(0, 1, n_grid_pts))

        # Guess a constant input
        if problem_instance.input is not None:
            lb = np.asarray(problem_instance.input.lb(parameters_instance))
            ub = np.asarray(problem_instance.input.ub(parameters_instance))
            average_input = (lb + ub) / 2.0
            input_vars = np.tile(average_input, (n_grid_pts, 1))

        # Initialize the final time tf
        tf_init = np.asarray(problem_instance.final_time.lb(parameters_instance))

        # Create the decision variable vector by interleaving states and inputs, and adding tf at the start
        vars = np.insert(np.hstack((state_vars, input_vars)).ravel(), 0, tf_init)
        
        assert len(vars) == n_vars, "The length of the initial guess does not match the expected number of variables."

        return vars
        

def check_control_problem(problem_instance):
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

def check_problem(problem_instance):
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


def initialize_problem(problem_instance):
    '''
        Initializes a optimization problem instance from a ControlProblem or Problem namedtuple by performing error-checking 
        and making sure that functions are compatable with JAX transformations by wrapping them with jax.tree_util.Partial if not done so already.
    '''

    # Check if the instance is either a ControlProblem or a Problem
    if not isinstance(problem_instance, (ControlProblem, Problem)):
        raise TypeError('problem_instance must be of type ControlProblem or Problem (namedtuples)')
    
    # Check if the instance is a ControlProblem and wrap functions with Partial and perform error-checking
    if isinstance(problem_instance, ControlProblem):
        problem_instance = check_control_problem(problem_instance)
    else:
        problem_instance = check_problem(problem_instance)
    
    return problem_instance

def _filter_lb(g_value, lb_val):

    def lb_finite(_):
        return np.array([lb_val - g_value])
    
    def lb_infinite(_):
        return np.array([np.nan])
    
    return jax.lax.cond(np.isinf(lb_val), lb_infinite, lb_finite, None)

def _filter_ub(g_value, ub_val):
    
        def ub_finite(_):
            return np.array([g_value - ub_val])
        
        def ub_infinite(_):
            return np.array([np.nan])
        
        return jax.lax.cond(np.isinf(ub_val), ub_infinite, ub_finite, None)

_vmapped_filter_lb = jax.vmap(_filter_lb, in_axes=(0, 0))

_vmapped_filter_ub = jax.vmap(_filter_ub, in_axes=(0, 0))

def _filter_lbs(g_values, lb):
    lb, g_values = np.atleast_1d(lb), np.atleast_1d(g_values)
    results = _vmapped_filter_lb(g_values, lb)
    results = results.flatten()
    mask = ~np.isnan(results)
    return np.where(mask, results, -1.)  # Substitute NaNs with -1 or another placeholder value

def _filter_ubs(g_values, ub):
    ub, g_values = np.atleast_1d(ub), np.atleast_1d(g_values)
    results = _vmapped_filter_ub(g_values, ub)
    results = results.flatten()
    mask = ~np.isnan(results)
    return np.where(mask, results, -1.)  # Substitute NaNs with -1 or another placeholder value


def _get_B(n_states, grid_pts, d_tau):
    '''
    This function returns the matrix B which is used to transcribe the dynamics constraints. B is defined as follows (from Betts 4.6.4): 
    B = -1/2*delta_tau[[I, I, 0, 0, ...][0, 0, I, I, 0, 0, ...]...[0, 0, ..., I, I]]
    where delta_tau is the non-dimensional time step and I is the identity matrix of size n_states
    '''
    I = np.eye(n_states)
    
    # Construct the full matrix B using vstack (vertical stacking)
    B = np.zeros((n_states * (grid_pts - 1), n_states * grid_pts))

    for i in range(grid_pts - 1):
        B = B.at[i*n_states:(i+1)*n_states, i*n_states:(i+1)*n_states].set(I)
        B = B.at[i*n_states:(i+1)*n_states, (i+1)*n_states:(i+2)*n_states].set(I)

    return -B * d_tau * 0.5

def _get_A(n_states, n_inputs, grid_pts):
    '''
    This function returns the matrix A which is used to transcribe the dynamics constraints. A is defined as follows (from Betts 4.6.4):
    A = [[0_col, -I, 0, I, ...][0, 0, 0, -I, 0, I, ...]...[0, 0, ..., I, 0]]
    '''
    I = np.eye(n_states)
    zero_column = np.zeros((n_states, 1))
    zero_block_1 = np.zeros((n_states, n_inputs))
    zero_block_2  = np.zeros((n_states, n_states+n_inputs))

    # Construct row block using hstack (horizontal stacking)
    row_block = np.hstack([zero_column, -I, zero_block_1, I, zero_block_1] + [zero_block_2] * (grid_pts - 2))
    
    # Construct the full matrix A using vstack (vertical stacking)
    A = np.vstack([np.roll(row_block, shift=i*(n_states + n_inputs), axis=1) for i in range(grid_pts-1)])
    
    return A

def _summed_hessians_x_lambda(fns):
    """
    Computes the Hessian matrix of a function, `g(x, lambdas)`, with respect to `x`, 
    where `g` is a linear combination of the functions in `fns` weighted by `lambdas` the lagrange multipliers.

    `g(x, lambdas)` is defined as the dot product of the evaluation of each function 
    in `fns` at `x` and the `lambdas` vector.

    Args:
        fns (list[callable]): A list of functions to be combined. Each function should
            accept a single argument and return a scalar or array value.

    Returns:
        callable: A JIT-compiled function that computes the Hessian of `g` with respect 
        to `x`. The returned function accepts two arguments: `x` and `lambdas`.

    Example:
        >>> fns = [lambda x: x**2, lambda x: x**3]
        >>> hessian_func = _summed_hessians_x_lambda(fns)
        >>> hessian_at_point = hessian_func(2.0, [1.0, 2.0])
        
    Note:
        This function uses JAX for differentiation and JIT compilation and takes advantage of the fact that the Hessian of a sum is the sum of the Hessians.

    """
    def g(x, lambdas):
        fnx = np.array([f(x) for f in fns]).squeeze(0)
        return np.dot(fnx, lambdas)
    return jax.jit(jax.hessian(g, argnums=0))


def _summed_hessians_x(fns):
    """
    Computes the Hessian matrix of a function, `g(x)`, with respect to `x`, 
    where `g` is the sum of the evaluations of each function in `fns` at `x`.

    Args:
        fns (list[callable]): A list of functions to be summed. Each function should
            accept a single argument and return a scalar or array value.

    Returns:
        callable: A JIT-compiled function that computes the Hessian of `g` with respect 
        to `x`. The returned function accepts a single argument, `x`.

    Example:
        >>> fns = [lambda x: x**2, lambda x: x**3]
        >>> hessian_func = _summed_hessians_x(fns)
        >>> hessian_at_point = hessian_func(2.0)
        
    Note:
        This function uses JAX for differentiation and JIT compilation and takes advantage of the fact that the Hessian of a sum is the sum of the Hessians.

    """
    def g(x):
        fnx = np.array([f(x) for f in fns]).squeeze(0)
        return np.sum(fnx, axis=0)
    return jax.jit(jax.hessian(g))
