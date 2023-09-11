import jax
import jax.numpy as np
from collections import namedtuple
from jax.tree_util import Partial
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
        parameters_instance = problem_instance.param_tup(*parameters_instance)

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

def _summed_hessians_xx_lambda(fns):
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
    def g(x, p, lambdas):
        fnx = np.array([f(x, p) for f in fns]).squeeze(0)
        return np.dot(fnx, lambdas)
    return jax.jit(jax.hessian(g, argnums=0))

def _summed_hessians_px_lambda(fns):
    """
    Computes the mixed partial derivatives of a function, `g(x, p, lambdas)`, with respect to `x` and `p`, 
    where `g` is a linear combination of the functions in `fns` weighted by `lambdas` the Lagrange multipliers.

    `g(x, p, lambdas)` is defined as the dot product of the evaluation of each function 
    in `fns` at `(x, p)` and the `lambdas` vector.

    Args:
        fns (list[callable]): A list of functions to be combined. Each function should
            accept two arguments (x, p) and return a scalar or array value.

    Returns:
        callable: A JIT-compiled function that computes the mixed partial derivatives of `g` with respect 
        to `x` and `p`. The returned function accepts three arguments: `x`, `p`, and `lambdas`.

    Example:
        >>> fns = [lambda x, p: x**2 + p, lambda x, p: x**3 - p]
        >>> jacobian_func = _summed_hessians_x_lambda(fns)
        >>> jacobian_at_point = jacobian_func(2.0, 1.0, np.array([1.0, 2.0]))
    """
    def g(x, p, lambdas):
        fn_xp = np.array([f(x, p) for f in fns]).squeeze(0)
        return np.dot(fn_xp, lambdas)

    return jax.jit(jax.jacobian(jax.grad(g, argnums=0), argnums=1))


def _summed_hessians_xx(fns):
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
    def g(x, p):
        fnx = np.array([f(x, p) for f in fns]).squeeze(0)
        return np.sum(fnx, axis=0)
    return jax.jit(jax.hessian(g, argnums=0))
