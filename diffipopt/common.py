import jax
import jax.numpy as np
from collections import namedtuple
jax.config.update("jax_enable_x64", True)

bound_box_func = namedtuple(
            "bound_box_func", ["name", "start", "end", "bound_func1", "bound_func2", "filter_func"]
        )

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