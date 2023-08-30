from collections import namedtuple
import jax
import jax.numpy as np
import numpy as onp
import cyipopt
from jax import core
from jax.tree_util import Partial
from functools import partial
import inspect
from typing import Callable, Any, Optional, NamedTuple, Tuple, List, Dict
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
        'final_time', 'state_tup', 'input_tup', 'dynamics', 'grid_pts', 'options']

ControlProblem = namedtuple("ControlProblem", control_fields, defaults=(None for i in range(len(control_fields))))

Constraint  = namedtuple('Constraint',['g'])
"""
Attributes:
    g (Callable[[NamedTuple, NamedTuple], Union[jax.numpy.ndarray, NamedTuple, List, float]]]):
        Constraint function with signature g(x, p) <= 0 where:
        x: Decision variables.
        p: Parameters (user-defined namedtuples).

Example:
    -10 <= x[0] + x[1] <= 1
    
   >>> def g(x, p):
   >>>      return x[0] + x[1] - 1, -x[0] - x[1] - 10
"""
BoundingBox = namedtuple('BoundingBox', ['lb', 'ub'])
class Trapezoidal():
    def __init__(self, Problem, Parameters):
        '''
        This class transcribes a trajectory optimization problem using the (seperated) trapezoidal collocation outlined in

        Betts, John T. Practical methods for optimal control and estimation using nonlinear programming. Society for Industrial and Applied Mathematics, 2010.

        Chapter 4.6.4 (pg. 139) Right-Hand-Side Sparsity (Trapezoidal Collocation)

        Notation:
        x = [tf, s1, u1, s2, u2, ...] are the decision variables where s is the state and u is the input
        xL and xU are the lower and upper bounds on the decision variables
        g(x, p) <= 0 are the constraints
        f(x, p) is the objective function

        '''
        self.problem = Problem
        self.params = Parameters
        self.n_states = len(Problem.state_tup._fields)
        self.n_inputs = len(Problem.input_tup._fields)
        self.states = Problem.state_tup
        self.inputs = Problem.input_tup
        self.grid_pts = Problem.grid_pts
        self.n_vars = Problem.grid_pts * (self.n_states + self.n_inputs) + 1
        self.d_tau = 1/(Problem.grid_pts - 1) # this is the non-dimensional time step

        self.xL, self.xU = self._get_bounds()
        self.A = self._get_A()
        self.B = self._get_B()
        self._setup_functions()

        self.constraints_x_p, self.g_bounds = self._stack_constraints()
        self.g_lb, self.g_ub = np.concatenate([b.lb for b in self.g_bounds]), np.concatenate([b.ub for b in self.g_bounds])
        self.constraints_x = [Partial(g, p=self.params) for g in self.constraints_x_p]
        self.constraints_x_p_jit = [jax.jit(g) for g in self.constraints_x_p]
        self.constraints_x_jit = [jax.jit(g) for g in self.constraints_x]

        self.constraint_jac_x = [jax.jit(jax.jacobian(g)) for g in self.constraints_x]
        self.constraint_jac_x_p = [jax.jit(jax.jacobian(g)) for g in self.constraints_x_p]


        self.objectives_x_p = self._setup_objective()
        self.objectives_x = [Partial(f, p=self.params) for f in self.objectives_x_p]
        self.objectives_x_p_jit = [jax.jit(f) for f in self.objectives_x_p]
        self.objectives_x_jit = [jax.jit(f) for f in self.objectives_x]
        self.obj_grad_x = [jax.jit(jax.grad(f)) for f in self.objectives_x]
        self.obj_grad_x_p = [jax.jit(jax.grad(f)) for f in self.objectives_x_p]




        test_x = np.arange(self.n_vars)
        

        #test1 = [f(x=test_x, p=self.params) for f in self.stacked_constraints_x_p]
        
        #test2 = [f(x=test_x) for f in self.stack_constraints_x]

        test3 = [f(x=test_x, p=self.params) for f in self.objectives_x_p]

    def objective(self, x):
        return np.sum([f(x) for f in self.objectives_x])
    
    def _setup_functions(self):
        self.dynamics = self._wrap_n_jit(self.problem.dynamics)
        self.path_cost = self._wrap_n_jit(self.problem.path_cost) if self.problem.path_cost is not None else None
        self.final_cost = self._wrap_n_jit(self.problem.final_cost) if self.problem.final_cost is not None else None
        self.initial_g = self._wrap_n_jit(self.problem.initial_g.g) if self.problem.initial_g is not None else None
        self.path_g = self._wrap_n_jit(self.problem.path_g.g) if self.problem.path_g is not None else None
        self.final_g = self._wrap_n_jit(self.problem.final_g.g) if self.problem.final_g is not None else None

    def _setup_objective(self):
        obj_list = []
        if self.problem.final_cost is not None:
            def _transcribed_final_cost(x, p, f, n_states, n_inputs):
                x = x[-n_states-n_inputs:]
                return f(x, p)
            obj_list.append(Partial(_transcribed_final_cost, f=self.final_cost, n_states=self.n_states, n_inputs=self.n_inputs))

        if self.problem.path_cost is not None:
            def _transcribed_path_cost(x, p, f, n_states, n_inputs, grid_pts):
                x = x[1:-n_states-n_inputs]
                x = x.reshape((grid_pts-1, n_states + n_inputs))
                mapped = jax.vmap(f, in_axes=(0, None))(x, p)
                return sum(mapped)
            obj_list.append(Partial(_transcribed_path_cost, f=self.path_cost, n_states=self.n_states, n_inputs=self.n_inputs, grid_pts=self.grid_pts))

        return obj_list
    
    def _stack_constraints(self):

        stacked_constraints = []
        stacked_bounds = []

        # defining these constraint functions in here and not using self.n_states, self.n_inputs etc. directly so that they are pure functions to keep jax happy
        # honestly shouldnt be a problem though anyway because those attributes are set in the constructor and never changed but just to be safe and keep things pure
        def transcribed_dynamics(x, p, f, n_states, n_inputs, grid_pts, A, B):
            # extract the final time
            tf = x[0]
            # extract the states and inputs which are interleaved like [s1, u1, s2, u2, ...]
            # and shaping the states and inputs into a matrix like [[s1, s2, ...], [u1, u2, ...]]
            x_new = x[1:].reshape((grid_pts, n_states + n_inputs))
            # vmapping the dynamics function over the grid points aka the columns of the matrix
            mapped_dynamics = jax.vmap(f, in_axes=(0, None))(x_new, p)
            # reshaping the mapped dynamics into a vector
            mapped_dynamics = mapped_dynamics.reshape((n_states * grid_pts, ))
            outs = np.matmul(A, x) + tf*np.matmul(B, mapped_dynamics)
            return outs
        
        def transcribed_initial_g(x, p, g_0, n_states, n_inputs):
            # grabbing the intial states and inputs
            x = x[1:n_states+n_inputs+1]

            return g_0(x, p)
        
        def transcribed_path_g(x, p, g_path, n_states, n_inputs, grid_pts):
            # grabbing the path states and inputs
            x = x[n_states+n_inputs+1:-n_states-n_inputs]
            # reshaping the states and inputs into a matrix like [[s1, s2, ...], [u1, u2, ...]]
            x = x.reshape(( grid_pts-2, n_states + n_inputs))

            # evaluating the path constraint function of the form g(x, p) <= 0
            return jax.vmap(g_path, in_axes=(0, None))(x, p)
        
        def transcribed_final_g(x, p, g_f, n_states, n_inputs):
                # grabbing the final states and inputs
                x = x[-n_states-n_inputs:]
                # evaluating the final constraint function and subtracting the upper bound and lower bound so that the constraint is of the form g(x, p) <= 0
                return g_f(x, p)
        

        # add the dynamics constraints to the stack, wrap them in a partial to evaluate the function with the correct number of states, inputs, and grid points
        stacked_constraints.append(Partial(transcribed_dynamics, f=self.dynamics, n_states=self.n_states, n_inputs=self.n_inputs, grid_pts=self.grid_pts, A=self.A, B=self.B))
        dynamics_bounds = np.zeros((self.n_states * (self.grid_pts-1), ))
        stacked_bounds.append(BoundingBox(lb=dynamics_bounds, ub=dynamics_bounds))

        if self.problem.initial_g is not None:
            
            stacked_constraints.append(Partial(transcribed_initial_g, g_0=self.initial_g, n_states=self.n_states, n_inputs=self.n_inputs))
            # probe the initial constraint function to get the upper and lower bound size
            initial_g_size = self.initial_g(np.zeros((self.n_states + self.n_inputs, )), self.params).shape[0]
            initial_g_ub = np.zeros((initial_g_size, ))
            initial_g_lb = np.zeros((initial_g_size, )) - np.inf
            stacked_bounds.append(BoundingBox(lb=initial_g_lb, ub=initial_g_ub))

        
        if self.problem.path_g is not None:
            
            stacked_constraints.append(Partial(transcribed_path_g, g_path=self.path_g, n_states=self.n_states, n_inputs=self.n_inputs, grid_pts=self.grid_pts))
            # probe the path constraint function to get the upper and lower bound size
            path_g_size = self.path_g(np.zeros((self.n_states + self.n_inputs, )), self.params).shape[0]
            path_g_size = path_g_size * (self.grid_pts - 2)
            path_g_ub = np.zeros((path_g_size, ))
            path_g_lb = np.zeros((path_g_size, )) - np.inf
            stacked_bounds.append(BoundingBox(lb=path_g_lb, ub=path_g_ub))

        if self.problem.final_g is not None:
            
            stacked_constraints.append(Partial(transcribed_final_g, g_f=self.final_g, n_states=self.n_states, n_inputs=self.n_inputs))
            # probe the final constraint function to get the upper and lower bound size
            final_g_size = self.final_g(np.zeros((self.n_states + self.n_inputs, )), self.params).shape[0]
            final_g_ub = np.zeros((final_g_size, ))
            final_g_lb = np.zeros((final_g_size, )) - np.inf
            stacked_bounds.append(BoundingBox(lb=final_g_lb, ub=final_g_ub))

            stacked_bounds = BoundingBox(*np.concatenate(stacked_bounds, axis=0))

        return stacked_constraints, stacked_bounds
    
    def _wrap_n_jit(self, f):
        #@jax.jit
        def wrapped_f(x, p):
            states = self.states(*x[:self.n_states])
            inputs = self.inputs(*x[self.n_states:])
            return np.array(f(states, inputs, p)).T.squeeze()
        return wrapped_f
    
    def _get_B(self):
        '''
        This function returns the matrix B which is used to transcribe the dynamics constraints. B is defined as follows (from Betts 4.6.4): 
        B = -1/2*delta_tau[[I, I, 0, 0, ...][0, 0, I, I, 0, 0, ...]...[0, 0, ..., I, I]]
        where delta_tau is the non-dimensional time step and I is the identity matrix of size n_states
        '''
        for i in range(self.grid_pts-1):
            row_block = np.zeros((self.n_states, self.n_states*self.grid_pts))
            row_block = row_block.at[: , i*(2 * self.n_states): i*(2 * self.n_states) + self.n_states].set(np.eye(self.n_states))
            row_block = row_block.at[: , i*(2 * self.n_states) + self.n_states: (i + 1)*(2 * self.n_states)].set(np.eye(self.n_states))
            if i == 0:
                B = row_block
            else:
                B = np.concatenate((B, row_block), axis=0)
        return -B * self.d_tau * 0.5
    
    def _get_A(self):
        '''
        This function returns the matrix A which is used to transcribe the dynamics constraints. A is defined as follows (from Betts 4.6.4):
        A = [[0, -I, 0, I, ...][0, 0, 0, -I, 0, I, ...]...[0, 0, ..., I, 0]]
        '''
        for i in range(self.grid_pts-1):
            row_block = np.zeros((self.n_states, self.n_vars))
            row_block = row_block.at[:, i*(self.n_states + self.n_inputs) + 1: i*(self.n_states + self.n_inputs) + self.n_states + 1].set(-np.eye(self.n_states))
            row_block = row_block.at[:, (i + 1)*(self.n_states + self.n_inputs) + 1: (i + 1)*(self.n_states + self.n_inputs) + self.n_states + 1].set(np.eye(self.n_states))
            if i == 0:
                A = row_block
            else:
                A = np.concatenate((A, row_block), axis=0)
        return A

    def extract_values(self, x, start, end):
        return x[start:end]

    def _generate_bound_box_funcs(self):
        """
        This function turns the bounding box constraints into a list of functions that can be evaluated with the parameters and the variables

        xL <= x <= xU -> xL -x <= 0 and  x - xU <= 0 while filtering out the constraints with infinite bounds or None bounds
        """
        constraint_definitions = [
        # (name, starting index, ending index, bound_func1, bound_func2, filter_func)
        ("final_time_lb", 0, 1, self.problem.final_time.lb, None, _filter_lb),
        ("initial_state_lb", 1 , self.n_states+1, self.problem.initial_state.lb, None, _filter_lb),
        ("initial_input_lb", self.n_states+1, self.n_states+self.n_inputs+1, self.problem.input.lb, None, _filter_lb),
        ("path_lb", self.n_states+self.n_inputs+1, self.n_vars-self.n_states-self.n_inputs, self.problem.path_state.lb, self.problem.input.lb, _filter_lb),
        ("final_state_lb", self.n_vars-self.n_states-self.n_inputs, self.n_vars-self.n_inputs, self.problem.final_state.lb, None, _filter_lb),
        ("final_input_lb", self.n_vars-self.n_inputs, self.n_vars, self.problem.input.lb, None, _filter_lb),
        ("final_time_ub", 0, 1, self.problem.final_time.ub, None, _filter_ub),
        ("initial_state_ub", 1 , self.n_states+1, self.problem.initial_state.ub, None, _filter_ub),
        ("initial_input_ub", self.n_states+1, self.n_states+self.n_inputs+1, self.problem.input.ub, None, _filter_ub),
        ("path_ub", self.n_states+self.n_inputs+1, self.n_vars-self.n_states-self.n_inputs, self.problem.path_state.ub, self.problem.input.ub, _filter_ub),
        ("final_state_ub", self.n_vars-self.n_states-self.n_inputs, self.n_vars-self.n_inputs, self.problem.final_state.ub, None, _filter_ub),
        ("final_input_ub", self.n_vars-self.n_inputs, self.n_vars, self.problem.input.ub, None, _filter_ub)
        ]

        bounding_box_funcs = []

        def path_bounding_box_constraints(x, p, start, end, bound_func1, bound_func2, filter_func, bounds_size):
            x = self.extract_values(x, start, end)
            return apply_path_bounding_box(x, p, bound_func1, bound_func2, filter_func)
        
        def apply_path_bounding_box(x, p, bounds_func1, bounds_func2, filter_func, bounds_size):
            state_bounds = np.array(bounds_func1(p)).squeeze()
            input_bounds = np.array(bounds_func2(p)).squeeze()
            bound_vals = np.concatenate([state_bounds, input_bounds])
            bound_vals = bound_vals.repeat(bounds_size-2)
            return filter_func(x, bound_vals)
        
        def apply_bounding_box(x, p, bound_func, filter_func):
            bound_vals = np.array(bound_func(p)).squeeze()
            return filter_func(x, bound_vals)  

        def bounding_box_constraint(x, p, start, end, bound_func, filter_func):
            x = self.extract_values(x, start, end)
            return apply_bounding_box(x, p, bound_func, filter_func)          

        for name, start, end, bound_func1, bound_func2, filter_func in constraint_definitions:
            if name == "path_lb" or name == "path_ub":
                bounding_box_funcs.append(Partial(path_bounding_box_constraints, start=start, end=end, bound_func1=bound_func1, bound_func2=bound_func2, filter_func=filter_func, bounds_size=self.grid_pts))
            else:
                bounding_box_funcs.append(Partial(bounding_box_constraint, start=start, end=end, bound_func=bound_func1, filter_func=filter_func))

        return bounding_box_funcs


    def _get_bounds(self):
        '''
        This function returns the bounds for the optimization problem that are fed to IPOPT. This is not the function that is used to calculate the KKT conditions.
        The bounds are defined as follows (from Betts 4.6.4):
        '''
        xL = np.zeros(self.n_vars, dtype=np.float64)
        xU = np.zeros(self.n_vars, dtype=np.float64)

        xL = xL.at[0].set(self.problem.final_time.lb(self.params))
        xU = xU.at[0].set(self.problem.final_time.ub(self.params))

        x_0_lb = np.array(self.problem.initial_state.lb(self.params)).squeeze()
        x_0_ub = np.array(self.problem.initial_state.ub(self.params)).squeeze()
        u_lb = np.array(self.problem.input.lb(self.params)).squeeze()
        u_ub = np.array(self.problem.input.ub(self.params)).squeeze()

        xL = _set_slice(xL, 1, self.n_states + 1, x_0_lb)
        xU = _set_slice(xU, 1, self.n_states + 1, x_0_ub)

        xL = _set_slice(xL, self.n_states + 1, self.n_states + self.n_inputs + 1, u_lb)
        xU = _set_slice(xU, self.n_states + 1, self.n_states + self.n_inputs + 1, u_ub)

        x_f_lb = np.array(self.problem.final_state.lb(self.params)).squeeze()
        x_f_ub = np.array(self.problem.final_state.ub(self.params)).squeeze()

        xL = _set_slice(xL, self.n_vars-self.n_inputs, self.n_vars, u_lb)
        xU = _set_slice(xU, self.n_vars-self.n_inputs, self.n_vars, u_ub)

        xL = _set_slice(xL, self.n_vars-(self.n_states + self.n_inputs), self.n_vars-self.n_inputs, x_f_lb)
        xU = _set_slice(xU, self.n_vars-(self.n_states +self.n_inputs), self.n_vars-self.n_inputs, x_f_ub)

        x_path_lb = np.array(self.problem.path_state.lb(self.params)).squeeze()
        x_path_ub = np.array(self.problem.path_state.ub(self.params)).squeeze()

        # Define the start and end indices for path states and path inputs
        path_slice_starts = np.arange(1, self.problem.grid_pts - 1) * (self.n_states + self.n_inputs) + 1
        path_state_ends = path_slice_starts + self.n_states
        path_input_ends = path_state_ends + self.n_inputs

        # Using JAX's dynamic functions to set slices for path states and inputs
        for start, end in zip(path_slice_starts, path_state_ends):
            xL = _set_slice(xL, start, end, x_path_lb)
            xU = _set_slice(xU, start, end, x_path_ub)

        for start, end in zip(path_state_ends, path_input_ends):
            xL = _set_slice(xL, start, end, u_lb)
            xU = _set_slice(xU, start, end, u_ub)

        return xL, xU

class HermiteSimpson():
    def __init__(self, Problem, Parameters):
        self.problem = Problem
        self.params = Parameters

    def objective(self, x):
        return self.problem.cost.fun(x, self.params.p)
    
class Standard():
    def __init__(self, Problem, Parameters):
        self.problem = Problem
        self.params = Parameters

    def objective(self, x):
        return self.problem.cost.fun(x, self.params.p)
    
def _set_slice(array, start, end, value):
    """Helper function to update a slice of a JAX array"""
    slice_size = int(end - start)
    slice_indices = (int(start), )
    current_slice = jax.lax.dynamic_slice(array, slice_indices, (slice_size, ))
    updated_slice = current_slice.at[:slice_size].set(value)
    return jax.lax.dynamic_update_slice(array, updated_slice, slice_indices)


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

_vmapped_filter_lb = jax.vmap(_filter_lb, in_axes=(0, None))

_vmapped_filter_ub = jax.vmap(_filter_ub, in_axes=(0, None))

def _filter_lbs(g_values, lb):
    results = _vmapped_filter_lb(g_values, lb)
    results = results.flatten()
    mask = ~np.isnan(results)
    return np.where(mask, results, -1.)  # Substitute NaNs with -1 or another placeholder value

def _filter_ubs(g_values, ub):
    results = _vmapped_filter_ub(g_values, ub)
    results = results.flatten()
    mask = ~np.isnan(results)
    return np.where(mask, results, -1.)  # Substitute NaNs with -1 or another placeholder value


