
import jax
import jax.numpy as np
from jax.tree_util import Partial
import scipy.sparse as sparse
from collections import namedtuple
from api import BoundingBox
from common import _filter_lbs, _filter_ubs, _get_B, _get_A, bound_box_func, _summed_hessians_x_lambda, _summed_hessians_x
jax.config.update("jax_enable_x64", True)


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
        self.params = Problem.param_tup(*Parameters)
        self.n_states = len(Problem.state_tup._fields)
        self.n_inputs = len(Problem.input_tup._fields)
        self.states = Problem.state_tup
        self.inputs = Problem.input_tup
        self.grid_pts = Problem.grid_pts
        self.n_vars = Problem.grid_pts * (self.n_states + self.n_inputs) + 1
        self.d_tau = 1/(Problem.grid_pts - 1) # this is the non-dimensional time step

        self.xL, self.xU = self._get_bounds()
        self.A = _get_A(self.n_states, self.n_inputs, self.grid_pts)
        self.B = _get_B(self.n_states, self.grid_pts, self.d_tau)

        self._setup_functions()
        self.objectives_x_p = self._setup_objective()
        self.bound_box_x_p = self._generate_bound_box_funcs()
        self.constraints_x_p, self.g_bounds = self._stack_constraints()

        self.gL, self.gU = np.concatenate([b.lb for b in self.g_bounds]), np.concatenate([b.ub for b in self.g_bounds])
        self.constraints_x = [Partial(g, p=self.params) for g in self.constraints_x_p]
        self.constraints_x_p_jit = [jax.jit(g) for g in self.constraints_x_p]
        self.constraints_x_jit = [jax.jit(g) for g in self.constraints_x]
        self.constraints_jac_x = [jax.jit(jax.jacobian(g)) for g in self.constraints_x]
        self.constraints_jac_x_p = [jax.jit(jax.jacobian(g)) for g in self.constraints_x_p]
        
        self.constraints_hess_x_lamb = _summed_hessians_x_lambda(self.constraints_x)
        self.constraints_hess_x = _summed_hessians_x(self.constraints_x)
        
        self.objectives_x = [Partial(f, p=self.params) for f in self.objectives_x_p]
        self.objectives_x_p_jit = [jax.jit(f) for f in self.objectives_x_p]
        self.objectives_x_jit = [jax.jit(f) for f in self.objectives_x]
        self.obj_grad_x = [jax.jit(jax.grad(f)) for f in self.objectives_x]
        self.obj_grad_x_p = [jax.jit(jax.grad(f)) for f in self.objectives_x_p]
        self.obj_hess_x = [jax.jit(jax.hessian(f)) for f in self.objectives_x]

        self.bound_box_x = [Partial(g, p=self.params) for g in self.bound_box_x_p]

        self.j_struct = self._jacobianstructure()
        self.h_struct = self._hessianstructure()

    def objective(self, x):
        return np.sum(np.array([f(x) for f in self.objectives_x_jit]))
    
    def gradient(self, x):
        return np.array([f(x) for f in self.obj_grad_x])
    
    def constraints(self, x):
        return np.array([g(x) for g in self.constraints_x_jit])

    def jacobian(self, x):
        Jac = np.array([g(x) for g in self.constraints_jac_x]).squeeze(0)
        return Jac[self.j_struct.row, self.j_struct.col]
    
    def hessian(self, x, lagrange, obj_factor):
        H_obj = np.array([f(x) for f in self.obj_hess_x])
        H_constraints = self.constraints_hess_x_lamb(x, lagrange)
        Hess = np.array(obj_factor * H_obj + H_constraints)
        return Hess[self.h_struct.row, self.h_struct.col]
    
    def jacobianstructure(self):
        return (self.j_struct.row, self.j_struct.col)

    def hessianstructure(self):
        return (self.h_struct.row, self.h_struct.col)

    def _jacobianstructure(self):
        point = jax.random.normal(jax.random.PRNGKey(0), (self.n_vars, ))*10.
        jac = np.array([g(point) for g in self.constraints_jac_x]).squeeze(0)
        js = sparse.coo_matrix(jac != 0)
        return js
    
    def _hessianstructure(self):
        point = jax.random.normal(jax.random.PRNGKey(0), (self.n_vars,))*10.
        # Assuming the functions return full Hessians
        hess_obj = np.array([f(point) for f in self.obj_hess_x]).squeeze(0)
        hess_obj_structure = (hess_obj != 0).astype(int)

        # Get the non-zero structures of each constraint Hessian
        hess_constraints_structure = (self.constraints_hess_x(point) != 0).astype(int)

        # Combine the objective and constraints structures
        combined_structure = np.logical_or(hess_obj_structure, (hess_constraints_structure > 0))
        hs = sparse.coo_matrix(combined_structure)
        return hs

    def _setup_functions(self):
        
        def _get_jit_wrapped_function(function):
            if function is not None:
                return self._wrap_n_jit(function)
            return None

        functions = {
            "dynamics": self.problem.dynamics,
            "path_cost": self.problem.path_cost,
            "final_cost": self.problem.final_cost,
            "initial_g": getattr(self.problem.initial_g, "g", None),
            "path_g": getattr(self.problem.path_g, "g", None),
            "final_g": getattr(self.problem.final_g, "g", None)
        }

        for name, func in functions.items():
            setattr(self, name, _get_jit_wrapped_function(func))

    def _setup_objective(self):
        obj_list = []
        if self.problem.final_cost is not None:
            obj_list.append(Partial(_transcribed_final_cost, f=self.final_cost, n_states=self.n_states, n_inputs=self.n_inputs))

        if self.problem.path_cost is not None:
            obj_list.append(Partial(_transcribed_path_cost, f=self.path_cost, n_states=self.n_states, n_inputs=self.n_inputs, grid_pts=self.grid_pts))

        return obj_list
    
    def _stack_constraints(self):

        stacked_constraints = []
        stacked_bounds = []
        
        # add the dynamics constraints to the stack, wrap them in a partial to evaluate the function with the correct number of states, inputs, and grid points
        stacked_constraints.append(jax.jit(Partial(_transcribed_dynamics, f=self.dynamics, n_states=self.n_states, n_inputs=self.n_inputs, grid_pts=self.grid_pts, A=self.A, B=self.B)))
        dynamics_bounds = np.zeros((self.n_states * (self.grid_pts-1), ))
        stacked_bounds.append(BoundingBox(lb=dynamics_bounds, ub=dynamics_bounds))

        if self.problem.initial_g is not None:
            
            stacked_constraints.append(Partial(_transcribed_initial_g, g_0=self.initial_g, n_states=self.n_states, n_inputs=self.n_inputs))
            # probe the initial constraint function to get the upper and lower bound size
            initial_g_size = self.initial_g(np.zeros((self.n_states + self.n_inputs, )), self.params).shape[0]
            initial_g_ub = np.zeros((initial_g_size, ))
            initial_g_lb = np.zeros((initial_g_size, )) - np.inf
            stacked_bounds.append(BoundingBox(lb=initial_g_lb, ub=initial_g_ub))

        
        if self.problem.path_g is not None:
            
            stacked_constraints.append(Partial(_transcribed_path_g, g_path=self.path_g, n_states=self.n_states, n_inputs=self.n_inputs, grid_pts=self.grid_pts))
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
        @jax.jit
        def wrapped_f(x, p):
            states = self.states(*x[:self.n_states])
            inputs = self.inputs(*x[self.n_states:])
            return np.array(f(states, inputs, p)).T.squeeze()
        return wrapped_f

    def _generate_bound_box_funcs(self):
        """
        This function turns the bounding box constraints into a list of functions that can be evaluated with the parameters and the variables

        xL <= x <= xU -> xL -x <= 0 and  x - xU <= 0 while filtering out the constraints with infinite bounds or None bounds
        """
        # bound_box_func is a named tuple with the following fields:
        # name: name of the bounding box constraint
        # start: start index of the constraint
        # end: end index of the constraint
        # bound_func1: function to evaluate the bound (this is typically all thats needed for initial and final states and inputs)
        # bound_func2: this is used for path constraints states and inputs need to be bounded along the trajectory
        # filter_func: (_filter_lb or _filter_ub) this is used to filter out the constraints with infinite bounds or None bounds
        # Lower bound constraints
        final_time_lb = bound_box_func("final_time_lb", 0, 1, self.problem.final_time.lb, None, _filter_lbs)
        initial_state_lb = bound_box_func("initial_state_lb", 1, self.n_states+1, self.problem.initial_state.lb, None, _filter_lbs)
        initial_input_lb = bound_box_func("initial_input_lb", self.n_states+1, self.n_states+self.n_inputs+1, self.problem.input.lb, None, _filter_lbs)
        path_lb = bound_box_func("path_lb", self.n_states+self.n_inputs+1, self.n_vars-self.n_states-self.n_inputs, self.problem.path_state.lb, self.problem.input.lb, _filter_lbs)
        final_state_lb = bound_box_func("final_state_lb", self.n_vars-self.n_states-self.n_inputs, self.n_vars-self.n_inputs, self.problem.final_state.lb, None, _filter_lbs)
        final_input_lb = bound_box_func("final_input_lb", self.n_vars-self.n_inputs, self.n_vars, self.problem.input.lb, None, _filter_lbs)

        # Upper bound constraints
        final_time_ub = bound_box_func("final_time_ub", 0, 1, self.problem.final_time.ub, None, _filter_ubs)
        initial_state_ub = bound_box_func("initial_state_ub", 1, self.n_states+1, self.problem.initial_state.ub, None, _filter_ubs)
        initial_input_ub = bound_box_func("initial_input_ub", self.n_states+1, self.n_states+self.n_inputs+1, self.problem.input.ub, None, _filter_ubs)
        path_ub = bound_box_func("path_ub", self.n_states+self.n_inputs+1, self.n_vars-self.n_states-self.n_inputs, self.problem.path_state.ub, self.problem.input.ub, _filter_ubs)
        final_state_ub = bound_box_func("final_state_ub", self.n_vars-self.n_states-self.n_inputs, self.n_vars-self.n_inputs, self.problem.final_state.ub, None, _filter_ubs)
        final_input_ub = bound_box_func("final_input_ub", self.n_vars-self.n_inputs, self.n_vars, self.problem.input.ub, None, _filter_ubs)

        bound_box_functions = [
            final_time_lb, initial_state_lb, initial_input_lb, path_lb, final_state_lb, final_input_lb,
            final_time_ub, initial_state_ub, initial_input_ub, path_ub, final_state_ub, final_input_ub
        ]

        bounding_box_funcs = []

        for box_func in bound_box_functions:
            if box_func.name == "path_lb" or box_func.name == "path_ub":
                bounding_box_funcs.append(Partial(path_bounding_box_constraints, 
                                                start=box_func.start, 
                                                end=box_func.end, 
                                                bound_func1=box_func.bound_func1, 
                                                bound_func2=box_func.bound_func2, 
                                                filter_func=box_func.filter_func, 
                                                bounds_size=self.grid_pts))
            else:
                bounding_box_funcs.append(Partial(bounding_box_constraint, 
                                                start=box_func.start, 
                                                end=box_func.end, 
                                                bound_func=box_func.bound_func1, 
                                                filter_func=box_func.filter_func))

        return bounding_box_funcs

    def _get_bounds(self):
        '''
        This function returns the bounds for the optimization problem that are fed to IPOPT. This is not the function that is used to calculate the KKT conditions post-optimization.
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

        xL = xL.at[1:self.n_states+1].set(x_0_lb)
        xU = xU.at[1:self.n_states+1].set(x_0_ub)

        xL = xL.at[self.n_states + 1:self.n_states + self.n_inputs + 1].set(u_lb)
        xU = xU.at[self.n_states + 1:self.n_states + self.n_inputs + 1].set(u_ub)

        x_f_lb = np.array(self.problem.final_state.lb(self.params)).squeeze()
        x_f_ub = np.array(self.problem.final_state.ub(self.params)).squeeze()

        xL = xL.at[self.n_vars-self.n_inputs:self.n_vars].set(u_lb)
        xU = xU.at[self.n_vars-self.n_inputs:self.n_vars].set(u_ub)

        xL = xL.at[self.n_vars-(self.n_states + self.n_inputs):self.n_vars-self.n_inputs].set(x_f_lb)
        xU = xU.at[self.n_vars-(self.n_states + self.n_inputs):self.n_vars-self.n_inputs].set(x_f_ub)

        x_path_lb = np.array(self.problem.path_state.lb(self.params)).squeeze()
        x_path_ub = np.array(self.problem.path_state.ub(self.params)).squeeze()

        # Define the start and end indices for path states and path inputs
        path_slice_starts = np.arange(1, self.problem.grid_pts - 1) * (self.n_states + self.n_inputs) + 1
        path_state_ends = path_slice_starts + self.n_states
        path_input_ends = path_state_ends + self.n_inputs

        # Using JAX's dynamic functions to set slices for path states and inputs
        for start, end in zip(path_slice_starts, path_state_ends):
            xL = xL.at[start:end].set(x_path_lb)
            xU = xU.at[start:end].set(x_path_ub)

        for start, end in zip(path_state_ends, path_input_ends):
            xL = xL.at[start:end].set(u_lb)
            xU = xU.at[start:end].set(u_ub)

        return xL, xU


# defining these constraint functions in here and not using self.n_states, self.n_inputs etc. directly so that they are pure functions to keep jax happy
# honestly shouldnt be a problem though anyway because those attributes are set in the constructor and never changed but just to be safe and keep things pure
def _transcribed_dynamics(x, p, f, n_states, n_inputs, grid_pts, A, B):
    """
    Compute the transcribed dynamics by mapping the dynamics function `f` over a grid of points.
    As described in Betts, defects c(x) == 0 are:

    c(x) = A*x + tf*B*q(x) = 0
    where x is the vector of decision variables, and q(x) is the mapped dynamics function q(x) = [f(s1, u1), f(s2, u2), ...]

    A and B are transformation matrices that are defined as follows:

    A = [[0_col, -I, 0, I, ...][0_col, 0, 0, -I, 0, I, ...]...[0_col, 0, ..., I, 0]]

    B = -1/2*delta_tau[[I, I, 0, 0, ...][0, 0, I, I, 0, 0, ...]...[0, 0, ..., I, I]]

    The input vector `x` contains the final time as its first element, followed by interleaved 
    states and inputs (e.g., [s1, u1, s2, u2, ...]).

    Args:
        x (array_like): Input vector with the first element as the final time followed by 
            interleaved states and inputs.
        p (namedtuple): Additional parameters for the dynamics function `f`.
        f (callable): Dynamics function to be mapped over the grid points. It should accept
            the reshaped states and inputs and the parameter `p` as arguments.
        n_states (int): Number of state variables in the system.
        n_inputs (int): Number of input variables in the system.
        grid_pts (int): Number of grid points.
        A (array_like): Transformation matrix.
        B (array_like): Transformation matrix used with the mapped dynamics.

    Returns:
        np.ndarray: Transcribed dynamics after transforming with matrices `A` and `B`.

    Example:
        >>> def example_dynamics(x_new, p):
        ...     return x_new * p
        >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> p = params(p=np.array([2.5]))
        >>> A = np.array([[1, 1], [3, 4]])
        >>> B = np.array([[1, 2], [3, 4]])
        >>> result = _transcribed_dynamics(x, p, example_dynamics, 2, 1, 2, A, B)
        
    Note:
        This function uses JAX for vectorized mapping operations.

    """
    # extract the final time
    tf = x[0]
    # extract the states and inputs which are interleaved like [s1, u1, s2, u2, ...]
    # and shaping the states and inputs into a matrix like [[s1, s2, ...], [u1, u2, ...]]
    x_new = x[1:].reshape((grid_pts, n_states + n_inputs))
    # vmapping the dynamics function over the grid points aka the columns of the matrix
    mapped_dynamics = jax.vmap(f, in_axes=(0, None))(x_new, p)
    # reshaping the mapped dynamics into a vector
    mapped_dynamics = mapped_dynamics.reshape((n_states * grid_pts, ))
    return np.matmul(A, x) + tf*np.matmul(B, mapped_dynamics)


def _transcribed_initial_g(x, p, g_0, n_states, n_inputs):
    """
    Apply an inequality constraint of the form `g_0(x) <= 0` to the initial states and inputs.

     where x is the vector of decision variables with the first element as the final time followed by interleaved states and inputs (e.g., [s1, u1, s2, u2, ...]).
    `g_0` is applied to these initial values to evaluate the inequality constraint.

    Args:
        x (array_like): Input vector with the first element as the final time followed by
            interleaved states and inputs.
        p (namedtuple): Additional parameters for the constraint function `g_0`.
        g_0 (callable): Inequality constraint function to be applied on the initial conditions. 
            It should accept the initial states, inputs, and the parameter `p` as arguments and 
            return values that are expected to be less than or equal to zero.
        n_states (int): Number of state variables in the system.
        n_inputs (int): Number of input variables in the system.

    Returns:
        np.ndarray: Evaluation of the inequality constraint `g_0` on the initial states and inputs. 
        The returned values are expected to be less than or equal to zero for the constraint to be satisfied.

    Example:
        >>> def inequality_constraint(x, p):
        ...     return x - p
        >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> p = params(p=np.array([2.5]))
        >>> result = transcribed_initial_g(x, p, inequality_constraint, 2, 1)

    """
    # grabbing the intial states and inputs
    x = x[1:n_states+n_inputs+1]

    return g_0(x, p)

def _transcribed_path_g(x, p, g_path, n_states, n_inputs, grid_pts):
    """
    Apply an inequality constraint of the form `g_path(x) <= 0` to the path states and inputs.

    where x is the vector of decision variables with the first element as the final time followed by interleaved states and inputs (e.g., [s1, u1, s2, u2, ...]).
    `g_path` is applied to these path values to evaluate the inequality constraint.


    Args:
        x (array_like): Input vector with the first element as the final time followed by
            interleaved states and inputs.
        p (namedtuple): Additional parameters for the constraint function `g_path`.
        g_path (callable): Inequality constraint function to be applied on the path states and inputs.
            It should accept the path states, inputs, and the parameter `p` as arguments and
            return values that are expected to be less than or equal to zero.
        n_states (int): Number of state variables in the system.
        n_inputs (int): Number of input variables in the system.
        grid_pts (int): Number of grid points.

    Returns:
        np.ndarray: Evaluation of the inequality constraint `g_path` on the path states and inputs.
        The returned values are expected to be less than or equal to zero for the constraint to be satisfied.

    Example:
        >>> def inequality_constraint(x, p):
        ...     return x - p
        >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> p = params(p=np.array([2.5]))
        >>> result = transcribed_path_g(x, p, inequality_constraint, 2, 1, 3)

    """
    # grabbing the path states and inputs
    x = x[n_states+n_inputs+1:-n_states-n_inputs]
    # reshaping the states and inputs into a matrix like [[s1, s2, ...], [u1, u2, ...]]
    x = x.reshape(( grid_pts-2, n_states + n_inputs))

    # evaluating the path constraint function of the form g(x, p) <= 0
    return jax.vmap(g_path, in_axes=(0, None))(x, p)

def transcribed_final_g(x, p, g_f, n_states, n_inputs):
        """
        Apply an inequality constraint of the form `g_f(x) <= 0` to the final states and inputs.

        where x is the vector of decision variables with the first element as the final time followed by interleaved states and inputs (e.g., [s1, u1, s2, u2, ...]).
        `g_f` is applied to these final values to evaluate the inequality constraint.

        Args:
            x (array_like): Input vector with the first element as the final time followed by
                interleaved states and inputs.
            p (namedtuple): Additional parameters for the constraint function `g_f`.
            g_f (callable): Inequality constraint function to be applied on the final conditions.
                It should accept the final states, inputs, and the parameter `p` as arguments and
                return values that are expected to be less than or equal to zero.
            n_states (int): Number of state variables in the system.
            n_inputs (int): Number of input variables in the system.

        Returns:
            np.ndarray: Evaluation of the inequality constraint `g_f` on the final states and inputs.
            The returned values are expected to be less than or equal to zero for the constraint to be satisfied.

        Example:
            >>> def inequality_constraint(x, p):
            ...     return x - p
            >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            >>> p = params(p=np.array([2.5]))
            >>> result = transcribed_final_g(x, p, inequality_constraint, 2, 1)
        
        """
        # grabbing the final states and inputs
        x = x[-n_states-n_inputs:]
        # evaluating the final constraint function and subtracting the upper bound and lower bound so that the constraint is of the form g(x, p) <= 0
        return g_f(x, p)

def path_bounding_box_constraints(x, p, start, end, bound_func1, bound_func2, filter_func, bounds_size):
    x = x[start: end]
    return apply_path_bounding_box(x, p, bound_func1, bound_func2, filter_func, bounds_size)

def apply_path_bounding_box(x, p, bounds_func1, bounds_func2, filter_func, bounds_size):
    state_bounds = np.array(bounds_func1(p)).squeeze()
    input_bounds = np.array(bounds_func2(p)).squeeze()
    state_bounds, input_bounds = np.atleast_1d(state_bounds), np.atleast_1d(input_bounds)
    bound_vals = np.concatenate([state_bounds, input_bounds])
    bound_vals = np.tile(bound_vals, bounds_size-2)
    return filter_func(x, bound_vals)

def apply_bounding_box(x, p, bound_func, filter_func):
    bound_vals = np.array(bound_func(p)).squeeze()
    return filter_func(x, bound_vals)  

def bounding_box_constraint(x, p, start, end, bound_func, filter_func):
    x = x[start:end] 
    return apply_bounding_box(x, p, bound_func, filter_func)

def _transcribed_final_cost(x, p, f, n_states, n_inputs):
    x = x[-n_states-n_inputs:]
    return f(x, p)

def _transcribed_path_cost(x, p, f, n_states, n_inputs, grid_pts):
    dt = x[0]/(grid_pts-1)
    xk = x[1:-2*(n_states+n_inputs)]
    xk = xk.reshape((grid_pts-2, n_states + n_inputs))

    xk1 = x[1+n_states+n_inputs:-n_states-n_inputs]
    xk1 = xk1.reshape((grid_pts-2, n_states + n_inputs))
    mapped = .5*dt*(jax.vmap(f, in_axes=(0, None))(xk, p) + jax.vmap(f, in_axes=(0, None))(xk1, p))
    return sum(mapped)