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
"""
ControlProblem namedtuple representing a trajectory optimization problem.

Attributes:
        integration_type (str): either 'trapezoidal' or 'hermite-simpson'

        final_cost (Optional[Callable[[float, NamedTuple, NamedTuple, NamedTuple], Union[jax.numpy.ndarray, float]]]):
            Final cost function with signature f(tf, s, u, p) where:
            tf: Decision variable for the final time.
            s: States namedtuple at the final time.
            u: Inputs namedtuple at the final time.
            p: Parameters (user-defined namedtuples).
        
        path_cost (Optional[Callable[[NamedTuple, NamedTuple, NamedTuple], Union[jax.numpy.ndarray, float]]]):
            Path cost function with signature f(s, u, p) where:
            s: States namedtuple at the current time.
            u: Inputs namedtuple at the current time.
            p: Parameters (user-defined namedtuples).
        
        initial_state (BoundingBox):
            BoundingBox namedtuple representing the bounds on the initial states.
        
        path_state (BoundingBox):
            BoundingBox namedtuple representing the bounds on the states at each time step.
        
        final_state (BoundingBox):
            BoundingBox namedtuple representing the bounds on the final states.

        input (BoundingBox):
            BoundingBox namedtuple representing the bounds on the inputs at each time step.
        
        initial_g (Optional[Constraint]):
            Constraint function with signature g(s, u, p) <= 0 passed in as an instance of the 
                namedtuple Constraint with the field g and applied to the first time-step.
        
        path_g (Optional[Constraint]):
            Constraint function with signature g(s, u, p) <= 0 passed in as an instance of the 
                namedtuple Constraint with the field g and applied to each time-step between t0 and tf.
        
        final_g (Optional[Constraint]):
            Constraint function with signature g(tf, s, u, p) <= 0 passed in as an instance of the 
                namedtuple Constraint with the field g and applied to the final time-step.
        
        final_time (BoundingBox):
            BoundingBox namedtuple representing the bounds on the final time.
        
        state_tup (NamedTuple):
            NamedTuple representing the states of the dynamical system.
        
        input_tup (NamedTuple):
            NamedTuple representing the inputs at of the dynamical system.
        
        param_tup (NamedTuple):
            NamedTuple representing the parameters of the optimization problem and dynamical system that you wish to be differentiable.

        dynamics (Callable[[NamedTuple, NamedTuple, NamedTuple], NamedTuple]):
            Function with signature f(s, u, p) where:
            s: States namedtuple at the current time.
            u: Inputs namedtuple at the current time.
            p: Parameters (user-defined namedtuples).

                outputs: States namedtuple or jax.numpy.array of the time derivatives of the states.

        grid_pts (int):
            Number of grid points used to discretize the trajectory.
        
        options (Optional[Dict[str, Any]):
            Additional optional configurations or settings for ipopt.

"""

Constraint  = namedtuple('Constraint',['g'])
"""
Namedtuple representing a constraint function with the signature g <= 0.

Attributes:
    g (Callable,[...], Union[jax.numpy.ndarray, float]):
    The signature of the constraint function varies depending on the type of problem being solved and the constraint.
    For a conventional optimization problem, the signature is g(x, p) <= 0 where:
        x: Decision variables.
        p: Parameters (user-defined namedtuples).

    For a trajectory optimization problem, the signature is generally g(s, u, p) <= 0 where:
        s: States namedtuple.
        u: Inputs namedtuple.
        p: Parameters (user-defined namedtuples).

    unless the constraint is applied to the final time, in which case the signature is g(tf, s, u, p) <= 0 where:
        tf: Decision variable for the final time.
        s: States namedtuple at the final time.
        u: Inputs namedtuple at the final time.
        p: Parameters (user-defined namedtuples).

"""

BoundingBox = namedtuple('BoundingBox', ['lb', 'ub'])
"""
Namedtuple representing the bounds on a variable.

Attributes:
    lb (Callable[[NamedTuple], Union[jax.numpy.ndarray, float]]):
        Lower bound function with signature f(p) where:
        p: Parameters (user-defined namedtuples).
    
    ub (Callable[[NamedTuple], Union[jax.numpy.ndarray, float]]):
        Upper bound function with signature f(p) where:
        p: Parameters (user-defined namedtuples).
"""

def solve(params, prob):
    """
    Solves an optimization problem using IPOPT.

    Args:
        params (NamedTuple):
            Instance of  user-defined parameters namedtuple for the optimization problem.
        
        prob (Union[ControlProblem, Problem]):
            Instance of the ControlProblem or Problem namedtuple representing the optimization problem.
    
    Returns:
        x (jax.numpy.ndarray):
            The optimal decision variables or primals.

        lam (jax.numpy.ndarray):
            The optimal Lagrange multipliers or duals.
    """

    # initialize the problem instance
    print("Initializing problem")
    prblm = _initialize_problem(prob)
    params = jax.lax.stop_gradient(params)

    meta_dat_shape_dtype = jax.ShapeDtypeStruct(shape=(1, 3), dtype=np.int64)
    dat_shapes = jax.pure_callback(_define_meta_getter(prblm), meta_dat_shape_dtype, params, vectorized=False)
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
    #states = prob.state_tup(*states.T)
    #inputs = prob.input_tup(*inputs.T)
            
    return x, lam

def _define_meta_getter(prblm):
    """
    This function is used to define the meta_getter function for jax.pure_callback.

    Args:
        prblm (Union[ControlProblem, Problem]):
            Instance of the ControlProblem or Problem namedtuple representing the optimization problem.
        
    Returns:
        meta_getter (Callable[[NamedTuple], jax.numpy.ndarray]):
    """
    
    def _meta_getter(params):
        """
        This function is used to get the metadata for the optimization problem.
        """

        if hasattr(prblm, 'integration_type'):
            if prblm.integration_type == 'trapezoidal':
                prob_cls = Trapezoidal_Meta_Data(prblm, params)
            #elif prblm.integration_type == 'hermite-simpson':
            #    prob_cls = HermiteSimpson_Meta_Data(prblm, params)
            #else:
            #    raise ValueError('integration_type must be either "trapezoidal" or "hermite-simpson"')

        #elif isinstance(prblm, Problem):
        #    prob_cls = Standard(prblm, params)

        n_vars = prob_cls.n_vars
        n_constraints = prob_cls.n_constraints
        n_params = prob_cls.n_params
        
        return np.array([n_vars, n_constraints, n_params]).reshape((1, 3)).astype(np.int64)
    
    return _meta_getter


def _define_solve(prblm):
    """
    This function is used to define the solve function for jax.pure_callback.

    Args:
        prblm (Union[ControlProblem, Problem]):
            Instance of the ControlProblem or Problem namedtuple representing the optimization problem.
    
    Returns:
        _solve (Callable[[NamedTuple], jax.numpy.ndarray]):
    """
    
    def _solve(params):

        """
        This function is used to solve the optimization problem by initilizaing respective problem classes and calling the cyipopt solver.

        Args:
            params (NamedTuple):
                Instance of a user-defined parameters namedtuple for the optimization problem.

        Returns:
            x (jax.numpy.ndarray):
                The optimal decision variables or primals.

            lam (jax.numpy.ndarray):
                The optimal Lagrange multipliers or duals.
        """


        if hasattr(prblm, 'integration_type'):
            if prblm.integration_type == 'trapezoidal':
                prob_cls = Trapezoidal(prblm, params)
            elif prblm.integration_type == 'hermite-simpson':
                prob_cls = HermiteSimpson(prblm, params)
            else:
                raise ValueError('integration_type must be either "trapezoidal" or "hermite-simpson"')

        elif isinstance(prblm, Problem):
            prob_cls = Standard(prblm, params)

        xL, xU = prob_cls.xL, prob_cls.xU
        gL, gU = prob_cls.gL, prob_cls.gU
        x0 = initial_guess_traj_opt(prblm, params)
        nlp = cyipopt.Problem(n=len(x0), m=len(gL), problem_obj=prob_cls, lb=xL, ub=xU, cl=gL, cu=gU)
        nlp.add_option('tol', 1e-4)
        nlp.add_option('max_iter', int(1e5))
        nlp.add_option('nlp_scaling_method', 'gradient-based')

        print("Solving problem")
        x, info = nlp.solve(x0)
        
        lam_g = info['mult_g']
        lam_xL = info['mult_x_L']
        lam_xU = info['mult_x_U']
        lam = np.concatenate([lam_g, lam_xL, lam_xU])

        return x.astype(np.float64).reshape((-1, prob_cls.n_vars)), lam.astype(np.float64).reshape((-1, prob_cls.n_constraints))

    return _solve
        


solve = jax.custom_jvp(solve, nondiff_argnums=(1,))

def _define_KKT_solve(prblm):
    """
    This function is used to define the solve function for jax.pure_callback in custom jvp. Overall similar to solve but returns derivatives of the solution with respect to the parameters.
    This is seperated for comuptational efficiency. If the user only wants the solution, then the KKT system does not need to be solved.

    Args:
        prblm (Union[ControlProblem, Problem]):
            Instance of the ControlProblem or Problem namedtuple representing the optimization problem.
        
    Returns:
        _KKT_solve (Callable[[NamedTuple], jax.numpy.ndarray]):

    """

    def _KKT_solve(params):
        """
        This function is used to solve the optimization problem by initilizaing respective problem classes and calling the cyipopt solver. It also computes the derivatives of the solution with respect to the parameters.

        Args:
            params (NamedTuple):
                Instance of a user-defined parameters namedtuple for the optimization problem.
        
        Returns:
            x (jax.numpy.ndarray):
                The optimal decision variables or primals.

            lam (jax.numpy.ndarray):
                The optimal Lagrange multipliers or duals.

            dx_lam_dp (jax.numpy.ndarray):
                The derivatives of the decision variables and Lagrange multipliers with respect to the parameters.
        """

        if hasattr(prblm, 'integration_type'):
            if prblm.integration_type == 'trapezoidal':
                prob_cls = Trapezoidal(prblm, params)
            elif prblm.integration_type == 'hermite-simpson':
                prob_cls = HermiteSimpson(prblm, params)
            else:
                raise ValueError('integration_type must be either "trapezoidal" or "hermite-simpson"')

        elif isinstance(prblm, Problem):
            prob_cls = Standard(prblm, params)

        xL, xU = prob_cls.xL, prob_cls.xU
        gL, gU = prob_cls.gL, prob_cls.gU
        x0 = initial_guess_traj_opt(prblm, params)
        nlp = cyipopt.Problem(n=len(x0), m=len(gL), problem_obj=prob_cls, lb=xL, ub=xU, cl=gL, cu=gU)
        nlp.add_option('tol', 1e-4)
        nlp.add_option('max_iter', int(1e5))
        nlp.add_option('nlp_scaling_method', 'gradient-based')

        print("Solving problem")
        x, info = nlp.solve(x0)
        
        lam_g = info['mult_g']
        lam_xL = info['mult_x_L']
        lam_xU = info['mult_x_U']
        lam = np.concatenate([lam_g, lam_xL, lam_xU])

        dKKT_dp = _del_KKT_del_p(prob_cls, params, (x, lam_g, lam_xL, lam_xU))
        dKKT_dx_lambda = _del_KKT_del_x_lambda(prob_cls, params, (x, lam_g, lam_xL, lam_xU))
        dKKT_dx_lambda = dKKT_dx_lambda + np.eye(dKKT_dx_lambda.shape[0]) * 1e-10
        dx_lam_dp = -np.linalg.solve(dKKT_dx_lambda, dKKT_dp)

        x = x.astype(np.float64).reshape((-1, prob_cls.n_vars))
        lam = lam.astype(np.float64).reshape((-1, prob_cls.n_constraints))

        dx_lam_dp = dx_lam_dp.astype(np.float64).reshape((prob_cls.n_vars + prob_cls.n_constraints, prob_cls.n_params))

        return x, lam, dx_lam_dp
    
    return _KKT_solve


@solve.defjvp
def _solve_jvp(prblm, primals, tangents):
    """
    This function is used to define the jvp of solve. It is used to compute the derivatives of the solution with respect to the parameters.

    Args:
        prblm (Union[ControlProblem, Problem]):
            Instance of the ControlProblem or Problem namedtuple representing the optimization problem.
        
        primals (Tuple[jax.numpy.ndarray]):
            Tuple of the primal variables.
        
        tangents (Tuple[jax.numpy.ndarray]):
            Tuple of the derivatives of the primal variables with respect to the parameters.
    
    Returns:
        (Tuple[jax.numpy.ndarray]):
            Tuple of the primal variables and Lagrange multipliers.
        
        (Tuple[jax.numpy.ndarray]):
            Tuple of the derivatives of the primal variables and Lagrange multipliers with respect to the parameters.
    
    """


    (params, ) = primals
    (params_dot, ) = tangents

    # initialize the problem instance
    print("Initializing problem")
    prblm = _initialize_problem(prblm)
    params = jax.lax.stop_gradient(params)

    meta_dat_shape_dtype = jax.ShapeDtypeStruct(shape=(1, 3), dtype=np.int64)
    dat_shapes = jax.pure_callback(_define_meta_getter(prblm), meta_dat_shape_dtype, params, vectorized=False)
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

    tanget_out = dx_lam_dp @ np.array([*params_dot])

    tanget_out_x = tanget_out[:n_vars].reshape((1, n_vars))
    tanget_out_lam = tanget_out[n_vars:].reshape((1, n_constraints))

    return (x, lam), (tanget_out_x, tanget_out_lam)



def _del_KKT_del_p(prob_cls, params, solution):
    """
    This function is used to compute the derivative of the KKT system with respect to the parameters.

    Args:
        prob_cls (Trapezoidal, HermiteSimpson, Problem):
            Instance of the a problem class instance representing the optimization problem that is input to cyipopt.
        
        params (NamedTuple):
            Instance of a user-defined parameters namedtuple for the optimization problem.
        
        solution (Tuple[jax.numpy.ndarray]):
            Tuple of the primal variables and Lagrange multipliers.

    Returns:
        dKKT_dp (jax.numpy.ndarray):
            Derivative of the KKT system with respect to the parameters.
    """

    prob = prob_cls
    x, lam_g, lam_xL, lam_xU = solution
    lams = np.concatenate([lam_g, lam_xL, lam_xU])

    obj_hess_px = np.array([f(x, params) for f in prob.obj_hess_px]).squeeze(0).T
    con_hess_px_dot_lam = np.vstack([*prob.all_constraints_hess_px_lam(x, params, lams)]).T
    top_block = obj_hess_px + con_hess_px_dot_lam

    jac_g_p = [g(x, params) for g in prob.all_constraints_jac_p]

    jac_g_p = np.vstack([np.vstack([*jac]).T for jac in jac_g_p])

    bottom_block = jac_g_p * lams[:, np.newaxis]

    return np.vstack([top_block, bottom_block])

def _del_KKT_del_x_lambda(prob_cls, params, solution):
    """
    This function is used to compute the derivative of the KKT system with respect to the decision variables and Lagrange multipliers.

    Args:   
        prob_cls (Trapezoidal, HermiteSimpson, Problem):
            Instance of the a problem class instance representing the optimization problem that is input to cyipopt.
        
        params (NamedTuple):
            Instance of a user-defined parameters namedtuple for the optimization problem.

        solution (Tuple[jax.numpy.ndarray]):
            Tuple of the primal variables and Lagrange multipliers.

    Returns:
        dKKT_dx_lambda (jax.numpy.ndarray):
            Derivative of the KKT system with respect to the decision variables and Lagrange multipliers.
    """

    prob = prob_cls
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


def _check_control_problem(prob):
    '''
    This function is used to check if the fields/subfields of a ControlProblem namedtuple are valid.

    Args:
        prob (ControlProblem):
            Instance of the ControlProblem namedtuple representing the optimization problem.
        
    Raises:
        ValueError: If any of the fields/subfields of the ControlProblem namedtuple are invalid.
    
    '''

    for field in prob._fields:
        attr = getattr(prob, field)
        
        if field in ['final_cost', 'path_cost']:
            if attr is None:
                continue
            if not callable(attr):
                raise ValueError(f"{field} should be a callable function")
            if len(inspect.signature(attr).parameters) != 3:
                raise ValueError(f"{field} should accept exactly 3 arguments (states namedtuple, inputs namedtuple, parameters namedtuple)")
        
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
                else:
                    raise ValueError(f"{field} should only have 'lb' and 'ub' fields")
            
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
                else:
                    raise ValueError(f"{field} should only have 'g' field")

        elif field in ['state_tup', 'input_tup', 'param_tup']:
            if not issubclass(attr, tuple) or not hasattr(attr, '_fields'):
                raise ValueError(f"{field} should be of type namedtuple")

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

def _check_problem(prob):
    """
    This function is used to check if the fields/subfields of a Problem namedtuple are valid.

    Args:
        prob (Problem):
            Instance of the Problem namedtuple representing the optimization problem.
        
    Raises:
        ValueError: If any of the fields/subfields of the Problem namedtuple are invalid.
    """

    for field in prob._fields:
        attr = getattr(prob, field)

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

                else:
                    raise ValueError(f"{field} should only have 'g' field")

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
                else:
                    raise ValueError(f"{field} should only have 'lb' and 'ub' fields")

        elif field == 'options':
            if attr is None:
                continue
            if not isinstance(attr, dict):
                raise ValueError(f"{field} should be of type dict")


def _initialize_problem(prob):
    '''
    This function is used to initialize the problem instance aka perform error checking.
    '''

    # Check if the instance is either a ControlProblem or a Problem
    if not isinstance(prob, (ControlProblem, Problem)):
        raise TypeError('prob must be of type ControlProblem or Problem (namedtuples)')
    
    # Check if the instance is a ControlProblem and wrap functions with Partial and perform error-checking
    if isinstance(prob, ControlProblem):
        _check_control_problem(prob)
    else:
        _check_problem(prob)
    
    return prob





