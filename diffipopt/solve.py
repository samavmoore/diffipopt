from collections import namedtuple
import jax
import jax.numpy as np
import numpy as onp
import cyipopt
from jax import core
from jax.tree_util import Partial
import inspect
import time
jax.config.update("jax_enable_x64", True)
from base_classes import Problem, ControlProblem, Constraint, BoundingBox, Trapezoidal, HermiteSimpson, Standard

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

    return x.astype(np.float64)

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