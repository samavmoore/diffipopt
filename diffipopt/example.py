from collections import namedtuple
import jax
import jax.numpy as np
import numpy as onp
import cyipopt
from jax import core
from jax.tree_util import Partial
import inspect
jax.config.update("jax_enable_x64", True)
from solve import solve
from base_classes import Problem, ControlProblem, Constraint, BoundingBox, Trapezoidal, HermiteSimpson, Standard



if __name__ == '__main__':
    # define structs for states, inputs, and parameters
    states = namedtuple('States', ['x', 'x_dot', 'theta', 'theta_dot'])
    inputs = namedtuple('Inputs', ['f_x'])
    params = namedtuple('Parameters', ['m_c', 'm_p', 'l', 'dist', 'f_max', 'tf'])

    # Instantiate structs for parameters and problem
    cartpole_params = params(m_c=2.0, m_p=.5, l=0.5, dist=0.8, f_max=100., tf=2.0)
    problem = ControlProblem(integration_type='trapezoidal', state_tup=states, input_tup=inputs)

    # Define dynamics and cost functions
    #@jax.jit
    def dynamics(s, u, p):
        sin = np.sin(s.theta)
        cos = np.cos(s.theta)
        g=9.81

        x_dot = s.x_dot
        x_ddot = 1/(p.m_c + p.m_p*(sin)**2)*(u.f_x + p.m_p*sin*(p.l*s.theta_dot**2 + g*cos))
        theta_dot = s.theta_dot
        theta_ddot = 1/(p.l*(p.m_c + p.m_p*(sin)**2))*(-u.f_x*cos - p.m_p*p.l*s.theta_dot**2*cos*sin - (p.m_c + p.m_p)*g*sin)

        return states(*np.array([x_dot, x_ddot, theta_dot, theta_ddot]))

    path_cost = lambda s, u, p: u.f_x**2
    problem = problem._replace(dynamics=dynamics, path_cost=path_cost)

    # Define constraints on the initial and final states
    ic_zeros = np.zeros((4, 1))
    ic_zeros = ic_zeros.at[2,0].set(None)
    ic = states(*ic_zeros)
    ic_bounds = BoundingBox(lb=lambda p: ic, ub=lambda p: ic)

    fc = lambda p: states(x=p.dist, x_dot=0.0, theta=np.pi, theta_dot=0.0)
    fc_bounds = BoundingBox(lb=fc, ub=fc)

    # Define path bounding box constraints
    path_lb = lambda p: states(x=-2*p.dist, x_dot=-np.inf, theta=-2*np.pi, theta_dot=-np.inf)
    path_ub = lambda p: states(x=2*p.dist, x_dot=np.inf, theta=2*np.pi, theta_dot=np.inf)
    path_bounds = BoundingBox(lb=path_lb, ub=path_ub)

    input_lb = lambda p: inputs(f_x=-p.f_max)
    input_ub = lambda p: inputs(f_x=p.f_max)
    input_bounds = BoundingBox(lb=input_lb, ub=input_ub)

    # constrain the final time
    tf = lambda p: p.tf
    tf_bound = BoundingBox(lb=tf, ub=tf)
    problem = problem._replace(initial_state=ic_bounds, final_state=fc_bounds, path_state=path_bounds, final_time=tf_bound, input=input_bounds, grid_pts=100)

    # add path constraints
    problem = problem._replace(path_g=Constraint(g=lambda s, u, p: [s.x_dot**2 + s.theta_dot**2 - 4.0, s.theta - np.pi, u.f_x**2 - p.f_max**2]))
    problem = problem._replace(param_tup=params)

    out = solve(problem, cartpole_params)
    #print(type(out))
    #print(out.params)



