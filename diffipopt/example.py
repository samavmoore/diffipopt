from collections import namedtuple
import jax
import jax.numpy as np
import numpy as onp
import cyipopt
from jax import core
from jax.tree_util import Partial
import inspect
jax.config.update("jax_enable_x64", True)
from api import ControlProblem, BoundingBox, solve
import matplotlib.pyplot as plt


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # define structs for states, inputs, and parameters
    states = namedtuple('States', ['x', 'x_dot', 'theta', 'theta_dot'])
    inputs = namedtuple('Inputs', ['f_x'])
    params = namedtuple('Parameters', ['m_c', 'm_p', 'l', 'dist', 'f_max', 'tf'])

    # Instantiate structs for parameters and problem
    cartpole_params = params(m_c=2.0, m_p=.5, l=0.5, dist=0.8, f_max=100., tf=2.0)

    # Define dynamics and cost functions
    @jax.jit
    def dynamics(s, u, p):
        sine_theta = np.sin(s.theta)
        cosine_theta = np.cos(s.theta)
        g = 9.81

        x_dot = s.x_dot
        denominator1 = p.m_c + p.m_p * sine_theta**2
        x_ddot = (u.f_x + p.m_p * sine_theta * (p.l * s.theta_dot**2 + g * cosine_theta)) / denominator1
        
        theta_dot = s.theta_dot
        denominator2 = p.l * (p.m_c + p.m_p * sine_theta**2)
        theta_ddot = (-u.f_x * cosine_theta - p.m_p * p.l * s.theta_dot**2 * cosine_theta * sine_theta - (p.m_c + p.m_p) * g * sine_theta) / denominator2

        return states(*np.array([x_dot, x_ddot, theta_dot, theta_ddot]))


    path_cost = lambda s, u, p: .1*u.f_x**2
    # Define constraints on the initial and final states
    ic = states(*np.zeros((4, 1)))
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

    grid_pts = 25

    problem = ControlProblem(
        integration_type='trapezoidal', 
        state_tup=states, 
        input_tup=inputs,
        param_tup=params,
        dynamics=dynamics, 
        path_cost=path_cost, 
        initial_state=ic_bounds, 
        final_state=fc_bounds,
        path_state=path_bounds, 
        final_time=tf_bound, 
        input=input_bounds, 
        grid_pts=grid_pts
    )

    tf, states, inputs = solve(cartpole_params, problem)
    #vmapped_solve = jax.vmap(Partial(solve, problem_instance=problem), in_axes=(0,))
    #params = np.array([cartpole_params, cartpole_params, cartpole_params, cartpole_params])
    #tf, states, inputs = vmapped_solve(params)


    t = np.linspace(0, tf, grid_pts)
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(t, states.x)
    plt.xlabel('Time (s)')
    plt.ylabel('Cart Position (m)')
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(t, states.x_dot)
    plt.xlabel('Time (s)')
    plt.ylabel('Cart Velocity (m/s)')
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.plot(t, states.theta)
    plt.xlabel('Time (s)')
    plt.ylabel('Pole Angle (rad)')
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(t, states.theta_dot)
    plt.xlabel('Time (s)')
    plt.ylabel('Pole Angular Velocity (rad/s)')
    plt.grid()

    plt.show()

    # plotting the input
    plt.figure()
    plt.plot(t, inputs.f_x)
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.grid()
    plt.show()


    #print(type(out))
    #print(out.params)


