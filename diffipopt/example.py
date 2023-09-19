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

    #x, lam = solve(cartpole_params, problem)
    #vmapped_solve = jax.vmap(Partial(solve, prob=problem), in_axes=(0,))
    #params = np.array([cartpole_params, cartpole_params, cartpole_params, cartpole_params])
    #x, lam = vmapped_solve(params)

    
    #dx_dp, dlam_dp = jax.jacobian(solve)(cartpole_params, problem)

    n_states = 4
    n_inputs = 1
    del_params = params(m_c=0.0, m_p=0.2, l=0.0, dist=-0.2, f_max=0.0, tf=0.0)
    solve_  = Partial(solve, prob=problem)
    soln, jvp = jax.jvp(solve_, (cartpole_params,), (del_params,))
    x, lam = soln
    dx_dp, dlam_dp = jvp

    perturbed_params = cartpole_params._replace(m_c=cartpole_params.m_c+del_params.m_c,
                                                 m_p=cartpole_params.m_p+del_params.m_p, 
                                                 l=cartpole_params.l+del_params.l, 
                                                 dist=cartpole_params.dist+del_params.dist, 
                                                 f_max=cartpole_params.f_max+del_params.f_max, 
                                                 tf=cartpole_params.tf+del_params.tf)
    
    x_perturbed, lam_perturbed = solve(perturbed_params, problem)

    tf = x[0, 0]
    states_inputs = x[0, 1:].reshape((grid_pts, n_states + n_inputs))
    states_soln = states_inputs[:, :n_states]
    inputs_soln = states_inputs[:, n_states:]
    states_soln = states(*states_soln.T)
    inputs_soln = inputs(*inputs_soln.T)

    d_tf = dx_dp[0, 0]
    d_states_inputs = dx_dp[0, 1:].reshape((grid_pts, n_states + n_inputs))
    d_states = d_states_inputs[:, :n_states]
    d_inputs = d_states_inputs[:, n_states:]
    d_states = states(*d_states.T)
    d_inputs = inputs(*d_inputs.T)

    tf_perturbed = x_perturbed[0, 0]
    states_inputs_perturbed = x_perturbed[0, 1:].reshape((grid_pts, n_states + n_inputs))
    states_soln_perturbed = states_inputs_perturbed[:, :n_states]
    inputs_soln_perturbed = states_inputs_perturbed[:, n_states:]
    states_soln_perturbed = states(*states_soln_perturbed.T)
    inputs_soln_perturbed = inputs(*inputs_soln_perturbed.T)

    # plot x vs xdot and dx_dp dxdot_dp with quiver
    plt.figure(figsize=(8, 6))
    plt.plot(states_soln.x, states_soln.x_dot, 'b', label='Nominal')
    plt.plot(states_soln_perturbed.x, states_soln_perturbed.x_dot, 'r--', label='Perturbed - True')
    plt.plot(states_soln.x+d_states.x, states_soln.x_dot+d_states.x_dot, 'y', label='Perturbed - Approx')
    #plt.quiver(states_soln.x, states_soln.x_dot, d_states.x, d_states.x_dot, color='r', width=0.005)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\dot{x}$')
    plt.xlim([-.25, 1.5])
    plt.ylim([-1.5, 3])
    plt.legend()
    #plt.savefig('/Users/Sam/Documents/DiffTrajOpt/DiffIPOPT/x_xdot.pdf')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(states_soln.theta, states_soln.theta_dot, 'b', label='Nominal')
    plt.plot(states_soln_perturbed.theta, states_soln_perturbed.theta_dot, 'r--', label='Perturbed - True')
    plt.plot(states_soln.theta+d_states.theta, states_soln.theta_dot+d_states.theta_dot, 'y', label='Perturbed - Approx')
    #plt.quiver(states_soln.theta, states_soln.theta_dot, d_states.theta, d_states.theta_dot, color='r', width=0.005, )
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\dot{\theta}$')
    plt.xlim([-np.pi/2, 4])
    plt.ylim([-4, 9])
    plt.legend()
    #plt.savefig('/Users/Sam/Documents/DiffTrajOpt/DiffIPOPT/theta_thetadot.pdf')
    plt.show()

    # plot inputs
    time = np.linspace(0, tf, grid_pts)
    plt.figure(figsize=(8, 6))
    plt.plot(time, inputs_soln.f_x, 'b', label='Nominal')
    plt.plot(time, inputs_soln_perturbed.f_x, 'r--', label='Perturbed - True')
    plt.plot(time, inputs_soln.f_x+d_inputs.f_x, 'y', label='Perturbed - Approx')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$f_x$')
    plt.legend()
    plt.show()



    #print(type(out))
    #print(out.params)


