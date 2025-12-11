# diffipopt
A differentiable JAX wrapper around IPOPT for direct collocation.

This code provides an interface for performing differentiable trajectory optimization using IPOPT, with bindings through cyipopt and automatic differentiation via JAX. Although the project has not been under active development for quite a while, it may still be useful to researchers interested in differentiable optimal control or parameter-sensitivity analysis/hardware-software co-design.

My original motivation was to explore how small perturbations in a system’s physical parameters influence the controllability, stability, or maneuverability of a closed-loop controller around a nominal trajectory obtained via trajectory optimization. The code includes an example (example.py) demonstrating differentiation through the open-loop solution for the cart-pole swing up problem wrt the physical params like mass and length. Only open-loop differentiation and trapezoidal collocation is implemented at present.
