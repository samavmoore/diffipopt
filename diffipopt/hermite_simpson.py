
import jax
import jax.numpy as np
jax.config.update("jax_enable_x64", True)

class HermiteSimpson():
    def __init__(self, Problem, Parameters):
        self.problem = Problem
        self.params = Parameters

    def objective(self, x):
        return self.problem.cost.fun(x, self.params.p)