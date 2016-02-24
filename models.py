import numpy as np
import sys
from model import Model

class ModelParameter:
    def __init__(self, name, default_value, default_min=None, default_max=None):
        self.name = name
        self.default_value = default_value
        self.default_min = default_min
        self.default_max = default_max

class Constant(Model):
    def __init__(self):
        const = ModelParameter('const', 1e-4)
        super(Constant, self).__init__([const])

    def evaluate(self, x):
        return self.params['const'].value

class Beta(Model):
    def __init__(self):
        s0 = ModelParameter('s0', 1e-2, default_min = 1e-12, default_max = None)
        beta = ModelParameter('beta', 0.7, default_min = 1e-12, default_max = None)
        rc = ModelParameter('rc', 0.1, default_min = 1e-12, default_max = None)
        super(Beta, self).__init__([s0, beta, rc])

    def evaluate_with_params(self, x, params):
        return params[0] * (1. + (x/params[2])**2) ** (0.5 - 3*params[1])

    def jacobian(self, x, params, i):
        if i == 0:
            return (1. + (x/params[2])**2) ** (0.5 - 3*params[1])
        if i == 1:
            return -3 * params[0] * np.log(1. + (x/params[2])**2) * \
                   (1. + (x/params[2])**2) ** (0.5 - 3*params[1])
        if i == 2:
            return -2 * params[0] * x**2 * (0.5 - 3*params[1]) / params[2]**3 \
                   * (1. + (x/params[2])**2) ** (-0.5 - 3*params[1])
