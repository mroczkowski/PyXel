import numpy as np
import sys
from model import Model

class ModelParameter:
    def __init__(self, name, default_value):
        self.name = name
        self.default_value = default_value

class Constant(Model):
    def __init__(self):
        const = ModelParameter('const', 1e-4)
        super(Constant, self).__init__([const])

    def evaluate(self, x):
        return self.params['const'].value

class Beta(Model):
    def __init__(self):
        s0 = ModelParameter('s0', 1e-2)
        beta = ModelParameter('beta', 0.7)
        rc = ModelParameter('rc', 0.1)
        super(Beta, self).__init__([s0, beta, rc])

    def evaluate_with_params(self, x, params):
        return params[0] * (1. + (x/params[2])**2) ** (0.5 - 3*params[1])
