import numpy as np
import sys
from model import Model

class ModelParameter:
    def __init__(self, name, default_value):
        self.name = name
        self.default_value = default_value

class Constant:
    def __init__(self):
        self.func_name = 'const'
        self.allowed_parameters = set(['constant'])

class Beta(Model):
    def __init__(self):
        s0 = ModelParameter('s0', 1e-2)
        beta = ModelParameter('beta', 0.7)
        rc = ModelParameter('rc', 0.1)
        super(Beta, self).__init__([s0, beta, rc])

    def evaluate(self, x):
        return self.params['s0'].value * (1. + (x/self.params['rc'].value)**2) \
               ** (0.5 - 3.*self.params['beta'].value)

mod = Beta()
mod.set_parameter('beta', 0.5/3)
mod.show_params()
