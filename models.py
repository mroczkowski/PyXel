import numpy as np
import sys
from model import Model
import scipy.integrate

class ModelParameter:
    def __init__(self, name, default_value, default_min=None, default_max=None):
        self.name = name
        self.default_value = default_value
        self.default_min = default_min
        self.default_max = default_max

class Constant(Model):
    def __init__(self):
        const = ModelParameter('const', 1e-4, default_min = 1e-12, default_max = None)
        super(Constant, self).__init__([const])

    def evaluate_with_params(self, x, params):
        if isinstance(x, list):
            print("hello")
            return params * len(x)
        else:
            return params[0]

    def jacobian(self, x, params, i):
        if i == 0:
            return 1

class Beta(Model):
    def __init__(self):
        s0 = ModelParameter('s0', 1e-2, default_min = 1e-12, default_max = None)
        beta = ModelParameter('beta', 0.7, default_min = 1e-12, default_max = None)
        rc = ModelParameter('rc', 0.1, default_min = 1e-12, default_max = None)
        const = ModelParameter('const', 1e-4, default_min = 1e-12, default_max = None)
        super(Beta, self).__init__([s0, beta, rc, const])

    def evaluate_with_params(self, x, params):
        return params[0] * (1. + (x/params[2])**2) ** (0.5 - 3*params[1]) \
               + params[3]

    def jacobian(self, x, params, i):
        if i == 0:
            return (1. + (x/params[2])**2) ** (0.5 - 3*params[1])
        if i == 1:
            return -3 * params[0] * np.log(1. + (x/params[2])**2) * \
                   (1. + (x/params[2])**2) ** (0.5 - 3*params[1])
        if i == 2:
            return -2 * params[0] * x**2 * (0.5 - 3*params[1]) / params[2]**3 \
                   * (1. + (x/params[2])**2) ** (-0.5 - 3*params[1])
        if i == 3:
            return 1.0

class BrokenPow(Model):
    def __init__(self):
        ind1 = ModelParameter('ind1', 0.0, default_min = None, default_max = None)
        ind2 = ModelParameter('ind2', 0.0, default_min = None, default_max = None)
        norm = ModelParameter('norm', 1e-3, default_min = 1e-12, default_max = None)
        r_cut = ModelParameter('r_cut', 1.0, default_min = 1e-12, default_max = None)
        c = ModelParameter('c', 1.0, default_min = 0.0, default_max = 4.0)
        const = ModelParameter('const', 1e-4, default_min = 1e-12, default_max = None)
        super(BrokenPow, self).__init__([ind1, ind2, norm, r_cut, c, const])

    def evaluate_with_params(self, x, params):
        norm_after_jump = params[2] / params[4]**2
        sx = x
        sx[x < params[3]] = scipy.integrate.quad(params[2] * (x/params[3])**(2*params[0]),
                                             1e-4, np.sqrt(params[3]**2 - x**2)) + \
                        scipy.integrate.quad(norm_after_jump * (x/params[3])**(2*params[1]),
                                             np.sqrt(params[3]**2 - x**2), 1e4)
        sx[x >= params[3]] = scipy.integrate.quad(norm_after_jump * (x/params[3])**(2*params[1]),
                                              1e-4, 1e4)
        return sx

    def jacobian(self, x, params, i):
        """This is messy...""" 
        pass
