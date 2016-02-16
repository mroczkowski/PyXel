import sys
import numpy as np
import inspect
import collections

from fitting import chi
from aux import call_model

class FitParameter:
    def __init__(self, model_parameter):
        self.name = model_parameter.name
        self.value = model_parameter.default_value
        self.frozen = False
        self.min = model_parameter.default_min
        self.max = model_parameter.default_max

class Model(object):
    def __init__(self, params):
        self.params = collections.OrderedDict([(x.name, FitParameter(x)) for x in params])
        self.constraints = ()

    def evaluate_with_params(self, x, params):
        raise NotImplementedError()

    def evaluate(self, x):
        param_values = [param.value for param in self.params.values()]
        return self.evaluate_with_params(x, param_values)

    def set_parameter(self, name, value, frozen=False,
                      min_bound=None, max_bound=None):
        self.params[name].value = value
        if frozen:
            self.params[name].min = value
            self.params[name].max = value
        else:
            is_frozen = self.params[name].min == self.params[name].max
            if min_bound is not None or is_frozen:
                if min_bound == -np.inf:
                    self.params[name].min = None
                else:
                    self.params[name].min = min_bound

            if max_bound is not None or is_frozen:
                if max_bound == +np.inf:
                    self.params[name].max = None
                else:
                    self.params[name].max = max_bound

    def min_fitrange(self, x, y, yerr, min_range=None):
        if min_range is not None and np.min(x) < min_range:
            x_limited = x[x >= min_range]
            y_limited = y[x >= min_range]
            yerr_limited = yerr[x >= min_range]
            return x_limited, y_limited, yerr_limited
        else:
            return x, y, yerr

    def max_fitrange(self, x, y, yerr, max_range=None):
        if max_range is not None and np.max(x) > max_range:
            x_limited = x[x <= max_range]
            y_limited = y[x <= max_range]
            yerr_limited = yerr[x <= max_range]
            return x_limited, y_limited, yerr_limited
        else:
            return x, y, yerr

    def set_lower_bound(self, name, min_bound):
        if name in self.params.keys():
            self.params[name].min = min_bound

    def set_upper_bound(self, name, max_bound):
        if name in self.params.keys():
            self.params[name].max = max_bound


    class NamedParameters:
        def __init__(self, param_names, param_values):
            for name, value in zip(param_names, param_values):
                setattr(self, name, value)

    def set_constraints(self, constraints):
        def make_func(x, user_func):
            obj = Model.NamedParameters(self.params, x)
            return user_func(obj)
        self.constraints = [{'type': constraint['type'],
                             'fun': lambda x: make_func(x, constraint['fun'])}
                             for constraint in constraints]

    def show_params(self):
        print()
        print("MODEL SUMMARY:")
        print()
        print("Parameter".rjust(10), "Value".rjust(10), "Frozen?".rjust(10),
              "Min Bound".rjust(12), "Max Bound".rjust(12))
        print("-"*60)
        for name in self.params.keys():
            print(self.params[name].name.rjust(10),
                  str('%.3e' % self.params[name].value).rjust(10),
                  repr(self.params[name].frozen).rjust(10))
#                  str('%.3e' % self.params[name].min).rjust(12),
#                  str('%.3e' % self.params[name].max).rjust(12))
        print()

    def fit(self, profile, statistics='cash'):
        if statistics == 'chi':
            return chi(profile, self)
        else:
            raise Exception('Statistics {} does not exist'.format(statistics))
