import sys
import numpy as np
import inspect
from aux import call_model

class FitParameter:
    def __init__(self, model_parameter):
        self.name = model_parameter.name
        self.value = model_parameter.default_value
        self.frozen = False
        self.min = -np.inf
        self.max = +np.inf

class Model(object):
    def __init__(self, params):
        self.params = {x.name: FitParameter(x) for x in params}

    def set_parameter(self, name, value, frozen=None,
                      min_bound=None, max_bound=None):
        self.params[name].val = value
        if frozen is not None:
            self.params[name].frozen = bool(frozen)
        if min_bound is not None:
            self.params[name].min = float(min_bound)
        if max_bound is not None:
            self.params[name].max = float(max_bound)

    def thaw(self, name):
        if name in self.params.keys():
            self.params[name]['frozen'] = False
        else:
            raise NameError("Parameter %s does not exist." % name)

    def freeze(self, name):
        if name in self.params.keys():
            self.params[name].frozen = True
        else:
            raise NameError("Parameter %s does not exist." % name)

    def set_lower_bound(self, name, min_bound):
        if name in self.params.keys():
            self.params[name].min = min_bound

    def set_upper_bound(self, param, max):
        if param in self.params.keys():
            self.params[param].max = max

    def show_params(self):
        print(self.params)



    def __add__(self, other):
        return AdditiveModel(self, other)



def get_model_defaults(func):
    try:
        defaults = inspect.getcallargs(func, 'x')
        del defaults['x']
    except NameError:
        defaults = inspect.getcallargs(func)
    return defaults

def initialize_params(params, allowed_params):
    param_defs = {}
    for key, val in params.items():
        if key in allowed_params:
            param_defs[key] = {}
            param_defs[key]['value'] = val
            param_defs[key]['frozen'] = False
            param_defs[key]['min'] = -np.inf
            param_defs[key]['max'] = np.inf
    return param_defs

class Model(object):
    def __init__(self, params):
        getattr(sys.modules[__name__], func_name).__init__(self)
        self.func = call_model(self.func_name)
        params = get_model_defaults(self.func)
        self.params = initialize_params(params, self.allowed_parameters)

    def set_param(self, param, val, frozen=False, min=-np.inf, max=np.inf):
        if param in self.params.keys():
            self.params[param]['value'] = val
            self.params[param]['frozen'] = frozen
            self.params[param]['min'] = min
            self.params[param]['max'] = max

    def thaw(self, param):
        if param in self.params.keys():
            self.params[param]['frozen'] = False

    def freeze(self, param):
        if param in self.params.keys():
            self.params[param]['frozen'] = True

    def set_lower_bound(self, param, min):
        if param in self.params.keys():
            self.params[param]['min'] = min

    def set_upper_bound(self, param, max):
        if param in self.params.keys():
            self.params[param]['max'] = max

    def show_params(self):
        print(self.params)
