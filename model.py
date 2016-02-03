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
        self.params[name].value = value
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

    def set_upper_bound(self, name, max_bound):
        if name in self.params.keys():
            self.params[name].max = max_bound

    def show_params(self):
        print()
        print("MODEL SUMMARY:")
        print()
        print("Parameter".rjust(10), "Value".rjust(10), "Frozen?".rjust(10),
              "Min Bound".rjust(12), "Max Bound".rjust(12))
        print("-"*60)
        for name in self.params.keys():
            print(self.params[name].name.rjust(10),
                  repr(round(self.params[name].value, 3)).rjust(10),
                  repr(self.params[name].frozen).rjust(10),
                  repr(round(self.params[name].min, 3)).rjust(12),
                  repr(round(self.params[name].max, 3)).rjust(12))
        print()
