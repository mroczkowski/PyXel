import sys
import numpy as np
import inspect
import collections

from fitting import chi, cash, calc_cash, calc_mod_profile
from aux import call_model, get_data_for_cash

import emcee
import corner
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator

class FitParameter:
    def __init__(self, model_parameter):
        self.name = model_parameter.name
        self.value = model_parameter.default_value
        self.frozen = False
        self.min = model_parameter.default_min
        self.max = model_parameter.default_max

    def __repr__(self):
        return 'value={}, frozen={}, min={}, max={}'.format(self.value, self.frozen, self.min, self.max)

class Model(object):
    def __init__(self, params):
        self.params = collections.OrderedDict([(x.name, FitParameter(x)) for x in params])
        self.constraints = ()

    def evaluate_with_params(self, x, params):
        raise NotImplementedError()

    def evaluate(self, x):
        param_values = [param.value for param in self.params.values()]
        print("param_values", param_values)
        print("x = ", x)
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

    def lnprior(self, params):
        prior = 0
        for param_value, param in zip(params, self.params.values()):
            if param.min is not None:
                if param_value < param.min:
                    prior = -np.inf
                    break
            if param.max is not None:
                if param_value > param.max:
                    prior = -np.inf
                    break
        return prior

    def lnprob(self, params, r, w, raw_cts, bkg, sb_to_counts_factor):
        lnp = self.lnprior(params)
        if not np.isfinite(lnp):
            return -np.inf
        mod_profile = calc_mod_profile(self, params, r, w, bkg, sb_to_counts_factor)
        lnc = calc_cash(raw_cts, mod_profile)
        return lnp - lnc

    def fit(self, profile, statistics='cash', method='L-BFGS-B',
            min_range=-np.inf, max_range=np.inf):
        if statistics == 'chi':
            return chi(profile, self, method=method,
                       min_range=min_range, max_range=max_range)
        elif statistics == 'cash':
            _, r, w, raw_cts, bkg, sb_to_counts_factor = get_data_for_cash(profile,
                                                                           min_range,
                                                                           max_range)
            result = cash(profile, self, method=method,
                          min_range=min_range, max_range=max_range)

            """ndim, nwalkers = len(self.params), 100
            pos = [result["x"] + 1e-4 * np.random.randn(ndim) * result["x"]
                   for i in range(nwalkers)]
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob,
                            args=(r, w, raw_cts, bkg, sb_to_counts_factor))
            sampler.run_mcmc(pos, 500)
            pl.clf()
            fig, axes = pl.subplots(ndim, 1, sharex=True, figsize=(8, 9))
            for i, param in enumerate(self.params):
                axes[i].plot(sampler.chain[:, :, i].T, color="k", alpha=0.4)
                axes[i].yaxis.set_major_locator(MaxNLocator(5))
                axes[i].axhline(result["x"][i], color="#888888", lw=2)
                axes[i].set_ylabel(param)
            pl.show()
            samples = sampler.chain[:, 100:, :].reshape((-1, ndim))
            fig = corner.corner(samples, labels=list(self.params.keys()))
            fig.savefig("triangle.png")
            val_with_errs = [(v[1], v[2]-v[1], v[1]-v[0])
                             for v in zip(*np.percentile(samples, [5, 50, 95],
                                                axis=0))]
            print(val_with_errs)
            print('optimize: params={}, nll={}'.format(result['x'], result['fun']))
            for walker in range(nwalkers):
                params = [sampler.chain[walker, -1, 0], sampler.chain[walker, -1, 1],
                          sampler.chain[walker, -1, 2], sampler.chain[walker, -1, 3]]
                mod_profile = calc_mod_profile(self, params, r, w, bkg, sb_to_counts_factor)
                nll = calc_cash(raw_cts, mod_profile)
                print('walker {}: params={}, nll={}'.format(walker, params, nll))"""
            return result
        else:
            raise Exception('Statistics {} does not exist'.format(statistics))
