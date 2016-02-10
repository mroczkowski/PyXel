import numpy as np
import scipy.optimize
import scipy.integrate

from aux import get_data_for_chi

"""Define the objective functions for the available statistics.

This functions will be minimized by the fitting routine. Available
statistics are chi-square and
"""
def do_fit(fun, model):
    x0 = np.array([param.value for param in model.params.values()])
    result = scipy.optimize.minimize(fun, x0, method='Nelder-Mead')
    if not result.success:
        print(result)
        raise Exception('Fit failed: {}'.format(result.message))
    for param, value in zip(model.params.values(), result.x):
        param.value = value
    #model.cov = result.hess_inv
    return result

def chi(obs_profile, model, minrange=-np.inf, maxrange=+np.inf):
    """Fit a profile using least squares statistics."""
    nbins, r, w, net, net_err = get_data_for_chi(obs_profile, minrange, maxrange)

    def calc_chi2(params):
        print('Iterating... ', params)
        mod_profile = np.array(
            [scipy.integrate.quad(model.evaluate_with_params,
                                  x - width, x + width, params)[0] / 2 / width
             for x, width in zip(r, w)])
        print('  => ', np.sum((net - mod_profile) ** 2 / net_err ** 2))
        #mod_profile = np.array([model.evaluate_with_params(x, params)
        #                        for x in r])
        return np.sum((net - mod_profile) ** 2 / net_err ** 2)

    return do_fit(calc_chi2, model)

#def cash(obs_profile, mod_profile):
#    nbins, r, src, npix, exp = get_data_for_cash(obs_profile)
#    counts = src * exp * npix
