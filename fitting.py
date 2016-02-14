import numpy as np
import scipy.optimize
import scipy.integrate

from aux import get_data_for_chi

def do_fit(fun, model):
    x0 = np.array([param.value for param in model.params.values()])
    result = scipy.optimize.minimize(fun, x0, method='SLSQP', jac=True, options={'ftol': 2.22e-9})
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

        chi2 = np.sum((net - mod_profile) ** 2 / net_err ** 2)
        jac = np.zeros(len(params))
        for i in range(len(params)):
            der = np.array(
                [scipy.integrate.quad(model.jacobian,
                            x - width, x + width, (params, i))[0] / 2 / width
                 for x, width in zip(r, w)])
            jac[i] = np.sum(-2*(net - mod_profile) / net_err**2 * der)
        print('analytical jacobian: ', jac)
        return chi2, jac

    return do_fit(calc_chi2, model)

#def cash(obs_profile, mod_profile):
#    nbins, r, src, npix, exp = get_data_for_cash(obs_profile)
#    counts = src * exp * npix
