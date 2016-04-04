import numpy as np
import scipy.optimize
import scipy.integrate

from aux import get_data_for_chi, get_data_for_cash

def do_fit(fun, model, method='L-BFGS-B'):
    x0 = np.array([param.value for param in model.params.values()])
    bnd = [(param.min, param.max) for param in model.params.values()]
    result = scipy.optimize.minimize(fun, x0, method=method, jac=True,
                                     bounds=bnd, constraints=model.constraints,
                                     options={'ftol': 2.22e-9, 'disp': True})
    if not result.success:
#        print(result)
        raise Exception('Fit failed: {}'.format(result.message))
    for param, value in zip(model.params.values(), result.x):
        param.value = value
    return result

def chi(obs_profile, model, method='L-BFGS-B',
        min_range=-np.inf, max_range=+np.inf):
    """Fit a profile using least squares statistics."""
    nbins, r, w, net, net_err = get_data_for_chi(obs_profile,
                                                 min_range, max_range)

    def calc_chi2(params):
#        print('Iterating... ', params)
        mod_profile = np.array(
            [scipy.integrate.quad(model.evaluate_with_params,
                                  x - width, x + width, params)[0] / 2 / width
             for x, width in zip(r, w)])
#        print('  => ', np.sum((net - mod_profile) ** 2 / net_err ** 2))
        #mod_profile = np.array([model.evaluate_with_params(x, params)
        #                        for x in r])

        #print(net)
        #print(mod_profile)
        chi2 = np.sum((net - mod_profile) ** 2 / net_err ** 2)
        jac = np.zeros(len(params))
        for i in range(len(params)):
            der = np.array(
                [scipy.integrate.quad(model.jacobian,
                            x - width, x + width, (params, i))[0] / 2 / width
                 for x, width in zip(r, w)])
            jac[i] = np.sum(-2*(net - mod_profile) / net_err**2 * der)
#        print('analytical jacobian: ', jac)
        return chi2, jac

    return do_fit(calc_chi2, model, method=method)

def calc_cash(raw_cts, mod_profile):
    cash = 2. * np.sum(mod_profile - raw_cts + raw_cts *
                       np.log(raw_cts/mod_profile))
    return cash

def calc_mod_profile(model, params, r, w, bkg, sb_to_counts_factor):
    mod_profile  = np.array(
        [scipy.integrate.quad(model.evaluate_with_params,
                              x - width, x + width, params)[0] / 2 / width
         for x, width in zip(r, w)])
    mod_profile = (mod_profile + bkg) * sb_to_counts_factor
    return mod_profile

def cash(obs_profile, model, method='L-BFGS-B',
         min_range=-np.inf, max_range=+np.inf):
    """Fit a profile using Cash statistics.

    This statistic is only valid if no background components are subtracted
    from the data. If (part of) the background is subtracted, use cstat
    statistics instead.
    """
    nbins, r, w, raw_cts, bkg, sb_to_counts_factor = get_data_for_cash(obs_profile,
                                                                       min_range, max_range)
    def get_cash_jac(params):
#        print('Iterating... ', params)
        mod_profile = calc_mod_profile(model, params, r, w, bkg, sb_to_counts_factor)
        # TODO: CHECK CASE FOR RAW_CTS == 0 (RAW_CTS IS ARRAY)
        cash = calc_cash(raw_cts, mod_profile)

#        print('==> ', cash)
        jac = np.zeros(len(params))
        for i in range(len(params)):
            der = np.array(
                [scipy.integrate.quad(model.jacobian,
                            x - width, x + width, (params, i))[0] / 2 / width
                 for x, width in zip(r, w)]) * sb_to_counts_factor
            jac[i] = 2. * np.sum((1. - raw_cts/mod_profile) * der)
#        print('analytical jacobian: ', jac)
        return cash, jac

    return do_fit(get_cash_jac, model, method=method)

#def cash(obs_profile, mod_profile):
#    nbins, r, src, npix, exp = get_data_for_cash(obs_profile)
#    counts = src * exp * npix
