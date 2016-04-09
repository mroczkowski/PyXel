import numpy as np

def cstat(measured_src_cts, measured_bkg_cts, t_src, t_bkg,
          updated_model, radius):
    """
    C-statistic implementation. [1][2]

    This algorithm requires source and background counts, as well as the
    factors that transform counts to rates.

    .. [1] Cash, W. (1979), "Parameter estimation in astronomy through
           application of the likelihood ratio", ApJ, 228, p. 939-947
    .. [2] Wachter, K., Leach, R., Kellogg, E. (1979), "Parameter estimation
           in X-ray astronomy using maximum likelihood", ApJ, 230, p. 274-287
    """
    










def cash(obs_profile, model, method='L-BFGS-B',
         min_range=-np.inf, max_range=+np.inf):
    """Fit a profile using Cash statistics (Cash 1979).

    This statistic is only valid if no background components are subtracted
    from the data. If (part of) the background is subtracted, use `cstat'
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

### Implement W-stat separately, because it's messier. Needs bkg counts and
### some correction for the exposure time.
