import numpy as np
from datetime import datetime

def cstat_derivatives(measured_raw_cts, updated_model, measured_bkg_cts,
                      t_raw, t_bkg, x):
    """
    Calculates the derivatives of the C-statistic.
    """
    model_derivs = updated_model.fit_deriv(x, *updated_model.parameters)

def cstat(measured_raw_cts, updated_model, measured_bkg_cts, t_raw, t_bkg, x):
    """
    C-statistic implementation. [1][2]

    This algorithm requires total and background counts, as well as the
    factors that transform counts to rates.

    .. [1] Cash, W. (1979), "Parameter estimation in astronomy through
           application of the likelihood ratio", ApJ, 228, p. 939-947
    .. [2] Wachter, K., Leach, R., Kellogg, E. (1979), "Parameter estimation
           in X-ray astronomy using maximum likelihood", ApJ, 230, p. 274-287
    """
    print("cstat eval - start time: ", str(datetime.now()))
    model_vals = updated_model(x)
    print(updated_model.parameters)
    d = np.sqrt(((t_raw + t_bkg) * model_vals - measured_raw_cts -
                measured_bkg_cts)**2 + 4. * (t_raw + t_bkg) *
                measured_bkg_cts * model_vals)
    f = (measured_raw_cts + measured_bkg_cts - (t_raw + t_bkg) * model_vals
         + d) / (2. * (t_raw + t_bkg))

    nbins = len(model_vals)
    cash = 0.
    for i in range(nbins):
        if measured_raw_cts[i] == 0 and measured_bkg_cts[i] > 0:
            cash += t_raw[i] * model_vals[i] - measured_bkg_cts[i] * \
                    np.log(t_bkg[i] / (t_raw[i] + t_bkg[i]))
        elif measured_bkg_cts[i] == 0 and measured_raw_cts[i] > 0:
            if model_vals[i] < measured_raw_cts[i] / (t_raw[i] + t_bkg[i]):
                cash += -t_bkg[i] * model_vals[i] - measured_raw_cts[i] * \
                        np.log(t_raw[i] / (t_raw[i] + t_bkg[i]))
            else:
                cash += t_raw[i] * model_vals[i] + measured_raw_cts[i] * \
                        (np.log(measured_raw_cts[i]) - np.log(t_raw[i] *
                        model_vals[i]) - 1)
        else:
            cash += np.sum(t_raw[i] * model_vals[i] + (t_raw[i] + t_bkg[i]) * f[i] -
                           measured_raw_cts[i] * np.log(t_raw[i] * (model_vals[i] + f[i]))
                           - measured_bkg_cts[i] * np.log(t_bkg[i] * f[i]) -
                           measured_raw_cts[i] * (1 - np.log(measured_raw_cts[i])) -
                           measured_bkg_cts[i] * (1 - np.log(measured_bkg_cts[i])))
    print('cash ==>', 2.*cash)
    print("cstat eval - end time: ", str(datetime.now()))
    return 2. * cash
