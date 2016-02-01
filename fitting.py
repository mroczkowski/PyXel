import numpy as np
from scipy.optimize import curve_fit
import models

class FitModel:
    """Fit a certain model to the data.

    Available models are: beta, power-law, broken power-law, and beta plus a
    power-law. To be added: double beta, cusp beta.
    """
    def __init__(self, fx_name):
        self.model = fx_name

    def lsq(self, profile, guess, bounds=None):
        """Fit a profile using least squares statistics.

        The fitting is done using the Levenberg-Marquardt algorithm.
        """
        nbins = len(profile)
        r = np.array([profile[i][0] for i in range(nbins)])
        sb = np.array([profile[i][6] for i in range(nbins)])
        sb_err = np.array([profile[i][7] for i in range(nbins)])

        if not bounds:
            bounds = (-np.inf, np.inf)

        fx = call_model(self.model)
        popt, pcov = curve_fit(fx, r, sb, p0=guess, sigma=sb_err)
        return popt

    def fit(self, profile, guess, method='leastsq'):
        if method == 'leastsq':
            return self.lsq(profile, guess)
