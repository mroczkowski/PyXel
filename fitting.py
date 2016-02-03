import numpy as np
import scipy.optimize as op
from model import Model

class Minimizer(Model):
    """Fit a certain model to the data.

    Available models are: constant, beta, power-law, broken power-law, and
    beta plus a power-law.
    """
    def leastsq(self, data, guess):
        """Fit a profile using least squares statistics."""
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
        objective_func = '''... function we're trying to minimize ...'''
        if method == 'leastsq':
            return self.leastsq(profile, guess)
