import numpy as np
import scipy.integrate
from astropy.modeling import Fittable1DModel, Parameter

class Beta(Fittable1DModel):
    s0 = Parameter(default = 1e-2, min = 1e-12)
    beta = Parameter(default = 0.7, min = 1e-12)
    rc = Parameter(default = 0.1, min = 1e-12)
    const = Parameter(default = 1e-3, min=1e-12)

    @staticmethod
    def evaluate(x, s0, beta, rc, const):
        result = s0 * (1. + (x/rc)**2) ** (0.5 - 3*beta) + const
        print(result)
        return result

    @staticmethod
    def fit_deriv(x, s0, beta, rc, const):
        print('deriv...')
        d_s0 = (1. + (x/rc)**2) ** (0.5 - 3*beta)
        d_beta = -3 * s0 * np.log(1. + (x/rc)**2) * \
                 (1. + (x/rc)**2) ** (0.5 - 3*beta)
        d_rc = -2 * s0 * x**2 * (0.5 - 3*beta) / rc**3 \
               * (1. + (x/rc)**2) ** (-0.5 - 3*beta)
        return [d_s0, d_beta, d_rc, list(np.ones(len(x)))]

class BrokenPow(Fittable1DModel):
    ind1 = Parameter(default = 0.)
    ind2 = Parameter(default = 0.)
    norm = Parameter(default = 1e-2, min = 1e-12)
    rbreak = Parameter(default = 0.1, min = 1e-12)
    jump = Parameter(default = 2.0, min = 1., max = 4.)
    const = Parameter(default = 1e-3)

    @staticmethod
    def evaluate(x, ind1, ind2, norm, rbreak, jump, const):
        norm_after_jump = norm / jump**2
        sx = np.zeros(len(x))
        for i in range(len(x)):
            xval = x[i]
            if xval <= rbreak:
                sx[i] = norm * scipy.integrate.quad(lambda z:
                                   ((xval**2 + z**2) / rbreak**2)**(-ind1),
                                   1e-4, np.sqrt(rbreak**2 - xval**2))[0] + \
                        norm_after_jump * scipy.integrate.quad(lambda z:
                                   ((xval**2 + z**2) / rbreak**2)**(-ind2),
                                   np.sqrt(rbreak**2 - xval**2), 1e4)[0]
            else:
                sx[i] = norm_after_jump * scipy.integrate.quad(lambda z:
                                   ((xval**2 + z**2) / rbreak**2)**(-ind2),
                                    1e-4, 1e4)[0]
        return sx+const

#    @staticmethod
#    def fit_deriv(x, ind1, ind2, norm, rbreak, jump):
