import numpy as np
import scipy.integrate
from astropy.modeling import Fittable1DModel, Parameter
from scipy import integrate

def IntModel(model_cls):
    class MyIntModel(model_cls):
        def __init__(self, widths, *args, **kwargs):
            self._widths = widths
            super(MyIntModel, self).__init__(*args, **kwargs)

        def evaluate(self, x, *params):
            fn = super(MyIntModel, self).evaluate

            result = np.array([integrate.quad(lambda r: fn(r, *params), d - dw, d + dw)[0] / (2.*dw)
                               for d, dw in zip(x, self._widths)])
            return result

        def fit_deriv(self, x, *params):
            fn = super(MyIntModel, self).fit_deriv

            result = [np.array([integrate.quad(lambda r: fn(r, *params)[i], d - dw, d + dw)[0] / (2.*dw)
                                for d, dw in zip(x, self._widths)])
                      for i in range(len(params))]

            return result

    return MyIntModel

class Beta(Fittable1DModel):
    s0 = Parameter(default = 1e-2, min = 1e-12)
    beta = Parameter(default = 0.7, min = 1e-12)
    rc = Parameter(default = 0.1, min = 1e-12)
    const = Parameter(default = 1e-3, min=1e-12)

    @staticmethod
    def evaluate(x, s0, beta, rc, const):
        result = s0 * (1. + (x/rc)**2) ** (0.5 - 3*beta) + const
        return result

    @staticmethod
    def fit_deriv(x, s0, beta, rc, const):
        d_s0 = (1. + (x/rc)**2) ** (0.5 - 3*beta)
        d_beta = -3 * s0 * np.log(1. + (x/rc)**2) * \
                 (1. + (x/rc)**2) ** (0.5 - 3*beta)
        d_rc = -2 * s0 * x**2 * (0.5 - 3*beta) / rc**3 \
               * (1. + (x/rc)**2) ** (-0.5 - 3*beta)
        return [d_s0, d_beta, d_rc, np.ones_like(x)]

class BrokenPow(Fittable1DModel):
    ind1 = Parameter(default = 0.)
    ind2 = Parameter(default = 0.)
    norm = Parameter(default = 1e-2, min = 1e-12)
    rbreak = Parameter(default = 0.1, min = 1e-12)
    jump = Parameter(default = 2.0, min = 1., max = 4.)
    const = Parameter(default = 1e-3)

    @staticmethod
    def evaluate(x, ind1, ind2, norm, rbreak, jump, const):
        if not isinstance(x, (int, float)):
            sx = np.zeros_like(x)
            for i in range(len(sx)):
                sx[i] = BrokenPow.evaluate_one(x[i], ind1, ind2, norm, rbreak, jump)
        else:
            sx = BrokenPow.evaluate_one(x, ind1, ind2, norm, rbreak, jump)

        return sx+const

    @staticmethod
    def evaluate_one(xval, ind1, ind2, norm, rbreak, jump):
        norm_after_jump = norm / jump**2
        if xval <= rbreak:
            return norm * scipy.integrate.quad(lambda z:
                               ((xval**2 + z**2) / rbreak**2)**(-ind1),
                               1e-4, np.sqrt(rbreak**2 - xval**2))[0] + \
                    norm_after_jump * scipy.integrate.quad(lambda z:
                               ((xval**2 + z**2) / rbreak**2)**(-ind2),
                               np.sqrt(rbreak**2 - xval**2), 1e4)[0]
        else:
            return norm_after_jump * scipy.integrate.quad(lambda z:
                               ((xval**2 + z**2) / rbreak**2)**(-ind2),
                                1e-4, 1e4)[0]

    @staticmethod
    def fit_deriv(xval, ind1, ind2, norm, rbreak, jump, const):
        fn1 = lambda z: ((xval**2 + z**2) / rbreak**2)**(-ind1)
        fn2 = lambda z: ((xval**2 + z**2) / rbreak**2)**(-ind2)
        norm_after_jump = norm / jump**2
        if xval <= rbreak:
            d_ind1 = -norm * scipy.integrate.quad(lambda z: fn1(z) *
                         np.log((xval**2 + z**2)/rbreak**2),
                         1e-4, np.sqrt(rbreak**2 - xval**2))[0]

            d_ind2 = -norm_after_jump * scipy.integrate.quad(lambda z: fn2(z) *
                         np.log((xval**2 + z**2)/rbreak**2),
                         np.sqrt(rbreak**2 - xval**2), 1e4)[0]

            d_norm = scipy.integrate.quad(lambda z:
                           ((xval**2 + z**2) / rbreak**2)**(-ind1),
                           1e-4, np.sqrt(rbreak**2 - xval**2))[0] + \
                     1./jump**2 * scipy.integrate.quad(lambda z:
                            ((xval**2 + z**2) / rbreak**2)**(-ind2),
                            np.sqrt(rbreak**2 - xval**2), 1e4)[0]

            d_rbreak = norm * (2. * ind1 * scipy.integrate.quad(lambda z:
                                    fn1(z) / rbreak,
                                    1e-4, np.sqrt(rbreak**2 - xval**2))[0] +
                       rbreak / np.sqrt(rbreak**2 - xval**2)) + \
                       norm_after_jump * (2. * ind2 * scipy.integrate.quad(
                                    lambda z: fn2(z) / rbreak,
                                    np.sqrt(rbreak**2 - xval**2), 1e4)[0] -
                       rbreak / np.sqrt(rbreak**2 - xval**2))

            d_jump = -2*norm / jump**3 * scipy.integrate.quad(lambda z:
                         ((xval**2 + z**2) / rbreak**2)**(-ind2),
                         np.sqrt(rbreak**2 - xval**2), 1e4)[0]

        else:
            d_ind1 = np.zeros_like(xval)

            d_ind2 = -norm_after_jump * scipy.integrate.quad(lambda z: fn2(z) *
                         np.log((xval**2 + z**2)/rbreak**2),
                         1e-4, 1e4)[0]

            d_norm = 1/jump**2 * scipy.integrate.quad(lambda z:
                               ((xval**2 + z**2) / rbreak**2)**(-ind2),
                                1e-4, 1e4)[0]

            d_rbreak = 2. * norm_after_jump * ind2 * scipy.integrate.quad(
                         lambda z: fn2(z) / rbreak, 1e-4, 1e4)[0]

            d_jump = -2*norm / jump**3 * scipy.integrate.quad(lambda z:
                         ((xval**2 + z**2) / rbreak**2)**(-ind2),
                         1e-4, 1e4)[0]

        d_const = np.ones_like(xval)
        return [d_ind1, d_ind2, d_norm, d_rbreak, d_jump, d_const]
