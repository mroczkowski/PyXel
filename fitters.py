import numpy as np

from astropy.modeling.fitting import Fitter

from optimizers import Minimize
from stats import cstat
import corner
import emcee

from astropy.modeling.fitting import (_validate_model,
                                      _fitter_to_model_params,
                                      _model_to_fit_params, Fitter,
                                      _convert_input)

class CstatFitter(Fitter):
    """
    Fit a model using the C-statistic. [1][2]

    Parameters
    ----------
    optimizer : class or callable
        one of the classes in optimizers.py or in astropy.modeling.optimizers
        (default: Minimize)

    .. [1] Cash, W. (1979), "Parameter estimation in astronomy through
           application of the likelihood ratio", ApJ, 228, p. 939-947
    .. [2] Wachter, K., Leach, R., Kellogg, E. (1979), "Parameter estimation
           in X-ray astronomy using maximum likelihood", ApJ, 230, p. 274-287
    """
    supported_constraints = ['bounds', 'eqcons', 'ineqcons', 'fixed', 'tied']

    def __init__(self, optimizer=None):
        if optimizer is None:
            optimizer = Minimize()

        def opt_func(*args, **kwargs):
            return optimizer(*args, **kwargs)

        super(CstatFitter, self).__init__(opt_func, statistic=cstat)

    def __call__(self, model, x, measured_raw_cts, measured_bkg_cts,
                 t_raw, t_bkg, **kwargs):
        model_copy = _validate_model(model,
                                     self.supported_constraints)
        farg = _convert_input(x, measured_raw_cts)
        farg = (model_copy, measured_bkg_cts, t_raw, t_bkg) + farg
        p0, _ = _model_to_fit_params(model_copy)

        fitparams, self.fit_info = self._opt_method(
            self.objective_function, p0, farg, **kwargs)
        _fitter_to_model_params(model_copy, fitparams)

        return model_copy

    def run_mcmc(self, model, x, measured_raw_cts, measured_bkg_cts,
                 t_raw, t_bkg, nburn=100, **kwargs):
        """Run Markov Chain Monte Carlo for parameter error estimation.

        `model` should be a fitted model as returned by `__call__`.

        Return the Markov Chain as a 3-dimensional array (walker, step, parameter).
        """
        model_copy = _validate_model(model,
                                     self.supported_constraints)
        params, _ = _model_to_fit_params(model_copy)
        ndim, nwalkers = len(params), 100
        pos = [params + 1e-4 * np.random.randn(ndim) * params
               for i in range(nwalkers)]

        def lnprob(mc_params, measured_raw_cts, measured_bkg_cts, t_raw, t_bkg, x):
            _fitter_to_model_params(model_copy, mc_params)
            lnp = 0. # TODO: evaluate prior based on bounds
            lnc = cstat(measured_raw_cts, model_copy, measured_bkg_cts, t_raw, t_bkg, x)
            return lnp - lnc

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                        args=(measured_raw_cts, measured_bkg_cts, t_raw, t_bkg, x))
        sampler.run_mcmc(pos, 500)
        samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))
        fig = corner.corner(samples)
        fig.savefig("triangle.png")
        return sampler.chain
#        ndim, nwalkers = len(self.params), 100
#        pos = [result["x"] + 1e-4 * np.random.randn(ndim) * result["x"]
#               for i in range(nwalkers)]
#        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob,
#                        args=(r, w, raw_cts, bkg, sb_to_counts_factor))
#        sampler.run_mcmc(pos, 500)
