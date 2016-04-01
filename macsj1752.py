import numpy as np
import matplotlib.pyplot as plt
import pyfits

from load_data import Image, load_region

from astropy.modeling.functional_models import Const1D
from modelsastropy import Beta
from astropy.modeling.fitting import LevMarLSQFitter

# Read the images used to create the surface brightness profile.
src_img = Image("comb-obj-im-500-4000.fits")
bkg_img = Image("comb-back-im-sky-500-4000.fits")
exp_img = Image("comb-exp-im-500-4000-nosrc.fits")

# Read the region file in which the surface brightness profile will be created.
region = load_region("hfs16.reg")

# Create the profile and bin it to a minimum of 100 counts/bin
p = region.sb_profile(src_img, bkg_img, exp_img, min_counts=30)

# Define the model that will be fitted to the data, which in this case is a
# simple beta model:
#
#       S = S0 * (1 + (r/rc)**2)**(0.5 - 3*beta)
mod = Beta(beta=0.8, rc=2.0, s0=1e-2, const=1e-4)

# Fit the data.
#mod.fit(p, statistics='cash', min_range=1., max_range=8.3)
r = np.array([pp[0] for pp in p])
r_err = np.array([pp[1] for pp in p])
sx = np.array([pp[7] for pp in p])
sx_err = np.array([pp[8] for pp in p])
fit = LevMarLSQFitter()
m = fit(mod, r, sx, weights=1./sx_err)
print(m)
print(fit.fit_info)
print('chi2=', sum(fit.fit_info['fvec']**2))
plt.figure(figsize=(8,5))
plt.scatter(r, sx, c="#1e8f1e", alpha=0.85, s=35, marker="s")
plt.errorbar(r, sx, xerr=r_err, yerr=sx_err,
             linestyle="None", color="#1e8f1e")
plt.plot(r, m(r), color="#ffa500", linewidth=2, alpha=0.75)
plt.semilogx()
plt.semilogy()
plt.savefig('blah.eps')

# Plot the data and the best-fitting model.
#region.plot_profile(p, xlog=True, ylog=True, xlims=(1.0, 8.3), \
#    model_name="Beta", model=mod, \
#    xlabel=r"Distance (arcmin)", \
#    ylabel=r"Surface Brightness (counts s$^{-1}$ arcmin$^{-2}$)")
