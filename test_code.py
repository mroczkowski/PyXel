import pyfits
import numpy as np
from load_data import Image, load_region
import matplotlib.pyplot as plt
#from profile import Box #, counts_profile

import os
import pickle

from astropy.modeling.functional_models import Const1D
from modelsastropy import Beta, BrokenPow
from astropy.modeling.fitting import LevMarLSQFitter

pkl = 'profile-core-circ.pkl'

if os.path.exists(pkl):
    with open(pkl, 'rb') as f:
        p = pickle.load(f)
else:
    # Read the images used to create the surface brightness profile.
    src_img = Image("srcfree_bin4_500-4000_band1_thresh.img")
    exp_img = Image("srcfree_bin4_500-4000_thresh.expmap_nosrcedg")
    bkg_img = Image("srcfree_bin4_500-4000_bgstow_renorm.img")

    # Read the region file in which the surface brightness profile will be created.
    region = load_region("core_sx_circ.reg")

    # Create the profile and bin it to a minimum of 100 counts/bin
    p = region.sb_profile(src_img, bkg_img, exp_img, min_counts=30, islog=False) 
    with open(pkl, 'wb') as f:
        pickle.dump(p, f)

rmin = 0.1
rmax = 5.0
r = np.array([pp[0] for pp in p if rmax >= pp[0] >= rmin])
r_err = np.array([pp[1] for pp in p if rmax >= pp[0] >= rmin])
sx = np.array([pp[7] for pp in p if rmax >= pp[0] >= rmin])
sx_err = np.array([pp[8] for pp in p if rmax >= pp[0] >= rmin])

mod = BrokenPow(ind1=0.5, ind2=1.2, norm=1e-4, rbreak=0.35, jump=2.0, const=9e-7)
#mod = Const1D(amplitude=1e-6)
mod.const.fixed = True
mod.rbreak.fixed = True

fit = LevMarLSQFitter()
m = fit(mod, r, sx, weights=1./sx_err, estimate_jacobian=True, maxiter=500) 
print(m)
print(fit.fit_info)

plt.figure(figsize=(8,5))
plt.scatter(r, sx, c="#1e8f1e", alpha=0.85, s=35, marker="s") 
plt.errorbar(r, sx, xerr=r_err, yerr=sx_err,
             linestyle="None", color="#1e8f1e")
plt.plot(r, m(r), color="#ffa500", linewidth=2, alpha=0.75)
plt.semilogx()
plt.semilogy()
plt.xlim(0.1, 8.0) 
plt.ylim(1e-7, 1e-3)
plt.savefig('blah.eps')




# mod = Beta()
# mod.set_parameter('beta', 0.6)
# mod.set_parameter('rc', 0.9)
# mod.set_parameter('s0', 3.7e-4, max_bound = 1e-2)
# mod.set_parameter('const', 3e-8)
# print(mod.fit(p, statistics='cash'))
# print(mod.params)
#
#
# mod = Model("beta") + Model("const")
# params = mod.set_params(beta=0.7, rc=0.4, s0=1e-4, const=3e-6)
#
# mod = Beta()
