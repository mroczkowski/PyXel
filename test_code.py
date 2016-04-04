import os
import pickle
import numpy as np

import matplotlib.pyplot as plt
from astropy.modeling.fitting import LevMarLSQFitter

from load_data import Image, load_region
from models import Beta, BrokenPow, IntModel

#from astropy.modeling.functional_models import Const1D

pkl = 'profile-core-ell.pkl'

if os.path.exists(pkl):
    with open(pkl, 'rb') as f:
        p = pickle.load(f)
else:
    # Read the images used to create the surface brightness profile.
    src_img = Image("srcfree_bin4_500-4000_band1_thresh.img")
    exp_img = Image("srcfree_bin4_500-4000_thresh.expmap_nosrcedg")
    bkg_img = Image("srcfree_bin4_500-4000_bgstow_renorm.img")
    bkg_err_img = Image("srcfree_bin4_500-4000_bgstow_err_renorm.img")

    # Read the region file in which the surface brightness profile will be created.
    region = load_region("core_sx_ell.reg")

    # Create the profile and bin it to a minimum of 100 counts/bin
    p = region.sb_profile(src_img, bkg_img, bkg_err_img, exp_img,
                          min_counts=30, islog=False)
    with open(pkl, 'wb') as f:
        pickle.dump(p, f)

rmin = 0.1
rmax = 1.0
r = np.array([pp[0] for pp in p if rmax >= pp[0] >= rmin])
r_err = np.array([pp[1] for pp in p if rmax >= pp[0] >= rmin])
bkg = np.array([pp[5] for pp in p if rmax >= pp[0] >= rmin])
bkg_err = np.array([pp[6] for pp in p if rmax >= pp[0] >= rmin])
sx = np.array([pp[7] for pp in p if rmax >= pp[0] >= rmin])
sx_err = np.array([pp[8] for pp in p if rmax >= pp[0] >= rmin])

#mod = Beta(s0=1e-3, beta=0.7, rc=0.25, const=9e-7)
mod = BrokenPow(ind1=0.5, ind2=1.2, norm=1e-4, rbreak=0.322, jump=2.0, const=1.30203247569e-06)
mod.const.fixed = True
intmod = IntModel(radius=r, widths=r_err, model=mod)

fit = LevMarLSQFitter()
m = fit(intmod, r, sx, weights=1./sx_err, estimate_jacobian=True, maxiter=500)
print(m)
print(fit.fit_info)
print(m.ind1, m.ind2, m.norm, m.rbreak, m.jump)
