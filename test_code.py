import pyfits
import numpy as np
from load_data import Image, load_region
import matplotlib.pyplot as plt
from models import Beta
#from profile import Box #, counts_profile

src_img = Image("srcfree_bin4_500-4000_band1_thresh.img")
bkg_img = Image("srcfree_bin4_500-4000_bgstow_renorm.img")
exp_img = Image("srcfree_bin4_500-4000_thresh.expmap_nosrcedg")

region = load_region("beta.reg")

#pixels = region.interior_pixels()
#for pixel in pixels:
#    img[pixel[0], pixel[1]] = 2.
#
#pyfits.writeto('test.fits', img, header=hdr, clobber=True)

# # Plot counts profile.
p = region.sb_profile(src_img, bkg_img, exp_img, min_counts=50)

# model = Model("beta").fit(p, guess, method='leastsq')
# region.plot_profile(p, xlog=True, ylog=True, \
#     with_model=True, model_name="beta", model_params=model, \
#     xlabel=r"Distance (arcmin)", \
#     ylabel=r"photons s$^{-1}$ cm$^{-2}$ pixel$^{-1}$")
#
#
# mod = Model("beta") + Model("const")
# params = mod.set_params(beta=0.7, rc=0.4, s0=1e-4, const=3e-6)

mod = Beta()
mod.show_params()
mod.set_parameter('beta', 1.2)
mod.set_parameter('rc', 1.2)
mod.set_parameter('s0', 4.5e-4)
#mod.set_constraints([{'type': 'ineq', 'fun': lambda p: +p.rc - p.beta}])

print(mod.fit(p, statistics='cash'))
