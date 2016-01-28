import pyfits
import numpy as np
from load_data import Image, load_region
import matplotlib.pyplot as plt
from fitting import FitModel
from models import beta
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


# Plot counts profile.
p = region.sb_profile(src_img, bkg_img, exp_img, min_counts=50)

model = FitModel("beta").lsq(p, [0.6, 0.5, 1e-2, 1e-6])

region.plot_profile(p, xlog=False, ylog=False, \
    with_model=True, model_name="beta", model_params=model, \
    xlabel=r"Distance (arcmin)", \
    ylabel=r"photons s$^{-1}$ cm$^{-2}$ pixel$^{-1}$")
