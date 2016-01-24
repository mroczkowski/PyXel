import pyfits
import numpy as np
from load_data import load_image, load_region
import matplotlib.pyplot as plt
from fitting import FitModel
from models import beta
#from profile import Box #, counts_profile

src_img, src_hdr = load_image("srcfree_bin4_500-4000_band1_thresh.img")
bkg_img, hdr_bkg = load_image("srcfree_bin4_500-4000_bgstow_renorm.img")
exp_img, hdr_exp = load_image("srcfree_bin4_500-4000_thresh.expmap_nosrcedg")

region = load_region("beta.reg")

#pixels = region.interior_pixels()
#for pixel in pixels:
#    img[pixel[0], pixel[1]] = 2.
#
#pyfits.writeto('test.fits', img, header=hdr, clobber=True)


# Plot counts profile.
p = region.sb_profile(src_img, bkg_img, exp_img, min_counts=50)

model = FitModel("beta").lsq(p, [0.6, 10.0, 50.0, 50.0])
region.plot_profile(p, src_hdr, \
    with_model=False, model_name="beta", model_params=model, \
    xlabel=r"Distance (pixels)", \
    ylabel=r"photons s$^{-1}$ cm$^{-2}$ pixel$^{-1}$")
