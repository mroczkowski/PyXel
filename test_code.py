# File: test_code.py
# Author: Georgiana Ogrean
# Created on Jan 13, 2016
#
# Test the code.
# Provisory file.
#

import pyfits
import numpy as np
from load_data import load_image, load_region
import matplotlib.pyplot as plt
from fitting import FitModel
from models import beta
#from profile import Box #, counts_profile

source_img, hdr_src = load_image("srcfree_bin4_500-4000_band1_thresh.img")
bkg_img, hdr_bkg = load_image("srcfree_bin4_500-4000_bgstow_renorm.img")
exp_img, hdr_exp = load_image("srcfree_bin4_500-4000_thresh.expmap_nosrcedg")

region = load_region("beta.reg")

#pixels = region.interior_pixels()
#for pixel in pixels:
#    img[pixel[0], pixel[1]] = 2.
#
#pyfits.writeto('test.fits', img, header=hdr, clobber=True)


# Plot counts profile.
p = region.count_profile(source_img, bkg_img, exp_img, min_counts=100)
# region.plot_count_profile(p)

model = FitModel("beta").lsq(p, [0.6, 10.0, 50.0, 50.0])
print(model)

nbins = len(p)

r = np.array([p[i][0] for i in range(nbins)])
r_err = np.array([p[i][1] for i in range(nbins)])
bkg = np.array([p[i][4] for i in range(nbins)])
net_cts = np.array([p[i][6] for i in range(nbins)])
err_net_cts = np.array([p[i][7] for i in range(nbins)])

plt.scatter(r, net_cts, c="black", alpha=0.85, s=35, marker="s")
plt.errorbar(r, net_cts, xerr=r_err, yerr=err_net_cts,
             linestyle="None", color="black")
plt.step(r, bkg, where="mid")
plt.xlabel("Distance (pixels)")
plt.ylabel("Counts")

plt.grid(True)

plt.semilogx()
plt.semilogy()
plt.plot(r, beta(r, model[0], model[1], model[2], model[3]))
plt.show()
