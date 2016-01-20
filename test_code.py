# File: test_code.py
# Author: Georgiana Ogrean
# Created on Jan 13, 2016
#
# Test the code.
# Provisory file.
#

import pyfits
from load_data import load_image, load_region
#from profile import Box #, counts_profile

img, hdr = load_image("counts_image.fits")
region = load_region("test_region.reg")
pixels = region.interior_pixels()
for pixel in pixels:
    img[pixel[0], pixel[1]] = 2.

pyfits.writeto('test.fits', img, header=hdr, clobber=True)

p = region.counts_profile(img, None, None, min_counts=101)
