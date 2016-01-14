# File: test_code.py
# Author: Georgiana Ogrean
# Created on Jan 13, 2016
#
# Test the code.
# Provisory file.
#

import pyfits
from load_data import load_image, load_region
from profile import interior_pixels

img, hdr = load_image("counts_image.fits")
region = load_region("test_region.reg")
pixels = interior_pixels(img, region)
for pixel in pixels:
    print(pixel, img[pixel[0], pixel[1]])
    img[pixel[0], pixel[1]] = 5000.
img[0,0] = 5000
img[0,1] = 5000

pyfits.writeto('test.fits', img, header=hdr, clobber=True)
