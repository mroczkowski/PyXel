# File: checks.py
# Author: Georgiana Ogrean
# Created on 06.05.2015
#
# Sanity checks to avoid running with incorrect input.
#
# Change log:

import pyfits
import numpy as np

def check_maps(img,expmap,bkgmap):
  size_img = np.size(img['IMAGE'].data)
  if expmap != None:
    size_exp = np.size(expmap['IMAGE'].data)
    if size_exp != size_img:
      'Catastrophic failure!'
  if bkgmap != None:
    size_bkg = np.size(bkgmap['IMAGE'].data)
    if size_bkg != size_img:
      'Catastrophic failure!'



