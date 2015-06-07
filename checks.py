# File: checks.py
# Author: Georgiana Ogrean
# Created on 06.05.2015
#
# Sanity checks to avoid running with incorrect input.
#
# Change log:

import pyfits
import numpy as np
from SurfMessages import *

def get_size(img):
  return np.size(img[0].data)

# Check that counts, exposure, and background maps have the same size
def check_map_size(img,expmap,bkgmap):
  if expmap != None:
    if get_size(img) != get_size(expmap):
      raise SizeError('Count map and exposure map should have the same size.')
  if bkgmap != None:
    if get_size(img) != get_size(bkgmap):
      raise SizeError('Count map and background map should have the same size.')





