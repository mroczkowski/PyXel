# File: profile.py
# Author: Georgiana Ogrean
# Created on 06.05.2015
# 
# Create surface brightness profile.
#
# Change log:

import pyfits
import numpy as np
from checks import check_map_size, check_shape, check_params

def profile(img,expmap,bkgmap,shape,params):
  check_maps(img,expmap,bkgmap)
  check_shape(shape)
  check_params(shape,params)

