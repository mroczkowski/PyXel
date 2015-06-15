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
  if expmap is not None:
    if get_size(img) != get_size(expmap):
      raise SizeError('Count map and exposure map should have the same size.')
  if bkgmap is not None:
    if get_size(img) != get_size(bkgmap):
      raise SizeError('Count map and background map should have the same size.')

# Check that the region in which the profile is 
# created is a valid region. 
def check_shape(shape):
  if shape.lower() not in ('circle', 'panda', 'ellpanda'):
    raise ShapeError("Invalid region. Valid region are 'circle', 'panda', and 'ellpanda'.")

def check_params(shape, params):
  if shape.lower() == 'circle':
    if params[0] < 0. or params[0] > 360. or params[1] < -180. or params[1] > 180.:
      raise RegionError('Incorrect region parameters. Circles should have 0 <= RA <= 180 and -180 <= DEC <= 180.')
  elif shape.lower() == 'panda':
    if params[0] < 0. or params[0] > 360. or params[1] < -180. or params[1] > 180. or params[2] < 0. or params[2] > 360. or params[3] < 0. or params[3] > 360. or params[4] > params[5]:
      raise RegionError('Incorrect region parameters. Panda should have 0 <= RA <= 180, -180 <= DEC <= 180, 0 <= ANGLE1 <= 360, 0 <= ANGLE2 <= 360, and R_INNER < R_OUTER.') 
  elif shape.lower() == 'epanda':



