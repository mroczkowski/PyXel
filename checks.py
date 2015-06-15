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
  if shape.lower() not in ('circle', 'panda', 'epanda'):
    raise ShapeError("Invalid region. Valid region are 'circle', 'panda', and 'epanda'.")

def check_params(shape, params):
  if shape.lower() == 'circle':
    if not 0 <= params[0] <= 360 or not -90 <= params[1] <= 90:
      raise RegionError('Incorrect region parameters. Circles should have 0 <= RA <= 360 and -90 <= DEC <= 90.')
  elif shape.lower() == 'panda':
    if not 0 <= params[0] <= 360 or not -90 <= params[1] <= 90 or not 0 <= params[2] <= 360 or not 0 <= params[3] <= 360 or not params[4] > params[5]:
      raise RegionError('Incorrect region parameters. Panda should have 0 <= RA <= 360, -90 <= DEC <= 90, 0 <= ANGLE1 <= 360, 0 <= ANGLE2 <= 360, and R_INNER < R_OUTER.') 
  elif shape.lower() == 'epanda':
    if not 0 <= params[0] <= 360 or not -90 <= params[1] <= 90 or not 0 <= params[2] <= 360 or not 0 <= params[3] <= 360 or not params[4] < params[5] or not params[6] > params[4] or not 0 <= params[7] <= 360:
      raise RegionError('Incorrect region parameters. Elliptical panda should have 0 <= RA <= 360, -90 <= DEC <= 90, 0 <= ANGLE1 <= 360, 0 <= ANGLE2 <= 360, R_MIN_INNER < R_MAJ_OUTER, R_MAJ_INNER < R_MAJ_OUTER, and 0 <= ROTANGLE <= 360.')



