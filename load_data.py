# File: profiles.py
# Author: Georgiana Ogrean
# Created on 03.22.2015
#
# Load counts image, exposure map, and background map 
#
# Change log:

import pyfits
import numpy as np

def load_image(filename):
  img = pyfits.open(filename)
  return img

def load_exp(filename):
  exp = pyfits.open(filename)
  return exp

def load_bkg(filename):
  bkg = pyfits.open(filename)
  return bkg

