# File: models.py
# Author: Georgiana Ogrean
# Created on 03.22.2015
#

def const(c):
    """Fit a constant to the data."""
    return c

def beta(r, beta, rc, s0, bkg):
   """Fit beta-model to the data."""
   mod = bkg + s0 * (1. + (r/rc)**2.) ** (-3.*beta+0.5)
   return mod
