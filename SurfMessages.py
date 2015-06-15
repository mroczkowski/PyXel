# File: SurfMessages.py
# Author: Georgiana Ogrean
# 
# Errors and warnings raised by SurfFit.

class SizeError(Exception):
  # Raised when two images do not have the same size
  pass

class RegionError(Exception):
  # Raised when something is wrong with the region defined by the user.
  pass
