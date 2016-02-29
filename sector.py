from math import floor, ceil
import pyfits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from SurfMessages import ErrorMessages, InfoMessages
from aux import rotate_point, call_model
import profile

class Sector(profile.Region):
    """Generate elliptical sector."""
    def __init__(self, x0, y0, start_angle, end_angle,
                 major_axis, minor_axis, rot_angle):
       self.x0 = x0
       self.y0 = y0
       self.start_angle = start_angle
       self.end_angle = end_angle
       self.major_axis = major_axis
       self.minor_axis = minor_axis
       self.rot_angle = rot_angle

    @classmethod
    def from_params(cls, params):
        """Make box parameters Python-compliant.

        Because DS9 pixels are 1-based, while Python arrays are 0-based, 1 is
        subtracted from the coordinates of the sector origin. The rotation angle
        is also converted from degrees (as defined in DS9) to radians.
        """
        start_angle = params[2] * np.pi / 180.
        end_angle = params[3] * np.pi / 180.
        rot_angle = params[6] * np.pi / 180.
        return Box(params[0] - 1, params[1] - 1, start_angle, end_angle,
                   params[4], params[5], rot_angle)
