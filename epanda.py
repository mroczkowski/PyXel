from math import floor, ceil
import pyfits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from SurfMessages import ErrorMessages, InfoMessages
from aux import rotate_point, call_model
import profile

class Epanda(profile.Region):
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
       print(self.x0, self.y0, self.start_angle, self.end_angle,
             self.major_axis, self.minor_axis, self.rot_angle)

    @classmethod
    def from_params(cls, params):
        """Make elliptical sector parameters Python-compliant.

        Because DS9 pixels are 1-based, while Python arrays are 0-based, 1 is
        subtracted from the coordinates of the sector origin. The rotation angle
        is also converted from degrees (as defined in DS9) to radians.
        """
        start_angle = params[2] * np.pi / 180.
        end_angle = params[3] * np.pi / 180.
        rot_angle = params[10] * np.pi / 180.
        return Epanda(params[0] - 1, params[1] - 1, start_angle, end_angle,
                   params[7], params[8], rot_angle)

    def get_bounds(self):
        x_min_bound = floor(self.x0 - self.major_axis)
        x_max_bound = floor(self.x0 + self.major_axis)
        y_min_bound = ceil(self.y0 - self.major_axis)
        y_max_bound = ceil(self.y0 + self.major_axis)
        bounds = [[x_min_bound, y_min_bound], [x_max_bound, y_max_bound]]
        return bounds

    def interior_pixels(self):
        """Find the pixels inside an elliptical sector.

        Returns an array of tuples containing the coordinates (row, col) of the
        pixels whose centers are within a certain annulus of the elliptical
        sector.
        """
        pixels = []
        [[x_min_bound, y_min_bound],
         [x_max_bound, y_max_bound]] = self.get_bounds()
        for x in range(x_min_bound, x_max_bound+1):
            for y in range(y_min_bound, y_max_bound+1):
                x_rel = x - self.x0
                y_rel = y - self.y0
                x_rot_back, y_rot_back = rotate_point(self.x0, self.y0, x_rel,
                                                      y_rel, -self.rot_angle)
                ellipse_eq = (x_rot_back - self.x0)**2 / self.major_axis**2 + \
                             (y_rot_back - self.y0)**2 / self.minor_axis**2
                if x_rot_back - self.x0 >= 0:
                    r = np.sqrt((x_rot_back - self.x0)**2 + \
                                (y_rot_back - self.y0)**2)
                    xy_angle = np.arcsin((y_rot_back - self.y0) / r)
                    if xy_angle < 0:
                        xy_angle = 2 * np.pi + xy_angle
                    count1 += 1
                else:
                    xy_angle = np.arctan((y_rot_back - self.y0) /
                                         (x_rot_back - self.x0)) + np.pi
                if ellipse_eq <= 1:
                    if self.start_angle <= xy_angle <= self.end_angle:
                        pixels.append((y, x))
        return pixels
