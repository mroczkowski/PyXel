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
    def from_epanda_params(cls, params):
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

    @classmethod
    def from_panda_params(cls, params):
        """Make circular sector parameters Python-compliant.

        Because DS9 pixels are 1-based, while Python arrays are 0-based, 1 is
        subtracted from the coordinates of the sector origin. The rotation angle
        is set to 0, while the major axis is set equal to the minor axis. This
        is simply a special case for an elliptical sector.
        """
        start_angle = params[2] * np.pi / 180.
        end_angle = params[3] * np.pi / 180.
        return Epanda(params[0] - 1, params[1] - 1, start_angle, end_angle,
                   params[6], params[6], 0.)

    @classmethod
    def from_ellipse_params(cls, params):
        """Make ellipse parameters Python-compliant.

        Because DS9 pixels are 1-based, while Python arrays are 0-based, 1 is
        subtracted from the coordinates of the center. This is treated as a
        special case for an elliptical sector, so the start and end angles are
        set to 0 and 360, respectively.
        """
        rot_angle = params[4] * np.pi / 180.
        return Epanda(params[0] - 1, params[1] - 1, 0., 360.,
                   params[2], params[3], rot_angle)

    @classmethod
    def from_circle_params(cls, params):
        """Make circle parameters Python-compliant.

        Because DS9 pixels are 1-based, while Python arrays are 0-based, 1 is
        subtracted from the coordinates of the center. This is treated as a
        special case for an elliptical sector, so the start and end angles are
        set to 0 and 360, respectively, the major and minor axes are set
        equal to the radius of the circle, and the rotation angle is set to 0.
        """
        return Epanda(params[0] - 1, params[1] - 1, 0., 360.,
                   params[2], params[2], 0.)

    def get_bounds(self):
        x_min_bound = floor(self.x0 - self.major_axis)
        x_max_bound = floor(self.x0 + self.major_axis)
        y_min_bound = ceil(self.y0 - self.major_axis)
        y_max_bound = ceil(self.y0 + self.major_axis)
        bounds = [[x_min_bound, y_min_bound], [x_max_bound, y_max_bound]]
        return bounds

    def interior_pixels(self, start_edge, end_edge):
        """Find the pixels inside an elliptical sector annulus.

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
                minor_end_edge = end_edge / self.major_axis * self.minor_axis
                minor_start_edge = start_edge / self.major_axis * \
                                   self.minor_axis
                outer_ellipse_eq = (x_rot_back - self.x0)**2 / end_edge**2 + \
                                   (y_rot_back - self.y0)**2 / minor_end_edge**2
                inner_ellipse_eq = (x_rot_back - self.x0)**2 / start_edge**2 + \
                                   (y_rot_back - self.y0)**2 / \
                                   minor_start_edge**2
                if x_rot_back - self.x0 >= 0:
                    r = np.sqrt((x_rot_back - self.x0)**2 + \
                                (y_rot_back - self.y0)**2)
                    xy_angle = np.arcsin((y_rot_back - self.y0) / r)
                    if xy_angle < 0:
                        xy_angle = 2 * np.pi + xy_angle
                else:
                    xy_angle = np.arctan((y_rot_back - self.y0) /
                                         (x_rot_back - self.x0)) + np.pi
                if outer_ellipse_eq < 1 and inner_ellipse_eq >= 1:
                    if self.start_angle <= xy_angle <= self.end_angle:
                        pixels.append((y, x))
        return pixels

    def get_bin_vals(self, counts_img_data, bkg_img_data, bkg_norm_factor,
        exp_img_data, pixels_in_bin, only_counts=False, only_net_cts=False):
        """Calculate the number of counts in a bin."""
        src = 0
        bkg = 0
        err_src = 0
        err_bkg = 0
        exp = 0

#        print("pixels in bin in get_bin_vals ", pixels_in_bin)
        npix = len(pixels_in_bin)

        raw_cts = 0
        for pixel in pixels_in_bin:
            if exp_img_data[pixel[0], pixel[1]] != 0:
                exp_val = exp_img_data[pixel[0], pixel[1]]
                # In case the profile needed is a counts profile...
                if only_counts:
                    exp_val = 1.
                raw_cts += counts_img_data[pixel[0], pixel[1]]
                src += counts_img_data[pixel[0], pixel[1]] / exp_val
                bkg += bkg_img_data[pixel[0], pixel[1]] / exp_val \
                    / bkg_norm_factor
                err_src += counts_img_data[pixel[0], pixel[1]] / exp_val**2
                err_bkg += bkg_img_data[pixel[0], pixel[1]] / exp_val**2 \
                    / bkg_norm_factor**2
                exp += exp_val

        net = src - bkg

        if only_net_cts:
            return net
        else:
            net = net / npix
            src = src / npix
            bkg = bkg / npix
            err_net = np.sqrt(err_src + err_bkg) / npix
            err_src = np.sqrt(err_src) / npix
            err_bkg = np.sqrt(err_bkg) / npix
            exp /= npix
            print('src, bkg, net, raw_cts: ', src, bkg, net, raw_cts)
            return raw_cts, src, err_src, bkg, err_bkg, net, err_net

    def rebin_data(self, counts_img, bkg_img, exp_img, min_counts, islog=True):
        bkg_img_data, bkg_norm_factor, exp_img_data = \
            get_bkg_exp(bkg_img, exp_img)
        edges = get_edges(self.major_axis, islog)
        raw_bins = self.distribute_pixels(edges)
        rollovered_pixels = []
        for bin in raw_bins:
            pixels_in_bin = bin[2] + rollovered_pixels
            net_counts = self.get_bin_vals(counts_img.data,
                bkg_img_data, bkg_norm_factor, exp_img_data, pixels_in_bin,
                only_counts=True, only_net_cts=True)
            if net_counts < min_counts:
