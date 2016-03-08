from math import floor, ceil
import pyfits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from SurfMessages import ErrorMessages, InfoMessages
from aux import rotate_point, call_model, get_edges, get_bkg_exp
import profile

class Box(profile.Region):
    """Generate box object."""
    def __init__(self, x0, y0, width, height, angle):
       self.x0 = x0
       self.y0 = y0
       self.width = width
       self.height = height
       self.angle = angle

    @classmethod
    def from_params(cls, params):
        """Make box parameters Python-compliant.

        Because DS9 pixels are 1-based, while Python arrays are 0-based, 1 is
        subtracted from the coordinates of the box center. The rotation angle
        of the box is also converted from degrees (as defined in DS9) to radians.
        """
        angle = params[4] * np.pi / 180.
        return Box(params[0] - 1, params[1] - 1, params[2], params[3], angle)

    def get_corners(self):
        """Get the coordinates of the corners of a box."""
        corners = [(-self.width/2, -self.height/2),
                   (-self.width/2, self.height/2),
                   (self.width/2, self.height/2),
                   (self.width/2, -self.height/2)]
        rotated_corners = [rotate_point(self.x0, self.y0, x, y, self.angle)
            for x, y in corners]
#        print(rotated_corners)
        return rotated_corners

    def make_edges(self, islog):
        return get_edges(self.height, islog)

    def distribute_pixels(self, edges):
        corners = self.get_corners()
        reg_path = Path(corners)
        # Get region boundaries.
        bounds = reg_path.get_extents().get_points()
        [[x_min_bound, y_min_bound], [x_max_bound, y_max_bound]] = bounds
        # For cases when the boundary pixels are not integers:
        x_min_bound = floor(x_min_bound)
        y_min_bound = floor(y_min_bound)
        x_max_bound = ceil(x_max_bound)
        y_max_bound = ceil(y_max_bound)
        pixels_in_bins = []
        for x in range(x_min_bound, x_max_bound+1):
            for y in range(y_min_bound, y_max_bound+1):
                if reg_path.contains_point((x, y)):
                    x_nonrotated, y_nonrotated = rotate_point(self.x0, self.y0,
                                                              x - self.x0,
                                                              y - self.y0,
                                                              -self.angle)
                    dist_from_box_bottom = self.height/2. - \
                                           (self.y0 - y_nonrotated)
                    for i, edge in enumerate(edges[1:]):
                        if edge > dist_from_box_bottom:
                            pixels_in_bins.append((y, x, i))
                            break
        return pixels_in_bins

    def merge_bins(self, counts_img, bkg_img, exp_img, min_counts, islog=True):
        bkg_img_data, bkg_norm_factor, exp_img_data = \
            get_bkg_exp(bkg_img, exp_img)
        edges = self.make_edges(islog)
        print('edges = ', edges)
        pixels_in_bins = self.distribute_pixels(edges)
        nbins = len(edges) - 1
        npix = len(pixels_in_bins)
        bins = []
        start_edge = edges[0]
        end_edge = edges[1]
        pixels_in_current_bin = []
        for i in range(nbins):
            end_edge = edges[i+1]
            pixels_in_current_bin.extend(
                    [(pixels_in_bins[j][0], pixels_in_bins[j][1])
                      for j in range(npix) if pixels_in_bins[j][2] == i])
            net_counts = self.get_bin_vals(counts_img.data,
                bkg_img_data, bkg_norm_factor, exp_img_data,
                pixels_in_current_bin, only_counts=True, only_net_cts=True)
            print('net_counts = ', net_counts)
            if net_counts < min_counts:
                if end_edge == edges[-1] and len(bins) != 0:
                    bins[-1][2].extend(pixels_in_current_bin)
                    updated_last_bin = (bins[-1][0], end_edge, bins[-1][2])
                    list(bins)[-1] = updated_last_bin
                elif end_edge == edges[-1] and len(bins) == 0:
                    error_message = ErrorMessages('001')
                    raise ValueError(error_message)
                else:
                    continue
            else:
                print(start_edge, end_edge, pixels_in_current_bin)
                bins.append((start_edge, end_edge, pixels_in_current_bin))
                start_edge = end_edge
                pixels_in_current_bin = []
        return bins
