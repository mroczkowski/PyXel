from math import floor, ceil
import pyfits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from SurfMessages import ErrorMessages, InfoMessages
from aux import rotate_point, call_model
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

    # Return an array containing the coordinates of the points inside
    # the region of interest.
    def interior_pixels(self):
        """Find the pixels inside a box.

        Returns an array of tuples containing the coordinates (row, col) of the
        pixels whose centers are within a certain box region. This works okay
        for most regions. However, for boxes rotated by a multiple of 90 deg and
        whose edges go precisely through the middle of the corresponding pixels,
        those pixels are not included in the box. While this is a very special
        case, it causes about half of the pixels to be ignored.
        """
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
        pixels = []
        for x in range(x_min_bound, x_max_bound+1):
            for y in range(y_min_bound, y_max_bound+1):
                if reg_path.contains_point((x, y)):
                    pixels.append((y, x))
        return pixels

    def update_last_bin(self, prev_bin, current_bin_pixels):
        """Update the last bin of the profile.

        If the last bin cannot accumulate enough counts before the end of the
        region is reached, then the previous bin's radius and width are
        increased to include the remaining area. The pixel lists are merged.
        """
        new_bin_radius = prev_bin[0] + self.height / 2.
        new_bin_height = prev_bin[1] + self.height / 2.
        prev_bin[2].extend(current_bin_pixels)
        return (new_bin_radius, new_bin_height, prev_bin[2])

    def get_bin_vals(self, counts_img_data, bkg_img_data, bkg_norm_factor,
        exp_img_data, pixels_in_bin, only_counts=False, only_net_cts=False):
        """Calculate the number of counts in a bin."""
        src = 0
        bkg = 0
        err_src = 0
        err_bkg = 0

#        print("pixels in bin in get_bin_vals ", pixels_in_bin)
        npix = len(pixels_in_bin)

        for pixel in pixels_in_bin:
            if exp_img_data[pixel[0], pixel[1]] != 0:
                exp_val = exp_img_data[pixel[0], pixel[1]]
                # In case the profile needed is a counts profile...
                if only_counts:
                    exp_val = 1.
                src += counts_img_data[pixel[0], pixel[1]] / exp_val
                bkg += bkg_img_data[pixel[0], pixel[1]] / exp_val \
                    / bkg_norm_factor
                err_src += counts_img_data[pixel[0], pixel[1]] / exp_val**2
                err_bkg += bkg_img_data[pixel[0], pixel[1]] / exp_val**2 \
                    / bkg_norm_factor**2

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
            return src, err_src, bkg, err_bkg, net, err_net

    def bin_region(self, counts_img, bkg_img, exp_img, min_counts=None):
        if bkg_img is None:
            bkg_img_data = np.zeros(np.shape(counts_img.data))
            bkg_norm_factor = 1
        else:
            bkg_img_data = bkg_img.data
            if 'BKG_NORM' in bkg_img.hdr:
                bkg_norm_factor = bkg_img.hdr['BKG_NORM']
            else:
                print(InfoMessages('003'))
                bkg_norm_factor = 1

        if exp_img is None:
            exp_img_data = np.ones(np.shape(counts_img.data))
        else:
            exp_img_data = exp_img.data

        bins = []
        i = 1
        while True:
            if len(bins) != 0:
                total_height_bins = bins[-1][0] + bins[-1][1]
                if total_height_bins >= self.height:
                    break
            else:
                total_height_bins = 0
            i = min(i, (self.height - total_height_bins) / 2)
            x0_bin_nonrotated = 0
            if len(bins) == 0:
                y0_bin_nonrotated = -self.height/2 + i
            else:
                y0_bin_nonrotated = -self.height/2 + i + \
                                    bins[-1][0] + bins[-1][1]
            x0_bin, y0_bin = rotate_point(self.x0, self.y0,
                                          x0_bin_nonrotated, y0_bin_nonrotated,
                                          self.angle)
            new_bin = Box(x0_bin, y0_bin, self.width, 2*i, self.angle)
#            print(i)
#            print(x0_bin, y0_bin, self.width, 2*i, self.angle)
            pixels_in_bin = new_bin.interior_pixels()
#            print("pixels in bin: ", pixels_in_bin)
#            print("npix in bins: ", len(pixels_in_bin))
            if not pixels_in_bin:
                break
            net_counts = new_bin.get_bin_vals(counts_img.data, bkg_img_data,
                bkg_norm_factor, exp_img_data, pixels_in_bin, only_counts=True,
                only_net_cts=True)
            if net_counts < min_counts:
                if len(bins) != 0:
                    # If the last bin cannot accumulate the minimum
                    # number of counts and the end of the region has been
                    # reached...
                    total_height_bins = bins[-1][0] + bins[-1][1] + 2*i
                    if total_height_bins >= self.height:
                        bins[-1] = new_bin.update_last_bin(bins[-1],
                            pixels_in_bin)
                        break
                    # ... otherwise increase the width of the bin a little more.
                    else:
                        i += 1
                # If no bins with the minimum number of counts have been found
                # and the end of the bin has been reached, then throw an error
                # message.
                elif 2*i >= self.height:
                    error_message = ErrorMessages('001')
                    raise ValueError(error_message)
                # If the profile is currently empty but the width of the bin
                # is smaller than the width of the region, then just increase
                # the bin width.
                else:
                    i += 1
            # If the minimum number of counts has been reached, then simply
            # add the previously calculated bin values to the surface brightness
            # profile.
            else:
                if len(bins) != 0:
                    bin_radius = bins[-1][0] + bins[-1][1] + \
                        new_bin.height/2
                else:
                    bin_radius = new_bin.height/2
                bins.append((bin_radius, new_bin.height/2, pixels_in_bin))
                i = 1
            #print(i, net_counts)
        return bins
