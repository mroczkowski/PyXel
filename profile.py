# File: profile.py
# Author: Georgiana Ogrean
# Created on 06.05.2015
#
# Create surface brightness profile.
#

from math import floor, ceil
import pyfits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from SurfMessages import ErrorMessages

def rotate_point(x0, y0, x, y, angle):
    """Rotate point (x,y) counter-clockwise around (x0,y0)."""
    x_rot = x0 + x * np.cos(angle) - y * np.sin(angle)
    y_rot = y0 + x * np.sin(angle) + y * np.cos(angle)
    return (x_rot, y_rot)

class Box:
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

    def update_last_bin(self, prev_bin, src_counts, err_src_counts,
        bkg_counts, err_bkg_counts, net_counts, err_net_counts):
        """Update last bin of the profile.

        If the  last bin cannot accumulate enough counts befor the end of the
        region is reached, then the previous bin's radius and width are
        increased to include the remaining area. The counts in the remaining
        area are added to those in the previous bin.
        """
        return (prev_bin[0] + self.height / 2,
               prev_bin[1] + self.height / 2,
               prev_bin[2] + src_counts,
               np.sqrt(prev_bin[3]**2 + err_src_counts**2),
               prev_bin[4] + bkg_counts,
               np.sqrt(prev_bin[5]**2 + err_bkg_counts**2),
               prev_bin[6] + net_counts,
               np.sqrt(prev_bin[7]**2 + err_net_counts**2))

    def bin_counts(self, counts_img, bkg_img, exp_img, pixels_in_bin):
        """Calculate the number of counts in a bin."""
        src_counts = 0
        bkg_counts = 0
        for pixel in pixels_in_bin:
            if exp_img[pixel[0], pixel[1]] != 0:
                src_counts += counts_img[pixel[0], pixel[1]]
                bkg_counts += bkg_img[pixel[0], pixel[1]]
#                print(pixel[0], pixel[1], counts_img[pixel[0], pixel[1]])
        err_src_counts = np.sqrt(src_counts)
        err_bkg_counts = np.sqrt(bkg_counts)
        net_counts = src_counts - bkg_counts
        err_net_counts = np.sqrt(err_src_counts**2 + err_bkg_counts**2)
        return src_counts, err_src_counts, \
               bkg_counts, err_bkg_counts, \
               net_counts, err_net_counts

    def count_profile(self, counts_img, bkg_img, exp_img, min_counts=None):
        """Generate count profiles.

        The box is divided into bins based on a minimum number of counts or a
        minimum S/N. The box is divided into bins starting from the bottom up,
        where the bottom is defined as the bin starting at the lowest row in
        the nonrotated box. E.g., if the box is rotated by 135 deg and it
        can be divided into three bins, then the bins will be distributed as:

                                 x
                               x   x
                             x       x
                           x    1st    x
                         x   x           x
                       x       x    bin    x
                     x    2nd    x       x
                   x   x           x   x
                 x       x    bin    x
               x    3rd    x       x
                 x           x   x
                   x    bin    x
                     x       x
                       x   x
                         x

        If a background map is not read in, then the background is set to zero.
        If an exposure map is not read in, then the exposure is set to one
        everywhere. However, if point source exclusion is important, then an
        exposure map is necessary; regions with zero exposure will be ignored,
        so they need to be masked in the exposure map. Removing point sources
        from the exposure map with dmcopy can be very slow if the full field of
        view is not included in the region file. So to quicken things up, the
        region file should look something like:

                field()
                -ellipse(......)
                -circle(......)

        The function returns a list of tuples of the form:
        (bin radius, bin width, source counts, background counts, net counts)
        """
        if bkg_img is None:
            print("hello")
            bkg_img = np.zeros(np.shape(counts_img))
        if exp_img is None:
            exp_img = np.ones(np.shape(counts_img))
        sb_profile = []
        i = 1
        while True:
            if len(sb_profile) != 0:
                total_height_bins = sb_profile[-1][0] + sb_profile[-1][1]
                if total_height_bins >= self.height:
                    break
            else:
                total_height_bins = 0
            i = min(i, (self.height - total_height_bins) / 2)
            x0_bin_nonrotated = 0
            if len(sb_profile) == 0:
                y0_bin_nonrotated = -self.height/2 + i
            else:
                y0_bin_nonrotated = -self.height/2 + i + sb_profile[-1][0] + \
                                    sb_profile[-1][1]
            x0_bin, y0_bin = rotate_point(self.x0, self.y0,
                                          x0_bin_nonrotated, y0_bin_nonrotated,
                                          self.angle)
#            print(i, x0_bin_nonrotated+self.x0, y0_bin_nonrotated+self.y0, x0_bin, y0_bin)
            new_bin = Box(x0_bin, y0_bin, self.width, 2*i, self.angle)
            pixels_in_bin = new_bin.interior_pixels()
            src_counts, err_src_counts, \
                bkg_counts, err_bkg_counts, \
                net_counts, err_net_counts = new_bin.bin_counts(
                counts_img, bkg_img, exp_img, pixels_in_bin)
            if net_counts < min_counts:
                if len(sb_profile) != 0:
                    # If the last bin cannot accumulate the minimum
                    # number of counts and the end of the region has been
                    # reached...
                    total_height_bins = sb_profile[-1][0] + \
                                        sb_profile[-1][1] + 2*i
                    if total_height_bins >= self.height:
#                        print('total...')
                        sb_profile[-1] = new_bin.update_last_bin(
                            sb_profile[-1], src_counts, err_src_counts,
                            bkg_counts, err_bkg_counts,
                            net_counts, err_net_counts)
#                        print(sb_profile[-1])
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
                if len(sb_profile) != 0:
                    bin_radius = sb_profile[-1][0] + sb_profile[-1][1] + \
                        new_bin.height/2
                else:
                    bin_radius = new_bin.height/2
                sb_profile.append((bin_radius, new_bin.height/2, \
                    src_counts, err_src_counts, bkg_counts, err_bkg_counts,
                    net_counts, err_net_counts))
                i = 1
        return sb_profile

    def plot_count_profile(self, cts_profile, xlog=True, ylog=True,
        xlims=None, ylims=None):
        """Plot count profile.

        Plots the net count profile (with error bars) and the background
        profile (step function without error bars - the uncertainties are
        difficult to get from a renormalized background image without knowing
        the normalization). The plotting can easily be done without the use of
        this routine, by just calling count_profile to get the data. This would
        allow for more customization than this routine provides.
        """

        nbins = len(cts_profile)

        r = np.array([cts_profile[i][0] for i in range(nbins)])
        r_err = np.array([cts_profile[i][1] for i in range(nbins)])
        bkg = np.array([cts_profile[i][4] for i in range(nbins)])
        net_cts = np.array([cts_profile[i][6] for i in range(nbins)])
        err_net_cts = np.array([cts_profile[i][7] for i in range(nbins)])

        plt.scatter(r, net_cts, c="black", alpha=0.85, s=35, marker="s")
        plt.errorbar(r, net_cts, xerr=r_err, yerr=err_net_cts,
                     linestyle="None", color="black")
        plt.step(r, bkg, where="mid")

        plt.xlabel("Distance (pixels)")
        plt.ylabel("Counts")

        plt.grid(True)

        if xlims is not None:
            plt.xlim([xlims[0], xlims[1]])
        else:
            plt.xlim([0, np.max(r + r_err)])

        if ylims is not None:
            plt.ylim([ylims[0], ylims[1]])

        if xlog:
            plt.semilogx()

        if ylog:
            plt.semilogy()

        plt.show()
