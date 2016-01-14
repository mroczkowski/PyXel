# File: profile.py
# Author: Georgiana Ogrean
# Created on 06.05.2015
#
# Create surface brightness profile.
#

import pyfits
import numpy as np
from matplotlib.path import Path

# If the region is a polygon, get the coordinates of the corners.
# This needs to be done separately for box and polygon regions because of
# the way they are defined in DS9.
def get_corners(region):
    region_shape = region[0]
    if region_shape == 'box':
        x0 = region[1][0]
        y0 = region[1][1]
        width = region[1][2]
        height = region[1][3]
        angle = -region[1][4] / 180. * np.pi
        corners = [[ - height/2, - width/2], \
                   [ height/2, - width/2], \
                   [ height/2, width/2], \
                   [ - height/2, width/2]]
        for i in range(4):
            x = corners[i][0]
            y = corners[i][1]
            corners[i][0] = np.cos(angle) * x - np.sin(angle) * y + y0 - 1
            corners[i][1] = np.sin(angle) * x + np.cos(angle) * y + x0 - 1
    return corners

# Return an array containing the coordinates of the points inside
# the region of interest.
# Should points on the edges be returned as interior points? There could be
# two adjacent regions that would then have common pixels.
def interior_pixels(img, region):
    region_shape = region[0]
    img_width, img_height = np.shape(img)
    if region_shape == 'box' or region_shape == 'polygon':
        corners = get_corners(region)
        reg_path = Path(corners)
        pixels = []
        for i in range(img_height):
            for j in range(img_width):
                if reg_path.contains_point((i,j)):
                    pixels.append([i,j])
    return pixels

# Create "unbinned" profiles.
def profile(counts_img, bkg_img, exp_img, region, min_counts=None):
    region_shape = region[0]
    [x0, y0, width, height, angle] = region[1]
    if region_shape == 'box' and min_counts is not None:
        i = 1
        while True:
            x0_bin = np.cos(angle) * (-height/2 + i) + y0 - 1
            y0_bin = np.sin(angle) * (-height/2 + i) + x0 - 1
            bin_region = ['box', [x0_bin, y0_bin, 2*i, height, angle]]
            pixels_in_bin = interior_pixels(img, bin_region)
            src_counts = 0
            bkg_counts = 0
            if bkg_img is not None:
                for pixel in pixels_in_bin:
                    src_counts += counts_img[pixel]
                    bkg_counts += bkg_img[pixel]
            else:
                for pixel in pixels_in_bin:
                    src_counts += counts_img[pixel]
            net_counts = src_counts - bkg_counts
            if net_counts < min_counts:
                i += 2
                if len(bin_r) != 0 and bin_r[-1] + bin_width[-1] + 2*i >= width:
                    
            else if len(r) >= 2 and r[len(r)] + 2*i:
                profile.append([r, net_counts])
                break
