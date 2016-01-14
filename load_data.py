# File: load_data.py
# Author: Georgiana Ogrean
# Created on 03.22.2015
#
# Load counts image, exposure map, and background map.
#

import pyfits
import numpy as np

# Use pyfits to open hdu.
# Image and header info are then read in load_image.
def open_file(filename, ext=0):
    try:
        img_hdu = pyfits.open(filename)
    except IOError:
        print('Cannot open file %s' %filename)
    else:
        return img_hdu

def load_image(filename, ext=0):
    img_hdu = open_file(filename, ext)
    if img_hdu:
        img = img_hdu[ext].data
        hdr = clean_header(img_hdu[ext].header)
        return img, hdr
    else:
        return None, None

# Remove history and comments.
# Some radio images are 4D. Force them to be 2D by removing
# the keywords associated to the 3rd and 4th axes.
def clean_header(hdr):
    if not hdr:
        return None
    else:
        forbidden_keywords = ['HISTORY', 'COMMENT', 'NAXIS3', 'NAXIS4', \
            'CTYPE3', 'CTYPE4', 'CRVAL3', 'CRVAL4', 'CDELT3', 'CDELT4', \
            'CRPIX3', 'CRPIX4', 'CUNIT3', 'CUNIT4']
        existing_keywords = [key for key in forbidden_keywords if key in hdr]
        if any(existing_keywords):
            for key in existing_keywords:
                del hdr[key]
        return hdr

# Read DS9 region file.
# For now, this only deals with region files that only have
# one (including) region. In the future, the function should be able to handle
# more complex regions, including regions consisting of a combination of
# including and excluding regions.
def load_region(filename):
    try:
        reg_file = open(filename)
    except IOError:
        print('Cannot open file %s' %filename)
    else:
        data = reg_file.readlines()
        if len(data) != 4 or data[2].strip() != 'image':
            error_message = """Currently only region files with one
                region defined in image coordinates are supported."""
            raise ValueError(error_message)
        else:
            reg_def = data[3].strip()
            shape = reg_def.split(r'(')[0]
            params = [float(i) \
                for i in reg_def.split(r'(')[1].strip(')').split(',')]
            return (shape, params)
