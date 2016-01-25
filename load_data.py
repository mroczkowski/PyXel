# File: load_data.py
# Author: Georgiana Ogrean
# Created on 03.22.2015
#
# Load counts image, exposure map, and background map.
#

import pyfits
import numpy as np
from box import Box

class Image():
    def __init__(self, filename, ext=0):
        """Return a FITS image and the associated header.

        The image is returned as a numpy array. By default, the first HDU is read.
        A different HDU can be specified with the argument 'ext'. The header of the
        image is modified to remove unncessary keywords such as HISTORY and COMMENT,
        as well as keywords associated with a 3rd and 4th dimension (e.g. NAXIS3,
        NAXIS4).
        """
        img_hdu = pyfits.open(filename)
        self.data = img_hdu[ext].data
        self.hdr = self.clean_header(img_hdu[ext].header)

    def clean_header(self, hdr):
        """Remove unwanted keywords from the image header.

        Deletes from the image header unncessary keywords such as HISTORY and
        COMMENT, as well as keywords associated with a 3rd and 4th dimension
        (e.g. NAXIS3, NAXIS4). Some radio images are 4D, but the 3rd and 4th
        dimensions are not necessary for plotting the brightness and may
        occasionally cause problems with PyFITS routines.
        """
        forbidden_keywords = {'HISTORY', 'COMMENT', 'NAXIS3', 'NAXIS4',
            'CTYPE3', 'CTYPE4', 'CRVAL3', 'CRVAL4', 'CDELT3', 'CDELT4',
            'CRPIX3', 'CRPIX4', 'CUNIT3', 'CUNIT4'}
        existing_keywords = [key for key in forbidden_keywords if key in hdr]
        if any(existing_keywords):
            for key in existing_keywords:
                del hdr[key]
        return hdr

def read_shape(data):
    """Get region shape and parameters.

    Splits the DS9 region definition into a string describing the region shape
    (e.g. box, circle) and a list of floats containing the parameters of the
    region.
    """
    reg_def = data[3].strip()
    shape = reg_def.split(r'(')[0]
    params = [float(i) for i in reg_def.split(r'(')[1].strip(')').split(',')]
    return (shape, params)

def load_region(filename):
    """Read DS9 region file.

    Reads a DS9 region file in which the region is defined in image coordinates.
    The region file should contain a single region. Compound regions are not
    supported currently.
    """
    try:
        reg_file = open(filename)
    except IOError:
        print('Cannot open file %s' % filename)
    else:
        regions = {'box': Box.from_params}
        data = reg_file.readlines()
        if len(data) != 4 or data[2].strip() != 'image':
            error_message = """Currently only region files with one
                region defined in image coordinates are supported."""
            raise ValueError(error_message)
        else:
            shape, params = read_shape(data)
            region = regions[shape](params)
            return region
