from astropy.io import fits

def clean_header(hdr):
    """Remove unwanted keywords from the image header.

    Deletes from the image header unncessary keywords such as HISTORY and
    COMMENT, as well as keywords associated with a 3rd and 4th dimension
    (e.g. NAXIS3, NAXIS4). Some radio images are 4D, but the 3rd and 4th
    dimensions are not necessary for plotting the brightness and may
    occasionally cause problems.
    """
    forbidden_keywords = {'HISTORY', 'COMMENT', 'NAXIS3', 'NAXIS4',
        'CTYPE3', 'CTYPE4', 'CRVAL3', 'CRVAL4', 'CDELT3', 'CDELT4',
        'CRPIX3', 'CRPIX4', 'CUNIT3', 'CUNIT4'}
    existing_keywords = [key for key in forbidden_keywords if key in hdr]
    if any(existing_keywords):
        for key in existing_keywords:
            del hdr[key]
    return hdr

class Image():
    def __init__(self, filename, ext=0):
        """Return a FITS image and the associated header.

        The image is returned as a numpy array. By default, the first HDU is read.
        A different HDU can be specified with the argument 'ext'. The header of the
        image is modified to remove unncessary keywords such as HISTORY and COMMENT,
        as well as keywords associated with a 3rd and 4th dimension (e.g. NAXIS3,
        NAXIS4).
        """
        if not isinstance(filename, list):
            img_hdu = fits.open(filename)
            self.data = img_hdu[ext].data
            self.hdr = clean_header(img_hdu[ext].header)
        else:
            img_hdr = []
            img_data = []
            if ext == 0:
                ext = [ext] * len(filename)
            elif len(ext) != len(filename):
                raise IndexError('Length of the extension array must match \
                    number of images.')
            for i in range(len(filename)):
                img_hdu = fits.open(filename[i])
                img_data.append(img_hdu[ext[i]].data)
                img_hdr.append(clean_header(img_hdu[ext[i]].header))
            self.data = img_data
            self.hdr = img_hdr


