from .image import Image
from .box import Box
from .epanda import Epanda

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
    reg_file = open(filename)
    regions = {'box': Box.from_params,
               'epanda': Epanda.from_epanda_params,
               'panda': Epanda.from_panda_params,
               'circle': Epanda.from_circle_params,
               'ellipse': Epanda.from_ellipse_params}
    data = reg_file.readlines()
    if len(data) != 4 or data[2].strip() != 'image':
        error_message = ErrorMessages('002')
        raise ValueError(error_message)
    else:
        shape, params = read_shape(data)
        region = regions[shape](params)
        print("Region loaded. Its shape and parameters are listed below: ")
        print(shape, params)
        return region
