import pyfits
import numpy as np
from load_data import Image, load_region
import matplotlib.pyplot as plt
from model import Model
#from profile import Box #, counts_profile

src_img = Image("srcfree_bin4_500-4000_band1_thresh.img")
bkg_img = Image("srcfree_bin4_500-4000_bgstow_renorm.img")
exp_img = Image("srcfree_bin4_500-4000_thresh.expmap_nosrcedg")

region = load_region("beta.reg")

#pixels = region.interior_pixels()
#for pixel in pixels:
#    img[pixel[0], pixel[1]] = 2.
#
#pyfits.writeto('test.fits', img, header=hdr, clobber=True)

# # Plot counts profile.
# p = region.sb_profile(src_img, bkg_img, exp_img, min_counts=50)
#
# guess = [0.6, 0.5, 1e-2, 1e-6]
# model = Model("beta").fit(p, guess, method='leastsq')
# region.plot_profile(p, xlog=True, ylog=True, \
#     with_model=True, model_name="beta", model_params=model, \
#     xlabel=r"Distance (arcmin)", \
#     ylabel=r"photons s$^{-1}$ cm$^{-2}$ pixel$^{-1}$")
#
#
# mod = Model("beta") + Model("const")
# params = mod.set_params(beta=0.7, rc=0.4, s0=1e-4, const=3e-6)

mod = Beta()

mod = Model('Beta')
mod.show_params()
mod.set_param('beta', 0.8, frozen=True, min=0.0)
mod.show_params()

const = Constant()
const.set_param(...)

beta = Beta()

const + beta --> const.__add__(beta)

class ModelParameter:
    def __init__(self, name, default_value):
        self.name = name
        ...

class Parameter:
    def __init__(self, model_parameter):
        self.val = model_parameter.default_value
        self.frozen = False
        self.min = -inf
        self.max = +inf

class Model:
    def __init__(self, params):
        self.params = {x.name: Parameter(x) for x in params}
        self.params['s0'].min

    def set_parameter(self, name, value, frozen=None, min=None, max=None):
        self.params[name].val = value

    def __add__(self, other):
        return AdditiveModel(self, other)

mod = Beta()
mod.set_parameter('beta', 0.5)

class Beta(Model):
    def __init__(self):
        s0 = ModelParameter('s0', 1.2)
        Model.__init__(self, [s0])

    def evaluate(x):

class AdditiveModel(Model):
    def __init__(self, model1, model2)

    def evaluate(x):
        return self.model1.evaluate(x) + self.model2.evaluate(x)
