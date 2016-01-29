import numpy as np
from model_defs import const, beta

class Constant(Model):
    def __init__(self, **kwargs):
        super(Constant, self).__init__(const, **kwargs)

    def set_params(self, **kwargs):
        params = {}
        for key, value in kwargs.items():
            


class BetaModel(Model):
    def __init__(self, **kwargs):
        super(BetaModel, self).__init__(beta, **kwargs)
