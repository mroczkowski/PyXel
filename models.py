def const(c):
    """Fit a constant to the data."""
    return c

def beta(r, beta=0.7, rc=1., s0=1e-2, bkg=1e-4):
   """Fit beta-model to the data."""
   mod = bkg + s0 * (1. + (r/rc)**2.) ** (-3.*beta+0.5)
   return mod
