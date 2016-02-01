import numpy as np
import sys

def const(constant=1e-2):
    return constant

def beta(x, beta=0.7, rc=1.0, s0=1e-2):
    return s0 * (1 + (x/rc)**2)**(0.5 - 3.*beta)
