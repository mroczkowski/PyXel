import numpy as np

def rotate_point(x0, y0, x, y, angle):
    """Rotate point (x,y) counter-clockwise around (x0,y0)."""
    x_rot = x0 + x * np.cos(angle) - y * np.sin(angle)
    y_rot = y0 + x * np.sin(angle) + y * np.cos(angle)
    return (x_rot, y_rot)

def convert_pix2arcmin(x, pix2arcmin):
    return x * pix2arcmin * 60.
