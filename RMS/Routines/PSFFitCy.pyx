# cython: language_level=3
""" Compiled model for fitting a 2D Gaussian to star image cutouts (see RMS.ExtractStars.fitPSF). """

import numpy as np

cimport cython
from libc.math cimport cos, exp, fabs, sin


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def twoDGaussianResidual(double[::1] params, double[::1] x, double[::1] y, double saturation,
    double[::1] data):
    """ Residual of the 2D Gaussian (the same model as RMS.Math.twoDGaussian, limited to the saturation level)
        minus the data, for scipy.optimize.leastsq. The model is evaluated in compiled code, which is several
        times faster than with NumPy on the small cutouts of single stars.

    Arguments:
        params: [ndarray] amplitude, xo, yo, sigma_x, sigma_y, theta (radians), offset.
        x: [ndarray] First coordinate of every pixel of the cutout (float64, flattened).
        y: [ndarray] Second coordinate of every pixel of the cutout (float64, flattened).
        saturation: [float] Saturation level, the model is limited to it.
        data: [ndarray] Pixel values of the cutout (float64, flattened).

    Return:
        [ndarray] Model minus data for every pixel.
    """

    cdef double amplitude = fabs(params[0])
    cdef double xo = params[1]
    cdef double yo = params[2]
    cdef double sigma_x = fabs(params[3])
    cdef double sigma_y = fabs(params[4])
    cdef double theta = params[5]
    cdef double offset = params[6]

    cdef double cos_t = cos(theta)
    cdef double sin_t = sin(theta)
    cdef double sin_2t = sin(2*theta)

    cdef double a = (cos_t**2)/(2*sigma_x**2) + (sin_t**2)/(2*sigma_y**2)
    cdef double b = -sin_2t/(4*sigma_x**2) + sin_2t/(4*sigma_y**2)
    cdef double c = (sin_t**2)/(2*sigma_x**2) + (cos_t**2)/(2*sigma_y**2)

    cdef Py_ssize_t i
    cdef Py_ssize_t n = x.shape[0]
    cdef double dx, dy, g

    residual = np.empty(n, dtype=np.float64)
    cdef double[::1] res = residual

    for i in range(n):

        dx = x[i] - xo
        dy = y[i] - yo

        g = offset + amplitude*exp(-(a*(dx*dx) + 2*b*dx*dy + c*(dy*dy)))

        # Limit values to saturation level
        if g > saturation:
            g = saturation

        res[i] = g - data[i]

    return residual
