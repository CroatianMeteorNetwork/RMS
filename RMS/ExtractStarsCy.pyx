#!python
#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: initializedcheck=False

""" Cythonized 2D Gaussian PSF model for fast star extraction.

scipy's curve_fit evaluates the model function hundreds of times per star (it estimates the
Jacobian numerically), so the model evaluation dominates the star extraction run time. This module
provides a C implementation of RMS.Math.twoDGaussian that is a numerical drop-in replacement:
same parameters, same abs() folding of amplitude and sigmas, same saturation clipping, same
raveled float64 output.
"""

import numpy as np
cimport numpy as np
cimport cython

# Initialize NumPy C API
np.import_array()

FLOAT_TYPE = np.float64
ctypedef np.float64_t FLOAT_TYPE_t
ctypedef np.int32_t INT32_TYPE_t

cdef extern from "math.h":
    double fabs(double)
    double sin(double)
    double cos(double)
    double exp(double)


def twoDGaussian(params, double amplitude, double xo, double yo,
                 double sigma_x, double sigma_y, double theta, double offset):
    """ Fast 2D Gaussian, numerically identical to RMS.Math.twoDGaussian.

    Arguments:
        params: [tuple] (x, y, saturation)
            - x, y: [ndarray] 2D coordinate arrays
            - saturation: [scalar or ndarray] saturation level
        amplitude: [float] amplitude of the PSF
        xo: [float] PSF center, X component
        yo: [float] PSF center, Y component
        sigma_x: [float] standard deviation X component
        sigma_y: [float] standard deviation Y component
        theta: [float] PSF rotation in radians
        offset: [float] PSF offset from 0

    Return:
        g: [ndarray] raveled values of the Gaussian at the (x, y) coordinates
    """

    cdef np.ndarray[FLOAT_TYPE_t, ndim=2] x_arr, y_arr
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] result
    cdef double saturation, a, b, c, dx, dy, value
    cdef double cos_theta, sin_theta, sin_2theta
    cdef int i, j, nrows, ncols, idx

    x_arr = np.ascontiguousarray(params[0], dtype=FLOAT_TYPE)
    y_arr = np.ascontiguousarray(params[1], dtype=FLOAT_TYPE)

    # Match RMS.Math.twoDGaussian: saturation may be given as an array
    saturation_param = params[2]
    if isinstance(saturation_param, np.ndarray):
        saturation = float(saturation_param.flat[0])
    else:
        saturation = float(saturation_param)

    # Match the abs() folding of the Python implementation. A zero sigma divides to IEEE inf,
    # giving exp(-inf) = 0, exactly like the numpy implementation.
    amplitude = fabs(amplitude)
    sigma_x = fabs(sigma_x)
    sigma_y = fabs(sigma_y)

    cos_theta = cos(theta)
    sin_theta = sin(theta)
    sin_2theta = sin(2.0*theta)

    a = (cos_theta*cos_theta)/(2.0*sigma_x*sigma_x) + (sin_theta*sin_theta)/(2.0*sigma_y*sigma_y)
    b = -sin_2theta/(4.0*sigma_x*sigma_x) + sin_2theta/(4.0*sigma_y*sigma_y)
    c = (sin_theta*sin_theta)/(2.0*sigma_x*sigma_x) + (cos_theta*cos_theta)/(2.0*sigma_y*sigma_y)

    nrows = x_arr.shape[0]
    ncols = x_arr.shape[1]
    result = np.empty(nrows*ncols, dtype=FLOAT_TYPE)

    idx = 0
    for i in range(nrows):
        for j in range(ncols):
            dx = x_arr[i, j] - xo
            dy = y_arr[i, j] - yo
            value = offset + amplitude*exp(-(a*dx*dx + 2.0*b*dx*dy + c*dy*dy))

            # Limit values to the saturation level
            if value > saturation:
                value = saturation

            result[idx] = value
            idx += 1

    return result


def twoDGaussianResiduals(np.ndarray[FLOAT_TYPE_t, ndim=1] p,
                          np.ndarray[FLOAT_TYPE_t, ndim=2] x_arr,
                          np.ndarray[FLOAT_TYPE_t, ndim=2] y_arr,
                          double saturation,
                          np.ndarray[FLOAT_TYPE_t, ndim=1] data):
    """ Residuals of the 2D Gaussian against the data, for scipy.optimize.leastsq.

    Computing the residuals in C lets leastsq call this function directly, without the per-call
    Python wrapper that curve_fit adds - that wrapper dominates the run time once the model
    itself is fast.

    Arguments:
        p: [ndarray] Parameter vector (amplitude, xo, yo, sigma_x, sigma_y, theta, offset).
        x_arr: [ndarray] 2D X coordinate array.
        y_arr: [ndarray] 2D Y coordinate array.
        saturation: [float] Saturation level.
        data: [ndarray] Raveled measured pixel values.

    Return:
        residuals: [ndarray] model - data, raveled.
    """

    cdef double amplitude = fabs(p[0])
    cdef double xo = p[1]
    cdef double yo = p[2]
    cdef double sigma_x = fabs(p[3])
    cdef double sigma_y = fabs(p[4])
    cdef double theta = p[5]
    cdef double offset = p[6]

    cdef double a, b, c, dx, dy, value
    cdef double cos_theta, sin_theta, sin_2theta
    cdef int i, j, nrows, ncols, idx

    cos_theta = cos(theta)
    sin_theta = sin(theta)
    sin_2theta = sin(2.0*theta)

    a = (cos_theta*cos_theta)/(2.0*sigma_x*sigma_x) + (sin_theta*sin_theta)/(2.0*sigma_y*sigma_y)
    b = -sin_2theta/(4.0*sigma_x*sigma_x) + sin_2theta/(4.0*sigma_y*sigma_y)
    c = (sin_theta*sin_theta)/(2.0*sigma_x*sigma_x) + (cos_theta*cos_theta)/(2.0*sigma_y*sigma_y)

    nrows = x_arr.shape[0]
    ncols = x_arr.shape[1]

    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] result = np.empty(nrows*ncols, dtype=FLOAT_TYPE)

    idx = 0
    for i in range(nrows):
        for j in range(ncols):
            dx = x_arr[i, j] - xo
            dy = y_arr[i, j] - yo
            value = offset + amplitude*exp(-(a*dx*dx + 2.0*b*dx*dy + c*dy*dy))

            # Limit values to the saturation level
            if value > saturation:
                value = saturation

            result[idx] = value - data[idx]
            idx += 1

    return result


def twoDGaussianResidualsEncoded(np.ndarray[FLOAT_TYPE_t, ndim=1] p,
                                 np.ndarray[FLOAT_TYPE_t, ndim=2] x_arr,
                                 np.ndarray[FLOAT_TYPE_t, ndim=2] y_arr,
                                 np.ndarray[FLOAT_TYPE_t, ndim=1] lut,
                                 np.ndarray[INT32_TYPE_t, ndim=1] lut_index,
                                 double index_scale,
                                 np.ndarray[FLOAT_TYPE_t, ndim=1] data):
    """ Residuals of a 2D Gaussian in LINEAR light, encoded through the camera response, against the
        encoded (code) data, for scipy.optimize.leastsq.

    A star is a Gaussian in light, not in the codes of a non-linear camera. The model is therefore
    evaluated in linear units, clipped at the white point and mapped back to a fractional code by
    inverting the decode table, and compared with the measured codes. In the codes the photon noise
    is roughly uniform (gamma ~0.5 stabilises it), so the unweighted residuals stay appropriate.

    Arguments:
        p: [ndarray] Parameter vector (amplitude, xo, yo, sigma_x, sigma_y, theta, offset), with the
            amplitude and offset in LINEAR units (the scale of lut).
        x_arr: [ndarray] 2D X coordinate array.
        y_arr: [ndarray] 2D Y coordinate array.
        lut: [ndarray] Strictly increasing decode table: lut[c] = linear value of code c, so lut[-1]
            is the linear white point.
        lut_index: [ndarray] lut_index[k] = the last code c with lut[c] <= k/index_scale, for k = 0..M
            (see ExtractStars.psfEncodeIndex): it points the inversion at the right table interval,
            so encoding costs a step or two per pixel instead of a binary search.
        index_scale: [float] M/lut[-1].
        data: [ndarray] Raveled measured pixel values (codes, fractional allowed).

    Return:
        residuals: [ndarray] encoded model - data, raveled.
    """

    cdef double amplitude = fabs(p[0])
    cdef double xo = p[1]
    cdef double yo = p[2]
    cdef double sigma_x = fabs(p[3])
    cdef double sigma_y = fabs(p[4])
    cdef double theta = p[5]
    cdef double offset = p[6]

    cdef double a, b, c, dx, dy, value, code, white
    cdef double cos_theta, sin_theta, sin_2theta
    cdef int i, j, nrows, ncols, idx, n, lo, k, m

    cos_theta = cos(theta)
    sin_theta = sin(theta)
    sin_2theta = sin(2.0*theta)

    a = (cos_theta*cos_theta)/(2.0*sigma_x*sigma_x) + (sin_theta*sin_theta)/(2.0*sigma_y*sigma_y)
    b = -sin_2theta/(4.0*sigma_x*sigma_x) + sin_2theta/(4.0*sigma_y*sigma_y)
    c = (sin_theta*sin_theta)/(2.0*sigma_x*sigma_x) + (cos_theta*cos_theta)/(2.0*sigma_y*sigma_y)

    nrows = x_arr.shape[0]
    ncols = x_arr.shape[1]
    n = lut.shape[0]
    m = lut_index.shape[0] - 1
    white = lut[n - 1]

    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] result = np.empty(nrows*ncols, dtype=FLOAT_TYPE)

    idx = 0
    for i in range(nrows):
        for j in range(ncols):
            dx = x_arr[i, j] - xo
            dy = y_arr[i, j] - yo
            value = offset + amplitude*exp(-(a*dx*dx + 2.0*b*dx*dy + c*dy*dy))

            # Encode: invert the decode table (indexed interval + linear interpolation). Above the
            # white point the camera saturates; below the first entry it outputs code 0.
            if value >= white:
                code = n - 1
            elif value <= lut[0]:
                code = 0.0
            else:
                k = <int>(value*index_scale)
                if k > m:
                    k = m
                lo = lut_index[k]
                while lo < n - 2 and lut[lo + 1] <= value:
                    lo += 1
                code = lo + (value - lut[lo])/(lut[lo + 1] - lut[lo])

            result[idx] = code - data[idx]
            idx += 1

    return result
