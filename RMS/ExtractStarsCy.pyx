#!python
#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: initializedcheck=False

"""
Cythonized functions for fast star extraction and PSF fitting.

This module provides optimized implementations of computationally intensive
functions used during star detection and PSF fitting.
"""

import numpy as np
cimport numpy as np
cimport cython

# Initialize NumPy C API (required for NumPy 2.0+)
np.import_array()

# Define numpy types
FLOAT_TYPE = np.float64
ctypedef np.float64_t FLOAT_TYPE_t

# Declare math functions
cdef extern from "math.h":
    double fabs(double)
    double sin(double)
    double cos(double)
    double exp(double)
    double sqrt(double)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double gaussian2d_point(double x, double y, double amplitude, double xo, double yo,
                                     double sigma_x, double sigma_y, double theta,
                                     double offset, double saturation):
    """Evaluate 2D Gaussian at a single point (x, y).

    This is an optimized C-level function that can be called without the GIL.

    Arguments:
        x, y: Point coordinates
        amplitude: Gaussian amplitude
        xo, yo: Gaussian center
        sigma_x, sigma_y: Standard deviations
        theta: Rotation angle in radians
        offset: Background offset
        saturation: Saturation level

    Returns:
        Value of Gaussian at (x, y), clipped to saturation
    """
    cdef double a, b, c, dx, dy, value
    cdef double cos_theta, sin_theta, cos_2theta, sin_2theta

    # Ensure positive sigma values
    if sigma_x <= 0:
        sigma_x = 1e-10
    if sigma_y <= 0:
        sigma_y = 1e-10

    # Precompute trig values
    cos_theta = cos(theta)
    sin_theta = sin(theta)
    cos_2theta = cos(2.0 * theta)
    sin_2theta = sin(2.0 * theta)

    # Compute rotation matrix coefficients
    a = (cos_theta * cos_theta) / (2.0 * sigma_x * sigma_x) + \
        (sin_theta * sin_theta) / (2.0 * sigma_y * sigma_y)

    b = -(sin_2theta) / (4.0 * sigma_x * sigma_x) + \
        (sin_2theta) / (4.0 * sigma_y * sigma_y)

    c = (sin_theta * sin_theta) / (2.0 * sigma_x * sigma_x) + \
        (cos_theta * cos_theta) / (2.0 * sigma_y * sigma_y)

    # Compute offset from center
    dx = x - xo
    dy = y - yo

    # Evaluate Gaussian
    value = offset + fabs(amplitude) * exp(-(a * dx * dx + 2.0 * b * dx * dy + c * dy * dy))

    # Clip to saturation
    if value > saturation:
        value = saturation

    return value


def twoDGaussian(params, double amplitude, double xo, double yo,
                 double sigma_x, double sigma_y, double theta, double offset):
    """Fast 2D Gaussian function compatible with scipy.optimize.curve_fit.

    This is a drop-in replacement for RMS.Math.twoDGaussian that uses
    Cython for ~10-20x speedup.

    Arguments:
        params: Tuple or array-like of (x_indices, y_indices, saturation)
            - x, y: 2D arrays of coordinates
            - saturation: Scalar or array for saturation level
        amplitude: Gaussian amplitude
        xo, yo: Gaussian center coordinates
        sigma_x, sigma_y: Standard deviations in x and y
        theta: Rotation angle in radians
        offset: Background offset

    Returns:
        Raveled array of Gaussian values at each (x, y) point
    """
    cdef np.ndarray[FLOAT_TYPE_t, ndim=2] x_arr, y_arr
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] result
    cdef double saturation_val
    cdef int i, j, nrows, ncols, idx

    # Handle params as tuple, list, or array
    # scipy.curve_fit sometimes passes params as ndarray during optimization
    if isinstance(params, tuple) or isinstance(params, list):
        x_arr = np.asarray(params[0], dtype=FLOAT_TYPE)
        y_arr = np.asarray(params[1], dtype=FLOAT_TYPE)
        saturation_param = params[2]
    else:
        # params is array-like, treat as having 3 elements
        x_arr = np.asarray(params[0], dtype=FLOAT_TYPE)
        y_arr = np.asarray(params[1], dtype=FLOAT_TYPE)
        saturation_param = params[2]

    # Handle saturation
    if isinstance(saturation_param, np.ndarray):
        # Take first element if it's an array
        saturation_val = float(saturation_param.flat[0])
    else:
        saturation_val = float(saturation_param)

    # Get dimensions
    nrows = x_arr.shape[0]
    ncols = x_arr.shape[1]

    # Allocate output
    result = np.empty(nrows * ncols, dtype=FLOAT_TYPE)

    # Evaluate Gaussian at each point
    idx = 0
    for i in range(nrows):
        for j in range(ncols):
            result[idx] = gaussian2d_point(
                x_arr[i, j], y_arr[i, j],
                amplitude, xo, yo, sigma_x, sigma_y, theta, offset,
                saturation_val
            )
            idx += 1

    return result


def twoDGaussianCircular(params, double amplitude, double xo, double yo,
                          double sigma, double offset):
    """Fast circular (symmetric) 2D Gaussian for round stars.

    This is faster than the full elliptical Gaussian as it has fewer parameters
    and simpler computation (no rotation).

    Arguments:
        params: Tuple or array-like of (x_indices, y_indices, saturation)
        amplitude: Gaussian amplitude
        xo, yo: Gaussian center coordinates
        sigma: Standard deviation (same in x and y)
        offset: Background offset

    Returns:
        Raveled array of Gaussian values
    """
    cdef np.ndarray[FLOAT_TYPE_t, ndim=2] x_arr, y_arr
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] result
    cdef double saturation_val, dx, dy, r_sq, value
    cdef int i, j, nrows, ncols, idx
    cdef double sigma_sq, amp_abs

    # Handle params as tuple, list, or array
    if isinstance(params, tuple) or isinstance(params, list):
        x_arr = np.asarray(params[0], dtype=FLOAT_TYPE)
        y_arr = np.asarray(params[1], dtype=FLOAT_TYPE)
        saturation_param = params[2]
    else:
        x_arr = np.asarray(params[0], dtype=FLOAT_TYPE)
        y_arr = np.asarray(params[1], dtype=FLOAT_TYPE)
        saturation_param = params[2]

    # Handle saturation
    if isinstance(saturation_param, np.ndarray):
        saturation_val = float(saturation_param.flat[0])
    else:
        saturation_val = float(saturation_param)

    # Get dimensions
    nrows = x_arr.shape[0]
    ncols = x_arr.shape[1]

    # Precompute
    amp_abs = fabs(amplitude)
    if sigma <= 0:
        sigma = 1e-10
    sigma_sq = 2.0 * sigma * sigma

    # Allocate output
    result = np.empty(nrows * ncols, dtype=FLOAT_TYPE)

    # Evaluate circular Gaussian at each point
    idx = 0
    for i in range(nrows):
        for j in range(ncols):
            dx = x_arr[i, j] - xo
            dy = y_arr[i, j] - yo
            r_sq = dx * dx + dy * dy
            value = offset + amp_abs * exp(-r_sq / sigma_sq)

            # Clip to saturation
            if value > saturation_val:
                value = saturation_val

            result[idx] = value
            idx += 1

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def computeIntensity(np.ndarray[FLOAT_TYPE_t, ndim=2] star_seg,
                     double background, double gamma):
    """Compute gamma-corrected intensity of a star segment.

    This function applies gamma correction and computes the total
    intensity by subtracting background and summing pixels.

    Arguments:
        star_seg: 2D array of star segment pixels
        background: Background level to subtract
        gamma: Gamma correction value

    Returns:
        Total gamma-corrected intensity
    """
    cdef int i, j, nrows, ncols
    cdef double intensity, bg_corrected, pixel_val
    cdef double gamma_inv

    nrows = star_seg.shape[0]
    ncols = star_seg.shape[1]

    intensity = 0.0
    gamma_inv = 1.0 / gamma if gamma > 0 else 1.0

    # Gamma correct background
    if background > 0:
        bg_corrected = background ** gamma_inv
    else:
        bg_corrected = 0.0

    # Apply gamma correction and sum
    for i in range(nrows):
        for j in range(ncols):
            pixel_val = star_seg[i, j]
            if pixel_val > 0:
                intensity += pixel_val ** gamma_inv - bg_corrected
            else:
                intensity -= bg_corrected

    return intensity
