""" Cythonized functions for saturation correction. """


import numpy as np

# Cython import
cimport numpy as np
cimport cython

from libc.math cimport floor, ceil, exp

# Define numpy types
INT_TYPE = np.uint32
ctypedef np.uint32_t INT_TYPE_t

FLOAT_TYPE = np.float64 
ctypedef np.float64_t FLOAT_TYPE_t





@cython.boundscheck(False)
@cython.wraparound(False) 
@cython.cdivision(True)
def addGaussian(np.ndarray[FLOAT_TYPE_t, ndim=2] frame, int x, int y, float amp, float sigma, int window):

    cdef int x_min, x_max, y_min, y_max, xp, yp
    cdef float r_sq, val


    # Compute the limits of the window for adding the Gaussian to the frame

    x_min = int(floor(x - window))
    if x_min < 0:
        x_min = 0

    x_max = int(ceil(x + window))
    if x_max > frame.shape[1] - 1:
        x_max = int(frame.shape[1] - 1)

    y_min = int(floor(y - window))
    if y_min < 0:
        y_min = 0

    y_max = int(ceil(y + window))
    if y_max > frame.shape[0] - 1:
        y_max = int(frame.shape[0] - 1)


    # Add the value of the Gaussian for every pixel on the frame
    for xp in range(y_min, y_max):
        for yp in range(x_min, x_max):

            # Compute the squared radius from the centre of the gaussian
            r_sq = float((xp - x)**2 + (yp - y)**2)

            # Compute the value of the Gaussian
            val = amp*exp(-0.5*r_sq/(sigma**2))

            frame[yp, xp] += val

    return frame
