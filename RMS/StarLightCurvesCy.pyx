# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
""" Photometry kernel of RMS.StarLightCurves: per frame and star, the linear sum over a square box
at the star's fractional position and the median of the ring pixels around its rounded position.
"""

import numpy as np
cimport numpy as np
from libc.math cimport floor


cdef inline double median8(double* v) nogil:
    """ Median of 8 values (mean of the 4th and 5th smallest, as np.median), sorts v in place. """
    cdef int i, j
    cdef double t
    for i in range(1, 8):
        t = v[i]
        j = i - 1
        while (j >= 0) and (v[j] > t):
            v[j + 1] = v[j]
            j -= 1
        v[j + 1] = t
    return 0.5*(v[3] + v[4])


def boxPhotometry(const unsigned char[:, :, ::1] frames, const float[::1] lut,
                  const double[::1] x_first, const double[::1] x_last,
                  const double[::1] y_first, const double[::1] y_last, const double[::1] frac,
                  int half, const Py_ssize_t[:, ::1] ring):
    """ Box sums, ring medians and saturated-frame counts of moving positions on a frame block.

    Arguments:
        frames: [3D ndarray uint8] (N, H, W) frame block.
        lut: [ndarray float32] (256,) decode table (code -> linear).
        x_first, x_last, y_first, y_last: [ndarray float64] (S,) positions on the first and the
            last frame; frame n is at first + frac[n]*(last - first).
        frac: [ndarray float64] (N,) fraction of the block of each frame.
        half: [int] Box half-width; the box is 2*half + 1 px wide, centred on the position.
        ring: [ndarray intp] (8, 2) ring offsets (dy, dx) around the rounded position.

    Return:
        (flux, bg, n_sat): flux [N, S] float32 box sum of linear values (pixel weights = overlap of
            each pixel with the box), bg [N, S] float32 ring median, n_sat [S] uint16 frames with a
            code-255 pixel of non-zero weight in the box. Positions must keep the box and the ring
            inside the frame (the caller's edge margin).
    """

    cdef Py_ssize_t n_frames = frames.shape[0]
    cdef Py_ssize_t n_stars = x_first.shape[0]
    cdef Py_ssize_t n, s, i, j, k
    cdef int size = 2*half + 2

    if ring.shape[0] != 8:
        raise ValueError("the ring must have 8 offsets")

    flux_arr = np.empty((n_frames, n_stars), dtype=np.float32)
    bg_arr = np.empty((n_frames, n_stars), dtype=np.float32)
    nsat_arr = np.zeros(n_stars, dtype=np.uint16)
    cdef float[:, ::1] flux = flux_arr
    cdef float[:, ::1] bg = bg_arr
    cdef unsigned short[::1] nsat = nsat_arr

    cdef double xc, yc, ux, uy, wy, w, acc
    cdef Py_ssize_t ix0, iy0, xi, yi
    cdef double wx[64]
    cdef double rv[8]
    cdef bint sat
    cdef unsigned char c
    cdef const unsigned char* row

    if size > 64:
        raise ValueError("box too large")

    with nogil:
        for n in range(n_frames):
            for s in range(n_stars):

                xc = x_first[s] + frac[n]*(x_last[s] - x_first[s])
                yc = y_first[s] + frac[n]*(y_last[s] - y_first[s])

                # The box [c - half - 0.5, c + half + 0.5] covers size pixels from i0: the first by
                # 1 - u, the last by u, the others fully
                ix0 = <Py_ssize_t>floor(xc - half)
                ux = xc - half - ix0
                iy0 = <Py_ssize_t>floor(yc - half)
                uy = yc - half - iy0

                wx[0] = 1.0 - ux
                for i in range(1, size - 1):
                    wx[i] = 1.0
                wx[size - 1] = ux

                acc = 0.0
                sat = False
                for j in range(size):
                    if j == 0:
                        wy = 1.0 - uy
                    elif j == size - 1:
                        wy = uy
                    else:
                        wy = 1.0
                    if wy == 0.0:
                        continue
                    row = &frames[n, iy0 + j, ix0]
                    for i in range(size):
                        w = wy*wx[i]
                        if w == 0.0:
                            continue
                        c = row[i]
                        acc += w*lut[c]
                        if c == 255:
                            sat = True

                flux[n, s] = <float>acc
                if sat:
                    nsat[s] += 1

                xi = <Py_ssize_t>floor(xc + 0.5)
                yi = <Py_ssize_t>floor(yc + 0.5)
                for k in range(8):
                    rv[k] = lut[frames[n, yi + ring[k, 0], xi + ring[k, 1]]]
                bg[n, s] = <float>median8(rv)

    return flux_arr, bg_arr, nsat_arr
