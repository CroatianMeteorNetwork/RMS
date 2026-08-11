
import numpy as np
import time

# Cython import
cimport numpy as np
cimport cython

# Define numpy types
INT16_TYPE = np.uint16
ctypedef np.uint16_t INT16_TYPE_t

INT64_TYPE = np.uint64
ctypedef np.uint64_t INT64_TYPE_t

FLOAT_TYPE = np.float64
ctypedef np.float64_t FLOAT_TYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef class FFMimickInterface:
    cdef public int nrows, ncols, nframes
    cdef public object dtype
    cdef public double gamma
    cdef public np.npy_bool calibrated, successful
    cdef public np.ndarray maxpixel, minpixel, acc, stdpixel, avepixel

    # Full-precision average in 8.8 fixed point (uint16, units of 1/256 ADU), matching the
    # avepixel16 plane of FF files, so consumers can use the same attribute for both. Only set
    # for 8-bit content (wider content does not fit the fixed point), and holds the RAW average -
    # preprocessFF calibrates only avepixel/maxpixel, so consumers of avepixel16 must apply
    # dark/flat themselves (as the existing ones do)
    cdef public object avepixel16

    cdef np.ndarray acc_lin, decode_lut
    cdef bint use_gamma
    cdef double wp

    def __init__(self, nrows, ncols, dtype, gamma=1.0, bit_depth=None):
        """ Structure which is used to make FF file format data. It mimicks the interface of an FF structure.

        Arguments:
            nrows: [int] Number of image rows.
            ncols: [int] Number of image columns.
            dtype: [data-type] Output dtype of the FF planes. May be updated after construction
                (it only affects the final cast).

        Keyword arguments:
            gamma: [float] Camera gamma. If not 1, the average is computed in the linear domain
                (decode, average, re-encode), which removes the Jensen bias of averaging
                gamma-encoded samples. The planes stay in the gamma-encoded domain.
            bit_depth: [int] Bit depth of the camera encoding, used as the white point of the
                gamma decode and the fixed-point gate. If None, it is derived from dtype - but
                note some callers only know the final dtype after the first frame, while the
                camera bit depth is known upfront, so passing it explicitly is preferred.
        """

        # Init the empty structures
        cdef np.ndarray[INT16_TYPE_t, ndim=2] maxpixel = np.zeros(shape=(nrows, ncols), \
            dtype=INT16_TYPE)
        cdef np.ndarray[INT16_TYPE_t, ndim=2] minpixel = np.full(shape=(nrows, ncols), \
            fill_value=65535, dtype=INT16_TYPE)
        cdef np.ndarray[INT64_TYPE_t, ndim=2] acc = np.zeros(shape=(nrows, ncols), \
            dtype=INT64_TYPE)
        cdef np.ndarray[INT64_TYPE_t, ndim=2] avepixel = np.zeros(shape=(nrows, ncols), \
            dtype=INT64_TYPE)
        cdef np.ndarray[INT64_TYPE_t, ndim=2] stdpixel = np.zeros(shape=(nrows, ncols), \
            dtype=INT64_TYPE)

        self.nrows = nrows
        self.ncols = ncols
        self.dtype = dtype
        self.nframes = 0

        # White point of the content, used for the gamma decode/encode and the fixed-point gate
        if bit_depth is not None:
            self.wp = float(2**bit_depth - 1)
        elif np.issubdtype(np.dtype(dtype), np.integer):
            self.wp = float(np.iinfo(dtype).max)
        else:
            self.wp = 255.0

        self.gamma = gamma
        self.use_gamma = (gamma != 1.0)

        if self.use_gamma:

            # Decode LUT over the full uint16 input range, and the linear-domain accumulator
            self.decode_lut = self.wp*(np.arange(65536)/self.wp)**(1.0/gamma)
            self.acc_lin = np.zeros(shape=(nrows, ncols), dtype=FLOAT_TYPE)

        else:

            # Unused placeholders (frameProc never indexes them when use_gamma is False)
            self.decode_lut = np.zeros(1, dtype=FLOAT_TYPE)
            self.acc_lin = np.zeros(shape=(1, 1), dtype=FLOAT_TYPE)

        # False if dark and flat weren't applied, True otherwise (False be default)
        self.calibrated = False

        # Flag to inicate if making the FF was success or not
        self.successful = False

        self.maxpixel = maxpixel
        self.minpixel = minpixel
        self.acc = acc
        self.avepixel = avepixel
        self.stdpixel = stdpixel
        self.avepixel16 = None


    cpdef addFrame(self, np.ndarray[INT16_TYPE_t, ndim=2] frame):
        """ Add raw frame for computation of FF data. """

        self.frameProc(frame, self.maxpixel, self.minpixel, self.acc, self.stdpixel, \
            self.decode_lut)

        self.nframes += 1


    cdef frameProc(self, np.ndarray[INT16_TYPE_t, ndim=2] frame, \
        np.ndarray[INT16_TYPE_t, ndim=2] maxpixel, np.ndarray[INT16_TYPE_t, ndim=2] minpixel, \
        np.ndarray[INT64_TYPE_t, ndim=2] acc, np.ndarray[INT64_TYPE_t, ndim=2] stdpixel, \
        np.ndarray[FLOAT_TYPE_t, ndim=1] lut):

        cdef int val
        cdef int i, j
        cdef int nrows, ncols
        cdef bint use_gamma = self.use_gamma
        cdef np.ndarray[FLOAT_TYPE_t, ndim=2] acc_lin = self.acc_lin
        nrows = self.nrows
        ncols = self.ncols

        for i in range(nrows):
            for j in range(ncols):

                val = <long> frame[i, j]

                # Track the extreme values (both are trimmed from the average - a one-sided max
                # trim biases the mean low)
                if val > maxpixel[i, j]:
                    maxpixel[i, j] = val

                if val < minpixel[i, j]:
                    minpixel[i, j] = val

                acc[i, j] += val
                stdpixel[i, j] += val*val

                # Accumulate the linear-domain (gamma-decoded) sum
                if use_gamma:
                    acc_lin[i, j] += lut[val]


    cpdef finish(self):
        """ Finish making an FF structure. """

        # If there are less than 4 frames, don't trim the extremes from the average (the trimmed
        # variance divisor would be zero)
        if self.nframes < 4:

            # Compute normal average, rounded
            self.avepixel = (2*self.acc + self.nframes)//(2*self.nframes)

            # Don't compute the standard deviation
            self.stdpixel *= 0


        else:

            # Number of frames in the trimmed sample
            n = self.nframes - 2

            # Remove the contribution of the extreme frames (symmetric trim: the max suppresses
            # meteors and wakes, and the min balances the trim - a one-sided trim biases the
            # mean low by ~0.04 sigma for symmetric noise)
            self.acc -= self.maxpixel
            self.acc -= self.minpixel

            if self.use_gamma:

                # Average in the linear domain and re-encode into the gamma domain the planes
                # are stored in (the trim removes the same frames in both domains - the decode
                # is monotone)
                acc_lin = (self.acc_lin - self.decode_lut[self.maxpixel]
                    - self.decode_lut[self.minpixel])
                enc_mean = self.wp*((acc_lin/n)/self.wp)**self.gamma

            else:
                enc_mean = None

            # Full-precision average in 8.8 fixed point, matching the FF avepixel16 plane. Only
            # for 8-bit content - wider content (16-bit cameras, sum-binned frames) does not
            # fit uint16 fixed point, so check the actual data range as well
            if (self.wp <= 255) and (int(self.maxpixel.max()) <= 255):

                if enc_mean is None:
                    ave16 = (256*self.acc + n//2)//n
                else:
                    ave16 = np.floor(256.0*enc_mean + 0.5).astype(np.uint64)

                self.avepixel16 = ave16.astype(np.uint16)

                # The integer average is derived from the fixed point one, the same way FF
                # readers derive the 8-bit view
                self.avepixel = (ave16 + 128) >> 8

            else:

                if enc_mean is None:
                    self.avepixel = (2*self.acc + n)//(2*n)
                else:
                    self.avepixel = np.rint(enc_mean)

            # Compute the trimmed standard deviation (encoded domain) with the correct sample
            # variance formula, in double precision. Using the truncated integer average in the
            # acc**2/n term (as done previously) inflates the variance by ~mean*frac(mean)
            var = (self.stdpixel.astype(np.float64)
                - (self.maxpixel.astype(np.float64))**2
                - (self.minpixel.astype(np.float64))**2
                - (self.acc.astype(np.float64))**2/n)/(n - 1)

            # Guard against small negative values from floating point rounding
            np.clip(var, 0, None, out=var)

            # Rounded standard deviation
            self.stdpixel = np.floor(np.sqrt(var) + 0.5).astype(np.uint64)


        # Make sure there are no zeros in standard deviation
        self.stdpixel[self.stdpixel == 0] = 1

        # Convert frames to the appropriate format
        self.maxpixel = self.maxpixel.astype(self.dtype)
        self.avepixel = self.avepixel.astype(self.dtype)
        self.stdpixel = self.stdpixel.astype(self.dtype)

        self.successful = True

        return True