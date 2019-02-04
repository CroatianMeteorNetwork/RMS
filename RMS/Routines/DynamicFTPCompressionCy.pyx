
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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef class FFMimickInterface:
    cdef public int nrows, ncols, nframes
    cdef public object dtype
    cdef public np.npy_bool calibrated, successful
    cdef public np.ndarray maxpixel, acc, stdpixel, avepixel

    def __init__(self, nrows, ncols, dtype):
        """ Structure which is used to make FF file format data. It mimicks the interface of an FF structure. 
    
        Arguments:
            nrows: [int] Number of image rows.
            ncols: [int] Number of image columns.
        """

        # Init the empty structures
        cdef np.ndarray[INT16_TYPE_t, ndim=2] maxpixel = np.zeros(shape=(nrows, ncols), \
            dtype=INT16_TYPE)
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

        # False if dark and flat weren't applied, True otherwise (False be default)
        self.calibrated = False

        # Flag to inicate if making the FF was success or not
        self.successful = False

        self.maxpixel = maxpixel
        self.acc = acc
        self.avepixel = avepixel
        self.stdpixel = stdpixel


    cpdef addFrame(self, np.ndarray[INT16_TYPE_t, ndim=2] frame):
        """ Add raw frame for computation of FF data. """

        self.maxpixel, self.acc, self.stdpixel = self.frameProc(frame, self.maxpixel, self.acc, self.stdpixel)

        self.nframes += 1


    cdef frameProc(self, np.ndarray[INT16_TYPE_t, ndim=2] frame, np.ndarray[INT16_TYPE_t, ndim=2] maxpixel, np.ndarray[INT64_TYPE_t, ndim=2] acc, np.ndarray[INT64_TYPE_t, ndim=2] stdpixel):

        cdef int val, maxval
        cdef int i, j
        cdef int nrows, ncols
        nrows = self.nrows
        ncols = self.ncols

        for i in range(nrows):
            for j in range(ncols):

                val = <long> frame[i, j]

                # Find the larger values between the current max value and the given frame
                maxval = maxpixel[i, j]
                if val > maxval:
                    maxval = val

                maxpixel[i, j] = maxval
                acc[i, j] += val
                stdpixel[i, j] += val*val


        return maxpixel, acc, stdpixel


    cpdef finish(self):
        """ Finish making an FF structure. """

        # If there are less than 3 frames, don't subtract the max from average
        if self.nframes < 3:

            # Compute normal average
            self.avepixel = self.acc//self.nframes

            # Don't compute the standard deviation
            self.stdpixel *= 0


        else:
            
            # Remove the contribution of the maxpixel to the avepixel
            self.acc -= self.maxpixel

            self.avepixel = self.acc//(self.nframes - 1)
            
            # Compute the max subtracted standard deviation
            self.stdpixel -= (self.maxpixel.astype(np.uint64))**2
            self.stdpixel -= self.acc*self.avepixel
            self.stdpixel  = np.sqrt(self.stdpixel/(self.nframes - 2))


        # Make sure there are no zeros in standard deviation
        self.stdpixel[self.stdpixel == 0] = 1

        # Convert frames to the appropriate format
        self.maxpixel = self.maxpixel.astype(self.dtype)
        self.avepixel = self.avepixel.astype(self.dtype)
        self.stdpixel = self.stdpixel.astype(self.dtype)

        self.successful = True

        return True