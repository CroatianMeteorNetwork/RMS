
import numpy as np
import time

# Cython import
cimport numpy as np
cimport cython
from libc.math cimport fabsf, fmaxf
from libc.stdlib cimport rand

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
    
    # Stored frames for Reservoir Sampling (to estimate robust background)
    cdef np.ndarray sample_buf
    cdef int res_size
    
    # Public output arrays (matching your original interface types)
    cdef public np.ndarray maxpixel, avepixel, stdpixel

    def __init__(self, nrows, ncols, dtype):
        """ Structure which is used to make FF file format data. It mimicks the interface of an FF structure. 
    
        Arguments:
            nrows: [int] Number of image rows.
            ncols: [int] Number of image columns.
        """
        self.nrows = nrows
        self.ncols = ncols
        self.dtype = dtype
        self.nframes = 0
        self.calibrated = False
        self.successful = False

        # Init outputs - all internal processing done in uint16 for robustness
        self.maxpixel = np.zeros((nrows, ncols), dtype=np.uint16)
        self.avepixel = np.zeros((nrows, ncols), dtype=np.uint16)
        self.stdpixel = np.zeros((nrows, ncols), dtype=np.uint16)

        # Internal buffer for the Reservoir Sampling
        self.res_size = 256
        self.sample_buf = np.zeros((self.res_size, nrows, ncols), dtype=np.uint16)


    cpdef addFrame(self, np.ndarray[INT16_TYPE_t, ndim=2] frame):
        """ Add raw frame and update sampling buffer for robust background estimation. """
        
        # Initialize maxpixel on the first frame
        if self.nframes == 0:
            self.maxpixel[:, :] = frame
        else:
            # Update maxpixel (Standard, always applied)
            # Use NumPy maximum for speed
            self.maxpixel[...] = np.maximum(self.maxpixel, frame)
        
        # Reservoir sampling to fill/update the buffer
        # This ensuring the buffer always contains a representative sample of all frames
        if self.nframes < self.res_size:
            # Fill the buffer sequentially for the first N frames
            self.sample_buf[self.nframes, :, :] = frame
        else:
            # Randomly replace an existing frame in the buffer with probability res_size/n_total
            # This is the Reservoir Sampling algorithm (Algorithm R)
            if (rand()%(self.nframes + 1)) < self.res_size:
                self.sample_buf[rand() % self.res_size, :, :] = frame
        
        self.nframes += 1


    cpdef finish(self):
        """ Finalize the arrays by calculating Median and MAD from the sample buffer. """
        
        # Check if we have any frames
        if self.nframes == 0:
            self.successful = False
            return False

        # Number of samples actually in the buffer
        cdef int n_samples = min(self.nframes, self.res_size)
        
        # Use NumPy's optimized median along the temporal axis (axis 0)
        # Slicing the buffer to only include valid samples
        cdef np.ndarray valid_samples = self.sample_buf[:n_samples]
        
        # 1. Calculate Median (avepixel)
        # We compute this in float32 for precision during MAD calculation
        cdef np.ndarray median_float = np.median(valid_samples, axis=0).astype(np.float32)
        
        # 2. Calculate Median Absolute Deviation (MAD)
        # MAD = median(|x - median|)
        # The factor 1.4826 converts MAD to an unbiased estimate of Standard Deviation for normal distribution
        cdef np.ndarray abs_diff = np.abs(valid_samples.astype(np.float32) - median_float)
        cdef np.ndarray mad = np.median(abs_diff, axis=0)
        
        cdef np.ndarray std_float = mad * 1.4826

        # Safety for zero noise (Standard Deviation must be at least 1 for thresholding)
        std_float[std_float <= 0] = 1

        # Determine clipping bounds based on target dtype (default to uint8 range if not set)
        cdef float min_val = 0.0
        cdef float max_val = 65535.0
        try:
            info = np.iinfo(self.dtype)
            min_val = <float>info.min
            max_val = <float>info.max
        except:
            pass

        # Final clipping and casting to target dtype
        self.maxpixel = np.clip(self.maxpixel, min_val, max_val).astype(self.dtype)
        self.avepixel = np.clip(median_float, min_val, max_val).astype(self.dtype)
        self.stdpixel = np.clip(std_float,    min_val, max_val).astype(self.dtype)
        
        self.successful = True
        return True