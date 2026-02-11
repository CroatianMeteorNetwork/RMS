
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
    
    # We use Float32 for the running accumulators to allow for sub-integer precision
    cdef np.ndarray med_buf, mad_buf 
    
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

        # Init outputs
        self.maxpixel = np.zeros((nrows, ncols), dtype=dtype)
        self.avepixel = np.zeros((nrows, ncols), dtype=dtype) # Will hold the Median
        self.stdpixel = np.zeros((nrows, ncols), dtype=dtype) # Will hold the MAD

        # Internal float buffers for the running approximation
        self.med_buf = np.zeros((nrows, ncols), dtype=np.float32)
        self.mad_buf = np.zeros((nrows, ncols), dtype=np.float32)


    cpdef addFrame(self, np.ndarray[INT16_TYPE_t, ndim=2] frame):
        """ Add raw frame. Handles initialization on the first frame automatically. """
        
        # Initialization: If it's the first frame, jump start the buffers
        if self.nframes == 0:
            
            self.med_buf[:, :] = frame.astype(np.float32)
            
            # Initialize noise floor to a small value
            self.mad_buf[:, :] = 1.0
            
            # Initialize maxpixel
            self.maxpixel[:, :] = frame

        else:
            self.frameProc(frame)
        
        self.nframes += 1

    cdef frameProc(self, np.ndarray[INT16_TYPE_t, ndim=2] frame):
        cdef int i, j
        cdef int nrows = self.nrows
        cdef int ncols = self.ncols
        cdef float pix_val, med_val, mad_val, diff
        
        # Access raw data pointers for speed
        cdef float[:, :] med_view = self.med_buf
        cdef float[:, :] mad_view = self.mad_buf
        cdef INT16_TYPE_t[:, :] frame_view = frame
        cdef INT16_TYPE_t[:, :] max_view = self.maxpixel

        for i in range(nrows):
            for j in range(ncols):
            
                pix_val = <float>frame_view[i, j]
                med_val = med_view[i, j]
                mad_val = mad_view[i, j]

                # --- 1. Update Max Pixel (Standard) ---
                if pix_val > max_view[i, j]:
                    max_view[i, j] = <INT16_TYPE_t>pix_val

                # --- 2. Update Approximate Median (Sigma-Delta) ---
                # If pixel > median, increment median. If pixel < median, decrement.
                # This converges to the median without sorting.
                if pix_val > med_val:
                    med_val += 1.0
                elif pix_val < med_val:
                    med_val -= 1.0
                
                # Write back to buffer
                med_view[i, j] = med_val

                # --- 3. Update Approximate MAD (Noise Estimation) ---
                # Calculate deviation from our current median estimate
                diff = abs(pix_val - med_val)
                
                # Same Sigma-Delta logic for the deviation
                if diff > mad_val:
                    mad_val += 1.0
                elif diff < mad_val:
                    mad_val -= 1.0
                
                mad_view[i, j] = mad_val

    cpdef finish(self):
        """ Finalize the arrays. """
        
        # Convert the float buffers to the output format
        self.avepixel = self.med_buf.astype(self.dtype)
        
        # Convert MAD to approximate Standard Deviation
        # Sigma approx = 1.4826*MAD
        # We can do this math on the whole array at once (vectorized)
        self.stdpixel = (self.mad_buf*1.4826).astype(self.dtype)
        
        # Safety for zero noise
        self.stdpixel[self.stdpixel == 0] = 1
        
        self.successful = True
        return True