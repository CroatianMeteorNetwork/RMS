
import numpy as np
import time

# Cython import
cimport numpy as np
cimport cython
from libc.math cimport fabsf, fmaxf

# Define numpy types
INT16_TYPE = np.uint16
ctypedef np.uint16_t INT16_TYPE_t

INT64_TYPE = np.uint64
ctypedef np.uint64_t INT64_TYPE_t



@cython.boundscheck(False)
@cython.wraparound(False)
cdef void updateMaxFrame(INT16_TYPE_t[:, ::1] maxpixel, np.uint32_t[:, ::1] maxframe, INT16_TYPE_t[:, :] frame,
        np.uint32_t frame_index):
    """ Update the maximum of every pixel with the frame, and the index of the frame where it is brighter. """

    cdef Py_ssize_t i, j
    cdef INT16_TYPE_t value

    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            value = frame[i, j]
            if value > maxpixel[i, j]:
                maxpixel[i, j] = value
                maxframe[i, j] = frame_index


# Image rows per block in sampleMedianMAD
MEDIAN_BLOCK_ROWS = 8

# Sample types of sampleMedianMAD: the raw frames of the FF files, and the calibrated frames of the matched filter
ctypedef fused sample_t:
    np.uint16_t
    np.float32_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double selectKth(double *a, Py_ssize_t n, Py_ssize_t k) nogil:
    """ The k-th smallest of the n values (0-based), by quickselect (Hoare partitioning), in place. On return,
        the values before position k are not larger than it.
    """

    cdef Py_ssize_t lo = 0, hi = n - 1, i, j
    cdef double pivot, tmp

    while hi > lo:

        # Partition the range around the middle value: the values up to j are not larger than the pivot, the
        #   values from i on are not smaller, and the ones between are equal to it
        pivot = a[(lo + hi)//2]
        i = lo
        j = hi
        while i <= j:
            while a[i] < pivot:
                i += 1
            while a[j] > pivot:
                j -= 1
            if i <= j:
                tmp = a[i]
                a[i] = a[j]
                a[j] = tmp
                i += 1
                j -= 1

        # Continue in the part which contains position k, or stop if it is equal to the pivot
        if k <= j:
            hi = j
        elif k >= i:
            lo = i
        else:
            break

    return a[k]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double medianInPlace(double *a, Py_ssize_t n) nogil:
    """ The median of the n values, as np.median computes it: the middle value, or the mean of the two middle
        values for an even n. The values are reordered.
    """

    cdef Py_ssize_t k = n//2, i
    cdef double upper = selectKth(a, n, k), lower

    if n % 2 == 1:
        return upper

    # For an even n, the other middle value is the largest of the values before position k
    lower = a[0]
    for i in range(1, k):
        if a[i] > lower:
            lower = a[i]

    return 0.5*(lower + upper)


@cython.boundscheck(False)
@cython.wraparound(False)
def blockMedianMAD(sample_t[:, ::1] block, float[::1] median, float[::1] mad):
    """ The median and the median absolute deviation of every pixel of a block, the samples of every pixel
        contiguous (a row of the block). The results are identical to np.median: the median is computed from the
        values (as float64), and the MAD from the absolute deviations of the samples (as float32) from the median
        (as float32), as np.median(np.abs(samples.astype(np.float32) - median), axis=1) computes it.

    Arguments:
        block: [ndarray] Samples, shape (n_pixels, n_samples).
        median: [ndarray] float32 output, the median of every pixel.
        mad: [ndarray] float32 output, the MAD of every pixel.
    """

    cdef Py_ssize_t n_pix = block.shape[0], n = block.shape[1], p, s
    cdef double[::1] buf = np.empty(max(n, 1), dtype=np.float64)
    cdef double *b = &buf[0]
    cdef float med

    with nogil:
        for p in range(n_pix):

            # The median of the samples of the pixel
            for s in range(n):
                b[s] = <double>block[p, s]
            med = <float>medianInPlace(b, n)
            median[p] = med

            # The median of the absolute deviations from the median, in float32
            for s in range(n):
                b[s] = <double>fabsf(<float>block[p, s] - med)
            mad[p] = <float>medianInPlace(b, n)


def sampleMedianMAD(samples, block_rows=MEDIAN_BLOCK_ROWS):
    """ Compute the median and the median absolute deviation (MAD) of every pixel over the sampled frames.

    The samples of a pixel are one frame apart in memory, so taking the median along the first axis reads
    memory with a large power of two stride (e.g. 512 KB for 512x512 16-bit frames), which is very slow on
    some CPUs (more than 10 times slower on an AMD Ryzen 3950X than on a recent Intel CPU). The frames are
    therefore processed in blocks of image rows: the rows of a block are copied into a buffer whose row
    length is not a power of two, and the buffer is transposed so that the samples of every pixel are
    contiguous. The results are identical to np.median along the first axis.

    Arguments:
        samples: [ndarray] Sampled frames, shape (n_samples, height, width).

    Keyword arguments:
        block_rows: [int] Image rows per block. MEDIAN_BLOCK_ROWS by default.

    Return:
        (median, mad): [tuple of ndarrays] float32 images of the median and of the median absolute deviation.
    """

    n_samples, height, width = samples.shape

    median = np.empty((height, width), dtype=np.float32)
    mad = np.empty((height, width), dtype=np.float32)

    # Pad the buffer rows, so their length is not a power of two
    row_len = block_rows*width + 16
    if (row_len & (row_len - 1)) == 0:
        row_len += 16

    buf = np.empty((n_samples, row_len), dtype=samples.dtype)

    for row in range(0, height, block_rows):

        rows = min(block_rows, height - row)
        n_pix = rows*width

        # Samples of every pixel of the block in a contiguous row
        buf[:, :n_pix] = samples[:, row:row + rows, :].reshape(n_samples, n_pix)
        block = np.ascontiguousarray(buf[:, :n_pix].T)

        # The median and the MAD of every pixel of the block (compiled, for the raw and the calibrated frames;
        #   np.median for other types)
        if block.dtype in (np.uint16, np.float32):
            block_median = np.empty(n_pix, dtype=np.float32)
            block_mad = np.empty(n_pix, dtype=np.float32)
            blockMedianMAD(block, block_median, block_mad)
        else:
            block_median = np.median(block, axis=1).astype(np.float32)
            block_mad = np.median(np.abs(block.astype(np.float32) - block_median[:, None]), axis=1)

        median[row:row + rows] = block_median.reshape(rows, width)
        mad[row:row + rows] = block_mad.reshape(rows, width)

    return median, mad


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef class FFMimickInterface:

    cdef public int nrows, ncols, nframes
    cdef public object dtype
    cdef public np.npy_bool calibrated, successful
    
    # Stored frames for Reservoir Sampling (to estimate robust background)
    cdef np.ndarray sample_buf
    # Public so consumers (e.g. SkyFit photometry) can read the reservoir size that bounds
    # how many frames actually contribute to the avepixel median.
    cdef public int res_size

    # State of the random number generator of the reservoir sampling
    cdef unsigned long long rng_state
    
    # Public output arrays (matching your original interface types)
    cdef public np.ndarray maxpixel, avepixel, stdpixel

    # Index of the frame in which every pixel was at its maximum (counted from the first added frame)
    cdef public np.ndarray maxframe

    def __init__(self, nrows, ncols, dtype, res_size=64):
        """ Structure which is used to make FF file format data. It mimicks the interface of an FF structure. 
    
        Arguments:
            nrows: [int] Number of image rows.
            ncols: [int] Number of image columns.
            dtype: [numpy dtype] Target output data type.
            res_size: [int] Number of frames in the reservoir sampling buffer used for background estimation.
                Default is 64. Higher values (e.g. 256) provide better precision but significantly increase
                memory and CPU time.
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
        self.maxframe = np.zeros((nrows, ncols), dtype=np.uint32)

        # Internal buffer for the Reservoir Sampling (increase for better background estimation, 
        # e.g. 256, but comes with a significant performance hit - approx 4x)
        self.res_size = res_size
        self.sample_buf = np.zeros((self.res_size, nrows, ncols), dtype=np.uint16)

        # The reservoir sampling uses its own generator with a fixed seed, so the same frames always give the
        #   same background. The libc rand() state is shared by the whole process, including other libraries
        self.rng_state = 0x2545F4914F6CDD1D


    cdef unsigned long long nextRandom(self):
        """ Return the next 31-bit random number of a 64-bit linear congruential generator (Knuth's MMIX
            constants), using the high bits, which are the most random.
        """

        self.rng_state = self.rng_state*6364136223846793005ULL + 1442695040888963407ULL

        return self.rng_state >> 33


    cpdef addFrame(self, np.ndarray[INT16_TYPE_t, ndim=2] frame):
        """ Add raw frame and update sampling buffer for robust background estimation. """

        cdef unsigned long long j
        
        # Initialize maxpixel on the first frame
        if self.nframes == 0:
            self.maxpixel[:, :] = frame
        else:
            # Update the maxpixel, and the maxframe where the pixel is brighter than in all previous frames
            updateMaxFrame(self.maxpixel, self.maxframe, frame, self.nframes)
        
        # Reservoir sampling to fill/update the buffer
        # This ensuring the buffer always contains a representative sample of all frames
        if self.nframes < self.res_size:
            # Fill the buffer sequentially for the first N frames
            self.sample_buf[self.nframes, :, :] = frame
        else:
            # Randomly replace an existing frame in the buffer with probability res_size/n_total
            # This is the Reservoir Sampling algorithm (Algorithm R)
            j = self.nextRandom()%(self.nframes + 1)
            if j < self.res_size:
                self.sample_buf[j, :, :] = frame
        
        self.nframes += 1


    cpdef finish(self):
        """ Finalize the arrays by calculating Median and MAD from the sample buffer. """
        
        # Check if we have any frames
        if self.nframes == 0:
            self.successful = False
            return False

        # Number of samples actually in the buffer
        cdef int n_samples = min(self.nframes, self.res_size)
        
        # Median (avepixel) and median absolute deviation of every pixel over the valid samples
        median_float, mad = sampleMedianMAD(self.sample_buf[:n_samples])

        # The factor 1.4826 converts MAD to an unbiased estimate of Standard Deviation for normal distribution
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

        # The sampling reservoir is only needed while building the FF statistics.
        # Drop it so cached finished chunks do not retain all sampled frames.
        self.sample_buf = np.empty((0, 0, 0), dtype=np.uint16)
        
        self.successful = True
        return True
