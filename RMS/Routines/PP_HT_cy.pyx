import numpy as np
cimport numpy as np
cimport cython

INT_TYPE = np.int16
ctypedef np.int16_t INT_TYPE_t

UINT_TYPE = np.uint16
ctypedef np.uint16_t UINT_TYPE_t

FLOAT_TYPE = np.float32
ctypedef np.float32_t FLOAT_TYPE_t

def generateTrigLookup(trig_function, deg_range, float delta):
    """ Generate a given trigonometric function lookup table. 

    @param: trig_function: [numpy function] np.sin or np.cos, etc.
    @param: deg_range: [tuple] a tuple of degree ranges for input, e.g. (-180, 180)
    @param: delta: [float] step of thedegree range (e.g. delta = 0.5, then degrees are 0, 0.5, 1, 1.5,...)

    @return lookup: [1D ndarray] array of lookup values, access values: lookup[degree/delta]
    """

    range_min, range_max = deg_range

    # Precalculate trigonometric values
    lookup = trig_function(np.radians(np.arange(range_min, range_max, delta))).astype(FLOAT_TYPE)

    return lookup

def generateAtan2Lookup(int img_h, int img_w):
    """ Generates atan2 values for a given image dimensions.

    @param: img_h: [int] image height in pixels
    @param: img_w: [int] image width in pixels

    @return: atan2_lookup: [2D ndarray] atan2 looup table, usage: atan2_lookup[x, y]

    """

    # Preallocate memory for arctan2 values
    cdef np.ndarray[FLOAT_TYPE_t, ndim=2] atan2_lookup = np.empty((img_w*2, img_h*2), FLOAT_TYPE)

    # Make an array of x and y indices
    x_inds = np.arange(img_w*2)
    y_inds = np.arange(img_h*2)

    # Calculate arctan2 values for all indices
    atan2_lookup[np.meshgrid(x_inds,y_inds)] = np.degrees(np.arctan2(*np.meshgrid(x_inds-img_w, y_inds-img_h)))

    return atan2_lookup

def rebinMean(a, shape):
    """ Rebin the given array into chuncks, average values in each chunk. """
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

def rebinStd(a, shape):
    """ Rebin the given array into chuncks, find stddev of each chunk. """
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).std(-1).std(1)

def rebinMax(a, shape):
    """ Rebin the given array into chuncks, find max value in each chunk. """
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).max(-1).max(1)

def getSubFactor(size, sub_factor):
    """ Generates a subdivision factor with which the axis is divisible, but is closest to a predefined 
        number.
    """
    for i in xrange(1,size/2+1):
        if size%i == 0:
            if i >= sub_factor:
                sub_factor = i
                break

    return sub_factor


@cython.cdivision(True) # Don't check for zero division
@cython.boundscheck(False) # Don't check for index bounds
def pixelPairHT(np.ndarray[INT_TYPE_t, ndim=2] points, int img_h, int img_w, int ht_sigma_factor, 
        int ht_sigma_abs, int sub_factor, np.ndarray[FLOAT_TYPE_t, ndim=2] atan2_lookup, 
        np.ndarray[FLOAT_TYPE_t, ndim=1] sin_lookup, np.ndarray[FLOAT_TYPE_t, ndim=1] cos_lookup, 
        float delta = 0.5):
    """ Do a pixel pair Hough Transform. Sacrifices processor time (N**2 operations), but removes butterfly
        noise, which is nice.

        @param points: [2D ndarray uint16] (X, Y) positions of image points on which you want to perform HT
        @param img_h: [int] image height in pixels
        @param img_W: [int] image width in pixels
        @param ht_sigma_factor: [int] standard deviations above avreage in HT space to take the line as valid
        @param ht_sigma_abs: [int] minimum absolute counts above the usual threshold
        @param sub_factor: [int] subdivision factor of the HT space for local peak estimates
        @param atan2_lookup: [2D ndarray float32] preallocated atan2 values (bounded for image coordinates)
        @param sin_lookup: [1D ndarray float32] preallocated sine values (bounded for -180, 180)
        @param cos_lookup: [1D ndarray float32] preallocated cosine values (bounded for -180, 180)
        @param delta: [float] subdivision of the HT space (e.g. if delta = 0.5, HT space will be subdivided 
            every half degree)

        @return ht_lines: [2D ndarray] (rho, theta, count) which define the line in Hough space and their 
            respective counts

    """

    cdef int i, j
    cdef int x1, x2, y1, y2

    cdef float theta, rho

    cdef int theta_ind, rho_ind

    # Calculate image center
    cdef int center_h = img_h / 2
    cdef int center_w = img_w / 2

    # Allocate HT accumulator
    cdef int max_rho = <int>(np.sqrt(center_h**2 + center_w**2)) + 1
    cdef int rho_num = <int>(max_rho / delta)

    cdef int theta_num = <int> (360 / delta)

    cdef np.ndarray[UINT_TYPE_t, ndim=2] ht_space = np.zeros((rho_num, theta_num), UINT_TYPE)

    # Shift point positions in respect to the image center
    points[:,0] -= center_w
    points[:,1] -= center_h

    # Get point list size
    cdef int point_list_size = points.shape[0]

    # Go through every point in point list
    for i in range(point_list_size):

        # Load points
        x1 = points[i,0]
        y1 = points[i,1]

        # Go through every point in post list after the (x1, y1)
        
        # for j in range(point_list_size - i - 1):

        #     # Load points
        #     x2 = points[i + j + 1, 0]
        #     y2 = points[i + j + 1, 1]

        for j in range(point_list_size):

            # Load points
            x2 = points[j, 0]
            y2 = points[j, 1]

            # Calculate line slope angle
            theta = -atan2_lookup[(x2-x1) + img_w, (y2-y1) + img_h]

            # Calculate rho (distance to line)
            theta_ind = <int> (theta/delta)
            rho = (<float> x1) * cos_lookup[theta_ind] + (<float> y1) * sin_lookup[theta_ind]

            if rho < 0:
                rho = - rho
                theta = theta - 180
            
            # Calculate indices of HT accumulator to increment (add 0.5 and int = rounding)
            theta_ind = <int> ((theta + 360) / delta + 0.5)
            rho_ind = <int> (rho / delta + 0.5)

            # Increment HT accumulator
            ht_space[rho_ind, theta_ind] += 1

    # # Calculate HT space mean and standard deviation
    # ht_mean = np.mean(ht_space)
    # ht_stddev = np.std(ht_space)

    # # Calculate the HT thresholding value
    # ht_threshold = ht_mean + ht_sigma_factor * ht_stddev

    # # Get indices of points above the accumulator threshold
    # ht_lines = np.transpose((ht_space >= ht_threshold).nonzero())   

    # Get the appropriate factors which will subdivide the HT space into theta_sub_factor x rho_sub_factor size chunks
    theta_sub_factor = getSubFactor(theta_num, sub_factor)
    rho_sub_factor = getSubFactor(rho_num, sub_factor)

    ## HT space is not uniform and a global mean and stddev can't be calculated, so you find local values
    ## Subdivide HT space into theta_sub x rho_sub chunks (e.g. 7x8) and calculate:

    # - mean of each chunk
    ht_sub_mean = rebinMean(ht_space, (rho_num/rho_sub_factor, theta_num/theta_sub_factor))

    # - stddev of each chunk
    ht_sub_std = rebinStd(ht_space, (rho_num/rho_sub_factor, theta_num/theta_sub_factor))

    # - max element in each chunk
    ht_sub_max = rebinMax(ht_space, (rho_num/rho_sub_factor, theta_num/theta_sub_factor))

    # Calculate the threshold max > avg + factor * stddev
    ht_sub_threshold = ht_sub_mean + ht_sigma_factor * ht_sub_std + ht_sigma_abs

    # Get positions of threshold passers
    cdef np.ndarray[UINT_TYPE_t, ndim=2] ht_sub_max_inds = np.transpose(np.where(ht_sub_max > ht_sub_threshold)).astype(UINT_TYPE)

    ht_lines = []

    # Get HT space indices of all threshold passers + find all non-max values also
    for rho, theta in ht_sub_max_inds:

        rho_slice = rho_sub_factor * rho
        theta_slice = theta_sub_factor * theta
        
        # Get the original slice formt he HT space that was previously averaged
        ht_slice = ht_space[rho_slice : rho_slice + rho_sub_factor, theta_slice : theta_slice + 
            theta_sub_factor]
        
        # Find all threshold passers in ths slice
        # ht_peaks = np.transpose(np.where(ht_slice > ht_sub_mean[rho, theta] + ht_sigma_factor * 
        #     ht_sub_std[rho, theta] + ht_sigma_abs))
        
        # Find the one max value
        ht_peaks = np.transpose(np.where(np.max(ht_slice)))

        # Calculate the original HT space indices position from the subsampled positions
        ht_peaks = ht_peaks + np.array([rho_slice, theta_slice])

        # Add found peaks to the list
        for rho, theta in ht_peaks:
            ht_lines.append([rho, theta])

    # Check the case if no lines were found
    if ht_lines:
        ht_lines = np.array(ht_lines, dtype = UINT_TYPE)
    else:
        return np.array([])


    # Get HT accumulator counts
    ht_counts = ht_space[ht_lines[:,0], ht_lines[:,1]].reshape(-1,1)

    # Recalculate to distance and angle
    ht_lines = ht_lines * delta

    # Add third row to be counts in the accumulator
    ht_lines = np.hstack((ht_lines, ht_counts))

    # Sort by descending count
    ht_lines = ht_lines[ht_lines[:,2].argsort()][::-1]

    return ht_lines
