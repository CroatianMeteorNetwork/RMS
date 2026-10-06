import numpy as np

# Cython import
cimport numpy as np

# Initialize the NumPy C API (explicit for clarity: Cython 3 (required to build against NumPy 2) emits it itself)
np.import_array()
cimport cython

# Define numpy types
INT8_TYPE = np.uint8
ctypedef np.uint8_t INT8_TYPE_t

INT16_TYPE = np.uint16
ctypedef np.uint16_t INT16_TYPE_t

INT32_TYPE = np.uint32
ctypedef np.uint32_t INT32_TYPE_t

FLOAT_TYPE = np.float64 
ctypedef np.float64_t FLOAT_TYPE_t


# Declare math functions
cdef extern from "math.h":
    double sqrt(double)
    double pow(double, double)

from libc.string cimport memset


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double encodeLUT(double v, const double* lut) nogil:
    """ Inverse of a strictly increasing 256-entry decode LUT (piecewise linear, so the inverse
    is exact): linear value v -> fractional code 0..255. A raw pointer, not a memoryview: this
    runs once per pixel per block, and a memoryview argument is reference-counted per call. """
    cdef int lo = 0, hi = 255, mid
    if v <= lut[0]:
        return 0.0
    if v >= lut[255]:
        return 255.0
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if lut[mid] <= v:
            lo = mid
        else:
            hi = mid
    return lo + (v - lut[lo])/(lut[hi] - lut[lo])


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def compressFrames(np.ndarray[INT8_TYPE_t, ndim=3] frames, int deinterlace_order, double gamma=1.0,
    response=None):
    """ Compress a block of frames into the four FF planes plus the full-precision average and
        standard deviation.

    The samples of one pixel are spread over the frames, 2 MB apart at 1080p, so a loop over the
    frames of a single pixel misses the cache on every sample: a 64-byte line is fetched for one
    byte. The loop below instead sweeps a strip of BLOCK consecutive pixels of one row through all
    frames, so each line fetched serves BLOCK samples, and keeps the per-pixel state of the strip
    (histograms, maxima) in small L1-resident arrays. The sums, sums of squares, linear-domain
    sums and both trims are then taken from each pixel's histogram, scanned over its occupied
    range, so the per-sample work is one counter increment, a row sum and the maximum test.
    Measured on a 1080p block: 1.35 s against 3.5 s for the per-pixel loop it replaced on a
    night sky (the histogram scans are short), equal to it in the worst case of pixels spanning
    the whole code range, with the extra statistics included.

    Arguments:
        frames: [3D ndarray uint8] (N, rows, cols), N >= 64 (256 in production).
        deinterlace_order: [int] -1 or -2 for progressive video (one field sum per frame), else
            the field order for interlaced video (two field sums per frame).

    Keyword arguments:
        gamma: [float] Camera gamma; != 1.0 averages in the linear domain (see below).
        response: [ResponseCurve] The camera's own decode table; when it is not a power law it
            replaces the power law of gamma for the linear-domain averaging.

    Return:
        (ftp_array, ave16_array, std16_array, fieldsum):
            - ftp_array: [3D ndarray uint8] maxpixel, maxframe, avepixel, stdpixel.
            - ave16_array: [2D ndarray uint16] the trimmed mean in 8.8 fixed point (1/256 code).
            - std16_array: [2D ndarray uint16] the trimmed sigma in 8.8 fixed point, >= 128.
            - fieldsum: [1D ndarray uint32] intensity sum per frame (or per field).
    """

    cdef unsigned int height = frames.shape[1]
    cdef unsigned int width = frames.shape[2]
    cdef unsigned int frames_num = frames.shape[0]

    if frames_num < 64:
        raise ValueError("compressFrames needs at least 64 frames for the trimmed statistics, got %d"
            % frames_num)

    # Init the output four frame temporal pixel array
    cdef np.ndarray[INT8_TYPE_t, ndim=3] ftp_array = np.empty([4, height, width], dtype=INT8_TYPE)

    # Full-precision average in 8.8 fixed point (units of 1/256 ADU). The 8-bit average in ftp_array
    # rounds away the sub-ADU precision of the 256-frame mean; this plane keeps it. The 8-bit plane
    # is derived from it by rounding off the fractional bits ((ave16 + 128) >> 8), the same
    # derivation the FF writer uses, so the stored planes always agree
    cdef np.ndarray[INT16_TYPE_t, ndim=2] ave16_array = np.empty([height, width], dtype=INT16_TYPE)

    # Full-precision standard deviation in 8.8 fixed point (units of 1/256 code). The 8-bit plane
    # rounds a ~2.4 code night-sky sigma to 2 or 3 (a 20% error that differs from pixel to pixel);
    # this plane keeps the fraction. Floored at half a code (128): the 8-bit plane derived from it,
    # (std16 + 128) >> 8, is then never 0 - the same floor the 8-bit plane always had - and the
    # split into the 8-bit plane plus a fractional byte (FFfits) inverts exactly
    cdef np.ndarray[INT16_TYPE_t, ndim=2] std16_array = np.empty([height, width], dtype=INT16_TYPE)

    # Array for field/frame intensity sums. If the video is interlaced, then there with will twice the
    # number of fields as there are frames
    cdef np.ndarray[INT32_TYPE_t, ndim=1] fieldsum = np.zeros((2*frames_num), INT32_TYPE)

    cdef unsigned int deinterlace_multiplier = 2

    # Init the field intensity sums array
    if deinterlace_order < 0:

        # If there's no deinterlacing, then only the values from the whole frame will be summed up
        deinterlace_multiplier = 1

    else:

        # Otherwise, values from every field will be summed up
        deinterlace_multiplier = 2

    # The mean is computed on a symmetrically trimmed sample: the top 4 values are removed to
    # suppress meteors and wakes, and the bottom 4 are removed to balance the trim - a one-sided
    # trim biases the mean low by ~0.04 sigma for symmetric noise
    cdef unsigned int n_trim = frames_num - 8

    # The standard deviation is computed on a more heavily trimmed sample, the top 16 and bottom
    # 16 values removed. The variance is far more sensitive to outliers than the mean: an object
    # of +40 codes dwelling 8 frames in a pixel (a wake, a slow bright meteor, a satellite) more
    # than doubles a 4/4-trimmed sigma, and a k*sigma threshold then rises against the object
    # itself. With 16 trimmed at the top the estimate stays within 0.4 codes of the truth up to a
    # 16-frame dwell (Monte Carlo, n = 256). Trimming narrows the sample, so the trimmed sample
    # standard deviation is scaled back to the full-population sigma with the Gaussian factor
    # below (E[s_trim]/sigma = 0.7563 for 16/16 of 256; the former 4/4 trim was 0.9108 and was
    # never corrected, so the old plane read 9% low). The estimate's own scatter is 5.1% per pixel
    # against 4.6% for the 4/4 trim. Both trims come from a per-pixel histogram of the 8-bit
    # samples, scanned in from both ends (exact with ties: the trimmed multiset)
    cdef unsigned int n_trim_sigma = frames_num - 32
    cdef double trim_sigma_corr = 1.3223

    # When a camera gamma is given, average in the LINEAR domain: the mean of gamma-encoded
    # samples is biased low relative to the encoding of the linear-domain mean (Jensen's
    # inequality), by ~gamma*(1 - gamma)*(sigma/mean)**2/2 of the level. The decoded values are
    # averaged and the result is re-encoded, so the stored plane stays in the same gamma-encoded
    # domain all consumers expect (they apply their own gamma correction downstream). With
    # gamma = 1 an exact integer path is used. Note this uses the same pure power-law convention
    # (black point 0) as the rest of RMS - a camera pedestal inside the power law makes both
    # approximations
    #
    # response: an RMS.Routines.Response.ResponseCurve. When it is the camera's own table (not
    # a power law), decode and re-encode with that table instead of the power law of gamma
    cdef bint use_table = (response is not None) and (not response.isPower)
    cdef bint use_gamma = (gamma != 1.0) or use_table
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] decode_lut = np.empty(256, dtype=FLOAT_TYPE)
    if use_table:
        decode_lut = np.ascontiguousarray(response.decodeLUT(255), dtype=FLOAT_TYPE)
    elif use_gamma:
        decode_lut = (255.0*(np.arange(256)/255.0)**(1.0/gamma)).astype(FLOAT_TYPE)
    cdef const double* decode_ptr = <const double*> np.PyArray_DATA(decode_lut)
    cdef double mean_lin

    # Populate the randomN array with 2**16 random numbers (ties of the maximum pick their frame
    # at random, so a static source that saturates in many frames does not always report the
    # first one)
    cdef np.ndarray[INT8_TYPE_t, ndim=1] randomN = np.empty(shape=[65536], dtype=INT8_TYPE)
    cdef unsigned int arand = randomN[0]
    cdef unsigned short rand_count = 1
    cdef unsigned int n
    for n in range(65536):
        arand = (arand*32719 + 3)%32749
        randomN[n] = <unsigned char>(32767.0/<double>(1 + arand%32767))

    # Per-strip state. BLOCK pixels of one row are swept through all frames together; their
    # histograms (BLOCK x 256 x 2 bytes = 32 KB) and sums live in L1
    DEF BLOCK = 64
    cdef unsigned short hist[BLOCK][256]
    cdef unsigned int max_val[BLOCK]
    cdef unsigned int max_frame[BLOCK]
    cdef unsigned int num_equal[BLOCK]

    cdef unsigned int x0, nb, i, x, y, pixel, rowsum, fieldsum_indx
    cdef unsigned int remaining, remaining4, c, c4, v
    cdef unsigned int sum_top4, sum_bot4, sum_top16, sum_bot16, sq_top16, sq_bot16
    cdef unsigned int acc_mean, acc_sigma, var_sigma, ave16, std16, mean, var
    cdef double lin4, var_d, lin_i
    cdef unsigned int v_min, acc_i, sq_i
    cdef const unsigned char* row

    for y in range(height):
        for x0 in range(0, width, BLOCK):

            nb = width - x0
            if nb > BLOCK:
                nb = BLOCK

            # Reset the strip state
            memset(hist, 0, sizeof(hist))
            for i in range(nb):
                max_val[i] = 0
                max_frame[i] = 0
                num_equal[i] = 0

            # Sweep the strip through all frames
            for n in range(frames_num):

                row = <const unsigned char*> &frames[n, y, x0]
                rowsum = 0

                for i in range(nb):

                    pixel = row[i]
                    rowsum += pixel
                    hist[i][pixel] += 1

                    # The maximum and its frame
                    if pixel > max_val[i]:

                        max_val[i] = pixel
                        max_frame[i] = n
                        num_equal[i] = 1

                    elif pixel == max_val[i]:

                        # Randomize taken frame number for max_val pixel if there are several frames
                        # with the maximum value
                        num_equal[i] += 1

                        # rand_count is unsigned short, which means it will overflow back to 0 after
                        # 65535
                        rand_count = (rand_count + 1)%65536

                        # Select the frame by random
                        if num_equal[i] <= randomN[rand_count]:
                            max_frame[i] = n

                # Calculate the index for fieldsum, dependent on the deinterlace order (and if there's
                # any deinterlacing at all), and sum the strip's intensity into its field
                fieldsum_indx = deinterlace_multiplier*n \
                    + (deinterlace_multiplier - 1)*((y + deinterlace_order)%2)
                fieldsum[fieldsum_indx] += rowsum

            # Finish each pixel of the strip from its histogram
            for i in range(nb):

                x = x0 + i

                # Scan in from the bottom: the 16 smallest samples for the sigma, of which the 4
                # smallest for the mean (the trims are nested)
                remaining = 16
                remaining4 = 4
                sum_bot16 = 0
                sq_bot16 = 0
                sum_bot4 = 0
                lin4 = 0
                v = 0
                while hist[i][v] == 0:
                    v += 1
                v_min = v
                while remaining > 0:
                    c = hist[i][v]
                    if c > remaining:
                        c = remaining
                    sum_bot16 += c*v
                    sq_bot16 += c*v*v
                    remaining -= c
                    if remaining4 > 0:
                        c4 = c
                        if c4 > remaining4:
                            c4 = remaining4
                        sum_bot4 += c4*v
                        if use_gamma:
                            lin4 += c4*decode_ptr[v]
                        remaining4 -= c4
                    v += 1

                # Scan in from the top: the 16 largest, of which the 4 largest for the mean
                remaining = 16
                remaining4 = 4
                sum_top16 = 0
                sq_top16 = 0
                sum_top4 = 0
                v = max_val[i]
                while remaining > 0:
                    c = hist[i][v]
                    if c > remaining:
                        c = remaining
                    sum_top16 += c*v
                    sq_top16 += c*v*v
                    remaining -= c
                    if remaining4 > 0:
                        c4 = c
                        if c4 > remaining4:
                            c4 = remaining4
                        sum_top4 += c4*v
                        if use_gamma:
                            lin4 += c4*decode_ptr[v]
                        remaining4 -= c4
                    v -= 1

                # Totals over the occupied range of the histogram (typically a few tens of bins):
                # the sum, the sum of squares and the linear-domain sum, which the per-sample loop no
                # longer accumulates
                acc_i = 0
                sq_i = 0
                lin_i = 0
                for v in range(v_min, max_val[i] + 1):
                    c = hist[i][v]
                    if c > 0:
                        acc_i += c*v
                        sq_i += c*v*v
                        if use_gamma:
                            lin_i += c*decode_ptr[v]

                ### Mean on the 4/4 trimmed sample ###

                acc_mean = acc_i - sum_top4 - sum_bot4

                if use_gamma:

                    # Average in the linear domain and re-encode into the gamma domain the file
                    # stores. The trim removes the same frames in both domains (the decode is
                    # monotone)
                    mean_lin = (lin_i - lin4)/n_trim
                    if use_table:
                        ave16 = <unsigned int>(256.0*encodeLUT(mean_lin, decode_ptr) + 0.5)
                    else:
                        ave16 = <unsigned int>(256.0*255.0*pow(mean_lin/255.0, gamma) + 0.5)

                else:

                    # Full-precision mean, rounded to 1/256 ADU. No overflow: acc <= 248*255, so
                    # 256*acc fits comfortably in 32 bits, and the result is at most 255*256
                    ave16 = (256*acc_mean + n_trim/2)/n_trim

                ave16_array[y, x] = <unsigned short>ave16

                # 8-bit mean, rounded off the fixed-point mean - the same derivation the FF writer
                # uses, so the two planes always agree
                mean = (ave16 + 128) >> 8

                ### Standard deviation on the 16/16 trimmed sample (encoded domain) ###

                acc_sigma = acc_i - sum_top16 - sum_bot16
                var_sigma = sq_i - sq_top16 - sq_bot16

                # Sample variance of the remaining values. acc**2 overflows 32 bits, so compute in
                # double precision (exact: both terms are far below 2**53)
                var_d = (var_sigma - (<double>acc_sigma)*acc_sigma/n_trim_sigma)/(n_trim_sigma - 1)

                # Guard against small negative values from floating point rounding
                if var_d < 0:
                    var_d = 0

                # Standard deviation in 8.8 fixed point, scaled from the trimmed sample to the full
                # population and floored at half a code (see std16_array)
                std16 = <unsigned int>(256.0*trim_sigma_corr*sqrt(var_d) + 0.5)
                if std16 < 128:
                    std16 = 128
                std16_array[y, x] = <unsigned short>std16

                # 8-bit standard deviation, rounded off the fixed-point value - the same derivation
                # the FF writer uses, so the two planes always agree. The floor above keeps it >= 1,
                # which prevents a divide by zero afterwards (as the old explicit floor did)
                var = (std16 + 128) >> 8

                # Output results
                ftp_array[0, y, x] = max_val[i]
                ftp_array[1, y, x] = max_frame[i]
                ftp_array[2, y, x] = mean
                ftp_array[3, y, x] = var

    return ftp_array, ave16_array, std16_array, fieldsum[:frames_num*deinterlace_multiplier]
