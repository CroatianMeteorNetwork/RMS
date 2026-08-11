import numpy as np

# Cython import
cimport numpy as np
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


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def compressFrames(np.ndarray[INT8_TYPE_t, ndim=3] frames, int deinterlace_order, double gamma=1.0):

    # Init the output four frame temporal pixel array
    cdef np.ndarray[INT8_TYPE_t, ndim=3] ftp_array = np.empty([4, frames.shape[1], frames.shape[2]],
        dtype=INT8_TYPE)

    # Full-precision average in 8.8 fixed point (units of 1/256 ADU). The 8-bit average in ftp_array
    # rounds away the sub-ADU precision of the 256-frame mean; this plane keeps it. The 8-bit plane
    # is derived from it by rounding off the fractional bits ((ave16 + 128) >> 8), the same way
    # readers derive the 8-bit view
    cdef np.ndarray[INT16_TYPE_t, ndim=2] ave16_array = np.empty([frames.shape[1], frames.shape[2]],
        dtype=INT16_TYPE)


    # Array for field/frame intensity sums. If the video is interlaced, then there with will twice the number 
    # of fields as there are frames
    cdef np.ndarray[INT32_TYPE_t, ndim=1] fieldsum = np.zeros((2*frames.shape[0]), INT32_TYPE)

    cdef unsigned int deinterlace_multiplier = 2

    # Init the field intensity sums array
    if deinterlace_order < 0:

        # If there's no deinterlacing, then only the values from the whole frame will be summed up
        deinterlace_multiplier = 1

    else:

        # Otherwise, values from every field will be summed up
        deinterlace_multiplier = 2

    
    cdef unsigned short rand_count = 1

    cdef unsigned int var, max_val, max_val_2, max_val_3, max_val_4, max_frame, mean, pixel, n, num_equal

    cdef unsigned int min_val, min_val_2, min_val_3, min_val_4

    cdef unsigned int x, y, acc, ave16
    cdef double var_d, acc_lin, mean_lin
    cdef unsigned int height = frames.shape[1]
    cdef unsigned int width = frames.shape[2]
    cdef unsigned int frames_num = frames.shape[0]

    # The mean and stddev are computed on a symmetrically trimmed sample: the top 4 values are
    # removed to suppress meteors and wakes, and the bottom 4 are removed to balance the trim -
    # a one-sided trim biases the mean low by ~0.04 sigma for symmetric noise
    cdef unsigned int n_trim = frames_num - 8
    cdef unsigned int n_trim_minus_one = frames_num - 9

    cdef unsigned int fieldsum_indx

    # When a camera gamma is given, average in the LINEAR domain: the mean of gamma-encoded
    # samples is biased low relative to the encoding of the linear-domain mean (Jensen's
    # inequality), by ~gamma*(1 - gamma)*(sigma/mean)**2/2 of the level. The decoded values are
    # averaged and the result is re-encoded, so the stored plane stays in the same gamma-encoded
    # domain all consumers expect (they apply their own gamma correction downstream). With
    # gamma = 1 an exact integer path is used. Note this uses the same pure power-law convention
    # (black point 0) as the rest of RMS - a camera pedestal inside the power law makes both
    # approximations
    cdef bint use_gamma = (gamma != 1.0)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] decode_lut = np.empty(256, dtype=FLOAT_TYPE)
    if use_gamma:
        decode_lut = (255.0*(np.arange(256)/255.0)**(1.0/gamma)).astype(FLOAT_TYPE)
    
    # Populate the randomN array with 2**16 random numbers
    cdef np.ndarray[INT8_TYPE_t, ndim=1] randomN = np.empty(shape=[65536], dtype=INT8_TYPE)
    cdef unsigned int arand = randomN[0]


    for n in range(65536):
        arand = (arand*32719 + 3)%32749
        randomN[n] = <unsigned char>(32767.0/<double>(1 + arand%32767))


    for y in range(height):
        for x in range(width):
        
            acc = 0
            acc_lin = 0
            var = 0
            max_val = 0
            max_val_2 = 0
            max_val_3 = 0
            max_val_4 = 0
            min_val = 255
            min_val_2 = 255
            min_val_3 = 255
            min_val_4 = 255
            num_equal = 0

            # Calculate mean, stddev, max_val, and max_val frame
            for n in range(frames_num):
            
                pixel = frames[n, y, x]
                acc += pixel
                var += pixel**2

                # Accumulate the linear-domain (gamma-decoded) sum
                if use_gamma:
                    acc_lin += decode_lut[pixel]

                # Assign the maximum value
                if pixel > max_val:
                    
                    # Track the top 4 maximum values
                    max_val_4 = max_val_3
                    max_val_3 = max_val_2
                    max_val_2 = max_val
                    max_val = pixel

                    max_frame = n
                    num_equal = 1


                else:

                    # Randomize taken frame number for max_val pixel if there are several frames with the 
                    # maximum value
                    if max_val == pixel:
                    
                        num_equal += 1
                        
                        # rand_count is unsigned short, which means it will overflow back to 0 after 65535
                        rand_count = (rand_count + 1)%65536

                        # Select the frame by random
                        if num_equal <= randomN[rand_count]:
                            max_frame = n


                    # Track the top 4 maximum values, which is used to remove wakes from mean and stddev
                    if pixel > max_val_2:
                        max_val_4 = max_val_3
                        max_val_3 = max_val_2
                        max_val_2 = pixel

                    elif pixel > max_val_3:
                        max_val_4 = max_val_3
                        max_val_3 = pixel

                    elif pixel > max_val_4:
                        max_val_4 = pixel


                # Track the bottom 4 minimum values, which balance the trim of the top 4
                if pixel < min_val:
                    min_val_4 = min_val_3
                    min_val_3 = min_val_2
                    min_val_2 = min_val
                    min_val = pixel

                elif pixel < min_val_2:
                    min_val_4 = min_val_3
                    min_val_3 = min_val_2
                    min_val_2 = pixel

                elif pixel < min_val_3:
                    min_val_4 = min_val_3
                    min_val_3 = pixel

                elif pixel < min_val_4:
                    min_val_4 = pixel


                # Calculate the index for fieldsum, dependent on the deinterlace order (and if there's any
                # detinerlacing at all)
                fieldsum_indx = deinterlace_multiplier*n \
                    + (deinterlace_multiplier - 1)*((y + deinterlace_order)%2)

                # Sum intensity per every field
                fieldsum[fieldsum_indx] += <unsigned long> pixel

            
            
            # Sum without the top 4 and bottom 4 values (symmetric trim)
            acc -= max_val + max_val_2 + max_val_3 + max_val_4
            acc -= min_val + min_val_2 + min_val_3 + min_val_4

            if use_gamma:

                # Average in the linear domain and re-encode into the gamma domain the file
                # stores. The trim removes the same frames in both domains (the decode is
                # monotone)
                acc_lin -= decode_lut[max_val] + decode_lut[max_val_2] + decode_lut[max_val_3] \
                    + decode_lut[max_val_4]
                acc_lin -= decode_lut[min_val] + decode_lut[min_val_2] + decode_lut[min_val_3] \
                    + decode_lut[min_val_4]

                mean_lin = acc_lin/n_trim
                ave16 = <unsigned int>(256.0*255.0*pow(mean_lin/255.0, gamma) + 0.5)

            else:

                # Full-precision mean, rounded to 1/256 ADU. No overflow: acc <= 248*255, so
                # 256*acc fits comfortably in 32 bits, and the result is at most 255*256
                ave16 = (256*acc + n_trim/2)/n_trim

            ave16_array[y, x] = <unsigned short>ave16

            # 8-bit mean, rounded off the fixed-point mean - the same derivation readers use for
            # the 8-bit view, so the two planes always agree
            mean = (ave16 + 128) >> 8



            ### Calculate stddev on the symmetrically trimmed sample (encoded domain) ##

            # Remove the top 4 and bottom 4 values
            var -= max_val**2 + max_val_2**2 + max_val_3**2 + max_val_4**2
            var -= min_val**2 + min_val_2**2 + min_val_3**2 + min_val_4**2

            # Sample variance of the remaining values. acc**2 overflows 32 bits, so compute in
            # double precision (exact: both terms are far below 2**53)
            var_d = (var - (<double>acc)*acc/n_trim)/n_trim_minus_one

            # Guard against small negative values from floating point rounding
            if var_d < 0:
                var_d = 0

            # Compute the standard deviation, rounded
            var = <unsigned int>(sqrt(var_d) + 0.5)

            # Make sure that the stddev is not 0, to prevent divide by zero afterwards
            if var == 0:
                var = 1

            ###
            
            
            # Output results
            ftp_array[0, y, x] = max_val
            ftp_array[1, y, x] = max_frame
            ftp_array[2, y, x] = mean
            ftp_array[3, y, x] = var


    return ftp_array, ave16_array, fieldsum[:frames_num*deinterlace_multiplier]