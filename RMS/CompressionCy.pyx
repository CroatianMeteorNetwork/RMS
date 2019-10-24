import numpy as np

# Cython import
cimport numpy as np
cimport cython

# Define numpy types
INT8_TYPE = np.uint8
ctypedef np.uint8_t INT8_TYPE_t

INT32_TYPE = np.uint32
ctypedef np.uint32_t INT32_TYPE_t

FLOAT_TYPE = np.float64 
ctypedef np.float64_t FLOAT_TYPE_t


# Declare math functions
cdef extern from "math.h":
    double sqrt(double)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def compressFrames(np.ndarray[INT8_TYPE_t, ndim=3] frames, int deinterlace_order):

    # Init the output four frame temporal pixel array
    cdef np.ndarray[INT8_TYPE_t, ndim=3] ftp_array = np.empty([4, frames.shape[1], frames.shape[2]], 
        dtype=INT8_TYPE)


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
    
    cdef unsigned int x, y, acc
    cdef unsigned int height = frames.shape[1]
    cdef unsigned int width = frames.shape[2]
    cdef unsigned int frames_num = frames.shape[0]
    cdef unsigned int frames_num_minus_four = frames_num - 4
    cdef unsigned int frames_num_minus_five = frames_num - 5

    cdef unsigned int fieldsum_indx
    
    # Populate the randomN array with 2**16 random numbers
    cdef np.ndarray[INT8_TYPE_t, ndim=1] randomN = np.empty(shape=[65536], dtype=INT8_TYPE)
    cdef unsigned int arand = randomN[0]


    for n in range(65536):
        arand = (arand*32719 + 3)%32749
        randomN[n] = <unsigned char>(32767.0/<double>(1 + arand%32767))


    for y in range(height):
        for x in range(width):
        
            acc = 0
            var = 0
            max_val = 0
            max_val_2 = 0
            max_val_3 = 0
            max_val_4 = 0
            num_equal = 0
            
            # Calculate mean, stddev, max_val, and max_val frame
            for n in range(frames_num):
            
                pixel = frames[n, y, x]
                acc += pixel
                var += pixel**2
                
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


                # Calculate the index for fieldsum, dependent on the deinterlace order (and if there's any
                # detinerlacing at all)
                fieldsum_indx = deinterlace_multiplier*n \
                    + (deinterlace_multiplier - 1)*((y + deinterlace_order)%2)

                # Sum intensity per every field
                fieldsum[fieldsum_indx] += <unsigned long> pixel

            
            
            # Calculate mean without top 4 max values
            acc -= max_val + max_val_2 + max_val_3 + max_val_4
            mean = acc/frames_num_minus_four
            


            
            ### Calculate stddev without top 4 max values ##
            
            # Remove top 4 max values
            var -= max_val**2 + max_val_2**2 + max_val_3**2 + max_val_4**2
            
            # Subtract average squared sum of all values (acc*mean = acc*acc/frames_num_minus_four)
            var -= acc*mean    

            # Compute the standard deviation
            var = <unsigned int> sqrt(var/frames_num_minus_five)

            # Make sure that the stddev is not 0, to prevent divide by zero afterwards
            if var == 0:
                var = 1

            ###
            
            
            # Output results
            ftp_array[0, y, x] = max_val
            ftp_array[1, y, x] = max_frame
            ftp_array[2, y, x] = mean
            ftp_array[3, y, x] = var


    return ftp_array, fieldsum[:frames_num*deinterlace_multiplier]