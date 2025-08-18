import numpy as np
# import cv2

# Cython import
cimport numpy as np
cimport cython

# Define numpy types
INT16_TYPE = np.uint16
ctypedef np.uint16_t INT16_TYPE_t

INT32_TYPE = np.uint32
ctypedef np.uint32_t INT32_TYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def binImage(np.ndarray[INT16_TYPE_t, ndim=2] img, int bin_factor, method='avg'):
    """ Bin the given image. The binning has to be a factor of 2, e.g. 2, 4, 8, etc.
    
    Arguments:
        img: [ndarray] Numpy array representing an image.
        bin_factor: [int] The binning factor. Has to be a factor of 2 (e.g. 2, 4, 8).

    Keyword arguments:
        method: [str] Binning method.  'avg' by default.
            - 'sum' will sum all values in the binning window and assign it to the new pixel.
            - 'avg' will take the average.

    Return:
        out_img: [ndarray] Binned image.
    """

    cdef int i, j
    cdef int img_h = img.shape[0]
    cdef int img_w = img.shape[1]



    # If the bin factor is 1, do nothing
    if bin_factor == 1:
        return img

    # Check that the given bin size is a factor of 2
    if np.log2(bin_factor)/int(np.log2(bin_factor)) != 1:
        print('The given binning factor is not a factor of 2!')
        return img

    # Init the new image size
    cdef np.ndarray[INT32_TYPE_t, ndim=2] out_img = np.zeros((img_h//bin_factor, img_w//bin_factor), \
        dtype=INT32_TYPE)


    # Go through all pixels and perform the binning
    for i in range(img_h):
        for j in range(img_w):
            out_img[i//bin_factor, j//bin_factor] += img[i, j]


    if method == 'avg':
        
        # If the average is needed, divide the whole image by the bin factor^2
        out_img = out_img//(bin_factor**2)


    return out_img