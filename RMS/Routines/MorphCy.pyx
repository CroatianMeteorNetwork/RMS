
import numpy as np
import cv2

# Cython import
cimport numpy as np
cimport cython

# Define numpy types
INT_TYPE = np.uint8
ctypedef np.uint8_t INT_TYPE_t



@cython.boundscheck(False)
@cython.wraparound(False) 
def morphApply(np.ndarray[INT_TYPE_t, ndim=2] img, operations):
    """ Apply morphological operations on the given image.

    1 - clean
    2 - brigde
    3 - close
    4 - thin
    5 - dilate

    """

    cdef int operation

    for operation in operations:

        if (operation == 1):
            img = clean(img)

        elif (operation == 2):
            img = bridge(img)

        elif (operation == 3):
            img = close(img)

        elif (operation == 4):
            img = thin(img)

        elif (operation == 5):
            img = dilate(img)


    return img



@cython.boundscheck(False)
@cython.wraparound(False) 
def clean(np.ndarray[INT_TYPE_t, ndim=2] img):
    """ Clean isolated pixels, as in:
     0  0  0      0  0  0
     0  1  0  =>  0  0  0
     0  0  0      0  0  0
    
    @param img: input image
    
    @return cleaned image
    """

    cdef int y, x
    cdef bint p2, p3, p4, p5, p6, p7, p8, p9

    cdef int y_size = img.shape[0]
    cdef int x_size = img.shape[1]

    cdef int ym = y_size - 1
    cdef int xm = x_size - 1

    for y in range(1, img.shape[0]):
        for x in range(1, img.shape[1]):

            # Skip if it is not a bright pixel
            if not img[y, x]:
                continue

            # Get neighbouring pixels
            p2 = 0 if (y == 0)             else img[y-1, x]
            p3 = 0 if (y == 0  or x == xm) else img[y-1, x+1]
            p4 = 0 if (x == xm)            else img[y,   x+1]
            p5 = 0 if (y == ym or x == xm) else img[y+1, x+1]
            p6 = 0 if (y == ym)            else img[y+1, x]
            p7 = 0 if (y == ym or x == 0)  else img[y+1, x-1]
            p8 = 0 if (x == 0)             else img[y,   x-1]
            p9 = 0 if (y == 0  or x == 0)  else img[y-1, x-1]

            if not (p2 or p3 or p4 or p5 or p6 or p7 or p8 or p9):
                img[y, x] = 0

    return img



@cython.boundscheck(False)
@cython.wraparound(False) 
def bridge(np.ndarray[INT_TYPE_t, ndim=2] img):
    """ Connect pixels on opposite sides, if other pixels are 0, as in:
     0  0  1      0  0  1
     0  0  0  =>  0  1  0
     1  0  0      1  0  0
    
    @param img: input image
    
    @return bridged image
    """

    cdef int y, x
    cdef bint p2, p3, p4, p5, p6, p7, p8, p9

    cdef int y_size = img.shape[0]
    cdef int x_size = img.shape[1]

    # Init mask array
    cdef np.ndarray[INT_TYPE_t, ndim=2] mask = np.zeros(shape=(y_size, x_size), dtype=INT_TYPE)

    
    for y in range(1, img.shape[0]-1):
        for x in range(1, img.shape[1]-1):

            # Continue if both the image and the mask pixels are bright
            if (img[y, x] and mask[y, x]):
                continue

            # Get neighbouring pixels
            p2 = img[y-1, x]
            p3 = img[y-1, x+1]
            p4 = img[y,   x+1]
            p5 = img[y+1, x+1]
            p6 = img[y+1, x]
            p7 = img[y+1, x-1]
            p8 = img[y,   x-1]
            p9 = img[y-1, x-1]

            if((p2 and not p3 and not p4 and not p5 and p6 and not p7 and not p8 and not p9) or
               (not p2 and not p3 and not p4 and p5 and not p6 and not p7 and not p8 and p9) or
               (not p2 and not p3 and p4 and not p5 and not p6 and not p7 and p8 and not p9) or
               (not p2 and p3 and not p4 and not p5 and not p6 and p7 and not p8 and not p9)):
                mask[y, x] = 1;
    

    # Perform a bitwise OR on the image and the mask (i.e. "stack" the images)
    np.bitwise_or(img, mask, img)

    return img



@cython.boundscheck(False)
@cython.wraparound(False) 
def close_cy(np.ndarray[INT_TYPE_t, ndim=2] img):
    """ Morphological closing (dilation followed by erosion) with a 3x3 kernel.

        NOTE: Slower than the OpenCV implementation! See the other closing function for a faster 
        implementation.
    
    @param img: input image
    
    @return cleaned image
    """

    cdef int y, x
    cdef int p2, p3, p4, p5, p6, p7, p8, p9

    cdef int y_size = img.shape[0]
    cdef int x_size = img.shape[1]

    # Init mask array
    cdef np.ndarray[INT_TYPE_t, ndim=2] mask = np.zeros(shape=(y_size, x_size), dtype=INT_TYPE)

    # Apply morphological dilation
    for y in range(1, img.shape[0]-1):
        for x in range(1, img.shape[1]-1):

            # Continue if the pixel is bright
            if img[y, x]:
                continue

            # Get neighbouring pixels
            p2 = img[y-1, x]
            p3 = img[y-1, x+1]
            p4 = img[y,   x+1]
            p5 = img[y+1, x+1]
            p6 = img[y+1, x]
            p7 = img[y+1, x-1]
            p8 = img[y,   x-1]
            p9 = img[y-1, x-1]

            # If any of the neighbours are bright, also make the current px bright
            if (p2 or p3 or p4 or p5 or p6 or p7 or p8 or p9):
                mask[y, x] = 1

    # Apply the mask
    np.bitwise_or(img, mask, img)

    # Set all values in mask to 1
    mask.fill(1)

    # Apply morphological erosion
    for y in range(1, img.shape[0]-1):
        for x in range(1, img.shape[1]-1):

            # Continue if the pixel is not bright
            if not img[y, x]:
                continue

            # Get neighbouring pixels
            p2 = img[y-1, x]
            p3 = img[y-1, x+1]
            p4 = img[y,   x+1]
            p5 = img[y+1, x+1]
            p6 = img[y+1, x]
            p7 = img[y+1, x-1]
            p8 = img[y,   x-1]
            p9 = img[y-1, x-1]

            # If any of the neighbours are dark, also make the current px dark
            if not (p2 and p3 and p4 and p5 and p6 and p7 and p8 and p9):
                mask[y, x] = 0

    # Apply the mask
    np.bitwise_and(img, mask, img)

    return img



@cython.boundscheck(False)
@cython.wraparound(False) 
def close(np.ndarray[INT_TYPE_t, ndim=2] img):
    """ Morphological closing (dilation followed by erosion) with OpenCV.
    
    @param image: input image
    
    @return closed image
    """
    
    kernel = np.ones((3, 3), np.uint8)
    
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    
    return img


@cython.boundscheck(False)
@cython.wraparound(False) 
def dilate(np.ndarray[INT_TYPE_t, ndim=2] img):
    """ Morphological dilation with OpenCV.
    
    @param image: input image
    
    @return dilated image
    """
    
    kernel = np.ones((3, 3), np.uint8)
    
    img = cv2.dilate(img, kernel)
    
    return img


@cython.boundscheck(False)
@cython.wraparound(False) 
def thin(np.ndarray[INT_TYPE_t, ndim=2] img):
    """ Zhang-Suen fast thinning algorithm. """

    cdef int y, x
    cdef int p2, p3, p4, p5, p6, p7, p8, p9
    cdef int A, B, m1, m2

    cdef int iteration

    cdef int y_size = img.shape[0]
    cdef int x_size = img.shape[1]

    # Init mask array
    cdef np.ndarray[INT_TYPE_t, ndim=2] mask = np.zeros(shape=(y_size, x_size), dtype=INT_TYPE)

    # Previous thinning solution
    cdef np.ndarray[INT_TYPE_t, ndim=2] previous = np.zeros(shape=(y_size, x_size), dtype=INT_TYPE)    

    while True:

        for iteration in range(2):

            for y in range(1, img.shape[0]-1):
                for x in range(1, img.shape[1]-1):

                    # Get neighbouring pixels
                    p2 = img[y-1, x]
                    p3 = img[y-1, x+1]
                    p4 = img[y,   x+1]
                    p5 = img[y+1, x+1]
                    p6 = img[y+1, x]
                    p7 = img[y+1, x-1]
                    p8 = img[y,   x-1]
                    p9 = img[y-1, x-1]

                    A = ((p2 == 0 and p3 == 1) + (p3 == 0 and p4 == 1) +
                        (p4 == 0 and p5 == 1) + (p5 == 0 and p6 == 1) +
                        (p6 == 0 and p7 == 1) + (p7 == 0 and p8 == 1) +
                        (p8 == 0 and p9 == 1) + (p9 == 0 and p2 == 1))

                    B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9

                    m1 = (p2*p4*p6) if (iteration == 0) else (p2*p4*p8)
                    m2 = (p4*p6*p8) if (iteration == 0) else (p2*p6*p8)

                    if (A == 1 and B >= 2 and B <= 6 and m1 == 0 and m2 == 0):
                        mask[y, x] = 1

        
            # Bitwise AND image with inverted mask
            img = (img & ~mask)

            # Reset mask to 0
            mask.fill(0)

        # Check the difference with the previous thinning solution
        diff = np.absolute(img - previous)
        if np.sum(diff) == 0:
            break

        # Set previous
        previous = np.copy(img)

    return img
