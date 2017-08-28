""" Image processing routines. """

from __future__ import print_function, division, absolute_import

import numpy as np


def adjustLevels(img_array, minv, gamma, maxv, nbits=8):
    """ Adjusts levels on image with given parameters.

    Arguments:
        img_array: [ndarray] Input image array.
        minv: [int] Minimum level.
        gamma: [float] gamma value
        Mmaxv: [int] maximum level.

    Keyword arguments:
        nbits: [int] Image bit depth.

    Return:
        [ndarray] Image with adjusted levels.

    """

    # Calculate maximum image level
    max_lvl = 2**nbits - 1.0

    # Check that the image adjustment values are in fact given
    if (minv == None) and (gamma == None) and (maxv == None):
        return img_array

    minv = minv/max_lvl
    maxv = maxv/max_lvl
    interval = maxv - minv
    invgamma = 1.0/gamma

    img_array = img_array.astype(np.float)

    #Reduce array to 0-1 values
    img_array = img_array/max_lvl

    #Calculate new levels
    img_array = ((img_array - minv)/interval)**invgamma 

    img_array = img_array*max_lvl

    #Convert back to 0-255 values
    img_array = np.clip(img_array, 0, max_lvl)

    # WARNING: This limits the number of image levels to 256!
    img_array = img_array.astype(np.uint8)

    return img_array




if __name__ == "__main__":

    import matplotlib.pyplot as plt


    # Generate image data
    img_data = np.zeros(shape=(256, 256))
    for i in range(256):
        img_data[:, i] += i


    plt.imshow(img_data, cmap='gray')
    plt.show()

    # Adjust levels
    img_data = adjustLevels(img_data, 100, 1.2, 240)

    plt.imshow(img_data, cmap='gray')
    plt.show()