""" Image processing routines. """

from __future__ import print_function, division, absolute_import

import os

import numpy as np
import scipy.misc


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




class FlatStruct(object):
    def __init__(self, flat_img, flat_avg):
        """ Structure containing the flat field.

        Arguments:
            flat_img: [ndarray] Flat field.
            flat_avg: [float] Average value of the flat field.

        """

        self.flat_img = flat_img
        self.flat_avg = flat_avg



def loadFlat(dir_path, file_name):
    """ Load the flat field image. 

    Arguments:
        dir_path: [str] Directory where the flat image is.
        file_name: [str] Name of the flat field file.

    Return:
        flat_struct: [Flat struct] Structure containing the flat field info.
    """

    # Load the flat image
    flat_img = scipy.misc.imread(os.path.join(dir_path, file_name))

    # Make sure there are not 0s, as images are divided by flats
    flat_img[flat_img == 0] = 1

    # Convert the flat to float64
    flat_img = flat_img.astype(np.float64)

    # Calculate the average of the flat
    flat_avg = np.mean(flat_img)

    # Init a new Flat structure
    flat_struct = FlatStruct(flat_img, flat_avg)

    return flat_struct





def applyFlat(img, flat_struct):
    """ Apply a flat field to the image.

    Arguments:
        img: [ndarray] Image to flat field.
        flat_struct: [Flat struct] Structure containing the flat field.
        

    Return:
        [ndarray] Flat corrected image.

    """

    input_type = img.dtype

    # Apply the flat
    img = flat_struct.flat_avg*img.astype(np.float64)/flat_struct.flat_img

    # Limit the image values to image type range
    dtype_info = np.iinfo(input_type)
    img = np.clip(img, dtype_info.min, dtype_info.max)

    # Make sure the output array is the same as the input type
    img = img.astype(input_type)

    return img



if __name__ == "__main__":

    import time

    import matplotlib.pyplot as plt

    from RMS.Formats import FFfile
    import RMS.ConfigReader as cr


    # Load config file
    config = cr.parse(".config")

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



    #### Apply the flat

    # Load an FF file
    dir_path = "/home/dvida/DATA/Dropbox/Apps/Elginfield RPi RMS data/ArchivedFiles/CA0001_20171018_230520_894458_detected"
    file_name = "FF_CA0001_20171019_013239_841_0264704.fits"

    ff = FFfile.read(dir_path, file_name)

    # Load the flat
    flat_struct = loadFlat(os.getcwd(), config.flat_file)


    t1 = time.clock()

    # Apply the flat
    img = applyFlat(ff.maxpixel, flat_struct)

    print('Flat time:', time.clock() - t1)

    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()
