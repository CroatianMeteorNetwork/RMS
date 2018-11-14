""" Image processing routines. """

from __future__ import print_function, division, absolute_import

import os
import math

import numpy as np
import scipy.misc

from RMS.Decorators import memoizeSingle




def thresholdImg(maxpixel, avepixel, stdpixel, k1, j1):
    """ Threshold the image with given parameters.
    
    Arguments:
        maxpixel: [2D ndarray]
        avepixel: [2D ndarray]
        stdpixel: [2D ndarray]
        k1: [float] relative thresholding factor (how many standard deviations above mean the maxpixel image 
            should be)
        j1: [float] absolute thresholding factor (how many minimum abuolute levels above mean the maxpixel 
            image should be)
    
    Return:
        [ndarray] thresholded 2D image
    """

    return maxpixel - avepixel > (k1 * stdpixel + j1)


@memoizeSingle
def thresholdFF(ff, k1, j1):
    """ Threshold the FF with given parameters.
    
    Arguments:
        ff: [FF object] input FF image object on which the thresholding will be applied
        k1: [float] relative thresholding factor (how many standard deviations above mean the maxpixel image 
            should be)
        j1: [float] absolute thresholding factor (how many minimum abuolute levels above mean the maxpixel 
            image should be)
    
    Return:
        [ndarray] thresholded 2D image
    """

    return thresholdImg(ff.maxpixel, ff.avepixel, ff.stdpixel, k1, j1)



@np.vectorize
def gammaCorrection(intensity, gamma, bp=0, wp=255):
    """ Correct the given intensity for gamma. 
        
    Arguments:
        intensity: [int] Pixel intensity
        gamma: [float] Gamma.

    Keyword arguments:
        bp: [int] Black point.
        wp: [int] White point.

    Return:
        [float] Gamma corrected image intensity.
    """

    if intensity < 0:
        intensity = 0

    x = (intensity - bp)/(wp - bp)

    if x > 0:

        # Compute the corrected intensity
        return bp + (wp - bp)*(x**(1.0/gamma))

    else:
        return bp



def applyBrightnessAndContrast(img, brightness, contrast):
    """ Applies brightness and contrast corrections to the image. 
    
    Arguments:
        img: [2D ndarray] Image array.
        brightness: [int] A number in the range -255 to 255.
        contrast: [float] A number in the range -255 to 255.

    Return:
        img: [2D ndarray] Image array with the brightness applied.
    """

    contrast = float(contrast)

    # Compute the contrast factor
    f = (259.0*(contrast + 255.0))/(255*(259 - contrast))

    img_type = img.dtype

    # Convert image to float
    img = img.astype(np.float)

    # Apply brightness
    img = img + brightness

    # Apply contrast
    img = f*(img - 128.0) + 128.0

    # Clip the values to 0-255 range
    img = np.clip(img, 0, 255)

    # Preserve image type
    img = img.astype(img_type)

    return img 



def adjustLevels(img_array, minv, gamma, maxv, nbits=None):
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

    if nbits is None:
        
        # Get the bit depth from the image type
        nbits = 8*img_array.itemsize


    input_type = img_array.dtype

    # Calculate maximum image level
    max_lvl = 2**nbits - 1.0

    # Limit the maximum level
    if maxv > max_lvl:
        maxv = max_lvl

    # Check that the image adjustment values are in fact given
    if (minv is None) or (gamma is None) or (maxv is None):
        return img_array

    minv = minv/max_lvl
    maxv = maxv/max_lvl
    interval = maxv - minv
    invgamma = 1.0/gamma

    img_array = img_array.astype(np.float64)

    # Reduce array to 0-1 values
    img_array = np.divide(img_array, max_lvl)

    # Calculate new levels
    img_array = np.divide((img_array - minv), interval)

    # Cut values lower than 0
    img_array[img_array < 0] = 0

    img_array = np.power(img_array, invgamma)

    img_array = np.multiply(img_array, max_lvl)

    # Convert back to 0-maxval values
    img_array = np.clip(img_array, 0, max_lvl)


    # Convert the image back to input type
    img_array.astype(input_type)
        

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



def loadFlat(dir_path, file_name, byteswap=False):
    """ Load the flat field image. 

    Arguments:
        dir_path: [str] Directory where the flat image is.
        file_name: [str] Name of the flat field file.

    Keyword arguments:
        byteswap: [bool] Byteswap the flat image. False by default.

    Return:
        flat_struct: [Flat struct] Structure containing the flat field info.
    """

    # Load the flat image
    flat_img = scipy.misc.imread(os.path.join(dir_path, file_name), -1)

    if byteswap:
        flat_img = flat_img.byteswap()


    # Convert the flat to float64
    flat_img = flat_img.astype(np.float64)

    # Calculate the median of the flat
    flat_avg = np.median(flat_img)

    # Make sure there are no values close to 0, as images are divided by flats
    flat_img[(flat_img < flat_avg/10) | (flat_img < 10)] = flat_avg

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

    # Check that the input image and the flat have the same dimensions, otherwise do not apply it
    if img.shape != flat_struct.flat_img.shape:
        return img

    input_type = img.dtype

    # Apply the flat
    img = flat_struct.flat_avg*img.astype(np.float64)/flat_struct.flat_img

    # Limit the image values to image type range
    dtype_info = np.iinfo(input_type)
    img = np.clip(img, dtype_info.min, dtype_info.max)

    # Make sure the output array is the same as the input type
    img = img.astype(input_type)

    return img




def applyDark(img, dark_img):
    """ Apply the dark frame to an image. 
    
    Arguments:
        img: [ndarray] Input image.
        dark_img: [ndarray] Dark frame.
    """

    # Check that the image sizes are the same
    if img.shape != dark_img.shape:
        return img


    # Save input type
    input_type = img.dtype


    # Convert the image to integer (with negative values)
    img = img.astype(np.int64)

    # Subtract dark
    img -= dark_img.astype(np.int64)

    # Make sure there aren't any values smaller than 0
    img[img < 0] = 0

    # Convert the image back to the input type
    img = img.astype(input_type)


    return img



def deinterlaceOdd(img):
    """ Deinterlaces the numpy array image by duplicating the odd frame. 
    """
    
    # Deepcopy img to new array
    deinterlaced_image = np.copy(img) 

    deinterlaced_image[1::2, :] = deinterlaced_image[:-1:2, :]

    # Move the image one row up
    deinterlaced_image[:-1, :] = deinterlaced_image[1:, :]
    deinterlaced_image[-1, :] = 0

    return deinterlaced_image



def deinterlaceEven(img):
    """ Deinterlaces the numpy array image by duplicating the even frame. 
    """
    
    # Deepcopy img to new array
    deinterlaced_image = np.copy(img)

    deinterlaced_image[:-1:2, :] = deinterlaced_image[1::2, :]

    return deinterlaced_image



def blendLighten(arr1, arr2):
    """ Blends two image array with lighen method (only takes the lighter pixel on each spot).
    """

    # Store input type
    input_type = arr1.dtype

    arr1 = arr1.astype(np.int64)

    temp = arr1 - arr2
    temp[temp > 0] = 0

    new_arr = arr1 - temp
    new_arr = new_arr.astype(input_type)

    return  new_arr




def deinterlaceBlend(img):
    """ Deinterlaces the image by making an odd and even frame, then blends them by lighten method.
    """

    img_odd_d = deinterlaceOdd(img)
    img_even = deinterlaceEven(img)

    proc_img = blendLighten(img_odd_d, img_even)

    return proc_img




def fillCircle(photom_mask, x_cent, y_cent, radius):

    y_min = math.floor(y_cent - 1.41*radius)
    y_max = math.ceil(y_cent + 1.41*radius)

    if y_min < 0: y_min = 0
    if y_max > photom_mask.shape[0]: y_max = photom_mask.shape[0]

    x_min = math.floor(x_cent - 1.41*radius)
    x_max = math.ceil(x_cent + 1.41*radius)

    if x_min < 0: x_min = 0
    if x_max > photom_mask.shape[1]: x_max = photom_mask.shape[1]

    for y in range(y_min, y_max):
        for x in range(x_min, x_max):

            if ((x - x_cent)**2 + (y - y_cent)**2) <= radius**2:
                photom_mask[y, x] = 1

    return photom_mask



def thickLine(img_h, img_w, x_cent, y_cent, length, rotation, radius):
    """ Given the image size, return the mask where indices which are inside a thick rounded line are 1s, and
    the rest are 0s. The Bresenham algorithm is used to compute line indices.

    Arguments:
        img_h: [int] Image height (px).
        img_w: [int] Image width (px).
        x_cent: [float] X centroid.
        y_cent: [float] Y centroid.
        length: [float] Length of the line segment (px).
        rotation: [float] Rotation of the line (deg).
        radius: [float] Aperture radius (px).

    Return:
        photom_mask: [ndarray] Photometric mask.
    """

    # Init the photom_mask array
    photom_mask = np.zeros((img_h, img_w), dtype=np.uint8)

    rotation = np.radians(rotation)

    # Compute the bounding box
    x0 = math.floor(x_cent - np.cos(rotation)*length/2.0)
    y0 = math.floor(y_cent - np.sin(rotation)*length/2.0)

    y1 = math.ceil(y_cent + np.sin(rotation)*length/2.0)
    x1 = math.ceil(x_cent + np.cos(rotation)*length/2.0)

    # Init the photom_mask array
    photom_mask = np.zeros((img_h, img_w))


    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1

    if dx > dy:
        err = dx / 2.0
        while x != x1:

            photom_mask = fillCircle(photom_mask, int(x), int(y), radius)

            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0

        while y != y1:

            photom_mask = fillCircle(photom_mask, int(x), int(y), radius)

            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy        

    photom_mask = fillCircle(photom_mask, int(x), int(y), radius)

    return photom_mask






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
    dir_path = "/home/dvida/Dropbox/Apps/Elginfield RPi RMS data/ArchivedFiles/CA0001_20171018_230520_894458_detected"
    file_name = "FF_CA0001_20171019_092744_161_1118976.fits"

    ff = FFfile.read(dir_path, file_name)

    # Load the flat
    flat_struct = loadFlat(os.getcwd(), config.flat_file)


    t1 = time.clock()

    # Apply the flat
    img = applyFlat(ff.maxpixel, flat_struct)

    print('Flat time:', time.clock() - t1)

    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()



    ### TEST THICK LINE

    x_cent = 20.1
    y_cent = 20

    rotation = 90
    length = 0
    radius = 2


    indices = thickLine(200, 200, x_cent, y_cent, length, rotation, radius)


    plt.imshow(indices)
    plt.show()