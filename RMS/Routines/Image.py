""" Image processing routines. """

from __future__ import print_function, division, absolute_import

import os
import sys
import math

import numpy as np
import scipy.misc
import cv2

from PIL import Image, ImageFont, ImageDraw 
import datetime

# Check which imread funtion to use
try:
    imread = scipy.misc.imread
    imsave = scipy.misc.imsave
    USING_SCIPY_IMREAD = True
except AttributeError:
    import imageio
    imread = imageio.imread
    imsave = imageio.imwrite
    USING_SCIPY_IMREAD = False


# Rawpy for DFN images
try:
    import rawpy
except ImportError:
    pass


from RMS.Decorators import memoizeSingle

# Cython init
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
import RMS.Routines.MorphCy as morph
from RMS.Routines.BinImageCy import binImage as binImageCy


def loadRaw(img_path):
    """ Load a raw images such as the DFN NEF or Canon CR2 image. 
    
        Arguments:
            img_path: [str] Path to the image.
    """

    if 'rawpy' in sys.modules:

        # Get raw data from .nef file and get image from it
        # Disable automated levels scaling and image orientation
        raw = rawpy.imread(img_path)
        frame = raw.postprocess(gamma=(1,1), output_bps=16, no_auto_bright=True, no_auto_scale=True, \
            output_color=rawpy.ColorSpace.sRGB, user_flip=0)

        # Convert the image to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        return frame

    else:
        print("WARNING! Rawpy not installed, cannot load raw images! To enable, run:\npip install rawpy")
        print()
        return None


def loadImage(img_path, flatten=-1):
    """
    Load the given image. Handle loading it using different libraries.

    Arguments:
        img_path: [str] Path to the image.

    Keyword arguments:
        flatten: [int] Convert color image to grayscale if -1. -1 by default.
    """

    if USING_SCIPY_IMREAD:
        img = imread(img_path, flatten)

    else:

        try:
            img = imread(img_path, as_gray=bool(flatten))

        except TypeError:
            
            img = imread(img_path)

            # If there more than 16 bits, convert to uint16
            if img.nbytes >= 2**16:
                img = img.astype("uint16")

            # Convert the time to grayscale, making sure to preserve the bit depth
            if img.shape == 3 and flatten == -1:
                img = img.mean(axis=2).astype(img.dtype)

    return img


def saveImage(img_path, img, add_timestamp=False):
    """ Save image to disk.

    Arguments:
        img_path: [str] Image path.
        img: [ndarray] Image as numpy array.
        add_timestamp: [boolean] optionally add a timestamp title to the image

    """

    imsave(img_path, img)
    if add_timestamp is True: 
        my_image = Image.open(img_path)
        try:
            _, height = my_image.size
            image_editable = ImageDraw.Draw(my_image)
            _, fname = os.path.split(img_path)
            splits = fname.split('_')
            dtstr = splits[2] + '_' + splits[3] + '.' + splits[4]
            imgdt = datetime.datetime.strptime(dtstr, '%Y%m%d_%H%M%S.%f')
            title = splits[1] + ' ' + imgdt.strftime('%Y-%m-%d %H:%M:%S UTC') 
            #fnt = ImageFont.truetype("arial.ttf", 15)
            fnt = ImageFont.load_default()
            image_editable.text((15,height-20), title, font=fnt, fill=(255))
        except:
            print('unable to add title')
        my_image.save(img_path)


def binImage(img, bin_factor, method='avg'):
    """ Bin the given image. The binning has to be a factor of 2, e.g. 2, 4, 8, etc.
    This is just a wrapper function for a cythonized function that does the binning.
    
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

    input_type = img.dtype

    # Make sure the input image is of the correct type
    if img.dtype != np.uint16:
        img = img.astype(np.uint16)
    
    # Perform the binning
    img = binImageCy(img, bin_factor, method=method)

    # Convert the image back to the input type
    img = img.astype(input_type)

    return img



def thresholdImg(img, avepixel, stdpixel, k1, j1, ff=False, mask=None, mask_ave_bright=True):
    """ Threshold the image with given parameters.
    
    Arguments:
        img: [2D ndarray]
        avepixel: [2D ndarray]
        stdpixel: [2D ndarray]
        k1: [float] relative thresholding factor (how many standard deviations above mean the maxpixel image 
            should be)
        j1: [float] absolute thresholding factor (how many minimum abuolute levels above mean the maxpixel 
            image should be)

    Keyword arguments:
        ff: [bool] If true, it indicated that the FF file is being thresholded.
        mask: [ndarray] Mask image. None by default.
        mask_ave_bright: [bool] Mask out regions that are 5 sigma brighter in avepixel than the mean.
            This gets rid of very bright stars, saturating regions, static bright parts, etc.
    
    Return:
        [ndarray] thresholded 2D image
    """

    # If the FF file is used, then values in max will always be larger than values in average
    if ff:
        img_avg_sub = img - avepixel
    else:
        
        # Subtract input image and average, making sure there are no values below 0 which will wrap around
        img_avg_sub = applyDark(img, avepixel)

    # Compute the thresholded image
    img_thresh = img_avg_sub > (k1 * stdpixel + j1)


    # Mask out regions that are very bright in avepixel
    if mask_ave_bright:

        # Compute the average saturation mask and mask out everything that's saturating in avepixel
        ave_saturation_mask = avepixel >= np.min([np.mean(avepixel) + 5*np.std(avepixel), \
            np.iinfo(avepixel.dtype).max])

        # Dilate the mask 2 times
        input_type = ave_saturation_mask.dtype
        ave_saturation_mask = morph.morphApply(ave_saturation_mask.astype(np.uint8), [5, 5]).astype(input_type)

        img_thresh = img_thresh & ~ave_saturation_mask


    # If the mask was given, set all areas of the thresholded image convered by the mask to false
    if mask is not None:
        if img_thresh.shape == mask.img.shape:
            img_thresh[mask.img == 0] = False

    # The thresholded image is always 8 bit
    return img_thresh.astype(np.uint8)


@memoizeSingle
def thresholdFF(ff, k1, j1, mask=None, mask_ave_bright=False):
    """ Threshold the FF with given parameters.
    
    Arguments:
        ff: [FF object] input FF image object on which the thresholding will be applied
        k1: [float] relative thresholding factor (how many standard deviations above mean the maxpixel image 
            should be)
        j1: [float] absolute thresholding factor (how many minimum abuolute levels above mean the maxpixel 
            image should be)

    Keyword arguments:
        mask: [ndarray] Mask image. None by default.
        mask_ave_bright: [bool] Mask out regions that are 5 sigma brighter in avepixel than the mean.
            This gets rid of very bright stars, saturating regions, static bright parts, etc.
    
    Return:
        [ndarray] thresholded 2D image
    """

    return thresholdImg(ff.maxpixel, ff.avepixel, ff.stdpixel, k1, j1, ff=True, mask=mask, \
        mask_ave_bright=mask_ave_bright)



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
    img = img.astype(float)

    # Apply brightness
    img = img + brightness

    # Apply contrast
    img = f*(img - 128.0) + 128.0

    # Clip the values to 0-255 range
    img = np.clip(img, 0, 255)

    # Preserve image type
    img = img.astype(img_type)

    return img 



def adjustLevels(img_array, minv, gamma, maxv, nbits=None, scaleto8bits=False):
    """ Adjusts levels on image with given parameters.

    Arguments:
        img_array: [ndarray] Input image array.
        minv: [int] Minimum level.
        gamma: [float] gamma value
        Mmaxv: [int] maximum level.

    Keyword arguments:
        nbits: [int] Image bit depth.
        scaleto8bits: [bool] If True, the maximum value will be scaled to 255 and the image will be converted
            to 8 bits.

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

    # Make sure the interval is at least 10 levels of difference
    if interval*max_lvl < 10:

        minv *= 0.9
        maxv *= 1.1

        interval = maxv - minv
        


    # Make sure the minimum and maximum levels are in the correct range
    if minv < 0:
        minv = 0

    if maxv*max_lvl > max_lvl:
        maxv = 1.0
    

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


    # Scale the image to 8 bits so the maximum value is set to 255
    if scaleto8bits:
        img_array *= 255.0/np.max(img_array)
        img_array = img_array.astype(np.uint8)

    else:

        # Convert the image back to input type
        img_array = img_array.astype(input_type)


    return img_array




class FlatStruct(object):
    def __init__(self, flat_img, dark=None):
        """ Structure containing the flat field.

        Arguments:
            flat_img: [ndarray] Flat field.

        """

        # Convert the flat to float64
        self.flat_img = flat_img.astype(np.float64)

        # Store the original flat
        self.flat_img_raw = np.copy(self.flat_img)

        # Apply the dark, if given
        self.applyDark(dark)

        # Compute the flat median
        self.computeAverage()

        # Fix values close to 0
        self.fixValues()


    def applyDark(self, dark):
        """ Apply a dark to the flat. """

        # Apply a dark frame to the flat, if given
        if dark is not None:
            self.flat_img = applyDark(self.flat_img_raw, dark)
            self.dark_applied = True

        else:
            self.flat_img = np.copy(self.flat_img_raw)
            self.dark_applied = False

        # Compute flat median
        self.computeAverage()

        # Fix values close to 0
        self.fixValues()


    def computeAverage(self):
        """ Compute the reference level. """


        # Bin the flat by a factor of 4 using the average method
        flat_binned = binImage(self.flat_img, 4, method='avg')

        # Take the maximum average level of pixels that are in a square of 1/4*height from the centre
        radius = flat_binned.shape[0]//4
        img_h_half = flat_binned.shape[0]//2
        img_w_half = flat_binned.shape[1]//2
        self.flat_avg = np.max(flat_binned[img_h_half-radius:img_h_half+radius, \
            img_w_half-radius:img_w_half+radius])

        # Make sure the self.flat_avg value is relatively high
        if self.flat_avg < 1:
            self.flat_avg = 1


    def fixValues(self):
        """ Handle values close to 0 on flats. """

        # Make sure there are no values close to 0, as images are divided by flats
        self.flat_img[(self.flat_img < self.flat_avg/10) | (self.flat_img < 10)] = self.flat_avg


    def binFlat(self, binning_factor, binning_method):
        """ Bin the flat. """

        # Bin the processed flat
        self.flat_img = binImage(self.flat_img, binning_factor, binning_method)

        # Bin the raw flat image
        self.flat_img_raw = binImage(self.flat_img_raw, binning_factor, binning_method)



def loadFlat(dir_path, file_name, dtype=None, byteswap=False, dark=None):
    """ Load the flat field image. 

    Arguments:
        dir_path: [str] Directory where the flat image is.
        file_name: [str] Name of the flat field file.

    Keyword arguments:
        dtype: [bool] A given file type fill be force if given (e.g. np.uint16).
        byteswap: [bool] Byteswap the flat image. False by default.

    Return:
        flat_struct: [Flat struct] Structure containing the flat field info.
    """

    # Load the flat image
    flat_img = loadImage(os.path.join(dir_path, file_name), -1)

    # Change the file type if given
    if dtype is not None:
        flat_img = flat_img.astype(dtype)

    # If the flat isn't a 8 bit integer, convert it to uint16
    elif flat_img.dtype != np.uint8:
        flat_img = flat_img.astype(np.uint16)


    if byteswap:
        flat_img = flat_img.byteswap()
        

    # Init a new Flat structure
    flat_struct = FlatStruct(flat_img, dark=dark)

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



def loadDark(dir_path, file_name, dtype=None, byteswap=False):
    """ Load the dark frame. 

    Arguments:
        dir_path: [str] Path to the directory which containes the dark frame.
        file_name: [str] Name of the dark frame file.

    Keyword arguments:
        dtype: [bool] A given file type fill be force if given (e.g. np.uint16).
        byteswap: [bool] Byteswap the dark. False by default.
    
    Return:
        dark: [ndarray] Dark frame.

    """

    try:

        # If the image is a raw file, load it as such
        if file_name.lower().endswith(".nef") or file_name.lower().endswith(".cr2"):

            # Load the dark from a raw file
            dark = loadRaw(os.path.join(dir_path, file_name))

        else:

            # Load the dark image
            dark = loadImage(os.path.join(dir_path, file_name), -1)

    except OSError as e:
        print('Dark could not be loaded:', e)
        return None


    # Change the file type if given
    if dtype is not None:
        dark = dark.astype(dtype)

    # If the flat isn't a 8 bit integer, convert it to uint16
    if dark.dtype != np.uint8:
        dark = dark.astype(np.uint16)


    if byteswap:
        dark = dark.byteswap()


    return dark



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

    # Load the default flat
    flat_struct = loadFlat(config.config_file_path, config.flat_file)


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