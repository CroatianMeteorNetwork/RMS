""" Stacks all maxpixles in the given folder to one image. """

from __future__ import print_function, division, absolute_import

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

from RMS.Formats.FFfile import read as readFF
from RMS.Formats.FFfile import validFFName


def truth_generator():
    """ Generates True/False intermittently by calling:

    gen = truth_generator() 
    gen.next() #True
    gen.next() #False
    gen.next() #True
    ...

    """

    while 1:
        yield True
        yield False


def deinterlaceOdd(ff_image):
    """ Deinterlaces the numpy array image by duplicating the odd frame. 
    """

    truth_gen = truth_generator()
    deinterlaced_image = np.copy(ff_image) #deepcopy ff_image to new array
    old_row = ff_image[0]
    for row_num in range(len(ff_image)):
        if truth_gen.next() == True:
            deinterlaced_image[row_num] = np.copy(ff_image[row_num])
            old_row = ff_image[row_num]
        else:
            deinterlaced_image[row_num] = np.copy(old_row)

    deinterlaced_image = moveArrayOneUp(deinterlaced_image)

    return deinterlaced_image



def deinterlaceEven(ff_image):
    """ Deinterlaces the numpy array image by duplicating the even frame. 
    """

    truth_gen = truth_generator()
    deinterlaced_image = np.copy(ff_image) #deepcopy ff_image to new array
    old_row = ff_image[-1]
    for row_num in reversed(range(len(ff_image))):
        if truth_gen.next() == True:
            deinterlaced_image[row_num] = np.copy(ff_image[row_num])
            old_row = ff_image[row_num]
        else:
            deinterlaced_image[row_num] = np.copy(old_row)

    return deinterlaced_image



def blendLighten(arr1, arr2):
    """ Blends two image array with lighen method (only takes the lighter pixel on each spot).
    """

    arr1 = arr1.astype(np.int16)

    temp = arr1 - arr2 
    temp[temp > 0] = 0
    new_arr = arr1 - temp
    new_arr = new_arr.astype(np.uint8)

    #Return "greater than" values
    return new_arr


def moveArrayOneUp(array):
    """ Moves image array 1 pixel up, and fills the bottom with zeroes.
    """

    array = np.delete(array, (0), axis=0)
    array = np.vstack([array, np.zeros(len(array[0]), dtype = np.uint8)])

    return array



def deinterlaceBlend(image_array):
    """ Deinterlaces the image by making an odd and even frame, then blends them by lighten method.
    """

    image_odd_d = deinterlaceOdd(image_array)
    image_even = deinterlaceEven(image_array)
    full_proc_image = blendLighten(image_odd_d, image_even)

    return full_proc_image


if __name__ == '__main__':

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Stacks all maxpixles in the given folder to one image.")

    arg_parser.add_argument('dir_path', nargs=1, metavar='DIR_PATH', type=str, \
        help='Path to directory with FF files.')

    arg_parser.add_argument('file_format', nargs=1, metavar='FILE_FORMAT', type=str, \
        help='File format of the image, e.g. jpg or png.')

    arg_parser.add_argument('-d', '--deinterlace', action="store_true", help="""Deinterlace the image before stacking. """)

    arg_parser.add_argument('-s', '--subavg', action="store_true", help="""Subtract the average image from maxpixel before stacking. """)

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################


    dir_path = cml_args.dir_path[0]

    first_img = True

    # List all FF files in the current dir
    for ff_name in os.listdir(dir_path):
        if validFFName(ff_name):

            print('Stacking: ', ff_name)

            # Load FF file
            ff = readFF(dir_path, ff_name)

            maxpixel = ff.maxpixel
            avepixel = ff.avepixel

            # Dinterlace the images
            if cml_args.deinterlace:
                maxpixel = deinterlaceBlend(maxpixel)
                avepixel = deinterlaceBlend(avepixel)

            # Subtract the average from maxpixel
            if cml_args.subavg:
                img = maxpixel - avepixel

            else:
                img = maxpixel


            if first_img:
                merge_img = np.copy(img)
                first_img = False
                continue

            # Blend images 'if lighter'
            merge_img = blendLighten(merge_img, img)


    stack_path = os.path.join(dir_path, 'stacked.' + cml_args.file_format[0])

    print("Saving to:", stack_path)
    
    # Save the blended image
    scipy.misc.imsave(stack_path, merge_img)

    # Plot the blended image
    plt.imshow(merge_img, cmap='gray')

    plt.show()




