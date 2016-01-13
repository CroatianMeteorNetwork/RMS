

####
# TO DO:
# - analyze results and find hot pixels ("stars" which don't move through the night), hold a record of 
#   hot pixels and apply the correction on subsequent star extractions

###

import time
import sys
import os
import cv2
from skimage import feature, data
import matplotlib.pyplot as plt
import numpy as np

import skimage.morphology as morph
import skimage.exposure as skie

# RMS imports
from RMS.Formats import FFbin


def maskBright(ff, max_abs_chunk_intensity=80, max_global_intensity=80, divider=16):
    """ Masks too bright parts of the image so that star extraction isn't performed on them. """

    # Generate indices for subdivision
    x_range = np.arange(0, ff.ncols, divider)
    y_range = np.arange(0, ff.nrows, divider)
    x_range[0] = 0
    y_range[0] = 0

    avepixel_cpy = np.copy(ff.avepixel)

    # Calculate image mean and stddev
    global_mean = np.mean(avepixel_cpy)

    # Check if the image is too bright
    if global_mean > max_global_intensity:
        return False

    global_std = np.std(avepixel_cpy)

    for x in x_range:
        for y in y_range:

            # Extract image segment
            img_chunk = ff.avepixel[y : y+divider, x : x+divider]
            chunk_mean = np.mean(img_chunk)

            # Check if the image sigment is too bright
            if (chunk_mean > global_mean  + 2 * global_std or chunk_mean > max_abs_chunk_intensity):
                avepixel_cpy[y : y+divider, x : x+divider] = global_mean

    return avepixel_cpy

if __name__ == "__main__":


    if len(sys.argv) == 1:
        print "Usage: python -m RMS.ExtractStars /path/to/bin/files/"
        sys.exit()
    
    # Get paths to every FF bin file in a directory 
    ff_list = [ff for ff in os.listdir(sys.argv[1]) if ff[0:2]=="FF" and ff[-3:]=="bin"]

    # Check if there are any file in the directory
    if(len(ff_list) == None):
        print "No files found!"
        sys.exit()


    for ff_name in sorted(ff_list):

        print ff_name

        # Load the FF bin file
        ff = FFbin.read(sys.argv[1], ff_name)

        t1 = time.clock()

        # Mask too bright regions of the image
        masked_average = maskBright(ff)

        # Continue if the image is too bright
        if not np.any(masked_average):
            continue

        limg = np.arcsinh(masked_average.astype(np.float32))
        limg = limg / limg.max()
        low = np.percentile(limg, 1.0)
        high = np.percentile(limg, 99.5)
        opt_img  = skie.exposure.rescale_intensity(limg, in_range=(low,high))

        lm = feature.peak_local_max(limg, min_distance=20, num_peaks=40)

        # Skip if no local maxima found
        if not np.any(lm):
            continue

        y1, x1 = np.hsplit(lm, 2)

        v = limg[(y1,x1)]
        lim = 0.7
        x2, y2 = x1[v > lim], y1[v > lim]

        print 'Time for finding: ', time.clock() - t1

        print x2, y2

        # # Adaptive thresholding on average image
        # avg_thresholded = cv2.adaptiveThreshold(ff.avepixel,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,5,2) * -1 + 255

        # stars = feature.blob_doh(avg_thresholded, min_sigma=1, max_sigma=3, threshold=0.001)

        # Plot image
        plt.imshow(opt_img, cmap='gray')

        # Plot stars
        for star in zip(list(y2), list(x2)):
            y, x = star
            c = plt.Circle((x, y), 5, fill=False, color='r')
            plt.gca().add_patch(c)

        plt.show()

        plt.clf()
        plt.close()


