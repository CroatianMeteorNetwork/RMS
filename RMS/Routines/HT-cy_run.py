import sys, os
import numpy as np
from time import time
import cv2
import matplotlib.pyplot as plt

# Cython init
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})

from RMS.Routines.PP_HT_cy import pixelPairHT, generateAtan2Lookup, generateTrigLookup

from RMS.Detection import thresholdImg, show2
from RMS.Formats import FFbin
from RMS.Routines import MorphologicalOperations as morph

def pixelPairHT_wrapper(img_array, img_h, img_w, ht_sigma_factor, ht_sigma_abs, sub_factor, delta):
    """ Wraps the Cython funtion for:
        Do a pixel pair Hough Transform. Sacrifices processor time (N**2 operations), but removes butterfly
        noise, which is nice.

    @param img_array: [2D ndarray] image on which to perfrom the pixel pair HT
    @param img_h: [int] image height in pixels
    @param img_W: [int] image width in pixels
    @param ht_sigma_factor: [int] standard deviations above avreage in HT space to take the line as valid
    @param ht_sigma_abs: [int] minimum absolute counts above the usual threshold
    @param sub_factor: [int] subdivision factor of the HT space for local peak estimates
    @param delta: [float] subdivision of the HT space (e.g. if delta = 0.5, HT space will be subdivided every half degree)

    @return ht_lines: [2D ndarray] (rho, theta, count) which define the line in Hough space and their respective counts
    """

    # Delta should never be 0, always must be > 0
    if delta <= 0:
        print 'Delta must be larger than 0!'
        return []

    # Get indices of pixel exceedances
    points = np.transpose(np.where(img_array)).astype(np.int16)
    points[:,[0, 1]] = points[:,[1, 0]]

    print points.shape

    # Generate arctan2 lookup
    atan2_lookup = generateAtan2Lookup(img_h, img_w)

    # Generate sin lookup table
    sin_lookup = generateTrigLookup(np.sin, (0, 360), delta)

    # Generate cos lookup table
    cos_lookup = generateTrigLookup(np.cos, (0, 360), delta)

    t1 = time()

    ht_lines = pixelPairHT(points, img_h, img_w, ht_sigma_factor, ht_sigma_abs, sub_factor, atan2_lookup, 
        sin_lookup, cos_lookup, delta)

    print 'ht time: ', time() - t1

    return ht_lines



# points = np.array([[40, 50], [50, 60], [41, 51], [51, 61], [45, 55], [55, 65], [74, 56], [98, 13]], np.int16)
# print points

# # Generate fake points
# for i in range(1):
#     pnts = np.arange(0, 300)
#     if i == 0:
#         points = np.array((pnts, pnts), np.int16).reshape(-1, 2)
#     else:
#         points = np.vstack((points, np.array((pnts, pnts+i*2), np.int16).reshape(-1, 2)))

# print points.shape

if __name__ == "__main__":

    delta = 0.5
    ht_sigma_factor = 7
    ht_sigma_abs = 10
    sub_factor = 10

    if len(sys.argv) == 1:
        print "Usage: python -m RMS.Routines.HT-cy_run /path/to/bin/files/"
        sys.exit()
    
    # Get paths to every FF bin file in a directory 
    ff_list = [ff for ff in os.listdir(sys.argv[1]) if ff[0:2]=="FF" and ff[-3:]=="bin"]

    # Check if there are any file in the directory
    if(len(ff_list) == None):
        print "No files found!"
        sys.exit()

    # Run meteor search on every file
    for ff_name in ff_list:

        print ff_name

        # Load FF bin
        ff = FFbin.read(sys.argv[1], ff_name)

        thresh_img = thresholdImg(ff, 1.6, 9)

        thresh_img = morph.clean(thresh_img)

        plt.imshow(thresh_img, cmap='gray')
        plt.show()

        # Clear plot memory
        plt.clf()
        plt.close()

        ht_lines = pixelPairHT_wrapper(thresh_img, ff.nrows, ff.ncols, ht_sigma_factor, ht_sigma_abs, sub_factor, delta)



        print ht_lines
        print ht_lines.shape

        # Skip if not lines found
        if not len(ht_lines):
            continue

        max_lines = 500

        hh = ff.nrows/2.0
        hw = ff.ncols/2.0
        thresh_img = thresh_img.astype(np.uint8)*255

        c = 0
        for rho, theta, count in ht_lines:
            c += 1
            if c > max_lines:
                break

            theta = np.deg2rad(theta)
            
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho

            x1 = int(x0 + 1000*(-b) + hw)
            y1 = int(y0 + 1000*(a) + hh)
            x2 = int(x0 - 1000*(-b) + hw)
            y2 = int(y0 - 1000*(a) + hh)
            
            cv2.line(thresh_img, (x1, y1), (x2, y2), (255, 0, 255), 1)
            
        show2("PP-HT", thresh_img)