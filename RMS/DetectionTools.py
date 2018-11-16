""" Functions for detection. """

from __future__ import print_function, division, absolute_import


import numpy as np
import matplotlib.pyplot as plt


from RMS.Formats.FFfile import selectFFFrames
from RMS.Routines import Image
from RMS.Routines import MaskImage

# Morphology - Cython init
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
import RMS.Routines.MorphCy as morph




def getStripeIndices(rho, theta, stripe_width, img_h, img_w):
    """ Get indices of the stripe centered on a line. Line parameters are in Hough Transform form.
    
    Arguments:
        rho: [float] Line distance from the center in HT space (pixels).
        theta: [float] Angle in degrees in HT space.
        stripe_width: [int] Width of the stripe around the line.
        img_h: [int] Original image height in pixels.
        img_w: [int] Original image width in pixels.

    Return:
        (indicesy, indicesx): [tuple] a tuple of x and y indices of stripe pixels

    """

    # minimum angle offset from 90 degrees
    angle_eps = 0.2

    # Check for vertical/horizontal lines and set theta to a small angle
    if (theta%90 < angle_eps):
        theta = 90 + angle_eps

    # Normalize theta to 0-360 range
    theta = theta%360

    hh = img_h/2.0
    hw = img_w/2.0

    indicesy = []
    indicesx = []
     
    if theta < 45 or (theta > 90 and theta < 135):

        theta = np.radians(theta)
        half_limit = (stripe_width/2)/np.cos(theta)

        a = -np.tan(theta)
        b = rho/np.cos(theta)
         
        for y in range(int(-hh), int(hh)):

            x0 = a*y + b
             
            x1 = int(x0 - half_limit + hw)
            x2 = int(x0 + half_limit + hw)
             
            if x1 > x2:
                x1, x2 = x2, x1
             
            if x2 < 0 or x1 >= img_w:
                continue
            
            for x in range(x1, x2):
                if x < 0 or x >= img_w:
                    continue
                 
                indicesy.append(y + hh)
                indicesx.append(x)
                 
    else:

        theta = np.radians(theta)
        half_limit = (stripe_width/2)/np.sin(theta)

        a = -1/np.tan(theta)
        b = rho/np.sin(theta)
         
        for x in range(int(-hw), int(hw)):
            y0 = a*x + b
             
            y1 = int(y0 - half_limit + hh)
            y2 = int(y0 + half_limit + hh)
             
            if y1 > y2:
                y1, y2 = y2, y1
             
            if y2 < 0 or y1 >= img_h:
                continue
                
            for y in range(y1, y2):
                if y < 0 or y >= img_h:
                    continue
                 
                indicesy.append(y)
                indicesx.append(x + hw)

    # Convert indices to integer
    indicesx = list(map(int, indicesx))
    indicesy = list(map(int, indicesy))

    return (indicesy, indicesx)



def getThresholdedStripe3DPoints(config, img_handle, frame_min, frame_max, rho, theta, mask, flat_struct, stripe_width_factor=1.0):
    """ Threshold the image and get a list of pixel positions and frames of threshold passers. 
        This function handles all input types of data.

    """


    # Get indices of stripe pixels around the line
    img_h, img_w = img_handle.ff.maxpixel.shape
    stripe_indices = getStripeIndices(rho, theta, stripe_width_factor*config.stripe_width, img_h, img_w)

    # If the FF files is given, extract the points from FF after threshold
    if img_handle.input_type == 'ff':

        # Threshold the FF file
        img_thres = Image.thresholdFF(img_handle.ff, config.k1_det, config.j1_det)

        # Extract the thresholded image by min and max frames from FF file
        img = selectFFFrames(np.copy(img_thres), img_handle.ff, frame_min, frame_max)

        # Remove lonely pixels
        img = morph.clean(img)

        # Extract the stripe from the thresholded image
        stripe = np.zeros(img.shape, img.dtype)
        stripe[stripe_indices] = img[stripe_indices]

        # Show stripe
        # show2("stripe", stripe*255)

        # Show 3D could
        # show3DCloud(ff, stripe)

        # Get stripe positions (x, y, frame)
        stripe_positions = stripe.nonzero()
        xs = stripe_positions[1]
        ys = stripe_positions[0]
        zs = img_handle.ff.maxframe[stripe_positions]

        return xs, ys, zs


    # If video frames are available, extract indices on all frames in the given range
    else:

        xs_array = []
        ys_array = []
        zs_array = []

        # Go through all frames in the frame range
        for fr in range(frame_min, frame_max + 1):


            # Break the loop if outside frame size
            if fr == (img_handle.total_frames - 1):
                break

            # Set the frame number
            img_handle.setFrame(fr)

            # Load the frame
            fr_img = img_handle.loadFrame()

            # Mask the image
            fr_img = MaskImage.applyMask(fr_img, mask)

            # Apply the flat to frame
            if flat_struct is not None:

                fr_img = Image.applyFlat(fr_img, flat_struct)
                


            # plt.imshow(fr_img, cmap='gray', vmin=100, vmax=10000)
            # plt.show()

            # Threshold the frame
            img_thres = Image.thresholdImg(fr_img, img_handle.ff.avepixel, img_handle.ff.stdpixel, \
                config.k1_det, config.j1_det)


            print(fr)
            fig, (ax1, ax2) = plt.subplots(nrows=2)
            ax1.imshow(img_thres, cmap='gray')
            ax2.imshow(fr_img, cmap='gray', vmax=10000)
            plt.show()

            # Remove lonely pixels
            img_thres = morph.clean(img_thres)


            # Extract the stripe from the thresholded image
            stripe = np.zeros(img_thres.shape, img_thres.dtype)
            stripe[stripe_indices] = img_thres[stripe_indices]

            # plt.imshow(stripe, cmap='gray')
            # plt.show()

            # Get stripe positions (x, y, frame)
            stripe_positions = stripe.nonzero()
            xs = stripe_positions[1]
            ys = stripe_positions[0]
            zs = np.zeros_like(xs) + fr

            # Add the points to the list
            xs_array.append(xs)
            ys_array.append(ys)
            zs_array.append(zs)

        
        # Flatten the arrays
        xs_array = np.concatenate(xs_array)
        ys_array = np.concatenate(ys_array)
        zs_array = np.concatenate(zs_array)


        return xs_array, ys_array, zs_array