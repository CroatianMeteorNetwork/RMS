""" Functions for detection. """

from __future__ import print_function, division, absolute_import

from time import time
import os

import numpy as np
import matplotlib.pyplot as plt


# Try importing numba
try:
    from numba import njit
    NUMBA_AVAILABLE = True

    # If it's available, set the numba debug level to WARNING
    import logging
    logging.getLogger("numba").setLevel(logging.WARNING)

except ImportError:
    NUMBA_AVAILABLE = False



from RMS.Formats.FFfile import selectFFFrames
from RMS.Logger import getLogger
from RMS.Routines import Image
from RMS.Routines import MaskImage
from RMS.Math import vectNorm

# Morphology - Cython init
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
import RMS.Routines.MorphCy as morph


# Get the logger from the main module
log = getLogger("logger")


def loadImageCalibration(dir_path, config, dtype=None, byteswap=False):
    """ Load the mask, dark and flat. 
    
    Arguments:
        dir_path: [str] Path to the directory with calibration.
        config: [ConfigStruct]

    Keyword arguments:
        dtype: [object] Numpy array dtype for the image. None by default, if which case it will be determined
            from the input image.
        byteswap: [bool] If the dark and flat should be byteswapped. False by default, and should be True for
            UWO PNGs.

    Return:
        mask, dark, flat_struct: [tuple of ndarrays]
    """

    mask_path = None
    mask = None

    # Try loading the mask from CaptureFiles directory
    if os.path.exists(os.path.join(dir_path, config.mask_file)):
        mask_path = os.path.join(dir_path, config.mask_file)

    # Try loading the default mask
    elif os.path.exists(os.path.join(config.config_file_path, config.mask_file)):
        mask_path = os.path.join(config.config_file_path, config.mask_file)

    # Load the mask if given
    if mask_path:
        mask = MaskImage.loadMask(mask_path)

        log.info('Loaded mask: {:s}'.format(mask_path))

        # If the mask is all white, set it to None
        if (mask is not None) and np.all(mask.img == 255):
            log.info('Mask is all white, setting it to None.')
            mask = None

    else:
        log.info('No mask file has been found.')
        

    # Try loading the dark frame
    dark = None
    if config.use_dark:

        dark_path = None

        # Check if dark is in the data directory
        if os.path.exists(os.path.join(dir_path, config.dark_file)):
            dark_path = os.path.join(dir_path, config.dark_file)

        # Try loading the default dark
        elif os.path.exists(config.dark_file):
            dark_path = os.path.abspath(config.dark_file)

        if dark_path is not None:

            # Load the dark
            dark = Image.loadDark(*os.path.split(dark_path), dtype=dtype, byteswap=byteswap)

        if dark is not None:
            print('Loaded dark:', dark_path)
            log.info('Loaded dark: {:s}'.format(dark_path))



    # Try loading a flat field image
    flat_struct = None
    if config.use_flat:

        flat_path = None
        
        # Check if there is flat in the data directory
        if os.path.exists(os.path.join(dir_path, config.flat_file)):
            flat_path = os.path.join(dir_path, config.flat_file)
            
        # Try loading the default flat
        elif os.path.exists(config.flat_file):
            flat_path = os.path.abspath(config.flat_file)

        if flat_path is not None:
            
            # Load the flat
            flat_struct = Image.loadFlat(*os.path.split(flat_path), dtype=dtype, byteswap=byteswap)


        if flat_struct is not None:
            print('Loaded flat:', flat_path)
            log.info('Loaded flat: {:s}'.format(flat_path))



    return mask, dark, flat_struct



def binImageCalibration(config, mask, dark, flat_struct):
    """ Bin the calibration images. """

    # Bin the mask
    if mask is not None:
        mask.img = Image.binImage(mask.img, config.detection_binning_factor, 'avg')

    # Bin the dark
    if dark is not None:
        dark = Image.binImage(dark, config.detection_binning_factor, 'avg')

    # Bin the flat
    if flat_struct is not None:
        flat_struct.binFlat(config.detection_binning_factor, 'avg')


    return mask, dark, flat_struct



def htLinePerpendicular(rho, theta, x_inters, y_inters, img_h, img_w):
    """ Compute a parpendicular line to the one given in Hough polar coordinates. The new line will intersect
        the given line in point (x_inters, y_inters).
    
    Arguments:
        rho: [float] Distance of the line from image centre.
        theta: [float] Angle of the line in degrees (positive clockwise from the vertical).
        x_inters: [float] X coordinate of the point on the line described by (rho, theta) where the 
            perpendicular line will intersect.
        y_inters: [float] Y coordinate of the point on the line described by (rho, theta) where the 
            perpendicular line will intersect.
        img_w: [int] Image width.
        img_h: [int] Image height.

    Return:
        (rho, theta): [tuple of floats] Parameters of the perpendicular line.

    """

    x_inters = -img_w/2 - rho*np.cos(np.radians(theta)) + x_inters
    y_inters = -img_h/2 - rho*np.sin(np.radians(theta)) + y_inters

    theta += 90
    theta = theta%360

    # If the direction of the line is close to up/down, use X to compute the rho because X is not defined
    th_check = theta + 45
    if (((th_check > 0) and (th_check < 45)) or ((th_check > 180) and (th_check < 270))) or \
        (x_inters !=0) or (y_inters == 0):
        rho = x_inters/np.cos(np.radians(theta))
    else:
        rho = y_inters/np.sin(np.radians(theta))
        

    return rho, theta




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
        theta = theta + angle_eps

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
    indicesx = np.array(indicesx, dtype=np.uint16)
    indicesy = np.array(indicesy, dtype=np.uint16)

    return (indicesy, indicesx)


def dilateCoordinates(coords, img_h, img_w, width=1):
    """
    Dilate the given coordinates by adding neighboring coordinates.

    Arguments:
        coords: [np.ndarray] An (N, 3) array where each row is [x, y, frame_no].
        img_h: [int] Height of the image (px).
        img_w: [int] Width of the image (px).

    Keyword arguments:
        width: [int] The dilation width (number of steps). Default is 1.

    Returns:
        np.ndarray: A (M, 3) array of dilated coordinates.

    """

        # If the width is 0, return the original coordinates
    if width == 0:
        return coords

    # Ensure width is at least 1
    width = max(1, int(width))
    
    # Generate shifts for dilation
    shift_range = np.arange(-width, width + 1)
    dx, dy = np.meshgrid(shift_range, shift_range)
    shifts = np.column_stack((dx.ravel(), dy.ravel()))
    
    # Generate all shifted coordinates
    coords_xy = coords[:, :2][:, np.newaxis, :]  # Shape: (N, 1, 2)
    shifted_coords = coords_xy + shifts[np.newaxis, :, :]  # Shape: (N, S, 2)
    shifted_coords = shifted_coords.reshape(-1, 2)  # Shape: (N*S, 2)
    
    # Repeat frame_no for each shifted coordinate
    frame_no = coords[:, 2]
    frame_no_repeated = np.repeat(frame_no, shifts.shape[0])
    
    # Combine x, y, and frame_no
    dilated_coords = np.column_stack((shifted_coords, frame_no_repeated))
    
    # Remove duplicate coordinates
    dilated_coords_unique = np.unique(dilated_coords, axis=0)
    
    # Ensure the coordinates are within image boundaries
    x_within_bounds = (dilated_coords_unique[:, 0] >= 0) & (dilated_coords_unique[:, 0] < img_w)
    y_within_bounds = (dilated_coords_unique[:, 1] >= 0) & (dilated_coords_unique[:, 1] < img_h)
    within_bounds = x_within_bounds & y_within_bounds
    dilated_coords_within_bounds = dilated_coords_unique[within_bounds]
    
    return dilated_coords_within_bounds



def checkCentroidBounds(model_pos, img_w, img_h):
    """ Checks if the given position is within the image. 
    
    Arguments:
        model_pos: [array like] (X, Y) coordinate to check.
        img_w: [int] Image width.
        img_h: [int] Image height.

    Return:
        [bool] True if within image, False otherwise.

    """

    # Get the rho, theta of the line perpendicular to the meteor line
    x_inters, y_inters = model_pos

    # If any of the model positions are out of bounds, skip this frame
    if (x_inters < 0) or (x_inters >= img_w) or (y_inters < 0) or (y_inters >= img_h):
        return False

    return True



### Define the function for finding the nonzero positions in a numpy array ###
### Choose the fastest implementation depending on whether numba is available ###

def findNonzeroPositionsNumpy(arr):
    """ Find the nonzero positions in a numpy array using numpy functions.
    
    Arguments:
        arr: [ndarray] Numpy array.

    Return:
        y, x: [tuple of lists] Indices of nonzero elements.
        
    """


    y, x = np.where(arr != 0)

    return y, x


if NUMBA_AVAILABLE:
    
    @njit
    def findNonzeroPositionsNumba(arr):
        """ Find the nonzero positions in a numpy array using numba functions.
        
        Arguments:
            arr: [ndarray] Numpy array.

        Return:
            y, x: [tuple of lists] Indices of nonzero elements.
            
        """

        y = []
        x = []

        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if arr[i, j] != 0:
                    y.append(i)
                    x.append(j)

        return np.array(y), np.array(x)
    
    findNonzeroPositions = findNonzeroPositionsNumba

else:
    findNonzeroPositions = findNonzeroPositionsNumpy

### ###



def getThresholdedStripe3DPoints(config, img_handle, frame_min, frame_max, rho, theta, mask, flat_struct, \
    dark, stripe_width_factor=1.0, centroiding=False, point1=None, point2=None, debug=False):
    """ Threshold the image and get a list of pixel positions and frames of threshold passers. 
        This function handles all input types of data.

    Arguments;
        config: [config object] configuration object (loaded from the .config file).
        img_handle: [FrameInterface instance] Object which has a common interface to various input files.
        frame_min: [int] First frame to process.
        frame_max: [int] Last frame to process.
        rho: [float] Line distance from the center in HT space (pixels).
        theta: [float] Angle in degrees in HT space.
        mask: [ndarray] Image mask.
        flat_struct: [Flat struct] Structure containing the flat field. None by default.
        dark: [ndarray] Dark frame.

    Keyword arguments:
        stripe_width_factor: [float] Multiplier by which the default stripe width will be multiplied. Default
            is 1.0
        centroiding: [bool] If True, the indices will be returned in the centroiding mode, which means
            that point1 and point2 arguments must be given.
        point1: [list] (x, y, frame) Of the first reference point of the detection.
        point2: [list] (x, y, frame) Of the second reference point of the detection.
        debug: [bool] If True, extra debug messages and plots will be shown.
    
    Return:
        xs, ys, zs: [tuple of lists] Indices of (x, y, frame) of threshold passers for every frame.
    """


    img_h, img_w = img_handle.ff.maxpixel.shape

    t_stripe = time()

    # Get indices of stripe pixels around the line of the meteor (this is quite fast)
    stripe_indices = getStripeIndices(rho, theta, stripe_width_factor*config.stripe_width, img_h, img_w)
    
    if debug: 
        strip_indices_time = time() - t_stripe
        print('  - Stripe indices time: {:.4f} s'.format(strip_indices_time))


    # If centroiding should be done, prepare everything for cutting out parts of the image for photometry
    if centroiding:

        t_centroid_prep = time()

        # Compute the unit vector which describes the motion of the meteor in the image domain
        point1 = np.array(point1)
        point2 = np.array(point2)
        motion_vect = point2[:2] - point1[:2]
        motion_vect_unit = vectNorm(motion_vect)

        # Get coordinates of 2 points that describe the line
        x1, y1, z1 = point1
        x2, y2, z2 = point2

        # Compute the average angular velocity in px per frame
        ang_vel = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)/(z2 - z1)

        # Compute the vector describing the length and direction of the meteor per frame
        motion_vect = ang_vel*motion_vect_unit

        if debug:
            centroid_prep_time = time() - t_centroid_prep
            print('  - Centroiding prep time: {:.4f} s'.format(centroid_prep_time))


    # If the FF files is given, extract the points from FF after threshold
    if img_handle.input_type == 'ff':

        # Threshold the FF file
        img_thres = Image.thresholdFF(img_handle.ff, config.k1_det, config.j1_det, mask=mask, \
            mask_ave_bright=False)

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

        # Use findNonzeroPositions to get y and x indices of non-zero elements in stripe
        ys, xs = findNonzeroPositions(stripe)

        # Retrieve the frame (z) values corresponding to the non-zero positions
        # Assuming img_handle.ff.maxframe is a numpy array with the same shape as stripe
        zs = img_handle.ff.maxframe[ys, xs]

        return xs, ys, zs


    # If video frames are available, extract indices on all frames in the given range
    else:

        xs_array = []
        ys_array = []
        zs_array = []

        # Track the time it takes to do specific operations
        frame_conditioning_times = []
        thresholding_times = []
        morph_times = []
        extract_stripe_times = []
        centroiding_times = []
        nonzero_times = []

        # Go through all frames in the frame range
        for fr in range(frame_min, frame_max + 1):


            # Break the loop if outside frame size
            if fr == (img_handle.total_frames - 1):
                break
            
            t_frame_conditioning = time()

            # Set the frame number
            img_handle.setFrame(fr)

            # Load the frame
            fr_img = img_handle.loadFrame()


            # Apply the dark frame
            if dark is not None:
                fr_img = Image.applyDark(fr_img, dark)

            # Apply the flat to frame
            if flat_struct is not None:
                fr_img = Image.applyFlat(fr_img, flat_struct)

            # Mask the image
            if mask is not None:
                fr_img = MaskImage.applyMask(fr_img, mask)
            
            frame_conditioning_times.append(time() - t_frame_conditioning)


            
            t_threshold = time()

            # Threshold the frame (THIS IS CURRENTLY SLOW)
            img_thres = Image.thresholdImg(fr_img, img_handle.ff.avepixel, img_handle.ff.stdpixel, \
                config.k1_det, config.j1_det, mask=mask, mask_ave_bright=False)
            
            thresholding_times.append(time() - t_threshold)


            t_morph = time()

            # Remove lonely pixels
            img_thres = morph.clean(img_thres)

            morph_times.append(time() - t_morph)


            t_extract_stripe = time()

            # Extract the stripe from the thresholded image
            stripe = np.zeros(img_thres.shape, img_thres.dtype)
            stripe[stripe_indices] = img_thres[stripe_indices]

            extract_stripe_times.append(time() - t_extract_stripe)

            # Include more pixels for centroiding and photometry and mask out per frame pixels
            if centroiding:

                t_centroid = time()
                
                # Dilate the pixels in the stripe twice, to include more pixels for photometry
                stripe = morph.dilate(stripe)
                stripe = morph.dilate(stripe)

                # Get indices of the stripe that is perpendicular to the meteor, and whose thickness is the 
                # length of the meteor on this particular frame - this is called stripe_indices_motion

                # Compute the previous, current, and the next linear model position of the meteor on the 
                #   image
                model_pos_prev = point1[:2] + (fr - 1 - z1)*motion_vect
                model_pos = point1[:2] + (fr - z1)*motion_vect
                model_pos_next = point1[:2] + (fr + 1 - z1)*motion_vect

                # Get the rho, theta of the line perpendicular to the meteor line
                x_inters, y_inters = model_pos

                # Check if the previous, current or the next centroids are outside bounds, and if so, skip the
                #   frame
                if (not checkCentroidBounds(model_pos_prev, img_w, img_h)) or \
                    (not checkCentroidBounds(model_pos, img_w, img_h)) or \
                    (not checkCentroidBounds(model_pos_next, img_w, img_h)):

                    centroiding_times.append(time() - t_centroid)
                    continue

                # Get parameters of the perpendicular line to the meteor line
                rho2, theta2 = htLinePerpendicular(rho, theta, x_inters, y_inters, img_h, img_w)

                # Compute the image indices of this position which will be the intersection with the stripe
                #   The width of the line will be 2x the angular velocity
                stripe_length = 6*ang_vel
                if stripe_length < stripe_width_factor*config.stripe_width:
                    stripe_length = stripe_width_factor*config.stripe_width
                stripe_indices_motion = getStripeIndices(rho2, theta2, stripe_length, img_h, img_w)

                # Mark only those parts which overlap both lines, which effectively creates a mask for
                #    photometry an centroiding, excluding other influences
                stripe_new = np.zeros_like(stripe)
                stripe_new[stripe_indices_motion] = stripe[stripe_indices_motion]
                stripe = stripe_new

                centroiding_times.append(time() - t_centroid)


                # if debug:

                #     # Show the extracted stripe
                #     img_stripe = np.zeros_like(stripe)
                #     img_stripe[stripe_indices] = 1
                #     final_stripe = np.zeros_like(stripe)
                #     final_stripe[stripe_indices_motion] = img_stripe[stripe_indices_motion]

                #     plt.imshow(final_stripe)
                #     plt.show()


            # if debug and centroiding:

            #     print(fr)
            #     print('mean stdpixel3:', np.mean(img_handle.ff.stdpixel))
            #     print('mean avepixel3:', np.mean(img_handle.ff.avepixel))
            #     print('mean frame:', np.mean(fr_img))
            #     fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, sharey=True)


            #     fr_img_noavg = Image.applyDark(fr_img, img_handle.ff.avepixel)
            #     #fr_img_noavg = fr_img

            #     # Auto levels
            #     min_lvl = np.percentile(fr_img_noavg[2:, :], 1)
            #     max_lvl = np.percentile(fr_img_noavg[2:, :], 99.0)

            #     # Adjust levels
            #     fr_img_autolevel = Image.adjustLevels(fr_img_noavg, min_lvl, 1.0, max_lvl)

            #     ax1.imshow(stripe, cmap='gray')
            #     ax2.imshow(fr_img_autolevel, cmap='gray')
            #     plt.show()

            #     pass

            t_nonzero = time()

            # Get stripe positions (x, y)
            ys, xs = findNonzeroPositions(stripe)
            zs = np.zeros_like(xs) + fr

            nonzero_times.append(time() - t_nonzero)

            # Add the points to the list
            xs_array.append(xs)
            ys_array.append(ys)
            zs_array.append(zs)


            # if debug:
            #     print('---')
            #     print(stripe.nonzero())
            #     print(xs, ys, zs)


        t_concatenate = time()

        if len(xs_array) > 0:
            
            # Flatten the arrays
            xs_array = np.concatenate(xs_array)
            ys_array = np.concatenate(ys_array)
            zs_array = np.concatenate(zs_array)

        else:
            xs_array = np.array(xs_array)
            ys_array = np.array(ys_array)
            zs_array = np.array(zs_array)

        concatenate_time = time() - t_concatenate


        if debug:
            
            total_tracked_time = 0
            total_tracked_time += strip_indices_time

            if centroiding:
                total_tracked_time += centroid_prep_time

            print('  - Frame conditioning time:')
            print('    - Mean:  {:.4f} +/- {:.4f} s'.format(np.mean(frame_conditioning_times), np.std(frame_conditioning_times)))
            print('    - Total: {:.4f} s'.format(np.sum(frame_conditioning_times)))
            total_tracked_time += np.sum(frame_conditioning_times)

            print('  - Thresholding time:')
            print('    - Mean:  {:.4f} +/- {:.4f} s'.format(np.mean(thresholding_times), np.std(thresholding_times)))
            print('    - Total: {:.4f} s'.format(np.sum(thresholding_times)))
            total_tracked_time += np.sum(thresholding_times)

            print('  - Morphology time:')
            print('    - Mean:  {:.4f} +/- {:.4f} s'.format(np.mean(morph_times), np.std(morph_times)))
            print('    - Total: {:.4f} s'.format(np.sum(morph_times)))
            total_tracked_time += np.sum(morph_times)

            print('  - Extract stripe time:')
            print('    - Mean:  {:.4f} +/- {:.4f} s'.format(np.mean(extract_stripe_times), np.std(extract_stripe_times)))
            print('    - Total: {:.4f} s'.format(np.sum(extract_stripe_times)))
            total_tracked_time += np.sum(extract_stripe_times)
            
            if centroiding:
                print('  - Centroiding time:')
                print('    - Mean:  {:.4f} +/- {:.4f} s'.format(np.mean(centroiding_times), np.std(centroiding_times)))
                print('    - Total: {:.4f} s'.format(np.sum(centroiding_times)))
                total_tracked_time += np.sum(centroiding_times)

            print('  - Nonzero time:')
            print('    - Mean:  {:.4f} +/- {:.4f} s'.format(np.mean(nonzero_times), np.std(nonzero_times)))
            print('    - Total: {:.4f} s'.format(np.sum(nonzero_times)))
            total_tracked_time += np.sum(nonzero_times)

            print('  - Concatenate time: {:.6f} s'.format(concatenate_time))
            total_tracked_time += concatenate_time

            print('  - TOTAL: {:.4f} s'.format(total_tracked_time))



        return xs_array, ys_array, zs_array