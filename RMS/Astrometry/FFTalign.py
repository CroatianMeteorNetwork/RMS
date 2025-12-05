""" Using FFT image registration to find larger offsets between the platepar and the image. """

from __future__ import print_function, division, absolute_import

import os
import sys
import copy
import datetime
import shutil
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, ifft2
from scipy.ndimage import map_coordinates
from skimage.transform import rotate
from skimage.registration import phase_cross_correlation
from skimage.filters import window

from RMS.Astrometry import ApplyAstrometry
from RMS.Astrometry.Conversions import date2JD, jd2Date, JD2HourAngle, raDec2AltAz
import RMS.ConfigReader as cr
from RMS.Formats import CALSTARS
from RMS.Formats.FFfile import getMiddleTimeFF
from RMS.Formats import Platepar
from RMS.Formats import StarCatalog
from RMS.Math import rotatePoint
from RMS.Logger import LoggingManager, getLogger

# Import Cython functions
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
from RMS.Astrometry.CyFunctions import subsetCatalog


log = getLogger('rmslogger')


def addPoint(img, xc, yc, radius):
    """ Add a point to the image. """

    img_w = img.shape[1]
    img_h = img.shape[0]

    sigma, mu = 1.0, 0.0

    # Generate a small array with a gaussian
    grid_arr = np.linspace(-radius, radius, 2*radius + 1, dtype=int)
    x, y = np.meshgrid(grid_arr, grid_arr)
    d = np.sqrt(x**2 + y**2)
    gauss = 255*np.exp(-((d - mu)**2/(2.0*sigma**2)))

    # Overlay the Gaussian on the image
    for xi, i in enumerate(grid_arr):
        for yj, j in enumerate(grid_arr):

            # Compute the coordinates of the point
            xp = int(i + xc)
            yp = int(j + yc)

            # Check that the point is inside the image
            if (xp >=0) and (xp < img_w) and (yp >= 0) and (yp < img_h):

                # Set the value of the gaussian to the image
                img[yp, xp] = max(gauss[yj, xi], img[yp, xp])


    return img




def constructImage(img_size, point_list, dot_radius):
    """ Construct the image that will be fed into the FFT registration algorithm. """

    # Construct images using given star positions. Map coordinates to img_size x img_size image
    img = np.zeros((img_size, img_size), dtype=np.uint8)

    # Add all given points to the imge
    for point in point_list:
        x, y = point
        img = addPoint(img, x, y, dot_radius)

    return img


def logpolarFilter(shape):
    """ Make a radial cosine filter that suppresses low frequencies in the FFT.

    This filter is essential for accurate rotation detection - it suppresses the
    DC component and low frequencies that would otherwise dominate the correlation.
    """
    yy = np.linspace(-np.pi/2, np.pi/2, shape[0])[:, np.newaxis]
    xx = np.linspace(-np.pi/2, np.pi/2, shape[1])[np.newaxis, :]
    rads = np.sqrt(yy**2 + xx**2)
    filt = 1.0 - np.cos(rads)**2
    filt[np.abs(rads) > np.pi/2] = 1
    return filt


def getLogBase(shape, new_r):
    """ Calculate the logarithmic base for the log-polar transform. """
    EXCESS_CONST = 1.1
    old_r = shape[0] * EXCESS_CONST / 2.0
    log_base = np.exp(np.log(old_r) / new_r)
    return log_base


def logpolarTransform(image, shape, log_base):
    """ Apply a log-polar transform to the image.

    Arguments:
        image: Input image (typically FFT magnitude)
        shape: Output shape (rows=angles, cols=radii)
        log_base: Base for logarithmic radial sampling

    Returns:
        Log-polar transformed image
    """
    bgval = np.percentile(image, 1)
    imshape = np.array(image.shape)
    center = imshape[0] / 2.0, imshape[1] / 2.0

    # Build coordinate grids
    theta = np.zeros(shape, dtype=np.float64)
    theta -= np.linspace(0, np.pi, shape[0])[:, np.newaxis]

    radius = np.zeros(shape, dtype=np.float64)
    radius += np.power(log_base, np.arange(shape[1], dtype=float))[np.newaxis, :]

    # Convert polar to cartesian
    y = radius * np.sin(theta) + center[0]
    x = radius * np.cos(theta) + center[1]

    output = np.empty_like(y)
    map_coordinates(image, [y, x], output=output, order=3, mode="constant", cval=bgval)
    return output


def subpixelPeak(cps, rad=2):
    """ Find the subpixel peak location using center-of-mass interpolation.

    Arguments:
        cps: Cross-power spectrum (shifted so peak is near center)
        rad: Radius around peak for center-of-mass calculation

    Returns:
        Subpixel peak coordinates as numpy array [y, x]
    """
    peak = np.unravel_index(np.argmax(cps), cps.shape)
    peak = np.array(peak)

    # Extract sub-region around peak
    y0, x0 = max(0, peak[0]-rad), max(0, peak[1]-rad)
    y1, x1 = min(cps.shape[0], peak[0]+rad+1), min(cps.shape[1], peak[1]+rad+1)
    subarr = cps[y0:y1, x0:x1]

    # Compute center of mass
    col = np.arange(subarr.shape[0])[:, np.newaxis]
    row = np.arange(subarr.shape[1])[np.newaxis, :]
    arrsum = subarr.sum()

    if arrsum == 0:
        return peak.astype(float)

    com_y = np.sum(subarr * col) / arrsum
    com_x = np.sum(subarr * row) / arrsum

    return np.array([y0 + com_y, x0 + com_x])


def phaseCorrelationSubpixel(im0, im1):
    """ Compute phase correlation with subpixel precision.

    Arguments:
        im0: Reference image
        im1: Moved image

    Returns:
        Subpixel shift as numpy array [y, x]
    """
    f0, f1 = fft2(im0), fft2(im1)
    eps = np.abs(f1).max() * 1e-15
    cps = np.abs(ifft2((f0 * f1.conjugate()) / (np.abs(f0) * np.abs(f1) + eps)))
    scps = fftshift(cps)
    peak = subpixelPeak(scps)
    ret = peak - np.array(f0.shape) // 2
    return ret



def findStarsTransform(config, reference_list, moved_list, img_size=256, dot_radius=2, show_plot=False):
    """ Given a list of reference and predicted star positions, return a transform (rotation, scale, \
        translation) between the two lists using FFT image registration. This is achieved by creating a
        synthetic star image using both lists and searching for the transform using phase correlation.

    Arguments:
        config: [Config instance]
        reference_list: [2D list] A list of reference (x, y) star coordinates.
        moved_list: [2D list] A list of moved (x, y) star coordinates.
    Keyword arguments:
        img_size: [int] Power of 2 image size (e.g. 128, 256, etc.) which will be created and fed into the
            FFT registration algorithm.
        dot_radius: [int] The radius of the dot which will be drawn on the synthetic image.
        show_plot: [bool] Show the comparison between the reference and image synthetic images.
    Return:
        angle, scale, translation_x, translation_y:
            - angle: [float] Angle of rotation (deg).
            - scale: [float] Image scale difference.
            - translation_x: [float]
            - translation_y: [float]
    """

    # Set input types
    reference_list = np.array(reference_list).astype(float)
    moved_list = np.array(moved_list).astype(float)

    # Rescale the coordinates so the whole image fits inside the square (rescale by the smaller image axis)
    rescale_factor = min(config.width, config.height)/img_size

    reference_list /= rescale_factor
    moved_list /= rescale_factor

    # Take only those coordinates which are inside img_size/2 distance from the centre, and
    #   shift the coordinates
    shift_x = img_size/2 - config.width/(2*rescale_factor)
    shift_y = img_size/2 - config.height/(2*rescale_factor)

    reference_list[:, 0] += shift_x
    reference_list[:, 1] += shift_y
    moved_list[:, 0] += shift_x
    moved_list[:, 1] += shift_y

    # Construct the reference and moved images
    img_ref = constructImage(img_size, reference_list, dot_radius).astype(np.float64)
    img_mov = constructImage(img_size, moved_list, dot_radius).astype(np.float64)

    # Run the FFT registration using custom imreg_dft-style algorithm
    try:
        # Apply Hann window to reduce spectral leakage from image borders
        winhann = window('hann', img_ref.shape)
        img_ref_windowed = img_ref * winhann
        img_mov_windowed = img_mov * winhann

        # Compute FFT magnitude spectra (translation-invariant representation)
        dft_ref = fftshift(fft2(img_ref_windowed))
        dft_mov = fftshift(fft2(img_mov_windowed))

        # Apply log-polar filter to suppress low frequencies (essential for rotation accuracy)
        filt = logpolarFilter(img_ref.shape)
        dft_ref_filt = dft_ref * filt
        dft_mov_filt = dft_mov * filt

        # Apply custom log-polar transform
        pcorr_shape = (int(max(img_ref.shape)),) * 2
        log_base = getLogBase(img_ref.shape, pcorr_shape[1])

        lp_ref = logpolarTransform(np.abs(dft_ref_filt), pcorr_shape, log_base)
        lp_mov = logpolarTransform(np.abs(dft_mov_filt), pcorr_shape, log_base)

        # Find rotation and scale via phase correlation on log-polar images
        shifts = phaseCorrelationSubpixel(lp_ref, lp_mov)
        arg_ang, arg_rad = shifts

        # Convert shifts to rotation angle and scale factor
        angle = -np.pi * arg_ang / float(pcorr_shape[0])
        angle = np.rad2deg(angle)
        angle = ((angle + 180) % 360) - 180  # Normalize to [-180, 180]

        scale = log_base ** arg_rad

        # Invert to get the transform from reference to moved (matching imreg_dft convention)
        angle = -angle
        scale = 1.0 / scale

        # Check for unreasonable scale values
        if scale < 0.5 or scale > 2.0:
            log.warning('FFT registration error: The scale correction is too high ({:.3f})!'.format(scale))
            return 0.0, 1.0, 0.0, 0.0

        # Apply rotation correction to moved image to find translation
        # Use skimage.transform.rotate which handles center rotation properly
        img_mov_corrected = rotate(img_mov, angle, resize=False, mode='constant', cval=0, order=1)

        # Handle scale correction if significant
        if abs(scale - 1.0) > 0.001:
            from scipy.ndimage import zoom
            img_mov_corrected = zoom(img_mov_corrected, 1.0/scale, order=1)
            # Crop or pad to match original size
            target_shape = img_mov.shape
            current_shape = img_mov_corrected.shape
            if current_shape[0] > target_shape[0] or current_shape[1] > target_shape[1]:
                start_y = (current_shape[0] - target_shape[0]) // 2
                start_x = (current_shape[1] - target_shape[1]) // 2
                img_mov_corrected = img_mov_corrected[
                    max(0, start_y):max(0, start_y)+target_shape[0],
                    max(0, start_x):max(0, start_x)+target_shape[1]
                ]
            if img_mov_corrected.shape[0] < target_shape[0] or img_mov_corrected.shape[1] < target_shape[1]:
                pad_y = target_shape[0] - img_mov_corrected.shape[0]
                pad_x = target_shape[1] - img_mov_corrected.shape[1]
                img_mov_corrected = np.pad(
                    img_mov_corrected,
                    ((pad_y // 2, pad_y - pad_y // 2), (pad_x // 2, pad_x - pad_x // 2)),
                    mode='constant'
                )

        # Find translation between reference and rotation/scale-corrected moved image
        translation_shifts, _, _ = phase_cross_correlation(
            img_ref, img_mov_corrected, upsample_factor=10, normalization=None
        )

        # Extract translation (note: phase_cross_correlation returns (row, col) = (y, x))
        # Keep same sign convention as imreg_dft (transform from moved to reference)
        translate_y, translate_x = translation_shifts[0], translation_shifts[1]

    except (ValueError, IndexError) as e:
        log.warning('FFT registration error: {}'.format(str(e)))
        return 0.0, 1.0, 0.0, 0.0

    # Rescale translation back to original image coordinates
    translation_x = rescale_factor * translate_x
    translation_y = rescale_factor * translate_y

    log.info('Platepar correction:')
    log.info('    Rotation: {:.5f} deg'.format(angle))
    log.info('    Scale: {:.5f}'.format(scale))
    log.info('    Translation X, Y: ({:.2f}, {:.2f}) px'.format(translation_x, translation_y))

    # Plot comparison
    if show_plot:

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

        ax1.imshow(img_ref, cmap='gray')
        ax1.set_title('Reference')

        ax2.imshow(img_mov, cmap='gray')
        ax2.set_title('Moved')

        ax3.imshow(img_mov_corrected, cmap='gray')
        ax3.set_title('Transformed')

        ax4.imshow(np.abs(img_mov_corrected - img_ref), cmap='gray')
        ax4.set_title('Difference')

        plt.tight_layout()

        plt.show()


    return angle, scale, translation_x, translation_y




def alignPlatepar(config, platepar, calstars_time, calstars_coords, scale_update=False, show_plot=False,
                  translation_limit=200, rotation_limit=30):
    """ Align the platepar using FFT registration between catalog stars and the given list of image stars.
    Arguments:
        config:
        platepar: [Platepar instance] Initial platepar.
        calstars_time: [list] A single entry of (year, month, day, hour, minute, second, millisecond) of the middle of
            the FF file used for alignment.
        calstars_coords: [ndarray] A 2D numpy array of (x, y) coordinates of image stars.
    
    
    Keyword arguments:
        scale_update: [bool] Update the platepar scale. False by default.
        show_plot: [bool] Show the comparison between the reference and image synthetic images.
        translation_limit: [int] Maximum allowed translation in pixels. Default is 200.
        rotation_limit: [int] Maximum allowed rotation in degrees. Default is 30.


    Return:
        platepar_aligned: [Platepar instance] The aligned platepar.
    """

    # Create a copy of the config not to mess with the original config parameters
    config = copy.deepcopy(config)

    year, month, day, hour, minute, second, millisecond = calstars_time
    ts = datetime.datetime(year, month, day, hour, minute, second, int(round(millisecond * 1000)))
    J2000 = datetime.datetime(2000, 1, 1, 12, 0, 0)

    # Compute the number of years from J2000
    years_from_J2000 = (ts - J2000).total_seconds()/(365.25*24*3600)
    # log.info('Loading star catalog with years from J2000: {:.2f}'.format(years_from_J2000))

    # Try to optimize the catalog limiting magnitude until the number of image and catalog stars are matched
    maxiter = 10
    search_fainter = True
    mag_step = 0.2
    for inum in range(maxiter):

        # Load the catalog stars
        catalog_stars, _, _ = StarCatalog.readStarCatalog(
            config.star_catalog_path,
            config.star_catalog_file,
            years_from_J2000=years_from_J2000,
            lim_mag=config.catalog_mag_limit,
            mag_band_ratios=config.star_catalog_band_ratios)

        # Get the RA/Dec of the image centre
        _, ra_centre, dec_centre, _ = ApplyAstrometry.xyToRaDecPP([calstars_time], [platepar.X_res/2], \
                [platepar.Y_res/2], [1], platepar, extinction_correction=False, precompute_pointing_corr=True)

        ra_centre = ra_centre[0]
        dec_centre = dec_centre[0]

        # Compute Julian date
        jd = date2JD(*calstars_time)

        # Calculate the FOV radius in degrees
        fov_y, fov_x = ApplyAstrometry.computeFOVSize(platepar)
        fov_radius = ApplyAstrometry.getFOVSelectionRadius(platepar)

        # Take only those stars which are inside the FOV
        filtered_indices, _ = subsetCatalog(catalog_stars, ra_centre, dec_centre, jd, platepar.lat, \
            platepar.lon, fov_radius, config.catalog_mag_limit)

        # Take those catalog stars which should be inside the FOV
        ra_catalog, dec_catalog, _ = catalog_stars[filtered_indices].T
        catalog_xy = ApplyAstrometry.raDecToXYPP(ra_catalog, dec_catalog, jd, platepar)

        catalog_x, catalog_y = catalog_xy
        catalog_xy = np.c_[catalog_x, catalog_y]

        # Cut all stars that are outside image coordinates
        catalog_xy = catalog_xy[catalog_xy[:, 0] > 0]
        catalog_xy = catalog_xy[catalog_xy[:, 0] < config.width]
        catalog_xy = catalog_xy[catalog_xy[:, 1] > 0]
        catalog_xy = catalog_xy[catalog_xy[:, 1] < config.height]


        # If there are more catalog than image stars, this means that the limiting magnitude is too faint
        #   and that the search should go in the brighter direction
        if len(catalog_xy) > len(calstars_coords):
            search_fainter = False
        else:
            search_fainter = True

        # print('Catalog stars:', len(catalog_xy), 'Image stars:', len(calstars_coords), \
        #     'Limiting magnitude:', config.catalog_mag_limit)

        # Search in mag_step magnitude steps
        if search_fainter:
            config.catalog_mag_limit += mag_step
        else:
            config.catalog_mag_limit -= mag_step

    # log.info('Final catalog limiting magnitude: {:.3f}'.format(config.catalog_mag_limit))


    # Find the transform between the image coordinates and predicted platepar coordinates
    res = findStarsTransform(config, calstars_coords, catalog_xy, show_plot=show_plot)
    angle, scale, translation_x, translation_y = res

    # Check if the translation and rotation are within the limits
    if (np.hypot(translation_x, translation_y) > translation_limit) or (abs(angle) > rotation_limit):
        
        log.warning("The translation or rotation is too large! The platepar will not be updated!")
        log.warning("Translation: x = {:.2f}, y = {:.2f} px, limit of {:.2f} px".format(
            translation_x, translation_y, translation_limit))
        log.warning("Rotation: {:.2f} deg, limit of {:.2f} deg".format(angle, rotation_limit))

        return platepar

    # print()
    # print('Angle:', angle)
    # print('Scale:', scale)
    # print('Translation:', translation_x, translation_y)
    

    ### Update the platepar ###

    platepar_aligned = copy.deepcopy(platepar)

    # Correct the rotation
    platepar_aligned.pos_angle_ref = (platepar_aligned.pos_angle_ref - angle)%360

    # Update the scale if needed
    if scale_update:
        platepar_aligned.F_scale *= scale

    # Compute the new reference RA and Dec
    # Translation tells us how catalog needs to shift to match detected stars,
    # so we shift the platepar pointing in the opposite direction
    _, ra_centre_new, dec_centre_new, _ = ApplyAstrometry.xyToRaDecPP([jd2Date(platepar_aligned.JD)], \
        [platepar_aligned.X_res/2 - translation_x], \
        [platepar_aligned.Y_res/2 - translation_y], [1], platepar_aligned, \
        extinction_correction=False)
    

    # print("RA:")
    # print(" - old: {:.5f}".format(ra_centre_old[0]))
    # print(" - new: {:.5f}".format(ra_centre_new[0]))
    # print("Dec:")
    # print(" - old: {:.5f}".format(dec_centre_old[0]))
    # print(" - new: {:.5f}".format(dec_centre_new[0]))

    # Correct RA/Dec
    platepar_aligned.RA_d = ra_centre_new[0]
    platepar_aligned.dec_d = dec_centre_new[0]

    # Recompute the FOV centre in Alt/Az and update the rotation
    platepar_aligned.updateRefAltAz()

    ###

    return platepar_aligned





if __name__ == "__main__":


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Align the platepar with the extracted stars from the CALSTARS file. The FF file in CALSTARS with most detected stars will be used for alignment.")

    arg_parser.add_argument('dir_path', nargs=1, metavar='DIR_PATH', type=str, \
        help='Path to night folder.')

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
        help="Path to a config file which will be used instead of the default one.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    dir_path = cml_args.dir_path[0]


    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, dir_path)

    # Initialize the logger
    log_manager = LoggingManager()
    log_manager.initLogging(config, 'fftalign_')

    # Get the logger handle
    log = getLogger("rmslogger", level="INFO")

    # Get a list of files in the night folder
    file_list = os.listdir(dir_path)

    # Find and load the platepar file
    if config.platepar_name in file_list:

        # Load the platepar
        platepar = Platepar.Platepar()
        platepar_path = os.path.join(dir_path, config.platepar_name)
        platepar.read(platepar_path, use_flat=config.use_flat)

    else:
        log.error('Cannot find the platepar file in the night directory: {}'.format(config.platepar_name))
        sys.exit()


    # Find the CALSTARS file in the given folder
    calstars_file = None
    for calstars_file in file_list:
        if ('CALSTARS' in calstars_file) and ('.txt' in calstars_file):
            break

    if calstars_file is None:
        log.error('CALSTARS file could not be found in the given directory!')
        sys.exit()

    # Load the calstars file
    calstars_data = CALSTARS.readCALSTARS(dir_path, calstars_file)
    calstars_list, ff_frames = calstars_data

    # Bail out gracefully if the CALSTARS list is empty
    if not calstars_list:
        log.warning("FFTalign: CALSTARS list is empty - nothing to align")
        sys.exit()
        
    calstars_dict = {ff_file: star_data for ff_file, star_data in calstars_list}

    log.info('CALSTARS file: ' + calstars_file + ' loaded!')

    # Extract star list from CALSTARS file from FF file with most stars
    max_len_ff = max(calstars_dict, key=lambda k: len(calstars_dict[k]))

    # Take only X, Y (change order so X is first)
    calstars_coords = np.array(calstars_dict[max_len_ff])[:, :2]
    calstars_coords[:, [0, 1]] = calstars_coords[:, [1, 0]]

    # Get the time of the FF file
    calstars_time = getMiddleTimeFF(max_len_ff, config.fps, ret_milliseconds=True, ff_frames=ff_frames)



    # Align the platepar with stars in CALSTARS
    platepar_aligned = alignPlatepar(config, platepar, calstars_time, calstars_coords, show_plot=False)


    # Backup the old platepar
    shutil.copy(platepar_path, platepar_path + '.bak')

    # Save the updated platepar
    platepar_aligned.write(platepar_path)

    ### Testing
    sys.exit()



    # class Config(object):
    #     def __init__(self):
    #         self.width = 1280
    #         self.height = 720



    # config = Config()


    # # Generate some random points as stars
    # npoints = 100
    # reference_list = np.c_[np.random.randint(-100, config.width + 100, size=npoints),
    #                        np.random.randint(-100, config.height + 100, size=npoints)]


    # # Create a list of shifted stars
    # moved_list = np.copy(reference_list)

    # # Move the dots
    # moved_list[:, 0] += 50
    # moved_list[:, 1] += 40

    # # Rotate the moved points
    # rot_angle = np.radians(25)
    # origin = np.array([config.width/2, config.height/2])
    # for i, mv_pt in enumerate(moved_list):
    #     moved_list[i] = rotatePoint(origin, mv_pt, rot_angle)

    # # Choose every second star on the moved list
    # moved_list = moved_list[::2]


    # # Find the transform between the star lists
    # print(findStarsTransform(config, reference_list, moved_list, img_size=128, dot_radius=2))
