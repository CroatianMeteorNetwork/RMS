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
from scipy.ndimage import map_coordinates, zoom
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



def findStarsTransform(config, reference_list, moved_list, img_size=256, dot_radius=2, show_plot=False,
                       distortion_offset=(0.0, 0.0), residual_threshold=5.0):
    """ Given a list of reference and predicted star positions, return a transform (rotation, scale, \
        translation) between the two lists using RANSAC with SimilarityTransform.

    Arguments:
        config: [Config instance]
        reference_list: [2D list] A list of reference (x, y) star coordinates (catalog projected via platepar).
        moved_list: [2D list] A list of moved (x, y) star coordinates (detected stars in image).
    Keyword arguments:
        img_size: [int] Unused, kept for backward compatibility.
        dot_radius: [int] Unused, kept for backward compatibility.
        show_plot: [bool] Show the comparison between reference and transformed positions.
        distortion_offset: [tuple] (x0, y0) offset of distortion center from image center in pixels.
            Rotation and scale are applied around this point. Default (0, 0) uses image center.
            Computed as: x0 = x_poly_fwd[0] * (X_res/2), y0 = x_poly_fwd[1] * (Y_res/2)
        residual_threshold: [float] RANSAC inlier threshold in pixels. Default 5.0.
    Return:
        angle, scale, translation_x, translation_y:
            - angle: [float] Angle of rotation (deg).
            - scale: [float] Image scale difference.
            - translation_x: [float]
            - translation_y: [float]
    """
    from skimage.measure import ransac
    from skimage.transform import SimilarityTransform

    x0, y0 = distortion_offset

    # Distortion center in image coordinates
    dc_x = config.width / 2.0 + x0
    dc_y = config.height / 2.0 + y0

    # Set input types
    ref = np.array(reference_list).astype(float)
    mov = np.array(moved_list).astype(float)

    # Need at least 3 points for SimilarityTransform
    if len(ref) < 3 or len(mov) < 3:
        log.warning('RANSAC registration error: Need at least 3 points')
        return 0.0, 1.0, 0.0, 0.0

    # Center both point sets on distortion center
    # This ensures rotation/scale are computed around the correct point
    ref_centered = ref - np.array([dc_x, dc_y])
    mov_centered = mov - np.array([dc_x, dc_y])

    try:
        model, inliers = ransac(
            (ref_centered, mov_centered),
            SimilarityTransform,
            min_samples=3,
            residual_threshold=residual_threshold,
            max_trials=1000
        )

        angle = np.rad2deg(model.rotation)
        scale = model.scale
        translation_x, translation_y = model.translation

        n_inliers = np.sum(inliers)

    except Exception as e:
        log.warning('RANSAC registration error: {}'.format(str(e)))
        return 0.0, 1.0, 0.0, 0.0

    # Check for unreasonable values
    if scale < 0.5 or scale > 2.0:
        log.warning('RANSAC registration error: Scale out of range ({:.3f})'.format(scale))
        return 0.0, 1.0, 0.0, 0.0

    log.info('Platepar correction:')
    log.info('    Rotation: {:.5f} deg'.format(angle))
    log.info('    Scale: {:.5f}'.format(scale))
    log.info('    Translation X, Y: ({:.2f}, {:.2f}) px'.format(translation_x, translation_y))
    log.info('    Inliers: {}/{}'.format(n_inliers, len(ref)))

    # Plot comparison
    if show_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Apply detected transform to reference points
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        ref_transformed = ref_centered.copy()
        x_rot = scale * (ref_transformed[:, 0] * cos_a - ref_transformed[:, 1] * sin_a)
        y_rot = scale * (ref_transformed[:, 0] * sin_a + ref_transformed[:, 1] * cos_a)
        ref_transformed[:, 0] = x_rot + translation_x + dc_x
        ref_transformed[:, 1] = y_rot + translation_y + dc_y

        # Before correction
        ax1.scatter(ref[:, 0], ref[:, 1], c='blue', marker='o', label='Reference (catalog)', alpha=0.7)
        ax1.scatter(mov[:, 0], mov[:, 1], c='red', marker='x', label='Moved (detected)', alpha=0.7)
        ax1.scatter(dc_x, dc_y, c='green', marker='*', s=200, label='Distortion center')
        ax1.set_title('Before correction')
        ax1.legend()
        ax1.set_xlim(0, config.width)
        ax1.set_ylim(config.height, 0)
        ax1.set_aspect('equal')

        # After correction
        ax2.scatter(ref_transformed[:, 0], ref_transformed[:, 1], c='blue', marker='o',
                    label='Reference (transformed)', alpha=0.7)
        ax2.scatter(mov[:, 0], mov[:, 1], c='red', marker='x', label='Moved (detected)', alpha=0.7)
        ax2.scatter(dc_x, dc_y, c='green', marker='*', s=200, label='Distortion center')

        # Mark outliers
        if inliers is not None:
            outliers = ~inliers
            if np.any(outliers):
                ax2.scatter(mov[outliers, 0], mov[outliers, 1], c='orange', marker='s',
                            s=100, facecolors='none', linewidths=2, label='Outliers')

        ax2.set_title('After correction (inliers: {}/{})'.format(n_inliers, len(ref)))
        ax2.legend()
        ax2.set_xlim(0, config.width)
        ax2.set_ylim(config.height, 0)
        ax2.set_aspect('equal')

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


    # Compute the distortion center offset from image center
    # (see CyFunctions.pyx lines 1501-1511 for the calculation)
    # This is where pos_angle_ref rotates around
    x0 = platepar.x_poly_fwd[0] * (platepar.X_res / 2.0)
    y0 = platepar.x_poly_fwd[1] * (platepar.Y_res / 2.0)

    # Find the transform between the image coordinates and predicted platepar coordinates
    res = findStarsTransform(config, calstars_coords, catalog_xy, show_plot=show_plot,
                             distortion_offset=(x0, y0))
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

    # Correct the rotation FIRST - this changes the pixel-to-sky mapping
    platepar_aligned.pos_angle_ref = (platepar_aligned.pos_angle_ref - angle) % 360

    # Update the scale if needed (before computing translation)
    if scale_update:
        platepar_aligned.F_scale *= scale

    # Compute the new reference RA and Dec using the ROTATION-CORRECTED platepar.
    # Translation tells us how catalog needs to shift to match detected stars,
    # so we shift the platepar pointing in the opposite direction.
    _, ra_centre_new, dec_centre_new, _ = ApplyAstrometry.xyToRaDecPP(
        [jd2Date(platepar_aligned.JD)],
        [platepar_aligned.X_res/2 - translation_x],
        [platepar_aligned.Y_res/2 - translation_y],
        [1], platepar_aligned,  # Use ROTATION-CORRECTED platepar
        extinction_correction=False)

    # Apply the translation correction
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
