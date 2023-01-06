""" Using FFT image registration to find larger offsets between the platepar and the image. """

from __future__ import print_function, division, absolute_import

try:
    import imreg_dft
    IMREG_INSTALLED = True

except ImportError:
    IMREG_INSTALLED = False


import os
import sys
import copy
import shutil
import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt

from RMS.Astrometry import ApplyAstrometry
from RMS.Astrometry.Conversions import date2JD, jd2Date, JD2HourAngle, raDec2AltAz
import RMS.ConfigReader as cr
from RMS.Formats import CALSTARS
from RMS.Formats.FFfile import getMiddleTimeFF
from RMS.Formats import Platepar
from RMS.Formats import StarCatalog
from RMS.Math import rotatePoint
from RMS.Logger import initLogging

# Import Cython functions
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
from RMS.Astrometry.CyFunctions import subsetCatalog


log = logging.getLogger('logger')


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

    # If the image registration library is not installed, return nothing
    if not IMREG_INSTALLED:
        log.warning("WARNING:")
        log.warning('The imreg_dft library is not installed! Install it by running either:')
        log.warning(' a) pip install imreg_dft')
        log.warning(' b) conda install -c conda-forge imreg_dft')

        return 0.0, 1.0, 0.0, 0.0


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
    img_ref = constructImage(img_size, reference_list, dot_radius)
    img_mov = constructImage(img_size, moved_list, dot_radius)


    # Run the FFT registration
    try:
        res = imreg_dft.imreg.similarity(img_ref, img_mov)

    except ValueError:
        log.warning('imreg_dft error: The scale correction is too high!')
        return 0.0, 1.0, 0.0, 0.0

    except IndexError:
        log.warning('imreg_dft error: Index out of bounds!')
        return 0.0, 1.0, 0.0, 0.0


    angle = res['angle']
    scale = res['scale']
    translate = res['tvec']

    # Extract translation and rescale it
    translation_x = rescale_factor*translate[1]
    translation_y = rescale_factor*translate[0]


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

        ax3.imshow(res['timg'], cmap='gray')
        ax3.set_title('Transformed')

        ax4.imshow(np.abs(res['timg'].astype(int) - img_ref.astype(int)).astype(np.uint8), cmap='gray')
        ax4.set_title('Difference')

        plt.tight_layout()

        plt.show()


    return angle, scale, translation_x, translation_y




def alignPlatepar(config, platepar, calstars_time, calstars_coords, scale_update=False, show_plot=False):
    """ Align the platepar using FFT registration between catalog stars and the given list of image stars.
    Arguments:
        config:
        platepar: [Platepar instance] Initial platepar.
        calstars_time: [list] A list of (year, month, day, hour, minute, second, millisecond) of the middle of
            the FF file used for alignment.
        calstars_coords: [ndarray] A 2D numpy array of (x, y) coordinates of image stars.
    Keyword arguments:
        scale_update: [bool] Update the platepar scale. False by default.
        show_plot: [bool] Show the comparison between the reference and image synthetic images.
    Return:
        platepar_aligned: [Platepar instance] The aligned platepar.
    """

    # Create a copy of the config not to mess with the original config parameters
    config = copy.deepcopy(config)


    # Try to optimize the catalog limiting magnitude until the number of image and catalog stars are matched
    maxiter = 10
    search_fainter = True
    mag_step = 0.2
    for inum in range(maxiter):

        # Load the catalog stars
        catalog_stars, _, _ = StarCatalog.readStarCatalog(config.star_catalog_path, config.star_catalog_file, \
            lim_mag=config.catalog_mag_limit, mag_band_ratios=config.star_catalog_band_ratios)

        # Get the RA/Dec of the image centre
        _, ra_centre, dec_centre, _ = ApplyAstrometry.xyToRaDecPP([calstars_time], [platepar.X_res/2], \
                [platepar.Y_res/2], [1], platepar, extinction_correction=False)

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

    log.info('Final catalog limiting magnitude: {:.3f}'.format(config.catalog_mag_limit))


    # Find the transform between the image coordinates and predicted platepar coordinates
    res = findStarsTransform(config, calstars_coords, catalog_xy, show_plot=show_plot)
    angle, scale, translation_x, translation_y = res


    ### Update the platepar ###

    platepar_aligned = copy.deepcopy(platepar)

    # Correct the rotation
    platepar_aligned.pos_angle_ref = (platepar_aligned.pos_angle_ref - angle)%360

    # Update the scale if needed
    if scale_update:
        platepar_aligned.F_scale *= scale

    # Compute the new reference RA and Dec
    _, ra_centre_new, dec_centre_new, _ = ApplyAstrometry.xyToRaDecPP([jd2Date(platepar.JD)], \
        [platepar.X_res/2 - platepar.x_poly_fwd[0] - translation_x], \
        [platepar.Y_res/2 - platepar.y_poly_fwd[0] - translation_y], [1], platepar, \
        extinction_correction=False)

    # Correct RA/Dec
    platepar_aligned.RA_d = ra_centre_new[0]
    platepar_aligned.dec_d = dec_centre_new[0]

    # # Update the reference time and hour angle
    # platepar_aligned.JD = jd
    # platepar_aligned.Ho = JD2HourAngle(jd)

    # Recompute the FOV centre in Alt/Az and update the rotation
    platepar_aligned.az_centre, platepar_aligned.alt_centre = raDec2AltAz(platepar.RA_d, \
        platepar.dec_d, platepar.JD, platepar.lat, platepar.lon)
    platepar_aligned.rotation_from_horiz = ApplyAstrometry.rotationWrtHorizon(platepar_aligned)

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
    initLogging(config, 'fftalign_')

    # Get the logger handle
    log = logging.getLogger("logger")
    log.setLevel(logging.INFO)

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
    calstars_list = CALSTARS.readCALSTARS(dir_path, calstars_file)
    calstars_dict = {ff_file: star_data for ff_file, star_data in calstars_list}

    log.info('CALSTARS file: ' + calstars_file + ' loaded!')

    # Extract star list from CALSTARS file from FF file with most stars
    max_len_ff = max(calstars_dict, key=lambda k: len(calstars_dict[k]))

    # Take only X, Y (change order so X is first)
    calstars_coords = np.array(calstars_dict[max_len_ff])[:, :2]
    calstars_coords[:, [0, 1]] = calstars_coords[:, [1, 0]]

    # Get the time of the FF file
    calstars_time = getMiddleTimeFF(max_len_ff, config.fps, ret_milliseconds=True)



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
