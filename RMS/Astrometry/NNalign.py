""" Align platepar using nearest-neighbor optimization.

This module provides platepar alignment by fitting pointing parameters (RA, Dec, rotation)
using a nearest-neighbor cost function. This uses direct optimization to minimize the
distances between detected stars and their nearest catalog matches.
"""

from __future__ import print_function, division, absolute_import

import os
import sys
import copy
import datetime
import argparse

import numpy as np

from RMS.Astrometry import ApplyAstrometry
from RMS.Astrometry.Conversions import date2JD
import RMS.ConfigReader as cr
from RMS.Formats import CALSTARS
from RMS.Formats.FFfile import getMiddleTimeFF
from RMS.Formats import Platepar
from RMS.Formats import StarCatalog
from RMS.Logger import LoggingManager, getLogger

# Import Cython functions
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
from RMS.Astrometry.CyFunctions import subsetCatalog


log = getLogger('rmslogger')


def alignPlatepar(config, platepar, calstars_time, calstars_coords, scale_update=False, show_plot=False):
    """ Align the platepar using nearest-neighbor optimization.

    This function fits the platepar pointing parameters (RA_d, dec_d, pos_angle_ref) by minimizing
    the sum of angular separations between detected stars and their nearest catalog neighbors.

    Arguments:
        config: [Config instance]
        platepar: [Platepar instance] Initial platepar.
        calstars_time: [list] A single entry of (year, month, day, hour, minute, second, millisecond)
            of the middle of the FF file used for alignment.
        calstars_coords: [ndarray] A 2D numpy array of (x, y) coordinates of image stars.

    Keyword arguments:
        scale_update: [bool] Update the platepar scale. False by default.
        show_plot: [bool] Unused, kept for backward compatibility.

    Return:
        platepar_aligned: [Platepar instance] The aligned platepar, or original if fit failed.
    """

    # Create a copy of the config not to mess with the original config parameters
    config = copy.deepcopy(config)

    year, month, day, hour, minute, second, millisecond = calstars_time
    ts = datetime.datetime(year, month, day, hour, minute, second, int(round(millisecond * 1000)))
    J2000 = datetime.datetime(2000, 1, 1, 12, 0, 0)

    # Compute the number of years from J2000
    years_from_J2000 = (ts - J2000).total_seconds() / (365.25 * 24 * 3600)

    # Compute Julian date
    jd = date2JD(*calstars_time)

    # Load the catalog stars
    catalog_stars, _, _ = StarCatalog.readStarCatalog(
        config.star_catalog_path,
        config.star_catalog_file,
        years_from_J2000=years_from_J2000,
        lim_mag=config.catalog_mag_limit,
        mag_band_ratios=config.star_catalog_band_ratios)

    # Get the RA/Dec of the image centre
    _, ra_centre, dec_centre, _ = ApplyAstrometry.xyToRaDecPP(
        [calstars_time], [platepar.X_res / 2], [platepar.Y_res / 2], [1], platepar,
        extinction_correction=False, precompute_pointing_corr=True)

    ra_centre = ra_centre[0]
    dec_centre = dec_centre[0]

    # Calculate the FOV radius in degrees
    fov_radius = ApplyAstrometry.getFOVSelectionRadius(platepar)

    # Take only those stars which are inside the FOV (with margin for alignment)
    fov_radius_margin = fov_radius * 1.5  # Larger margin for initial alignment
    filtered_indices, _ = subsetCatalog(catalog_stars, ra_centre, dec_centre, jd, platepar.lat,
                                        platepar.lon, fov_radius_margin, config.catalog_mag_limit)

    catalog_stars_fov = catalog_stars[filtered_indices]

    if len(catalog_stars_fov) < 5:
        log.warning("alignPlatepar: Not enough catalog stars in FOV ({})".format(len(catalog_stars_fov)))
        return platepar

    # Convert calstars_coords to img_stars format (x, y, intensity)
    # Add dummy intensity column
    img_stars = np.column_stack([calstars_coords, np.ones(len(calstars_coords))])

    if len(img_stars) < 5:
        log.warning("alignPlatepar: Not enough detected stars ({})".format(len(img_stars)))
        return platepar

    # Create aligned platepar copy
    platepar_aligned = copy.deepcopy(platepar)

    # Use the NN-based pointing fit
    log.info("alignPlatepar: Input platepar RA={:.2f} Dec={:.2f} Ho={:.2f} JD={:.6f}".format(
        platepar.RA_d, platepar.dec_d, platepar.Ho, platepar.JD))
    log.info("alignPlatepar: Current JD={:.6f}, FOV center RA={:.2f} Dec={:.2f}".format(
        jd, ra_centre, dec_centre))
    log.info("alignPlatepar: Fitting pointing with {} detected stars, {} catalog stars".format(
        len(img_stars), len(catalog_stars_fov)))

    success, rmsd_pixels = platepar_aligned.fitPointingNN(
        jd, img_stars, catalog_stars_fov, fixed_scale=(not scale_update))

    if success:
        log.info("alignPlatepar: Fit successful, RMSD = {:.2f} px".format(rmsd_pixels))
        log.info("    RA:  {:.4f} -> {:.4f} deg (delta={:.4f})".format(
            platepar.RA_d, platepar_aligned.RA_d, platepar_aligned.RA_d - platepar.RA_d))
        log.info("    Dec: {:.4f} -> {:.4f} deg (delta={:.4f})".format(
            platepar.dec_d, platepar_aligned.dec_d, platepar_aligned.dec_d - platepar.dec_d))
        log.info("    Rot: {:.4f} -> {:.4f} deg (delta={:.4f})".format(
            platepar.pos_angle_ref, platepar_aligned.pos_angle_ref,
            platepar_aligned.pos_angle_ref - platepar.pos_angle_ref))
        log.info("    Ho unchanged: {:.4f} (platepar_aligned.Ho)".format(platepar_aligned.Ho))
        if scale_update:
            log.info("    Scale: {:.4f} -> {:.4f}".format(platepar.F_scale, platepar_aligned.F_scale))
    else:
        log.warning("alignPlatepar: Fit did not converge, returning original platepar")
        return platepar

    return platepar_aligned


if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(
        description="Align the platepar with the extracted stars from the CALSTARS file. "
                    "The FF file in CALSTARS with most detected stars will be used for alignment.")

    arg_parser.add_argument('dir_path', nargs=1, metavar='DIR_PATH', type=str,
                            help='Path to night folder.')

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str,
                            help="Path to a config file which will be used instead of the default one.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    dir_path = cml_args.dir_path[0]

    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, dir_path)

    # Initialize the logger
    log_manager = LoggingManager()
    log_manager.initLogging(config, 'align_')

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
        log.warning("CALSTARS list is empty - nothing to align")
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

    # Save the aligned platepar
    platepar_aligned.write(platepar_path)
    log.info("Aligned platepar saved to: {}".format(platepar_path))
