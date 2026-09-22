""" Align the platepar using nearest-neighbour optimization.

    This module provides platepar alignment by fitting the pointing parameters (RA, Dec, rotation)
    using a nearest-neighbour cost function. Direct optimization is used to minimize the distances
    between detected stars and their nearest catalog matches.
"""

# The MIT License

# Copyright (c) 2016 Denis Vida

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from __future__ import print_function, division, absolute_import

import os
import sys
import copy
import argparse

import numpy as np

from RMS.Astrometry import ApplyAstrometry
from RMS.Astrometry.Conversions import date2JD, jd2YearsFromJ2000
from RMS.Math import angularSeparationDeg
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


# Bounds of the catalog limiting magnitude inferred from the photometric calibration (mag). The floor
#   guards against a handful of bright detections producing a useless shallow catalog, the cap against
#   a mis-calibrated zero point pulling in an unmatchable deep catalog.
INFERRED_LM_MIN = 4.0
INFERRED_LM_MAX = 12.0

# Percentile of the estimated catalog magnitudes of the detected stars taken as the LM estimate, and the
#   margin added to it so the catalog covers all detections (mag)
INFERRED_LM_PERCENTILE = 95
INFERRED_LM_MARGIN = 1.0


def alignPlatepar(config, platepar, calstars_time, calstars_coords, scale_update=False, show_plot=False,
                  lm_callback=None):
    """ Align the platepar using nearest-neighbour optimization.

        This function fits the platepar pointing parameters (RA_d, dec_d, pos_angle_ref) by minimizing
        the sum of angular separations between detected stars and their nearest catalog neighbours.

    Arguments:
        config: [Config instance]
        platepar: [Platepar instance] Initial platepar.
        calstars_time: [list] A single entry of (year, month, day, hour, minute, second, millisecond)
            of the middle of the FF file used for alignment.
        calstars_coords: [ndarray] A 2D numpy array of star data. Can be:
            - (x, y) coordinates only (legacy format)
            - Full CALSTARS format: (y, x, intensity, amplitude, fwhm, bg, snr, saturated)
            If intensities are available, catalog LM is inferred from detected star magnitudes.

    Keyword arguments:
        scale_update: [bool] Update the platepar scale. False by default.
        show_plot: [bool] Unused, kept for backward compatibility.
        lm_callback: [callable] Optional callback called when catalog LM changes during balancing.
            Signature: callback(lim_mag, n_catalog, n_detected, ratio)
            Used for visual debugging of the LM balancing process.

    Return:
        (platepar_aligned, catalog_mag_limit): [tuple]
            platepar_aligned: [Platepar instance] The aligned platepar, or original if fit failed.
            catalog_mag_limit: [float] The catalog LM used (inferred from photometry if calibrated,
                               otherwise config value). Caller can use this for subsequent fitting.
    """

    # Create a copy of the config not to mess with the original config parameters
    config = copy.deepcopy(config)

    # Compute the Julian date (the tuple is not turned into a datetime directly, so a millisecond value that
    #   rounds up to a full second cannot overflow the datetime microsecond field)
    jd = date2JD(*calstars_time)

    # Compute the number of years from J2000 for the proper motion correction
    years_from_J2000 = jd2YearsFromJ2000(jd)

    # Extract coordinates and optionally infer catalog LM from intensities
    calstars_coords = np.array(calstars_coords)

    # Bail out if there are no stars or the input is not a 2D table of star entries
    if (calstars_coords.ndim != 2) or (len(calstars_coords) == 0):
        log.warning("alignPlatepar: No usable star data (shape {}), returning original platepar".format(
            calstars_coords.shape))
        return platepar, config.catalog_mag_limit

    # Full CALSTARS format: (y, x, intensity, ...)
    if calstars_coords.shape[1] >= 3:

        det_x = calstars_coords[:, 1]
        det_y = calstars_coords[:, 0]
        det_intens = calstars_coords[:, 2]

        # Infer catalog LM from detected star magnitudes if photometry is calibrated. A fresh Platepar has
        #   mag_lev = 1.0 and mag_lev_stddev = 0.0; the photometric fit sets both, so a non-default zero
        #   point together with a positive stddev is taken as "photometry has been fitted".
        photometry_calibrated = (platepar.mag_lev != 1.0) and (platepar.mag_lev_stddev > 0)

        if photometry_calibrated:

            # Estimate the catalog magnitude of every detection from its intensity and the zero point
            valid_intens = det_intens[det_intens > 0]
            if len(valid_intens) > 0:
                inst_mags = -2.5 * np.log10(valid_intens)
                est_cat_mags = inst_mags + platepar.mag_lev

                # Take a high percentile plus a margin so the catalog covers all detections, clamped to
                #   the plausible LM range
                inferred_lim_mag = np.percentile(est_cat_mags, INFERRED_LM_PERCENTILE) + INFERRED_LM_MARGIN
                inferred_lim_mag = min(max(inferred_lim_mag, INFERRED_LM_MIN), INFERRED_LM_MAX)

                log.info("alignPlatepar: Inferred LM={:.1f} from {} detected stars (mag_lev={:.1f})".format(
                    inferred_lim_mag, len(valid_intens), platepar.mag_lev))
                config.catalog_mag_limit = inferred_lim_mag
        else:
            log.info("alignPlatepar: Photometry not calibrated (mag_lev={:.1f}), using config LM={:.1f}".format(
                platepar.mag_lev, config.catalog_mag_limit))

        # Reformat to (x, y) for the rest of the function
        calstars_coords = np.column_stack([det_x, det_y])

    # Legacy (x, y) format - use the config LM
    else:
        log.info("alignPlatepar: Using config LM={:.1f} (no intensity data)".format(
            config.catalog_mag_limit))

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
    fov_radius_margin = fov_radius * 1.5
    filtered_indices, _ = subsetCatalog(catalog_stars, ra_centre, dec_centre, jd, platepar.lat,
                                        platepar.lon, fov_radius_margin, config.catalog_mag_limit)

    catalog_stars_fov = catalog_stars[filtered_indices]

    log.info("alignPlatepar: LM={:.1f}, {} catalog stars in FOV, {} detected stars".format(
        config.catalog_mag_limit, len(catalog_stars_fov), len(calstars_coords)))

    # Cap the catalog fed to NN matching to the brightest stars. A mis-inferred (over-deep) catalog
    #   LM on a wide FOV can pull in 100k+ mostly-unmatchable stars and bog down the fit. The
    #   detected stars are the brightest in the image, so the brightest catalog stars are the
    #   matchable ones; this is a bounded, cause-agnostic guard (independent of why the LM is deep).
    nn_catalog_cap = 8000
    if len(catalog_stars_fov) > nn_catalog_cap:
        log.info("alignPlatepar: Capping catalog from {} to brightest {} stars".format(
            len(catalog_stars_fov), nn_catalog_cap))
        brightest_idx = np.argsort(catalog_stars_fov[:, 2])[:nn_catalog_cap]
        catalog_stars_fov = catalog_stars_fov[brightest_idx]

    if len(catalog_stars_fov) < 5:
        log.warning("alignPlatepar: Not enough catalog stars in FOV ({})".format(len(catalog_stars_fov)))
        return platepar, config.catalog_mag_limit

    # Convert calstars_coords to the img_stars format (x, y, intensity) by adding a dummy intensity column
    img_stars = np.column_stack([calstars_coords, np.ones(len(calstars_coords))])

    if len(img_stars) < 5:
        log.warning("alignPlatepar: Not enough detected stars ({})".format(len(img_stars)))
        return platepar, config.catalog_mag_limit

    # Fit on a copy so the original platepar is returned untouched if the fit fails
    platepar_aligned = copy.deepcopy(platepar)

    # Use the NN-based pointing fit
    log.info("alignPlatepar: Fitting pointing with {} detected stars, {} catalog stars".format(
        len(img_stars), len(catalog_stars_fov)))

    success, rmsd, inlier_fraction, inlier_rmsd = platepar_aligned.fitPointingNN(
        jd, img_stars, catalog_stars_fov, fixed_scale=(not scale_update))

    if success:

        # Compute the pointing shift and rotation change for logging
        _, ra_centre_new, dec_centre_new, _ = ApplyAstrometry.xyToRaDecPP(
            [calstars_time], [platepar.X_res / 2], [platepar.Y_res / 2], [1],
            platepar_aligned, extinction_correction=False, precompute_pointing_corr=True)
        ra_centre_new = ra_centre_new[0]
        dec_centre_new = dec_centre_new[0]

        pointing_shift = angularSeparationDeg(ra_centre, dec_centre, ra_centre_new, dec_centre_new)

        # Wrap the rotation change to the [0, 180] deg range
        rot_change = abs(platepar_aligned.pos_angle_ref - platepar.pos_angle_ref) % 360
        if rot_change > 180:
            rot_change = 360 - rot_change

        log.info("alignPlatepar: Fit successful")
        log.info("    Apparent RA:  {:.4f} -> {:.4f} deg (delta={:.4f})".format(
            ra_centre, ra_centre_new, ra_centre_new - ra_centre))
        log.info("    Apparent Dec: {:.4f} -> {:.4f} deg (delta={:.4f})".format(
            dec_centre, dec_centre_new, dec_centre_new - dec_centre))
        log.info("    Rot: {:.4f} -> {:.4f} deg (delta={:.4f})".format(
            platepar.pos_angle_ref, platepar_aligned.pos_angle_ref,
            platepar_aligned.pos_angle_ref - platepar.pos_angle_ref))
        if scale_update:
            log.info("    Scale: {:.4f} -> {:.4f}".format(platepar.F_scale, platepar_aligned.F_scale))
        log.info("    Pointing shift: {:.2f} deg, Rotation shift: {:.2f} deg".format(
            pointing_shift, rot_change))
        log.info("    Inliers: {:.1f}%, Inlier RMSD: {:.2f} px".format(
            100*inlier_fraction, inlier_rmsd))

    # Keep the original platepar if the fit did not converge
    else:
        log.warning("alignPlatepar: Fit did not converge (inliers: {:.1f}%), returning original platepar".format(
            100*inlier_fraction))
        return platepar, config.catalog_mag_limit

    return platepar_aligned, config.catalog_mag_limit


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

    # Find the CALSTARS file in the given folder (do not reuse the loop variable, otherwise it is left
    #   pointing at the last listed file when nothing matches)
    calstars_file = None
    for file_name in file_list:
        if ('CALSTARS' in file_name) and ('.txt' in file_name):
            calstars_file = file_name
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

    # Extract the star list from the FF file with the most stars
    max_len_ff = max(calstars_dict, key=lambda k: len(calstars_dict[k]))

    # Pass the full CALSTARS data - alignPlatepar will extract the coordinates and use the intensities
    #   to infer the appropriate catalog limiting magnitude
    calstars_data = np.array(calstars_dict[max_len_ff])

    # Get the time of the FF file
    calstars_time = getMiddleTimeFF(max_len_ff, config.fps, ret_milliseconds=True, ff_frames=ff_frames)

    # Align the platepar with stars in CALSTARS
    platepar_aligned, _ = alignPlatepar(config, platepar, calstars_time, calstars_data, show_plot=False)

    # Save the aligned platepar
    platepar_aligned.write(platepar_path)
    log.info("Aligned platepar saved to: {}".format(platepar_path))
