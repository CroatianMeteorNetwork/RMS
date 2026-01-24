"""
Automatic platepar creation from CALSTARS data.

This module provides functionality to automatically create a platepar (plate solution)
from a directory containing CALSTARS data, without requiring GUI interaction.

The main entry point is autoFitPlatepar() which:
1. Loads CALSTARS data from the directory
2. Selects the best frame based on star distribution quality
3. Runs astrometry.net plate solving on that frame
4. Performs iterative NN-based refinement
5. Applies star filtering (photometric outliers, blended stars, high FWHM)
6. Does final fit with user-configurable settings
7. Returns a fitted Platepar object

The output should match exactly what would be obtained by clicking "Auto Fit"
in SkyFit2 on the selected image.
"""

from __future__ import print_function, division, absolute_import

import datetime
import os

import numpy as np

from RMS.Astrometry.Conversions import date2JD, jd2Date, JD2HourAngle, JD2LST
from RMS.Astrometry.Conversions import trueRaDec2ApparentAltAz, apparentAltAz2TrueRADec
from RMS.Astrometry.ApplyAstrometry import rotationWrtStandardToPosAngle
from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP
from RMS.Astrometry.AstrometryNet import astrometryNetSolve
from RMS.Astrometry.StarClasses import CatalogStar, PairedStars
from RMS.Astrometry.StarFilters import (filterPhotometricOutliers, filterBlendedStars, filterHighFWHMStars,
                                         DEFAULT_PHOTOMETRIC_SIGMA, DEFAULT_BLEND_FWHM_MULT,
                                         DEFAULT_BLEND_MAG_MARGIN, DEFAULT_HIGH_FWHM_FRACTION)
from RMS.Formats.Platepar import getCatalogStarsImagePositions
from RMS.Formats import CALSTARS, StarCatalog
from RMS.Formats.Platepar import Platepar
from RMS.Formats.FFfile import filenameToDatetime
from RMS.Routines.MaskImage import getMaskFile
from RMS.Math import angularSeparation, RMSD


# Default fitting parameters (matching SkyFit2's "Restore Defaults" button)
DEFAULT_DISTORTION_TYPE = "radial7-odd"
DEFAULT_EQUAL_ASPECT = True
DEFAULT_ASYMMETRY_CORR = True
DEFAULT_FORCE_DISTORTION_CENTRE = False
DEFAULT_REFRACTION = True


def scoreFrameDistribution(star_data, img_width, img_height, n_grid=4):
    """
    Score a frame's star distribution based on spatial coverage.

    Divides the image into a grid and scores based on how well stars are
    distributed across the grid cells. A good distribution has stars in
    most cells without excessive clustering.

    Arguments:
        star_data: [ndarray] Star data array with columns [y, x, IntensSum, Ampltd, FWHM, BgLvl, SNR, NSatPx]
        img_width: [int] Image width in pixels
        img_height: [int] Image height in pixels
        n_grid: [int] Number of grid divisions per axis (default 4 = 16 cells)

    Returns:
        score: [float] Distribution score (0-1, higher is better)
        details: [dict] Detailed breakdown of the score components
    """
    if len(star_data) == 0:
        return 0.0, {'n_stars': 0, 'cells_occupied': 0, 'total_cells': n_grid*n_grid}

    star_data = np.array(star_data)
    n_stars = len(star_data)

    # Extract coordinates (CALSTARS format: y in col 0, x in col 1)
    y_coords = star_data[:, 0]
    x_coords = star_data[:, 1]

    # Define grid
    cell_width = img_width / n_grid
    cell_height = img_height / n_grid
    total_cells = n_grid * n_grid

    # Count stars per cell
    cell_counts = np.zeros((n_grid, n_grid), dtype=int)
    for x, y in zip(x_coords, y_coords):
        cell_x = min(int(x / cell_width), n_grid - 1)
        cell_y = min(int(y / cell_height), n_grid - 1)
        cell_counts[cell_y, cell_x] += 1

    # Metrics
    cells_occupied = np.sum(cell_counts > 0)
    coverage_fraction = cells_occupied / total_cells

    # Penalize very uneven distribution (high variance in cell counts)
    occupied_counts = cell_counts[cell_counts > 0]
    if len(occupied_counts) > 1:
        cv = np.std(occupied_counts) / np.mean(occupied_counts)
        uniformity_score = 1.0 / (1.0 + cv)
    else:
        uniformity_score = 0.5

    score = 0.7 * coverage_fraction + 0.3 * uniformity_score

    details = {
        'n_stars': n_stars,
        'cells_occupied': cells_occupied,
        'total_cells': total_cells,
        'coverage_fraction': coverage_fraction,
        'uniformity_score': uniformity_score,
        'cell_counts': cell_counts
    }

    return score, details


def scoreFrameQuality(star_data, min_stars=10, max_stars=200):
    """
    Score a frame's star quality based on SNR, saturation, and count.

    Arguments:
        star_data: [ndarray] Star data array with columns [y, x, IntensSum, Ampltd, FWHM, BgLvl, SNR, NSatPx]
        min_stars: [int] Minimum number of stars for a valid frame
        max_stars: [int] Maximum number of stars (more may indicate noise/clouds)

    Returns:
        score: [float] Quality score (0-1, higher is better)
        details: [dict] Detailed breakdown of the score components
    """
    if len(star_data) == 0:
        return 0.0, {'n_stars': 0, 'valid': False}

    star_data = np.array(star_data)
    n_stars = len(star_data)

    if n_stars < min_stars:
        return 0.0, {'n_stars': n_stars, 'valid': False, 'reason': 'too_few_stars'}

    snr = star_data[:, 6] if star_data.shape[1] > 6 else np.ones(n_stars)
    n_saturated_px = star_data[:, 7] if star_data.shape[1] > 7 else np.zeros(n_stars)

    non_saturated_count = np.sum(n_saturated_px == 0)
    saturation_fraction = non_saturated_count / n_stars

    non_sat_mask = n_saturated_px == 0
    if np.sum(non_sat_mask) > 0:
        mean_snr = np.mean(snr[non_sat_mask])
    else:
        mean_snr = np.mean(snr)

    snr_score = min(mean_snr / 10.0, 1.0)

    if n_stars <= max_stars:
        count_score = (n_stars - min_stars) / (max_stars - min_stars)
        count_score = max(0, min(1, count_score))
    else:
        count_score = max(0, 1.0 - (n_stars - max_stars) / max_stars)

    score = 0.4 * count_score + 0.4 * saturation_fraction + 0.2 * snr_score

    details = {
        'n_stars': n_stars,
        'valid': True,
        'non_saturated_count': non_saturated_count,
        'saturation_fraction': saturation_fraction,
        'mean_snr': mean_snr,
        'snr_score': snr_score,
        'count_score': count_score
    }

    return score, details


def selectBestFrame(calstars, img_width, img_height, min_stars=10, max_stars=200, verbose=False):
    """
    Select the best frame from CALSTARS data based on star distribution and quality.

    Arguments:
        calstars: [dict] Dictionary mapping FF filenames to star data arrays
        img_width: [int] Image width in pixels
        img_height: [int] Image height in pixels
        min_stars: [int] Minimum number of stars for a valid frame
        max_stars: [int] Maximum stars before penalizing
        verbose: [bool] Print detailed scoring info

    Returns:
        best_ff: [str] Filename of the best frame (or None if no valid frames)
        best_score: [float] Score of the best frame
        all_scores: [dict] Dictionary mapping FF filenames to score details
    """
    all_scores = {}

    for ff_name, star_data in calstars.items():
        star_data = np.array(star_data)

        dist_score, dist_details = scoreFrameDistribution(star_data, img_width, img_height)
        qual_score, qual_details = scoreFrameQuality(star_data, min_stars, max_stars)

        if qual_details.get('valid', False):
            combined_score = 0.5 * dist_score + 0.5 * qual_score
        else:
            combined_score = 0.0

        all_scores[ff_name] = {
            'combined_score': combined_score,
            'distribution_score': dist_score,
            'quality_score': qual_score,
            'distribution_details': dist_details,
            'quality_details': qual_details
        }

        if verbose:
            print("  {:s}: {:.3f} (dist={:.3f}, qual={:.3f}, n_stars={:d})".format(
                ff_name, combined_score, dist_score, qual_score, len(star_data)))

    best_ff = None
    best_score = 0.0

    for ff_name, scores in all_scores.items():
        if scores['combined_score'] > best_score:
            best_score = scores['combined_score']
            best_ff = ff_name

    return best_ff, best_score, all_scores


def printFitResiduals(paired_stars, platepar, jd, ff_dt):
    """
    Print fit residuals matching SkyFit2's output format.

    Arguments:
        paired_stars: [PairedStars] Paired stars object
        platepar: [Platepar] Fitted platepar
        jd: [float] Julian date
        ff_dt: [datetime] Datetime of the image

    Returns:
        rmsd_img: [float] RMSD in pixels
        rmsd_angular: [float] RMSD in arcmin/arcsec/deg
        angular_error_label: [str] Unit label for angular RMSD
    """
    # Print platepar
    print()
    print(repr(platepar))

    # Print time info
    print()
    print("Image time = {:s} UTC".format(ff_dt.strftime("%Y-%m-%d %H:%M:%S.%f")))
    print("Image JD = {:.8f}".format(jd))
    print("Image LST = {:.8f}".format(JD2LST(jd, platepar.lon)[0]))

    # Get catalog positions for matched stars
    sky_coords = np.array(paired_stars.skyCoords())
    catalog_x, catalog_y, catalog_mag = getCatalogStarsImagePositions(sky_coords, jd, platepar)

    # Print header
    print()
    print('Residuals')
    print('----------')
    print(' No,       Img X,       Img Y, RA cat (deg), Dec cat (deg),    Cat X,    Cat Y, RA img (deg), Dec img (deg), Err amin,  Err px, Direction,  FWHM,    Mag, -2.5*LSP')

    residuals = []
    all_coords = paired_stars.allCoords()

    for star_no, (cat_x, cat_y, cat_coords, paired_star_data) in enumerate(
            zip(catalog_x, catalog_y, sky_coords, all_coords)):

        img_x, img_y, fwhm, sum_intens, snr, saturated = paired_star_data[0]
        ra, dec, mag = cat_coords

        delta_x = cat_x - img_x
        delta_y = cat_y - img_y

        # Compute image residual and angle
        angle = np.arctan2(delta_y, delta_x)
        distance = np.sqrt(delta_x**2 + delta_y**2)

        # Compute RA/Dec from image position (using JD time format)
        _, ra_img, dec_img, _ = xyToRaDecPP(
            [jd], [img_x], [img_y], [1], platepar, extinction_correction=False, jd_time=True
        )
        ra_img = ra_img[0]
        dec_img = dec_img[0]

        # Compute angular distance
        angular_distance = np.degrees(angularSeparation(
            np.radians(ra), np.radians(dec),
            np.radians(ra_img), np.radians(dec_img)
        ))

        residuals.append([img_x, img_y, angle, distance, angular_distance])

        lsp = -2.5*np.log10(sum_intens) if sum_intens and sum_intens > 0 else 0
        fwhm_val = fwhm if fwhm is not None else 0.0
        mag_val = mag if mag is not None else 0.0

        # Print residual line
        print('{:3d}, {:11.6f}, {:11.6f}, {:>12.6f}, {:>+13.6f}, {:8.2f}, {:8.2f}, {:>12.6f}, {:>+13.6f}, {:8.2f}, {:7.2f}, {:+9.1f}, {:5.2f}, {:+6.2f}, {:8.2f}'.format(
            star_no + 1, img_x, img_y, ra, dec, cat_x, cat_y,
            ra_img, dec_img, 60*angular_distance, distance, np.degrees(angle),
            fwhm_val, mag_val, lsp
        ))

    # Compute RMSD errors
    rmsd_angular = 60*RMSD([entry[4] for entry in residuals])
    rmsd_img = RMSD([entry[3] for entry in residuals])

    # Determine appropriate angular unit
    if rmsd_angular > 60:
        rmsd_angular /= 60
        angular_error_label = 'deg'
    elif rmsd_angular > 0.5:
        angular_error_label = 'arcmin'
    else:
        rmsd_angular *= 60
        angular_error_label = 'arcsec'

    print()
    print('RMSD: {:.2f} px, {:.2f} {:s}'.format(rmsd_img, rmsd_angular, angular_error_label))

    return rmsd_img, rmsd_angular, angular_error_label


def autoFitPlatepar(dir_path, config, catalog_stars, platepar_template=None,
                    fov_w_hint=None, ff_name=None, distortion_type=DEFAULT_DISTORTION_TYPE,
                    equal_aspect=DEFAULT_EQUAL_ASPECT, asymmetry_corr=DEFAULT_ASYMMETRY_CORR,
                    force_distortion_centre=DEFAULT_FORCE_DISTORTION_CENTRE,
                    refraction=DEFAULT_REFRACTION,
                    photometric_sigma=DEFAULT_PHOTOMETRIC_SIGMA,
                    fwhm_mult=DEFAULT_BLEND_FWHM_MULT,
                    high_fwhm_fraction=DEFAULT_HIGH_FWHM_FRACTION,
                    verbose=True):
    """
    Automatically create a platepar from CALSTARS data in a directory.

    This function replicates the behavior of SkyFit2's "Auto Fit" button:
    1. Loads CALSTARS data from the directory
    2. Selects the best frame based on star distribution (or uses specified frame)
    3. Runs astrometry.net plate solving
    4. Performs NN-based refinement with intermediate settings
    5. Applies star filtering (photometric outliers, blended stars, high FWHM)
    6. Does final fit with user-specified settings
    7. Returns a fitted Platepar

    Arguments:
        dir_path: [str] Path to directory containing CALSTARS file and FF files
        config: [Config] RMS configuration object
        catalog_stars: [ndarray] Star catalog array (RA, Dec, mag, ...)
        platepar_template: [Platepar] Optional template platepar with station location.
                          If None, uses config values.
        fov_w_hint: [float] Optional FOV width hint in degrees. If None, uses config.fov_w.
        ff_name: [str] Optional specific FF filename to use. If None, selects best frame
                 from CALSTARS automatically.

    Keyword arguments (final fit settings - match SkyFit2 defaults):
        distortion_type: [str] Distortion model for final fit (default: "radial5-odd")
        equal_aspect: [bool] Equal aspect ratio constraint (default: True)
        asymmetry_corr: [bool] Asymmetry correction (default: False)
        force_distortion_centre: [bool] Force distortion centre to image centre (default: False)
        refraction: [bool] Apply refraction correction (default: True)

    Keyword arguments (filtering parameters):
        photometric_sigma: [float] Sigma threshold for photometric outlier removal (default: 2.5)
        fwhm_mult: [float] Multiplier of FWHM for blend detection radius (default: 2.0)
        high_fwhm_fraction: [float] Fraction of high-FWHM stars to remove (default: 0.10)

        verbose: [bool] Print progress information

    Returns:
        platepar: [Platepar] Fitted platepar object, or None if fitting failed
        matched_stars: [list] List of matched star pairs
        best_ff: [str] Filename of the frame used for fitting
    """
    if verbose:
        print("=" * 70)
        print("Auto Platepar Fitting")
        print("=" * 70)
        print("Directory: {:s}".format(dir_path))

    # Find and load CALSTARS file
    calstars_file = None
    for f in os.listdir(dir_path):
        if 'CALSTARS' in f and f.endswith('.txt'):
            calstars_file = f
            break

    if calstars_file is None:
        if verbose:
            print("ERROR: No CALSTARS file found in directory")
        return None, None, None

    calstars_list, chunk_frames = CALSTARS.readCALSTARS(dir_path, calstars_file)
    calstars = {ff_file: star_data for ff_file, star_data in calstars_list}

    if verbose:
        print("Loaded CALSTARS: {:s} ({:d} frames)".format(calstars_file, len(calstars)))

    img_width = config.width
    img_height = config.height

    # Use specified frame or find the best one
    if ff_name is not None:
        # Use the specified frame
        if ff_name not in calstars:
            if verbose:
                print("ERROR: Specified frame '{:s}' not found in CALSTARS".format(ff_name))
            return None, None, None

        best_ff = ff_name
        best_score = None
        all_scores = {}

        if verbose:
            print()
            print("Using specified frame: {:s}".format(best_ff))
            print("  Stars: {:d}".format(len(calstars[best_ff])))
    else:
        # Find the best frame automatically
        if verbose:
            print()
            print("Scoring frames for star distribution...")

        best_ff, best_score, all_scores = selectBestFrame(
            calstars, img_width, img_height,
            min_stars=10, max_stars=200,
            verbose=verbose
        )

        if best_ff is None:
            if verbose:
                print("ERROR: No valid frames found in CALSTARS")
            return None, None, None

        if verbose:
            print()
            print("Best frame: {:s} (score={:.3f})".format(best_ff, best_score))
            details = all_scores[best_ff]
            print("  Stars: {:d}, Coverage: {:.1f}%, Non-saturated: {:.1f}%".format(
                details['quality_details']['n_stars'],
                details['distribution_details']['coverage_fraction'] * 100,
                details['quality_details']['saturation_fraction'] * 100
            ))

    # Get star data for best frame
    star_data = np.array(calstars[best_ff])

    # Create or copy platepar
    if platepar_template is not None:
        platepar = Platepar()
        platepar.lat = platepar_template.lat
        platepar.lon = platepar_template.lon
        platepar.elev = platepar_template.elev
        platepar.X_res = platepar_template.X_res
        platepar.Y_res = platepar_template.Y_res
        platepar.station_code = platepar_template.station_code
    else:
        platepar = Platepar()
        platepar.lat = config.latitude
        platepar.lon = config.longitude
        platepar.elev = config.elevation
        platepar.X_res = config.width
        platepar.Y_res = config.height
        platepar.station_code = config.stationID

    # Initialize vignetting coefficient with resolution-scaled default
    # We don't apply flat during platepar creation, so photometry needs vignetting correction
    platepar.addVignettingCoeff(use_flat=False)

    # Get time from FF filename
    ff_dt = filenameToDatetime(best_ff)
    jd = date2JD(ff_dt.year, ff_dt.month, ff_dt.day,
                 ff_dt.hour, ff_dt.minute, ff_dt.second, ff_dt.microsecond/1000.0)

    platepar.JD = jd
    platepar.Ho = JD2HourAngle(jd) % 360

    # FOV hint
    if fov_w_hint is None:
        fov_w_hint = config.fov_w
    fov_w_range = [0.75 * fov_w_hint, 1.5 * fov_w_hint]

    # Load mask if available
    mask = getMaskFile(dir_path, config)

    # Extract star coordinates (CALSTARS format: y, x, ...)
    y_data = star_data[:, 0]
    x_data = star_data[:, 1]
    input_intensities = star_data[:, 2] if star_data.shape[1] > 2 else None
    input_fwhm = star_data[:, 4] if star_data.shape[1] > 4 else np.zeros(len(x_data))
    input_snr = star_data[:, 6] if star_data.shape[1] > 6 else np.ones(len(x_data))
    input_saturated = star_data[:, 7] if star_data.shape[1] > 7 else np.zeros(len(x_data))

    if verbose:
        print()
        print("Running astrometry.net plate solving...")
        print("  Stars: {:d}".format(len(x_data)))
        print("  FOV range: {:.1f} - {:.1f} deg".format(fov_w_range[0], fov_w_range[1]))

    # Call astrometry.net
    solution = astrometryNetSolve(
        x_data=x_data, y_data=y_data,
        fov_w_range=fov_w_range,
        fov_w_hint=fov_w_hint,
        mask=mask,
        x_center=platepar.X_res / 2,
        y_center=platepar.Y_res / 2,
        lat=platepar.lat,
        lon=platepar.lon,
        jd=jd,
        input_intensities=input_intensities,
        verbose=verbose
    )

    if solution is None:
        if verbose:
            print("ERROR: Astrometry.net failed to find a solution")
        return None, None, best_ff

    # Extract solution
    ra, dec, rot_standard, scale, fov_w, fov_h, matched_star_data, solution_info = solution

    if verbose:
        print()
        print("Astrometry.net solution:")
        print("  RA = {:.2f} deg".format(ra))
        print("  Dec = {:.2f} deg".format(dec))
        print("  Scale = {:.3f} arcmin/px".format(60 / scale))
        print("  FOV = {:.2f} x {:.2f} deg".format(fov_w, fov_h))

    # Apply solution to platepar
    platepar.F_scale = scale

    # Compute azimuth and altitude from RA/Dec
    azim, alt = trueRaDec2ApparentAltAz(ra, dec, jd, platepar.lat, platepar.lon)
    platepar.az_centre = azim
    platepar.alt_centre = alt

    platepar.updateRefRADec(skip_rot_update=True)
    platepar.pos_angle_ref = rotationWrtStandardToPosAngle(platepar, rot_standard)

    # Set INTERMEDIATE fitting parameters (matching SkyFit2)
    # These are used for NN refinement, NOT the final fit
    platepar.refraction = True
    platepar.equal_aspect = True
    platepar.asymmetry_corr = False
    platepar.force_distortion_centre = False
    platepar.setDistortionType("radial5-odd", reset_params=True)

    if verbose:
        print()
        print("Performing NN-based refinement...")

    # Prepare detected stars array
    img_stars_arr = np.column_stack([x_data, y_data,
                                     input_intensities if input_intensities is not None else np.ones(len(x_data))])

    try:
        # Fit using NN cost function (intermediate fit)
        result = platepar.fitAstrometry(
            jd, img_stars_arr, catalog_stars,
            first_platepar_fit=True,
            use_nn_cost=True
        )

        if verbose:
            print("  NN fit complete")
            print("  RA = {:.2f} deg, Dec = {:.2f} deg, Scale = {:.3f} arcmin/px".format(
                platepar.RA_d, platepar.dec_d, 60 / platepar.F_scale))

    except Exception as e:
        if verbose:
            print("ERROR: NN fitting failed: {:s}".format(str(e)))
        return None, None, best_ff

    # Build paired_stars from NN fit results
    paired_stars = PairedStars()

    if hasattr(platepar, 'star_list') and platepar.star_list:
        for entry in platepar.star_list:
            # star_list format: [jd, x, y, intensity, ra, dec, mag]
            _, img_x, img_y, intensity, cat_ra, cat_dec, cat_mag = entry
            sky_obj = CatalogStar(cat_ra, cat_dec, cat_mag)

            # Find closest detected star to get FWHM, SNR, saturation
            fwhm, snr, saturated = 2.5, 1.0, False
            if len(x_data) > 0:
                distances = np.sqrt((x_data - img_x)**2 + (y_data - img_y)**2)
                closest_idx = np.argmin(distances)
                if distances[closest_idx] < 3.0:
                    fwhm = input_fwhm[closest_idx]
                    snr = input_snr[closest_idx]
                    saturated = input_saturated[closest_idx] > 0

            paired_stars.addPair(img_x, img_y, fwhm, intensity, sky_obj, snr=snr, saturated=saturated)

        if verbose:
            print("  Matched pairs: {:d}".format(len(paired_stars)))

    # Check if we have enough stars for final fit
    if len(paired_stars) < 10:
        if verbose:
            print("ERROR: Not enough matched stars for final fit ({:d} < 10)".format(len(paired_stars)))
        return platepar, [], best_ff

    # Apply star filtering (matching SkyFit2)
    if verbose:
        print()
        print("Filtering stars...")

    if len(paired_stars) >= 15:
        paired_stars, removed = filterPhotometricOutliers(
            paired_stars, platepar, jd, sigma_threshold=photometric_sigma, verbose=verbose)

    if len(paired_stars) >= 15:
        paired_stars, removed = filterBlendedStars(
            paired_stars, catalog_stars, platepar, jd, config.catalog_mag_limit,
            fwhm_mult=fwhm_mult, verbose=verbose)

    if len(paired_stars) >= 15:
        paired_stars, removed = filterHighFWHMStars(
            paired_stars, fraction=high_fwhm_fraction, verbose=verbose)

    if verbose:
        print("  Stars after filtering: {:d}".format(len(paired_stars)))

    # Check again after filtering
    if len(paired_stars) < 10:
        if verbose:
            print("ERROR: Not enough stars after filtering ({:d} < 10)".format(len(paired_stars)))
        return platepar, [], best_ff

    # Apply USER's settings for final fit
    if verbose:
        print()
        print("Final fit with user settings...")
        print("  distortion_type: {:s}".format(distortion_type))
        print("  equal_aspect: {:s}".format(str(equal_aspect)))
        print("  asymmetry_corr: {:s}".format(str(asymmetry_corr)))
        print("  force_distortion_centre: {:s}".format(str(force_distortion_centre)))
        print("  refraction: {:s}".format(str(refraction)))

    platepar.equal_aspect = equal_aspect
    platepar.asymmetry_corr = asymmetry_corr
    platepar.force_distortion_centre = force_distortion_centre
    platepar.refraction = refraction
    platepar.setDistortionType(distortion_type, reset_params=False)

    # Extract coordinates for final fit
    img_coords = np.array(paired_stars.imageCoords())
    sky_coords = np.array(paired_stars.skyCoords())

    # Do the final fit
    try:
        platepar.fitAstrometry(jd, img_coords, sky_coords, first_platepar_fit=True)
    except Exception as e:
        if verbose:
            print("ERROR: Final fit failed: {:s}".format(str(e)))
        # Return the intermediate fit result
        return platepar, [], best_ff

    if verbose:
        # Print full residuals report (matching SkyFit2 output)
        try:
            printFitResiduals(paired_stars, platepar, jd, ff_dt)
        except Exception as e:
            print("WARNING: Could not print residuals: {:s}".format(str(e)))
            print()
            print("Final platepar:")
            print("  RA = {:.4f} deg".format(platepar.RA_d))
            print("  Dec = {:.4f} deg".format(platepar.dec_d))
            print("  Scale = {:.4f} arcmin/px".format(60 / platepar.F_scale))
            print("  Matched stars: {:d}".format(len(paired_stars)))

    return platepar, paired_stars, best_ff


def loadCatalogStars(config, lim_mag, jd=None):
    """
    Load star catalog for plate solving.

    Arguments:
        config: [Config] RMS configuration object
        lim_mag: [float] Limiting magnitude
        jd: [float] Julian date for proper motion correction (optional)

    Returns:
        catalog_stars: [ndarray] Star catalog array
    """
    star_catalog_path = config.star_catalog_path
    if not os.path.isdir(star_catalog_path):
        star_catalog_path = os.path.join(config.rms_root_dir, 'Catalogs')

    if jd is not None:
        dt = jd2Date(jd, dt_obj=True)
        years_from_J2000 = (dt - datetime.datetime(2000, 1, 1, 12, 0, 0)).days / 365.25
    else:
        years_from_J2000 = 0.0

    catalog_results = StarCatalog.readStarCatalog(
        star_catalog_path, config.star_catalog_file,
        lim_mag=lim_mag, mag_band_ratios=config.star_catalog_band_ratios,
        years_from_J2000=years_from_J2000
    )

    if catalog_results is None:
        return None

    catalog_stars, _, _ = catalog_results
    return catalog_stars


if __name__ == "__main__":
    import argparse
    import RMS.ConfigReader as cr

    parser = argparse.ArgumentParser(description="Auto-fit platepar from CALSTARS data")
    parser.add_argument("dir_path", help="Path to directory with CALSTARS file")
    parser.add_argument("-c", "--config", help="Path to config file", default=None)

    # Fitting parameters (matching SkyFit2 defaults)
    parser.add_argument("--distortion", default=DEFAULT_DISTORTION_TYPE,
                        help="Distortion type (default: {:s})".format(DEFAULT_DISTORTION_TYPE))
    parser.add_argument("--no-equal-aspect", action="store_true",
                        help="Disable equal aspect ratio constraint (default: enabled)")
    parser.add_argument("--no-asymmetry-corr", action="store_true",
                        help="Disable asymmetry correction (default: enabled)")
    parser.add_argument("--force-distortion-centre", action="store_true",
                        help="Force distortion centre to image centre (default: disabled)")
    parser.add_argument("--no-refraction", action="store_true",
                        help="Disable refraction correction (default: enabled)")

    # Filtering parameters
    parser.add_argument("--photom-sigma", type=float, default=DEFAULT_PHOTOMETRIC_SIGMA,
                        help="Photometric outlier sigma threshold (default: {:.1f})".format(DEFAULT_PHOTOMETRIC_SIGMA))
    parser.add_argument("--fwhm-mult", type=float, default=DEFAULT_BLEND_FWHM_MULT,
                        help="FWHM multiplier for blend detection radius (default: {:.1f})".format(DEFAULT_BLEND_FWHM_MULT))
    parser.add_argument("--fwhm-fraction", type=float, default=DEFAULT_HIGH_FWHM_FRACTION,
                        help="Fraction of high-FWHM stars to remove (default: {:.2f})".format(DEFAULT_HIGH_FWHM_FRACTION))

    parser.add_argument("-o", "--output", help="Output platepar filename", default="platepar_auto.cal")

    args = parser.parse_args()

    # Load config
    config = cr.loadConfigFromDirectory(args.config if args.config else '.', args.dir_path)

    # Load star catalog
    catalog_stars = loadCatalogStars(config, config.catalog_mag_limit)

    if catalog_stars is None:
        print("ERROR: Could not load star catalog")
        exit(1)

    print("Loaded star catalog: {:d} stars".format(len(catalog_stars)))

    # Run auto-fit
    platepar, matched_stars, best_ff = autoFitPlatepar(
        args.dir_path, config, catalog_stars,
        distortion_type=args.distortion,
        equal_aspect=not args.no_equal_aspect,
        asymmetry_corr=not args.no_asymmetry_corr,
        force_distortion_centre=args.force_distortion_centre,
        refraction=not args.no_refraction,
        photometric_sigma=args.photom_sigma,
        fwhm_mult=args.fwhm_mult,
        high_fwhm_fraction=args.fwhm_fraction,
        verbose=True
    )

    if platepar is not None:
        print()
        print("=" * 70)
        print("SUCCESS: Platepar created")
        print("=" * 70)

        output_path = os.path.join(args.dir_path, args.output)
        platepar.write(output_path)
        print("Saved to: {:s}".format(output_path))
    else:
        print()
        print("FAILED: Could not create platepar")
        exit(1)
