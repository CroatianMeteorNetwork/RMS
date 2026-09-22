""" Automatic platepar creation from CALSTARS data.

    This module provides functionality to automatically create a platepar (plate solution) from a
    directory containing CALSTARS data, without requiring GUI interaction.

    The main entry point is autoFitPlatepar() which:
        1. Loads the CALSTARS data from the directory
        2. Selects the best frame based on the star distribution quality
        3. Runs astrometry.net plate solving on that frame
        4. Performs an iterative NN-based refinement
        5. Applies star filtering (photometric outliers, blended stars)
        6. Does the final fit with user-configurable settings
        7. Returns a fitted Platepar object

    The output should match exactly what would be obtained by clicking "Auto Fit" in SkyFit2 on the
    selected image.
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

import numpy as np
from scipy.spatial import cKDTree

from RMS.Astrometry.Conversions import date2JD, JD2HourAngle, JD2LST, jd2YearsFromJ2000
from RMS.Astrometry.Conversions import trueRaDec2ApparentAltAz
from RMS.Astrometry.ApplyAstrometry import rotationWrtStandardToPosAngle
from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP
from RMS.Astrometry.AstrometryNet import astrometryNetSolve
from RMS.Astrometry.StarClasses import CatalogStar, PairedStars
from RMS.Astrometry.StarFilters import (filterPhotometricOutliers, filterBlendedStars,
                                         DEFAULT_PHOTOMETRIC_SIGMA, DEFAULT_BLEND_FWHM_MULT)
from RMS.Formats.Platepar import getCatalogStarsImagePositions
from RMS.Formats import CALSTARS, StarCatalog
from RMS.Formats.Platepar import Platepar
from RMS.Formats.FFfile import getMiddleTimeFF, validFFName
from RMS.ExtractStars import extractStarsAndSave
from RMS.Routines.MaskImage import getMaskFile
from RMS.Math import angularSeparation, RMSD


# Default fitting parameters (matching SkyFit2's "Restore Defaults" button)
DEFAULT_DISTORTION_TYPE = "radial7-odd"
DEFAULT_EQUAL_ASPECT = True
DEFAULT_ASYMMETRY_CORR = True
DEFAULT_FORCE_DISTORTION_CENTRE = False
DEFAULT_REFRACTION = True

# Maximum pixel distance between an NN-fitted star position and a detected star for the detection's
#   FWHM/SNR/saturation to be attached to the pair (the fitted star_list only carries x, y, intensity)
NN_PAIR_ASSOC_RADIUS_PX = 3.0



### Detection association ###

def associateDetections(star_x, star_y, x_data, y_data, radius=NN_PAIR_ASSOC_RADIUS_PX):
    """ Find the closest detection to every fitted star, if it is within the association radius.

        Equivalent to taking, for every star, np.argmin of the distances to all detections and accepting it
        if the distance is below the radius, but done with one KD-tree query in O((N + M) log M) instead of
        an O(N*M) Python loop.

    Arguments:
        star_x: [ndarray] X image coordinates of the fitted stars.
        star_y: [ndarray] Y image coordinates of the fitted stars.
        x_data: [ndarray] X image coordinates of the detections.
        y_data: [ndarray] Y image coordinates of the detections.

    Keyword arguments:
        radius: [float] Maximum association distance (px). NN_PAIR_ASSOC_RADIUS_PX by default.

    Return:
        closest_idx: [ndarray of int] Index of the associated detection for every star, -1 if none is
            within the radius.
    """

    star_x = np.atleast_1d(np.asarray(star_x, dtype=np.float64))
    star_y = np.atleast_1d(np.asarray(star_y, dtype=np.float64))
    x_data = np.asarray(x_data, dtype=np.float64)
    y_data = np.asarray(y_data, dtype=np.float64)

    closest_idx = -np.ones(len(star_x), dtype=int)

    # Without detections nothing can be associated. The same holds when any coordinate is non-finite: the
    #   per-star argmin over distances containing a NaN selected that NaN, which never passed the radius
    #   test, so no star was associated - keep that so the result does not change
    if (len(x_data) == 0) or (not np.all(np.isfinite(x_data))) or (not np.all(np.isfinite(y_data))) \
        or (not np.all(np.isfinite(star_x))) or (not np.all(np.isfinite(star_y))):

        return closest_idx

    # One nearest-neighbour query for all stars; the upper bound only prunes the search, the strict
    #   radius test below is the same as the original comparison
    tree = cKDTree(np.column_stack([x_data, y_data]))
    distances, indices = tree.query(np.column_stack([star_x, star_y]), k=1, distance_upper_bound=radius)
    associated = distances < radius
    closest_idx[associated] = indices[associated]

    return closest_idx



### Frame scoring ###

def scoreFrameDistribution(star_data, img_width, img_height, n_grid=4):
    """ Score a frame's star distribution based on spatial coverage.

        Divides the image into a grid and scores based on how well the stars are distributed across the
        grid cells. A good distribution has stars in most cells without excessive clustering.

    Arguments:
        star_data: [ndarray] Star data array with columns
            [y, x, IntensSum, Ampltd, FWHM, BgLvl, SNR, NSatPx].
        img_width: [int] Image width in pixels.
        img_height: [int] Image height in pixels.

    Keyword arguments:
        n_grid: [int] Number of grid divisions per axis. 4 by default, i.e. 16 cells.

    Return:
        score: [float] Distribution score (0-1, higher is better).
        details: [dict] Detailed breakdown of the score components.
    """

    # An empty frame scores zero
    if len(star_data) == 0:
        return 0.0, {'n_stars': 0, 'cells_occupied': 0, 'total_cells': n_grid*n_grid}

    star_data = np.array(star_data)
    n_stars = len(star_data)

    # Extract coordinates (CALSTARS format: y in col 0, x in col 1)
    y_coords = star_data[:, 0]
    x_coords = star_data[:, 1]

    # Define the grid
    cell_width = img_width / n_grid
    cell_height = img_height / n_grid
    total_cells = n_grid * n_grid

    # Count the stars per cell (stars on the far edge are clipped into the last cell)
    cell_x = np.clip((x_coords / cell_width).astype(int), 0, n_grid - 1)
    cell_y = np.clip((y_coords / cell_height).astype(int), 0, n_grid - 1)
    cell_counts = np.zeros((n_grid, n_grid), dtype=int)
    np.add.at(cell_counts, (cell_y, cell_x), 1)

    # Fraction of the image covered by stars
    cells_occupied = np.sum(cell_counts > 0)
    coverage_fraction = cells_occupied / total_cells

    # Penalize a very uneven distribution (high variance in cell counts)
    occupied_counts = cell_counts[cell_counts > 0]
    if len(occupied_counts) > 1:
        cv = np.std(occupied_counts) / np.mean(occupied_counts)
        uniformity_score = 1.0 / (1.0 + cv)
    else:
        uniformity_score = 0.5

    # Coverage matters more than uniformity for constraining the distortion
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



def scoreFrameQuality(star_data, min_stars=10, max_stars=200, penalty_stars=None):
    """ Score a frame's star quality based on SNR, saturation, and count.

    Arguments:
        star_data: [ndarray] Star data array with columns
            [y, x, IntensSum, Ampltd, FWHM, BgLvl, SNR, NSatPx].

    Keyword arguments:
        min_stars: [int] Minimum number of stars for a valid frame. 10 by default.
        max_stars: [int] Number of stars at which the count score saturates. 200 by default.
        penalty_stars: [int] Number of stars above which the frame is penalized (more may indicate
            noise/clouds). None by default, in which case max_stars is used.

    Return:
        score: [float] Quality score (0-1, higher is better).
        details: [dict] Detailed breakdown of the score components.
    """

    # An empty frame is not usable
    if len(star_data) == 0:
        return 0.0, {'n_stars': 0, 'valid': False}

    star_data = np.array(star_data)
    n_stars = len(star_data)

    # Too few stars cannot constrain a fit
    if n_stars < min_stars:
        return 0.0, {'n_stars': n_stars, 'valid': False, 'reason': 'too_few_stars'}

    # Older CALSTARS files lack the SNR and saturation columns, so assume neutral values for them
    snr = star_data[:, 6] if star_data.shape[1] > 6 else np.ones(n_stars)
    n_saturated_px = star_data[:, 7] if star_data.shape[1] > 7 else np.zeros(n_stars)

    # Fraction of stars with no saturated pixels (higher is better for the score)
    non_saturated_count = np.sum(n_saturated_px == 0)
    non_saturated_fraction = non_saturated_count / n_stars

    # Mean SNR of the unsaturated stars, falling back to all stars if every one is saturated
    non_sat_mask = n_saturated_px == 0
    if np.sum(non_sat_mask) > 0:
        mean_snr = np.mean(snr[non_sat_mask])
    else:
        mean_snr = np.mean(snr)

    # An SNR of 10 or more is considered good enough
    snr_score = min(mean_snr / 10.0, 1.0)

    # Reward more stars up to max_stars, then penalize frames with suspiciously many detections (more
    #   than penalty_stars)
    if penalty_stars is None:
        penalty_stars = max_stars
    penalty_stars = max(penalty_stars, max_stars)

    if n_stars <= max_stars:
        count_score = (n_stars - min_stars) / (max_stars - min_stars)
        count_score = max(0, min(1, count_score))
    elif n_stars <= penalty_stars:
        count_score = 1.0
    else:
        count_score = max(0, 1.0 - (n_stars - penalty_stars) / penalty_stars)

    score = 0.4 * count_score + 0.4 * non_saturated_fraction + 0.2 * snr_score

    details = {
        'n_stars': n_stars,
        'valid': True,
        'non_saturated_count': non_saturated_count,
        'non_saturated_fraction': non_saturated_fraction,
        'mean_snr': mean_snr,
        'snr_score': snr_score,
        'count_score': count_score
    }

    return score, details



def selectBestFrame(calstars, img_width, img_height, min_stars=10, max_stars=200, verbose=False):
    """ Select the best frame from the CALSTARS data based on the star distribution and quality.

    Arguments:
        calstars: [dict] Dictionary mapping FF file names to star data arrays.
        img_width: [int] Image width in pixels.
        img_height: [int] Image height in pixels.

    Keyword arguments:
        min_stars: [int] Minimum number of stars for a valid frame. 10 by default.
        max_stars: [int] Maximum number of stars before penalizing. 200 by default.
        verbose: [bool] Print detailed scoring info. False by default.

    Return:
        best_ff: [str] File name of the best frame (or None if no valid frames).
        best_score: [float] Score of the best frame.
        all_scores: [dict] Dictionary mapping FF file names to score details.
    """

    all_scores = {}

    # A camera which routinely detects more than max_stars stars must not have every good frame penalized,
    #   so only penalize frames with more than twice the night's median star count (of the frames with
    #   enough stars). On nights with a median of up to max_stars/2 this is the same as max_stars
    star_counts = [len(star_data) for star_data in calstars.values() if len(star_data) >= min_stars]
    penalty_stars = max_stars
    if star_counts:
        penalty_stars = max(max_stars, int(2*np.median(star_counts)))

    # Score every frame on both criteria
    for ff_name, star_data in calstars.items():
        star_data = np.array(star_data)

        dist_score, dist_details = scoreFrameDistribution(star_data, img_width, img_height)
        qual_score, qual_details = scoreFrameQuality(star_data, min_stars, max_stars,
                                                     penalty_stars=penalty_stars)

        # Frames that failed the quality checks are excluded regardless of their distribution
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

    # Pick the frame with the highest combined score (a zero score never wins, so it stays None)
    best_ff = None
    best_score = 0.0

    for ff_name, scores in all_scores.items():
        if scores['combined_score'] > best_score:
            best_score = scores['combined_score']
            best_ff = ff_name

    return best_ff, best_score, all_scores



### Fit reporting ###

def frameReferenceTime(ff_name, fps, ff_frames=256):
    """ Compute the reference time of an FF file, i.e. the middle of the frames it contains.

        The star positions in an FF file are averaged over all its frames, so the middle of the FF is
        used as its time everywhere else in the pipeline (e.g. recalibration and SkyFit2).

    Arguments:
        ff_name: [str] FF file name.
        fps: [float] Frames per second.

    Keyword arguments:
        ff_frames: [int] Number of frames in the FF file. 256 by default.

    Return:
        (ff_dt, jd): [tuple]
            - ff_dt: [datetime] Time of the middle of the FF file.
            - jd: [float] Julian date of the middle of the FF file.
    """

    ff_dt = getMiddleTimeFF(ff_name, fps, ff_frames=ff_frames, dt_obj=True)
    jd = date2JD(ff_dt.year, ff_dt.month, ff_dt.day, ff_dt.hour, ff_dt.minute, ff_dt.second,
                 ff_dt.microsecond/1000.0)

    return ff_dt, jd


def directoryReferenceJD(dir_path, fps):
    """ Return the Julian date of the first FF file listed in the directory's CALSTARS file or directory.

        Used to apply the star catalog proper motion for the night before the frame is chosen - the
        change of the proper motion over one night is negligible.

    Arguments:
        dir_path: [str] Path to the directory with the FF and CALSTARS files.
        fps: [float] Frames per second.

    Return:
        [float] Julian date, or None if no FF file name could be found.
    """

    file_list = sorted(os.listdir(dir_path))

    # Prefer the FF files listed in the CALSTARS file, as the FF files themselves may not be present
    ff_frames = 256
    ff_names = []
    for file_name in file_list:
        if ('CALSTARS' in file_name) and file_name.endswith('.txt'):
            calstars_data = CALSTARS.readCALSTARS(dir_path, file_name)
            if calstars_data:
                calstars_list, ff_frames = calstars_data
                ff_names = sorted(entry[0] for entry in calstars_list)
            break

    # Fall back to the FF files in the directory
    if not ff_names:
        ff_names = [file_name for file_name in file_list if validFFName(file_name)]

    if not ff_names:
        return None

    return frameReferenceTime(ff_names[0], fps, ff_frames=ff_frames)[1]


def printFitResiduals(paired_stars, platepar, jd, ff_dt):
    """ Print the fit residuals matching SkyFit2's output format.

    Arguments:
        paired_stars: [PairedStars] Paired stars object.
        platepar: [Platepar] Fitted platepar.
        jd: [float] Julian date.
        ff_dt: [datetime] Datetime of the image.

    Return:
        rmsd_img: [float] RMSD in pixels.
        rmsd_angular: [float] RMSD in the unit given by angular_error_label.
        angular_error_label: [str] Unit label for the angular RMSD (deg, arcmin or arcsec).
    """

    # Print the platepar
    print()
    print(repr(platepar))

    # Print the time info
    print()
    print("Image time = {:s} UTC".format(ff_dt.strftime("%Y-%m-%d %H:%M:%S.%f")))
    print("Image JD = {:.8f}".format(jd))
    print("Image LST = {:.8f}".format(JD2LST(jd, platepar.lon)[0]))

    # Get the catalog positions for the matched stars
    sky_coords = np.array(paired_stars.skyCoords())
    catalog_x, catalog_y, catalog_mag = getCatalogStarsImagePositions(sky_coords, jd, platepar)

    # Print the header
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

        # Compute the image residual and angle
        angle = np.arctan2(delta_y, delta_x)
        distance = np.sqrt(delta_x**2 + delta_y**2)

        # Compute RA/Dec from the image position (using the JD time format)
        _, ra_img, dec_img, _ = xyToRaDecPP(
            [jd], [img_x], [img_y], [1], platepar, extinction_correction=False, jd_time=True
        )
        ra_img = ra_img[0]
        dec_img = dec_img[0]

        # Compute the angular distance
        angular_distance = np.degrees(angularSeparation(
            np.radians(ra), np.radians(dec),
            np.radians(ra_img), np.radians(dec_img)
        ))

        residuals.append([img_x, img_y, angle, distance, angular_distance])

        # Guard against missing photometry and star shape data in older CALSTARS files
        lsp = -2.5*np.log10(sum_intens) if sum_intens and sum_intens > 0 else 0
        fwhm_val = fwhm if fwhm is not None else 0.0
        mag_val = mag if mag is not None else 0.0

        # Print the residual line
        print('{:3d}, {:11.6f}, {:11.6f}, {:>12.6f}, {:>+13.6f}, {:8.2f}, {:8.2f}, {:>12.6f}, {:>+13.6f}, {:8.2f}, {:7.2f}, {:+9.1f}, {:5.2f}, {:+6.2f}, {:8.2f}'.format(
            star_no + 1, img_x, img_y, ra, dec, cat_x, cat_y,
            ra_img, dec_img, 60*angular_distance, distance, np.degrees(angle),
            fwhm_val, mag_val, lsp
        ))

    # Compute the RMSD errors
    rmsd_angular = 60*RMSD([entry[4] for entry in residuals])
    rmsd_img = RMSD([entry[3] for entry in residuals])

    # Determine the appropriate angular unit
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



### Automatic platepar fit ###

def autoFitPlatepar(dir_path, config, catalog_stars, platepar_template=None,
                    fov_w_hint=None, ff_name=None, distortion_type=DEFAULT_DISTORTION_TYPE,
                    equal_aspect=DEFAULT_EQUAL_ASPECT, asymmetry_corr=DEFAULT_ASYMMETRY_CORR,
                    force_distortion_centre=DEFAULT_FORCE_DISTORTION_CENTRE,
                    refraction=DEFAULT_REFRACTION,
                    photometric_sigma=DEFAULT_PHOTOMETRIC_SIGMA,
                    fwhm_mult=DEFAULT_BLEND_FWHM_MULT,
                    wide_fov_search=False,
                    final_catalog_stars=None,
                    verbose=True):
    """ Automatically create a platepar from the CALSTARS data in a directory.

        This function replicates the behaviour of SkyFit2's "Auto Fit" button:
            1. Loads the CALSTARS data from the directory
            2. Selects the best frame based on the star distribution (or uses the specified frame)
            3. Runs astrometry.net plate solving
            4. Performs an NN-based refinement with intermediate settings
            5. Applies star filtering (photometric outliers, blended stars)
            6. Does the final fit with the user-specified settings
            7. Returns a fitted Platepar

    Arguments:
        dir_path: [str] Path to the directory containing the CALSTARS file and FF files.
        config: [Config] RMS configuration object.
        catalog_stars: [ndarray] Star catalog array (RA, Dec, mag, ...).

    Keyword arguments:
        platepar_template: [Platepar] Optional template platepar with the station location. If None,
            the config values are used.
        fov_w_hint: [float] Optional FOV width hint in degrees. If None, config.fov_w is used.
        ff_name: [str] Optional specific FF file name to use. If None, the best frame from CALSTARS is
            selected automatically.
        distortion_type: [str] Distortion model for the final fit. DEFAULT_DISTORTION_TYPE by default.
        equal_aspect: [bool] Equal aspect ratio constraint. DEFAULT_EQUAL_ASPECT by default.
        asymmetry_corr: [bool] Asymmetry correction. DEFAULT_ASYMMETRY_CORR by default.
        force_distortion_centre: [bool] Force the distortion centre to the image centre.
            DEFAULT_FORCE_DISTORTION_CENTRE by default.
        refraction: [bool] Apply the refraction correction. DEFAULT_REFRACTION by default.
        photometric_sigma: [float] Sigma threshold for the photometric outlier removal.
            DEFAULT_PHOTOMETRIC_SIGMA by default.
        fwhm_mult: [float] Multiplier of the FWHM for the blend detection radius. DEFAULT_BLEND_FWHM_MULT
            by default.
        wide_fov_search: [bool] If True, use a wide FOV search range (2 to 200 deg) instead of the
            config-based range. Used as a fallback when the tight search fails. False by default.
        final_catalog_stars: [ndarray] Optional deeper star catalog used for the last stage of the NN
            fit. None by default, in which case catalog_stars is used throughout.
        verbose: [bool] Print progress information. True by default.

    Return:
        platepar: [Platepar] Fitted platepar object, or None if fitting failed.
        matched_stars: [PairedStars] Matched star pairs (an empty list if the fit failed after the
            frame was chosen, None if it failed before).
        best_ff: [str] File name of the frame used for fitting.
    """

    if verbose:
        print("=" * 70)
        print("Auto Platepar Fitting")
        print("=" * 70)
        print("Directory: {:s}".format(dir_path))

    # Find the CALSTARS file (the number of frames per FF is read from it, 256 by default)
    calstars_file = None
    ff_frames = 256
    for f in os.listdir(dir_path):
        if 'CALSTARS' in f and f.endswith('.txt'):
            calstars_file = f
            break

    # Run the star extraction if there is no CALSTARS file yet
    if calstars_file is None:
        if verbose:
            print("No CALSTARS file found, generating automatically...")

        try:
            calstars_list = extractStarsAndSave(config, dir_path)
        except Exception as e:
            if verbose:
                print("ERROR: Failed to generate CALSTARS file: {:s}".format(str(e)))
            return None, None, None

        if calstars_list is None or len(calstars_list) == 0:
            if verbose:
                print("ERROR: Failed to generate CALSTARS file")
            return None, None, None

    # Otherwise load the existing one
    else:
        calstars_list, ff_frames = CALSTARS.readCALSTARS(dir_path, calstars_file)

    calstars = {ff_file: star_data for ff_file, star_data in calstars_list}

    if verbose:
        if calstars_file is not None:
            print("Loaded CALSTARS: {:s} ({:d} frames)".format(calstars_file, len(calstars)))
        else:
            print("Generated CALSTARS with {:d} frames".format(len(calstars)))

    img_width = config.width
    img_height = config.height

    # Use the specified frame
    if ff_name is not None:

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

    # Otherwise find the best frame automatically
    else:
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
                details['quality_details']['non_saturated_fraction'] * 100
            ))

    # Get the star data for the best frame
    star_data = np.array(calstars[best_ff])

    # Init the platepar with the station location and resolution from the template
    if platepar_template is not None:
        platepar = Platepar()
        platepar.lat = platepar_template.lat
        platepar.lon = platepar_template.lon
        platepar.elev = platepar_template.elev
        platepar.X_res = platepar_template.X_res
        platepar.Y_res = platepar_template.Y_res
        platepar.station_code = platepar_template.station_code

    # Or from the config
    else:
        platepar = Platepar()
        platepar.lat = config.latitude
        platepar.lon = config.longitude
        platepar.elev = config.elevation
        platepar.X_res = config.width
        platepar.Y_res = config.height
        platepar.station_code = config.stationID

    # Initialize the vignetting coefficient with the resolution-scaled default. The flat is not applied
    #   during platepar creation, so the photometry needs the vignetting correction
    platepar.addVignettingCoeff(use_flat=False)

    # Get the time of the middle of the FF file (the FF file name only gives the time of its first frame)
    ff_dt, jd = frameReferenceTime(best_ff, config.fps, ff_frames=ff_frames)

    platepar.JD = jd
    platepar.Ho = JD2HourAngle(jd)

    # Take the FOV hint from the config if not given
    if fov_w_hint is None:
        fov_w_hint = config.fov_w

    # Construct the FOV width search range
    if wide_fov_search:

        # The wide search range covers all common lens types (2 deg telephoto to 200 deg fisheye)
        fov_w_range = [2, max(200, 1.5 * fov_w_hint)]

    else:

        # Tight search range based on the config (0.75x to 1.5x)
        fov_w_range = [0.75 * fov_w_hint, 1.5 * fov_w_hint]

    # Load the mask if available
    mask = getMaskFile(dir_path, config)

    # Extract the star coordinates (CALSTARS format: y, x, ...). Older CALSTARS files lack the photometry
    #   and star shape columns, so neutral values are assumed for them
    y_data = star_data[:, 0]
    x_data = star_data[:, 1]
    input_intensities = star_data[:, 2] if star_data.shape[1] > 2 else None
    input_fwhm = star_data[:, 4] if star_data.shape[1] > 4 else np.zeros(len(x_data))
    input_snr = star_data[:, 6] if star_data.shape[1] > 6 else np.ones(len(x_data))
    input_saturated = star_data[:, 7] if star_data.shape[1] > 7 else np.zeros(len(x_data))

    if verbose:
        print()
        search_mode = "wide" if wide_fov_search else "tight"
        print("Running astrometry.net plate solving ({:s} FOV search)...".format(search_mode))
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

        # If the tight FOV search failed, retry with the wide search on the same frame as a fallback
        if not wide_fov_search:
            if verbose:
                print("Tight FOV search failed, trying wide FOV search...")
            return autoFitPlatepar(
                dir_path, config, catalog_stars, platepar_template=platepar_template,
                fov_w_hint=fov_w_hint, ff_name=best_ff, distortion_type=distortion_type,
                equal_aspect=equal_aspect, asymmetry_corr=asymmetry_corr,
                force_distortion_centre=force_distortion_centre,
                refraction=refraction,
                photometric_sigma=photometric_sigma,
                fwhm_mult=fwhm_mult,
                wide_fov_search=True,
                final_catalog_stars=final_catalog_stars,
                verbose=verbose
            )

        if verbose:
            print("ERROR: Astrometry.net failed to find a solution")
        return None, [], best_ff

    # Extract the solution
    ra, dec, rot_standard, scale, fov_w, fov_h, matched_star_data, solution_info = solution

    if verbose:
        print()
        print("Astrometry.net solution:")
        print("  RA = {:.2f} deg".format(ra))
        print("  Dec = {:.2f} deg".format(dec))
        print("  Scale = {:.3f} arcmin/px".format(60 / scale))
        print("  FOV = {:.2f} x {:.2f} deg".format(fov_w, fov_h))

    # Apply the solution to the platepar
    platepar.F_scale = scale

    # Compute the azimuth and altitude from RA/Dec
    azim, alt = trueRaDec2ApparentAltAz(ra, dec, jd, platepar.lat, platepar.lon)
    platepar.az_centre = azim
    platepar.alt_centre = alt

    # Set the reference pointing and convert the astrometry.net rotation to the platepar convention
    platepar.updateRefRADec(skip_rot_update=True)
    platepar.pos_angle_ref = rotationWrtStandardToPosAngle(platepar, rot_standard)

    # Set the INTERMEDIATE fitting parameters (matching SkyFit2). These are used for the NN refinement,
    #   NOT the final fit
    platepar.refraction = True
    platepar.equal_aspect = True
    platepar.asymmetry_corr = False
    platepar.force_distortion_centre = False
    platepar.setDistortionType("radial5-odd", reset_params=True)

    if verbose:
        print()
        print("Performing NN-based refinement...")

    # Prepare the detected stars array
    img_stars_arr = np.column_stack([x_data, y_data,
                                     input_intensities if input_intensities is not None else np.ones(len(x_data))])

    try:

        # Fit using the NN cost function (intermediate fit)
        result = platepar.fitAstrometry(
            jd, img_stars_arr, catalog_stars,
            first_platepar_fit=True,
            use_nn_cost=True,
            final_catalog_stars=final_catalog_stars
        )

        if verbose:
            print("  NN fit complete")
            print("  RA = {:.2f} deg, Dec = {:.2f} deg, Scale = {:.3f} arcmin/px".format(
                platepar.RA_d, platepar.dec_d, 60 / platepar.F_scale))

    except Exception as e:
        if verbose:
            print("ERROR: NN fitting failed: {:s}".format(str(e)))
        return None, [], best_ff

    # Build the paired stars from the NN fit results
    paired_stars = PairedStars()

    if hasattr(platepar, 'star_list') and platepar.star_list:

        # Find the closest detected star of every fitted star (one KD-tree query for all of them) to get
        #   the FWHM, SNR and saturation
        # star_list format: [jd, x, y, intensity, ra, dec, mag]
        star_arr = np.array([entry[1:3] for entry in platepar.star_list], dtype=np.float64)
        closest_indices = associateDetections(star_arr[:, 0], star_arr[:, 1], x_data, y_data,
                                              radius=NN_PAIR_ASSOC_RADIUS_PX)

        for entry, closest_idx in zip(platepar.star_list, closest_indices):
            _, img_x, img_y, intensity, cat_ra, cat_dec, cat_mag = entry
            sky_obj = CatalogStar(cat_ra, cat_dec, cat_mag)

            # Fall back to nominal star shape values if no detection is close enough
            fwhm, snr, saturated = 2.5, 1.0, False
            if closest_idx >= 0:
                fwhm = input_fwhm[closest_idx]
                snr = input_snr[closest_idx]
                saturated = input_saturated[closest_idx] > 0

            paired_stars.addPair(img_x, img_y, fwhm, intensity, sky_obj, snr=snr, saturated=saturated)

        if verbose:
            print("  Matched pairs: {:d}".format(len(paired_stars)))

    # Check if there are enough stars for the final fit
    if len(paired_stars) < 10:
        if verbose:
            print("ERROR: Not enough matched stars for final fit ({:d} < 10)".format(len(paired_stars)))
        return None, [], best_ff

    # Apply the star filtering (matching SkyFit2)
    if verbose:
        print()
        print("Filtering stars...")

    # Only filter when there are enough stars left over for the filters to be meaningful
    if len(paired_stars) >= 15:
        paired_stars, _ = filterPhotometricOutliers(
            paired_stars, platepar, jd, sigma_threshold=photometric_sigma, verbose=verbose)

    if len(paired_stars) >= 15:
        paired_stars, _ = filterBlendedStars(
            paired_stars, catalog_stars, platepar, jd, config.catalog_mag_limit,
            fwhm_mult=fwhm_mult, verbose=verbose)

    if verbose:
        print("  Stars after filtering: {:d}".format(len(paired_stars)))

    # Check again after filtering
    if len(paired_stars) < 10:
        if verbose:
            print("ERROR: Not enough stars after filtering ({:d} < 10)".format(len(paired_stars)))
        return None, [], best_ff

    # Apply the USER's settings for the final fit
    if verbose:
        print()
        print("Final fit with user settings...")
        print("  distortion_type: {:s}".format(distortion_type))
        print("  equal_aspect: {:s}".format(str(equal_aspect)))
        print("  asymmetry_corr: {:s}".format(str(asymmetry_corr)))
        print("  force_distortion_centre: {:s}".format(str(force_distortion_centre)))
        print("  refraction: {:s}".format(str(refraction)))

    # Use remapCoeffsForFlagChange to properly handle the coefficient structure changes
    platepar.remapCoeffsForFlagChange('equal_aspect', equal_aspect)
    platepar.remapCoeffsForFlagChange('asymmetry_corr', asymmetry_corr)
    platepar.remapCoeffsForFlagChange('force_distortion_centre', force_distortion_centre)
    platepar.refraction = refraction
    platepar.setDistortionType(distortion_type, reset_params=False)

    # Extract the coordinates for the final fit
    img_coords = np.array(paired_stars.imageCoords())
    sky_coords = np.array(paired_stars.skyCoords())

    # Do the final fit
    try:
        platepar.fitAstrometry(jd, img_coords, sky_coords, first_platepar_fit=True)
    except Exception as e:
        if verbose:
            print("ERROR: Final fit failed: {:s}".format(str(e)))
        return None, [], best_ff

    if verbose:

        # Print the full residuals report (matching the SkyFit2 output)
        try:
            printFitResiduals(paired_stars, platepar, jd, ff_dt)

        # Fall back to a short summary if the report cannot be produced
        except Exception as e:
            print("WARNING: Could not print residuals: {:s}".format(str(e)))
            print()
            print("Final platepar:")
            print("  RA = {:.4f} deg".format(platepar.RA_d))
            print("  Dec = {:.4f} deg".format(platepar.dec_d))
            print("  Scale = {:.4f} arcmin/px".format(60 / platepar.F_scale))
            print("  Matched stars: {:d}".format(len(paired_stars)))

    return platepar, paired_stars, best_ff



### Catalog loading ###

def loadCatalogStars(config, lim_mag, jd=None):
    """ Load the star catalog for plate solving.

    Arguments:
        config: [Config] RMS configuration object.
        lim_mag: [float] Limiting magnitude.

    Keyword arguments:
        jd: [float] Julian date for the proper motion correction. None by default, in which case the
            J2000 positions are used.

    Return:
        catalog_stars: [ndarray] Star catalog array, or None if the catalog could not be read.
    """

    # Fall back to the catalogs shipped with RMS if the configured directory does not exist
    star_catalog_path = config.star_catalog_path
    if not os.path.isdir(star_catalog_path):
        star_catalog_path = os.path.join(config.rms_root_dir, 'Catalogs')

    if jd is not None:
        # Computed from the JD directly so the day fraction is kept (a .days difference truncates)
        years_from_J2000 = jd2YearsFromJ2000(jd)
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

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    parser = argparse.ArgumentParser(description="Auto-fit platepar from CALSTARS data")
    parser.add_argument("dir_path", help="Path to directory with CALSTARS file")
    parser.add_argument("-c", "--config", nargs=1, metavar='CONFIG_PATH', type=str,
                        help="Path to config file", default=None)

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
                        help="Photometric outlier sigma threshold (default: {:.1f})".format(
                            DEFAULT_PHOTOMETRIC_SIGMA))
    parser.add_argument("--fwhm-mult", type=float, default=DEFAULT_BLEND_FWHM_MULT,
                        help="FWHM multiplier for blend detection radius (default: {:.1f})".format(
                            DEFAULT_BLEND_FWHM_MULT))

    parser.add_argument("-o", "--output", help="Output platepar filename", default="platepar_auto.cal")

    # Parse the command line arguments
    args = parser.parse_args()

    #########################

    # Load the config
    config = cr.loadConfigFromDirectory(args.config if args.config is not None else '.', args.dir_path)

    # Load the star catalog, with the proper motion applied to the time of the data
    catalog_stars = loadCatalogStars(config, config.catalog_mag_limit,
                                     jd=directoryReferenceJD(args.dir_path, config.fps))

    if catalog_stars is None:
        print("ERROR: Could not load star catalog")
        exit(1)

    print("Loaded star catalog: {:d} stars".format(len(catalog_stars)))

    # Run the auto-fit
    platepar, matched_stars, best_ff = autoFitPlatepar(
        args.dir_path, config, catalog_stars,
        distortion_type=args.distortion,
        equal_aspect=not args.no_equal_aspect,
        asymmetry_corr=not args.no_asymmetry_corr,
        force_distortion_centre=args.force_distortion_centre,
        refraction=not args.no_refraction,
        photometric_sigma=args.photom_sigma,
        fwhm_mult=args.fwhm_mult,
        verbose=True
    )

    # Save the platepar next to the input data
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
