"""
Star filtering functions for astrometry operations.

This module provides functions to filter paired stars based on various criteria:
- Photometric outliers (magnitude residuals)
- Blended stars (nearby bright neighbors)
- High FWHM stars (poor PSF quality)

These functions are used by both SkyFit2 and AutoPlatepar.
"""

from __future__ import print_function, division, absolute_import

import numpy as np

from scipy.spatial import cKDTree

from RMS.Astrometry.StarClasses import PairedStars
from RMS.Astrometry.ApplyAstrometry import extinctionCorrectionTrueToApparent, raDecToXYPP, \
    getFOVSelectionRadius, xyToRaDecPP


# Default filtering parameters
DEFAULT_PHOTOMETRIC_SIGMA = 2.5
DEFAULT_BLEND_FWHM_MULT = 2.0  # Multiplier of FWHM for blending detection radius
DEFAULT_BLEND_MAG_MARGIN = 0.3  # Margin above limiting magnitude for blend check

# Margin applied to the FOV selection radius when pre-filtering the catalog. The radius
#   already circumscribes the image corners, so the margin only has to cover neighbours
#   sitting a few pixels outside the frame - it is generous rather than tight.
DEFAULT_FOV_RADIUS_MARGIN = 1.5


def catalogStarsInFOV(catalog_ra, catalog_dec, platepar, jd, margin=DEFAULT_FOV_RADIUS_MARGIN):
    """ Mask of the catalog stars which lie inside the FOV cone at the given time.

    Stars behind the camera can reverse-project into valid-looking pixel coordinates, so a
    cone around the pointing direction is used to reject them before projecting.

    Arguments:
        catalog_ra: [ndarray] Catalog star right ascensions (deg).
        catalog_dec: [ndarray] Catalog star declinations (deg).
        platepar: [Platepar] Platepar for the FOV geometry.
        jd: [float] Julian date.

    Keyword arguments:
        margin: [float] Multiplier applied to the FOV selection radius.
            Default is DEFAULT_FOV_RADIUS_MARGIN.

    Returns:
        in_fov: [ndarray] Boolean mask, True for stars inside the cone.
    """
    # Radius which includes the image corners, computed by projecting them through the
    #   platepar (distortion included) instead of assuming the central F_scale holds all
    #   the way out to the corners
    fov_radius = min(getFOVSelectionRadius(platepar)*margin, 90)

    # Centre the cone on the pointing at THIS jd. platepar.RA_d/dec_d is the pointing at
    #   platepar.JD, and on an alt-az camera the two drift ~15 deg/hour apart, which would
    #   eat the margin above whenever a platepar is reused across a night.
    _, ra_centre, dec_centre, _ = xyToRaDecPP([jd], [platepar.X_res/2.0], [platepar.Y_res/2.0], [1],
                                              platepar, extinction_correction=False, jd_time=True)

    ra_centre = np.radians(ra_centre[0])
    dec_centre = np.radians(dec_centre[0])

    ra_rad = np.radians(catalog_ra)
    dec_rad = np.radians(catalog_dec)

    # Spherical angular distance from the pointing direction to each catalog star
    cos_ang_dist = (np.sin(dec_centre)*np.sin(dec_rad)
                    + np.cos(dec_centre)*np.cos(dec_rad)*np.cos(ra_rad - ra_centre))
    cos_ang_dist = np.clip(cos_ang_dist, -1, 1)

    return np.degrees(np.arccos(cos_ang_dist)) < fov_radius


def filterPhotometricOutliers(paired_stars, platepar, jd, sigma_threshold=DEFAULT_PHOTOMETRIC_SIGMA,
                               verbose=False):
    """
    Filter paired_stars by removing photometric outliers.

    Stars whose magnitude residuals (catalog - instrumental) deviate by more than
    sigma_threshold standard deviations from the median are removed.

    Arguments:
        paired_stars: [PairedStars] Paired stars object.
        platepar: [Platepar] Current platepar for extinction correction.
        jd: [float] Julian date.

    Keyword arguments:
        sigma_threshold: [float] Number of standard deviations for outlier detection.
            Default is 2.5.
        verbose: [bool] Print filtering info. Default is False.

    Returns:
        new_paired_stars: [PairedStars] Filtered paired stars.
        removed_count: [int] Number of stars removed.
    """
    if len(paired_stars) < 10:
        return paired_stars, 0

    residuals = []
    valid_indices = []
    ra_list = []
    dec_list = []

    for i, (x, y, fwhm, intens_acc, obj, snr, saturated) in enumerate(paired_stars.paired_stars):
        if saturated:
            continue

        if hasattr(obj, 'pick_type') and obj.pick_type == "geopoint":
            continue

        ra, dec, cat_mag = obj.coords()

        if intens_acc <= 0 or np.isnan(intens_acc) or np.isinf(intens_acc):
            continue

        inst_mag = -2.5 * np.log10(intens_acc)

        residuals.append((cat_mag, inst_mag))
        valid_indices.append(i)
        ra_list.append(ra)
        dec_list.append(dec)

    if len(residuals) < 5:
        return paired_stars, 0

    cat_mags = np.array([r[0] for r in residuals])
    inst_mags = np.array([r[1] for r in residuals])

    cat_mags_corrected = extinctionCorrectionTrueToApparent(cat_mags, ra_list, dec_list, jd, platepar)

    mag_residuals = cat_mags_corrected - inst_mags

    median = np.median(mag_residuals)
    std = np.std(mag_residuals)

    if std < 0.01:
        return paired_stars, 0

    outlier_mask = np.abs(mag_residuals - median) > sigma_threshold * std
    outlier_indices = set(valid_indices[i] for i, is_outlier in enumerate(outlier_mask) if is_outlier)

    if len(outlier_indices) > 0:
        new_paired_stars = PairedStars()
        for i, (x, y, fwhm, intens_acc, obj, snr, saturated) in enumerate(paired_stars.paired_stars):
            if i not in outlier_indices:
                new_paired_stars.addPair(x, y, fwhm, intens_acc, obj, snr, saturated)

        if verbose:
            print("  Removed {:d} photometric outliers (>{:.1f} sigma)".format(
                len(outlier_indices), sigma_threshold))

        return new_paired_stars, len(outlier_indices)

    return paired_stars, 0


def filterBlendedStars(paired_stars, catalog_stars, platepar, jd, lim_mag,
                       fwhm_mult=DEFAULT_BLEND_FWHM_MULT,
                       mag_margin=DEFAULT_BLEND_MAG_MARGIN, verbose=False):
    """
    Filter paired_stars by removing likely blended stars.

    A star is considered blended if there are other catalog stars (brighter than
    lim_mag + mag_margin) within fwhm_mult * FWHM pixels of the star.

    Arguments:
        paired_stars: [PairedStars] Paired stars object.
        catalog_stars: [ndarray] Full catalog stars array with columns [ra, dec, mag, ...].
        platepar: [Platepar] Platepar for coordinate conversion.
        jd: [float] Julian date.
        lim_mag: [float] Current limiting magnitude for star detection.

    Keyword arguments:
        fwhm_mult: [float] Multiplier of the star's FWHM for blend detection radius.
            Default is 2.0.
        mag_margin: [float] Margin above lim_mag - only consider catalog stars
            brighter than (lim_mag + mag_margin). Default is 0.3.
        verbose: [bool] Print filtering info. Default is False.

    Returns:
        new_paired_stars: [PairedStars] Filtered paired stars.
        removed_count: [int] Number of stars removed.
    """
    if len(paired_stars) < 5 or catalog_stars is None:
        return paired_stars, 0

    # Only consider catalog stars bright enough to be detectable
    max_mag = lim_mag + mag_margin
    bright_mask = catalog_stars[:, 2] < max_mag

    if np.sum(bright_mask) == 0:
        return paired_stars, 0

    # Get bright catalog star coordinates
    catalog_ra = catalog_stars[bright_mask, 0]
    catalog_dec = catalog_stars[bright_mask, 1]

    # Filter to stars actually in front of the camera (within FOV + margin)
    # This prevents false positives from stars behind the camera that could
    # project to valid-looking pixel coordinates
    in_fov = catalogStarsInFOV(catalog_ra, catalog_dec, platepar, jd)
    catalog_ra = catalog_ra[in_fov]
    catalog_dec = catalog_dec[in_fov]

    if len(catalog_ra) == 0:
        return paired_stars, 0

    # Convert FOV-filtered catalog stars to pixel coordinates
    catalog_x, catalog_y = raDecToXYPP(catalog_ra, catalog_dec, jd, platepar)

    blended_indices = set()

    # Collect matched star data for batch projection
    check_indices = []
    matched_ra_list = []
    matched_dec_list = []
    blend_radii = []
    for i, (x, y, fwhm, intens_acc, obj, snr, saturated) in enumerate(paired_stars.paired_stars):
        if hasattr(obj, 'pick_type') and obj.pick_type == "geopoint":
            continue
        ra, dec, mag = obj.coords()
        check_indices.append(i)
        matched_ra_list.append(ra)
        matched_dec_list.append(dec)
        blend_radii.append(fwhm_mult * fwhm)

    if len(check_indices) > 0:
        # Batch project all matched stars to pixel coordinates in one call
        all_matched_x, all_matched_y = raDecToXYPP(
            np.array(matched_ra_list), np.array(matched_dec_list), jd, platepar)
        blend_radii = np.array(blend_radii)

        # Every matched star is inside the image, so a catalog star further outside the
        # frame than the largest blend radius can never be a blend neighbour. This is
        # exact in the units that matter (pixels) and removes most of what the cone
        # pre-filter necessarily lets through.
        edge_margin = np.max(blend_radii)
        in_img = ((catalog_x > -edge_margin) & (catalog_x < platepar.X_res + edge_margin)
                  & (catalog_y > -edge_margin) & (catalog_y < platepar.Y_res + edge_margin))
        catalog_x = catalog_x[in_img]
        catalog_y = catalog_y[in_img]

        if len(catalog_x) == 0:
            return paired_stars, 0

        # Nearest-neighbour lookup rather than an (n_matched x n_catalog) distance matrix.
        # O(N log M) with constant memory, so a deep catalog cannot allocate the multi-GB
        # matrices that used to get the process OOM-killed, and there is no chunk size to
        # tune. Same reasoning as RMS/Astrometry/MatchStars.py.
        tree = cKDTree(np.column_stack([catalog_x, catalog_y]))

        # A matched star normally coincides with its own catalog entry (d ~ 0), which the
        # d > 0.1 test below drops. Ask for a few neighbours rather than one so the search
        # still reaches a genuine neighbour when the catalog holds duplicate entries at
        # that position. Missing neighbours come back as inf, which fails both tests.
        n_neighbors = min(3, len(catalog_x))
        nn_dist, _ = tree.query(np.column_stack([all_matched_x, all_matched_y]), k=n_neighbors)
        nn_dist = np.reshape(nn_dist, (len(check_indices), n_neighbors))

        # Neighbours within each star's own blend radius (excluding self)
        has_neighbor = np.any((nn_dist < blend_radii[:, np.newaxis]) & (nn_dist > 0.1), axis=1)

        for k, idx in enumerate(check_indices):
            if has_neighbor[k]:
                blended_indices.add(idx)

    if len(blended_indices) > 0:
        new_paired_stars = PairedStars()
        for i, (x, y, fwhm, intens_acc, obj, snr, saturated) in enumerate(paired_stars.paired_stars):
            if i not in blended_indices:
                new_paired_stars.addPair(x, y, fwhm, intens_acc, obj, snr, saturated)

        if verbose:
            print("  Removed {:d} blended stars (catalog neighbors within {:.1f}x FWHM, mag < {:.1f})".format(
                len(blended_indices), fwhm_mult, max_mag))

        return new_paired_stars, len(blended_indices)

    return paired_stars, 0
