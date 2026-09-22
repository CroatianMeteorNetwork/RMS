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
from RMS.Astrometry.ApplyAstrometry import extinctionCorrectionTrueToApparent, raDecToXYPP
from RMS.Math import angularSeparationDeg


# Default filtering parameters
DEFAULT_PHOTOMETRIC_SIGMA = 2.5
DEFAULT_BLEND_FWHM_MULT = 2.0  # Multiplier of FWHM for blending detection radius
DEFAULT_BLEND_MAG_MARGIN = 0.3  # Margin above limiting magnitude for blend check

# Multiplicative margin on the FOV radius used when pre-selecting catalog stars
DEFAULT_FOV_RADIUS_MARGIN = 1.5



def fovRadiusDeg(platepar, margin=DEFAULT_FOV_RADIUS_MARGIN):
    """ Estimate the angular radius of the field of view from the platepar.

        Half the image diagonal is converted from pixels to degrees with the plate scale. The result is
        capped at 90 deg, as nothing further from the pointing than that can be in front of the camera.

    Arguments:
        platepar: [Platepar object] Platepar with X_res, Y_res (px) and F_scale (px/deg).

    Keyword arguments:
        margin: [float] Multiplicative margin on the radius. 1.5 by default, i.e. a 50% margin.

    Return:
        fov_radius: [float] FOV radius (deg), at most 90 deg.
    """

    # Half the image diagonal in pixels
    half_diagonal_px = np.sqrt(platepar.X_res**2 + platepar.Y_res**2)/2

    # Convert to degrees with the plate scale (px/deg) and apply the margin
    fov_radius = margin*half_diagonal_px/platepar.F_scale

    return min(fov_radius, 90.0)



def catalogInFOVMask(catalog_ra, catalog_dec, platepar, margin=DEFAULT_FOV_RADIUS_MARGIN):
    """ Select the catalog stars within the FOV radius of the platepar pointing.

        This is used before projecting catalog stars to image coordinates, as stars behind the camera
        can otherwise project to valid-looking pixel positions.

    Arguments:
        catalog_ra: [ndarray] Catalog right ascensions (deg).
        catalog_dec: [ndarray] Catalog declinations (deg).
        platepar: [Platepar object] Platepar with the reference pointing RA_d, dec_d (deg).

    Keyword arguments:
        margin: [float] Multiplicative margin on the FOV radius. 1.5 by default.

    Return:
        in_fov: [ndarray of bool] True for catalog stars within the FOV radius.
    """

    # Angular distance from the pointing to every catalog star
    ang_dist_deg = angularSeparationDeg(platepar.RA_d, platepar.dec_d, np.asarray(catalog_ra),
        np.asarray(catalog_dec))

    return ang_dist_deg < fovRadiusDeg(platepar, margin=margin)



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

    # Keep only the stars in front of the camera (within the FOV radius plus a margin). This prevents
    # false positives from stars behind the camera that could project to valid-looking pixel coordinates.
    in_fov = catalogInFOVMask(catalog_ra, catalog_dec, platepar)
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

        # Find the catalog stars within each star's blend radius with a KD-tree (O(N log M) instead of the
        #   dense N x M distance matrix). The ball query is inclusive (dist <= r) and returns everything
        #   for a negative radius, so non-positive or non-finite radii are queried with 0 and the exact
        #   (0.1 < dist < r) rule is re-applied on the candidates below. The candidate order does not
        #   matter, but return_sorted is not passed: it only exists in scipy >= 1.2 and requirements.txt
        #   still allows 1.0
        matched_coords = np.column_stack([all_matched_x, all_matched_y])
        query_radii = np.where(np.isfinite(blend_radii) & (blend_radii > 0), blend_radii, 0.0)
        catalog_tree = cKDTree(np.column_stack([catalog_x, catalog_y]))
        candidates = catalog_tree.query_ball_point(matched_coords, query_radii)

        # Flatten the candidate lists into (matched, catalog) index pairs
        n_candidates = np.array([len(c) for c in candidates], dtype=int)
        matched_idx = np.repeat(np.arange(len(candidates)), n_candidates)
        catalog_idx = np.concatenate([np.asarray(c, dtype=int) for c in candidates]) \
            if np.any(n_candidates) else np.zeros(0, dtype=int)

        # Check for neighbors within each star's blend radius (excluding self)
        dist = np.sqrt((all_matched_x[matched_idx] - catalog_x[catalog_idx])**2
            + (all_matched_y[matched_idx] - catalog_y[catalog_idx])**2)
        is_neighbor = (dist < blend_radii[matched_idx]) & (dist > 0.1)
        has_neighbor = np.zeros(len(candidates), dtype=bool)
        has_neighbor[matched_idx[is_neighbor]] = True

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
