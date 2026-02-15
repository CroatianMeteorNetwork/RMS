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

from RMS.Astrometry.StarClasses import PairedStars
from RMS.Astrometry.ApplyAstrometry import extinctionCorrectionTrueToApparent, raDecToXYPP


# Default filtering parameters
DEFAULT_PHOTOMETRIC_SIGMA = 2.5
DEFAULT_BLEND_FWHM_MULT = 2.0  # Multiplier of FWHM for blending detection radius
DEFAULT_BLEND_MAG_MARGIN = 0.3  # Margin above limiting magnitude for blend check
DEFAULT_HIGH_FWHM_FRACTION = 0.10


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
    ra_rad = np.radians(catalog_ra)
    dec_rad = np.radians(catalog_dec)
    ra_center = np.radians(platepar.RA_d)
    dec_center = np.radians(platepar.dec_d)

    # Spherical angular distance from camera pointing to each catalog star
    cos_ang_dist = (np.sin(dec_center) * np.sin(dec_rad) +
                    np.cos(dec_center) * np.cos(dec_rad) * np.cos(ra_rad - ra_center))
    cos_ang_dist = np.clip(cos_ang_dist, -1, 1)
    ang_dist_deg = np.degrees(np.arccos(cos_ang_dist))

    # Estimate FOV radius from platepar (diagonal / 2 * scale, with margin)
    fov_diagonal = np.sqrt(platepar.X_res**2 + platepar.Y_res**2)
    fov_radius = (fov_diagonal / 2) * platepar.F_scale * 1.5  # 50% margin
    fov_radius = min(fov_radius, 90)  # Cap at 90 degrees

    in_fov = ang_dist_deg < fov_radius
    catalog_ra = catalog_ra[in_fov]
    catalog_dec = catalog_dec[in_fov]

    if len(catalog_ra) == 0:
        return paired_stars, 0

    # Convert FOV-filtered catalog stars to pixel coordinates
    catalog_x, catalog_y = raDecToXYPP(catalog_ra, catalog_dec, jd, platepar)

    blended_indices = set()

    # Check each paired star's matched catalog position against other catalog stars
    for i, (x, y, fwhm, intens_acc, obj, snr, saturated) in enumerate(paired_stars.paired_stars):
        if hasattr(obj, 'pick_type') and obj.pick_type == "geopoint":
            continue

        # Compute blend radius based on this star's FWHM
        blend_radius = fwhm_mult * fwhm

        # Get the matched catalog star's RA/Dec and project to pixels
        matched_ra, matched_dec, mag = obj.coords()
        matched_x, matched_y = raDecToXYPP(np.array([matched_ra]), np.array([matched_dec]), jd, platepar)
        matched_x, matched_y = matched_x[0], matched_y[0]

        # Compute pixel distances from matched catalog star to all bright catalog stars
        pixel_dist = np.sqrt((matched_x - catalog_x)**2 + (matched_y - catalog_y)**2)

        # Find neighbors within radius (excluding self - use small threshold for floating point)
        neighbor_mask = (pixel_dist < blend_radius) & (pixel_dist > 0.1)

        if np.sum(neighbor_mask) > 0:
            blended_indices.add(i)

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


def filterHighFWHMStars(paired_stars, fraction=DEFAULT_HIGH_FWHM_FRACTION, verbose=False):
    """
    Filter paired_stars by removing the worst fraction of stars by FWHM.

    Stars with high FWHM tend to have worse centroiding precision due to:
    - Blended sources
    - Extended objects (galaxies)
    - Poor atmospheric seeing
    - Saturation/defocus

    Arguments:
        paired_stars: [PairedStars] Paired stars object.

    Keyword arguments:
        fraction: [float] Fraction of stars to remove (0.10 = top 10% highest FWHM).
            Default is 0.10.
        verbose: [bool] Print filtering info. Default is False.

    Returns:
        new_paired_stars: [PairedStars] Filtered paired stars.
        removed_count: [int] Number of stars removed.
    """
    if len(paired_stars) < 10:
        return paired_stars, 0

    fwhm_list = []
    valid_indices = []

    for i, (x, y, fwhm, intens_acc, obj, snr, saturated) in enumerate(paired_stars.paired_stars):
        if hasattr(obj, 'pick_type') and obj.pick_type == "geopoint":
            continue

        if fwhm is not None and fwhm > 0:
            fwhm_list.append(fwhm)
            valid_indices.append(i)

    if len(fwhm_list) < 10:
        return paired_stars, 0

    fwhm_array = np.array(fwhm_list)

    # Calculate the FWHM threshold (remove top fraction)
    threshold_percentile = (1.0 - fraction) * 100
    fwhm_threshold = np.percentile(fwhm_array, threshold_percentile)

    # Find indices to remove
    high_fwhm_indices = set()
    for idx, fwhm_val in zip(valid_indices, fwhm_array):
        if fwhm_val > fwhm_threshold:
            high_fwhm_indices.add(idx)

    if len(high_fwhm_indices) > 0:
        new_paired_stars = PairedStars()
        for i, (x, y, fwhm, intens_acc, obj, snr, saturated) in enumerate(paired_stars.paired_stars):
            if i not in high_fwhm_indices:
                new_paired_stars.addPair(x, y, fwhm, intens_acc, obj, snr, saturated)

        if verbose:
            median_fwhm = np.median(fwhm_array)
            print("  Removed {:d} high-FWHM stars (FWHM > {:.2f}, median={:.2f})".format(
                len(high_fwhm_indices), fwhm_threshold, median_fwhm))

        return new_paired_stars, len(high_fwhm_indices)

    return paired_stars, 0
