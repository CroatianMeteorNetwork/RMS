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
from RMS.Astrometry.ApplyAstrometry import extinctionCorrectionTrueToApparent


# Default filtering parameters
DEFAULT_PHOTOMETRIC_SIGMA = 2.5
DEFAULT_BLEND_RADIUS_ARCSEC = 30.0
DEFAULT_BLEND_MAG_DIFF = 2.0
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


def filterBlendedStars(paired_stars, catalog_stars, blend_radius_arcsec=DEFAULT_BLEND_RADIUS_ARCSEC,
                       mag_diff_limit=DEFAULT_BLEND_MAG_DIFF, verbose=False):
    """
    Filter paired_stars by removing likely blended stars.

    A star is considered blended if there are other bright catalog stars
    within blend_radius_arcsec of the matched catalog star position.

    Arguments:
        paired_stars: [PairedStars] Paired stars object.
        catalog_stars: [ndarray] Full catalog stars array with columns [ra, dec, mag, ...].

    Keyword arguments:
        blend_radius_arcsec: [float] Radius in arcseconds to check for neighbors.
            Default is 30.0.
        mag_diff_limit: [float] Only consider neighbors within this mag of matched star.
            Default is 2.0.
        verbose: [bool] Print filtering info. Default is False.

    Returns:
        new_paired_stars: [PairedStars] Filtered paired stars.
        removed_count: [int] Number of stars removed.
    """
    if len(paired_stars) < 5 or catalog_stars is None:
        return paired_stars, 0

    catalog_ra = catalog_stars[:, 0]
    catalog_dec = catalog_stars[:, 1]
    catalog_mag = catalog_stars[:, 2]

    blend_radius_deg = blend_radius_arcsec / 3600.0

    blended_indices = set()

    for i, (x, y, fwhm, intens_acc, obj, snr, saturated) in enumerate(paired_stars.paired_stars):
        if hasattr(obj, 'pick_type') and obj.pick_type == "geopoint":
            continue

        ra, dec, mag = obj.coords()

        # Compute angular distances using small-angle approximation
        cos_dec = np.cos(np.radians(dec))
        d_ra = (catalog_ra - ra) * cos_dec
        d_dec = catalog_dec - dec
        angular_dist = np.sqrt(d_ra**2 + d_dec**2)

        # Find neighbors within radius (excluding self)
        neighbor_mask = (angular_dist < blend_radius_deg) & (angular_dist > 0.001)

        # Only count bright neighbors (within mag_diff_limit of matched star)
        bright_neighbor_mask = neighbor_mask & (catalog_mag < mag + mag_diff_limit)

        if np.sum(bright_neighbor_mask) > 0:
            blended_indices.add(i)

    if len(blended_indices) > 0:
        new_paired_stars = PairedStars()
        for i, (x, y, fwhm, intens_acc, obj, snr, saturated) in enumerate(paired_stars.paired_stars):
            if i not in blended_indices:
                new_paired_stars.addPair(x, y, fwhm, intens_acc, obj, snr, saturated)

        if verbose:
            print("  Removed {:d} blended stars (neighbors within {:.0f}\")".format(
                len(blended_indices), blend_radius_arcsec))

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
