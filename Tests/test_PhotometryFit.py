"""
Tests of the photometric offset fit (ApplyAstrometry.photometryFit and ApplyAstrometry.photometryFitRobust).

The fit is done on the logarithm of the star intensities. A star with an intensity <= 0 (e.g. an integer IntensSum
of 0 in CALSTARS) or with a non-finite value cannot be fitted, and a single one of them used to make the cost of the
fit non-finite, so the fit returned its initial guess (mag_lev = 10.0) with a standard deviation of 0. The tests use
synthetic stars whose intensities follow mag = -2.5*log10(intensity) + 11.3 with a 0.1 mag scatter.
"""

import numpy as np
import pytest

from RMS.Astrometry.ApplyAstrometry import photometryFit, photometryFitRobust


# Photometric offset of the synthetic stars
MAG_LEV = 11.3


def photometryData(n_stars=30, seed=0):
    """ Synthetic stars with intensities which follow the photometric offset MAG_LEV with a 0.1 mag scatter.

    Keyword arguments:
        n_stars: [int] Number of stars. 30 by default.
        seed: [int] Seed of the random numbers. 0 by default.

    Return:
        (intensities, radii, catalog_mags): [tuple of ndarrays] Star intensities, radii from the image centre
            (px) and catalog magnitudes.
    """

    # Random numbers of the synthetic stars
    rng = np.random.default_rng(seed)

    # Catalog magnitudes, and the intensities which go with them with some scatter
    catalog_mags = rng.uniform(1.0, 5.0, n_stars)
    intensities = 10**((MAG_LEV - catalog_mags + rng.normal(0, 0.1, n_stars))/2.5)

    # Positions in the image
    radii = rng.uniform(0, 500, n_stars)

    return intensities, radii, catalog_mags


@pytest.mark.parametrize("bad_intensity", [0.0, -5.0, np.nan])
def test_fit_leaves_out_unusable_intensities(bad_intensity):
    """ One star with an intensity <= 0 or NaN must not turn the fit into its initial guess (mag_lev = 10.0). The
        residuals still have one entry per star, not finite for the star which was left out. """

    # One of the stars cannot be fitted
    intensities, radii, catalog_mags = photometryData()
    intensities[4] = bad_intensity

    photom_params, fit_stddev, fit_resids = photometryFit(intensities, radii, catalog_mags, fixed_vignetting=0.0)

    # The offset of the other stars, with their scatter
    assert abs(photom_params[0] - MAG_LEV) < 0.1
    assert np.isfinite(fit_stddev) and (0 < fit_stddev < 0.3)

    # One residual per star, the unusable star is not fitted
    assert len(fit_resids) == len(intensities)
    assert not np.isfinite(fit_resids[4])
    assert np.all(np.isfinite(np.delete(fit_resids, 4)))


def test_robust_fit_with_a_zero_intensity_star():
    """ The robust fit gives the real offset and a non-zero standard deviation with a zero-intensity star, and the
        star is not among the returned stars. """

    # One of the stars has a zero intensity
    intensities, radii, catalog_mags = photometryData()
    intensities[0] = 0.0

    photom_params, fit_stddev, _, px_intens, _, _ = photometryFitRobust(intensities, radii, catalog_mags,
                                                                       fixed_vignetting=0.0)

    # The offset of the other stars with their scatter, without the zero-intensity star
    assert abs(photom_params[0] - MAG_LEV) < 0.1
    assert 0 < fit_stddev < 0.3
    assert np.all(px_intens > 0)


def test_robust_fit_of_a_few_stars_returns_only_fitted_stars():
    """ With few stars the outlier rejection stops before it removes anything, as it would leave fewer than
        MIN_PHOTOM_FIT_STARS stars. The returned stars must still be only the ones which could be fitted, as the
        callers plot the logarithm of their intensities. """

    # Four stars, one of them with a zero intensity
    intensities, radii, catalog_mags = photometryData(n_stars=4)
    intensities[0] = 0.0

    # Star 3 is 1.5 mag too bright, so the rejection of the residuals larger than 1 mag would leave two stars
    intensities[3] *= 10**(1.5/2.5)

    photom_params, fit_stddev, fit_resid, px_intens, radius, mags = photometryFitRobust(
        intensities, radii, catalog_mags, fixed_vignetting=0.0)

    # The fit of the three stars which can be fitted
    assert abs(photom_params[0] - MAG_LEV) < 0.2
    assert np.isfinite(fit_stddev) and (fit_stddev > 0)

    # Only those three stars are returned, with finite residuals
    assert np.array_equal(px_intens, intensities[1:])
    assert np.array_equal(radius, radii[1:])
    assert np.array_equal(mags, catalog_mags[1:])
    assert np.all(np.isfinite(fit_resid))


def test_excluded_stars_keep_their_residuals():
    """ Guard (the same before and after the change): a star marked in exclude_list is left out of the fit, but its
        residual is still computed, as SkyFit2 shows the residuals of the excluded stars. """

    # The synthetic stars
    intensities, radii, catalog_mags = photometryData()

    # Star 3 is 2 mag too bright and excluded from the fit
    intensities[3] *= 10**(2.0/2.5)
    exclude_list = np.zeros(len(intensities), dtype=int)
    exclude_list[3] = 1

    photom_params, fit_stddev, fit_resids = photometryFit(intensities, radii, catalog_mags, fixed_vignetting=0.0,
                                                          exclude_list=exclude_list)

    # The fit of the other stars, and the 2 mag residual of the excluded star
    assert abs(photom_params[0] - MAG_LEV) < 0.1
    assert 0 < fit_stddev < 0.3
    assert np.all(np.isfinite(fit_resids))
    assert abs(fit_resids[3] - 2.0) < 0.4
