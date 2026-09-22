""" Tests for the photometric offset fit in RMS.Astrometry.ApplyAstrometry. """

from __future__ import print_function, division, absolute_import

import numpy as np
import pytest

from RMS.Astrometry.ApplyAstrometry import photometryFit, photometryFitRobust


def _photometryData(n_stars=30, mag_lev=11.3, seed=0):
    """ Synthetic star intensities which follow mag = -2.5*log10(intensity) + mag_lev with some noise. """

    rng = np.random.default_rng(seed)
    catalog_mags = rng.uniform(1.0, 5.0, n_stars)
    intensities = 10**((mag_lev - catalog_mags + rng.normal(0, 0.1, n_stars))/2.5)
    radii = rng.uniform(0, 500, n_stars)

    return intensities, radii, catalog_mags


@pytest.mark.parametrize("bad_intensity", [0.0, -5.0, np.nan])
def testPhotometryFitIgnoresUnusableIntensities(bad_intensity):
    """ One star with an intensity <= 0 or NaN must not turn the fit into the initial guess. """

    intensities, radii, catalog_mags = _photometryData()
    intensities[4] = bad_intensity

    photom_params, fit_stddev, fit_resids = photometryFit(intensities, radii, catalog_mags,
        fixed_vignetting=0.0)

    assert abs(photom_params[0] - 11.3) < 0.1
    assert np.isfinite(fit_stddev) and (0 < fit_stddev < 0.3)
    assert len(fit_resids) == len(intensities)
    assert not np.isfinite(fit_resids[4])


def testPhotometryFitRobustIgnoresZeroIntensity():
    """ The robust fit must give the real offset and a non-zero stddev with a zero-intensity star. """

    intensities, radii, catalog_mags = _photometryData()
    intensities[0] = 0.0

    photom_params, fit_stddev, _, px_intens, _, _ = photometryFitRobust(intensities, radii, catalog_mags,
        fixed_vignetting=0.0)

    assert abs(photom_params[0] - 11.3) < 0.1
    assert 0 < fit_stddev < 0.3
    assert np.all(px_intens > 0)
