"""
Tests of the FWHM of the stars: the star extractor gives the FWHM of a round star (2.355 sigma), and the CALSTARS
files keep it.
"""

import numpy as np
import pytest

from RMS.ExtractStars import extractStars
from RMS.Formats import CALSTARS


def _starImage(sigma, size=200, n=5, peak=150.0, sky=20.0, seed=1):
    """ An 8-bit image with round Gaussian stars of the given sigma on a noisy sky. """

    # The sky with a noise of 1 ADU, and the stars on a diagonal, at subpixel positions
    rng = np.random.default_rng(seed)
    img = sky + rng.normal(0, 1.0, (size, size))
    yy, xx = np.mgrid[0:size, 0:size]
    for k in range(n):
        x, y = 40 + 30*k + 0.3, 60 + 25*k + 0.6
        img += peak*np.exp(-0.5*((xx - x)**2 + (yy - y)**2)/sigma**2)

    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.mark.parametrize('sigma', [1.2, 2.0])
def test_extracted_fwhm_is_the_fwhm_of_a_round_star(sigma):
    """ The FWHM given by the star extractor for round stars is 2.355 sigma (and not sqrt(2) times more, as
        with the sum of the variances of the two axes).
    """

    x, y, amplitude, intensity, fwhm, background, snr, n_sat = extractStars(_starImage(sigma), segment_radius=8)

    # Most stars are found, and their median FWHM is within 10% of the true one
    assert len(fwhm) >= 3
    assert np.median(fwhm) == pytest.approx(2.355*sigma, rel=0.1)


def test_calstars_fwhm_round_trip(tmp_path):
    """ The FWHM written to a CALSTARS file is read back unchanged. """

    # One star: (y, x, intensity, amplitude, fwhm, background, snr, saturated pixels)
    stars = [['FF_XX0001_20260101_000000_000_0000000.fits', [(10.0, 20.0, 1000, 100, 2.5, 30, 10.0, 0)]]]
    CALSTARS.writeCALSTARS(stars, str(tmp_path), 'CALSTARS_test.txt', 'XX0001', 100, 100)

    star_list, _ = CALSTARS.readCALSTARS(str(tmp_path), 'CALSTARS_test.txt')
    assert star_list[0][1][0][4] == pytest.approx(2.5)
