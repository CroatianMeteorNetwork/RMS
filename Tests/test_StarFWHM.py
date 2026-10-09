"""
Tests of the FWHM of the stars: the star extractor gives the FWHM of a round star (2.355 sigma), and the CALSTARS
files keep it, including older files written with the FWHM larger by sqrt(2).
"""

import json
import math

import numpy as np
import pytest

from RMS.ExtractStars import extractStars
from RMS.Formats import CALSTARS
from Utils.Flux import getSensorCharacterization


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
    """ A CALSTARS file written now has the definition of the FWHM in its header, so its FWHM is read as it was
        written.
    """

    # One star: (y, x, intensity, amplitude, fwhm, background, snr, saturated pixels)
    stars = [['FF_XX0001_20260101_000000_000_0000000.fits', [(10.0, 20.0, 1000, 100, 2.5, 30, 10.0, 0)]]]
    CALSTARS.writeCALSTARS(stars, str(tmp_path), 'CALSTARS_test.txt', 'XX0001', 100, 100)

    star_list, _ = CALSTARS.readCALSTARS(str(tmp_path), 'CALSTARS_test.txt')
    assert star_list[0][1][0][4] == pytest.approx(2.5)


def test_older_calstars_fwhm_is_converted(tmp_path):
    """ A file without the line of the definition of the FWHM has the FWHM larger by sqrt(2), and a FWHM of -1 means
        it was not measured.
    """

    # A CALSTARS file as written before the definition of the FWHM was added to the header, with two stars
    lines = ["==========================================================================",
             "RMS star extractor",
             "Cal time = FF header time plus Nframes/(2*FPS) seconds",
             "      Y       X IntensSum Ampltd  FWHM  BgLvl   SNR NSatPx",
             "==========================================================================",
             "FF folder = /tmp",
             "Cam #   = XX0001",
             "Nrows   = 100",
             "Ncols   = 100",
             "Nframes = 256",
             "Nstars  = -1",
             "==========================================================================",
             "FF_XX0001_20260101_000000_000_0000000.fits",
             "Star area dim = -1",
             "Integ pixels  = -1",
             "  10.00   20.00      1000    100  2.83     30 10.00      0",
             "  11.00   21.00      1000    100 -1.00     30 10.00      0",
             "##########################################################################"]
    (tmp_path/'CALSTARS_old.txt').write_text('\n'.join(lines) + '\n')

    # The measured FWHM is divided by sqrt(2), the -1 is kept
    star_list, _ = CALSTARS.readCALSTARS(str(tmp_path), 'CALSTARS_old.txt')
    stars = star_list[0][1]
    assert stars[0][4] == pytest.approx(2.83/math.sqrt(2))
    assert stars[1][4] == -1.0


def test_flux_sensor_cache_of_older_definition_is_converted(tmp_path):
    """ The FWHM of a sensor characterization cache of the flux saved before the definition of the FWHM was stored
        is converted like the FWHM of older CALSTARS files, and a cache of the new definition is read as is.
    """

    ff_name = 'FF_XX0001_20260101_000000_000_0000000.fits'

    # A cache of the older definition (no definition stored)
    path = tmp_path/'flux_sensor_characterization.json'
    path.write_text(json.dumps({ff_name: [2.83], '-1': {"FF file name": ['median star FWHM']}}))
    sensor_data = getSensorCharacterization(str(tmp_path), None, None, None)
    assert sensor_data[ff_name][0] == pytest.approx(2.83/math.sqrt(2))

    # A cache of the new definition
    path.write_text(json.dumps({ff_name: [2.0], 'fwhm_definition': CALSTARS.FWHM_DEFINITION}))
    sensor_data = getSensorCharacterization(str(tmp_path), None, None, None)
    assert sensor_data[ff_name][0] == pytest.approx(2.0)
