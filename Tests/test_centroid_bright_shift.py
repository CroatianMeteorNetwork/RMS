""" The brightness-dependent centroid shift (config centroid_bright_shift*): the correction of the image coordinates
    by the magnitude or by the intensity, its use in the astrometry of detections, and the config options. Run with:
    python -m pytest Tests/test_centroid_bright_shift.py """

from __future__ import print_function, division, absolute_import

import os
import shutil
import tempfile
import unittest

import numpy as np

import RMS.ConfigReader as cr
from RMS.Astrometry.ApplyAstrometry import correctBrightShift, applyPlateparToCentroids, brightShiftOptions
from RMS.Formats.Platepar import Platepar
from RMS.Misc import getRmsRootDir


TEMPLATE = os.path.join(getRmsRootDir(), 'share', 'platepar_templates', 'template_generic_720p_4mm.cal')
FF_NAME = 'FF_XX0001_20260313_021955_123_000000.fits'

# Coefficients of the order of the ones measured on a real camera: about 1.6 px along x at magnitude 3
SHIFT = [0.0053, 0.0393, -0.0047, 0.0002]


def detectionRows(mag_level):
    """ Rows [calib_status, frame_n, x, y, ra, dec, azim, elev, inten, mag, background, snr, saturated] of a
        detection crossing the image, with the given intensity """

    n = 20
    frames = np.arange(n, dtype=np.float64)
    x = 300.0 + 10.0*frames
    y = 200.0 + 5.0*frames
    rows = np.zeros((n, 13))
    rows[:, 0] = 1
    rows[:, 1] = frames
    rows[:, 2] = x
    rows[:, 3] = y
    rows[:, 8] = mag_level
    rows[:, 10] = 100.0
    rows[:, 11] = 50.0
    return rows


def separationArcsec(ra1, dec1, ra2, dec2):
    """ Angle between directions given in degrees (arcsec) """

    # The chord between the unit vectors (precise at small angles, unlike the arc cosine)
    ra1, dec1, ra2, dec2 = [np.radians(np.asarray(v, dtype=np.float64)) for v in (ra1, dec1, ra2, dec2)]
    u1 = np.column_stack([np.cos(dec1)*np.cos(ra1), np.cos(dec1)*np.sin(ra1), np.sin(dec1)])
    u2 = np.column_stack([np.cos(dec2)*np.cos(ra2), np.cos(dec2)*np.sin(ra2), np.sin(dec2)])
    return np.degrees(2*np.arcsin(np.linalg.norm(u1 - u2, axis=1)/2))*3600


class TestCorrectBrightShift(unittest.TestCase):

    def test_model(self):
        # The shift A*F + C*F^2 along each axis, F the brightness relative to magnitude 5, is subtracted
        x, y = np.array([100.0, 100.0, 100.0]), np.array([50.0, 50.0, 50.0])
        mags = np.array([5.0, 4.0, 10.0])
        xc, yc = correctBrightShift(x, y, mags, SHIFT, mag_limit=3.0)
        f = 10**(-0.4*(mags - 5.0))
        np.testing.assert_allclose(xc, x - (SHIFT[0]*f + SHIFT[1]*f**2))
        np.testing.assert_allclose(yc, y - (SHIFT[2]*f + SHIFT[3]*f**2))

        # Faint sources are not moved measurably
        self.assertLess(abs(xc[2] - x[2]), 1e-3)

    def test_magnitude_limit_and_missing_magnitudes(self):
        # Brighter than the limit: the shift at the limit; no magnitude: no shift
        x, y = np.full(3, 100.0), np.full(3, 50.0)
        xc, yc = correctBrightShift(x, y, np.array([-3.0, 3.0, np.nan]), SHIFT, mag_limit=3.0)
        self.assertAlmostEqual(xc[0], xc[1])
        self.assertAlmostEqual(yc[0], yc[1])
        self.assertEqual(xc[2], 100.0)
        self.assertEqual(yc[2], 50.0)

    def test_zero_coefficients(self):
        x, y = np.array([10.0, 20.0]), np.array([30.0, 40.0])
        xc, yc = correctBrightShift(x, y, np.array([0.0, 2.0]), [0, 0, 0, 0])
        np.testing.assert_array_equal(xc, x)
        np.testing.assert_array_equal(yc, y)


class TestIntensityBasis(unittest.TestCase):

    def test_model(self):
        # A*u + C*v^2, u and v the intensity above the knee of each term (0 below it), in units of 1e5 ADU
        x, y = np.full(5, 100.0), np.full(5, 50.0)
        inten = np.array([5e4, 9e4, 1e5, 1.5e5, 2e5])
        shift = [0.05, 0.4, 0.01, 0.02]
        xc, yc = correctBrightShift(x, y, None, shift, intensities=inten, basis='intensity', knees=(8e4, 1e5),
                                    intensity_limit=2.5e5)
        u = np.maximum(inten - 8e4, 0)/1e5
        v = np.maximum(inten - 1e5, 0)/1e5
        np.testing.assert_allclose(xc, x - (0.05*u + 0.4*v**2))
        np.testing.assert_allclose(yc, y - (0.01*u + 0.02*v**2))

        # Below both knees not moved; between them only the linear term
        self.assertEqual(xc[0], 100.0)
        self.assertAlmostEqual(x[1] - xc[1], 0.05*0.1)

        # Knees at 0: the terms grow from the faintest levels
        xc0, _ = correctBrightShift(x, y, None, shift, intensities=inten, basis='intensity')
        np.testing.assert_allclose(xc0, x - (0.05*inten/1e5 + 0.4*(inten/1e5)**2))

    def test_limit_and_missing_intensities(self):
        # Above the limit: the shift at the limit; no finite positive intensity: no shift
        x, y = np.full(5, 100.0), np.full(5, 50.0)
        xc, yc = correctBrightShift(x, y, None, [0.05, 0.4, 0.0, 0.0], intensities=[3e5, 2e5, np.nan, 0.0, -5.0],
                                    basis='intensity', knees=(0.0, 1e5), intensity_limit=2e5)
        self.assertAlmostEqual(xc[0], xc[1])
        np.testing.assert_array_equal(xc[2:], 100.0)
        np.testing.assert_array_equal(yc, 50.0)

    def test_magnitudes_not_used(self):
        # The intensity basis doesn't depend on the magnitudes
        x, y = np.full(2, 100.0), np.full(2, 50.0)
        a = correctBrightShift(x, y, np.array([3.0, 9.0]), SHIFT, intensities=[2e5, 2e5], basis='intensity')
        b = correctBrightShift(x, y, None, SHIFT, intensities=[2e5, 2e5], basis='intensity')
        np.testing.assert_array_equal(a[0], b[0])
        self.assertAlmostEqual(a[0][0], a[0][1])

    def test_unknown_basis(self):
        with self.assertRaises(ValueError):
            correctBrightShift([1.0], [1.0], [5.0], SHIFT, basis='peak')


class TestApplyPlateparToCentroids(unittest.TestCase):

    def setUp(self):
        self.pp = Platepar()
        self.pp.read(TEMPLATE)

        # A level giving a bright detection (about magnitude 3) with the template's photometric zero point
        self.level = 10**(-0.4*(3.0 - self.pp.mag_lev))

    def test_without_correction(self):
        # No coefficients or zero coefficients: the astrometry as before
        rows = detectionRows(self.level)
        ref = applyPlateparToCentroids(FF_NAME, 25.0, rows, self.pp)
        for shift in (None, [0, 0, 0, 0]):
            out = applyPlateparToCentroids(FF_NAME, 25.0, rows, self.pp, bright_shift=shift)
            np.testing.assert_array_equal(out, ref)

    def test_correction(self):
        rows = detectionRows(self.level)
        plain = applyPlateparToCentroids(FF_NAME, 25.0, rows, self.pp)
        out = applyPlateparToCentroids(FF_NAME, 25.0, rows, self.pp, bright_shift=SHIFT, bright_shift_mag_limit=3.0)

        # The output keeps the measured image coordinates and the magnitudes
        np.testing.assert_array_equal(out[:, 1:3], plain[:, 1:3])
        np.testing.assert_array_equal(out[:, 8], plain[:, 8])

        # The sky positions are the ones of the corrected image coordinates
        xc, yc = correctBrightShift(rows[:, 2], rows[:, 3], plain[:, 8], SHIFT, mag_limit=3.0)
        moved = rows.copy()
        moved[:, 2], moved[:, 3] = xc, yc
        ref = applyPlateparToCentroids(FF_NAME, 25.0, moved, self.pp)
        self.assertLess(np.max(separationArcsec(out[:, 3], out[:, 4], ref[:, 3], ref[:, 4])), 1e-6)

        # A bright detection is moved by about the modelled shift (the plate scale of the wide-field template
        #   changes across the image)
        sep = separationArcsec(out[:, 3], out[:, 4], plain[:, 3], plain[:, 4])
        px_scale = 3600/self.pp.F_scale
        ratio = sep/(np.hypot(rows[:, 2] - xc, rows[:, 3] - yc)*px_scale)
        self.assertTrue(np.all((ratio > 0.5) & (ratio < 2.0)), ratio)


    def test_intensity_basis(self):
        # The intensity basis takes the intensity of every centroid (its level), the sky positions are the ones of
        #   the corrected coordinates, and the output keeps the measured coordinates
        rows = detectionRows(self.level)
        rows[:, 8] = np.linspace(5e4, 2.5e5, len(rows))
        shift = [0.05, 0.4, 0.0, 0.0]
        opts = dict(bright_shift=shift, bright_shift_basis='intensity', bright_shift_knees=(8e4, 1e5),
                    bright_shift_intensity_limit=2.5e5)
        plain = applyPlateparToCentroids(FF_NAME, 25.0, rows, self.pp)
        out = applyPlateparToCentroids(FF_NAME, 25.0, rows, self.pp, **opts)
        np.testing.assert_array_equal(out[:, 1:3], plain[:, 1:3])
        xc, yc = correctBrightShift(rows[:, 2], rows[:, 3], None, shift, intensities=rows[:, 8], basis='intensity',
                                    knees=(8e4, 1e5), intensity_limit=2.5e5)
        moved = rows.copy()
        moved[:, 2], moved[:, 3] = xc, yc
        ref = applyPlateparToCentroids(FF_NAME, 25.0, moved, self.pp)
        self.assertLess(np.max(separationArcsec(out[:, 3], out[:, 4], ref[:, 3], ref[:, 4])), 1e-6)

        # The faintest centroid (below both knees) doesn't move, the brightest does
        sep = separationArcsec(out[:, 3], out[:, 4], plain[:, 3], plain[:, 4])
        self.assertLess(sep[0], 1e-6)
        self.assertGreater(sep[-1], 1.0)


class TestConfigOption(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.base = open(os.path.join(getRmsRootDir(), '.config')).read()

    def tearDown(self):
        shutil.rmtree(self.dir)

    def parse(self, text):
        path = os.path.join(self.dir, 'test.config')
        with open(path, 'w') as f:
            f.write(text)
        return cr.parse(path)

    def test_default_is_no_correction(self):
        config = self.parse(self.base)
        self.assertEqual(config.centroid_bright_shift, [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(config.centroid_bright_shift_mag_limit, 3.0)

    def test_values(self):
        text = self.base.replace('centroid_bright_shift: 0, 0, 0, 0',
                                 'centroid_bright_shift: 0.0053, 0.0393, -0.0047, 0.0002')
        text = text.replace('centroid_bright_shift_mag_limit: 3.0', 'centroid_bright_shift_mag_limit: 2.5')
        config = self.parse(text)
        self.assertEqual(config.centroid_bright_shift, SHIFT)
        self.assertEqual(config.centroid_bright_shift_mag_limit, 2.5)

    def test_intensity_options(self):
        text = self.base.replace('centroid_bright_shift_basis: magnitude', 'centroid_bright_shift_basis: Intensity')
        text = text.replace('centroid_bright_shift_knees: 0, 0', 'centroid_bright_shift_knees: 80000, 100000')
        text = text.replace('centroid_bright_shift_intensity_limit: 250000', 'centroid_bright_shift_intensity_limit: 210000')
        config = self.parse(text)
        self.assertEqual(config.centroid_bright_shift_basis, 'intensity')
        self.assertEqual(config.centroid_bright_shift_knees, [8e4, 1e5])
        self.assertEqual(config.centroid_bright_shift_intensity_limit, 2.1e5)

        # The arguments of the astrometry from the config
        opts = brightShiftOptions(config)
        self.assertEqual(opts['bright_shift_basis'], 'intensity')
        self.assertEqual(opts['bright_shift_knees'], [8e4, 1e5])
        self.assertEqual(opts['bright_shift_intensity_limit'], 2.1e5)
        self.assertEqual(opts['bright_shift'], config.centroid_bright_shift)

    def test_default_basis_and_unknown_basis(self):
        config = self.parse(self.base)
        self.assertEqual(config.centroid_bright_shift_basis, 'magnitude')
        self.assertEqual(config.centroid_bright_shift_knees, [0.0, 0.0])
        self.assertEqual(config.centroid_bright_shift_intensity_limit, 2.5e5)
        config = self.parse(self.base.replace('centroid_bright_shift_basis: magnitude', 'centroid_bright_shift_basis: peak')
                            .replace('centroid_bright_shift_knees: 0, 0', 'centroid_bright_shift_knees: 80000'))
        self.assertEqual(config.centroid_bright_shift_basis, 'magnitude')
        self.assertEqual(config.centroid_bright_shift_knees, [0.0, 0.0])

    def test_wrong_count_is_ignored(self):
        config = self.parse(self.base.replace('centroid_bright_shift: 0, 0, 0, 0', 'centroid_bright_shift: 0.1, 0.2'))
        self.assertEqual(config.centroid_bright_shift, [0.0, 0.0, 0.0, 0.0])


if __name__ == '__main__':
    unittest.main()
