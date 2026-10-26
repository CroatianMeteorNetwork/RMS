"""Tests for polynomial vignetting, saturation, and chromatic vignetting corrections.

Tests the production functions from RMS.Astrometry.ApplyAstrometry and RMS.Formats.Platepar.
Falls back to local implementations if the full RMS dependency chain is unavailable (no cv2).
"""

import numpy as np
import unittest
import sys
import os
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Mock pyximport if unavailable (Cython not always installed in test env)
if 'pyximport' not in sys.modules:
    sys.modules['pyximport'] = types.ModuleType('pyximport')
    sys.modules['pyximport'].install = lambda **kw: None

# Try importing real production code; fall back to local copies only if the deep
# dependency chain (cv2, etc.) is unavailable.
_USING_REAL_CODE = False
try:
    from RMS.Astrometry.ApplyAstrometry import (
        correctVignetting, photomLine, calculateMagnitudes, photometryFit
    )
    from RMS.Formats.Platepar import Platepar
    _USING_REAL_CODE = True
except ImportError:
    # Minimal local copies matching the production code for environments without cv2.
    # These MUST be kept in sync with ApplyAstrometry.py -- if these diverge, CI on
    # the full RMS environment will catch it (the real import path works there).
    def correctVignetting(px_sum, radius, vignetting_coeff):
        if vignetting_coeff is None:
            vignetting_coeff = 0.0
        return px_sum / (np.cos(vignetting_coeff * radius) ** 4)

    def photomLine(input_params, photom_offset, vignetting_coeff, vignetting_poly=None,
                   saturation_params=None, chromatic_params=None):
        px_sum, radius = input_params
        lsp = np.log10(correctVignetting(px_sum, radius, vignetting_coeff))
        mag = -2.5 * lsp + photom_offset
        if vignetting_poly is not None:
            a2, a4, half_diag = vignetting_poly
            if half_diag > 0 and (abs(a2) > 0 or abs(a4) > 0):
                rn = np.asarray(radius, dtype=np.float64) / half_diag
                mag = mag - (a2 * rn ** 2 + a4 * rn ** 4)
        if saturation_params is not None:
            slope, mag_break, catalog_mags = saturation_params
            catalog_mags = np.asarray(catalog_mags, dtype=np.float64)
            sat_corr = np.where(catalog_mags < mag_break, slope * (catalog_mags - mag_break), 0.0)
            mag = mag - sat_corr
        if chromatic_params is not None:
            ct0, ct_r2, ct_r4, pivot, half_diag, colors = chromatic_params
            colors = np.asarray(colors, dtype=np.float64)
            c = np.where(np.isfinite(colors), colors - pivot, 0.0)
            rn = np.asarray(radius, dtype=np.float64) / half_diag if half_diag > 0 else 0.0
            chrom_corr = (ct0 + ct_r2 * rn ** 2 + ct_r4 * rn ** 4) * c
            mag = mag - chrom_corr
        return mag

    def calculateMagnitudes(px_sum_arr, radius_arr, photom_offset, vignetting_coeff,
                            vignetting_poly=None):
        magnitude_data = np.zeros_like(px_sum_arr, dtype=np.float64)
        for i, (px_sum, radius) in enumerate(zip(px_sum_arr, radius_arr)):
            if px_sum is None:
                px_sum = 1
            px_sum_corr = correctVignetting(px_sum, radius, vignetting_coeff)
            magnitude_data[i] = -2.5 * np.log10(px_sum_corr) + photom_offset
        if vignetting_poly is not None:
            a2, a4, half_diag = vignetting_poly
            if half_diag > 0 and (abs(a2) > 0 or abs(a4) > 0):
                rn = np.asarray(radius_arr, dtype=np.float64) / half_diag
                magnitude_data = magnitude_data - (a2 * rn ** 2 + a4 * rn ** 4)
        return magnitude_data

    Platepar = None


class TestBackwardCompatibility(unittest.TestCase):
    """New parameters default to no correction -- old behavior preserved."""

    def setUp(self):
        self.px = np.array([1000.0, 500.0, 200.0, 50.0])
        self.r = np.array([100.0, 300.0, 500.0, 600.0])
        self.offset = 10.0
        self.vig = 0.001

    def test_no_extra_params_matches_original(self):
        old = photomLine((self.px, self.r), self.offset, self.vig)
        new = photomLine((self.px, self.r), self.offset, self.vig,
                         vignetting_poly=None, saturation_params=None, chromatic_params=None)
        np.testing.assert_array_almost_equal(old, new)

    def test_zero_poly_no_change(self):
        old = photomLine((self.px, self.r), self.offset, self.vig)
        new = photomLine((self.px, self.r), self.offset, self.vig,
                         vignetting_poly=(0.0, 0.0, 640.0))
        np.testing.assert_array_almost_equal(old, new)

    def test_zero_saturation_no_change(self):
        old = photomLine((self.px, self.r), self.offset, self.vig)
        cat_mags = np.array([3.0, 4.0, 6.0, 7.0])
        new = photomLine((self.px, self.r), self.offset, self.vig,
                         saturation_params=(0.0, 5.0, cat_mags))
        np.testing.assert_array_almost_equal(old, new)

    def test_zero_chromatic_no_change(self):
        old = photomLine((self.px, self.r), self.offset, self.vig)
        colors = np.array([0.5, 1.5, 2.5, 3.5])
        new = photomLine((self.px, self.r), self.offset, self.vig,
                         chromatic_params=(0.0, 0.0, 0.0, 1.5, 640.0, colors))
        np.testing.assert_array_almost_equal(old, new)


class TestPolynomialVignetting(unittest.TestCase):

    def test_positive_a2_brightens_edge(self):
        px = np.array([1000.0, 1000.0])
        r = np.array([0.0, 500.0])
        mag_no = photomLine((px, r), 10.0, 0.001)
        mag_poly = photomLine((px, r), 10.0, 0.001, vignetting_poly=(0.5, 0.0, 640.0))
        self.assertAlmostEqual(mag_poly[0], mag_no[0], places=5)
        self.assertLess(mag_poly[1], mag_no[1])

    def test_symmetric_at_center(self):
        px = np.array([1000.0])
        r = np.array([0.0])
        mag_no = photomLine((px, r), 10.0, 0.001)
        mag_poly = photomLine((px, r), 10.0, 0.001, vignetting_poly=(1.0, 1.0, 640.0))
        self.assertAlmostEqual(mag_poly[0], mag_no[0], places=5)

    def test_scales_with_radius(self):
        px = np.ones(3) * 1000.0
        r = np.array([200.0, 400.0, 600.0])
        mag = photomLine((px, r), 10.0, 0.001, vignetting_poly=(1.0, 0.0, 640.0))
        diffs = np.diff(mag)
        self.assertTrue(np.all(diffs < 0))

    def test_calculateMagnitudes_applies_poly(self):
        px = np.array([1000.0, 1000.0])
        r = np.array([0.0, 500.0])
        mag_no = calculateMagnitudes(px, r, 10.0, 0.001)
        mag_poly = calculateMagnitudes(px, r, 10.0, 0.001, vignetting_poly=(0.5, 0.0, 640.0))
        self.assertAlmostEqual(mag_poly[0], mag_no[0], places=5)
        self.assertLess(mag_poly[1], mag_no[1])


class TestSaturationCorrection(unittest.TestCase):

    def test_only_affects_bright_stars(self):
        px = np.array([10000.0, 1000.0, 100.0])
        r = np.array([200.0, 200.0, 200.0])
        cat = np.array([2.0, 4.0, 7.0])
        mag_no = photomLine((px, r), 10.0, 0.001)
        mag_sat = photomLine((px, r), 10.0, 0.001, saturation_params=(-0.1, 5.0, cat))
        self.assertAlmostEqual(mag_sat[2], mag_no[2], places=5)
        self.assertNotAlmostEqual(mag_sat[0], mag_no[0], places=3)

    def test_negative_slope_brightens_bright_stars(self):
        px = np.array([10000.0])
        r = np.array([200.0])
        cat = np.array([2.0])
        mag_no = photomLine((px, r), 10.0, 0.001)
        mag_sat = photomLine((px, r), 10.0, 0.001, saturation_params=(-0.1, 5.0, cat))
        self.assertLess(mag_sat[0], mag_no[0])

    def test_at_break_point_no_correction(self):
        px = np.array([500.0])
        r = np.array([200.0])
        cat = np.array([5.0])
        mag_no = photomLine((px, r), 10.0, 0.001)
        mag_sat = photomLine((px, r), 10.0, 0.001, saturation_params=(-0.1, 5.0, cat))
        self.assertAlmostEqual(mag_sat[0], mag_no[0], places=5)

    def test_custom_mag_break(self):
        px = np.array([5000.0])
        r = np.array([200.0])
        cat = np.array([4.5])
        mag_break3 = photomLine((px, r), 10.0, 0.001, saturation_params=(-0.1, 3.0, cat))
        mag_break6 = photomLine((px, r), 10.0, 0.001, saturation_params=(-0.1, 6.0, cat))
        mag_no = photomLine((px, r), 10.0, 0.001)
        # mag 4.5 < break 6.0 -> corrected; mag 4.5 > break 3.0 -> not corrected
        self.assertAlmostEqual(mag_break3[0], mag_no[0], places=5)
        self.assertNotAlmostEqual(mag_break6[0], mag_no[0], places=3)


class TestChromaticVignetting(unittest.TestCase):

    def test_no_correction_at_pivot_color(self):
        px = np.array([1000.0, 1000.0])
        r = np.array([0.0, 500.0])
        colors = np.array([1.5, 1.5])
        mag_no = photomLine((px, r), 10.0, 0.001)
        mag_chrom = photomLine((px, r), 10.0, 0.001,
                               chromatic_params=(0.1, 0.2, 0.0, 1.5, 640.0, colors))
        np.testing.assert_array_almost_equal(mag_chrom, mag_no, decimal=5)

    def test_color_dependent(self):
        px = np.array([1000.0, 1000.0])
        r = np.array([300.0, 300.0])
        colors_blue = np.array([0.5, 0.5])
        colors_red = np.array([3.0, 3.0])
        mag_blue = photomLine((px, r), 10.0, 0.001,
                               chromatic_params=(0.1, 0.0, 0.0, 1.5, 640.0, colors_blue))
        mag_red = photomLine((px, r), 10.0, 0.001,
                              chromatic_params=(0.1, 0.0, 0.0, 1.5, 640.0, colors_red))
        self.assertNotAlmostEqual(mag_blue[0], mag_red[0], places=3)

    def test_position_dependent(self):
        px = np.ones(2) * 1000.0
        r = np.array([0.0, 500.0])
        colors = np.array([3.0, 3.0])
        mag = photomLine((px, r), 10.0, 0.001,
                          chromatic_params=(0.0, 0.2, 0.0, 1.5, 640.0, colors))
        self.assertNotAlmostEqual(mag[0], mag[1], places=3)

    def test_nan_color_treated_as_pivot(self):
        px = np.array([1000.0])
        r = np.array([300.0])
        colors = np.array([np.nan])
        mag_no = photomLine((px, r), 10.0, 0.001)
        mag_chrom = photomLine((px, r), 10.0, 0.001,
                               chromatic_params=(0.1, 0.2, 0.0, 1.5, 640.0, colors))
        self.assertAlmostEqual(mag_chrom[0], mag_no[0], places=5)


class TestCombinedCorrections(unittest.TestCase):

    def test_all_corrections_stack(self):
        px = np.array([10000.0, 1000.0])
        r = np.array([500.0, 100.0])
        cat = np.array([2.0, 6.0])
        colors = np.array([0.5, 3.0])
        mag_none = photomLine((px, r), 10.0, 0.001)
        mag_all = photomLine((px, r), 10.0, 0.001,
                              vignetting_poly=(0.3, 0.1, 640.0),
                              saturation_params=(-0.1, 5.0, cat),
                              chromatic_params=(0.05, 0.1, 0.0, 1.5, 640.0, colors))
        self.assertFalse(np.allclose(mag_none, mag_all))

    def test_corrections_are_additive(self):
        px = np.array([5000.0])
        r = np.array([400.0])
        cat = np.array([3.0])
        colors = np.array([2.5])
        vig_poly = (0.3, 0.1, 640.0)
        sat = (-0.1, 5.0, cat)
        chrom = (0.05, 0.1, 0.0, 1.5, 640.0, colors)
        mag_base = photomLine((px, r), 10.0, 0.001)
        mag_vig = photomLine((px, r), 10.0, 0.001, vignetting_poly=vig_poly)
        mag_sat = photomLine((px, r), 10.0, 0.001, saturation_params=sat)
        mag_chrom = photomLine((px, r), 10.0, 0.001, chromatic_params=chrom)
        mag_all = photomLine((px, r), 10.0, 0.001, vignetting_poly=vig_poly,
                              saturation_params=sat, chromatic_params=chrom)
        vig_delta = mag_vig[0] - mag_base[0]
        sat_delta = mag_sat[0] - mag_base[0]
        chrom_delta = mag_chrom[0] - mag_base[0]
        combined_delta = mag_all[0] - mag_base[0]
        self.assertAlmostEqual(combined_delta, vig_delta + sat_delta + chrom_delta, places=5)


class TestPlatepar(unittest.TestCase):

    @unittest.skipIf(Platepar is None, "Full RMS deps not available")
    def test_new_fields_have_defaults(self):
        pp = Platepar()
        self.assertEqual(pp.vignetting_poly, [0.0, 0.0, 0.0])
        self.assertEqual(pp.saturation_slope, 0.0)
        self.assertEqual(pp.saturation_mag_break, 5.0)
        self.assertEqual(pp.chromatic_vignetting, [0.0, 0.0, 0.0, 1.5])

    @unittest.skipIf(Platepar is None, "Full RMS deps not available")
    def test_old_platepar_migration(self):
        pp = Platepar()
        old_dict = {k: v for k, v in pp.__dict__.items()
                    if k not in ('vignetting_poly', 'saturation_slope',
                                 'saturation_mag_break', 'chromatic_vignetting')}
        pp2 = Platepar()
        pp2.loadFromDict(old_dict)
        self.assertEqual(pp2.vignetting_poly, [0.0, 0.0, 0.0])
        self.assertEqual(pp2.saturation_slope, 0.0)
        self.assertEqual(pp2.saturation_mag_break, 5.0)
        self.assertEqual(pp2.chromatic_vignetting, [0.0, 0.0, 0.0, 1.5])

    @unittest.skipIf(Platepar is None, "Full RMS deps not available")
    def test_2element_poly_migration(self):
        pp = Platepar()
        old_dict = dict(pp.__dict__)
        old_dict['vignetting_poly'] = [0.3, 0.1]
        pp2 = Platepar()
        pp2.loadFromDict(old_dict)
        self.assertEqual(len(pp2.vignetting_poly), 3)
        self.assertAlmostEqual(pp2.vignetting_poly[0], 0.3)
        self.assertAlmostEqual(pp2.vignetting_poly[1], 0.1)
        # half_diag should be computed from X_res/Y_res, not zero
        expected_hd = np.hypot(pp2.X_res / 2, pp2.Y_res / 2)
        self.assertAlmostEqual(pp2.vignetting_poly[2], expected_hd)

    def test_defaults_produce_no_correction(self):
        px = np.array([1000.0, 500.0])
        r = np.array([200.0, 400.0])
        mag_base = photomLine((px, r), 10.0, 0.001)
        mag_defaults = photomLine((px, r), 10.0, 0.001,
                                   vignetting_poly=(0.0, 0.0, 640.0),
                                   saturation_params=(0.0, 5.0, np.array([4.0, 6.0])),
                                   chromatic_params=(0.0, 0.0, 0.0, 1.5, 640.0, np.array([1.0, 2.0])))
        np.testing.assert_array_almost_equal(mag_base, mag_defaults)


if __name__ == '__main__':
    if _USING_REAL_CODE:
        print("Testing with REAL production code from RMS.Astrometry.ApplyAstrometry")
    else:
        print("Testing with LOCAL copies (cv2/full RMS deps not available)")
    unittest.main()
