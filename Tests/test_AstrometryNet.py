""" Unit tests for local and remote astrometry.net solve hints and orientation in RMS.Astrometry. """

from __future__ import print_function, division, absolute_import

from types import SimpleNamespace

import numpy as np
import pytest

import RMS.Astrometry.AstrometryNet as AstrometryNet
from RMS.Astrometry.AstrometryNetNova import rotationEqStandard


def _fakeAstrometry(record):
    """ Build a stand-in for the astrometry package that records solver hints and input stars. """

    class SizeHint(object):
        def __init__(self, **kwargs):
            record['hint'] = kwargs

    class Solution(object):
        def has_match(self):
            return False

    class Solver(object):
        def __init__(self, index_files):
            return None

        def solve(self, stars=None, size_hint=None, position_hint=None, solution_parameters=None):
            record['stars'] = np.array(stars, copy=True) if stars is not None else None
            return Solution()

    def indexFiles(cache_directory=None, scales=None):
        record['scales'] = scales

    return SimpleNamespace(
        SizeHint=SizeHint, Solver=Solver, series_4100=SimpleNamespace(index_files=indexFiles),
        SolutionParameters=lambda **kwargs: None, Action=SimpleNamespace(STOP=0, CONTINUE=1)
    )


def _solveStarList(monkeypatch, fov_w_range, fov_w_hint, with_centre, with_intensities=True, verbose=False):
    """ Run a star-list solve with the fake solver and return the recorded hints and true scale. """

    record = {}
    monkeypatch.setattr(AstrometryNet, 'astrometry', _fakeAstrometry(record), raising=False)

    width, height = 1920, 1080
    rng = np.random.RandomState(0)
    x_data = rng.uniform(0, width, 400)
    y_data = rng.uniform(0, height, 400)

    centre = dict(x_center=width/2, y_center=height/2) if with_centre else {}
    intensities = rng.uniform(1, 10, 400) if with_intensities else None

    AstrometryNet.astrometryNetSolveLocal(
        x_data=x_data, y_data=y_data, fov_w_range=fov_w_range, fov_w_hint=fov_w_hint,
        input_intensities=intensities, verbose=verbose, **centre
    )

    true_scale = fov_w_hint*3600.0/width if fov_w_hint is not None else None
    return record, true_scale


@pytest.mark.parametrize("fov", [60, 100, 120, 150, 180])
def testStarListSizeHintContainsTruePixelScale(monkeypatch, fov):
    """ The pixel scale hint of a star-list solve must contain the true scale, also for wide FOVs. """

    record, true_scale = _solveStarList(monkeypatch, [0.75*fov, 1.5*fov], fov, with_centre=True)

    hint = record['hint']
    assert hint['lower_arcsec_per_pixel'] <= true_scale <= hint['upper_arcsec_per_pixel']
    assert np.isclose(hint['lower_arcsec_per_pixel'], 0.75*true_scale)
    assert np.isclose(hint['upper_arcsec_per_pixel'], 1.5*true_scale)


@pytest.mark.parametrize("fov", [60, 100, 120, 150])
def testStarListSizeHintWithoutImageCentre(monkeypatch, fov):
    """ Without an image or centre the width comes from the extent of all (unfiltered) stars. """

    record, true_scale = _solveStarList(monkeypatch, [0.75*fov, 1.5*fov], fov, with_centre=False)

    hint = record['hint']
    assert hint['lower_arcsec_per_pixel'] <= true_scale <= hint['upper_arcsec_per_pixel']
    assert np.isclose(hint['lower_arcsec_per_pixel'], 0.75*true_scale, rtol=0.02)


@pytest.mark.parametrize("with_centre", [True, False])
def testWideFOVHintWithoutRange(monkeypatch, with_centre):
    """ A wide FOV hint without a FOV range must still give a scale hint around the hinted FOV. """

    record, true_scale = _solveStarList(monkeypatch, None, 120, with_centre=with_centre)

    hint = record['hint']
    assert hint['lower_arcsec_per_pixel'] <= true_scale <= hint['upper_arcsec_per_pixel']
    assert np.isclose(hint['upper_arcsec_per_pixel'], 2*hint['lower_arcsec_per_pixel'])


@pytest.mark.parametrize("ra_mid, dra_true", [
    (0.001, 0.002),
    (359.999, -0.002),
    (120.0, 0.002),
    (359.999, 0.002),
])
def testRotationEqStandardWrapsRA(ra_mid, dra_true):
    """ The orientation must not flip by 180 deg when the RA difference straddles RA = 0 in either direction. """

    ra_right = (ra_mid - dra_true)%360
    rot = rotationEqStandard(ra_mid, 30.001, ra_right, 30.0)

    expected = np.degrees(np.arctan2(0.001, dra_true))%360
    assert np.isclose(rot, expected)


def testDebugPrintIsGatedBehindVerbose(monkeypatch, capsys):
    """ The FOV hint print must appear only when verbose=True. """

    _solveStarList(monkeypatch, [45.0, 90.0], 60.0, with_centre=True, verbose=False)
    out_quiet = capsys.readouterr().out
    assert "fov_w_hint" not in out_quiet

    _solveStarList(monkeypatch, [45.0, 90.0], 60.0, with_centre=True, verbose=True)
    out_verbose = capsys.readouterr().out
    assert "fov_w_hint" in out_verbose


def testRandomStarSubsampleIsReproducibleAndLeavesGlobalRNGAlone(monkeypatch):
    """ Without intensities, subsampling must pick identical stars across calls without mutating global RNG. """

    np.random.seed(12345)
    before_state = np.random.get_state()[1].copy()
    r1, _ = _solveStarList(monkeypatch, [45.0, 90.0], 60.0, with_centre=True, with_intensities=False)
    after_state = np.random.get_state()[1].copy()
    r2, _ = _solveStarList(monkeypatch, [45.0, 90.0], 60.0, with_centre=True, with_intensities=False)

    assert np.array_equal(r1['stars'], r2['stars'])
    assert np.array_equal(before_state, after_state)
