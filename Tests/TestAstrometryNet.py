""" Tests for the local astrometry.net solve hints in RMS.Astrometry.AstrometryNet. """

from __future__ import print_function, division, absolute_import

from types import SimpleNamespace

import numpy as np
import pytest

import RMS.Astrometry.AstrometryNet as AstrometryNet


def _fakeAstrometry(record):
    """ Build a stand-in for the astrometry package which records the solver hints. """

    class SizeHint(object):
        def __init__(self, **kwargs):
            record['hint'] = kwargs

    class Solution(object):
        def has_match(self):
            return False

    class Solver(object):
        def __init__(self, index_files):
            pass

        def solve(self, stars=None, size_hint=None, position_hint=None, solution_parameters=None):
            return Solution()

    def indexFiles(cache_directory=None, scales=None):
        record['scales'] = scales

    return SimpleNamespace(
        SizeHint=SizeHint, Solver=Solver, series_4100=SimpleNamespace(index_files=indexFiles),
        SolutionParameters=lambda **kwargs: None, Action=SimpleNamespace(STOP=0, CONTINUE=1)
    )


@pytest.mark.parametrize("fov", [60, 100, 120, 150, 180])
def testStarListSizeHintContainsTruePixelScale(monkeypatch, fov):
    """ The pixel scale hint of a star-list solve must contain the true scale, also for wide FOVs.

        For FOVs > 90 deg the stars are filtered to the image centre, which used to shrink the image width
        taken from the star positions (while the FOV range was replaced by ~120 deg), excluding the true
        arcsec/px of 100-150 deg lenses.
    """

    record = {}
    monkeypatch.setattr(AstrometryNet, 'astrometry', _fakeAstrometry(record), raising=False)

    width, height = 1920, 1080
    rng = np.random.default_rng(0)
    x_data = rng.uniform(0, width, 400)
    y_data = rng.uniform(0, height, 400)

    AstrometryNet.astrometryNetSolveLocal(
        x_data=x_data, y_data=y_data, fov_w_range=[0.75*fov, 1.5*fov], fov_w_hint=fov,
        x_center=width/2, y_center=height/2, input_intensities=rng.uniform(1, 10, 400)
    )

    true_scale = fov*3600.0/width
    hint = record['hint']

    assert hint['lower_arcsec_per_pixel'] <= true_scale <= hint['upper_arcsec_per_pixel']
    assert np.isclose(hint['lower_arcsec_per_pixel'], 0.75*true_scale)
    assert np.isclose(hint['upper_arcsec_per_pixel'], 1.5*true_scale)


@pytest.mark.parametrize("ra_mid", [0.001, 120.0, 359.999])
def testRotationEqStandardWrapsRA(ra_mid):
    """ The orientation must not flip by 180 deg when the RA difference straddles RA = 0. """

    from RMS.Astrometry.AstrometryNetNova import rotationEqStandard

    # The point right of the centre is 0.002 deg lower in RA and 0.001 deg lower in Dec
    ra_right = (ra_mid - 0.002)%360
    rot = rotationEqStandard(ra_mid, 30.0, ra_right, 29.999)

    expected = np.degrees(np.arctan2(0.001, 0.002))%360
    assert np.isclose(rot, expected)
