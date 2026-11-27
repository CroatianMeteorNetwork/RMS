""" Unit tests for preserving projection across radial distortion flag toggles in RMS.Formats.Platepar. """

from __future__ import print_function, division, absolute_import

import numpy as np
import pytest

from RMS.Astrometry.ApplyAstrometry import raDecToXYPP, xyToRaDecPP
from RMS.Formats.Platepar import Platepar


RADIAL_TYPES = [
    "radial3-all", "radial4-all", "radial5-all",
    "radial3-odd", "radial5-odd", "radial7-odd", "radial9-odd",
]

RADIAL_FLAGS = [
    ('force_distortion_centre', True, False),
    ('equal_aspect', True, False),
    ('asymmetry_corr', False, True),
]


def _radialPlatepar(dist_type, decentering_corr=False):
    """ Build a radial Platepar with all flags restricted and non-zero distortion terms. """

    pp = Platepar()
    pp.X_res, pp.Y_res = 1920, 1080
    pp.F_scale = 1920/90.0
    pp.JD = 2460000.5
    pp.RA_d, pp.dec_d = 100.0, 30.0
    pp.pos_angle_ref = 20.0
    pp.lat, pp.lon = 45.0, 15.0
    pp.refraction = False

    pp.force_distortion_centre = True
    pp.equal_aspect = True
    pp.asymmetry_corr = False
    pp.decentering_corr = decentering_corr
    pp.setDistortionType(dist_type, reset_params=True)

    # Set small non-zero radial (and optional decentering) terms
    n = pp.poly_length
    ks = 0.02*(np.arange(n) + 1)*(-1)**np.arange(n)
    if decentering_corr:
        ks[-4:] = np.array([1e-3, -1e-3, 5e-4, -5e-4])
    pp.x_poly_fwd = ks.copy()
    pp.x_poly_rev = -ks.copy()
    pp.y_poly_fwd = ks.copy()
    pp.y_poly_rev = -ks.copy()
    pp.x_poly = pp.x_poly_fwd
    pp.y_poly = pp.y_poly_fwd

    return pp


def _projection(pp):
    """ Project an image grid to sky coordinates and back as a regression fingerprint. """

    xs, ys = np.meshgrid(np.linspace(10, pp.X_res - 10, 9), np.linspace(10, pp.Y_res - 10, 7))
    xs, ys = xs.ravel(), ys.ravel()

    _, ra, dec, _ = xyToRaDecPP(
        len(xs)*[pp.JD], xs, ys, np.ones(len(xs)), pp, extinction_correction=False, jd_time=True
    )
    x_back, y_back = raDecToXYPP(ra, dec, pp.JD, pp)

    return np.c_[ra, dec], np.c_[x_back, y_back]


@pytest.mark.parametrize("dist_type", RADIAL_TYPES)
@pytest.mark.parametrize("decentering_corr", [False, True])
@pytest.mark.parametrize("flag_name, restricted_value, free_value", RADIAL_FLAGS)
def testRadialFlagToggleKeepsProjection(dist_type, decentering_corr, flag_name, restricted_value, free_value):
    """ Freeing a radial distortion flag and restricting it back must preserve the projection. """

    pp = _radialPlatepar(dist_type, decentering_corr=decentering_corr)
    sky_ref, img_ref = _projection(pp)

    # Free the flag and check that both forward and reverse projections stay identical
    assert pp.remapCoeffsForFlagChange(flag_name, free_value)
    assert len(pp.x_poly_fwd) == pp.poly_length
    sky, img = _projection(pp)
    assert np.max(np.abs(img - img_ref)) < 1e-6
    assert np.max(np.abs(sky - sky_ref)) < 1e-6/3600

    # Restrict the flag back and verify the round trip
    assert pp.remapCoeffsForFlagChange(flag_name, restricted_value)
    sky, img = _projection(pp)
    assert np.max(np.abs(img - img_ref)) < 1e-6
    assert np.max(np.abs(sky - sky_ref)) < 1e-6/3600


def testStashForcedCentreRoundTripRestoresFittedCentre():
    """ Toggling force_distortion_centre on and off via PlateparParameterManager must restore the fitted
        free centre instead of overwriting it with the forced 0.5/(res/2) offset.
    """

    pytest.importorskip("pyqtgraph")
    from types import SimpleNamespace
    from RMS.Routines.CustomPyqtgraphClasses import PlateparParameterManager

    pp = _radialPlatepar("radial5-odd")
    pp.remapCoeffsForFlagChange('force_distortion_centre', False)
    for name in ('x_poly_fwd', 'x_poly_rev'):
        getattr(pp, name)[0] = 0.05
        getattr(pp, name)[1] = -0.03
    k_before = np.array(pp.x_poly_fwd[2:])

    # Minimal stand-in holding the platepar and stash without instantiating a Qt widget
    class _StubManager(object):
        _isEmptyCoeff = staticmethod(PlateparParameterManager._isEmptyCoeff)
        resetCoeffStash = PlateparParameterManager.resetCoeffStash
        _checkCoeffStashOwner = PlateparParameterManager._checkCoeffStashOwner
        _stashCurrentCoeffs = PlateparParameterManager._stashCurrentCoeffs
        _restoreCoeffsFromStash = PlateparParameterManager._restoreCoeffsFromStash
        _remapCoeffsWithStash = PlateparParameterManager._remapCoeffsWithStash

    pm = _StubManager()
    pm.gui = SimpleNamespace(platepar=pp)
    pm.resetCoeffStash()

    pm._remapCoeffsWithStash('force_distortion_centre', True)
    pm._remapCoeffsWithStash('force_distortion_centre', False)

    for name in ('x_poly_fwd', 'x_poly_rev'):
        assert getattr(pp, name)[0] == pytest.approx(0.05)
        assert getattr(pp, name)[1] == pytest.approx(-0.03)
    assert np.allclose(pp.x_poly_fwd[2:], k_before)
