""" Headless tests for SkyFit2 Auto Fit failure restoration, distortion detection, LM restore and background lock. """

from __future__ import print_function, division, absolute_import

import copy
import time
from types import SimpleNamespace

import numpy as np
import pytest

from RMS.Formats.Platepar import Platepar


@pytest.fixture(scope="module")
def skyfit2():
    pg = pytest.importorskip("pyqtgraph")
    QtWidgets = pg.Qt.QtWidgets
    if QtWidgets.QApplication.instance() is None:
        QtWidgets.QApplication([])
    return pytest.importorskip("Utils.SkyFit2")


def _makePlatepar(dist_type="radial9-odd", with_distortion=True, force_centre=False):
    """ Build a radial or polynomial Platepar for auto-fit regression checks. """

    pp = Platepar()
    pp.X_res, pp.Y_res = 1280, 720
    pp.F_scale = 21.5
    pp.JD = 2461000.5
    pp.RA_d, pp.dec_d = 120.0, 45.0
    pp.pos_angle_ref = 15.0
    pp.lat, pp.lon, pp.elev = 43.0, -81.0, 329.0
    pp.refraction = False
    pp.equal_aspect = False
    pp.asymmetry_corr = True
    pp.decentering_corr = False
    pp.force_distortion_centre = force_centre
    pp.setDistortionType(dist_type, reset_params=True)

    if with_distortion:
        if dist_type.startswith("radial"):
            pp.x_poly_fwd[-1] = 0.05
            pp.x_poly_rev[-1] = -0.05
        else:
            pp.x_poly_fwd[1] = 0.02
            pp.y_poly_fwd[1] = -0.02

    pp.x_poly = pp.x_poly_fwd
    pp.y_poly = pp.y_poly_fwd
    return pp


def _makeAutoFitTool(skyfit2, pp):
    """ Build a lightweight PlateTool stand-in bound to the real Auto Fit methods. """

    PlateTool = skyfit2.PlateTool

    class _StubTool(object):
        _restorePlateparInPlace = PlateTool._restorePlateparInPlace
        _plateparHasDistortion = staticmethod(PlateTool._plateparHasDistortion)
        _applyPointingToOriginalPlatepar = PlateTool._applyPointingToOriginalPlatepar
        tryQuickAlignment = PlateTool.tryQuickAlignment
        _runQuickAlignment = PlateTool._runQuickAlignment
        quickAlign = PlateTool.quickAlign
        autoFitAstrometryNet = PlateTool.autoFitAstrometryNet
        getInitialParamsAstrometryNet = PlateTool.getInitialParamsAstrometryNet
        runInBackground = PlateTool.runInBackground

    pt = _StubTool()
    pt.platepar = pp
    pt.fit_only_pointing = True
    pt.cat_lim_mag = 6.5
    pt.config = SimpleNamespace(catalog_mag_limit=6.5, fov_w=60.0)
    pt.catalog_stars = np.column_stack([
        np.linspace(110.0, 130.0, 30),
        np.linspace(35.0, 55.0, 30),
        np.linspace(3.0, 6.0, 30),
    ])
    pt.paired_stars = skyfit2.PairedStars()
    pt.astrometry_solution_info = None
    pt._background_busy = False
    pt.status_bar = SimpleNamespace(showMessage=lambda msg: None)
    pt.img_handle = SimpleNamespace(currentTime=lambda: (2026, 11, 27, 4, 0, 0, 0))
    pt.tab = SimpleNamespace(
        param_manager=SimpleNamespace(
            setQuickAlignButtonBusy=lambda b: None,
            setAutoFitButtonBusy=lambda b: None,
            updatePlatepar=lambda: None,
        )
    )
    pt.updateStars = lambda **kw: None
    pt.updateLeftLabels = lambda: None
    pt.updateDistortion = lambda: None
    pt.drawPhotometryColoring = lambda: None
    pt.updateDistortionCenterMarker = lambda: None
    pt.loadCatalogStars = lambda lm: pt.catalog_stars
    pt.filterCatalogStarsInsideFOV = lambda cat: (np.arange(len(cat)), cat)
    pt.filterCatalogStarsByMask = lambda cx, cy, cat: (cat, np.ones(len(cat), dtype=bool))
    pt.getDetectedStars = lambda: np.column_stack([
        np.linspace(100, 600, 20), np.linspace(100, 1100, 20),
        np.full(20, 5000.0), np.full(20, 500.0), np.full(20, 2.5),
        np.full(20, 50.0), np.full(20, 20.0), np.zeros(20),
    ])
    return pt


@pytest.mark.parametrize("mode", ["none", "raise"])
def testQuickAlignmentFailureRestoresPlateparInPlace(skyfit2, monkeypatch, mode):
    """ Rejected or raising NN fit in tryQuickAlignment must return False and restore the same platepar object. """

    pp = _makePlatepar("radial9-odd", with_distortion=True)
    orig_id = id(pp)
    orig_coeffs = pp.x_poly_fwd.copy()
    pt = _makeAutoFitTool(skyfit2, pp)

    def _fakeAlign(config, platepar, t, det, lm_callback=None):
        aligned = copy.deepcopy(platepar)
        aligned.RA_d += 1.0
        return aligned, 7.5

    def _fakeFit(self, *args, **kwargs):
        if mode == "raise":
            raise RuntimeError("fit failure")
        return None

    monkeypatch.setattr(skyfit2, "alignPlatepar", _fakeAlign)
    monkeypatch.setattr(skyfit2, "getCatalogStarsImagePositions",
                        lambda cat, jd, p: (np.full(len(cat), 200.0), np.full(len(cat), 200.0), cat[:, 2]))
    monkeypatch.setattr(Platepar, "fitAstrometry", _fakeFit)

    ok = pt.tryQuickAlignment(pointing_only=False)

    assert ok is False
    assert id(pt.platepar) == orig_id
    assert pt.platepar.distortion_type == "radial9-odd"
    assert pt.platepar.equal_aspect is False
    assert pt.platepar.asymmetry_corr is True
    assert pt.platepar.refraction is False
    assert np.allclose(pt.platepar.x_poly_fwd, orig_coeffs)
    assert pt.fit_only_pointing is True
    assert pt.cat_lim_mag == 6.5


@pytest.mark.parametrize("mode", ["none", "raise"])
def testInitialParamsRejectedNNFitRestoresPlateparInPlace(skyfit2, monkeypatch, mode):
    """ Rejected or raising NN fit in getInitialParamsAstrometryNet must not fit stale star_list and must
        restore the original platepar in place.
    """

    pp = _makePlatepar("radial9-odd", with_distortion=True)
    pp.star_list = [[2460000.0, 100.0, 100.0, 5000.0, 10.0, 10.0, 4.0]]*15
    orig_id = id(pp)
    orig_coeffs = pp.x_poly_fwd.copy()
    pt = _makeAutoFitTool(skyfit2, pp)

    pt._solveAstrometryNet = lambda **kw: (125.0, 46.0, 20.0, 22.0, 58.0, 33.0, None, None)
    pt.balanceCatalogMagnitude = lambda: None
    fit_picked_called = []
    pt.fitPickedStars = lambda: fit_picked_called.append(True)

    monkeypatch.setattr(skyfit2, "qmessagebox", lambda **kw: None)
    monkeypatch.setattr(skyfit2.QtWidgets.QMessageBox, "question",
                        lambda *a, **k: skyfit2.QtWidgets.QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(skyfit2, "getCatalogStarsImagePositions",
                        lambda cat, jd, p: (np.full(len(cat), 200.0), np.full(len(cat), 200.0), cat[:, 2]))

    def _fakeFit(self, *args, **kwargs):
        if mode == "raise":
            raise RuntimeError("boom")
        return None

    monkeypatch.setattr(Platepar, "fitAstrometry", _fakeFit)

    res = pt.getInitialParamsAstrometryNet(upload_image=False)

    assert res is None
    assert fit_picked_called == []
    assert id(pt.platepar) == orig_id
    assert pt.platepar.distortion_type == "radial9-odd"
    assert np.allclose(pt.platepar.x_poly_fwd, orig_coeffs)
    assert pt.fit_only_pointing is True


@pytest.mark.parametrize("dist_type, force_centre, with_dist, expected", [
    ("poly3+radial", False, False, False),
    ("poly3+radial5", False, False, False),
    ("poly3+radial5", False, True, True),
    ("radial5-odd", False, False, False),
    ("radial5-odd", True, False, False),
    ("radial5-odd", False, True, True),
    ("radial9-odd", True, True, True),
])
def testPlateparHasDistortionIgnoresDefaultCentreOffsets(skyfit2, dist_type, force_centre, with_dist, expected):
    """ Default centre/offset entries (0.5 in poly, x0/y0 in radial) must not count as fitted distortion. """

    pp = _makePlatepar(dist_type, with_distortion=with_dist, force_centre=force_centre)
    assert skyfit2.PlateTool._plateparHasDistortion(pp) is expected


@pytest.mark.parametrize("branch", ["no_refinement", "too_few_pairs"])
@pytest.mark.parametrize("with_dist", [False, True])
def testInitialParamsKeepsUserDistortionWithoutFinalFit(skyfit2, monkeypatch, branch, with_dist):
    """ Answering No or matching <10 pairs restores user distortion and applies solved scale only when
        the original platepar had no distortion.
    """

    pp = _makePlatepar("radial7-odd", with_distortion=with_dist)
    orig_scale = pp.F_scale
    orig_coeffs = pp.x_poly_fwd.copy()
    pt = _makeAutoFitTool(skyfit2, pp)

    solved_scale = 28.0
    pt._solveAstrometryNet = lambda **kw: (125.0, 46.0, 20.0, solved_scale, 58.0, 33.0, None, None)
    pt.balanceCatalogMagnitude = lambda: None
    applied_pointing = []
    pt._applyAstrometryNetPointing = lambda sol: applied_pointing.append(sol)

    monkeypatch.setattr(skyfit2, "qmessagebox", lambda **kw: None)
    monkeypatch.setattr(skyfit2, "getCatalogStarsImagePositions",
                        lambda cat, jd, p: (np.full(len(cat), 200.0), np.full(len(cat), 200.0), cat[:, 2]))

    if branch == "no_refinement":
        monkeypatch.setattr(skyfit2.QtWidgets.QMessageBox, "question",
                            lambda *a, **k: skyfit2.QtWidgets.QMessageBox.StandardButton.No)
    else:
        monkeypatch.setattr(skyfit2.QtWidgets.QMessageBox, "question",
                            lambda *a, **k: skyfit2.QtWidgets.QMessageBox.StandardButton.Yes)
        # Return only 3 matched pairs (< 10)
        small_img = np.zeros((3, 3))
        small_cat = np.zeros((3, 3))
        monkeypatch.setattr(Platepar, "fitAstrometry", lambda self, *a, **k: (small_img, small_cat))

    pt.getInitialParamsAstrometryNet(upload_image=False)

    assert len(applied_pointing) == 1
    assert pt.platepar.distortion_type == "radial7-odd"
    assert np.allclose(pt.platepar.x_poly_fwd, orig_coeffs)
    if with_dist:
        assert pt.platepar.F_scale == pytest.approx(orig_scale)
    else:
        assert pt.platepar.F_scale == pytest.approx(solved_scale)


@pytest.mark.parametrize("method_name", ["quickAlign", "autoFitAstrometryNet"])
@pytest.mark.parametrize("raises", [False, True])
def testAutoFitRestoresConfigCatalogLM(skyfit2, method_name, raises):
    """ quickAlign and autoFitAstrometryNet must restore config.catalog_mag_limit and button state even on error. """

    pp = _makePlatepar("radial5-odd", with_distortion=False)
    pt = _makeAutoFitTool(skyfit2, pp)
    pt.config.catalog_mag_limit = 6.5

    busy_states = []
    pt.tab.param_manager.setQuickAlignButtonBusy = lambda b: busy_states.append(b)
    pt.tab.param_manager.setAutoFitButtonBusy = lambda b: busy_states.append(b)

    def _fakeBalance():
        pt.cat_lim_mag = 9.2
        pt.config.catalog_mag_limit = 9.2

    pt.balanceCatalogMagnitude = _fakeBalance

    if raises:
        def _boom(**kw):
            raise RuntimeError("alignment error")
        pt.tryQuickAlignment = _boom
        with pytest.raises(RuntimeError, match="alignment error"):
            getattr(pt, method_name)()
    else:
        pt.tryQuickAlignment = lambda **kw: True
        getattr(pt, method_name)()

    assert pt.cat_lim_mag == pytest.approx(6.5)
    assert pt.config.catalog_mag_limit == pytest.approx(6.5)
    assert busy_states == [True, False]


def testTryQuickAlignmentUncaughtExceptionRestoresPlateparInPlace(skyfit2):
    """ An uncaught exception inside _runQuickAlignment must still restore platepar in place. """

    pp = _makePlatepar("radial9-odd", with_distortion=True)
    orig_id = id(pp)
    orig_coeffs = pp.x_poly_fwd.copy()
    pt = _makeAutoFitTool(skyfit2, pp)
    pt.fit_only_pointing = False

    def _mutatingBoom(pointing_only=False):
        pt.platepar.setDistortionType("radial5-odd", reset_params=True)
        pt.fit_only_pointing = True
        pt.cat_lim_mag = 9.5
        raise RuntimeError("unexpected crash")

    pt._runQuickAlignment = _mutatingBoom

    with pytest.raises(RuntimeError, match="unexpected crash"):
        pt.tryQuickAlignment()

    assert pt.platepar is pp
    assert id(pt.platepar) == orig_id
    assert pt.platepar.distortion_type == "radial9-odd"
    assert np.allclose(pt.platepar.x_poly_fwd, orig_coeffs)
    assert pt.fit_only_pointing is False
    assert pt.cat_lim_mag == pytest.approx(6.5)


def testApplyAstrometryNetPointingUpdatesRotationFromHoriz(skyfit2):
    """ _applyAstrometryNetPointing must sync rotation_from_horiz with the new pointing and pos_angle_ref. """

    pp = _makePlatepar("radial5-odd", with_distortion=True)
    pp.lat = 43.0
    pp.lon = -81.0
    pp.elev = 250.0
    pp.rotation_from_horiz = -999.0
    pt = _makeAutoFitTool(skyfit2, pp)
    pt._applyAstrometryNetPointing = skyfit2.PlateTool._applyAstrometryNetPointing.__get__(pt)

    solution = (120.0, 45.0, 25.0, 22.0, 64.0, 36.0, None, {})
    pt._applyAstrometryNetPointing(solution)

    assert pt.platepar.rotation_from_horiz != pytest.approx(-999.0)
    assert np.isfinite(pt.platepar.rotation_from_horiz)


def testRunInBackgroundRefusesNestedCallAndClearsBusyFlag(skyfit2):
    """ runInBackground refuses a re-entrant call while busy and clears its flag after an exception. """

    pp = _makePlatepar("radial5-odd", with_distortion=False)
    pt = _makeAutoFitTool(skyfit2, pp)

    # While busy, a second call is refused and returns None
    pt._background_busy = True
    called = []
    assert pt.runInBackground(lambda: called.append(1)) is None
    assert called == []

    # Normal execution returns result and clears busy flag
    pt._background_busy = False
    assert pt.runInBackground(lambda x: x + 1, 41) == 42
    assert pt._background_busy is False

    # Exception still clears busy flag
    def _boom():
        time.sleep(0.01)
        raise ValueError("worker error")

    with pytest.raises(ValueError, match="worker error"):
        pt.runInBackground(_boom)
    assert pt._background_busy is False
