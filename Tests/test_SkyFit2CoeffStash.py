""" Headless tests for PlateparParameterManager coefficient stash ownership and DistortionDialog resets. """

from __future__ import print_function, division, absolute_import

from types import SimpleNamespace

import numpy as np
import pytest

from RMS.Formats.Platepar import Platepar


@pytest.fixture(scope="module")
def qt_app():
    pg = pytest.importorskip("pyqtgraph")
    QtWidgets = pg.Qt.QtWidgets
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _makeRadialPlatepar(dist_type="radial5-odd", force_centre=False, coeffs=None):
    """ Build a radial Platepar with optional custom forward/reverse polynomial values. """

    pp = Platepar()
    pp.X_res, pp.Y_res = 1280, 720
    pp.F_scale = 20.0
    pp.az_centre = 180.0
    pp.alt_centre = 45.0
    pp.rotation_from_horiz = 0.0
    pp.vignetting_coeff = 0.0
    pp.force_distortion_centre = force_centre
    pp.equal_aspect = False
    pp.asymmetry_corr = True
    pp.decentering_corr = False
    pp.setDistortionType(dist_type, reset_params=True)

    if coeffs is not None:
        arr = np.array(coeffs, dtype=np.float64)
        pp.x_poly_fwd = arr.copy()
        pp.x_poly_rev = arr.copy()
        pp.y_poly_fwd = arr.copy()
        pp.y_poly_rev = arr.copy()
        pp.x_poly = pp.x_poly_fwd
        pp.y_poly = pp.y_poly_fwd

    return pp


def _makeParamManager(pp):
    """ Instantiate PlateparParameterManager bound to a minimal QWidget GUI stand-in. """

    import pyqtgraph as pg
    from RMS.Routines.CustomPyqtgraphClasses import PlateparParameterManager

    gui = pg.Qt.QtWidgets.QWidget()
    gui.platepar = pp
    gui.fit_only_pointing = False
    gui.fixed_scale = False
    gui.paired_stars = []
    gui.getMinFitStars = lambda: 4
    gui.openHelpTopic = lambda topic: None
    gui.config = SimpleNamespace(platepar_distortion_type=pp.distortion_type)
    pm = PlateparParameterManager(gui)
    gui.tab = SimpleNamespace(param_manager=pm)
    return pm, gui


def testCoeffStashNotCarriedToNewPlatepar(qt_app):
    """ Loading a second platepar must not fill its zero slots with coefficients stashed from the first. """

    pp_a = _makeRadialPlatepar("radial7-odd", coeffs=[0.05, -0.04, 0.03, 0.02, -0.01, 0.11, -0.22, 0.33])
    pm, gui = _makeParamManager(pp_a)

    # Populate stash by reducing radial7-odd -> radial5-odd on platepar A
    pm._changeDistortionTypeWithStash("radial5-odd")
    assert pm._coeff_stash['x_fwd'].get('k3') == pytest.approx(0.33)

    # Load platepar B (radial7-odd with only k1 non-zero)
    pp_b = _makeRadialPlatepar("radial7-odd", coeffs=[0.0, 0.0, 0.0, 0.0, 0.0, 0.07, 0.0, 0.0])
    gui.platepar = pp_b
    pm.updatePlatepar()

    # Platepar B's zero slots must remain zero and stash must belong to B (empty)
    assert np.allclose(pp_b.x_poly_fwd, [0.0, 0.0, 0.0, 0.0, 0.0, 0.07, 0.0, 0.0])
    assert pm._coeff_stash['x_fwd'] == {}


def testDistortionTypeSwitchAcrossForcedResetDropsStash(qt_app):
    """ Switching between incompatible radial families (odd <-> all, radial <-> poly) drops the stash. """

    pp = _makeRadialPlatepar("radial5-odd", coeffs=[0.05, -0.04, 0.03, 0.02, -0.01, 0.11, -0.22])
    pm, _ = _makeParamManager(pp)

    # Switch odd -> all (forced parameter reset): stash must be dropped and array length must match
    pm._changeDistortionTypeWithStash("radial5-all")
    assert len(pp.x_poly_fwd) == pp.poly_length
    assert pm._coeff_stash['x_fwd'] == {}
    assert np.allclose(pp.x_poly_fwd[2:], 0.0)

    # Switching back must not resurrect the pre-reset odd coefficients
    pm._changeDistortionTypeWithStash("radial5-odd")
    assert np.allclose(pp.x_poly_fwd[2:], 0.0)


@pytest.mark.parametrize("reset_kind", ["resetToZero", "resetDistortion", "changeDistortionType", "firstFit"])
def testExplicitResetDropsCoeffStash(qt_app, reset_kind):
    """ Explicit distortion resets must drop the stash so later type changes cannot resurrect old values. """

    from Utils.SkyFit2 import PlateTool

    pp = _makeRadialPlatepar("radial7-odd", coeffs=[0.05, -0.04, 0.03, 0.02, -0.01, 0.11, -0.22, 0.33])
    pm, gui = _makeParamManager(pp)

    # Hide k3 in the stash
    pm._changeDistortionTypeWithStash("radial5-odd")
    assert pm._coeff_stash['x_fwd'].get('k3') == pytest.approx(0.33)

    if reset_kind == "resetToZero":
        pm.distortion_dialog.parent_widget = gui
        pm.distortion_dialog.resetToZero()
    elif reset_kind == "resetDistortion":
        gui.resetCoeffStash = lambda: PlateTool.resetCoeffStash(gui)
        gui.onFitParametersChanged = lambda: None
        gui.updateFitResiduals = lambda: None
        PlateTool.resetDistortion(gui)
    elif reset_kind == "changeDistortionType":
        gui.resetCoeffStash = lambda: PlateTool.resetCoeffStash(gui)
        gui.updateDistortion = lambda: None
        gui.dist_type_index = pp.distortion_type_list.index("radial5-odd")
        PlateTool.changeDistortionType(gui)
    elif reset_kind == "firstFit":
        gui.resetCoeffStash = lambda: PlateTool.resetCoeffStash(gui)
        gui.first_platepar_fit = True
        gui.paired_stars = [1, 2, 3, 4, 5, 6]
        gui.getMinFitStars = lambda: 4
        gui.img_handle = SimpleNamespace(currentTime=lambda: (2026, 11, 27, 4, 0, 0, 0))
        gui.tab.param_manager.setFitButtonBusy = lambda b: None

        # Stop right after fitPairs by raising a sentinel exception we catch
        class _StopAfterFit(Exception):
            """Sentinel exception."""

        def _stopFit(jd):
            raise _StopAfterFit()

        gui.fitPairs = _stopFit
        with pytest.raises(_StopAfterFit):
            PlateTool.fitPickedStars(gui)

    assert pm._coeff_stash['x_fwd'] == {}


@pytest.mark.parametrize("force_centre", [False, True])
def testDistortionResetToZeroForcedCentre(qt_app, force_centre):
    """ DistortionDialog.resetToZero zeros x_poly[0] when force_distortion_centre is True and zeros all
        of y_poly for radial models.
    """

    pp = _makeRadialPlatepar("radial5-odd", force_centre=force_centre)
    pp.x_poly_fwd[:] = np.arange(1, pp.poly_length + 1, dtype=np.float64)*0.1
    pp.x_poly_rev[:] = pp.x_poly_fwd.copy()
    pp.y_poly_fwd[:] = np.arange(1, pp.poly_length + 1, dtype=np.float64)*0.2
    pp.y_poly_rev[:] = pp.y_poly_fwd.copy()

    pm, _ = _makeParamManager(pp)
    pm.distortion_dialog.resetToZero()

    if force_centre:
        assert np.allclose(pp.x_poly_fwd, 0.0)
        assert np.allclose(pp.x_poly_rev, 0.0)
    else:
        assert pp.x_poly_fwd[0] == pytest.approx(0.1)
        assert pp.x_poly_fwd[1] == pytest.approx(0.2)
        assert np.allclose(pp.x_poly_fwd[2:], 0.0)

    # Radial y_poly has no centre slots and must be zeroed from index 0
    assert np.allclose(pp.y_poly_fwd, 0.0)
    assert np.allclose(pp.y_poly_rev, 0.0)
