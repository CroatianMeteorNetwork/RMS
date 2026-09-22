""" Regression tests for defects found in the second review of SkyFit2.

    The tests that need Qt run offscreen and are skipped cleanly when Qt or pyqtgraph is not available.
"""

from __future__ import print_function, division, absolute_import

import copy
import functools
import os
import shutil
import tarfile
import types

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

pytest.importorskip("pyqtgraph")

try:
    import Utils.SkyFit2 as SF
except SystemExit:
    pytest.skip("SkyFit2 could not import its Qt dependencies", allow_module_level=True)

from pyqtgraph.Qt import QtWidgets

import RMS.ConfigReader as cr
import RMS.Routines.CustomPyqtgraphClasses as CPC


REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_CONFIG = os.path.join(REPO_DIR, ".config")
STATIONS_ARCHIVE = os.path.join(REPO_DIR, "Tests", "ExampleStationData", "stations.tar.bz2")


###################################################################################################
# FIXTURES
###################################################################################################

@pytest.fixture(scope="module")
def qapp():
    """ The offscreen QApplication shared by the GUI tests. """

    app = QtWidgets.QApplication.instance()
    if app is None:
        try:
            app = QtWidgets.QApplication([])
        except Exception as e:
            pytest.skip("No Qt application could be created: {}".format(e))

    return app


@pytest.fixture(scope="module")
def stationArchive(tmp_path_factory):
    """ The example station AU000A, extracted once per module. """

    if not os.path.isfile(STATIONS_ARCHIVE):
        pytest.skip("The example station archive is missing")

    out_dir = str(tmp_path_factory.mktemp("stations"))
    with tarfile.open(STATIONS_ARCHIVE, "r:bz2") as tar:
        members = [m for m in tar.getmembers() if m.name.startswith(("Stations/AU000A", "Stations/AU000C"))]
        tar.extractall(out_dir, members=members)

    return os.path.join(out_dir, "Stations", "AU000A")


@pytest.fixture
def stationDir(stationArchive, tmp_path):
    """ A fresh copy of the example station for each test. """

    dir_path = str(tmp_path / "AU000A")
    shutil.copytree(stationArchive, dir_path)

    return dir_path


@pytest.fixture
def secondStationDir(stationArchive, tmp_path):
    """ A fresh copy of the example station AU000C, next to the AU000A copy. """

    dir_path = str(tmp_path / "AU000C")
    shutil.copytree(os.path.join(os.path.dirname(stationArchive), "AU000C"), dir_path)

    return dir_path


@pytest.fixture
def quietMessages(monkeypatch):
    """ Record the SkyFit2 message boxes instead of showing them, and answer questions with Yes. """

    messages = []
    record = lambda *a, **k: messages.append(k)
    monkeypatch.setattr(SF, "qmessagebox", record)
    monkeypatch.setattr(CPC, "qmessagebox", record)

    yes = QtWidgets.QMessageBox.StandardButton.Yes
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", staticmethod(lambda *a, **k: yes))
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QtWidgets.QMessageBox, name, staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", lambda self: 0, raising=False)
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec_", lambda self: 0, raising=False)

    return messages


@pytest.fixture
def plateTool(qapp, stationDir, quietMessages):
    """ A SkyFit2 PlateTool on a fresh copy of the example station. """

    config = cr.loadConfigFromDirectory('.config', stationDir)
    pt = SF.PlateTool(stationDir, config)
    qapp.processEvents()

    yield pt

    pt.close()
    pt.deleteLater()
    qapp.processEvents()


###################################################################################################
# CONFIG WRITING
###################################################################################################

def _fakePlateTool(config_path, **overrides):
    """ Build the minimal object that the config writer and the File Manager config save read.

    Arguments:
        config_path: [str] Path of the "loaded" config file.

    Keyword arguments:
        overrides: Star detection override values to use instead of the defaults.

    Return:
        pt: [SimpleNamespace] Stand-in for PlateTool.
    """

    pt = types.SimpleNamespace()
    pt.config = cr.parse(config_path)
    pt.override_intensity_threshold = overrides.get('intensity_threshold', 41)
    pt.override_neighborhood_size = overrides.get('neighborhood_size', 7)
    pt.override_max_stars = overrides.get('max_stars', 1200)
    pt.override_segment_radius = overrides.get('segment_radius', 5)
    pt.override_max_feature_ratio = overrides.get('max_feature_ratio', 0.77)
    pt.override_roundness_threshold = overrides.get('roundness_threshold', 0.33)
    pt.tuned_cat_lim_mag = overrides.get('tuned_cat_lim_mag', 6.3)
    pt.platepar_modified = False
    pt.configMaxStars = functools.partial(SF.PlateTool.configMaxStars, pt)
    pt._writeStarDetectionConfig = functools.partial(SF.PlateTool._writeStarDetectionConfig, pt)
    pt._updateConfigSaveButtonState = lambda: None
    pt.updateFileManagerButton = lambda: None

    return pt


@pytest.mark.parametrize("spelling", ["{k}: {v}", "{k}:{v}", "{k}={v}", "{k} = {v}", "{k} : {v}  ; note"])
def testWriteConfigAnyKeySpelling(tmp_path, quietMessages, spelling):
    """ Every key spelling configparser accepts is updated in place, never appended a second time. """

    # Respell the StarExtraction keys of the template config
    keys = ["intensity_threshold", "neighborhood_size", "max_stars", "segment_radius", "max_feature_ratio",
            "roundness_threshold"]
    lines = []
    section = None
    with open(TEMPLATE_CONFIG) as f:
        for line in f:
            if line.strip().startswith('['):
                section = line.strip()
            key = line.split(':')[0].strip()
            if section == "[StarExtraction]" and key in keys and not line.lstrip().startswith(';'):
                value = line.split(':', 1)[1].split(';')[0].strip()
                line = spelling.format(k=key, v=value) + "\n"
            lines.append(line)

    cfg_path = str(tmp_path / ".config")
    with open(cfg_path, 'w') as f:
        f.writelines(lines)

    pt = _fakePlateTool(cfg_path)
    pt._writeStarDetectionConfig(cfg_path, catalog_mag_limit=6.3)

    # The strict parser RMS starts with must accept the result, with the new values
    config = cr.parse(cfg_path, strict=True)
    assert config.intensity_threshold == 41
    assert config.max_stars == 1200
    assert config.max_feature_ratio == pytest.approx(0.77)
    assert config.catalog_mag_limit == pytest.approx(6.3)

    # The spelling and the inline comment are kept
    with open(cfg_path) as f:
        text = f.read()
    assert spelling.format(k="intensity_threshold", v="41") in text


def testUpdateConfigLinesAddsMissing():
    """ Missing keys go at the end of their section, missing sections at the end of the file. """

    lines = ["[StarExtraction]\n", "max_stars=100\n", "\n", "[Other]\n", "x: 1"]
    new = SF.updateConfigLines(lines, {"StarExtraction": {"max_stars": "5", "segment_radius": "4"},
                                       "Calibration": {"catalog_mag_limit": "6.0"}})

    assert new[:3] == ["[StarExtraction]\n", "max_stars=5\n", "segment_radius: 4\n"]
    assert "x: 1\n" in new
    assert new[-2:] == ["\n[Calibration]\n", "catalog_mag_limit: 6.0\n"]


def testSaveConfigKeepsTargetStation(tmp_path, quietMessages):
    """ Saving the config to another station patches that station's config instead of replacing it. """

    src_dir = tmp_path / "src"
    dst_dir = tmp_path / "dst"
    src_dir.mkdir()
    dst_dir.mkdir()

    src_path = str(src_dir / ".config")
    shutil.copy(TEMPLATE_CONFIG, src_path)

    # The target config has its own station ID and a line the source does not have
    with open(TEMPLATE_CONFIG) as f:
        text = f.read()
    text = text.replace("stationID: XX0001", "stationID: ZZ0009")
    text = text.replace("[StarExtraction]", "[StarExtraction]\n; target only comment line")
    dst_path = str(dst_dir / ".config")
    with open(dst_path, 'w') as f:
        f.write(text)

    pt = _fakePlateTool(src_path)
    dialog = types.SimpleNamespace(plate_tool=pt, _refreshAll=lambda: None)
    SF.CalibrationFilesDialog._saveFile(dialog, "Config", [str(dst_dir)])

    # The target keeps its identity and gets the new detection settings
    with open(dst_path) as f:
        new_text = f.read()
    assert "stationID: ZZ0009" in new_text
    assert "; target only comment line" in new_text
    assert cr.parse(dst_path).intensity_threshold == 41

    # The backup is of the original target, not of a copy of the source
    backups = [f for f in os.listdir(str(dst_dir)) if ".bak." in f]
    assert len(backups) == 1
    with open(str(dst_dir / backups[0])) as f:
        assert f.read() == text

    # The loaded config was not written, so its in-memory values stay as loaded
    assert pt.config.intensity_threshold == cr.parse(src_path).intensity_threshold


def testSaveConfigFailureNotReported(tmp_path, quietMessages, monkeypatch):
    """ A failed config write is reported as failed and leaves the in-memory config alone. """

    cfg_path = str(tmp_path / ".config")
    shutil.copy(TEMPLATE_CONFIG, cfg_path)
    pt = _fakePlateTool(cfg_path)
    orig_threshold = pt.config.intensity_threshold

    def failingWrite(*args, **kwargs):
        raise IOError("disk full")

    monkeypatch.setattr(SF, "updateConfigLines", failingWrite)

    dialog = types.SimpleNamespace(plate_tool=pt, _refreshAll=lambda: None)
    SF.CalibrationFilesDialog._saveFile(dialog, "Config", [str(tmp_path)])

    assert pt.config.intensity_threshold == orig_threshold
    assert any(m.get("message_type") == "error" and "failed" in m.get("message", "") for m in quietMessages)
    assert not any(m.get("title") == "Settings Saved" for m in quietMessages)


###################################################################################################
# PLATEPAR PARAMETER MANAGER
###################################################################################################

def testCoeffStashNotCarriedToNewPlatepar(plateTool, tmp_path):
    """ Loading another platepar must not fill its zero coefficients with the previous platepar's. """

    pt = plateTool
    pm = pt.tab.param_manager

    # Fill the stash from the current platepar with a flag round trip
    pm._stashCurrentCoeffs()
    assert any(pm._coeff_stash['x_fwd'].values())

    # Platepar B has zero distortion except one coefficient
    B = copy.deepcopy(pt.platepar)
    B.setDistortionType("radial5-odd", reset_params=True)
    B.x_poly_fwd[:] = 0
    B.x_poly_rev[:] = 0
    B.y_poly_fwd[:] = 0
    B.y_poly_rev[:] = 0
    B.x_poly_fwd[-2] = 0.123
    b_path = str(tmp_path / "B.cal")
    B.write(b_path)

    dialog = SF.CalibrationFilesDialog(pt)
    dialog._loadFile("Platepar", b_path)

    assert pt.platepar.distortion_type == "radial5-odd"
    assert np.allclose(pt.platepar.x_poly_fwd, B.x_poly_fwd)

    # A flag round trip on B restores B's values only
    pm._remapCoeffsWithStash('equal_aspect', not pt.platepar.equal_aspect)
    pm._remapCoeffsWithStash('equal_aspect', not pt.platepar.equal_aspect)
    assert np.allclose(pt.platepar.x_poly_fwd, B.x_poly_fwd)


@pytest.mark.parametrize("new_type", ["radial3-all", "radial4-all", "radial5-all", "radial3-odd", "radial7-odd",
                                      "radial9-odd"])
def testDistortionTypeSwitchKeepsArrayLength(plateTool, new_type):
    """ Switching the distortion type through the combo box always leaves poly_length coefficients, and a
        switch between the all and the odd powers does not restore coefficients of the other kind. """

    pt = plateTool
    pm = pt.tab.param_manager
    old_type = pt.platepar.distortion_type

    # A plain type change with the forced parameter reset, for comparison
    expected = copy.deepcopy(pt.platepar)
    expected.setDistortionType(new_type, reset_params=False)

    pm.distortion_type.setCurrentIndex(pt.platepar.distortion_type_list.index(new_type))

    pp = pt.platepar
    for arr in (pp.x_poly_fwd, pp.x_poly_rev, pp.y_poly_fwd, pp.y_poly_rev):
        assert len(arr) == pp.poly_length

    if new_type[-3:] != old_type[-3:]:
        assert np.allclose(pp.x_poly_fwd, expected.x_poly_fwd)
        assert np.allclose(pp.y_poly_rev, expected.y_poly_rev)


###################################################################################################
# STATE FILES
###################################################################################################

# Attributes of PlateTool.__init__ that are not in the state files saved by master
NEW_STATE_ATTRIBUTES = [
    "_original_catalog_file", "_original_band_ratios", "closest_planet_indx", "last_mask_dir",
    "platepar_modified", "mask_source_path", "flat_source_path", "dark_source_path",
    "_file_manager_custom_locations", "fig_astrometry", "fig_photometry", "auto_compute_sattracks",
    "show_spectral_type", "show_star_names", "apparent_mag_corr_enabled", "label_mag_limit",
    "show_constellations", "selected_stars_visible", "geo_marker_scale", "star_detection_override_enabled",
    "star_detection_override_data", "override_intensity_threshold", "override_neighborhood_size",
    "override_max_stars", "override_gamma", "override_segment_radius", "override_max_feature_ratio",
    "override_roundness_threshold", "_original_config_gamma", "mask_draw_mode", "mask_current_polygon",
    "mask_polygons", "mask_dragging_vertex", "mask_brush_mode", "mask_brush_radius", "mask_brush_painting",
    "mask_brush_erasing", "mask_brush_last_pos", "mask_paint_layer", "mask_brush_stroke_history",
    "mask_brush_max_undo", "flat_image_data", "mask_use_flat_background",
    "_show_calibration_dialog_on_start", "_show_file_manager_on_start",
]


def testLoadOldStateFile(plateTool, stationDir, qapp):
    """ A state file without the attributes introduced in this release loads and is usable. """

    from RMS.Pickling import loadPickle, savePickle

    plateTool.saveState()
    state = loadPickle(stationDir, 'skyFitMR_latest.state')
    for key in NEW_STATE_ATTRIBUTES:
        state.pop(key, None)
    savePickle(state, stationDir, 'old.state')

    # Load it at startup, the way SkyFit2 does it for a state file given on the command line
    pt = SF.PlateTool.__new__(SF.PlateTool)
    super(SF.PlateTool, pt).__init__()
    pt.loadState(stationDir, 'old.state')
    qapp.processEvents()

    pt.updateFileManagerButton()
    SF.CalibrationFilesDialog(pt)
    assert pt.isConfigModified() == plateTool.isConfigModified()
    assert pt.override_intensity_threshold == pt.config.intensity_threshold
    pt.updateCalstars()
    pt.nextImg()
    pt.changeMode('manualreduction')
    pt.changeMode('skyfit')

    pt.close()
    pt.deleteLater()


def testLoadStateMidSessionResetsMaskModes(plateTool, stationDir, qapp):
    """ Loading a state saved with the brush active does not leave the brush flags without their UI. """

    pt = plateTool
    tab = pt.tab
    mask_idx = tab.indexOf(tab.mask)
    tab.setCurrentIndex(mask_idx)
    tab.onTabBarClicked(mask_idx)
    tab.mask.brush_button.setChecked(True)
    pt.toggleMaskBrushMode()
    pt.brushStrokeBegin()
    pt.brushPaintAt(300, 300)
    pt.saveState()

    # Leave the mask tab, then load the state
    tab.setCurrentIndex(0)
    tab.onTabBarClicked(0)
    pt.loadState(stationDir, 'skyFitMR_latest.state')
    qapp.processEvents()

    assert not pt.mask_brush_mode
    assert not pt.mask_brush_painting
    assert not tab.mask.brush_button.isChecked()
    assert pt.img_frame.panning_enabled == (tab.currentIndex() != mask_idx)


###################################################################################################
# AUTOMATIC FITS
###################################################################################################

def _astrometryNetStandIn(pt):
    """ An astrometry.net answer made from the loaded platepar's own pointing.

    Arguments:
        pt: [PlateTool] The plate tool.

    Return:
        solution: [tuple] In the format returned by PlateTool._solveAstrometryNet.
    """

    from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP, rotationWrtStandard

    pp = pt.platepar
    t = pt.img_handle.currentTime()
    _, ra, dec, _ = xyToRaDecPP([t], [pp.X_res/2], [pp.Y_res/2], [1], pp, extinction_correction=False)

    return (ra[0], dec[0], rotationWrtStandard(pp), pp.F_scale, 53.7, 30.0, None, None)


def _plateparState(pp):
    """ The platepar values an automatic fit must not change when it fails. """

    return (pp.distortion_type, pp.equal_aspect, pp.asymmetry_corr, pp.force_distortion_centre,
            pp.refraction, pp.RA_d, pp.dec_d, pp.F_scale, tuple(pp.x_poly_fwd))


@pytest.mark.parametrize("failure", ["none", "raise"])
def testQuickAlignmentFailureRestoresPlatepar(plateTool, monkeypatch, failure):
    """ A rejected (None) or crashed NN fit makes the quick alignment fail and restores the platepar. """

    from RMS.Formats.Platepar import Platepar

    orig_fit = Platepar.fitAstrometry
    nn_calls = []

    def fakeFit(self, jd, img, cat, *args, **kwargs):

        # Emulate the NN exits after the RANSAC stages already changed the platepar
        if kwargs.get('use_nn_cost'):
            nn_calls.append(1)
            self.setDistortionType("radial7-odd", reset_params=False)
            self.RA_d += 1.0
            if failure == "raise":
                raise RuntimeError("simulated NN failure")
            return None

        return orig_fit(self, jd, img, cat, *args, **kwargs)

    pt = plateTool
    before = _plateparState(pt.platepar)
    lm_before = pt.cat_lim_mag

    monkeypatch.setattr(Platepar, "fitAstrometry", fakeFit)

    assert pt.tryQuickAlignment() is False
    assert nn_calls
    assert _plateparState(pt.platepar) == before
    assert pt.cat_lim_mag == lm_before


def testInitialParamsRejectedNNFitRestoresPlatepar(plateTool, monkeypatch):
    """ A rejected NN fit in the full recalibration does not fit the stale star_list of the platepar. """

    from RMS.Formats.Platepar import Platepar

    orig_fit = Platepar.fitAstrometry
    nn_calls = []

    def fakeFit(self, jd, img, cat, *args, **kwargs):
        if kwargs.get('use_nn_cost'):
            nn_calls.append(1)
            return None
        return orig_fit(self, jd, img, cat, *args, **kwargs)

    pt = plateTool

    # The example platepar carries the star list of the night it was made on
    assert len(pt.platepar.star_list) > 0
    before = _plateparState(pt.platepar)

    solution = _astrometryNetStandIn(pt)
    pt._solveAstrometryNet = lambda *a, **k: solution
    monkeypatch.setattr(Platepar, "fitAstrometry", fakeFit)

    assert pt.getInitialParamsAstrometryNet(upload_image=False) is None
    assert nn_calls
    assert len(pt.paired_stars) == 0
    assert _plateparState(pt.platepar) == before


def testRadialFitWithTooFewStarsFitsPointing(stationDir):
    """ A radial model fit with fewer stars than the distortion needs still fits the pointing. """

    import io
    import contextlib
    from RMS.Formats.Platepar import Platepar

    pp = Platepar()
    pp.read(os.path.join(stationDir, "platepar_cmn2010.cal"))
    assert pp.distortion_type.startswith("radial")

    star_list = np.array(pp.star_list)
    jd = star_list[0, 0]
    n_stars = pp.poly_length
    img_stars = star_list[:n_stars, 1:4]
    catalog_stars = star_list[:n_stars, 4:7]

    results = []
    for fit_only_pointing in (True, False):

        # Start from an offset pointing
        pp_fit = copy.deepcopy(pp)
        pp_fit.RA_d += 0.5
        pp_fit.updateRefAltAz()
        ra_start = pp_fit.RA_d

        with contextlib.redirect_stdout(io.StringIO()):
            pp_fit.fitAstrometry(jd, img_stars, catalog_stars, first_platepar_fit=True,
                                 fit_only_pointing=fit_only_pointing)

        results.append(pp_fit.RA_d)

    # The pointing moved, and the same way as a pointing-only fit
    assert abs(results[1] - ra_start) > 0.1
    assert results[1] == pytest.approx(results[0], abs=1e-6)


###################################################################################################
# STATION CHANGE
###################################################################################################

def testChangeStationResetsStationState(plateTool, secondStationDir):
    """ Calibration state of station A is not carried over to station B. """

    pt = plateTool

    # State that belongs to station A
    pt.flat_image_data = np.full((720, 1280), 7, np.uint8)
    pt.catalog_lm_tuned = True
    pt.tuned_cat_lim_mag = 4.3
    pt.dark = np.zeros_like(pt.img.data)
    pt.dark_source_path = os.path.join(pt.dir_path, 'dark.bmp')
    pt.star_detection_override_enabled = True
    pt.star_detection_override_data = {'FF_A.fits': [[1, 2, 3]]}
    pt._original_config_gamma = 0.7

    assert pt.changeStation(secondStationDir)

    assert pt.flat_image_data is None or not np.all(pt.flat_image_data == 7)
    assert pt.dark is None and pt.dark_source_path is None and pt.img.dark is None
    assert not pt.catalog_lm_tuned and not hasattr(pt, 'tuned_cat_lim_mag')
    assert not pt.star_detection_override_enabled and pt.star_detection_override_data == {}
    assert pt._original_config_gamma is None
    assert len(pt.unsuitable_stars) == 0


def testChangeStationCancelKeepsCurrentStation(plateTool, secondStationDir):
    """ Cancelling the platepar picker leaves the current station fully loaded. """

    pt = plateTool
    before = (pt.dir_path, pt.config.config_file_name, pt.platepar_file, pt.img_handle.dir_path,
              id(pt.catalog_stars), id(pt.mask))

    pt._findPlatepar = lambda *a, **k: None
    assert pt.changeStation(secondStationDir) is False

    after = (pt.dir_path, pt.config.config_file_name, pt.platepar_file, pt.img_handle.dir_path,
             id(pt.catalog_stars), id(pt.mask))
    assert after == before


###################################################################################################
# KEYBOARD
###################################################################################################

def _pressKey(pt, key, modifiers=None):
    """ Send a key press to the plate tool.

    Arguments:
        pt: [PlateTool] The plate tool.
        key: [Qt.Key] Key to press.

    Keyword arguments:
        modifiers: [Qt.KeyboardModifier] Modifiers of the event. None (default) for no modifier.
    """

    from pyqtgraph.Qt import QtCore, QtGui

    if modifiers is None:
        modifiers = QtCore.Qt.KeyboardModifier.NoModifier

    event = QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, key, modifiers)
    pt.keyPressEvent(event)


@pytest.mark.parametrize("key_name", ["Key_W", "Key_A", "Key_Q", "Key_E", "Key_Up", "Key_1", "Key_9"])
def testKeyboardEditMarksPlateparModified(plateTool, key_name):
    """ Keyboard edits of the platepar trigger the unsaved changes prompt. """

    from pyqtgraph.Qt import QtCore

    pt = plateTool
    assert pt.mode == 'skyfit'
    pt.platepar_modified = False

    _pressKey(pt, getattr(QtCore.Qt.Key, key_name))

    assert pt.platepar_modified


def testKeyboardViewKeysKeepPlateparUnmodified(plateTool):
    """ Keys that do not edit the platepar do not mark it modified. """

    from pyqtgraph.Qt import QtCore

    pt = plateTool
    pt.platepar_modified = False

    for key in (QtCore.Qt.Key.Key_M, QtCore.Qt.Key.Key_H, QtCore.Qt.Key.Key_Right, QtCore.Qt.Key.Key_Plus):
        _pressKey(pt, key)

    assert not pt.platepar_modified


###################################################################################################
# MASK EDITING
###################################################################################################

class _FakeMouseEvent(object):
    """ Minimal mouse press event at an image position, for PlateTool.onMousePressed. """

    def __init__(self, pt, x, y, button):
        from pyqtgraph.Qt import QtCore
        self._pos = pt.img_frame.mapViewToScene(QtCore.QPointF(x + 0.5, y + 0.5))
        self._button = button

    def button(self):
        return self._button

    def scenePos(self):
        return self._pos


def _loadHorizonMask(pt, dir_path):
    """ Load a mask with a horizon polygon along the bottom of the image. """

    import cv2

    mask = np.full((720, 1280), 255, np.uint8)
    cv2.fillPoly(mask, [np.array([[0, 600], [400, 550], [900, 620], [1279, 580], [1279, 719], [0, 719]])], 0)
    mask_path = os.path.join(dir_path, "mask_horizon.bmp")
    cv2.imwrite(mask_path, mask)
    pt.loadMaskFromFile(mask_path)

    return mask


def _vertexCount(pt):
    return sum(len(poly) for poly in pt.mask_polygons)


def testMaskVerticesOnlyEditedOnMaskTab(plateTool, stationDir, qapp):
    """ A right click near a mask vertex only deletes it on the Mask tab. """

    from pyqtgraph.Qt import QtCore

    pt = plateTool
    pt.show()
    pt.resize(1600, 1000)
    qapp.processEvents()

    _loadHorizonMask(pt, stationDir)
    n_vertices = _vertexCount(pt)
    right = QtCore.Qt.MouseButton.RightButton

    # On another tab, and in the manual reduction mode, the vertex is left alone
    vx, vy = pt.mask_polygons[0][1]
    assert not pt.isMaskTabCurrent()
    pt.onMousePressed(_FakeMouseEvent(pt, vx + 3, vy + 3, right))
    assert _vertexCount(pt) == n_vertices

    # On the Mask tab it is deleted
    mask_idx = pt.tab.indexOf(pt.tab.mask)
    pt.tab.setCurrentIndex(mask_idx)
    pt.tab.onTabBarClicked(mask_idx)
    pt.onMousePressed(_FakeMouseEvent(pt, vx + 3, vy + 3, right))
    assert _vertexCount(pt) == n_vertices - 1


def testStartupMaskKeptWithoutMaskFile(qapp, stationDir, quietMessages):
    """ A mask given at startup (--mask, RMS root) is edited instead of discarded when the data directory
        has no mask.bmp. """

    import cv2
    from RMS.Routines.MaskImage import MaskStructure

    os.remove(os.path.join(stationDir, "mask.bmp"))

    mask_img = np.full((720, 1280), 255, np.uint8)
    cv2.rectangle(mask_img, (0, 600), (1279, 719), 0, -1)

    config = cr.loadConfigFromDirectory('.config', stationDir)
    pt = SF.PlateTool(stationDir, config, mask=MaskStructure(mask_img))
    qapp.processEvents()

    try:
        assert len(pt.mask_polygons) > 0
        assert np.array_equal(pt.mask.img, mask_img)
        assert np.array_equal(pt.generateMaskImage(), mask_img)
    finally:
        pt.close()
        pt.deleteLater()


def testMaskLoadClearsBrushHistory(plateTool, stationDir):
    """ Loading a mask always drops the brush undo history of the previous mask. """

    pt = plateTool
    pt.brushStrokeBegin()
    pt.brushPaintAt(300, 300)
    assert pt.mask_brush_stroke_history

    # A clean rectangle decomposes without raster residuals
    import cv2
    mask_img = np.full((720, 1280), 255, np.uint8)
    cv2.rectangle(mask_img, (0, 600), (1279, 719), 0, -1)
    mask_path = os.path.join(stationDir, "rect.bmp")
    cv2.imwrite(mask_path, mask_img)

    assert pt.loadMaskFromFile(mask_path)
    assert pt.mask_paint_layer is None
    assert pt.mask_brush_stroke_history == []


def testUnreadableMaskNotReportedLoaded(plateTool, stationDir):
    """ A mask file that cannot be read does not become the mask source. """

    pt = plateTool
    source_before = pt.mask_source_path
    polygons_before = copy.deepcopy(pt.mask_polygons)

    bad_path = os.path.join(stationDir, "broken.bmp")
    with open(bad_path, 'w') as f:
        f.write("not an image")

    assert pt.loadMaskFromFile(bad_path) is False
    assert pt.mask_source_path == source_before
    assert pt.mask_polygons == polygons_before


def _openMaskTab(pt):
    """ Open the Mask tab the way a click on its tab does. """

    mask_idx = pt.tab.indexOf(pt.tab.mask)
    pt.tab.setCurrentIndex(mask_idx)
    pt.tab.onTabBarClicked(mask_idx)


@pytest.mark.parametrize("start_mode", ["skyfit", "manualreduction"])
def testModeSwitchKeepsMaskTabAndLeavesBrush(plateTool, qapp, start_mode):
    """ A mode switch on the Mask tab keeps the Mask tab open, leaves the brush mode and keeps the
        overlays hidden. """

    pt = plateTool
    pt.show()
    qapp.processEvents()

    pt.changeMode(start_mode)
    _openMaskTab(pt)
    pt.tab.mask.brush_button.setChecked(True)
    pt.toggleMaskBrushMode()
    assert pt.mask_brush_mode

    other_mode = "manualreduction" if start_mode == "skyfit" else "skyfit"
    pt.changeMode(other_mode)
    qapp.processEvents()

    assert pt.tab.currentWidget() is pt.tab.mask
    assert pt.tab.index == pt.tab.currentIndex()
    assert not pt.mask_brush_mode
    assert not pt.brush_cursor.isVisible()
    assert not pt.img_frame.panning_enabled
    assert not pt.pick_marker.isVisible()
    assert not pt.sel_cat_star_markers.isVisible()


def testModeSwitchFromRemovedTabShowsExistingTab(plateTool, qapp):
    """ Switching to manual reduction from a skyfit-only tab does not land on the Mask tab unannounced. """

    pt = plateTool
    station_idx = pt.tab.indexOf(pt.tab.geolocation)
    pt.tab.setCurrentIndex(station_idx)
    pt.tab.onTabBarClicked(station_idx)

    pt.changeMode("manualreduction")

    assert pt.tab.currentWidget() is not pt.tab.mask
    assert pt.img_frame.panning_enabled


def testInvertMaskIsExact(plateTool, stationDir):
    """ Inverting swaps masked and unmasked exactly, also for masks with holes. """

    import cv2

    pt = plateTool

    # An all-sky style mask: a masked ring around an unmasked disc, plus a masked tree inside
    mask_img = np.zeros((720, 1280), np.uint8)
    cv2.circle(mask_img, (640, 360), 340, 255, -1)
    cv2.rectangle(mask_img, (600, 600), (660, 690), 0, -1)
    mask_path = os.path.join(stationDir, "allsky.bmp")
    cv2.imwrite(mask_path, mask_img)
    assert pt.loadMaskFromFile(mask_path)
    assert np.array_equal(pt.generateMaskImage(), mask_img)

    pt.invertMaskPolygons()
    assert np.array_equal(pt.generateMaskImage(), 255 - mask_img)

    pt.invertMaskPolygons()
    assert np.array_equal(pt.generateMaskImage(), mask_img)


###################################################################################################
# STAR DETECTION TAB
###################################################################################################

def testStarDetectionSlidersKeepConfigValues(qapp, stationDir, quietMessages):
    """ Loading a config with values outside the slider ranges or between slider steps keeps the exact
        config values as the overrides and does not mark the config modified. """

    cfg_path = os.path.join(stationDir, ".config")
    with open(cfg_path) as f:
        lines = f.readlines()
    lines = SF.updateConfigLines(lines, {"StarExtraction": {
        "max_stars": "8000", "max_feature_ratio": "0.57999", "neighborhood_size": "57",
        "roundness_threshold": "0.333", "segment_radius": "25"}})
    with open(cfg_path, 'w') as f:
        f.writelines(lines)

    config = cr.loadConfigFromDirectory('.config', stationDir)
    pt = SF.PlateTool(stationDir, config)
    qapp.processEvents()

    try:
        assert pt.override_max_stars == 8000
        assert pt.override_max_feature_ratio == pytest.approx(0.57999)
        assert pt.override_neighborhood_size == 57
        assert pt.override_roundness_threshold == pytest.approx(0.333)
        assert pt.override_segment_radius == 25
        assert not pt.isConfigModified()

        # The sliders show the values instead of clamping them
        sd = pt.tab.star_detection
        assert sd.max_stars_slider.value() == 8000
        assert sd.max_feature_ratio_slider.value() == 58
        assert sd.neighborhood_size_slider.value() == 57
    finally:
        pt.close()
        pt.deleteLater()


###################################################################################################
# BEST FRAME PLACEHOLDER
###################################################################################################

def testValidFFNameRejectsPngWithoutFormat():
    """ The RMS directory scanners (validFFName without fmt) never take a PNG for an FF file. """

    from RMS.Formats.FFfile import validFFName
    from RMS.Formats.FrameInterface import validFFImageName

    name = "FF_AU000A_20250817_194808_532_0524288_placeholder.png"

    assert not validFFName(name)
    assert validFFName(name, fmt='png')
    assert validFFName("FF_AU000A_20250817_194808_532_0524288.fits")

    # The FF image handle opens the placeholders explicitly
    assert validFFImageName(name)


def testBestFramePlaceholderNotLeftInDataFolder(plateTool, monkeypatch):
    """ The auto fit on a frame without an image uses a placeholder that is not written to the data
        folder, and gets the re-detected stars of that frame. """

    from pyqtgraph.Qt import QtWidgets

    pt = plateTool
    real_ff = pt.img_handle.name()
    missing_ff = "FF_AU000A_20250817_200000_000_0600000.fits"

    stars = pt.calstars[real_ff]
    pt.calstars[missing_ff] = stars
    pt.star_detection_override_data[missing_ff] = stars

    # The missing frame is the best one
    monkeypatch.setattr(SF, "selectBestFrame",
                        lambda data, *a, **k: (missing_ff if missing_ff in data else real_ff, 1.0, {}))

    # Choose the auto fit in the dialog, and skip the fit itself
    def clickedButton(self):
        for button in self.buttons():
            if button.text().startswith("Auto Fit"):
                return button
        return None

    monkeypatch.setattr(QtWidgets.QMessageBox, "clickedButton", clickedButton)
    fits = []
    pt.autoFitAstrometryNet = lambda: fits.append(pt.img_handle.name())

    pt.findBestFrame()

    placeholder_name = "FF_AU000A_20250817_200000_000_0600000_placeholder.png"
    assert fits == [placeholder_name]
    assert not any("_placeholder" in f for f in os.listdir(pt.dir_path))
    assert pt.star_detection_override_data[placeholder_name] is stars

    # The placeholder can be shown again after navigating away
    pt.nextImg(n=1)
    pt.nextImg(n=-1)
    assert pt.img_handle.name() == placeholder_name
    assert np.all(pt.img_handle.loadChunk().avepixel <= 40)


###################################################################################################
# CATALOG
###################################################################################################

def testCatalogProbeKeepsLoadedCatalog(plateTool):
    """ A temporary catalog read at another LM leaves the loaded catalog and its per-star data alone. """

    pt = plateTool
    catalog = pt.catalog_stars
    names = pt.catalog_stars_common_names
    n_stars = len(catalog)

    deep = pt.readCatalogStars(pt.cat_lim_mag + 2.0)

    assert len(deep) > n_stars
    assert pt.catalog_stars is catalog
    assert pt.catalog_stars_common_names is names
    assert (names is None) or (len(names) == n_stars)


def testOptimalCatalogLMSearchKeepsLoadedCatalog(plateTool):
    """ The catalog LM search of the tuner probes other LMs without replacing the loaded catalog. """

    from RMS.Astrometry.Conversions import date2JD

    pt = plateTool
    catalog = pt.catalog_stars
    names = pt.catalog_stars_common_names

    stars = np.array(pt.calstars[pt.img_handle.name()])
    jd = date2JD(*pt.img_handle.currentTime())
    pt._findOptimalCatalogLM(jd, stars[:, 1], stars[:, 0], target_matches=len(stars)//2)

    assert pt.catalog_stars is catalog
    assert pt.catalog_stars_common_names is names


@pytest.mark.parametrize("method", ["quickAlign", "autoFitAstrometryNet"])
def testAutoFitRestoresConfigCatalogLM(plateTool, method):
    """ The LM the balancing sets in the config for alignPlatepar does not outlive the fit. """

    pt = plateTool
    config_lm = pt.config.catalog_mag_limit

    def fakeBalance():
        pt.cat_lim_mag = config_lm + 1.5
        pt.config.catalog_mag_limit = config_lm + 1.5
        return True

    pt.balanceCatalogMagnitude = fakeBalance
    pt.tryQuickAlignment = lambda *a, **k: True
    pt.paired_stars = SF.PairedStars()

    getattr(pt, method)()

    assert pt.config.catalog_mag_limit == config_lm


def testAbortedTuningRestoresConfig(plateTool):
    """ An aborted star detection tuning leaves the detection parameters of the config as loaded. """

    pt = plateTool
    names = ["max_stars", "intensity_threshold", "segment_radius", "max_feature_ratio", "roundness_threshold"]

    # Overrides that differ from the config, which the probes patch into the config
    pt.override_max_feature_ratio = pt.config.max_feature_ratio + 0.1
    pt.override_roundness_threshold = pt.config.roundness_threshold + 0.1
    before = {name: getattr(pt.config, name) for name in names}

    # Abort at the final success gate, after all the detection probes ran
    pt._findOptimalCatalogLM = lambda *a, **k: None

    pt.tuneStarDetection()

    assert {name: getattr(pt.config, name) for name in names} == before


###################################################################################################
# PLANET MAGNITUDES
###################################################################################################

@pytest.mark.parametrize("body_name, time_str, published_mag", [
    ('jupiter', '2022-09-26 20:00', -2.9),     # Opposition
    ('mars', '2020-10-13 23:00', -2.6),        # Opposition
    ('saturn', '2022-08-14 18:00', 0.3),       # Opposition, rings open ~14 deg
    ('venus', '2020-04-28 12:00', -4.7),       # Greatest brilliancy
    ('venus', '2021-03-26 07:00', -3.9),       # Superior conjunction
    ('mercury', '2021-01-24 00:00', -0.6),     # Greatest eastern elongation
    ('moon', '2022-01-17 23:48', -12.7),       # Full Moon
    ('moon', '2022-01-09 18:11', -10.1),       # First quarter
    ('neptune', '2022-09-16 12:00', 7.8),      # Opposition
])
def testSolarSystemMagnitudes(body_name, time_str, published_mag):
    """ The planet and Moon magnitudes agree with published values to 0.2 mag. """

    if not SF.ASTROPY_AVAILABLE:
        pytest.skip("astropy is not available")

    from astropy.coordinates import get_body, get_sun, EarthLocation
    from astropy.time import Time
    import astropy.units as u

    loc = EarthLocation(lat=45*u.deg, lon=15*u.deg, height=0*u.m)
    t = Time(time_str)

    mag = SF.computeSolarSystemMagnitude(body_name, get_body(body_name, t, loc), get_sun(t), t)

    assert mag == pytest.approx(published_mag, abs=0.2)


def testPlanetPhaseGeometry():
    """ Distances and phase angle of a body at quadrature seen from 1 AU. """

    r, delta, alpha = SF.planetPhaseGeometry([0.0, 1.0, 0.0], [1.0, 0.0, 0.0])

    assert r == pytest.approx(np.sqrt(2))
    assert delta == pytest.approx(1.0)
    assert alpha == pytest.approx(45.0)


def testConfigMagLimitQuickAlignUsesImageTime(plateTool, monkeypatch):
    """ The config-LM quick align test uses the same image time and stars as tryQuickAlignment. """

    pt = plateTool
    calls = []

    def fakeAlign(config, platepar, calstars_time, detected_stars, **kwargs):
        calls.append((list(calstars_time), len(detected_stars)))
        return platepar, config.catalog_mag_limit

    monkeypatch.setattr(SF, "alignPlatepar", fakeAlign)

    pt.testQuickAlignWithConfigMagLimit()

    assert calls == [(list(pt.img_handle.currentTime()), len(pt.calstars[pt.img_handle.name()]))]


###################################################################################################
# PHOTOMETRY
###################################################################################################

def testPhotometryExcludeListFallback():
    """ The exclusions are relaxed step by step until at least 3 stars remain. """

    # Enough stars left without the saturated and the variable ones
    sat = [True, False, False, False, False, False]
    var = [False, True, False, False, False, False]
    assert SF.photometryExcludeList(sat, var) == [True, True, False, False, False, False]

    # The variable star has to be used
    sat = [True, True, False, False, False]
    var = [False, False, True, False, False]
    assert SF.photometryExcludeList(sat, var) == [True, True, False, False, False]

    sat = [True, True, True, False]
    assert SF.photometryExcludeList(sat, [False]*4) == [False]*4


def testBandRatioFitWithSaturatedStars(plateTool, monkeypatch, capsys):
    """ The band ratio fit falls back to the saturated stars like photometry() instead of reporting a
        ratio fitted to no stars. """

    import matplotlib.pyplot as plt

    pt = plateTool
    if pt.catalog_stars_bp is None:
        pytest.skip("The example catalog has no BP/RP magnitudes")

    monkeypatch.setattr(plt, "show", lambda *a, **k: None)

    # Pair the brightest catalog stars in the image, all flagged saturated
    from RMS.Astrometry.Conversions import date2JD

    jd = date2JD(*pt.img_handle.currentTime())
    x, y, _ = SF.getCatalogStarsImagePositions(pt.catalog_stars, jd, pt.platepar)
    in_image = (x > 50) & (x < pt.platepar.X_res - 50) & (y > 50) & (y < pt.platepar.Y_res - 50)
    idx = np.where(in_image)[0]
    idx = idx[np.argsort(pt.catalog_stars[idx, 2])][:8]

    pt.paired_stars = SF.PairedStars()
    for i in idx:
        ra, dec, mag = pt.catalog_stars[i][:3]
        pt.paired_stars.addPair(x[i], y[i], 2.5, 10**(-0.4*(mag - 12.0)), SF.CatalogStar(ra, dec, mag),
                                snr=50, saturated=True)

    pt.fitBandRatio()

    out = capsys.readouterr().out
    assert "Fit stddev: inf" not in out
    assert "Fit stddev:" in out


###################################################################################################
# FILE MANAGER SAVES
###################################################################################################

def testFileManagerSavesLoadedPlateparFile(plateTool, stationDir):
    """ The File Manager saves the platepar into the file it was loaded from, not config.platepar_name. """

    pt = plateTool
    default_path = os.path.join(stationDir, pt.config.platepar_name)
    other_path = os.path.join(stationDir, "other.cal")
    shutil.copy(default_path, other_path)
    with open(default_path) as f:
        default_before = f.read()

    pt.loadPlatepar(update=True, platepar_file=other_path)
    pt.platepar.RA_d += 1.0

    dialog = SF.CalibrationFilesDialog(pt)
    dialog._saveFile("Platepar", [stationDir])

    with open(default_path) as f:
        assert f.read() == default_before

    from RMS.Formats.Platepar import Platepar
    saved = Platepar()
    saved.read(other_path)
    assert saved.RA_d == pytest.approx(pt.platepar.RA_d)
    assert pt.platepar_file == other_path


def testFileManagerMaskSaveUpdatesDetectionMask(plateTool, stationDir):
    """ Saving the mask through the File Manager makes it the mask used by star detection. """

    pt = plateTool
    pt.mask_polygons = [[(0, 0), (200, 0), (200, 200), (0, 200)]]
    expected = pt.generateMaskImage()

    dialog = SF.CalibrationFilesDialog(pt)
    dialog._saveFile("Mask", [stationDir])

    assert np.array_equal(pt.mask.img, expected)


def testArrowKeysLeftToSlidersAndCombos(plateTool):
    """ The arrow keys are not taken from focused sliders, combo boxes and lists. """

    from pyqtgraph.Qt import QtCore, QtGui

    pt = plateTool
    right = QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_Right,
                            QtCore.Qt.KeyboardModifier.NoModifier)

    sd = pt.tab.star_detection
    combo = pt.tab.param_manager.distortion_type
    for widget in (sd.max_stars_slider, combo, combo.view()):
        assert pt.eventFilter(widget, right) is False

    # Elsewhere they still navigate the images
    assert pt.eventFilter(pt.tab.hist, right) is True


@pytest.mark.parametrize("digit, dist_type", [(1, "poly3+radial"), (2, "poly3+radial3"), (3, "radial3-odd"),
                                              (4, "radial5-odd"), (5, "radial7-odd"), (6, "radial9-odd"),
                                              (7, None)])
def testCtrlDigitDistortionShortcuts(plateTool, monkeypatch, digit, dist_type):
    """ CTRL + digit sets the distortion type of the keyboard reference and nothing else; unbound
        combinations do nothing (no IndexError, no coefficient edit). """

    from pyqtgraph.Qt import QtCore, QtWidgets

    pt = plateTool
    ctrl = QtCore.Qt.KeyboardModifier.ControlModifier
    monkeypatch.setattr(QtWidgets.QApplication, "keyboardModifiers", staticmethod(lambda: ctrl))
    monkeypatch.setattr(QtWidgets.QApplication, "queryKeyboardModifiers", staticmethod(lambda: ctrl))

    before_type = pt.platepar.distortion_type
    before_x = np.array(pt.platepar.x_poly_fwd)

    _pressKey(pt, getattr(QtCore.Qt.Key, "Key_{:d}".format(digit)), ctrl)

    if dist_type is None:
        assert pt.platepar.distortion_type == before_type
        assert np.array_equal(pt.platepar.x_poly_fwd, before_x)

    else:
        assert pt.platepar.distortion_type == dist_type

        # setDistortionType resets the coefficients, the X offset key must not add 0.5 on top
        expected = copy.deepcopy(pt.platepar)
        expected.setDistortionType(dist_type)
        assert np.array_equal(pt.platepar.x_poly_fwd, expected.x_poly_fwd)


def testHelpFromMaskTabLeavesMaskModes(plateTool, qapp):
    """ Opening the Help from the Mask tab runs the Mask tab leave logic. """

    pt = plateTool
    pt.show()
    qapp.processEvents()

    _openMaskTab(pt)
    pt.tab.mask.brush_button.setChecked(True)
    pt.toggleMaskBrushMode()
    assert pt.mask_brush_mode and not pt.img_frame.panning_enabled

    pt.openHelp()

    assert pt.tab.currentWidget() is pt.tab.help
    assert not pt.mask_brush_mode
    assert not pt.brush_cursor.isVisible()
    assert pt.img_frame.panning_enabled


def testNextImageKeepsMaskFlatBackground(plateTool):
    """ Changing the image on the Mask tab keeps the flat shown as the mask editing background. """

    pt = plateTool
    flat = np.full((720, 1280), 77, np.uint8)
    pt.flat_image_data = flat
    pt.mask_use_flat_background = True
    _openMaskTab(pt)
    assert np.all(pt.img.image == 77)

    pt.nextImg(n=1)

    assert np.all(pt.img.image == 77)


def testCtrlZOnMaskTabNeverFits(plateTool, monkeypatch):
    """ CTRL + Z on the Mask tab with nothing to undo does not refit the plate. """

    from pyqtgraph.Qt import QtCore, QtWidgets

    pt = plateTool
    ctrl = QtCore.Qt.KeyboardModifier.ControlModifier
    monkeypatch.setattr(QtWidgets.QApplication, "keyboardModifiers", staticmethod(lambda: ctrl))
    monkeypatch.setattr(QtWidgets.QApplication, "queryKeyboardModifiers", staticmethod(lambda: ctrl))

    fits = []
    pt.fitPickedStars = lambda *a, **k: fits.append(1)

    _openMaskTab(pt)
    pt.mask_brush_stroke_history = []
    _pressKey(pt, QtCore.Qt.Key.Key_Z, ctrl)

    assert fits == []


def testManualReductionWithoutData(qapp, quietMessages, monkeypatch):
    """ Without data the manual reduction cannot be entered and the mode stays skyfit. """

    monkeypatch.setattr(SF.PlateTool, "showCalibrationFilesDialog", lambda self, *a, **k: None)
    pt = SF.PlateTool()
    qapp.processEvents()

    try:
        assert not pt.hasData()
        assert not pt.manualreduction_button.isEnabled()

        pt.changeMode('manualreduction')
        assert pt.mode == 'skyfit'
    finally:
        pt.close()
        pt.deleteLater()


@pytest.mark.parametrize("force_centre", [False, True])
def testDistortionResetToZeroForcedCentre(plateTool, force_centre):
    """ Reset to zero keeps the radial distortion centre only when it is in the coefficient array. """

    pt = plateTool
    pp = pt.platepar
    assert pp.distortion_type.startswith("radial")

    pp.remapCoeffsForFlagChange('force_distortion_centre', force_centre)
    for name in ('x_poly_fwd', 'x_poly_rev', 'y_poly_fwd', 'y_poly_rev'):
        getattr(pp, name)[:] = 0.123

    dialog = pt.tab.param_manager.distortion_dialog
    dialog.updatePlatepar(pp)
    dialog.resetToZero()

    n_centre = 0 if force_centre else 2
    assert np.all(pp.x_poly_fwd[:n_centre] == 0.123)
    assert not np.any(pp.x_poly_fwd[n_centre:])
    assert not np.any(pp.y_poly_fwd)


###################################################################################################
# CALIBRATION REPORT
###################################################################################################

def testCalibrationReportLMExcludesCappedAndSaturated():
    """ The limiting magnitude fit of the calibration report leaves out the S/N-capped and saturated
        stars, and works with CALSTARS files without the saturation column. """

    from Utils.CalibrationReport import calstarsLimitingMagnitudeExcludeMask as limitingMagnitudeExcludeMask

    # Y, X, IntensSum, Ampltd, FWHM, BgLvl, SNR, NSatPx
    stars = np.array([
        [10, 10, 1000, 100, 2.5, 20, 99.99, 0],
        [20, 20, 800, 90, 2.5, 20, 50.0, 3],
        [30, 30, 500, 60, 2.5, 20, 30.0, 0],
    ])

    assert list(limitingMagnitudeExcludeMask(stars)) == [True, True, False]
    assert list(limitingMagnitudeExcludeMask(stars[:, :7])) == [True, False, False]


@pytest.mark.parametrize("forced_centre_reported", [False, True])
def testForcedCentreRoundTripRestoresFittedCentre(plateTool, monkeypatch, forced_centre_reported):
    """ Forcing the distortion centre and releasing it again gives back the fitted free centre, whether
        extractRadialCoeffs reports 0 (old) or the forced 0.5/(res/2) value (new) for a forced centre. """

    from RMS.Formats.Platepar import Platepar

    if forced_centre_reported:

        orig_extract = Platepar.extractRadialCoeffs

        def extractWithForcedCentre(self, x_poly=None):
            coeffs = orig_extract(self, x_poly)
            if (coeffs is not None) and self.force_distortion_centre:
                coeffs['x0'] = 0.5/(self.X_res/2.0)
                coeffs['y0'] = 0.5/(self.Y_res/2.0)
            return coeffs

        monkeypatch.setattr(Platepar, "extractRadialCoeffs", extractWithForcedCentre)

    pt = plateTool
    pm = pt.tab.param_manager
    pp = pt.platepar
    assert pp.distortion_type.startswith("radial")

    pp.remapCoeffsForFlagChange('force_distortion_centre', False)
    for name in ('x_poly_fwd', 'x_poly_rev'):
        getattr(pp, name)[0] = 0.05
        getattr(pp, name)[1] = -0.03
    k_before = np.array(pp.x_poly_fwd[2:])

    pm._remapCoeffsWithStash('force_distortion_centre', True)
    pm._remapCoeffsWithStash('force_distortion_centre', False)

    for name in ('x_poly_fwd', 'x_poly_rev'):
        assert getattr(pp, name)[0] == pytest.approx(0.05)
        assert getattr(pp, name)[1] == pytest.approx(-0.03)
    assert np.allclose(pp.x_poly_fwd[2:], k_before)


@pytest.mark.parametrize("at_startup", [False, True])
def testStateOnPlaceholderReopensOnPlaceholder(plateTool, stationDir, qapp, at_startup):
    """ A state saved while the in-memory best frame placeholder is shown reopens on it, at its time. """

    pt = plateTool
    placeholder_name = "FF_AU000A_20250817_200000_000_0600000_placeholder.png"
    pt.calstars[placeholder_name] = pt.calstars[pt.img_handle.name()]
    idx = pt.img_handle.addMemoryFF(placeholder_name,
                                    SF.ffStructFromImage(np.full((720, 1280), 24, np.uint8)))
    pt.nextImg(n=idx - pt.img_handle.current_ff_index)
    saved_time = pt.img_handle.currentTime()
    assert pt.img_handle.name() == placeholder_name

    pt.saveState()

    if at_startup:
        pt2 = SF.PlateTool.__new__(SF.PlateTool)
        super(SF.PlateTool, pt2).__init__()
    else:
        config = cr.loadConfigFromDirectory('.config', stationDir)
        pt2 = SF.PlateTool(stationDir, config)

    try:
        pt2.loadState(stationDir, 'skyFitMR_latest.state')
        qapp.processEvents()

        assert pt2.img_handle.name() == placeholder_name
        assert pt2.img_handle.currentTime() == saved_time
        assert np.all(pt2.img_handle.loadChunk().avepixel == 24)
        assert not any("_placeholder" in f for f in os.listdir(stationDir))
    finally:
        pt2.close()
        pt2.deleteLater()


@pytest.mark.parametrize("failing_step", ["detectInputType", "loadCatalogStars"])
def testChangeStationFailureKeepsCurrentStationState(plateTool, secondStationDir, failing_step):
    """ A station change that fails part way keeps the calibration state of the current station. """

    pt = plateTool
    flat, dark = object(), np.zeros_like(pt.img.data)
    pt.flat_struct = flat
    pt.img.flat_struct = flat
    pt.dark = dark
    pt.catalog_lm_tuned = True
    pt.tuned_cat_lim_mag = 6.1
    pt.star_detection_override_enabled = True
    pt.star_detection_override_data = {'FF_A.fits': [[1, 2, 3]]}
    unsuitable = pt.unsuitable_stars

    def fail(*args, **kwargs):
        raise RuntimeError("simulated failure")

    setattr(pt, failing_step, fail)

    assert pt.changeStation(secondStationDir) is False

    assert pt.flat_struct is flat and pt.img.flat_struct is flat
    assert pt.dark is dark
    assert pt.catalog_lm_tuned and pt.tuned_cat_lim_mag == 6.1
    assert pt.star_detection_override_enabled
    assert pt.star_detection_override_data == {'FF_A.fits': [[1, 2, 3]]}
    assert pt.unsuitable_stars is unsuitable
    assert pt.tab.star_detection.use_override_checkbox.isChecked()


@pytest.mark.parametrize("reset", ["resetToZero", "resetDistortion", "changeDistortionType", "firstFit"])
def testExplicitResetDropsCoeffStash(plateTool, reset):
    """ After an explicit distortion reset, a type change does not bring the old coefficients back. """

    pt = plateTool
    pm = pt.tab.param_manager
    pp = pt.platepar
    assert pp.distortion_type == "radial9-odd"

    # radial9 -> radial7 stashes the highest term
    pm.distortion_type.setCurrentIndex(pp.distortion_type_list.index("radial7-odd"))
    assert any(pm._coeff_stash['x_fwd'].values())

    if reset == "resetToZero":
        pm.distortion_dialog.updatePlatepar(pp)
        pm.distortion_dialog.resetToZero()
    elif reset == "resetDistortion":
        pt.resetDistortion()
    elif reset == "changeDistortionType":
        pt.dist_type_index = pp.distortion_type_list.index("radial7-odd")
        pt.changeDistortionType()
    else:
        pt.first_platepar_fit = True
        pt.platepar.fitAstrometry = lambda *a, **k: None
        pt.paired_stars = SF.PairedStars()
        for x, y, intens, ra, dec, mag in np.array(pp.star_list)[:10, 1:]:
            pt.paired_stars.addPair(x, y, 2.0, intens, SF.CatalogStar(ra, dec, mag))
        pt.fitPickedStars()

    assert not any(any(stash.values()) for stash in pm._coeff_stash.values())

    # Back to radial9: no pre-reset coefficient comes back into the zero slots
    before = np.array(pp.x_poly_fwd)
    pm.distortion_type.setCurrentIndex(pp.distortion_type_list.index("radial9-odd"))
    assert np.allclose(pp.x_poly_fwd[:len(before)], before)
    assert pp.x_poly_fwd[-1] == 0.0


def testFailedQuickAlignmentKeepsPlateparObjectAndStash(plateTool, monkeypatch):
    """ A failed quick alignment restores the platepar into the same object, so the Fit Parameters
        stash and the distortion dialog stay attached to it. """

    from RMS.Formats.Platepar import Platepar

    pt = plateTool
    pm = pt.tab.param_manager
    pp = pt.platepar

    pm.distortion_type.setCurrentIndex(pp.distortion_type_list.index("radial7-odd"))
    stash = copy.deepcopy(pm._coeff_stash)
    assert any(stash['x_fwd'].values())

    # Every fit is rejected
    monkeypatch.setattr(Platepar, "fitAstrometry", lambda self, *a, **k: None)

    assert pt.tryQuickAlignment() is False
    assert pt.platepar is pp
    pm._checkCoeffStashOwner()
    assert pm._coeff_stash == stash
    assert pm.distortion_dialog.platepar is pp


def testConfigWriteKeepsCRLFAndEncoding(tmp_path, quietMessages):
    """ A CRLF config with non-ASCII text keeps every byte except the updated lines. """

    with open(TEMPLATE_CONFIG, encoding='utf-8') as f:
        text = f.read()
    text = text.replace("stationID: XX0001", "stationID: XX0001 ; Križevci, Čakovec")
    raw = text.replace("\n", "\r\n").encode('utf-8')

    cfg_path = str(tmp_path / ".config")
    with open(cfg_path, 'wb') as f:
        f.write(raw)

    pt = _fakePlateTool(cfg_path)
    pt._writeStarDetectionConfig(cfg_path, backup=False)

    with open(cfg_path, 'rb') as f:
        new_raw = f.read()

    old_lines = raw.split(b"\r\n")
    new_lines = new_raw.split(b"\r\n")
    assert b"\n" not in new_raw.replace(b"\r\n", b"")
    assert len(new_lines) == len(old_lines)

    changed = [(a, b) for a, b in zip(old_lines, new_lines) if a != b]
    changed_keys = {b.split(b":")[0] for _, b in changed}
    assert changed_keys <= {b"intensity_threshold", b"neighborhood_size", b"max_stars", b"segment_radius",
                            b"max_feature_ratio", b"roundness_threshold"}
    assert cr.parse(cfg_path).intensity_threshold == 41


@pytest.mark.parametrize("text, check", [
    # An empty value must not swallow the line ending
    ("[StarExtraction]\nmax_stars: \nsegment_radius: 3\n", None),
    # configparser does not strip section names, so this is not the StarExtraction section
    ("[ StarExtraction ]\nmax_stars: 1\n", "[ StarExtraction ]\nmax_stars: 1\n"),
    # An indented line is the continuation of the value above it
    ("[StarExtraction]\nmax_stars: 200\n  intensity_threshold: 5\n", None),
])
def testUpdateConfigLinesParserEdgeCases(text, check):
    """ The updated lines parse with the strict parser to exactly the new values. """

    import configparser

    updates = {"StarExtraction": {"max_stars": "300", "segment_radius": "6", "intensity_threshold": "25"}}
    out = "".join(SF.updateConfigLines(text.splitlines(True), updates))

    parser = configparser.RawConfigParser(inline_comment_prefixes=(';',), strict=True)
    parser.read_string(out)
    for key, value in updates["StarExtraction"].items():
        assert parser.get("StarExtraction", key) == value

    if check is not None:
        assert out.startswith(check)


def testSaveConfigWriteFailureKeepsConfigAndBackup(tmp_path, quietMessages, monkeypatch):
    """ A config write that fails while writing keeps the config and its backup and is reported. """

    cfg_path = str(tmp_path / ".config")
    shutil.copy(TEMPLATE_CONFIG, cfg_path)
    with open(cfg_path, 'rb') as f:
        original = f.read()

    pt = _fakePlateTool(cfg_path)

    # The lines are only produced while the file is being written
    class FailingLines(object):
        def __iter__(self):
            yield "[StarExtraction]\n"
            raise IOError("disk full")

    monkeypatch.setattr(SF, "updateConfigLines", lambda lines, updates: FailingLines())

    dialog = types.SimpleNamespace(plate_tool=pt, _refreshAll=lambda: None)
    SF.CalibrationFilesDialog._saveFile(dialog, "Config", [str(tmp_path)])

    with open(cfg_path, 'rb') as f:
        assert f.read() == original

    backups = [f for f in os.listdir(str(tmp_path)) if ".bak." in f]
    assert len(backups) == 1
    with open(str(tmp_path / backups[0]), 'rb') as f:
        assert f.read() == original

    assert not [f for f in os.listdir(str(tmp_path)) if f.endswith(".tmp")]
    assert any(m.get("message_type") == "error" and "failed" in m.get("message", "") for m in quietMessages)


@pytest.mark.parametrize("key_name, modifier_name", [("Key_U", "ControlModifier"),
                                                      ("Key_Space", "ShiftModifier")])
def testJumpKeysKeepPlateparUnmodified(plateTool, monkeypatch, key_name, modifier_name):
    """ Jumping to the next star re-derives RA/Dec from the pointing, which is not a platepar edit. """

    from pyqtgraph.Qt import QtCore, QtWidgets

    pt = plateTool
    modifier = getattr(QtCore.Qt.KeyboardModifier, modifier_name)
    monkeypatch.setattr(QtWidgets.QApplication, "keyboardModifiers", staticmethod(lambda: modifier))
    monkeypatch.setattr(QtWidgets.QApplication, "queryKeyboardModifiers", staticmethod(lambda: modifier))

    pt.star_pick_mode = True
    pt.platepar_modified = False

    _pressKey(pt, getattr(QtCore.Qt.Key, key_name), modifier)

    assert not pt.platepar_modified


def testPlateparFingerprintTolerance():
    """ Float rounding and the 0/360 wrap are not changes, a real nudge is. """

    fp = lambda az: ((az, 45.0, 10.0, 0.0, 14.5), (np.zeros(3),))

    assert not SF.PlateTool._plateparFingerprintsDiffer(fp(359.9999999999999), fp(0.0))
    assert not SF.PlateTool._plateparFingerprintsDiffer(fp(120.0), fp(120.0 + 1e-13))
    assert SF.PlateTool._plateparFingerprintsDiffer(fp(120.0), fp(120.001))


def testFileManagerSavesForeignPlateparUnderConfigName(plateTool, stationDir, tmp_path):
    """ A platepar loaded from another folder is saved into the station under config.platepar_name. """

    pt = plateTool
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    backup_path = str(backup_dir / "foo_backup.cal")
    shutil.copy(os.path.join(stationDir, pt.config.platepar_name), backup_path)

    pt.loadPlatepar(update=True, platepar_file=backup_path)
    pt.platepar.RA_d += 1.0

    dialog = SF.CalibrationFilesDialog(pt)
    dialog._saveFile("Platepar", [stationDir])

    from RMS.Formats.Platepar import Platepar
    saved = Platepar()
    saved.read(os.path.join(stationDir, pt.config.platepar_name))
    assert saved.RA_d == pytest.approx(pt.platepar.RA_d)
    assert not os.path.exists(os.path.join(stationDir, "foo_backup.cal"))


@pytest.mark.parametrize("path", ["no_refinement", "too_few_pairs"])
def testInitialParamsKeepsUserDistortionWithoutFinalFit(plateTool, monkeypatch, path):
    """ When the full recalibration does not get to the final fit, the user's distortion model and flags
        are kept and only the astrometry.net pointing is applied. """

    from pyqtgraph.Qt import QtWidgets
    from RMS.Formats.Platepar import Platepar

    pt = plateTool
    pp = pt.platepar
    before = (pp.distortion_type, pp.equal_aspect, pp.asymmetry_corr, pp.force_distortion_centre,
              pp.refraction, tuple(pp.x_poly_fwd), pp.F_scale)

    solution = _astrometryNetStandIn(pt)
    pt._solveAstrometryNet = lambda *a, **k: solution

    if path == "no_refinement":
        no = QtWidgets.QMessageBox.StandardButton.No
        monkeypatch.setattr(QtWidgets.QMessageBox, "question", staticmethod(lambda *a, **k: no))

    else:
        # The NN fit only returns a few pairs
        stars = np.array(pp.star_list)[:5]
        monkeypatch.setattr(Platepar, "fitAstrometry",
                            lambda self, *a, **k: (stars[:, 1:4], stars[:, 4:7]) if k.get('use_nn_cost')
                            else None)

    pt.getInitialParamsAstrometryNet(upload_image=False)

    pp = pt.platepar
    after = (pp.distortion_type, pp.equal_aspect, pp.asymmetry_corr, pp.force_distortion_centre,
             pp.refraction, tuple(pp.x_poly_fwd), pp.F_scale)
    assert after == before


def testSkyFitLMFitExcludesCappedStars(plateTool, monkeypatch):
    """ SkyFit2's limiting magnitude fit leaves out the S/N-capped stars like the calibration report. """

    pt = plateTool
    pt.paired_stars = SF.PairedStars()
    for i, (x, y, intens, ra, dec, mag) in enumerate(np.array(pt.platepar.star_list)[:12, 1:]):
        snr = 99.99 if i < 3 else 5.0 + i
        pt.paired_stars.addPair(x, y, 2.0, intens, SF.CatalogStar(ra, dec, mag), snr=snr,
                                saturated=(i == 3))

    masks = []

    def recordLM(mags, snr_arr, snr_targets=(5, 10), exclude_mask=None):
        masks.append(list(exclude_mask))
        return None

    monkeypatch.setattr(SF, "limitingMagnitude", recordLM)
    pt.photometry()

    assert masks
    assert masks[-1][:4] == [True, True, True, True]
    assert not any(masks[-1][4:])


def testKeyholeBridgeVertexMovesAndDeletesTogether(plateTool, stationDir):
    """ Dragging or deleting a bridge vertex of a keyhole polygon (mask with a hole) handles all its
        copies, so the polygon does not tear. """

    import cv2

    pt = plateTool
    mask_img = np.zeros((720, 1280), np.uint8)
    cv2.circle(mask_img, (640, 360), 300, 255, -1)
    mask_path = os.path.join(stationDir, "allsky.bmp")
    cv2.imwrite(mask_path, mask_img)
    assert pt.loadMaskFromFile(mask_path)
    assert len(pt.mask_polygons) == 1
    polygon = pt.mask_polygons[0]

    # A repeated vertex is a bridge vertex
    counts = {}
    for v in polygon:
        counts[v] = counts.get(v, 0) + 1
    bridge = [v for v, n in counts.items() if n > 1]
    assert bridge
    vert_idx = polygon.index(bridge[0])

    pt.moveMaskVertex((0, vert_idx), 700.0, 300.0)
    assert polygon.count((700.0, 300.0)) == counts[bridge[0]]
    assert bridge[0] not in polygon

    n_before = len(polygon)
    pt.deleteMaskVertex((0, polygon.index((700.0, 300.0))))
    assert (700.0, 300.0) not in pt.mask_polygons[0]
    assert len(pt.mask_polygons[0]) == n_before - counts[bridge[0]]


def testCtrlDigitShortcutsInKeyboardReference(plateTool):
    """ The keyboard reference lists every CTRL + digit distortion type with its digit. """

    from RMS.Routines import SkyFitHelp

    html = SkyFitHelp._topicShortcutsSkyfit(plateTool)

    for digit, (_, dist_type) in enumerate(SF.CTRL_DIGIT_DISTORTION_TYPES, start=1):
        assert "{:d} {:s}".format(digit, dist_type) in html
