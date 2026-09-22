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
        members = [m for m in tar.getmembers() if m.name.startswith("Stations/AU000A")]
        tar.extractall(out_dir, members=members)

    return os.path.join(out_dir, "Stations", "AU000A")


@pytest.fixture
def stationDir(stationArchive, tmp_path):
    """ A fresh copy of the example station for each test. """

    dir_path = str(tmp_path / "AU000A")
    shutil.copytree(stationArchive, dir_path)

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
