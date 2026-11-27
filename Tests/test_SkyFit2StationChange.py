""" Headless tests for SkyFit2 PlateTool.changeStation state reset, cancellation, and failure recovery. """

from __future__ import print_function, division, absolute_import

from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope="module")
def skyfit2():
    pg = pytest.importorskip("pyqtgraph")
    QtWidgets = pg.Qt.QtWidgets
    if QtWidgets.QApplication.instance() is None:
        QtWidgets.QApplication([])
    return pytest.importorskip("Utils.SkyFit2")


def _makeStationTool(skyfit2, dir_a):
    """ Build a lightweight PlateTool stand-in pre-populated with station A calibration state. """

    PlateTool = skyfit2.PlateTool

    class _StubTool(object):
        STATION_STATE_ATTRIBUTES = PlateTool.STATION_STATE_ATTRIBUTES
        STATION_ROLLBACK_ATTRIBUTES = PlateTool.STATION_ROLLBACK_ATTRIBUTES
        changeStation = PlateTool.changeStation
        _stationStateSnapshot = PlateTool._stationStateSnapshot
        _restoreStationState = PlateTool._restoreStationState
        _resetStationState = PlateTool._resetStationState
        _findPlatepar = PlateTool._findPlatepar

    checkbox_states = []
    status_states = []
    img_handles_seen = []
    zoom_handles_seen = []

    pt = _StubTool()
    pt.input_path = dir_a
    pt.dir_path = dir_a
    pt.config = SimpleNamespace(
        stationID="ST0001", platepar_name="platepar_a.cal", fps=25.0,
        media_backend="cv2", star_catalog_file="cat_a", star_catalog_band_ratios=[0, 1, 0, 0, 0, 0, 0],
        catalog_mag_limit=6.5,
    )
    pt._original_catalog_file = "cat_a"
    pt._original_band_ratios = [0, 1, 0, 0, 0, 0, 0]
    pt.frame_times_path = "/tmp/old_ft.json"
    pt.camera_fps = 25.0
    pt.use_fr_files = False
    pt.mode = "astrometry"
    pt.hasData = lambda: True
    pt.img_handle = SimpleNamespace(name="handle_a")

    # Station A calibration state that must not leak to station B
    pt.catalog_lm_tuned = True
    pt.tuned_cat_lim_mag = 8.4
    pt.star_detection_override_enabled = True
    pt.star_detection_override_data = {"FF_A": [1, 2]}
    pt._original_config_gamma = 0.45
    pt.first_platepar_fit = False
    pt.flat_struct = SimpleNamespace(tag="flat_a")
    pt.flat_source_path = "/a/flat.bmp"
    pt.dark = SimpleNamespace(tag="dark_a")
    pt.dark_source_path = "/a/dark.bmp"
    pt.flat_image_data = np.ones((10, 10), dtype=np.uint8)
    pt.mask_use_flat_background = True
    pt.mask_source_path = "/a/mask.bmp"
    pt.mask_polygons = [[(1.0, 1.0), (2.0, 1.0), (2.0, 2.0)]]
    pt.astrometry_solution_info = {"job": 123}
    pt.unsuitable_stars = skyfit2.PairedStars()
    pt.unsuitable_stars.addPair(10, 20, 2.0, 100, skyfit2.CatalogStar(1.0, 2.0, 3.0))

    pt.img = SimpleNamespace(
        data=np.zeros((320, 180), dtype=np.uint8),
        flat_struct=pt.flat_struct, dark=pt.dark,
        changeHandle=lambda h: img_handles_seen.append(h),
        setLevels=lambda lv: None, reloadImage=lambda: None,
    )
    pt.img_zoom = SimpleNamespace(
        flat_struct=pt.flat_struct, dark=pt.dark,
        changeHandle=lambda h: zoom_handles_seen.append(h),
        setLevels=lambda lv: None, reloadImage=lambda: None,
    )
    pt._img_handles_seen = img_handles_seen
    pt._zoom_handles_seen = zoom_handles_seen
    pt.img_frame = SimpleNamespace(
        height=lambda: 180, width=lambda: 320,
        setLimits=lambda **kw: None, autoRange=lambda **kw: None,
    )
    pt.saturation_mask = SimpleNamespace(setImage=lambda img: None)
    pt.tab = SimpleNamespace(
        hist=SimpleNamespace(auto_levels=False, setLevels=lambda a, b: None, getLevels=lambda: (0, 255)),
        settings=SimpleNamespace(populateCatalogList=lambda: None),
        debruijn=SimpleNamespace(updateTable=lambda: None),
        star_detection=SimpleNamespace(
            use_override_checkbox=SimpleNamespace(setChecked=lambda v: checkbox_states.append(v)),
            updateStatus=lambda v: status_states.append(v),
        ),
    )

    pt.initMaskFromFile = lambda: None
    pt.updateMaskOverlayImage = lambda: None
    pt.loadCatalogStars = lambda lm: np.zeros((5, 3))
    pt.loadCalstars = lambda: None
    pt.loadPlatepar = lambda **kw: None
    pt.updatePairedStars = lambda: None
    pt.updateFitResiduals = lambda: None
    pt.updatePicks = lambda: None
    pt.drawPhotometryColoring = lambda: None
    pt.photometry = lambda: None
    pt.updateStars = lambda: None
    pt.updateDistortion = lambda: None
    pt.updateLeftLabels = lambda: None
    pt.updateImageNavigationDisplay = lambda: None
    pt.initStarDetectionOverrides = lambda: None
    pt.updateFindBestFrameButton = lambda: None
    pt.changeMode = lambda m: None
    pt.usingUWOData = lambda: False

    pt._flat_setup_calls = []
    pt.checkAndSetupFlatForMask = lambda: pt._flat_setup_calls.append(True)
    pt._checkbox_states = checkbox_states
    pt._status_states = status_states

    return pt


def _configB():
    return SimpleNamespace(
        stationID="ST0002", platepar_name="platepar_b.cal", fps=30.0,
        media_backend="gst", star_catalog_file="cat_b", star_catalog_band_ratios=[0, 1, 0, 0, 0, 0, 0],
        catalog_mag_limit=7.0,
    )


def testChangeStationResetsStationState(skyfit2, monkeypatch, tmp_path):
    """ Switching to station B clears all per-station calibration state from station A and offers B's flat. """

    dir_a = str(tmp_path/'station_a')
    dir_b = str(tmp_path/'station_b')
    pt = _makeStationTool(skyfit2, dir_a)

    monkeypatch.setattr(skyfit2.cr, "loadConfigFromDirectory", lambda ov, d: _configB())
    pt._findPlatepar = lambda d=None, platepar_name=None: dir_b + "/platepar_b.cal"

    def _detect(**kw):
        pt.img_handle = SimpleNamespace(
            name="handle_b",
            loadChunk=lambda: SimpleNamespace(maxpixel=np.zeros((180, 320), dtype=np.uint8)),
        )

    pt.detectInputType = _detect

    assert pt.changeStation(dir_b) is True

    assert pt.dir_path == dir_b
    assert pt.config.stationID == "ST0002"
    assert pt.catalog_lm_tuned is False
    assert not hasattr(pt, 'tuned_cat_lim_mag')
    assert pt.star_detection_override_enabled is False
    assert pt.star_detection_override_data == {}
    assert pt._original_config_gamma is None
    assert pt.first_platepar_fit is True
    assert pt.flat_struct is None and pt.flat_source_path is None
    assert pt.dark is None and pt.dark_source_path is None
    assert pt.img.flat_struct is None and pt.img.dark is None
    assert pt.img_zoom.flat_struct is None and pt.img_zoom.dark is None
    assert pt.flat_image_data is None and pt.mask_use_flat_background is False
    assert pt.mask_source_path is None
    assert pt.astrometry_solution_info is None
    assert len(pt.unsuitable_stars) == 0
    assert pt._flat_setup_calls == [True]


def testChangeStationCancelKeepsCurrentStation(skyfit2, monkeypatch, tmp_path):
    """ Cancelling the platepar picker must leave station A completely untouched. """

    dir_a = str(tmp_path/'station_a')
    dir_b = str(tmp_path/'station_b')
    pt = _makeStationTool(skyfit2, dir_a)

    monkeypatch.setattr(skyfit2.cr, "loadConfigFromDirectory", lambda ov, d: _configB())

    seen_args = []
    def _cancelPick(d=None, platepar_name=None):
        seen_args.append((d, platepar_name))
        return None

    pt._findPlatepar = _cancelPick

    assert pt.changeStation(dir_b) is False

    # Picker used station B's directory and platepar_name before modifying station A
    assert seen_args == [(dir_b, "platepar_b.cal")]
    assert pt.dir_path == dir_a
    assert pt.config.stationID == "ST0001"
    assert pt.catalog_lm_tuned is True
    assert pt.tuned_cat_lim_mag == pytest.approx(8.4)
    assert pt.flat_source_path == "/a/flat.bmp"
    assert pt.dark_source_path == "/a/dark.bmp"


@pytest.mark.parametrize("fail_at", ["detectInputType", "loadCatalogStars"])
def testChangeStationFailureRestoresCurrentStationState(skyfit2, monkeypatch, tmp_path, fail_at):
    """ A failure mid-switch must restore station A's config, paths, handle, and calibration state. """

    dir_a = str(tmp_path/'station_a')
    dir_b = str(tmp_path/'station_b')
    pt = _makeStationTool(skyfit2, dir_a)
    orig_flat = pt.flat_struct
    orig_dark = pt.dark

    monkeypatch.setattr(skyfit2.cr, "loadConfigFromDirectory", lambda ov, d: _configB())
    monkeypatch.setattr(skyfit2, "qmessagebox", lambda **kw: None)
    pt._findPlatepar = lambda d=None, platepar_name=None: dir_b + "/platepar_b.cal"

    if fail_at == "detectInputType":
        def _failDetect(**kw):
            raise RuntimeError("no images")
        pt.detectInputType = _failDetect
    else:
        def _okDetect(**kw):
            pt.img_handle = SimpleNamespace(
                name="handle_b",
                loadChunk=lambda: SimpleNamespace(maxpixel=np.zeros((180, 320), dtype=np.uint8)),
            )
        pt.detectInputType = _okDetect
        pt.initMaskFromFile = lambda: setattr(pt, "mask_polygons", [])
        pt.loadCatalogStars = lambda lm: (_ for _ in ()).throw(RuntimeError("catalog error"))

    assert pt.changeStation(dir_b) is False

    assert pt.dir_path == dir_a
    assert pt.config.stationID == "ST0001"
    assert pt._original_catalog_file == "cat_a"
    assert pt.frame_times_path == "/tmp/old_ft.json"
    assert pt.camera_fps == 25.0
    assert pt.img_handle.name == "handle_a"
    if fail_at == "loadCatalogStars":
        assert pt._img_handles_seen[-1].name == "handle_a"
        assert pt._zoom_handles_seen[-1].name == "handle_a"
    assert pt.mask_source_path == "/a/mask.bmp"
    assert len(pt.mask_polygons) == 1
    assert pt.astrometry_solution_info == {"job": 123}
    assert pt.catalog_lm_tuned is True
    assert pt.tuned_cat_lim_mag == pytest.approx(8.4)
    assert pt.star_detection_override_enabled is True
    assert pt.star_detection_override_data == {"FF_A": [1, 2]}
    assert pt.flat_struct is orig_flat and pt.dark is orig_dark
    assert pt.img.flat_struct is orig_flat and pt.img.dark is orig_dark
    assert pt.img_zoom.flat_struct is orig_flat and pt.img_zoom.dark is orig_dark
    assert len(pt.unsuitable_stars) == 1
