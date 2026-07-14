""" Tests for the light-dome auto-priming logic in Utils.FitLightDome. """

from __future__ import absolute_import, division, print_function

import datetime
import json
import os

import pytest

from Utils.FitLightDome import (modelIsStale, ensureLightDomeModel, findLegacyModelFile,
    findSiblingStationConfigs, AUTO_ATTEMPT_MARKER)
import RMS.ConfigReader as cr


class DummyConfig(object):
    def __init__(self, data_dir, station_id="US005X", intensity_threshold=10,
                 latitude=32.0, longitude=-110.0, config_file_name=None):
        self.data_dir = data_dir
        self.stationID = station_id
        self.intensity_threshold = intensity_threshold
        self.latitude = latitude
        self.longitude = longitude
        self.config_file_name = config_file_name


class DummyPlatepar(object):
    def __init__(self, az_centre=180.0, alt_centre=45.0):
        self.az_centre = az_centre
        self.alt_centre = alt_centre


def freshModelDict(**overrides):
    model = dict(
        cams=["US005X"],
        LM0=[5.5],
        k=0.2, s=0.32, q0=1.0, h0=20.0, norder=0, harmonics=[],
        model="vanrhijn_harmonics",
        fit_date=datetime.datetime.utcnow().strftime("%Y-%m-%d"),
        n_trials=50000,
        pointing={"US005X": [180.0, 45.0]},
        intensity_threshold={"US005X": 10.0},
        auto_fitted=True,
    )
    model.update(overrides)
    return model


def legacyModelDict():
    return dict(cams=["US005X"], LM0=[5.5], k=0.2, s=0.32, q0=1.0, h0=20.0,
                ndom=0, domes=[], model="vanrhijn_brightness")


def test_fresh_model_not_stale(tmp_path):
    cfg = DummyConfig(str(tmp_path))
    assert modelIsStale(freshModelDict(), cfg, DummyPlatepar()) is None


def test_stale_by_age(tmp_path):
    cfg = DummyConfig(str(tmp_path))
    old = (datetime.datetime.utcnow() - datetime.timedelta(days=60)).strftime("%Y-%m-%d")
    reason = modelIsStale(freshModelDict(fit_date=old), cfg, DummyPlatepar())
    assert reason is not None and "age" in reason


def test_stale_by_threshold_change(tmp_path):
    cfg = DummyConfig(str(tmp_path), intensity_threshold=12)
    reason = modelIsStale(freshModelDict(), cfg, DummyPlatepar())
    assert reason is not None and "intensity_threshold" in reason


def test_stale_by_pointing_change(tmp_path):
    cfg = DummyConfig(str(tmp_path))
    reason = modelIsStale(freshModelDict(), cfg, DummyPlatepar(az_centre=190.0))
    assert reason is not None and "pointing" in reason

    # Azimuth wraparound: 359 vs 1 deg is only 2 deg apart, not stale
    model = freshModelDict(pointing={"US005X": [359.0, 45.0]})
    assert modelIsStale(model, cfg, DummyPlatepar(az_centre=1.0)) is None


def test_model_without_metadata_never_stale(tmp_path):
    # Models written before the staleness metadata existed must be treated as fresh,
    # not refit-looped
    cfg = DummyConfig(str(tmp_path))
    bare = dict(cams=["US005X"], LM0=[5.5], k=0.2, s=0.32, q0=1.0, h0=20.0,
                norder=0, harmonics=[], model="vanrhijn_harmonics")
    assert modelIsStale(bare, cfg, DummyPlatepar()) is None


def test_stale_manual_model_triggers_refit_attempt(tmp_path):
    # A stale manual model is refit by the auto path too (the auto fit pools co-located
    # stations itself, so a manual fit holds no advantage). With no archives available
    # the previous model is kept and the attempt marker is written.
    cfg = DummyConfig(str(tmp_path))
    old = (datetime.datetime.utcnow() - datetime.timedelta(days=90)).strftime("%Y-%m-%d")
    model = freshModelDict(fit_date=old)
    del model["auto_fitted"]

    path = os.path.join(str(tmp_path), "US005X_light_dome.json")
    with open(path, "w") as f:
        json.dump(model, f)
    before = open(path).read()

    # No archived nights in tmp_path: the refit cannot run, the model is kept
    assert ensureLightDomeModel(cfg, platepar=DummyPlatepar()) is True
    assert open(path).read() == before
    assert os.path.isfile(os.path.join(str(tmp_path), "US005X_" + AUTO_ATTEMPT_MARKER))


def test_legacy_basis_triggers_refit_attempt(tmp_path):
    # A model file from the retired dome basis is refit outright, even if fresh: it no
    # longer evaluates. With no archives the station falls back to scalar behavior.
    cfg = DummyConfig(str(tmp_path))

    path = os.path.join(str(tmp_path), "US005X_light_dome.json")
    with open(path, "w") as f:
        json.dump(legacyModelDict(), f)

    assert findLegacyModelFile(str(tmp_path), "US005X") == path

    assert ensureLightDomeModel(cfg, platepar=DummyPlatepar()) is False
    assert os.path.isfile(os.path.join(str(tmp_path), "US005X_" + AUTO_ATTEMPT_MARKER))


def test_shadowed_legacy_file_does_not_refit_loop(tmp_path):
    # A legacy site-generic file behind a fresh harmonic per-station file is shadowed
    # and must not trigger refits
    cfg = DummyConfig(str(tmp_path))

    with open(os.path.join(str(tmp_path), "light_dome.json"), "w") as f:
        json.dump(legacyModelDict(), f)
    with open(os.path.join(str(tmp_path), "US005X_light_dome.json"), "w") as f:
        json.dump(freshModelDict(), f)

    assert findLegacyModelFile(str(tmp_path), "US005X") is None
    assert ensureLightDomeModel(cfg, platepar=DummyPlatepar()) is True

    # No attempt marker: the fresh model short-circuits before the rate limiter
    assert not os.path.isfile(os.path.join(str(tmp_path), "US005X_" + AUTO_ATTEMPT_MARKER))


def test_no_model_no_archives_waits(tmp_path):
    # Fresh station with no archives: returns False (scalar fallback), writes the daily
    # attempt marker, and the second call the same day short-circuits
    cfg = DummyConfig(str(tmp_path))

    assert ensureLightDomeModel(cfg) is False

    marker = os.path.join(str(tmp_path), "US005X_" + AUTO_ATTEMPT_MARKER)
    assert os.path.isfile(marker)
    assert json.load(open(marker))["date"] == datetime.datetime.utcnow().strftime("%Y-%m-%d")

    assert ensureLightDomeModel(cfg) is False


def test_sibling_discovery(tmp_path, monkeypatch):
    # Multi-camera layout: Stations/<ID>/.config with the directory named by the
    # stationID. A far-away station and an unrelated RMS checkout (directory name does
    # not match its template stationID) must both be excluded.
    root = os.path.join(str(tmp_path), "Stations")
    for name in ["US005X", "US005Y", "US005Z", "XX9999", "RMS"]:
        os.makedirs(os.path.join(root, name))
        with open(os.path.join(root, name, ".config"), "w") as f:
            f.write("; dummy\n")

    def fakeParse(path, strict=True):
        name = os.path.basename(os.path.dirname(path))
        station_id = "XX0001" if name == "RMS" else name
        latitude = 45.0 if name == "XX9999" else 32.0
        return DummyConfig(os.path.join(str(tmp_path), "data", station_id),
            station_id=station_id, latitude=latitude, config_file_name=path)

    monkeypatch.setattr(cr, "parse", fakeParse)

    cfg = fakeParse(os.path.join(root, "US005X", ".config"))
    siblings = findSiblingStationConfigs(cfg)
    ids = [str(c.stationID) for c in siblings]

    assert ids[0] == "US005X"
    assert set(ids) == {"US005X", "US005Y", "US005Z"}

    # A sibling sharing this station's data_dir would double-count trials - excluded
    def fakeParseSameDataDir(path, strict=True):
        c = fakeParse(path)
        c.data_dir = str(cfg.data_dir)
        return c

    monkeypatch.setattr(cr, "parse", fakeParseSameDataDir)
    assert [str(c.stationID) for c in findSiblingStationConfigs(cfg)] == ["US005X"]

    # A single-camera station (config not in the multi-camera layout) fits alone
    solo = DummyConfig(str(tmp_path), station_id="US1234", config_file_name=None)
    assert [str(c.stationID) for c in findSiblingStationConfigs(solo)] == ["US1234"]


def _writeHistory(tmp_path, station, dratios, model_version):
    import json
    entries = {"%s_202606%02d" % (station, i + 1): dict(dratio=r, dmodel=model_version)
               for i, r in enumerate(dratios)}
    with open(os.path.join(str(tmp_path), "%s_flux_lm_history.json" % station), "w") as f:
        json.dump(entries, f)


def test_stale_by_normalization_drift(tmp_path):
    cfg = DummyConfig(str(tmp_path))
    model = freshModelDict()
    _writeHistory(tmp_path, "US005X", [0.65]*6, model["fit_date"])
    reason = modelIsStale(model, cfg, DummyPlatepar())
    assert reason is not None and "drift" in reason


def test_no_drift_when_ratios_near_one(tmp_path):
    cfg = DummyConfig(str(tmp_path))
    model = freshModelDict()
    _writeHistory(tmp_path, "US005X", [0.95, 1.02, 0.98, 1.05, 0.9, 1.0], model["fit_date"])
    assert modelIsStale(model, cfg, DummyPlatepar()) is None


def test_drift_ignores_other_model_versions(tmp_path):
    cfg = DummyConfig(str(tmp_path))
    model = freshModelDict()
    _writeHistory(tmp_path, "US005X", [0.5]*8, "2001-01-01")
    assert modelIsStale(model, cfg, DummyPlatepar()) is None
