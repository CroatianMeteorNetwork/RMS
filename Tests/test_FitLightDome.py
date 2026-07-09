""" Tests for the light-dome auto-priming logic in Utils.FitLightDome. """

from __future__ import absolute_import, division, print_function

import datetime
import json
import os

import pytest

from Utils.FitLightDome import modelIsStale, ensureLightDomeModel, AUTO_ATTEMPT_MARKER


class DummyConfig(object):
    def __init__(self, data_dir, station_id="US005X", intensity_threshold=10):
        self.data_dir = data_dir
        self.stationID = station_id
        self.intensity_threshold = intensity_threshold


class DummyPlatepar(object):
    def __init__(self, az_centre=180.0, alt_centre=45.0):
        self.az_centre = az_centre
        self.alt_centre = alt_centre


def freshModelDict(**overrides):
    model = dict(
        cams=["US005X"],
        LM0=[5.5],
        k=0.2, s=0.32, q0=1.0, h0=20.0, ndom=0, domes=[],
        model="vanrhijn_brightness",
        fit_date=datetime.datetime.utcnow().strftime("%Y-%m-%d"),
        n_trials=50000,
        pointing={"US005X": [180.0, 45.0]},
        intensity_threshold={"US005X": 10.0},
        auto_fitted=True,
    )
    model.update(overrides)
    return model


def test_fresh_model_not_stale(tmp_path):
    cfg = DummyConfig(str(tmp_path))
    assert modelIsStale(freshModelDict(), cfg, DummyPlatepar()) is None


def test_stale_by_age(tmp_path):
    cfg = DummyConfig(str(tmp_path))
    old = (datetime.datetime.utcnow() - datetime.timedelta(days=45)).strftime("%Y-%m-%d")
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


def test_legacy_model_without_metadata_never_stale(tmp_path):
    # Models written before the metadata existed (e.g. manual site-pooled fits) must be
    # treated as fresh, not refit-looped
    cfg = DummyConfig(str(tmp_path))
    legacy = dict(cams=["US005X"], LM0=[5.5], k=0.2, s=0.32, q0=1.0, h0=20.0,
                  ndom=0, domes=[], model="vanrhijn_brightness")
    assert modelIsStale(legacy, cfg, DummyPlatepar()) is None


def test_manual_pooled_model_never_overwritten(tmp_path):
    # A stale MANUAL model (no auto_fitted flag) must be kept as-is
    cfg = DummyConfig(str(tmp_path))
    old = (datetime.datetime.utcnow() - datetime.timedelta(days=90)).strftime("%Y-%m-%d")
    model = freshModelDict(fit_date=old)
    del model["auto_fitted"]

    path = os.path.join(str(tmp_path), "US005X_light_dome.json")
    with open(path, "w") as f:
        json.dump(model, f)
    before = open(path).read()

    assert ensureLightDomeModel(cfg, platepar=DummyPlatepar()) is True
    assert open(path).read() == before


def test_no_model_no_archives_waits(tmp_path):
    # Fresh station with no archives: returns False (scalar fallback), writes the daily
    # attempt marker, and the second call the same day short-circuits
    cfg = DummyConfig(str(tmp_path))

    assert ensureLightDomeModel(cfg) is False

    marker = os.path.join(str(tmp_path), "US005X_" + AUTO_ATTEMPT_MARKER)
    assert os.path.isfile(marker)
    assert json.load(open(marker))["date"] == datetime.datetime.utcnow().strftime("%Y-%m-%d")

    assert ensureLightDomeModel(cfg) is False
