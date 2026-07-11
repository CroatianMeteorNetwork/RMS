""" Tests for Utils.SkyQuality: continuous bias tracking and Bortle mapping. """

from __future__ import absolute_import, division, print_function

import datetime
import json
import os

import pytest

from Utils.SkyQuality import (bortleClass, resolveWorkingBias, loadRadiometricCalibration,
    BIAS_MIN_OBS, BIAS_STEP_ADU, FLOOR_GUARD_ADU, HISTORY_KEEP, BIAS_OBS_MAX_AGE_NIGHTS)


def nightsBack(obs, days_back_last=1):
    """ Nightly entries for consecutive nights ending days_back_last days ago. Dates are
        relative to today because the estimator ages observations out by calendar date. """
    today = datetime.datetime.utcnow()
    nights = {}
    for i, b in enumerate(obs):
        key = (today - datetime.timedelta(days=days_back_last + len(obs) - 1 - i)
               ).strftime("%Y%m%d")
        nights[key] = dict(bias=b, floor=(b or 60) + 8)
    return nights


def calWith(obs, seed=None):
    """ Build a cal dict with prior nightly observations on recent consecutive nights. """
    return dict(seed=seed, nights=nightsBack(obs))


def test_bortle_mapping():
    assert bortleClass(22.0) == "1"
    assert bortleClass(21.5) == "4"
    assert bortleClass(19.4) == "7"
    assert bortleClass(19.6) == "6"
    assert bortleClass(17.0) == "9"


def test_seed_used_until_enough_observations():
    cal = calWith([58.0, 58.5], seed=dict(bias=71.0, method="overlap-graph"))
    bias, source, _ = resolveWorkingBias(cal, None, 80.0)
    assert bias == 71.0
    assert "seed" in source


def test_tracking_takes_over_from_seed():
    obs = [58.0, 58.4, 57.9, 58.2, 58.1, 58.3]
    cal = calWith(obs, seed=dict(bias=71.0, method="overlap-graph"))
    bias, source, _ = resolveWorkingBias(cal, 58.0, 66.0)
    assert abs(bias - 58.1) < 0.5
    assert source.startswith("tracked")


def test_step_detector_resets_window():
    # Long stable history at ~58, then a pedestal step to ~70 for the last nights
    obs = [58.0]*8 + [70.1, 70.3]
    cal = calWith(obs)
    bias, source, _ = resolveWorkingBias(cal, 69.9, 78.0)
    assert bias > 68.0
    assert "step" in source


def test_gradual_drift_tracks_continuously():
    # Slow drift stays inside the step threshold and the median follows it
    obs = [58.0 + 0.2*i for i in range(12)]
    cal = calWith(obs)
    bias, source, _ = resolveWorkingBias(cal, 60.4, 70.0)
    assert 58.5 < bias < 60.5
    assert source.startswith("tracked")


def test_floor_guard_withdraws_bias():
    # Working bias ~70, but tonight's floor is 60: pedestal dropped, no tonight obs
    obs = [70.0]*6
    cal = calWith(obs)
    bias, source, _ = resolveWorkingBias(cal, None, 60.0)
    assert bias is None
    assert "floor guard" in source


def test_floor_guard_accepts_tonights_observation():
    obs = [70.0]*6
    cal = calWith(obs)
    bias, source, _ = resolveWorkingBias(cal, 57.5, 60.0)
    assert bias == 57.5
    assert "floor guard" in source


def test_history_is_trimmed():
    cal = calWith([58.0]*60)
    _, _, cal2 = resolveWorkingBias(cal, 58.0, 66.0)
    assert len(cal2["nights"]) <= HISTORY_KEEP


def test_retention_does_not_shrink_below_old_window():
    # The file must retain the long-term record, not just the estimator window
    cal = calWith([58.0]*60)
    _, _, cal2 = resolveWorkingBias(cal, 58.0, 66.0)
    assert len(cal2["nights"]) == 61


def test_stale_observations_age_out_and_seed_returns():
    # Enough observations, but all far older than the age window: the seed must regain
    # authority instead of stale observations outranking it forever
    nights = nightsBack([58.0]*(BIAS_MIN_OBS + 2),
        days_back_last=BIAS_OBS_MAX_AGE_NIGHTS + 30)
    cal = dict(seed=dict(bias=71.0, method="overlap-graph"), nights=nights)
    bias, source, _ = resolveWorkingBias(cal, None, 80.0)
    assert bias == 71.0
    assert "seed" in source


def test_handover_recorded_with_delta():
    obs = [58.0, 58.4, 57.9, 58.2, 58.1, 58.3]
    cal = calWith(obs, seed=dict(bias=71.0, method="overlap-graph"))
    bias, _, cal2 = resolveWorkingBias(cal, 58.0, 66.0)
    handover = cal2["handover"]
    assert handover["seed_bias"] == 71.0
    assert handover["tracked_bias"] == pytest.approx(bias, abs=0.01)
    assert handover["delta"] == pytest.approx(bias - 71.0, abs=0.01)


def test_handover_recorded_only_once():
    cal = calWith([58.0]*6, seed=dict(bias=71.0, method="overlap-graph"))
    _, _, cal2 = resolveWorkingBias(cal, 58.0, 66.0)
    first = dict(cal2["handover"])
    _, _, cal3 = resolveWorkingBias(cal2, 59.5, 66.0)
    assert cal3["handover"] == first


def test_handover_survives_save_and_load(tmp_path):
    cal = calWith([58.0]*6, seed=dict(bias=71.0, method="overlap-graph"))
    _, _, cal2 = resolveWorkingBias(cal, 58.0, 66.0)

    path = os.path.join(str(tmp_path), "US005X_radiometric.json")
    with open(path, "w") as f:
        json.dump(cal2, f)

    class DummyConfig(object):
        data_dir = str(tmp_path)
        stationID = "US005X"

    cal3 = loadRadiometricCalibration(DummyConfig())
    assert cal3["handover"] == cal2["handover"]


def test_replay_key_reproduces_historic_decision():
    # A replay passing historic night keys must age observations relative to that night,
    # not relative to today
    nights = nightsBack([58.0]*6, days_back_last=200)
    cal = dict(seed=None, nights=dict(list(nights.items())[:-1]))
    last_key, last_entry = sorted(nights.items())[-1]
    bias, source, _ = resolveWorkingBias(cal, last_entry["bias"], last_entry["floor"],
        night_key=last_key)
    assert bias == pytest.approx(58.0, abs=0.1)
    assert source.startswith("tracked")


def test_legacy_flat_file_loads_as_seed(tmp_path):
    path = os.path.join(str(tmp_path), "US005X_radiometric.json")
    with open(path, "w") as f:
        json.dump(dict(bias=71.0, method="overlap-graph", fit_date="2026-07-11"), f)

    class DummyConfig(object):
        data_dir = str(tmp_path)
        stationID = "US005X"

    cal = loadRadiometricCalibration(DummyConfig())
    assert cal["seed"]["bias"] == 71.0
    assert cal["seed"]["method"] == "overlap-graph"
    assert cal["nights"] == {}
