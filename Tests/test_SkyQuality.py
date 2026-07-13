""" Tests for Utils.SkyQuality: continuous bias tracking and Bortle mapping. """

from __future__ import absolute_import, division, print_function

import datetime
import json
import os

import pytest

import numpy as np

from Utils.SkyQuality import (bortleClass, resolveWorkingBias, loadRadiometricCalibration,
    resolveApertureCorrection, frameApertureSamples,
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


def test_floor_guard_rejects_unphysical_tonights_observation():
    # Tonight's obs is itself above the floor - the failure mode the guard exists to
    # catch - so it must not vouch for itself; the working value is withdrawn
    obs = [70.0]*6
    cal = calWith(obs)
    bias, source, _ = resolveWorkingBias(cal, 65.0, 60.0)
    assert bias is None
    assert "withdrawn" in source


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


def test_aperture_tracking_median_and_coexistence():
    # f observations ride in the same nightly entries as the bias without clobbering it
    cal = calWith([58.0]*6)
    f, n, cal2 = resolveApertureCorrection(cal, 0.72)
    assert f == pytest.approx(0.72)
    assert n == 1
    key = sorted(cal2["nights"])[-1]
    f2, n2, cal3 = resolveApertureCorrection(cal2, 0.70, night_key=key)
    assert n2 == 1  # same night overwrites, does not accumulate
    assert f2 == pytest.approx(0.70)
    # prior bias/floor entries are preserved
    old_key = sorted(cal3["nights"])[0]
    assert cal3["nights"][old_key]["bias"] == 58.0


def test_aperture_none_when_no_observations():
    cal = calWith([58.0]*3)
    f, n, _ = resolveApertureCorrection(cal, None)
    assert f is None
    assert n == 0


def test_aperture_median_over_window():
    cal = dict(seed=None, nights={})
    today = datetime.datetime.utcnow()
    for i, fv in enumerate([0.60, 0.70, 0.80, 0.72, 0.68]):
        key = (today - datetime.timedelta(days=5 - i)).strftime("%Y%m%d")
        f, n, cal = resolveApertureCorrection(cal, fv, night_key=key)
    assert n == 5
    assert f == pytest.approx(0.70)


def test_aperture_observations_age_out_slowly():
    # f ages on its own slow clock: still trusted after a bias-scale gap (stability
    # beats recency - f self-cancels in tracking), gone after its own limit
    from Utils.SkyQuality import APERTURE_OBS_MAX_AGE_NIGHTS
    cal = dict(seed=None, nights={})
    today = datetime.datetime.utcnow()

    key_mid = (today - datetime.timedelta(days=BIAS_OBS_MAX_AGE_NIGHTS + 10)).strftime("%Y%m%d")
    _, _, cal = resolveApertureCorrection(cal, 0.65, night_key=key_mid)
    f, n, _ = resolveApertureCorrection(dict(cal), None, night_key=today.strftime("%Y%m%d"))
    assert f == pytest.approx(0.65)
    assert n == 1

    cal2 = dict(seed=None, nights={})
    key_old = (today - datetime.timedelta(days=APERTURE_OBS_MAX_AGE_NIGHTS + 10)).strftime("%Y%m%d")
    _, _, cal2 = resolveApertureCorrection(cal2, 0.50, night_key=key_old)
    f, n, _ = resolveApertureCorrection(cal2, None, night_key=today.strftime("%Y%m%d"))
    assert f is None
    assert n == 0


def test_frame_aperture_samples_recovers_gaussian_capture_fraction():
    # Synthetic star: the extractor's windowed sum over a +/-4 px crop vs true total
    h, w = 120, 120
    sigma = 2.5
    yy, xx = np.mgrid[0:h, 0:w]
    x0, y0, amp, bg = 60.0, 60.0, 120.0, 20.0
    star = amp*np.exp(-((xx - x0)**2 + (yy - y0)**2)/(2*sigma**2))
    ave = bg + star

    seg = 4
    crop = star[int(y0) - seg:int(y0) + seg + 1, int(x0) - seg:int(x0) + seg + 1]
    windowed_intens = float(np.sum(crop))
    true_total = 2*np.pi*amp*sigma**2
    expected_f = windowed_intens/true_total

    star_list = [[0.0, y0, x0, windowed_intens, 0.0, 0.0, 3.0]]
    samples = frameApertureSamples(ave, star_list, bit_depth=8)
    assert len(samples) == 1
    assert samples[0] == pytest.approx(expected_f, abs=0.03)
