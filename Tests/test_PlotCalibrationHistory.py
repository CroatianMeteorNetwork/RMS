""" Tests for Utils.PlotCalibrationHistory: history replay and series extraction. """

from __future__ import absolute_import, division, print_function

import datetime

import pytest

from Utils.PlotCalibrationHistory import parseNightDate, replayBiasHistory, lmHistorySeries


def datesBack(n, days_back_last=1):
    today = datetime.datetime.utcnow()
    return [(today - datetime.timedelta(days=days_back_last + n - 1 - i)).strftime("%Y%m%d")
            for i in range(n)]


def test_parse_night_date():
    assert parseNightDate("20260701") == datetime.datetime(2026, 7, 1)
    assert parseNightDate("US005X_20260701_020000_0") == datetime.datetime(2026, 7, 1)
    assert parseNightDate("seed") is None
    assert parseNightDate("US005X_2026") is None


def test_replay_reproduces_handover():
    keys = datesBack(8)
    # Floors sit above the seed so the floor guard stays quiet and the handover is isolated
    nights = {k: dict(bias=58.0 + 0.1*i, floor=78.0) for i, k in enumerate(keys)}
    cal = dict(seed=dict(bias=71.0, method="overlap-graph"), nights=nights)

    records, handover = replayBiasHistory(cal)

    assert len(records) == len(keys)
    # Seed rules the early nights, tracking the later ones
    assert "seed" in records[0]["source"]
    assert records[-1]["source"].startswith("tracked")
    assert handover is not None
    assert handover["seed_bias"] == 71.0
    assert handover["delta"] == pytest.approx(records[-1]["bias"] - 71.0, abs=1.0)


def test_replay_marks_floor_guard():
    keys = datesBack(7)
    nights = {k: dict(bias=70.0, floor=78.0) for k in keys[:-1]}
    nights[keys[-1]] = dict(bias=None, floor=60.0)   # pedestal dropped, no observation
    cal = dict(seed=None, nights=nights)

    records, _ = replayBiasHistory(cal)

    assert records[-1]["bias"] is None
    assert "floor guard" in records[-1]["source"]


def test_lm_series_envelope_uses_prior_nights_only():
    keys = ["US005X_202606{:02d}".format(i + 1) for i in range(10)]
    history = {k: dict(depth=5.5, dratio=1.0, dmodel="v1") for k in keys}
    history[keys[-1]] = dict(depth=9.9, dratio=1.0, dmodel="v1")   # tonight is an outlier

    series = lmHistorySeries(history)

    assert len(series["depth"]) == 10
    # The envelope on the last night is built from the prior 5.5s, not tonight's 9.9
    assert series["envelope"][-1][1] == pytest.approx(5.5)
    assert series["version_changes"] == []


def test_lm_series_marks_version_change():
    keys = ["US005X_202606{:02d}".format(i + 1) for i in range(8)]
    history = {k: dict(dratio=1.0, dmodel=("v1" if i < 5 else "v2"))
               for i, k in enumerate(keys)}

    series = lmHistorySeries(history)

    assert len(series["version_changes"]) == 1
    assert series["version_changes"][0][1] == "v2"
