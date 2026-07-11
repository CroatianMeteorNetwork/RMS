""" Tests for Utils.SkyQuality tier logic and Bortle mapping. """

from __future__ import absolute_import, division, print_function

import pytest

from Utils.SkyQuality import bortleClass, _calibrationIsUsable


class DummyModel(object):
    def __init__(self, fit_date):
        self.model = dict(fit_date=fit_date)


def test_bortle_mapping():
    assert bortleClass(22.0) == "1"
    assert bortleClass(21.5) == "4"
    assert bortleClass(19.4) == "7"
    assert bortleClass(19.6) == "6"
    assert bortleClass(17.0) == "9"


def test_manual_calibration_always_trusted():
    cal = dict(bias=70.0, method="overlap-graph", fit_date="1999-01-01")
    assert _calibrationIsUsable(cal, DummyModel("2026-07-11")) is True


def test_auto_calibration_refreshes_when_stale():
    old = dict(bias=60.0, method="model-regression", fit_date="1999-01-01",
               model_fit_date="2026-07-11")
    assert _calibrationIsUsable(old, DummyModel("2026-07-11")) is False


def test_auto_calibration_refreshes_on_model_change():
    import datetime
    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    cal = dict(bias=60.0, method="model-regression", fit_date=today,
               model_fit_date="2026-07-01")
    assert _calibrationIsUsable(cal, DummyModel("2026-07-11")) is False
    cal["model_fit_date"] = "2026-07-11"
    assert _calibrationIsUsable(cal, DummyModel("2026-07-11")) is True


def test_missing_or_empty_calibration():
    assert _calibrationIsUsable(None, None) is False
    assert _calibrationIsUsable(dict(method="overlap-graph"), None) is False
