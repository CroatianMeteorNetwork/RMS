""" Tests for RMS.LightDomeModel. """

from __future__ import absolute_import, division, print_function

import json
import os

import numpy as np
import pytest

from RMS.LightDomeModel import LightDomeModel


@pytest.fixture
def model_dict():
    return dict(
        cams=["US005A", "US005B"],
        LM0=[6.0, 5.5],
        k=0.2,
        s=0.32,
        q0=1.0,      # LP bowl brightness 10x zenith-natural at alt 0
        h0=30.0,
        norder=1,
        harmonics=[dict(order=1, A=12.0, phi=200.0, h=20.0)],
        model="vanrhijn_harmonics",
    )


def test_lm0_per_station(model_dict):
    model = LightDomeModel(model_dict)

    lm_a = model.limitingMagnitude(0.0, 90.0, station_id="US005A")
    lm_b = model.limitingMagnitude(0.0, 90.0, station_id="US005B")

    # Same sky position: the difference is exactly the LM0 difference
    assert lm_a - lm_b == pytest.approx(0.5, abs=1e-9)

    # Unknown station falls back to the site mean
    lm_x = model.limitingMagnitude(0.0, 90.0, station_id="NOPE")
    assert lm_b < lm_x < lm_a


def test_lm_decreases_toward_horizon_and_glow(model_dict):
    model = LightDomeModel(model_dict)

    # Away from the glow, LM must fall monotonically with decreasing altitude
    alts = np.array([80.0, 60.0, 40.0, 20.0, 10.0])
    lms = model.limitingMagnitude(np.full_like(alts, 20.0), alts, station_id="US005A")
    assert np.all(np.diff(lms) < 0)

    # At low altitude, pointing into the glow must be shallower than away from it
    lm_into = model.limitingMagnitude(200.0, 15.0, station_id="US005A")
    lm_away = model.limitingMagnitude(20.0, 15.0, station_id="US005A")
    assert lm_into < lm_away - 0.5


def test_detection_probability(model_dict):
    model = LightDomeModel(model_dict)

    mags = np.array([1.0, 4.0, 8.0])
    p = model.detectionProbability(mags, np.full(3, 0.0), np.full(3, 80.0),
        station_id="US005A")

    # Bounded, and monotonically decreasing with magnitude
    assert np.all((p >= 0) & (p <= 1))
    assert p[0] > 0.95
    assert p[2] < 0.05
    assert np.all(np.diff(p) < 0)

    # A star exactly at the limiting magnitude has 50% detection probability
    lm = model.limitingMagnitude(0.0, 80.0, station_id="US005A")
    p_at_lm = model.detectionProbability(lm, 0.0, 80.0, station_id="US005A")
    assert p_at_lm == pytest.approx(0.5, abs=1e-9)


def test_amplitude_scales_light_pollution_only(model_dict):
    model = LightDomeModel(model_dict)

    lm_before = model.limitingMagnitude(200.0, 15.0, station_id="US005A")

    # More aerosols = brighter LP field = shallower LM
    model.amplitude = 2.0
    lm_after = model.limitingMagnitude(200.0, 15.0, station_id="US005A")
    assert lm_after < lm_before

    # With no LP at all, only van Rhijn + airmass remain and amplitude has no effect
    model_dark = LightDomeModel(dict(model_dict, q0=-10.0, harmonics=[]))
    lm1 = model_dark.limitingMagnitude(200.0, 15.0, station_id="US005A")
    model_dark.amplitude = 2.0
    lm2 = model_dark.limitingMagnitude(200.0, 15.0, station_id="US005A")
    assert lm1 == pytest.approx(lm2, abs=1e-6)


def test_dipole_direction(model_dict):
    model = LightDomeModel(model_dict)

    # Brighter (shallower LM) toward the dipole phase than away from it, at low altitude
    lm_toward = model.limitingMagnitude(200.0, 15.0, station_id="US005A")
    lm_away = model.limitingMagnitude(20.0, 15.0, station_id="US005A")
    assert lm_toward < lm_away

    # The negative half of the cosine cannot push total LP below zero: away-side sky is
    # never DARKER than the airglow-only sky
    dark = LightDomeModel(dict(model_dict, q0=-10.0, harmonics=[]))
    assert model.limitingMagnitude(20.0, 15.0, station_id="US005A") \
        <= dark.limitingMagnitude(20.0, 15.0, station_id="US005A") + 1e-9


def test_load_roundtrip(model_dict, tmp_path):
    path = os.path.join(str(tmp_path), "US005A_light_dome.json")
    with open(path, "w") as f:
        json.dump(model_dict, f)

    class DummyConfig(object):
        data_dir = str(tmp_path)
        stationID = "US005A"

    model = LightDomeModel.load(DummyConfig())
    assert model is not None
    assert model.lm0_map["US005A"] == 6.0

    # No file for a different data_dir
    class DummyConfig2(object):
        data_dir = os.path.join(str(tmp_path), "empty")
        stationID = "US005A"

    assert LightDomeModel.load(DummyConfig2()) is None


def test_load_rejects_legacy_basis(model_dict, tmp_path):
    # A file from the retired von Mises dome basis must not load - it is left for
    # ensureLightDomeModel to detect and refit with the harmonic basis
    legacy = dict(
        cams=["US005A"], LM0=[6.0], k=0.2, s=0.32, q0=1.0, h0=30.0,
        ndom=1, domes=[dict(az=200.0, B=100.0, kappa=5.0, h=15.0)],
        model="vanrhijn_brightness",
    )

    path = os.path.join(str(tmp_path), "US005A_light_dome.json")
    with open(path, "w") as f:
        json.dump(legacy, f)

    class DummyConfig(object):
        data_dir = str(tmp_path)
        stationID = "US005A"

    assert LightDomeModel.load(DummyConfig()) is None

    # A harmonic site-generic file behind the legacy per-station file still loads
    with open(os.path.join(str(tmp_path), "light_dome.json"), "w") as f:
        json.dump(model_dict, f)

    model = LightDomeModel.load(DummyConfig())
    assert model is not None
    assert model.lm0_map["US005A"] == 6.0


def test_load_rejects_lower_pinned_lm0(model_dict, tmp_path):
    # An LM0 pinned at the lower fit bound is a degenerate all-cloudy fit: it predicts
    # almost no stars, so ratios inflate and cloudy nights read as clear. The scalar
    # fallback is strictly better - load() must refuse the model for that station.
    model_dict["LM0"] = [4.05, 5.5]

    path = os.path.join(str(tmp_path), "US005A_light_dome.json")
    with open(path, "w") as f:
        json.dump(model_dict, f)

    class PinnedConfig(object):
        data_dir = str(tmp_path)
        stationID = "US005A"

    assert LightDomeModel.load(PinnedConfig()) is None

    # The sibling with a healthy LM0 keeps using the site model
    path_b = os.path.join(str(tmp_path), "US005B_light_dome.json")
    with open(path_b, "w") as f:
        json.dump(model_dict, f)

    class HealthyConfig(object):
        data_dir = str(tmp_path)
        stationID = "US005B"

    model = LightDomeModel.load(HealthyConfig())
    assert model is not None
    assert model.lm0_map["US005B"] == 5.5
