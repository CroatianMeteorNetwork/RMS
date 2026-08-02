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


def test_load_rejects_degenerate_model_for_all_stations(model_dict, tmp_path):
    # A joint fit with ANY camera pinned at the lower bound was optimized against
    # garbage data - the shared harmonics and s are co-fit, so NO station's slice is
    # trustworthy (observed: AUC0A6 at LM0=4.31 passed a per-station check while its
    # siblings sat at 4.00, and the model predicted a ridiculous star count).
    model_dict["LM0"] = [4.05, 5.5]

    for station in ("US005A", "US005B"):
        path = os.path.join(str(tmp_path), "{}_light_dome.json".format(station))
        with open(path, "w") as f:
            json.dump(model_dict, f)

        class Cfg(object):
            data_dir = str(tmp_path)
            stationID = station

        assert LightDomeModel.load(Cfg()) is None, station


def test_load_rejects_s_at_ceiling(model_dict, tmp_path):
    # s pinned at its upper bound = no depth discrimination; the fit is degenerate
    # even if every LM0 looks plausible
    model_dict["s"] = 1.15

    path = os.path.join(str(tmp_path), "US005A_light_dome.json")
    with open(path, "w") as f:
        json.dump(model_dict, f)

    class Cfg(object):
        data_dir = str(tmp_path)
        stationID = "US005A"

    assert LightDomeModel.load(Cfg()) is None


def test_load_accepts_upper_saturated_model(model_dict, tmp_path):
    # Upper-bound saturation is NOT blocking: ratio normalization compensates until
    # the refit lands, and rejecting would flip whole fleets to scalar overnight
    model_dict["LM0"] = [6.95, 5.5]

    path = os.path.join(str(tmp_path), "US005A_light_dome.json")
    with open(path, "w") as f:
        json.dump(model_dict, f)

    class Cfg(object):
        data_dir = str(tmp_path)
        stationID = "US005A"

    assert LightDomeModel.load(Cfg()) is not None


def test_fit_quality_warnings_bowl_degeneracy(model_dict):
    # A q0 pinned at its upper bound (LP bowl collinear with LM0 - the USC0K
    # symptom: fake extreme light pollution on the render at a pristine site)
    # must warn without blocking
    from RMS.LightDomeModel import (blockingQualityIssues, fitQualityWarnings)

    model_dict["q0"] = 2.97
    model_dict["h0"] = 38.0
    warnings = fitQualityWarnings(model_dict)
    assert len(warnings) == 1 and "q0=" in warnings[0]
    assert blockingQualityIssues(model_dict) == []

    # Healthy bowl: clean
    model_dict["q0"] = 1.0
    assert fitQualityWarnings(model_dict) == []

    # Harmonic amplitude at its bound
    model_dict["harmonics"] = [dict(order=1, A=10.0**3.45, phi=100.0, h=20.0)]
    warnings = fitQualityWarnings(model_dict)
    assert len(warnings) == 1 and "harmonic order 1 amplitude" in warnings[0]


def test_moon_penalty_physics(model_dict):
    # Krisciunas & Schaefer scattered-moonlight term: zero with the moon down,
    # grows with phase and with proximity to the moon, and detectionProbability
    # drops accordingly on moonlit frames
    import numpy as np
    model = LightDomeModel(model_dict)

    az = np.array([180.0, 180.0])
    alt = np.array([60.0, 60.0])

    # Moon below the horizon: no penalty
    assert np.allclose(model.moonPenalty(az, alt, 180.0, -10.0, 100.0), 0.0)

    # Full moon at alt 40, az 180: nearby star suffers far more than one 120 deg away
    pen_near = model.moonPenalty(np.array([180.0]), np.array([50.0]), 180.0, 40.0, 100.0)
    pen_far = model.moonPenalty(np.array([0.0]), np.array([50.0]), 180.0, 40.0, 100.0)
    assert pen_near[0] > pen_far[0] > 0.0
    assert pen_near[0] > 1.0        # bright moon nearby costs magnitudes

    # Quarter moon costs less than full at the same geometry
    pen_quarter = model.moonPenalty(np.array([180.0]), np.array([50.0]), 180.0, 40.0, 50.0)
    assert pen_quarter[0] < pen_near[0]

    # Detection probability drops under the moon and is unchanged without it
    p_dark = model.detectionProbability(5.0, 180.0, 50.0, station_id="US005A")
    p_moon = model.detectionProbability(5.0, 180.0, 50.0, station_id="US005A",
        moon=(180.0, 40.0, 100.0))
    p_none = model.detectionProbability(5.0, 180.0, 50.0, station_id="US005A", moon=None)
    assert p_moon < p_dark
    assert p_none == p_dark
