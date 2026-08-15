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
        floor_modeled=True,
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


def test_fresh_model_not_covering_siblings_triggers_refit(tmp_path, monkeypatch):
    # A FRESH single-camera model on a multi-camera computer is superseded by the pooled
    # site fit - it must not be kept until it ages out. With no archives the refit cannot
    # run: the model is kept for now and the attempt marker is written.
    import Utils.FitLightDome as fld

    cfg = DummyConfig(str(tmp_path))
    with open(os.path.join(str(tmp_path), "US005X_light_dome.json"), "w") as f:
        json.dump(freshModelDict(), f)

    sibling = DummyConfig(os.path.join(str(tmp_path), "sib"), station_id="US005Y")
    monkeypatch.setattr(fld, "findSiblingStationConfigs", lambda c: [c, sibling])

    assert ensureLightDomeModel(cfg, platepar=DummyPlatepar()) is True
    assert os.path.isfile(os.path.join(str(tmp_path), "US005X_" + AUTO_ATTEMPT_MARKER))

    # A fresh model that covers all siblings is kept - no refit attempt
    os.remove(os.path.join(str(tmp_path), "US005X_" + AUTO_ATTEMPT_MARKER))
    with open(os.path.join(str(tmp_path), "US005X_light_dome.json"), "w") as f:
        json.dump(freshModelDict(cams=["US005X", "US005Y"], LM0=[5.5, 5.6]), f)

    assert ensureLightDomeModel(cfg, platepar=DummyPlatepar()) is True
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


def test_radiometric_anchor_pools_site_measurements(tmp_path):
    # The anchor is a SITE property: measurements from all co-located cameras pool into
    # one offset, so every camera's map shows the same site zenith
    from RMS.LightDomeModel import LightDomeModel
    from Utils.FitLightDome import _radiometricAnchorOffset, NATURAL_ZENITH_SQM

    model = LightDomeModel(dict(cams=["US005X"], LM0=[5.5], k=0.2, s=0.32,
        q0=-10.0, h0=20.0, norder=0, harmonics=[], model="vanrhijn_harmonics"))

    def writeHistory(station, sqm):
        d = os.path.join(str(tmp_path), station)
        os.makedirs(d)
        nights = {"%s_2026070%d" % (station, i): dict(sqm=sqm, absolute=True,
                  az=0.0, alt=90.0) for i in range(1, 4)}
        with open(os.path.join(d, "%s_sky_quality_history.json" % station), "w") as f:
            json.dump(dict(nights=nights), f)
        return DummyConfig(d, station_id=station)

    cfg_a = writeHistory("US005X", 20.0)
    cfg_b = writeHistory("US005Y", 21.0)

    # Zenith measurements pass through untransposed: anchor = median(sqm) - model zenith
    single = _radiometricAnchorOffset(model, cfg_a)
    assert single == pytest.approx(20.0 - NATURAL_ZENITH_SQM, abs=1e-6)

    pooled = _radiometricAnchorOffset(model, [cfg_a, cfg_b])
    assert pooled == pytest.approx(20.5 - NATURAL_ZENITH_SQM, abs=1e-6)

    # Order-independent, and identical for whichever camera's map is being drawn
    assert pooled == pytest.approx(_radiometricAnchorOffset(model, [cfg_b, cfg_a]),
        abs=1e-9)


def test_anchor_surfaces_per_camera_residuals(tmp_path, caplog):
    # Cameras disagreeing about the site zenith (after transposition through the shared
    # field) is a glow-field misallocation diagnostic - it must be logged, and flagged
    # when beyond ANCHOR_RESIDUAL_WARN (via the rmslogger, so it lands in the
    # nightly station log with timestamps, not on bare stdout)
    from RMS.LightDomeModel import LightDomeModel
    from Utils.FitLightDome import _radiometricAnchorOffset

    model = LightDomeModel(dict(cams=["US005X"], LM0=[5.5], k=0.2, s=0.32,
        q0=-10.0, h0=20.0, norder=0, harmonics=[], model="vanrhijn_harmonics"))

    def writeHistory(station, sqm):
        d = os.path.join(str(tmp_path), station)
        os.makedirs(d)
        nights = {"%s_2026070%d" % (station, i): dict(sqm=sqm, absolute=True,
                  az=0.0, alt=90.0) for i in range(1, 4)}
        with open(os.path.join(d, "%s_sky_quality_history.json" % station), "w") as f:
            json.dump(dict(nights=nights), f)
        return DummyConfig(d, station_id=station)

    cfg_a = writeHistory("US005X", 20.0)
    cfg_b = writeHistory("US005Y", 21.0)

    import logging
    with caplog.at_level(logging.INFO, logger="rmslogger"):
        _radiometricAnchorOffset(model, [cfg_a, cfg_b])
    out = caplog.text

    assert "US005X: -0.50 mag vs site (n=3)" in out
    assert "US005Y: +0.50 mag vs site (n=3)" in out
    assert out.count("check the glow field allocation") == 2

    # A single camera has no cross-check - nothing logged
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="rmslogger"):
        _radiometricAnchorOffset(model, cfg_a)
    assert "vs site" not in caplog.text


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


def test_fit_quality_issues():
    from Utils.FitLightDome import fitQualityIssues, S_FIT_MAX

    # Healthy fit: no issues
    assert fitQualityIssues(freshModelDict(catalog_lim_mag=6.0)) == []

    # LM0 pinned at the lower fit bound (all-cloudy fit window)
    issues = fitQualityIssues(freshModelDict(LM0=[4.05], catalog_lim_mag=6.0))
    assert len(issues) == 1 and "lower fit bound" in issues[0]

    # LM0 pinned at the upper bound (saturated against the catalog depth)
    issues = fitQualityIssues(freshModelDict(LM0=[6.95], catalog_lim_mag=6.0))
    assert len(issues) == 1 and "upper fit bound" in issues[0]

    # Models without a stored depth are judged against the default depth
    issues = fitQualityIssues(freshModelDict(LM0=[6.95]))
    assert len(issues) == 1 and "upper fit bound" in issues[0]

    # s pinned at its bound: no depth discrimination
    issues = fitQualityIssues(freshModelDict(s=S_FIT_MAX))
    assert len(issues) == 1 and "s=" in issues[0]


def test_degenerate_installed_model_triggers_refit_attempt(tmp_path):
    # A fresh but saturated model (LM0 at the upper bound - the old fixed-depth fleet
    # symptom) is refit outright rather than left to poison verdicts. With no archives
    # available the model is kept and the attempt marker is written.
    cfg = DummyConfig(str(tmp_path))
    model = freshModelDict(LM0=[6.95], catalog_lim_mag=6.0)

    path = os.path.join(str(tmp_path), "US005X_light_dome.json")
    with open(path, "w") as f:
        json.dump(model, f)
    before = open(path).read()

    assert ensureLightDomeModel(cfg, platepar=DummyPlatepar()) is True
    assert open(path).read() == before
    assert os.path.isfile(os.path.join(str(tmp_path), "US005X_" + AUTO_ATTEMPT_MARKER))


def test_lower_pinned_installed_model_falls_back_to_scalar(tmp_path):
    # A model with LM0 at the LOWER bound is refused by LightDomeModel.load itself, so
    # with no archives to refit from, ensureLightDomeModel reports no usable model and
    # the caller falls back to the scalar path
    cfg = DummyConfig(str(tmp_path))
    model = freshModelDict(LM0=[4.0], catalog_lim_mag=6.0)

    path = os.path.join(str(tmp_path), "US005X_light_dome.json")
    with open(path, "w") as f:
        json.dump(model, f)

    assert ensureLightDomeModel(cfg, platepar=DummyPlatepar()) is False
    assert os.path.isfile(os.path.join(str(tmp_path), "US005X_" + AUTO_ATTEMPT_MARKER))


def test_dome_nll_gradient_matches_numerical():
    # The analytic gradient is what makes the site fit tractable; it must agree with
    # numerical differentiation of the same objective everywhere it is smooth
    import numpy as np
    from Utils.FitLightDome import _domeNLLAndGrad

    rng = np.random.RandomState(42)
    ncam = 3
    n = 4000

    az = rng.uniform(0.0, 360.0, n)
    alt = rng.uniform(6.0, 89.0, n)
    mag = rng.uniform(2.0, 7.0, n)
    det = (rng.uniform(0.0, 1.0, n) < 0.5).astype(float)
    ci = rng.randint(0, ncam, n)

    # Per-frame-style chance floor, nonzero so the floored gradient path is exercised
    pc = rng.uniform(0.0, 0.03, n)

    for norder in range(4):

        p = np.array([5.2, 5.6, 5.9] + [0.25, 0.4, 0.8, 22.0]
                     + [0.9, 130.0, 18.0, 0.4, 40.0, 12.0, -0.2, 15.0, 25.0][:3*norder])

        f0, grad = _domeNLLAndGrad(p, ncam, norder, az, alt, mag, det, ci, pc)

        # Central differences
        num = np.zeros_like(p)
        eps = 1e-6
        for i in range(len(p)):
            pp_hi, pp_lo = p.copy(), p.copy()
            pp_hi[i] += eps
            pp_lo[i] -= eps
            f_hi, _ = _domeNLLAndGrad(pp_hi, ncam, norder, az, alt, mag, det, ci, pc)
            f_lo, _ = _domeNLLAndGrad(pp_lo, ncam, norder, az, alt, mag, det, ci, pc)
            num[i] = (f_hi - f_lo)/(2*eps)

        scale = np.maximum(np.abs(num), 1.0)
        assert np.allclose(grad/scale, num/scale, atol=5e-5), \
            "norder={:d}: analytic {} vs numerical {}".format(norder, grad, num)


def test_fit_trial_subsampling_deterministic():
    # The subsample the optimizer sees must reproduce across refits on the same data
    import numpy as np
    from Utils.FitLightDome import MAX_FIT_TRIALS

    n = MAX_FIT_TRIALS + 1000
    sel_a = np.random.RandomState(0).choice(n, MAX_FIT_TRIALS, replace=False)
    sel_b = np.random.RandomState(0).choice(n, MAX_FIT_TRIALS, replace=False)
    assert np.array_equal(sel_a, sel_b)


def test_site_claim_is_exclusive_per_day(tmp_path):
    # One fit per site per day, whatever the overlap: the claim is what stops colocated
    # cameras from each fitting and each installing its own "site" model
    from Utils.FitLightDome import claimSiteFit, SITE_CLAIM_MARKER

    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")

    configs = []
    for station in ["US005X", "US005Y", "US005Z"]:
        d = os.path.join(str(tmp_path), station)
        os.makedirs(d)
        configs.append(DummyConfig(d, station_id=station))

    # Whoever calls first wins; the siblings are turned away
    assert claimSiteFit(configs, today) is True
    assert claimSiteFit(list(reversed(configs)), today) is False
    assert claimSiteFit([configs[2], configs[0], configs[1]], today) is False

    # The claim lives in the lexicographically first station's directory, so every
    # sibling computes the same path no matter what order it sees them in
    claim = os.path.join(str(tmp_path), "US005X",
                         "{:s}_{:s}.json".format(SITE_CLAIM_MARKER, today))
    assert os.path.isfile(claim)

    # A new day claims again, and the stale claim is cleaned up
    tomorrow = (datetime.datetime.utcnow() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    assert claimSiteFit(configs, tomorrow) is True
    assert not os.path.isfile(claim)


def test_site_claim_degrades_open(tmp_path):
    # A site that cannot write the claim (missing/read-only directory) must still get a
    # model - the guard may not be the reason a station never fits
    from Utils.FitLightDome import claimSiteFit

    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    missing = DummyConfig(os.path.join(str(tmp_path), "nonexistent"), station_id="US005X")

    assert claimSiteFit([missing], today) is True


def test_attempt_markers_stamped_before_the_fit(tmp_path):
    from Utils.FitLightDome import stampAttemptMarkers, AUTO_ATTEMPT_MARKER

    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")

    configs = []
    for station in ["US005X", "US005Y"]:
        d = os.path.join(str(tmp_path), station)
        os.makedirs(d)
        configs.append(DummyConfig(d, station_id=station))

    # An unwritable sibling must not stop the others being stamped
    configs.append(DummyConfig(os.path.join(str(tmp_path), "gone"), station_id="US005Z"))

    stampAttemptMarkers(configs, today)

    for station in ["US005X", "US005Y"]:
        marker = os.path.join(str(tmp_path), station,
                              "{:s}_{:s}".format(station, AUTO_ATTEMPT_MARKER))
        assert json.load(open(marker))["date"] == today


def test_install_never_rolls_back_a_newer_model(tmp_path):
    # A slow fit finishing after a faster one must not reinstate its older field
    from Utils.FitLightDome import _installIsNewer

    path = os.path.join(str(tmp_path), "US005X_light_dome.json")

    early = freshModelDict(fit_timestamp="2026-08-15T01:00:00")
    late = freshModelDict(fit_timestamp="2026-08-15T02:00:00")

    # Nothing in place yet
    assert _installIsNewer(early, path) is True

    with open(path, "w") as f:
        json.dump(late, f)

    # Same day, but ours is the older fit - refuse
    assert _installIsNewer(early, path) is False
    assert _installIsNewer(late, path) is True

    # A legacy model carrying only a date is replaceable by a same-day timestamped fit
    legacy = freshModelDict(fit_date="2026-08-15")
    legacy.pop("fit_timestamp", None)
    with open(path, "w") as f:
        json.dump(legacy, f)
    assert _installIsNewer(early, path) is True

    # An unreadable file must not block the install
    with open(path, "w") as f:
        f.write("{ truncated")
    assert _installIsNewer(early, path) is True


def test_concurrent_siblings_fit_once_and_share_one_model(tmp_path, monkeypatch):
    # The CAC0B failure: two colocated cameras both reached the fit, each installed its
    # own model over the whole site, and the pod ended up carrying different glow fields
    # for the same sky. Only one fit may run, and both cameras must end up identical.
    import Utils.FitLightDome as fld

    stations = ["US005X", "US005Y"]
    configs = []
    for station in stations:
        d = os.path.join(str(tmp_path), station)
        os.makedirs(d)
        cfg = DummyConfig(d, station_id=station)
        configs.append(cfg)

        # Both carry a stale model, so both want to refit
        old = (datetime.datetime.utcnow() - datetime.timedelta(days=90)).strftime("%Y-%m-%d")
        with open(os.path.join(d, "{:s}_light_dome.json".format(station)), "w") as f:
            json.dump(freshModelDict(cams=stations, LM0=[5.5, 5.6], fit_date=old,
                pointing={s: [180.0, 45.0] for s in stations},
                intensity_threshold={s: 10.0 for s in stations}), f)

    monkeypatch.setattr(fld, "findSiblingStationConfigs", lambda c: list(configs))
    monkeypatch.setattr(fld, "selectNightDirs",
        lambda cfg, **kw: ["/n/{:s}_2026081{:d}_000000_000000".format(cfg.stationID, i)
                           for i in range(4)])
    monkeypatch.setattr(fld, "renderLightDomeModel",
        lambda *a, **kw: None)

    calls = []

    def fakeFit(station_configs, dates=None, **kwargs):
        calls.append([str(c.stationID) for c in station_configs])
        n = len(calls)

        # THE RACE: the sibling's nightly check falls inside this fit, which is where the
        # real window is (a six-camera fit is long). Sequential calls never reproduce it -
        # the first fit installs a fresh model and the second camera short-circuits.
        if n == 1:
            fld.ensureLightDomeModel(configs[1], platepar=DummyPlatepar())

        # Two real fits land on different fields; whichever installs last would win
        return freshModelDict(cams=stations, LM0=[5.5, 5.6],
            q0=1.0 + 0.1*n, harmonics=[dict(order=1, A=1.0, phi=72.0*n, h=20.0)],
            fit_timestamp="2026-08-15T0{:d}:00:00".format(n),
            n_trials=50000, n_frames_used=200,
            pointing={s: [180.0, 45.0] for s in stations},
            intensity_threshold={s: 10.0 for s in stations})

    monkeypatch.setattr(fld, "fitLightDome", fakeFit)

    assert fld.ensureLightDomeModel(configs[0], platepar=DummyPlatepar()) is True

    # Exactly one fit ran for the site, and it pooled both cameras
    assert len(calls) == 1, calls
    assert set(calls[0]) == set(stations)

    # Both cameras carry the same field - the invariant the pod was violating
    installed = []
    for station in stations:
        with open(os.path.join(str(tmp_path), station,
                               "{:s}_light_dome.json".format(station))) as f:
            installed.append(json.load(f))

    assert installed[0] == installed[1]
    assert installed[0]["fit_timestamp"] == "2026-08-15T01:00:00"
    assert installed[0]["q0"] == pytest.approx(1.1)
