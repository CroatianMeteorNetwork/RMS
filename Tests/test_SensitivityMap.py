""" Tests for the per-camera sensitivity map (RMS.SensitivityMap) and its use in the flux collection
area: block interpolation, the best-block reference, placing a night's bin relative to the map
through the zero point (including a vignetting coefficient change), file loading with quality
gating, and the collection-area file name carrying the map identity.
"""

from __future__ import absolute_import, division, print_function

import json
import os

import numpy as np
import pytest

from RMS.SensitivityMap import (SensitivityMap, sensitivityMapQualityIssues, _areaMeanVignettingShift,
                                MIN_TRIALS_PER_BLOCK, SENSITIVITY_MAP_FILE_SUFFIX, MAX_ZERO_POINT_SHIFT)
from Utils.Flux import generateColAreaJSONFileName, stellarLMModel

X_RES, Y_RES = 1920, 1080


def _mapDict(lm_rows, **extra):
    nby = len(lm_rows)
    nbx = len(lm_rows[0])
    d = dict(stationID="US005X", nbx=nbx, nby=nby, X_res=X_RES, Y_res=Y_RES,
             LM=[v for row in lm_rows for v in row], s=0.4,
             n_trials=MIN_TRIALS_PER_BLOCK*nbx*nby + 1, fit_date="2026-10-04",
             nights=["US005X_20261001_000000_000000"], mag_lev_ref=10.5, vignetting_coeff_ref=0.00059)
    d.update(extra)
    return d


class FakePlatepar(object):
    def __init__(self, k):
        self.X_res, self.Y_res = X_RES, Y_RES
        self.vignetting_coeff = k


def testBlockInterpolationAndClamping():

    rows = [[5.0, 5.5, 5.0, 4.5],
            [5.5, 6.0, 5.5, 5.0],
            [5.0, 5.5, 5.0, 4.5]]
    m = SensitivityMap(_mapDict(rows))

    # Block centres return the block values exactly
    cx = (np.arange(4) + 0.5)*X_RES/4
    cy = (np.arange(3) + 0.5)*Y_RES/3
    for j in range(3):
        for i in range(4):
            assert m.lmAt(cx[i], cy[j]) == pytest.approx(rows[j][i])

    # Halfway between two centres is their mean
    assert m.lmAt((cx[0] + cx[1])/2, cy[1]) == pytest.approx(5.75)

    # Beyond the outer centres the value is held, so the corner pixel is the corner block
    assert m.lmAt(0, 0) == pytest.approx(5.0)
    assert m.lmAt(X_RES, Y_RES) == pytest.approx(4.5)

    # Vectorised call
    out = m.lmAt(np.array([cx[1], 0.0]), np.array([cy[1], 0.0]))
    assert np.allclose(out, [6.0, 5.0])

    # Reference is the best block, and its sensitivity ratio is 1; a block 1 mag shallower is 0.398
    assert m.lm_ref == pytest.approx(6.0)
    assert m.sensitivityRatio(cx[1], cy[1]) == pytest.approx(1.0)
    assert m.sensitivityRatio(cx[0], cy[0]) == pytest.approx(10**(-0.4*1.0))
    assert m.sensitivityRatio(cx[3], cy[0]) == pytest.approx(10**(-0.4*1.5))


def testBinStellarLMFollowsZeroPoint():

    m = SensitivityMap(_mapDict([[5.0, 6.0], [5.5, 5.0]]))
    pp = FakePlatepar(0.00059)

    # Same zero point as the map's reference: the measured best-block LM
    assert m.binStellarLM(10.5, pp, stellarLMModel) == pytest.approx(6.0)

    # One magnitude more throughput moves the LM by the zero-point model's slope
    expected = 6.0 + (stellarLMModel(11.5) - stellarLMModel(10.5))
    assert m.binStellarLM(11.5, pp, stellarLMModel) == pytest.approx(expected)
    assert expected > 6.0

    # No reference zero point in the map, or an undefined bin zero point: dark-sky level as is
    m_noref = SensitivityMap(_mapDict([[5.0, 6.0], [5.5, 5.0]], mag_lev_ref=None))
    assert m_noref.binStellarLM(12.0, pp, stellarLMModel) == pytest.approx(6.0)
    assert m.binStellarLM(np.nan, pp, stellarLMModel) == pytest.approx(6.0)

    # A zero point several magnitudes off the reference is a units change, not transparency:
    # the shift is clipped (both directions) and flagged once
    assert not m.shift_clipped
    assert m.binStellarLM(10.5 + 4.0, pp, stellarLMModel) == pytest.approx(6.0 + MAX_ZERO_POINT_SHIFT)
    assert m.shift_clipped
    assert m.binStellarLM(10.5 - 4.0, pp, stellarLMModel) == pytest.approx(6.0 - MAX_ZERO_POINT_SHIFT)


def testReferenceFollowsVignettingChange():

    m = SensitivityMap(_mapDict([[5.0, 6.0], [5.5, 5.0]], vignetting_coeff_ref=0.0008))

    # Lowering the platepar's vignetting coefficient lowers every fitted zero point by the
    # area-mean change of the correction; the reference must move with it
    shift = _areaMeanVignettingShift(X_RES, Y_RES, 0.0008, 0.00059)
    assert shift < 0
    assert m.referenceZeroPoint(FakePlatepar(0.0008)) == pytest.approx(10.5)
    assert m.referenceZeroPoint(FakePlatepar(0.00059)) == pytest.approx(10.5 + shift)

    # A night with the same sky, measured under the new coefficient, lands on the same LM
    assert m.binStellarLM(10.5 + shift, FakePlatepar(0.00059), stellarLMModel) == pytest.approx(6.0)

    # No change of coefficient, no shift
    assert _areaMeanVignettingShift(X_RES, Y_RES, 0.0008, 0.0008) == 0.0


def testLoadWithQualityGating(tmp_path):

    class Cfg(object):
        data_dir = str(tmp_path)
        stationID = "US005X"

    # No file
    assert SensitivityMap.load(Cfg()) is None

    path = os.path.join(str(tmp_path), "US005X_" + SENSITIVITY_MAP_FILE_SUFFIX)

    good = _mapDict([[5.0, 6.0, 5.5], [5.5, 5.8, 5.0]])
    with open(path, "w") as f:
        json.dump(good, f)
    m = SensitivityMap.load(Cfg())
    assert m is not None
    assert m.nbx == 3 and m.nby == 2
    assert m.lm_ref == pytest.approx(6.0)
    assert m.tag.startswith("sensmap-2026-10-04-")
    assert "3x2 blocks" in m.summary()

    # Another map has another tag, the same map the same tag
    other = SensitivityMap(_mapDict([[5.0, 6.0, 5.5], [5.5, 5.8, 4.0]]))
    assert other.tag != m.tag
    assert SensitivityMap(good).tag == m.tag

    # Wrong number of values, too few trials, absurd width or spread: refused
    for bad in (dict(good, LM=good["LM"][:-1]),
                dict(good, n_trials=10),
                dict(good, s=3.0),
                dict(good, LM=[1.0, 6.0, 5.5, 5.5, 5.8, 5.0])):
        assert sensitivityMapQualityIssues(bad)
        with open(path, "w") as f:
            json.dump(bad, f)
        assert SensitivityMap.load(Cfg()) is None

    # Unreadable file
    with open(path, "w") as f:
        f.write("{not json")
    assert SensitivityMap.load(Cfg()) is None


class FakeConfig(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.stationID = "US005X"
        self.gamma = 0.5
        self.bit_depth = 8
        self.fps = 25.0
        self.star_gate_factor = 3.5
        self.segment_radius = 6
        self.max_feature_ratio = 0.8
        self.roundness_threshold = 0.5
        self.camera_settings_path = os.path.join(data_dir, "camera_settings.json")
        self.config_file_path = data_dir


class PointedPlatepar(FakePlatepar):
    def __init__(self, k=0.00059, az=90.0, alt=30.0):
        FakePlatepar.__init__(self, k)
        self.az_centre, self.alt_centre = az, alt


def _writeRecalibrated(night_dir, zp, depth, k=0.00059, n_frames=5, matched=40):
    os.makedirs(night_dir, exist_ok=True)
    ppr = {}
    for i in range(n_frames):
        # Linear ramp whose 90th percentile is exactly `depth` (numpy interpolates linearly)
        mags = np.linspace(depth - 2.7, depth + 0.3, matched)
        star_list = [[2461000.5, 1.0, 1.0, 1.0, 1.0, 1.0, float(m)] for m in mags]
        ppr["FF_US005X_20261002_01{:04d}_000_0000000.fits".format(i)] = dict(
            mag_lev=zp, vignetting_coeff=k, auto_recalibrated=True, star_list=star_list)
    # A failed recalibration carries a fossil star list and must not count
    ppr["FF_US005X_20261002_019999_000_0000000.fits"] = dict(
        mag_lev=99.0, vignetting_coeff=k, auto_recalibrated=False,
        star_list=[[0, 0, 0, 0, 0, 0, 1.0]]*50)
    with open(os.path.join(night_dir, "platepars_flux_recalibrated.json"), "w") as f:
        json.dump(ppr, f)


def testFingerprintAndItsChanges(tmp_path):

    from RMS.SensitivityMap import configFingerprint, fingerprintChanges

    cfg = FakeConfig(str(tmp_path))
    with open(cfg.camera_settings_path, "w") as f:
        f.write('{"gain": 10}')
    pp = PointedPlatepar()

    fp = configFingerprint(cfg, pp)
    assert fp["gamma"] == 0.5 and fp["X_res"] == X_RES and fp["pointing"] == [90.0, 30.0]
    assert fp["camera_settings"] is not None
    assert fingerprintChanges(fp, configFingerprint(cfg, pp)) == ([], [])

    # Units and geometry are hard; the extraction gate is soft; a small pointing jitter is nothing
    cfg.gamma = 1.0
    cfg.star_gate_factor = 4.0
    hard, soft = fingerprintChanges(fp, configFingerprint(cfg, PointedPlatepar(az=91.0)))
    assert any("gamma" in h for h in hard) and not any("pointing" in h for h in hard)
    assert any("star_gate_factor" in s for s in soft)

    # The camera settings file content and a real re-aim are hard
    cfg.gamma = 0.5
    cfg.star_gate_factor = 3.5
    with open(cfg.camera_settings_path, "w") as f:
        f.write('{"gain": 20}')
    hard, soft = fingerprintChanges(fp, configFingerprint(cfg, PointedPlatepar(az=100.0)))
    assert any("camera_settings" in h for h in hard) and any("pointing" in h for h in hard)
    assert soft == []

    # Keys missing on either side are not compared, and an unknown value (unreadable settings
    # file) is not a change
    assert fingerprintChanges({"gamma": 0.5}, {"bit_depth": 8}) == ([], [])
    assert fingerprintChanges({"camera_settings": "abc"}, {"camera_settings": None}) == ([], [])

    # The settings file next to the station config wins over a working-directory-relative path
    cfg.camera_settings_path = "./camera_settings.json"
    assert configFingerprint(cfg, pp)["camera_settings"] == configFingerprint(
        FakeConfig(str(tmp_path)), pp)["camera_settings"]


def testNightPhotometryAndStaleness(tmp_path):

    from RMS.SensitivityMap import (configFingerprint, nightPhotometry, mapStaleness,
                                    REFIT_AGE_DAYS, _areaMeanVignettingShift)

    cfg = FakeConfig(str(tmp_path))
    pp = PointedPlatepar()
    night = os.path.join(str(tmp_path), "US005X_20261002_010000_000000")
    _writeRecalibrated(night, zp=6.6, depth=4.9)

    ph = nightPhotometry(night)
    assert ph["zp"] == pytest.approx(6.6) and ph["depth"] == pytest.approx(4.9)
    assert ph["n_frames"] == 5 and ph["matched"] == 40
    assert nightPhotometry(os.path.join(str(tmp_path), "nowhere")) is None

    base = _mapDict([[4.0, 4.5], [4.2, 4.0]], mag_lev_ref=6.5, depth_ref=4.8,
                    fingerprint=configFingerprint(cfg, pp), fit_date="2099-01-01")

    # Consistent night: fresh
    assert mapStaleness(base, cfg, platepar=pp, night_dir=night) == ([], [])

    # Zero point an intensity epoch away: hard. Camera clearly deeper: soft
    _writeRecalibrated(night, zp=9.8, depth=4.9)
    hard, soft = mapStaleness(base, cfg, platepar=pp, night_dir=night)
    assert any("units" in h for h in hard) and soft == []
    _writeRecalibrated(night, zp=6.6, depth=5.6)
    hard, soft = mapStaleness(base, cfg, platepar=pp, night_dir=night)
    assert hard == [] and any("deeper" in s for s in soft)

    # The reference zero point follows a vignetting coefficient change, so a night recalibrated
    # under a new coefficient is still the same epoch
    shift = _areaMeanVignettingShift(X_RES, Y_RES, 0.0008, 0.00059)
    old_k_map = dict(base, vignetting_coeff_ref=0.0008, mag_lev_ref=6.6 - shift)
    _writeRecalibrated(night, zp=6.6, depth=4.9, k=0.00059)
    assert mapStaleness(old_k_map, cfg, platepar=pp, night_dir=night) == ([], [])

    # Age is soft; a hard fingerprint change is hard
    old = dict(base, fit_date="2000-01-01")
    hard, soft = mapStaleness(old, cfg, platepar=pp)
    assert hard == [] and any("age" in s and str(REFIT_AGE_DAYS) in s for s in soft)
    hard, soft = mapStaleness(base, cfg, platepar=PointedPlatepar(alt=40.0))
    assert any("pointing" in h for h in hard)


def testLoadPrefersNightCopyAndRefusesHardStaleStationMap(tmp_path):

    from RMS.SensitivityMap import configFingerprint

    cfg = FakeConfig(str(tmp_path))
    pp = PointedPlatepar()
    night = os.path.join(str(tmp_path), "US005X_20261002_010000_000000")
    _writeRecalibrated(night, zp=6.6, depth=4.9)

    station_map = _mapDict([[4.0, 4.5], [4.2, 4.0]], mag_lev_ref=6.5, fingerprint=configFingerprint(cfg, pp))
    with open(SensitivityMap.stationMapPath(cfg), "w") as f:
        json.dump(station_map, f)

    # Station map, consistent with the night: used
    m = SensitivityMap.load(cfg, dir_path=night, platepar=pp, night_dir=night)
    assert m is not None and m.source_path == SensitivityMap.stationMapPath(cfg)

    # An archived night copy wins, and is not subjected to the current-config check
    night_map = _mapDict([[3.0, 3.5], [3.2, 3.0]], mag_lev_ref=9.8, fingerprint={"gamma": 1.0})
    with open(SensitivityMap.nightCopyPath(night), "w") as f:
        json.dump(night_map, f)
    m = SensitivityMap.load(cfg, dir_path=night, platepar=pp, night_dir=night)
    assert m.lm_ref == pytest.approx(3.5) and m.source_path == SensitivityMap.nightCopyPath(night)

    # Without the night copy, a station map from another intensity epoch is refused
    os.remove(SensitivityMap.nightCopyPath(night))
    _writeRecalibrated(night, zp=9.8, depth=4.9)
    assert SensitivityMap.load(cfg, dir_path=night, platepar=pp, night_dir=night) is None

    # And so is one whose configuration fingerprint changed in a hard key
    _writeRecalibrated(night, zp=6.6, depth=4.9)
    cfg.bit_depth = 16
    assert SensitivityMap.load(cfg, platepar=pp) is None
    cfg.bit_depth = 8
    assert SensitivityMap.load(cfg, platepar=pp) is not None


def testCollectionAreaFileNameCarriesMapIdentity():

    base = generateColAreaJSONFileName("US005X", 20, 60.0, 130.0, 2.0, 10.0)
    tagged = generateColAreaJSONFileName("US005X", 20, 60.0, 130.0, 2.0, 10.0, map_tag="sensmap-2026-10-04-abc123")

    assert base.endswith("elemin-10.0.json")
    assert tagged.endswith("elemin-10.0_sensmap-2026-10-04-abc123.json")
    assert base != tagged
