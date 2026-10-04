""" Tests for the sensitivity map's self-maintenance (Utils.FitCameraSensitivityMap): selecting
training nights from the camera's current intensity epoch, and the nightly ensure step that refits
a missing or stale map, rate-limits attempts, and never installs a fit that does not describe the
camera as it is now.
"""

from __future__ import absolute_import, division, print_function

import json
import os

import numpy as np
import pytest

import Utils.FitCameraSensitivityMap as fcs
from RMS.SensitivityMap import SensitivityMap, MIN_TRIALS_PER_BLOCK

X_RES, Y_RES = 1920, 1080


class Cfg(object):
    def __init__(self, data_dir, station="US005X"):
        self.data_dir = data_dir
        self.stationID = station
        self.gamma = 0.5
        self.bit_depth = 8
        self.fps = 25.0
        self.star_gate_factor = 3.5
        self.segment_radius = 6
        self.max_feature_ratio = 0.8
        self.roundness_threshold = 0.5
        self.camera_settings_path = os.path.join(data_dir, "camera_settings.json")
        self.config_file_path = data_dir


class FakePlatepar(object):
    def __init__(self, k=0.00059, az=90.0, alt=30.0):
        self.X_res, self.Y_res = X_RES, Y_RES
        self.vignetting_coeff = k
        self.az_centre, self.alt_centre = az, alt


def _writeNight(archive, name, zp, depth, matched, k=0.00059, n_frames=12):
    """ A night directory with flux-recalibrated platepars of the given photometry. """

    d = os.path.join(archive, name)
    os.makedirs(d)
    rng = np.random.RandomState(abs(hash(name)) % 1000)
    ppr = {}
    for i in range(n_frames):
        mags = rng.uniform(depth - 3.0, depth, matched)
        # Pin the 90th percentile at the requested depth so the statistic is exact
        mags[-1] = depth
        star_list = [[2461000.5, 100.0, 100.0, 50.0, 10.0, 20.0, float(m)] for m in sorted(mags)]
        ppr["FF_US005X_2026100{:d}_01{:04d}_000_0000000.fits".format(1, i)] = dict(
            mag_lev=zp, vignetting_coeff=k, auto_recalibrated=True, star_list=star_list)
    with open(os.path.join(d, "platepars_flux_recalibrated.json"), "w") as f:
        json.dump(ppr, f)
    return d


def _mapDict(best=4.5, zp=6.5, depth=4.9, fingerprint=None, fit_date="2026-10-04", k=0.00059):
    lm = [best - 0.3, best, best - 0.2, best - 0.5, best - 0.8, best - 0.4, best - 0.3, best - 0.9,
          best - 1.2, best - 1.0, best - 1.1, best - 1.4]
    return dict(stationID="US005X", nbx=4, nby=3, X_res=X_RES, Y_res=Y_RES, LM=lm, s=0.42,
                n_trials=MIN_TRIALS_PER_BLOCK*12 + 1, fit_date=fit_date, nights=["n1"],
                mag_lev_ref=zp, vignetting_coeff_ref=k, depth_ref=depth, fingerprint=fingerprint,
                pointing=[90.0, 30.0])


def testSelectMapNightsKeepsCurrentEpochAndClearest(tmp_path):

    cfg = Cfg(str(tmp_path))
    archive = os.path.join(str(tmp_path), "ArchivedFiles")
    os.makedirs(archive)

    # Old firmware epoch: zero point 3 mag higher, clear nights
    _writeNight(archive, "US005X_20260919_010000_000000", zp=9.8, depth=4.8, matched=120)
    _writeNight(archive, "US005X_20260920_010000_000000", zp=9.8, depth=4.8, matched=130)
    # New epoch: one hazy night, one shallow (pre gate fix) night, two clear nights
    _writeNight(archive, "US005X_20260926_010000_000000", zp=6.5, depth=3.9, matched=30)
    _writeNight(archive, "US005X_20260930_010000_000000", zp=6.5, depth=4.1, matched=60)
    _writeNight(archive, "US005X_20261002_010000_000000", zp=6.6, depth=4.9, matched=90)
    _writeNight(archive, "US005X_20261003_010000_000000", zp=6.5, depth=4.95, matched=85)
    # Too few frames to be a candidate
    _writeNight(archive, "US005X_20261004_010000_000000", zp=6.5, depth=4.9, matched=100, n_frames=3)

    nights = fcs.selectMapNights(cfg)
    names = [os.path.basename(n)[7:15] for n in nights]

    # Only the new epoch, only nights within the depth tolerance of the deepest, clearest first
    assert names == ["20261002", "20261003"]

    # An explicit reference zero point selects the old epoch instead
    nights_old = fcs.selectMapNights(cfg, reference_zp=9.8)
    assert sorted(os.path.basename(n)[7:15] for n in nights_old) == ["20260919", "20260920"]

    # No archive: nothing
    assert fcs.selectMapNights(Cfg(os.path.join(str(tmp_path), "nowhere"))) == []


def testEnsureFitsMissingMapAndRateLimits(tmp_path, monkeypatch):

    cfg = Cfg(str(tmp_path))
    archive = os.path.join(str(tmp_path), "ArchivedFiles")
    os.makedirs(archive)
    _writeNight(archive, "US005X_20261002_010000_000000", zp=6.6, depth=4.9, matched=90)
    pp = FakePlatepar()

    calls = []

    def fakeFit(config, night_dirs, nbx=4, nby=3, lim_mag=8.0):
        calls.append(list(night_dirs))
        from RMS.SensitivityMap import configFingerprint
        return _mapDict(fingerprint=configFingerprint(config, pp))

    monkeypatch.setattr(fcs, "fitSensitivityMap", fakeFit)

    # No map: fitted and installed
    assert fcs.ensureSensitivityMap(cfg, platepar=pp) is True
    assert len(calls) == 1
    assert os.path.isfile(SensitivityMap.stationMapPath(cfg))
    assert SensitivityMap.load(cfg, platepar=pp) is not None

    # Fresh map: nothing to do, no second fit
    assert fcs.ensureSensitivityMap(cfg, platepar=pp) is True
    assert len(calls) == 1

    # A hard change (gamma) retires the map; the daily marker blocks a second attempt today
    cfg.gamma = 1.0
    assert SensitivityMap.load(cfg, platepar=pp) is None
    assert fcs.ensureSensitivityMap(cfg, platepar=pp) is False
    assert len(calls) == 1

    # Next day: the refit runs and the map is usable again
    marker = os.path.join(str(tmp_path), "US005X_" + fcs.AUTO_ATTEMPT_MARKER)
    with open(marker, "w") as f:
        json.dump(dict(date="2000-01-01"), f)
    assert fcs.ensureSensitivityMap(cfg, platepar=pp) is True
    assert len(calls) == 2
    assert SensitivityMap.load(cfg, platepar=pp) is not None


def testEnsureKeepsUsableMapWhenRefitFails(tmp_path, monkeypatch):

    cfg = Cfg(str(tmp_path))
    archive = os.path.join(str(tmp_path), "ArchivedFiles")
    os.makedirs(archive)
    _writeNight(archive, "US005X_20261002_010000_000000", zp=6.6, depth=4.9, matched=90)
    pp = FakePlatepar()

    from RMS.SensitivityMap import configFingerprint
    installed = _mapDict(fingerprint=configFingerprint(cfg, pp), fit_date="2026-01-01")   # old: soft-stale
    with open(SensitivityMap.stationMapPath(cfg), "w") as f:
        json.dump(installed, f)

    # A failed fit leaves the (soft-stale but usable) map in place
    monkeypatch.setattr(fcs, "fitSensitivityMap", lambda *a, **k: None)
    assert fcs.ensureSensitivityMap(cfg, platepar=pp) is True
    with open(SensitivityMap.stationMapPath(cfg)) as f:
        assert json.load(f)["fit_date"] == "2026-01-01"

    # A fit that does not describe the current camera (pointing moved since the archive) is not
    # installed either
    with open(os.path.join(str(tmp_path), "US005X_" + fcs.AUTO_ATTEMPT_MARKER), "w") as f:
        json.dump(dict(date="2000-01-01"), f)
    moved = _mapDict(fingerprint=configFingerprint(cfg, FakePlatepar(az=120.0)))
    monkeypatch.setattr(fcs, "fitSensitivityMap", lambda *a, **k: moved)
    assert fcs.ensureSensitivityMap(cfg, platepar=pp) is True
    with open(SensitivityMap.stationMapPath(cfg)) as f:
        assert json.load(f)["fit_date"] == "2026-01-01"

    # Without any usable map and a failing fit, the answer is honest
    os.remove(SensitivityMap.stationMapPath(cfg))
    with open(os.path.join(str(tmp_path), "US005X_" + fcs.AUTO_ATTEMPT_MARKER), "w") as f:
        json.dump(dict(date="2000-01-01"), f)
    monkeypatch.setattr(fcs, "fitSensitivityMap", lambda *a, **k: None)
    assert fcs.ensureSensitivityMap(cfg, platepar=pp) is False
