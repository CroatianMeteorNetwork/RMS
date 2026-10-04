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


def testCollectionAreaFileNameCarriesMapIdentity():

    base = generateColAreaJSONFileName("US005X", 20, 60.0, 130.0, 2.0, 10.0)
    tagged = generateColAreaJSONFileName("US005X", 20, 60.0, 130.0, 2.0, 10.0, map_tag="sensmap-2026-10-04-abc123")

    assert base.endswith("elemin-10.0.json")
    assert tagged.endswith("elemin-10.0_sensmap-2026-10-04-abc123.json")
    assert base != tagged
