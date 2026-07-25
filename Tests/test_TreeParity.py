""" Regression gate for the Voronoi tree estimator against the validated
harness reference.

The port-by-rewrite of the harness estimator regressed twice in ways unit
tests could not see (a photometric zero-point convention and a missing
normalization in the BP message smoothing) - both only visible as a
disagreement with the harness's leaf-level output on a real night. This test
re-runs computeTreeSeries on a pinned dark slice of a reference night and
diffs the leaf dm series against the harness dump.

The fixture (a processed night directory plus the harness leaf dump) is far
too large for the repository, so the test is skipped unless
RMS_TREE_PARITY_DIR points at a fixture directory containing:

    <NIGHT_NAME>/              the processed night, under its real name
                               (scoring product, sidecar, CALSTARS,
                               platepars_flux_recalibrated.json, .config)
    harness_leafdump.npz       harness leaf_dm [n_frames, n_leaf], anchor_ids
    slice.json                 {"start": int, "end": int} pinned frame range

Build it with Utils/StarCalibration + the parity notes in the investigation
directory (see the tree-port memory).
"""

import glob
import json
import os

import numpy as np
import pytest


FIXTURE_ENV = "RMS_TREE_PARITY_DIR"

# Gates measured at parity (A6 20260721, commit e6ccf6e1): thin-band medians
# agreed to 0.06 and leaf correlation was 0.89 on the pinned slice. The
# tolerances leave room for numeric drift, not for a channel dying.
THIN_BAND_TOL = 0.15
MIN_CORR = 0.75


@pytest.mark.skipif(FIXTURE_ENV not in os.environ,
    reason="set {} to the parity fixture directory".format(FIXTURE_ENV))
def testTreeSliceParity():

    import RMS.ConfigReader as cr
    from RMS.Formats.StarScoring import loadStarScoring, scoringFileName
    from Utils.StarCalibration import computeNightStarStats
    from Utils.StillsSampler import fuseSidecarDetections
    from Utils.VoronoiTreeEstimator import computeTreeSeries

    fix = os.environ[FIXTURE_ENV]
    night_dirs = [d for d in glob.glob(os.path.join(fix, "*"))
                  if os.path.isdir(d)]
    assert len(night_dirs) == 1, "fixture must hold exactly one night dir"
    night = night_dirs[0]
    with open(os.path.join(fix, "slice.json")) as f:
        sl = json.load(f)
    s0, s1 = int(sl["start"]), int(sl["end"])

    config = cr.parse(os.path.join(night, ".config"))
    config.data_dir = fix

    night_name = os.path.basename(os.path.normpath(night))
    header, frames, stars = loadStarScoring(
        os.path.join(night, scoringFileName(night_name)))
    stars = fuseSidecarDetections(night, frames, stars)
    stats = computeNightStarStats(config, night)
    assert stats is not None, "night calibration failed"

    ref = np.load(os.path.join(fix, "harness_leafdump.npz"))
    h_leaf = ref["leaf_dm"].astype(np.float32)
    h_ids = ref["anchor_ids"].astype(np.int64)

    n_full = len(h_leaf)
    sub_frames = {k: (np.asarray(v)[s0:s1]
                      if np.ndim(v) and len(np.asarray(v)) == n_full else v)
                  for k, v in frames.items()}
    sf = np.asarray(stars["star_frame"])
    keep = (sf >= s0) & (sf < s1)
    sub_stars = {k: np.asarray(v)[keep] for k, v in stars.items()}
    sub_stars["star_frame"] = sub_stars["star_frame"] - s0

    calibration = (
        dict(catalog_lim_mag=stats["catalog_lim_mag"], k_ema=stats["k_fit"]),
        dict(rate_calstars=stats["rate_calstars"],
             rate_forced=stats["rate_forced"],
             base_mag=stats["base_mag"], sigma_mag=stats["sigma_mag"]))

    result = computeTreeSeries(config, night, header, sub_frames, sub_stars,
        calibration=calibration)
    assert result is not None, "tree estimator returned no result"
    _, _, _, leaf_ids, leaf_dm = result

    _, hi, oi = np.intersect1d(h_ids, leaf_ids, return_indices=True)
    a = h_leaf[s0:s1][:, hi]
    b = leaf_dm.astype(np.float32)[:, oi]
    both = np.isfinite(a) & np.isfinite(b)
    assert both.sum() > 1000, "too few comparable leaf samples"

    thin = both & (a > 0.3) & (a < 1.0)
    clear = both & (a < 0.15)
    thin_ops = float(np.median(b[thin]))
    thin_ref = float(np.median(a[thin]))
    corr = float(np.corrcoef(a[both], b[both])[0, 1])

    assert abs(thin_ops - thin_ref) < THIN_BAND_TOL, \
        "thin-band median {:.2f} vs harness {:.2f}".format(thin_ops, thin_ref)
    assert float(np.median(b[clear])) < 0.15, "clear sky reads cloudy"
    assert corr > MIN_CORR, "leaf correlation {:.2f}".format(corr)
