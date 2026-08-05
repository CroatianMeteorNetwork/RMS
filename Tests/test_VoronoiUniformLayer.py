""" Uniform-layer regression for the tree estimator.

A uniform semi-transparent layer must NOT be absorbed into the baseline: the
night-1 seed once normalized per-star logits by the night's OWN detection
rates, so a 1.0 mag layer read 0.30 and a 2.0 layer read 1.17 (measured on
the real-geometry suite). With a trailing ratio history the seed norm comes
from PRIOR nights and the layer reads correctly. This test pins that.

Deep layers (3+ mag) intentionally saturate to the ceiling ("at least this
opaque") - see the OPEN ITEM note in Utils.VoronoiTreeEstimator.
"""

from __future__ import absolute_import, division, print_function

import json
import os

import numpy as np
import pytest


def _buildNight(tmp_path, config, dm_layer, n_frames=40, seed=5):
    """ Synthetic scoring product on REAL catalog stars (the tree builds its
        Voronoi anchors from catalog positions of the product's cat ids). """

    from RMS.Formats import StarCatalog

    catalog_stars, _, _ = StarCatalog.readStarCatalog(
        config.star_catalog_path, config.star_catalog_file, lim_mag=6.0,
        mag_band_ratios=config.star_catalog_band_ratios)
    ra, dec, mag = (catalog_stars[:, 0], catalog_stars[:, 1], catalog_stars[:, 2])

    # A contiguous sky patch with a healthy star count
    sel = np.where((ra > 40) & (ra < 80) & (dec > 10) & (dec < 40))[0][:220]
    n_star = len(sel)
    assert n_star >= 150

    rng = np.random.RandomState(seed)
    dome_s = 0.4
    p_dark = 1.0/(1.0 + np.exp(-(6.0 - mag[sel])/dome_s))
    p_dark = np.clip(p_dark, 0.05, 0.995)

    # Image positions: spread the patch over a 1280x720 frame (static)
    x = 40 + 1200*(ra[sel] - 40)/40.0
    y = 40 + 640*(dec[sel] - 10)/30.0

    logit = np.log(p_dark/(1 - p_dark))
    p_lay = 1.0/(1.0 + np.exp(-(logit - dm_layer/dome_s)))

    sf, cid, sx, sy, sp, rowv = [], [], [], [], [], []
    for f in range(n_frames):
        det = rng.uniform(size=n_star) < p_lay
        sf.extend([f]*n_star)
        cid.extend(sel)
        sx.extend(x); sy.extend(y)
        sp.extend(p_dark)
        rowv.extend(np.where(det, 1, -1))

    frames = dict(
        frame_names=np.array(["FF_T_{:04d}.fits".format(f) for f in range(n_frames)]),
        frame_time_unix=1.7e9 + np.arange(n_frames)*10.0,
        sun_alt=np.full(n_frames, -30.0, np.float32),
        moon_alt=np.full(n_frames, -20.0, np.float32),
        moon_phase=np.zeros(n_frames, np.float32),
        n_detected=np.full(n_frames, n_star, np.int32),
        in_flux_domain=np.ones(n_frames, bool),
    )
    stars = dict(
        star_frame=np.array(sf, np.int32), star_cat_id=np.array(cid, np.int32),
        star_x=np.array(sx, np.float32), star_y=np.array(sy, np.float32),
        star_mag=mag[np.array(cid)].astype(np.float16),
        star_p=np.array(sp, np.float32),
        star_flux_snr=np.full(len(sf), np.nan, np.float16),
        calstars_row=np.array(rowv, np.int32),
    )
    header = dict(schema_version=5, catalog_lim_mag=6.0, dome_s=dome_s,
        dome_fit_date="2026-08-01", stationID=config.stationID)

    # Trailing ratio history: six calibrated prior nights for this model
    hist = {"202607{:02d}".format(20 + i): dict(dratio=1.0, dmodel="2026-08-01")
            for i in range(6)}
    with open(os.path.join(str(tmp_path),
            "{:s}_flux_lm_history.json".format(config.stationID)), "w") as f:
        json.dump(hist, f)

    return header, frames, stars


def test_uniform_layer_not_absorbed(tmp_path):
    import RMS.ConfigReader as cr
    from Utils.VoronoiTreeEstimator import computeTreeSeries

    config = cr.parse(".config")
    config.stationID = "US005X"
    config.data_dir = str(tmp_path)

    # Uniform 1.5 mag layer: must read near truth, not near half of it
    header, frames, stars = _buildNight(tmp_path, config, 1.5)
    out = computeTreeSeries(config, str(tmp_path), header, frames, stars)
    assert out is not None
    _, dm_cells, _, _, _ = out
    med = float(np.nanmedian(dm_cells[5:-5]))
    assert 1.1 <= med <= 2.1, med

    # Clear night through the same path stays clear
    header, frames, stars = _buildNight(tmp_path, config, 0.0)
    out = computeTreeSeries(config, str(tmp_path), header, frames, stars)
    _, dm_cells, _, _, _ = out
    assert float(np.nanmedian(dm_cells[5:-5])) <= 0.25
