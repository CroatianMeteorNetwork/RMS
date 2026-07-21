""" Tests for the experimental segmented cloud detector. """

from __future__ import absolute_import, division, print_function

import numpy as np

from Utils.SegmentedCloudDetector import (CELL_CLEAR, CELL_CLOUDY, CELL_NO_DATA,
    cloudCoverageSeries, computeCellSeries)


def _syntheticProduct(n_frames=30, n_side=24, width=1280.0, height=720.0, p=0.8,
        cloud_frames=(), cloud_box=None, p_scale=1.0, seed=7):
    """ Build a synthetic scoring product: a uniform star lattice, every star matched
        with probability p, except inside cloud_box during cloud_frames (never matched).
        p_scale mis-scales the STORED model P to emulate a stale dome model. """

    rng = np.random.RandomState(seed)

    xs = np.linspace(20, width - 20, n_side)
    ys = np.linspace(20, height - 20, n_side)
    gx, gy = np.meshgrid(xs, ys)
    gx, gy = gx.ravel(), gy.ravel()

    star_frame, star_x, star_y, star_p, calstars_row = [], [], [], [], []
    for f in range(n_frames):
        det = rng.uniform(size=len(gx)) < p
        if (f in cloud_frames) and (cloud_box is not None):
            x0, y0, x1, y1 = cloud_box
            in_cloud = (gx >= x0) & (gx < x1) & (gy >= y0) & (gy < y1)
            det = det & ~in_cloud
        star_frame.extend([f]*len(gx))
        star_x.extend(gx)
        star_y.extend(gy)
        star_p.extend([p*p_scale]*len(gx))
        calstars_row.extend(np.where(det, 1, -1))

    frames = dict(
        frame_names=np.array(["FF_TEST_{:04d}.fits".format(f) for f in range(n_frames)]),
        frame_time_unix=np.arange(n_frames)*300.0 + 1.6e9,
        sun_alt=np.full(n_frames, -30.0, dtype=np.float32),
        moon_alt=np.full(n_frames, -10.0, dtype=np.float32),
        moon_phase=np.zeros(n_frames, dtype=np.float32),
        n_detected=np.full(n_frames, len(gx), dtype=np.int32),
        in_flux_domain=np.ones(n_frames, dtype=bool),
    )
    stars = dict(
        star_frame=np.array(star_frame, dtype=np.int32),
        star_x=np.array(star_x, dtype=np.float32),
        star_y=np.array(star_y, dtype=np.float32),
        star_mag=np.zeros(len(star_x), dtype=np.float16),
        star_p=np.array(star_p),
        calstars_row=np.array(calstars_row, dtype=np.int32),
    )
    return frames, stars


def test_clear_night_all_clear():
    frames, stars = _syntheticProduct()
    result = computeCellSeries(frames, stars, nx=4, ny=3,
        width=1280.0, height=720.0)

    judged = result["verdict"][result["verdict"] != CELL_NO_DATA]
    assert len(judged)
    assert np.all(judged == CELL_CLEAR)

    cov = cloudCoverageSeries(result)
    assert np.nanmax(cov) == 0.0


def test_cloud_quadrant_flagged_only_there_and_then():
    cloud_frames = set(range(10, 20))
    # Top-left quadrant of a 1280x720 image
    frames, stars = _syntheticProduct(cloud_frames=cloud_frames,
        cloud_box=(0, 0, 640, 360))
    result = computeCellSeries(frames, stars, nx=4, ny=2,
        width=1280.0, height=720.0)

    verdict = result["verdict"]

    # Cells fully inside the cloud box: columns 0-1, row 0
    for f in range(12, 18):   # interior of the cloud window (smoothing softens edges)
        assert np.all(verdict[f, 0, :2] == CELL_CLOUDY), \
            "frame {:d}: {}".format(f, verdict[f])

    # The opposite quadrant stays clear throughout
    assert np.all(verdict[:, 1, 2:] == CELL_CLEAR)

    # Well before and after the cloud, everything judged is clear
    assert np.all(verdict[:8][verdict[:8] != CELL_NO_DATA] == CELL_CLEAR)
    assert np.all(verdict[22:][verdict[22:] != CELL_NO_DATA] == CELL_CLEAR)


def test_stale_model_normalization():
    # Stored P is half what it should be (stale shallow dome model, deep detection):
    # raw ratios ~2 - the nightly normalization must absorb it and still read clear
    frames, stars = _syntheticProduct(p_scale=0.5)
    result = computeCellSeries(frames, stars, nx=4, ny=3,
        width=1280.0, height=720.0)

    assert result["norm"] > 1.5
    judged = result["verdict"][result["verdict"] != CELL_NO_DATA]
    assert np.all(judged == CELL_CLEAR)


def test_empty_cells_carry_no_verdict():
    # Stars only in the left half - right-half cells must be no_data, not cloudy
    frames, stars = _syntheticProduct()
    keep = stars["star_x"] < 600.0
    stars = {k: v[keep] if len(v) == len(keep) else v for k, v in stars.items()}
    result = computeCellSeries(frames, stars, nx=4, ny=2,
        width=1280.0, height=720.0)

    assert np.all(result["verdict"][:, :, 3] == CELL_NO_DATA)
    left = result["verdict"][:, :, 0]
    assert np.all(left[left != CELL_NO_DATA] == CELL_CLEAR)
