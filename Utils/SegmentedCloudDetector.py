""" EXPERIMENTAL - Segmented (per-sky-cell) cloud detection from the nightly star
scoring product.

Consumes <night>_star_scoring.npz (see RMS.Formats.StarScoring): per-star projected
positions, catalog magnitudes, dome-model detection probabilities P and CALSTARS match
flags for EVERY scored frame, ungated. This detector applies its own domain judgment -
that is the point of the score-everything/gate-at-the-verdict architecture.

Method (catalog-first, statistics-pooled):
    - The image is divided into a fixed grid of cells. Per frame and cell,
      observed = number of matched catalog stars, expected = sum of the dome model's
      per-star P. Pooling many stars per cell is what makes a verdict statistical -
      a single star is a coin flip (scintillation, near-threshold flicker).
    - A nightly normalization (high percentile of the per-frame whole-image
      observed/expected ratio, bounded) absorbs model-epoch miscalibration - e.g. a
      dome model fit on shallower detections than tonight's - without letting an
      overcast night normalize itself into "clear".
    - Per cell, a normal approximation to the Poisson-binomial gives
      z = (obs - exp)/sqrt(sum p*(1-p)); a cell is cloudy when the deficit is both
      statistically significant and large, clear when the ratio is high, and
      carries no verdict when too little expectation falls in it.
    - Verdicts are time-smoothed with a short median filter per cell.

Geometry note: cells live in IMAGE coordinates. For a fixed camera, clouds cross the
image slowly while the star field drifts through it - each cell sees a changing star
population, which is fine: the expectation moves with the stars.

The related ClearSkyDetector draft (branch fix-clearsky-fov-filter) pioneered the
catalog-first segmentation idea with per-star Voronoi cells, but re-detected stars
itself with a fixed intensity threshold and judged visibility against a fixed limiting
magnitude. This detector replaces that measurement layer with the CALSTARS-derived
matches and dome-model expectations carried by the scoring product.

Usage:
    python -m Utils.SegmentedCloudDetector /path/to/<night>_star_scoring.npz \\
        [--grid 8x5] [--plot out.png]
"""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os

import numpy as np

from RMS.Formats.StarScoring import loadStarScoring


# Verdict codes (per frame, per cell)
CELL_NO_DATA = 0     # too little expectation in the cell to judge
CELL_CLEAR = 1
CELL_UNCERTAIN = 2   # deficit present but not decisive (thin/partial cloud, noise)
CELL_CLOUDY = 3

VERDICT_NAMES = {CELL_NO_DATA: "no_data", CELL_CLEAR: "clear",
                 CELL_UNCERTAIN: "uncertain", CELL_CLOUDY: "cloudy"}

# Statistical thresholds
EXP_MIN = 2.0          # minimum summed expectation for a cell to carry a verdict
Z_CLOUDY = 2.5         # deficit significance required to call a cell cloudy
RATIO_CLOUDY = 0.6     # ...and the ratio must actually be low (significance alone can
                       # trip on a mildly-low, very-well-populated cell)
RATIO_CLEAR = 0.75     # ratio at or above which a cell is called clear

# Nightly PER-CELL normalization (intra-night analog of Flux.domeRatioNormalization):
# each cell baselines against a high percentile of ITS OWN dark-hours ratio series, so
# a cloudy minority of frames is read through. Per-cell (not global) because the dome
# model carries a static spatial bias - one LM0 per camera cannot represent the
# center-to-corner sensitivity falloff (~0.8 mag measured), which otherwise reads as
# permanent "cloud" at the image edges. A static bias is not a cloud; only temporal
# deviation from the cell's own baseline is. The bounds stop a fully-overcast night
# from normalizing itself into "clear".
NORM_PCT = 80
NORM_MIN = 0.5
NORM_MAX = 3.0
NORM_CELL_MIN_FRAMES = 5   # baseline frames a cell needs for its own norm (else the
                           # global norm is used)
NORM_FRAME_EXP_MIN = 5.0   # frames with less total expectation don't inform the norm

SMOOTH_FRAMES = 3      # median-filter window (frames) applied to z per cell


def computeCellSeries(frames, stars, nx=8, ny=5, width=None, height=None):
    """ Pool the scoring product into per-frame, per-cell statistics and verdicts.

    Arguments:
        frames: [dict of ndarray] Frame arrays from loadStarScoring.
        stars: [dict of ndarray] Star arrays from loadStarScoring.

    Keyword arguments:
        nx, ny: [int] Grid cells across image width/height.
        width, height: [float] Image dimensions; inferred from the star positions
            (rounded up to a multiple of 16) when not given.

    Return:
        result: [dict]
            obs [n_frames, ny, nx] - matched star counts,
            exp [n_frames, ny, nx] - normalized summed P,
            var [n_frames, ny, nx] - Poisson-binomial variance of obs,
            ratio, z - derived fields (NaN where exp == 0),
            verdict [n_frames, ny, nx] - CELL_* codes,
            norm [float] - global fallback normalization,
            cell_norm [ny, nx] - per-cell normalization actually applied,
            nx, ny, width, height.
    """

    n_frames = len(frames["frame_names"])

    x = np.asarray(stars["star_x"], dtype=np.float64)
    y = np.asarray(stars["star_y"], dtype=np.float64)
    p = np.asarray(stars["star_p"], dtype=np.float64)
    fidx = np.asarray(stars["star_frame"], dtype=np.int64)
    matched = np.asarray(stars["calstars_row"], dtype=np.int64) >= 0

    if width is None:
        width = float(np.ceil(np.max(x)/16.0)*16.0) if len(x) else 1280.0
    if height is None:
        height = float(np.ceil(np.max(y)/16.0)*16.0) if len(y) else 720.0

    cx = np.clip((x/(width/nx)).astype(np.int64), 0, nx - 1)
    cy = np.clip((y/(height/ny)).astype(np.int64), 0, ny - 1)

    # Flat bin index over (frame, cell_y, cell_x) for one-pass accumulation
    cell = cy*nx + cx
    flat = fidx*(ny*nx) + cell
    n_bins = n_frames*ny*nx

    obs = np.bincount(flat[matched], minlength=n_bins).astype(np.float64)
    exp_raw = np.bincount(flat, weights=p, minlength=n_bins)

    obs_fc = obs.reshape(n_frames, ny*nx)
    exp_fc = exp_raw.reshape(n_frames, ny*nx)

    # Baseline frames for the norms: the fully dark hours (twilight frames carry the
    # brightened-sky regime and would inflate a high-percentile baseline)
    dark = np.asarray(frames["sun_alt"], dtype=np.float64) <= -18.0
    if not np.any(dark):
        dark = np.ones(n_frames, dtype=bool)

    # Global fallback norm from whole-image per-frame ratios
    frame_obs = obs_fc.sum(axis=1)
    frame_exp = exp_fc.sum(axis=1)
    informative = dark & (frame_exp >= NORM_FRAME_EXP_MIN)
    if np.any(informative):
        norm = float(np.clip(np.percentile(
            frame_obs[informative]/frame_exp[informative], NORM_PCT), NORM_MIN, NORM_MAX))
    else:
        norm = 1.0

    # Per-cell norms from each cell's own dark-hours ratio series
    cell_norm = np.full(ny*nx, norm)
    for c in range(ny*nx):
        ok = dark & (exp_fc[:, c] >= EXP_MIN)
        if ok.sum() >= NORM_CELL_MIN_FRAMES:
            cell_norm[c] = np.clip(np.percentile(
                obs_fc[ok, c]/exp_fc[ok, c], NORM_PCT), NORM_MIN, NORM_MAX)

    # Scale each star's P by its cell's norm (clipped to stay a probability), then
    # re-pool - the variance must be computed from the scaled per-star probabilities
    p_s = np.clip(p*cell_norm[cell], 0.0, 0.999)
    exp = np.bincount(flat, weights=p_s, minlength=n_bins)
    var = np.bincount(flat, weights=p_s*(1.0 - p_s), minlength=n_bins)

    obs = obs.reshape(n_frames, ny, nx)
    exp = exp.reshape(n_frames, ny, nx)
    var = var.reshape(n_frames, ny, nx)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(exp > 0, obs/exp, np.nan)
        z = np.where(var > 0, (obs - exp)/np.sqrt(var), np.nan)

    # Median-smooth z and ratio over time per cell (rolling window, edges shrink)
    z_s = _rollingMedian(z, SMOOTH_FRAMES)
    ratio_s = _rollingMedian(ratio, SMOOTH_FRAMES)

    verdict = np.full((n_frames, ny, nx), CELL_UNCERTAIN, dtype=np.int8)
    verdict[exp < EXP_MIN] = CELL_NO_DATA
    judged = exp >= EXP_MIN
    verdict[judged & (ratio_s >= RATIO_CLEAR)] = CELL_CLEAR
    verdict[judged & (z_s <= -Z_CLOUDY) & (ratio_s < RATIO_CLOUDY)] = CELL_CLOUDY

    return dict(obs=obs, exp=exp, var=var, ratio=ratio, z=z, verdict=verdict,
        norm=norm, cell_norm=cell_norm.reshape(ny, nx), nx=nx, ny=ny,
        width=width, height=height)


def _rollingMedian(a, window):
    """ Median filter along axis 0 with a centered window; NaNs propagate as NaN-aware
        medians (all-NaN windows stay NaN). """

    if window <= 1:
        return a

    half = window//2
    out = np.empty_like(a)
    n = a.shape[0]
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        with np.errstate(all="ignore"):
            out[i] = np.nanmedian(a[lo:hi], axis=0)
    return out


def cloudCoverageSeries(result):
    """ Per-frame cloud coverage: fraction of judged cells that are cloudy.

    Return:
        coverage: [ndarray n_frames] NaN where no cell carried a verdict.
    """

    verdict = result["verdict"]
    judged = (verdict != CELL_NO_DATA).sum(axis=(1, 2)).astype(np.float64)
    cloudy = (verdict == CELL_CLOUDY).sum(axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(judged > 0, cloudy/judged, np.nan)


def plotCellRaster(frames, result, out_path, title=None):
    """ Diagnostic raster: cells (rows) vs time (columns), colored by smoothed ratio,
        with cloudy verdicts marked and no-data cells blanked. """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import datetime

    n_frames, ny, nx = result["verdict"].shape
    times = [datetime.datetime.utcfromtimestamp(t) for t in frames["frame_time_unix"]]
    t_num = mdates.date2num(times)

    ratio = np.where(result["exp"] >= EXP_MIN, result["ratio"], np.nan)
    raster = ratio.reshape(n_frames, ny*nx).T
    verdict = result["verdict"].reshape(n_frames, ny*nx).T

    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
        gridspec_kw=dict(height_ratios=[1, 2.6]))

    # Top: whole-image ratio + cloud coverage
    frame_obs = result["obs"].sum(axis=(1, 2))
    frame_exp = result["exp"].sum(axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        whole = np.where(frame_exp > 0, frame_obs/frame_exp, np.nan)
    ax[0].plot(t_num, whole, "k.", ms=3, label="Image ratio (normalized)")
    cov = cloudCoverageSeries(result)
    ax[0].plot(t_num, cov, "-", color="firebrick", lw=1.2, label="Cloudy cell fraction")
    ax[0].axhline(1.0, color="gray", lw=0.5)
    ax[0].set_ylim(0, 1.6)
    ax[0].legend(fontsize=8, loc="upper right", ncol=2)
    ax[0].set_ylabel("Ratio / fraction")

    # Sun/moon context shading
    sun = frames["sun_alt"]
    moon_up = (frames["moon_alt"] > 0) & (frames["moon_phase"] > 25.0)
    for a in ax:
        _shade(a, t_num, sun > -18.0, "peachpuff")
        _shade(a, t_num, moon_up, "thistle")

    # Bottom: the cell raster
    extent = [t_num[0], t_num[-1], ny*nx - 0.5, -0.5]
    im = ax[1].imshow(raster, aspect="auto", extent=extent, cmap="viridis",
        vmin=0.0, vmax=1.4, interpolation="nearest")
    # Cloudy marks
    rows, cols = np.where(verdict == CELL_CLOUDY)
    if len(rows):
        ax[1].plot(t_num[cols], rows, "x", color="red", ms=3, mew=0.8, ls="none")
    ax[1].set_ylabel("Cell (row-major, {:d}x{:d})".format(ny, nx))
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax[1].set_xlabel("Time (UTC)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, label="Cell obs/exp")

    if title:
        ax[0].set_title(title, fontsize=11)

    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    return out_path


def _shade(ax, t_num, flag, color):
    """ Shade contiguous True runs of flag along the time axis. """

    if not np.any(flag):
        return
    idx = np.where(flag)[0]
    splits = np.where(np.diff(idx) > 1)[0]
    for run in np.split(idx, splits + 1):
        ax.axvspan(t_num[run[0]], t_num[run[-1]], color=color, alpha=0.45, zorder=0)


def main():
    parser = argparse.ArgumentParser(description="Segmented cloud detection from a "
        "nightly star scoring product (EXPERIMENTAL).")
    parser.add_argument("npz_path", help="Path to <night>_star_scoring.npz")
    parser.add_argument("--grid", default="8x5", help="Cell grid as NXxNY (default 8x5)")
    parser.add_argument("--plot", default=None, help="Output plot path (default: "
        "alongside the npz)")
    args = parser.parse_args()

    nx, ny = (int(v) for v in args.grid.lower().split("x"))

    header, frames, stars = loadStarScoring(args.npz_path)
    result = computeCellSeries(frames, stars, nx=nx, ny=ny)

    cov = cloudCoverageSeries(result)
    judged = np.isfinite(cov)
    print("Night: {:s}  frames: {:d}  norm: {:.2f}".format(
        str(header.get("night", "?")), len(frames["frame_names"]), result["norm"]))
    if np.any(judged):
        print("Cloudy-cell fraction: median {:.2f}, max {:.2f} over {:d} judged frames".format(
            float(np.nanmedian(cov)), float(np.nanmax(cov)), int(judged.sum())))

    out_path = args.plot
    if out_path is None:
        out_path = args.npz_path.replace("_star_scoring.npz", "_cloud_cells.png")
    plotCellRaster(frames, result, out_path,
        title="{:s} - segmented cloud detection (norm {:.2f})".format(
            str(header.get("night", "?")), result["norm"]))
    print("Plot: {:s}".format(out_path))


if __name__ == "__main__":
    main()
