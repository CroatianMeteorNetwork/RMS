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

DM_WINDOW = 3          # bins pooled (centered) per extinction estimate - clouds
                       # persist across bins, and pooling is what buys the inversion
                       # its statistics at depth-6 star densities
DM_MIN_STARS = 10      # pooled star RECORDS the inversion needs. A raw count, not an
                       # expectation: the per-cell norm (capped at 3) can inflate a
                       # mostly-obstructed cell's few-star expectation over any
                       # expectation floor, while an honest mid-sky cell with many
                       # faint stars fails it - the record count cannot be gamed


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


def extinctionSeries(frames, stars, result, dome_s):
    """ Invert each cell's pooled star-count deficit into an extinction estimate.

    A uniform extinction dm over a cell shifts every star's detection argument:
    P_i(dm) = sigmoid(logit(P_i) - dm/s). Per frame and cell, the dm that makes the
    expected count match the observed count is found by bisection (the expectation is
    monotone in dm). This is a TRANSPARENCY score in magnitudes - far more informative
    in the thin-cloud regime than a count ratio, whose response saturates through the
    logistic. Cells with a count excess get dm = 0 (no negative extinction).

    Note: the per-cell normalization is applied to P before inversion, so static
    image-plane sensitivity structure does not read as extinction.

    Arguments:
        frames, stars: [dict] Arrays from loadStarScoring.
        result: [dict] Output of computeCellSeries (same grid).
        dome_s: [float] Logistic rolloff width of the night's dome model (mag) -
            converts the logit shift into magnitudes.

    Return:
        dm: [ndarray n_frames, ny, nx] Extinction (mag); NaN where exp < EXP_MIN.
    """

    n_frames = len(frames["frame_names"])
    nx, ny = result["nx"], result["ny"]
    width, height = result["width"], result["height"]

    x = np.asarray(stars["star_x"], dtype=np.float64)
    y = np.asarray(stars["star_y"], dtype=np.float64)
    p_raw = np.asarray(stars["star_p"], dtype=np.float64)
    fidx = np.asarray(stars["star_frame"], dtype=np.int64)

    cx = np.clip((x/(width/nx)).astype(np.int64), 0, nx - 1)
    cy = np.clip((y/(height/ny)).astype(np.int64), 0, ny - 1)
    cell = cy*nx + cx

    cell_norm = result["cell_norm"].ravel()
    p_s = np.clip(p_raw*cell_norm[cell], 1e-4, 0.999)
    logit = np.log(p_s/(1.0 - p_s))

    # Group star records by (frame, cell) for the per-bin inversions
    flat = fidx*(ny*nx) + cell
    order = np.argsort(flat, kind="stable")
    flat_sorted = flat[order]
    logit_sorted = logit[order]
    bin_starts = np.searchsorted(flat_sorted, np.arange(n_frames*ny*nx))
    bin_ends = np.searchsorted(flat_sorted, np.arange(n_frames*ny*nx) + 1)

    obs = result["obs"].reshape(-1)

    n_cells = ny*nx
    half = DM_WINDOW//2

    dm = np.full(n_frames*n_cells, np.nan)
    for f in range(n_frames):
        for c in range(n_cells):

            # Pool the centered window of bins for this cell
            pool = []
            n_obs = 0.0
            for k in range(max(0, f - half), min(n_frames, f + half + 1)):
                b = k*n_cells + c
                pool.append(logit_sorted[bin_starts[b]:bin_ends[b]])
                n_obs += obs[b]
            l = np.concatenate(pool) if pool else np.array([])

            if len(l) < DM_MIN_STARS:
                continue

            # No deficit: fully transparent
            if np.sum(1.0/(1.0 + np.exp(-l))) <= n_obs:
                dm[f*n_cells + c] = 0.0
                continue

            # Bisection on the logit shift (expectation is monotone decreasing in it)
            lo, hi = 0.0, 25.0
            for _ in range(40):
                mid = 0.5*(lo + hi)
                if np.sum(1.0/(1.0 + np.exp(-(l - mid)))) > n_obs:
                    lo = mid
                else:
                    hi = mid
            dm[f*n_cells + c] = 0.5*(lo + hi)*dome_s

    return dm.reshape(n_frames, ny, nx)


def animateTransparency(frames, result, dm, out_path, fps=6, title=None):
    """ Animated per-cell extinction map over the night (GIF).

    Arguments:
        frames: [dict] Frame arrays from loadStarScoring.
        result: [dict] Output of computeCellSeries.
        dm: [ndarray] Output of extinctionSeries.
        out_path: [str] Output .gif path.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    import datetime

    n_frames = dm.shape[0]
    times = [datetime.datetime.utcfromtimestamp(t) for t in frames["frame_time_unix"]]
    sun = frames["sun_alt"]

    fig, (ax, ax_t) = plt.subplots(2, 1, figsize=(8, 6.2),
        gridspec_kw=dict(height_ratios=[5, 1]))

    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("lightgray")
    im = ax.imshow(np.ma.masked_invalid(dm[0]), cmap=cmap, vmin=0.0, vmax=3.0,
        interpolation="nearest", aspect="auto",
        extent=[0, result["width"], result["height"], 0])
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Extinction (mag)")
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    txt = ax.set_title("")

    # Time strip: mean extinction with a moving cursor
    with np.errstate(all="ignore"):
        mean_dm = np.nanmean(dm.reshape(n_frames, -1), axis=1)
    ax_t.plot(times, mean_dm, "-", color="steelblue", lw=1.0)
    ax_t.set_ylabel("Mean\n(mag)", fontsize=8)
    ax_t.set_ylim(bottom=0)
    cursor = ax_t.axvline(times[0], color="red", lw=1.2)
    for lbl in ax_t.get_xticklabels():
        lbl.set_fontsize(7)

    def update(i):
        im.set_array(np.ma.masked_invalid(dm[i]))
        txt.set_text("{:s}{:s} UTC   sun {:+.1f} deg".format(
            (title + "   ") if title else "",
            times[i].strftime("%H:%M:%S"), float(sun[i])))
        cursor.set_xdata([times[i], times[i]])
        return im, txt, cursor

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)

    return out_path


def loadFrameTimes(json_path):
    """ Load a frames-timelapse sidecar (see Utils.GenerateTimelapse): a dict mapping
        frame index (as string) to a timestamp string derived from the frame file name,
        e.g. "20260718_051032_123" (a trailing day/night letter is ignored).

    Return:
        times_unix: [ndarray] Frame times (unix seconds, UTC), ordered by frame index.
    """

    import datetime
    import calendar

    with open(json_path) as f:
        data = json.load(f)

    times = []
    for idx in sorted(data, key=int):
        parts = str(data[idx]).split("_")
        dt = datetime.datetime.strptime(parts[0] + parts[1], "%Y%m%d%H%M%S")
        ms = int(parts[2]) if (len(parts) > 2 and parts[2].isdigit()) else 0
        times.append(calendar.timegm(dt.timetuple()) + ms/1000.0)

    return np.array(times)


def _renderExtinctionPanel(dm_frame, width_px, height_px, vmax=3.0, mask_img=None,
        img_size=None):
    """ Render one extinction map as a BGR image (matplotlib -> array).

    mask_img (station mask, 0 = obstructed) is drawn as an opaque dark overlay: a cell
    verdict only speaks for the sky it can see, and the per-cell normalization absorbs
    static obstruction - without the overlay, terrain inside a partially-open cell
    would inherit the cell's sky color and read as "transparent ground".
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dpi = 100
    fig, ax = plt.subplots(figsize=(width_px/dpi, height_px/dpi), dpi=dpi)
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("lightgray")
    ext = None
    if img_size is not None:
        ext = [0, img_size[0], img_size[1], 0]
    im = ax.imshow(np.ma.masked_invalid(dm_frame), cmap=cmap, vmin=0.0, vmax=vmax,
        interpolation="nearest", aspect="auto", extent=ext)
    if (mask_img is not None) and (ext is not None):
        over = np.zeros(mask_img.shape + (4,))
        over[mask_img < 128] = (0.25, 0.25, 0.25, 1.0)
        ax.imshow(over, extent=ext, interpolation="nearest", aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, label="Extinction (mag)")
    fig.tight_layout(pad=0.4)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)

    # RGB -> BGR for OpenCV, and force the exact requested size
    import cv2
    return cv2.resize(buf[:, :, ::-1].copy(), (width_px, height_px))


def sideBySideVideo(video_path, frame_times_unix, frames, dm, out_path,
        fps=None, max_gap=900.0, vmax=3.0, mask_img=None, img_size=None):
    """ Compose the camera timelapse next to the per-cell transparency map, frame by
        frame - the contrail-attribution view: was that part of the FOV transparent
        enough at that moment.

    Each timelapse frame is matched to the nearest scored bin in time; frames farther
    than max_gap from any bin get a "no transparency data" panel instead of a stale map.

    Arguments:
        video_path: [str] Camera timelapse (e.g. <night>_frames_timelapse.mp4).
        frame_times_unix: [ndarray] Per-video-frame times (see loadFrameTimes), same
            order and count as the video frames.
        frames: [dict] Frame arrays from loadStarScoring (bin times).
        dm: [ndarray n_bins, ny, nx] Output of extinctionSeries.
        out_path: [str] Output .mp4.

    Keyword arguments:
        fps: [float] Output frame rate; source rate when None.
        max_gap: [float] Seconds beyond which no scored bin is considered current.
        vmax: [float] Color scale ceiling (mag).

    Return:
        out_path: [str]
    """

    import cv2
    import datetime

    bin_times = np.asarray(frames["frame_time_unix"], dtype=np.float64)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Could not open video: {:s}".format(video_path))

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    panel_w = vh  # square-ish right panel, same height as the video
    out_size = (vw + panel_w, vh)

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps else src_fps, out_size)

    if img_size is None:
        img_size = (vw, vh)

    # Pre-render one panel per scored bin, plus the no-data panel
    panels = [_renderExtinctionPanel(dm[j], panel_w, vh, vmax=vmax, mask_img=mask_img,
                                     img_size=img_size)
              for j in range(dm.shape[0])]
    no_data = np.full((vh, panel_w, 3), 40, dtype=np.uint8)
    cv2.putText(no_data, "no transparency data", (panel_w//8, vh//2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)

    i = 0
    n_written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if i >= len(frame_times_unix):
            break
        t = frame_times_unix[i]

        j = int(np.argmin(np.abs(bin_times - t)))
        if abs(bin_times[j] - t) <= max_gap:
            panel = panels[j]
        else:
            panel = no_data

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        combo = np.hstack([frame, panel])

        stamp = datetime.datetime.utcfromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S UTC")
        cv2.putText(combo, stamp, (vw + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(combo, stamp, (vw + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(combo)
        n_written += 1
        i += 1

    cap.release()
    writer.release()

    print("Side-by-side video: {:d} frames -> {:s}".format(n_written, out_path))
    return out_path


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
    parser.add_argument("--animate", default=None, help="Also write a transparency "
        "animation (GIF) to this path")
    parser.add_argument("--dome-s", type=float, default=0.5, help="Logistic width s of "
        "the night's dome model (mag), used to express extinction in magnitudes "
        "(default 0.5; read it from the night's light_dome.json)")
    parser.add_argument("--video", default=None, help="Camera timelapse mp4 to compose "
        "side by side with the transparency map")
    parser.add_argument("--frametimes", default=None, help="Frames-timelapse sidecar "
        "JSON with per-frame timestamps (see Utils.GenerateTimelapse)")
    parser.add_argument("--uniform", default=None, help="Fallback frame timing as "
        "START_ISO,DT_SECONDS (e.g. 2026-07-18T05:03:10,10.24) when no sidecar exists")
    parser.add_argument("--sidebyside", default=None, help="Output mp4 path for the "
        "side-by-side composition (requires --video and timing)")
    parser.add_argument("--mask", default=None, help="Station mask image (mask.bmp); "
        "obstructed regions are drawn opaque on the transparency panels")
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

    dm = None
    if args.animate or args.sidebyside:
        dm = extinctionSeries(frames, stars, result, args.dome_s)

    if args.animate:
        animateTransparency(frames, result, dm, args.animate,
            title=str(header.get("stationID", "")))
        print("Animation: {:s}".format(args.animate))

    if args.sidebyside:
        if not args.video:
            parser.error("--sidebyside requires --video")

        if args.frametimes:
            frame_times = loadFrameTimes(args.frametimes)
        elif args.uniform:
            import calendar
            import datetime
            start_str, dt_str = args.uniform.split(",")
            start_dt = datetime.datetime.strptime(start_str, "%Y-%m-%dT%H:%M:%S")
            start_unix = calendar.timegm(start_dt.timetuple())
            import cv2
            cap = cv2.VideoCapture(args.video)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            frame_times = start_unix + np.arange(n)*float(dt_str)
        else:
            parser.error("--sidebyside requires --frametimes or --uniform")

        mask_img = None
        if args.mask:
            import cv2
            mask_img = cv2.imread(args.mask, 0)

        sideBySideVideo(args.video, frame_times, frames, dm, args.sidebyside,
            mask_img=mask_img,
            img_size=(int(result["width"]), int(result["height"])))


if __name__ == "__main__":
    main()
