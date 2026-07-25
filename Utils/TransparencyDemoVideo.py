""" Render the transparency demo video on station: the night's 5 s stills
side-by-side with the transparency overlay and per-star evidence markers.

The point of the video is JUDGEABILITY: a human can verify the transparency
estimate against the raw sky frame by frame, because every piece of evidence
the estimator used is drawn on it. The rendering choices are deliberate and
were iterated against human judgment on demo nights:

- The overlay maps extinction dm to the BLOCKED-LIGHT fraction
  u = 1 - 10^(-0.4 dm) (dm 1 blocks 60%, dm 2.5 blocks 90%), because that is
  how a human reads cloud opacity - a linear-in-dm ramp makes thick cloud look
  thin. Viridis colormap, fixed 0.5 alpha above a 0.3 mag dead-band (values
  below the counting-noise floor stay invisible), diagonal hatching where the
  map has no data (flat gray is invisible on a black-and-white sky).
- Markers carry the evidence: green ring = member star detected on this still;
  orange ring = detected but dimmed (flux well below its clear median);
  light-blue ring = a forced-photometry-class member (saturated/extractor-
  culled star recovered by aperture flux); red dot = a reliable member
  (high measured rate) missing; small dark-red dot = a weak member missing.

Gated by config.transparency_demo_video. The video stays on station (written
to the night directory, after its archive was built - it never uploads).
"""

from __future__ import absolute_import, division, print_function

import calendar
import json
import os

import numpy as np


FILE_SUFFIX = "transparency_demo.mp4"

DEAD_BAND_MAG = 0.3    # overlay invisible below this dm
OVERLAY_ALPHA = 0.5
DIMMED_FLUX_FRAC = 0.6 # detected but flux below this fraction of clear median
                       # draws the "dimmed" marker (~0.55 mag)
STRONG_RATE = 0.5      # member rate above which an absence is "reliable missing"
FPS = 30
CRF = 25               # H.264 quality (same as the frames timelapse)


def _openEncoder(out_path, width, height):
    # Streaming H.264 encoder: raw BGR frames piped into ffmpeg stdin - the
    # same pattern (and rate caps) as the frames-timelapse encoder, which
    # produces files ~10x smaller than OpenCV's mp4v writer at equal quality.
    # Returns (proc, temp_path), or (None, None) if ffmpeg is unavailable.
    import subprocess
    from Utils.GenerateTimelapse import isFfmpegWorking

    ffmpeg_path = isFfmpegWorking()
    if not ffmpeg_path:
        return None, None

    maxrate, bufsize = ("4M", "8M") if width*height > 1280*720 else ("2M", "4M")
    temp_path = out_path + ".tmp.mp4"
    cmd = [ffmpeg_path, "-y", "-nostdin", "-hide_banner",
           "-loglevel", "error",
           "-f", "rawvideo", "-vcodec", "rawvideo",
           "-s", "{}x{}".format(width, height),
           "-pix_fmt", "bgr24", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-crf", str(CRF),
           "-maxrate", maxrate, "-bufsize", bufsize,
           "-preset", "veryfast", "-pix_fmt", "yuv420p",
           "-movflags", "faststart", "-threads", "1", "-g", "120",
           temp_path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE), temp_path


def demoVideoFileName(night_name):
    return "{:s}_{:s}".format(night_name, FILE_SUFFIX)


def _viridisLut():
    """ 255-entry BGR viridis LUT without requiring matplotlib at import. """

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap("viridis")
        rgb = np.array([cmap(i/254.0)[:3] for i in range(255)])
    except Exception:
        # Fallback: a perceptually-ordered blue->green->yellow ramp
        t = np.linspace(0, 1, 255)[:, None]
        rgb = np.hstack([0.27 + 0.7*t**2, 0.0 + 0.9*t, 0.33 + 0.5*(1 - t)])
        rgb = np.clip(rgb, 0, 1)
    return (rgb[:, ::-1]*255).astype(np.uint8)


def generateDemoVideo(config, night_dir, sidecar_path, max_stills=None):
    """ Render the demo video for a night whose stills sidecar was just written.

    Arguments:
        config: [Config]
        night_dir: [str] Night directory (transparency map + scoring product).
        sidecar_path: [str] The stills sidecar written by Utils.StillsSampler -
            its frame list defines the video; the stills must still exist.

    Keyword arguments:
        max_stills: [int] Render at most this many stills (None = all).

    Return:
        path: [str] Video path, or None if prerequisites are missing.
    """

    import cv2

    from RMS.Formats import StarCatalog
    from RMS.Formats.Platepar import Platepar
    from RMS.Formats.StarScoring import scoringFileName
    from RMS.Formats.TransparencyMap import mapFileName
    from RMS.Astrometry.ApplyAstrometry import raDecToXYPP
    from RMS.Astrometry.Conversions import date2JD
    from RMS.Formats import FFfile
    from Utils.StillsSampler import inFrontOfCamera, loadStillStarStates

    night_name = os.path.basename(os.path.normpath(night_dir))

    # Overlay the TREE map when it exists - the estimator being judged. The
    # grid map is only the fallback while the tree product is absent.
    from Utils.VoronoiTreeEstimator import treeMapFileName
    tree_path = os.path.join(night_dir, treeMapFileName(night_name))
    map_path = tree_path if os.path.isfile(tree_path)         else os.path.join(night_dir, mapFileName(night_name))
    if not (os.path.isfile(map_path) and os.path.isfile(sidecar_path)):
        return None

    import numpy as _np
    leaf_cat_id = leaf_dm = None
    with _np.load(map_path, allow_pickle=False) as _z:
        _mh = json.loads(str(_z["header"]))
        t_map = _z["t_unix"]
        dm = _z["dm"].astype(_np.float32)
        if "leaf_dm" in _z.files:
            leaf_cat_id = _z["leaf_cat_id"].astype(_np.int64)
            leaf_dm = _z["leaf_dm"].astype(_np.float32)
            # Display smoothing: zero-phase median over +/-1 map frame kills
            # the dead-band shimmer (5 s stills alternate between adjacent
            # 10 s map frames; leaves hovering at the threshold blink). The
            # PRODUCT stays unsmoothed - this is presentation only; the real
            # temporal layer is its own benchmarked roadmap item.
            if len(leaf_dm) >= 3:
                stack = _np.stack([leaf_dm[:-2], leaf_dm[1:-1], leaf_dm[2:]])
                with _np.errstate(all="ignore"):
                    leaf_dm[1:-1] = _np.nanmedian(stack, axis=0)
    estimator_tag = _mh.get("estimator", "grid")

    # Station mask: masked pixels (terrain, obstructions) are hatched, never
    # painted - a cell/leaf value only means anything on sky
    mask_img = None
    for mp_ in (os.path.join(night_dir, "mask.bmp"),):
        if os.path.isfile(mp_):
            mask_img = cv2.imread(mp_, cv2.IMREAD_GRAYSCALE)
    sc_header, sc = loadStillStarStates(sidecar_path)

    t_unix = sc["t_unix"]
    bright_ids = sc["bright_cat_id"].astype(np.int64)
    bright_flux = sc["bright_flux"].astype(np.float32)
    bright_det = sc["bright_detected"]

    # The stills themselves: rediscover their paths from the frame dir by time
    from Utils.GenerateTimelapse import IMAGE_PATTERN, _timestampFromName
    frame_dir = os.path.join(config.data_dir, config.frame_dir)
    stills_by_t = {}
    for root, _, files in os.walk(frame_dir):
        for fname in files:
            if IMAGE_PATTERN.match(fname):
                ts = _timestampFromName(fname)
                key = calendar.timegm(ts.timetuple()) + ts.microsecond/1e6
                stills_by_t[round(key, 1)] = os.path.join(root, fname)
    if not stills_by_t:
        return None

    # Member stats measured on this night's stills (rates weight the markers)
    meas = np.isfinite(bright_flux)
    with np.errstate(invalid="ignore"):
        still_rate = (bright_det & meas).sum(axis=1)/np.maximum(meas.sum(axis=1), 1)
        clear_med = np.nanmedian(np.where(bright_det, bright_flux, np.nan), axis=1)

    # Forced-class members (recovered saturated stars) from the scoring product
    forced_ids = set()
    try:
        from RMS.Formats.StarScoring import loadStarScoring
        _, _, stars = loadStarScoring(os.path.join(night_dir,
            scoringFileName(night_name)))
        cid = np.asarray(stars["star_cat_id"], dtype=np.int64)
        row = np.asarray(stars["calstars_row"], dtype=np.int64)
        forced_ids = set(int(i) for i in np.unique(cid[row == -2]))
    except Exception:
        pass
    is_forced = np.array([int(i) in forced_ids for i in bright_ids])

    # Star positions: catalog + platepar knots (same projection as the sampler)
    catalog_stars, _, _ = StarCatalog.readStarCatalog(
        config.star_catalog_path, config.star_catalog_file,
        lim_mag=float(sc_header["catalog_lim_mag"]),
        mag_band_ratios=config.star_catalog_band_ratios)
    cat_ra, cat_dec = catalog_stars[:, 0], catalog_stars[:, 1]

    pp_path = os.path.join(night_dir, config.platepars_flux_recalibrated_name)
    with open(pp_path) as f:
        ppr = json.load(f)
    knots = []
    for ff_name, pp_dict in ppr.items():
        if isinstance(pp_dict, dict) and pp_dict.get("auto_recalibrated"):
            pp = Platepar()
            pp.loadFromDict(pp_dict, use_flat=config.use_flat)
            knots.append((calendar.timegm(
                FFfile.filenameToDatetime(ff_name).timetuple()), pp))
    if not knots:
        return None
    knots.sort()
    knot_times = np.array([t for t, _ in knots])
    knot_pps = [pp for _, pp in knots]

    lut = _viridisLut()

    import datetime as _dt

    def _leafPositions(j):
        """ Leaf anchor image positions at still j (same projection as markers). """
        t = float(t_unix[j])
        ki = int(np.argmin(np.abs(knot_times - t)))
        pp = knot_pps[ki]
        d = _dt.datetime.utcfromtimestamp(t)
        jd = date2JD(d.year, d.month, d.day, d.hour, d.minute, d.second,
            d.microsecond/1000.0)
        x, y = raDecToXYPP(cat_ra[leaf_cat_id], cat_dec[leaf_cat_id], jd, pp)
        front = inFrontOfCamera(pp, cat_ra[leaf_cat_id], cat_dec[leaf_cat_id], jd)
        return np.where(front, x, np.nan), np.where(front, y, np.nan)

    _grid_pts = _grid_shape = None
    # Overlay layers depend only on the MAP frame, not the still: rebuild on
    # map-frame change (~every other still) and reuse between. The KDTree
    # pixel-grid labeling and colorizing were the render hot spot; leaf
    # anchors drift only a few pixels per still, well under the 4 px label
    # grid, so the reuse is visually lossless.
    ov_k = -1
    ov_inv = ov_add = ov_gray = None
    hole_gz = None

    n_stills = len(t_unix) if max_stills is None else min(len(t_unix), max_stills)
    writer = None
    out_path = os.path.join(night_dir, demoVideoFileName(night_name))

    ny, nx = dm.shape[1], dm.shape[2]

    for j in range(n_stills):

        path = stills_by_t.get(round(float(t_unix[j]), 1))
        if path is None:
            continue
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            continue
        H, W = frame.shape[:2]

        if writer is None:
            writer, temp_path = _openEncoder(out_path, 2*W, H)
            if writer is None:
                # No ffmpeg on this machine: OpenCV fallback (larger files)
                writer = cv2.VideoWriter(out_path,
                    cv2.VideoWriter_fourcc(*"mp4v"), FPS, (2*W, H))
                temp_path = None
            hatch = ((np.add.outer(np.arange(H), np.arange(W))) % 28) < 4
            gy_, gx_ = np.mgrid[0:H:4, 0:W:4]
            _grid_shape = gy_.shape
            _grid_pts = np.column_stack([gx_.ravel(), gy_.ravel()])

        right = frame.copy()

        # Transparency overlay from the nearest map frame. With the tree's
        # leaf channel: TRUE nearest-leaf rendering at pixel resolution (the
        # estimator's native Voronoi structure - the thing being judged).
        # Cell-resize is only the fallback for leafless (grid) maps.
        k = int(np.argmin(np.abs(t_map - t_unix[j])))
        map_current = abs(t_map[k] - t_unix[j]) <= 40.0
        if not map_current:
            # The FF product has a hole here (heavily overcast frames can
            # yield no CALSTARS entry at all, so they were never scored).
            # An absent estimate must be VISIBLY absent - hatch the whole
            # sky; a blank overlay reads as "clear", which is a lie.
            if hole_gz is None:
                hole_gz = hatch if mask_img is None else (hatch & (mask_img > 0))
            right[hole_gz] = (right[hole_gz] >> 1) + 90
        if map_current:
            if ov_k != k:
                big = None
                if leaf_dm is not None:
                    lx, ly = _leafPositions(j)
                    lv = leaf_dm[k]
                    okl = (np.isfinite(lx) & np.isfinite(ly) & np.isfinite(lv)
                           & (lx > -50) & (lx < W + 50)
                           & (ly > -50) & (ly < H + 50))
                    if okl.sum() > 30:
                        from scipy.spatial import cKDTree
                        tr_ = cKDTree(np.column_stack([lx[okl], ly[okl]]))
                        _, nn = tr_.query(_grid_pts, k=1)
                        small = lv[okl][nn].reshape(_grid_shape)
                        big = cv2.resize(small.astype(np.float32), (W, H),
                            interpolation=cv2.INTER_NEAREST)
                        nan_big = ~np.isfinite(big)
                if big is None:
                    cell = dm[k].astype(np.float32)
                    nan_cells = ~np.isfinite(cell)
                    big = cv2.resize(np.nan_to_num(cell), (W, H),
                        interpolation=cv2.INTER_LINEAR)
                    nan_big = cv2.resize(nan_cells.astype(np.float32), (W, H),
                        interpolation=cv2.INTER_LINEAR) > 0.5
                if mask_img is not None:
                    nan_big = nan_big | (mask_img == 0)
                u = 1.0 - 10.0**(-0.4*np.nan_to_num(np.maximum(big, 0.0)))
                color = lut[np.clip(u*254, 0, 254).astype(np.uint8)]
                alpha = np.where(np.nan_to_num(big) > DEAD_BAND_MAG,
                    OVERLAY_ALPHA, 0.0)
                alpha[nan_big] = 0.0
                # Integer blend layers: the float form promoted the whole
                # frame to float64 every still and was the render hot spot
                ov_inv = np.rint((1.0 - alpha)*256.0).astype(np.uint16)
                ov_add = np.rint(color*alpha[..., None]).astype(np.uint8)
                ov_gray = nan_big & hatch
                ov_k = k
            right = cv2.add(
                ((right.astype(np.uint16)*ov_inv[..., None]) >> 8)
                .astype(np.uint8), ov_add)
            right[ov_gray] = (right[ov_gray] >> 1) + 90

        # Evidence markers
        t = t_unix[j]
        ki = int(np.argmin(np.abs(knot_times - t)))
        pp = knot_pps[ki]
        dt = _dt.datetime.utcfromtimestamp(t)
        jd = date2JD(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second,
            dt.microsecond/1000.0)
        x, y = raDecToXYPP(cat_ra[bright_ids], cat_dec[bright_ids], jd, pp)
        front = inFrontOfCamera(pp, cat_ra[bright_ids], cat_dec[bright_ids], jd)
        in_frame = front & (x > 8) & (x < W - 8) & (y > 8) & (y < H - 8)

        det = bright_det[:, j]
        m_flux = bright_flux[:, j]
        with np.errstate(invalid="ignore"):
            dimmed = det & np.isfinite(clear_med) \
                & (m_flux < DIMMED_FLUX_FRAC*clear_med)
        missing = ~det & np.isfinite(m_flux)

        for b in np.where(in_frame)[0]:
            pt = (int(x[b]), int(y[b]))
            if det[b] and is_forced[b]:
                cv2.circle(right, pt, 6, (255, 220, 80), 1, cv2.LINE_AA)
            elif dimmed[b]:
                cv2.circle(right, pt, 5, (60, 165, 255), 1, cv2.LINE_AA)
            elif det[b]:
                cv2.circle(right, pt, 5, (80, 255, 80), 1, cv2.LINE_AA)
            elif missing[b] and still_rate[b] >= STRONG_RATE:
                cv2.circle(right, pt, 2, (50, 50, 255), -1, cv2.LINE_AA)
            elif missing[b]:
                cv2.circle(right, pt, 1, (40, 40, 170), -1, cv2.LINE_AA)

        combo = np.hstack([frame, right])
        stamp = _dt.datetime.utcfromtimestamp(t).strftime("%H:%M:%S UTC")
        for xoff in (10, W + 10):
            cv2.putText(combo, stamp, (xoff, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(combo, stamp, (xoff, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)
        leg = ("o detected   o dimmed   o forced-phot(sat)   . reliable missing"
               "   . weak missing   [{}]".format(
                   "TREE" if "tree" in estimator_tag else "GRID"))
        # Above the stills' own burned-in banner
        cv2.putText(combo, leg, (W + 10, H - 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(combo, leg, (W + 10, H - 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (230, 230, 230), 1, cv2.LINE_AA)

        if temp_path is not None:
            writer.stdin.write(np.ascontiguousarray(combo).tobytes())
        else:
            writer.write(combo)

    if writer is None:
        return None
    if temp_path is not None:
        writer.stdin.close()
        writer.wait()
        os.replace(temp_path, out_path)
    else:
        writer.release()

    print("Transparency demo video: {:s}".format(os.path.basename(out_path)))

    return out_path
