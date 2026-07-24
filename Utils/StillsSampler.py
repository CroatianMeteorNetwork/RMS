""" Instantaneous star states from the saved 5 s stills.

The FF-based products carry a 10.24 s maxpixel window: "detected" means visible
at SOME instant of the window, so a cloud edge's crossing time is only known to
+/- a window. The stills are single ~40 ms exposures every ~5 s - forced patch
photometry on them (RMS.PatchPhotometry) yields exact visible/occluded states
at the frames the contrail consumers already use, with edge timing at the still
cadence instead of the window width. Validated on LP and dark-site cameras:
per-star depth within ~0.5 mag of the FF channel through JPEG compression, and
BRIGHTER than the extractor's curve above ~mag 4 (no saturation/PSF gates).

Runs in the morning pipeline just before the frames are consumed into the
timelapse and deleted (RMS.Reprocess.processFramesFiles), when the night's
recalibrated platepars and star-scoring product already exist - so positions
are exact and the star set is chosen from MEASURED per-star behavior, not
magnitude constants:

- Per-star channel: stars the night's scoring product shows as reliably
  detectable (CALSTARS match rate >= BRIGHT_RATE_MIN on scored frames) plus
  every star the forced-photometry class recovered (calstars_row == -2,
  saturated/extractor-culled - the highest-SNR cloud probes there are).
  Their full flux series is stored, visible or not: the absences carry the
  occlusion evidence.
- Cell-stack channel: every other in-FOV catalog star's flux is summed per
  transparency-map cell per still - individually sub-threshold stars are
  recovered in aggregate (sqrt-N), giving a cell-level transparency signal at
  the still cadence without storing the faint tail.

Star identity is star_cat_id from the scoring product (schema v4), so the
sidecar joins against the scoring product and the trailing calibration.

Schema (npz, <night>_still_star_states.npz):
    header        - JSON: schema_version, stationID, night, catalog_lim_mag,
                    snr_min, clear_flux_frac, noise_floor, provenance counts.
    t_unix        - [n_stills] float64 still times (UTC, from the file name).
    still_mode    - [n_stills] uint8, 1 = night mode.
    frame_noise   - [n_stills] float32 per-still noise scale (pre-floor).
    bright_cat_id - [n_bright] int32 catalog ids of the per-star channel.
    bright_flux   - [n_bright, n_stills] float16 aperture flux (NaN = not
                    measurable: out of frame / too close to the edge).
    bright_detected - [n_bright, n_stills] bool, both floors applied.
    cell_flux     - [n_stills, ny, nx] float32 summed stack-channel flux.
    cell_count    - [n_stills, ny, nx] int16 stars in each stack.
"""

from __future__ import absolute_import, division, print_function

import calendar
import glob
import json
import os

import numpy as np

from RMS.Formats import FFfile, StarCatalog
from RMS.Formats.Platepar import Platepar
from RMS.PatchPhotometry import (detectStars, measurePatchFluxes,
    nightNoiseFloor)


SCHEMA_VERSION = 1

FILE_SUFFIX = "still_star_states.npz"

BRIGHT_RATE_MIN = 0.5    # CALSTARS match rate (this night, measured) above which a
                         # star joins the per-star channel

CELL_NX, CELL_NY = 8, 5  # stack-channel grid - matches the transparency map

MIN_NIGHT_STILLS = 20


def sidecarFileName(night_name):
    return "{:s}_{:s}".format(night_name, FILE_SUFFIX)


def sampleStillsForNight(config, image_blocks, night_dir):
    """ Measure the night's stills and write the sidecar into the night directory.

    Arguments:
        config: [Config]
        image_blocks: [list of list of str] Frame blocks from
            Utils.GenerateTimelapse.listImageBlocksBefore - consumed as-is so
            the sampler sees exactly the frames the timelapse is about to eat.
        night_dir: [str] The night's CapturedFiles directory (recalibrated
            platepars + star-scoring product must exist there).

    Return:
        path: [str] Written sidecar path, or None if the night could not be
            sampled (missing prerequisites are logged, never raised).
    """

    from Utils.GenerateTimelapse import _modeFromName, _timestampFromName

    import cv2

    # Night-mode stills, chronological
    stills = []
    for block in image_blocks:
        for p in block:
            if _modeFromName(p) == 'n':
                stills.append((_timestampFromName(p), p))
    stills.sort()

    if len(stills) < MIN_NIGHT_STILLS:
        print("Stills sampler: only {:d} night stills - skipping".format(len(stills)))
        return None

    # Prerequisites from the processed night
    from RMS.Formats.StarScoring import loadStarScoring, scoringFileName

    night_name = os.path.basename(os.path.normpath(night_dir))
    scoring_path = os.path.join(night_dir, scoringFileName(night_name))
    pp_path = os.path.join(night_dir, config.platepars_flux_recalibrated_name)

    if not (os.path.isfile(scoring_path) and os.path.isfile(pp_path)):
        print("Stills sampler: no scoring product or recalibrated platepars - skipping")
        return None

    header, frames, stars = loadStarScoring(scoring_path)

    if int(header.get("schema_version", 0)) < 4 or "star_cat_id" not in stars:
        print("Stills sampler: scoring product predates schema v4 (no star "
              "identity) - skipping")
        return None

    # Per-star channel membership from MEASURED behavior this night
    cat_id = np.asarray(stars["star_cat_id"], dtype=np.int64)
    row = np.asarray(stars["calstars_row"], dtype=np.int64)

    n_seen = np.bincount(cat_id)
    n_matched = np.bincount(cat_id, weights=(row >= 0).astype(np.float64))
    with np.errstate(invalid="ignore"):
        rate = n_matched/np.maximum(n_seen, 1)
    forced_ever = np.zeros(len(n_seen), dtype=bool)
    forced_ever[cat_id[row == -2]] = True

    bright_ids = np.where(((rate >= BRIGHT_RATE_MIN) & (n_seen >= 10))
                          | forced_ever)[0]

    if len(bright_ids) < 5:
        print("Stills sampler: {:d} per-star channel members - skipping".format(
            len(bright_ids)))
        return None

    # Catalog at the scoring depth so cat ids align
    catalog_stars, _, _ = StarCatalog.readStarCatalog(
        config.star_catalog_path, config.star_catalog_file,
        lim_mag=float(header["catalog_lim_mag"]),
        mag_band_ratios=config.star_catalog_band_ratios)
    cat_ra, cat_dec = catalog_stars[:, 0], catalog_stars[:, 1]

    # Platepar knots
    with open(pp_path) as f:
        ppr = json.load(f)
    knots = []
    for ff_name, pp_dict in ppr.items():
        if isinstance(pp_dict, dict) and pp_dict.get("auto_recalibrated"):
            pp = Platepar()
            pp.loadFromDict(pp_dict, use_flat=config.use_flat)
            knots.append((FFfile.filenameToDatetime(ff_name), pp))
    knots.sort(key=lambda kv: kv[0])

    if not knots:
        print("Stills sampler: no valid recalibrated platepar - skipping")
        return None

    knot_times = np.array([calendar.timegm(t.timetuple()) for t, _ in knots])
    knot_pps = [pp for _, pp in knots]

    from RMS.Astrometry.ApplyAstrometry import raDecToXYPP
    from RMS.Astrometry.Conversions import date2JD

    n_stills = len(stills)
    n_bright = len(bright_ids)
    bright_flux = np.full((n_bright, n_stills), np.nan, dtype=np.float32)
    frame_noise = np.full(n_stills, np.nan, dtype=np.float32)
    cell_flux = np.zeros((n_stills, CELL_NY, CELL_NX), dtype=np.float32)
    cell_count = np.zeros((n_stills, CELL_NY, CELL_NX), dtype=np.int16)
    t_unix = np.empty(n_stills, dtype=np.float64)

    # Stack channel: all catalog stars the scoring product retained, minus the
    # per-star channel (their flux is stored individually)
    stack_ids = np.setdiff1d(np.unique(cat_id), bright_ids)

    bright_set = set(int(i) for i in bright_ids)
    W = H = None

    for j, (t, path) in enumerate(stills):

        t_unix[j] = calendar.timegm(t.timetuple()) + t.microsecond/1e6

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if W is None:
            H, W = img.shape

        jd = date2JD(t.year, t.month, t.day, t.hour, t.minute, t.second,
            t.microsecond/1000.0)

        k = int(np.argmin(np.abs(knot_times - t_unix[j])))
        pp = knot_pps[k]

        # Project both channels at this instant. Plain projection (no spherical
        # polygon test) is safe here: membership came from the scoring product,
        # whose stars all sat inside the FOV polygon - the frame-bounds check
        # below only trims edge drift.
        ids = np.concatenate([bright_ids, stack_ids])
        x, y = raDecToXYPP(cat_ra[ids], cat_dec[ids], jd, pp)

        flux, nz = measurePatchFluxes(img, x, y)
        frame_noise[j] = nz

        bright_part = flux[:n_bright]
        bright_flux[:, j] = bright_part

        stack_part = flux[n_bright:]
        sx, sy = x[n_bright:], y[n_bright:]
        ok = np.isfinite(stack_part)
        if np.any(ok):
            cx = np.clip((sx[ok]/(W/CELL_NX)).astype(np.intp), 0, CELL_NX - 1)
            cy = np.clip((sy[ok]/(H/CELL_NY)).astype(np.intp), 0, CELL_NY - 1)
            np.add.at(cell_flux[j], (cy, cx), stack_part[ok])
            np.add.at(cell_count[j], (cy, cx), 1)

    noise_floor = nightNoiseFloor(frame_noise)

    # Clear-flux medians for the detection floor: each star's flux on stills
    # taken while an FF window in which CALSTARS matched it was current
    ff_t = np.asarray(frames["frame_time_unix"], dtype=np.float64)
    matched_times = {}
    for cid in bright_ids:
        sel = (cat_id == cid) & (row >= 0)
        if np.any(sel):
            matched_times[int(cid)] = ff_t[np.asarray(
                stars["star_frame"], dtype=np.int64)[sel]]

    clear_med = np.full(n_bright, np.nan, dtype=np.float32)
    for b, cid in enumerate(bright_ids):
        mt = matched_times.get(int(cid))
        if mt is None or not len(mt):
            continue
        near = np.min(np.abs(t_unix[None, :] - mt[:, None]), axis=0) <= 6.0
        vals = bright_flux[b, near]
        vals = vals[np.isfinite(vals)]
        if len(vals) >= 5:
            clear_med[b] = np.median(vals)

    bright_detected = np.zeros((n_bright, n_stills), dtype=bool)
    for j in range(n_stills):
        det, _ = detectStars(bright_flux[:, j], frame_noise[j], noise_floor,
            clear_flux_median=clear_med)
        bright_detected[:, j] = det

    out_header = dict(
        schema_version=SCHEMA_VERSION,
        stationID=str(config.stationID),
        night=night_name,
        catalog_lim_mag=float(header["catalog_lim_mag"]),
        bright_rate_min=BRIGHT_RATE_MIN,
        noise_floor=float(noise_floor),
        n_bright=int(n_bright),
        n_stills=int(n_stills),
        n_platepar_knots=len(knots),
        note="bright_detected applies both PatchPhotometry floors; "
             "bright_flux NaN = not measurable on that still",
    )

    path = os.path.join(night_dir, sidecarFileName(night_name))
    np.savez_compressed(path.replace(".npz", ""),
        header=json.dumps(out_header),
        t_unix=t_unix,
        still_mode=np.ones(n_stills, dtype=np.uint8),
        frame_noise=frame_noise,
        bright_cat_id=bright_ids.astype(np.int32),
        bright_flux=bright_flux.astype(np.float16),
        bright_detected=bright_detected,
        cell_flux=cell_flux,
        cell_count=cell_count)

    print("Stills sampler: {:s} ({:d} stills, {:d} per-star, stack over "
          "{:d} stars)".format(os.path.basename(path), n_stills, n_bright,
          len(stack_ids)))

    return path


def loadStillStarStates(path):
    """ Load a sidecar.

    Return:
        (header, arrays): header dict and dict of ndarrays per the schema.
    """

    with np.load(path, allow_pickle=False) as z:
        header = json.loads(str(z["header"]))
        arrays = {k: z[k] for k in z.files if k != "header"}

    return header, arrays


def findNightDirForStills(config, stills_t0):
    """ The CapturedFiles night directory whose span covers the stills' start.

    Arguments:
        config: [Config]
        stills_t0: [datetime] Time of the first night still.

    Return:
        night_dir: [str] or None.
    """

    captured = os.path.join(config.data_dir, config.captured_dir)
    best, best_dt = None, None
    for d in glob.glob(os.path.join(captured, "{:s}_*".format(config.stationID))):
        name = os.path.basename(d)
        try:
            parts = name.split("_")
            from datetime import datetime
            dt = datetime.strptime(parts[1] + parts[2], "%Y%m%d%H%M%S")
        except (IndexError, ValueError):
            continue
        if dt <= stills_t0 and (best_dt is None or dt > best_dt):
            best, best_dt = d, dt

    return best
