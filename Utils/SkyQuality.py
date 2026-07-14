""" Nightly sky quality (SQM) measurement from the camera's own frames.

The sky background of the avepixel is a radiometric measurement: with the sensor bias
subtracted, the photometric zero point (mag_lev) converts ADU per pixel to mag/arcsec^2.
The hard part is the bias, and stations differ in what they can support, so the bias is
resolved through safe tiers:

  Tier 1 - a stored radiometric calibration file (<stationID>_radiometric.json), written
      by a host-level cross-camera calibration (e.g. the FOV-overlap graph on multi-camera
      sites) or any other trusted source. Never overwritten automatically.
  Tier 2 - single-camera self-priming: if the station's light-dome model shows enough
      brightness CONTRAST across the FOV (auto-checked), the bias is the intercept of
      regressing patch ADU against the model brightness field. Cached to the same file
      (method "model-regression") and refreshed when stale.
  Tier 3 - no usable bias: the darkest patch level of the night bounds the bias from
      above, which bounds the sky brightness. Reported as a LIMIT (sky at least this
      bright), never as an absolute value.

Measurements are taken on the highest-altitude unmasked patch of the FOV, only on frames
inside clear intervals with no significant moon. The sky level is devignetted and scaled
by the tracked PSF capture fraction (see the APERTURE_* constants: the star extractor's
windowed sums miss the PSF wings, and mag_lev absorbs that deficit, so extended sources
need the same haircut to sit on the calibrated flux scale). Results are written to
<night>_sky_quality.json in the night directory and one line is logged.

Convention note: mag_lev maps counts to above-atmosphere magnitudes, so the SQM values
read ~0.2 mag brighter than the as-seen convention at the zenith. Beyond that, treat
the absolute scale as UNCALIBRATED against SQM meters: validation against a co-sited
SQM-LE at SI0002 (Aug+Nov 2024, 20 clear nights) measured excellent night-to-night
tracking (~0.07 mag RMS) but a stable total offset of ~1.5-2 mag bright of the meter.
The measured decomposition: extinction and the patch altitude ~0.5-0.7; PSF wing loss
+0.39 at SI0002 (now removed by the nightly aperture correction above); spectral band
~0 (a Gaia G/BP/RP band-ratio fit showed the camera response is close to the catalog
G band, color term -0.04 mag per BP-RP); the remaining ~0.5-1 mag is consistent with
lens-scattered light (veiling glare), which no software correction can separate from
sky signal. Bortle classes inherit whatever offset remains.
"""

from __future__ import absolute_import, division, print_function

import datetime
import json
import os

import numpy as np

from RMS.Logger import getLogger

log = getLogger("logger")


RADIOMETRIC_FILE_SUFFIX = "radiometric.json"
SKY_PATCH_HALF = 30              # px - half-size of the measurement patch
MAX_FRAMES = 10                  # frames sampled per night
MIN_SKY_ADU = 1.5                # below this the sky signal is in the noise - skip frame
MIN_LEVER = 3.0                  # min (p95/p05) model-brightness contrast for regression
MOON_PHASE_MAX = 25.0            # percent - frames with a brighter risen moon are skipped

# Continuous bias tracking: the working bias is a robust trailing statistic over nightly
# observations - there are no discrete refits and nothing is trusted forever
BIAS_WINDOW = 14                 # nightly regression observations in the trailing window
BIAS_MIN_OBS = 5                 # observations before tracking outranks a stored seed
BIAS_STEP_ADU = 4.0              # persistent offset of recent obs that signals a pedestal
                                 # step (firmware/gain change) - window resets, heals in
                                 # ~BIAS_STEP_NIGHTS nights
BIAS_STEP_NIGHTS = 3
FLOOR_GUARD_ADU = 3.0            # physics: night floor >= bias; a floor below the working
                                 # bias by more than this = pedestal dropped = demote to
                                 # limit rather than report a wrong absolute
BIAS_OBS_MAX_AGE_NIGHTS = 40     # calendar nights after which an observation no longer
                                 # enters the estimator, so a stored seed regains authority
                                 # after a long observation gap. This aging used to be an
                                 # implicit side effect of the 40-entry file retention.
HISTORY_KEEP = 3660              # nightly entries kept in the file (~10 years). This is
                                 # RETENTION for the long-term calibration record (see
                                 # Utils.PlotCalibrationHistory), not an estimator window -
                                 # the estimator is bounded by the BIAS_* constants above
                                 # and BIAS_OBS_MAX_AGE_NIGHTS

# Aperture (PSF wing loss) correction: the star extractor sums flux in a small segment
# window, so mag_lev absorbs the missing PSF wings (~30% of the flux at a typical
# station). Extended sky has no aperture to lose flux from, so converting it through
# mag_lev overestimates its brightness by 2.5*log10(1/f). The capture fraction f is
# measured nightly from growth curves of bright isolated stars on the same avepixels
# the sky is measured on, and tracked as a slow trailing median (see APERTURE_WINDOW).
APERTURE_R_MAX = 16              # px - widest growth-curve aperture (PSF wings converge
                                 # well inside this at typical focus)
APERTURE_BG_IN = 18              # px - background annulus (inner, outer)
APERTURE_BG_OUT = 26
APERTURE_MIN_INTENS = 400        # star_list intensity floor at 8 bits (scaled by depth) -
                                 # growth curves on fainter stars drown in sky noise
APERTURE_ISO_RADIUS = 30         # px - no other detected star this close
APERTURE_SAT_FRAC = 0.86         # of full scale - reject stars with clipped cores
APERTURE_MIN_STARS = 10          # star samples required for tonight's f observation
APERTURE_WINDOW = 60             # trailing nightly f observations in the estimator.
                                 # Deliberately much longer than BIAS_WINDOW: tonight's
                                 # TRUE f already cancels out of the sky value through
                                 # the nightly mag_lev refit (same stars, same windows),
                                 # so the correction only sets the ABSOLUTE scale and
                                 # recency buys nothing - a noisy nightly f measurably
                                 # degraded meter tracking (0.066 -> 0.120 mag RMS at
                                 # SI0002) while a long median matched the uncorrected
                                 # tracking exactly
APERTURE_OBS_MAX_AGE_NIGHTS = 365  # own aging clock, slower than the bias one for the
                                 # same reason - a stale f mis-sets the absolute scale
                                 # by at most the focus drift, which self-heals over
                                 # the window, while an emptied estimator would drop
                                 # the correction (a known 0.4 mag error) entirely
APERTURE_F_RANGE = (0.2, 1.5)    # sanity range for a single-star capture fraction

# Approximate zenith SQM (mag/arcsec^2) to Bortle class mapping
BORTLE_SCALE = [(21.99, "1"), (21.89, "2"), (21.69, "3"), (21.25, "4"), (20.49, "5"),
                (19.50, "6"), (18.94, "7"), (18.38, "8"), (-99.0, "9")]


def bortleClass(sqm):
    """ Approximate Bortle class for a zenith sky brightness. """

    for limit, name in BORTLE_SCALE:
        if sqm >= limit:
            return name

    return "9"


def _radiometricPath(config):
    return os.path.join(os.path.expanduser(config.data_dir),
        "{:s}_{:s}".format(str(config.stationID), RADIOMETRIC_FILE_SUFFIX))


def loadRadiometricCalibration(config):
    """ Load the station's radiometric tracking file: {seed: {...}, nights: {...}}.

    Legacy flat files ({bias, method, ...}) are read as a seed. Returns an empty
    structure if nothing is stored.
    """

    cal = dict(seed=None, nights={})

    try:
        path = _radiometricPath(config)
        if os.path.isfile(path):
            with open(path) as f:
                raw = json.load(f)

            if "nights" in raw or "seed" in raw:
                cal["seed"] = raw.get("seed")
                cal["nights"] = raw.get("nights", {})
                if raw.get("handover") is not None:
                    cal["handover"] = raw["handover"]
            elif "bias" in raw:
                # Legacy single-value file becomes the seed observation
                cal["seed"] = dict(bias=float(raw["bias"]),
                    method=raw.get("method", "stored"), date=raw.get("fit_date"))
    except Exception:
        pass

    return cal


def resolveWorkingBias(cal, tonight_obs, tonight_floor, night_key=None):
    """ Continuous bias tracking: fold tonight's observation into the history and return
        the working bias. Pure function - the caller persists the updated cal.

    The working bias is the median of the trailing regression observations. A stored seed
    (e.g. a host-level overlap-graph calibration) is used until enough observations
    accumulate, then tracking takes over - nothing is trusted forever. Observations older
    than BIAS_OBS_MAX_AGE_NIGHTS never enter the estimator, so after a long observation
    gap (weather, hardware down) the seed regains authority instead of stale observations
    outranking it. The first night tracking outranks a seed is recorded in cal["handover"]
    (date, both values, delta) and a disagreement larger than BIAS_STEP_ADU is logged as a
    warning - the two calibrations both claim authority, so a large delta means one of
    them is wrong (stale seed OR systematic regression error) and a human should glance at
    the calibration history plot. Two change detectors run every night:

      - Step detector: if the last BIAS_STEP_NIGHTS observations all sit on the same side
        of the trailing median by more than BIAS_STEP_ADU, the pedestal has stepped
        (firmware/gain change); the working value snaps to the recent observations.
      - Floor guard: physics demands night_floor >= bias. A floor below the working bias
        by more than FLOOR_GUARD_ADU proves the pedestal DROPPED; the working value is
        withdrawn (limit tier) unless tonight's own observation vouches for a new one -
        and only an observation that itself respects the floor may vouch.

    Arguments:
        cal: [dict] {seed, nights, ...} as loaded. Extra keys (e.g. handover) are preserved.
        tonight_obs: [float] Tonight's regression bias observation, or None.
        tonight_floor: [float] Tonight's darkest-patch level (always available).

    Keyword arguments:
        night_key: [str] YYYYMMDD key to record tonight under. None (default) uses the
            current UTC date; a historical replay (Utils.PlotCalibrationHistory) passes
            the keys from the stored history to reproduce every night's decision.

    Return:
        (bias, source, cal): working bias (or None -> limit), a source label, updated cal.
    """

    nights = dict(cal.get("nights", {}))
    key = night_key if night_key is not None else datetime.datetime.utcnow().strftime("%Y%m%d")
    nights[key] = dict(bias=(round(float(tonight_obs), 2) if tonight_obs is not None else None),
                       floor=round(float(tonight_floor), 2))
    nights = dict(sorted(nights.items())[-HISTORY_KEEP:])

    # Preserve keys beyond seed/nights (e.g. the handover record)
    cal = dict(cal)
    cal["nights"] = nights

    # Only observations recent enough to describe the current pedestal enter the estimator
    try:
        cutoff = (datetime.datetime.strptime(key, "%Y%m%d")
                  - datetime.timedelta(days=BIAS_OBS_MAX_AGE_NIGHTS)).strftime("%Y%m%d")
    except ValueError:
        cutoff = "00000000"

    obs = [(k, v["bias"]) for k, v in sorted(nights.items())
           if isinstance(v, dict) and v.get("bias") is not None and k >= cutoff]
    obs_vals = [b for _, b in obs][-BIAS_WINDOW:]

    seed = cal.get("seed")
    seed_bias = seed.get("bias") if isinstance(seed, dict) else None

    bias = None
    source = None

    if len(obs_vals) >= BIAS_MIN_OBS:
        med = float(np.median(obs_vals))
        recent = obs_vals[-BIAS_STEP_NIGHTS:]

        if len(obs_vals) >= BIAS_MIN_OBS + BIAS_STEP_NIGHTS and (
                all(r - med > BIAS_STEP_ADU for r in recent)
                or all(med - r > BIAS_STEP_ADU for r in recent)):
            bias = float(np.median(recent))
            source = "tracked (pedestal step detected)"
        else:
            bias = med
            source = "tracked ({:d} nights)".format(len(obs_vals))

        # First night tracking outranks a stored seed: record the handover so a large
        # disagreement is a visible event (log + history plot), not a silent jump
        if (seed_bias is not None) and (cal.get("handover") is None):
            delta = round(float(bias) - float(seed_bias), 2)
            cal["handover"] = dict(date=key, seed_bias=round(float(seed_bias), 2),
                tracked_bias=round(float(bias), 2), delta=delta)

            if abs(delta) > BIAS_STEP_ADU:
                log.warning("Radiometric tracking took over from the stored seed with a "
                    "{:+.1f} ADU disagreement (seed {:.1f}, tracked {:.1f}) - one of them "
                    "is wrong; inspect the calibration history".format(
                        delta, float(seed_bias), float(bias)))

    elif seed_bias is not None:
        bias = float(seed_bias)
        source = "seed ({:s})".format(str(seed.get("method", "stored")))

    elif obs_vals:
        bias = float(np.median(obs_vals))
        source = "tracking warmup ({:d} nights)".format(len(obs_vals))

    # Floor guard - runs regardless of where the working value came from. Tonight's own
    # observation may replace the withdrawn value only if it respects the same physics
    # (bias <= floor): an observation above the floor is exactly the failure mode the
    # guard exists to catch, so it must not vouch for itself.
    if (bias is not None) and (tonight_floor < bias - FLOOR_GUARD_ADU):
        if (tonight_obs is not None) and (tonight_obs <= tonight_floor):
            bias = float(tonight_obs)
            source = "tonight's observation (floor guard tripped)"
        else:
            bias = None
            source = "floor guard tripped - working bias withdrawn"

    return bias, source, cal


def frameApertureSamples(ave, star_list, bit_depth=8):
    """ Per-star PSF capture fractions f = windowed_intensity/total_flux on one avepixel.

    For every bright, isolated, unsaturated detected star, the total flux is the
    asymptote of an aperture growth curve with a local annulus background; the windowed
    intensity is the one the star extractor recorded (and the photometric calibration
    consumed), so f is exactly the deficit hidden in mag_lev.

    Arguments:
        ave: [2D ndarray] Avepixel image.
        star_list: [list] Recalibrated platepar star list rows [jd, y, x, intensity, ...].

    Keyword arguments:
        bit_depth: [int] Camera bit depth, used to scale the brightness/saturation cuts.

    Return:
        samples: [list of float] Per-star capture fractions.
    """

    h, w = ave.shape
    scale = 2.0**(bit_depth - 8)
    min_intens = APERTURE_MIN_INTENS*scale
    sat_level = APERTURE_SAT_FRAC*(2.0**bit_depth - 1)

    yy, xx = np.mgrid[-APERTURE_BG_OUT:APERTURE_BG_OUT + 1,
                      -APERTURE_BG_OUT:APERTURE_BG_OUT + 1]
    rr = np.hypot(yy, xx)
    bg_mask = (rr >= APERTURE_BG_IN) & (rr <= APERTURE_BG_OUT)

    positions = np.array([(row[2], row[1]) for row in star_list]) if star_list else None

    samples = []
    for row in (star_list or []):
        _, y, x, intens = row[0], row[1], row[2], row[3]
        if intens < min_intens:
            continue
        if not (APERTURE_BG_OUT < x < w - APERTURE_BG_OUT - 1
                and APERTURE_BG_OUT < y < h - APERTURE_BG_OUT - 1):
            continue

        # Isolation: any other detected star inside the growth region spoils the curve
        dist = np.hypot(positions[:, 0] - x, positions[:, 1] - y)
        if np.sum(dist < APERTURE_ISO_RADIUS) > 1:
            continue

        xi, yi = int(round(x)), int(round(y))
        cut = ave[yi - APERTURE_BG_OUT:yi + APERTURE_BG_OUT + 1,
                  xi - APERTURE_BG_OUT:xi + APERTURE_BG_OUT + 1]
        if cut.shape != rr.shape:
            continue
        if cut[APERTURE_BG_OUT - 2:APERTURE_BG_OUT + 3,
               APERTURE_BG_OUT - 2:APERTURE_BG_OUT + 3].max() > sat_level:
            continue

        bg = np.median(cut[bg_mask])
        growth = [float(np.sum(cut[rr <= r] - bg))
                  for r in range(2, APERTURE_R_MAX + 1)]
        total = float(np.mean(growth[-3:]))
        if total <= 0 or growth[6] <= 0:
            continue
        # A curve still climbing at the edge is a blend or a background gradient
        if growth[-1] > 1.15*growth[-4]:
            continue

        f = intens/total
        if APERTURE_F_RANGE[0] < f < APERTURE_F_RANGE[1]:
            samples.append(f)

    return samples


def resolveApertureCorrection(cal, tonight_f, night_key=None):
    """ Continuous tracking of the PSF capture fraction f, mirroring the bias tracker:
        fold tonight's observation into the nightly history and return the working value
        (trailing median). Pure function - the caller persists the updated cal.

    Unlike the bias there is no seed alternative and no floor physics: f is a throughput
    property of the optics + extraction window, so a robust median over recent nights is
    used as soon as a single observation exists (an uncorrected value is KNOWN to be
    wrong by 2.5*log10(1/f), so waiting for a long warmup would preserve a worse error).

    Stability beats recency here, hence the long window and slow aging: tonight's true
    f is already inside tonight's mag_lev (refit nightly from the same windowed sums)
    and cancels out of the sky value, so night-to-night tracking is protected with or
    without the correction - its only job is the absolute scale. Applying a noisy
    per-night f injects that noise directly into an otherwise clean series (measured
    at SI0002: 0.066 -> 0.120 mag RMS vs a meter), while the long median fixed the
    absolute scale at no tracking cost.

    Arguments:
        cal: [dict] {seed, nights, ...} as loaded. Only nights[key]["aperture"] is touched.
        tonight_f: [float] Tonight's median capture fraction, or None.

    Keyword arguments:
        night_key: [str] YYYYMMDD key, as in resolveWorkingBias.

    Return:
        (f, n_obs, cal): working capture fraction (or None -> no correction), the number
            of observations in the estimator, updated cal.
    """

    nights = dict(cal.get("nights", {}))
    key = night_key if night_key is not None else datetime.datetime.utcnow().strftime("%Y%m%d")

    entry = dict(nights.get(key, {}))
    if tonight_f is not None:
        entry["aperture"] = round(float(tonight_f), 3)
    nights[key] = entry
    nights = dict(sorted(nights.items())[-HISTORY_KEEP:])

    cal = dict(cal)
    cal["nights"] = nights

    try:
        cutoff = (datetime.datetime.strptime(key, "%Y%m%d")
                  - datetime.timedelta(days=APERTURE_OBS_MAX_AGE_NIGHTS)).strftime("%Y%m%d")
    except ValueError:
        cutoff = "00000000"

    obs = [v["aperture"] for k, v in sorted(nights.items())
           if isinstance(v, dict) and v.get("aperture") is not None
           and k >= cutoff][-APERTURE_WINDOW:]

    if not obs:
        return None, 0, cal

    return float(np.median(obs)), len(obs), cal


def estimateBiasByRegression(config, dir_path, dome_model, pps, mask):
    """ Tier 2: single-camera bias from regressing patch ADU against the dome model's
        brightness field across the FOV. Only valid when the FOV spans enough brightness
        contrast (lever); returns None otherwise.

    Arguments:
        config: [Config]
        dir_path: [str] Night directory with FF files.
        dome_model: [LightDomeModel]
        pps: [dict] ff_name -> recalibrated Platepar (auto_recalibrated only).
        mask: [Mask object]

    Return:
        bias: [float] ADU, or None if the FOV has no lever or the fit failed.
    """

    from RMS.Formats import FFfile
    from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP
    from RMS.Astrometry.Conversions import date2JD, raDec2AltAz

    ffs = sorted(pps.keys())
    if len(ffs) < 3:
        return None

    picks = [ffs[i] for i in np.unique(np.linspace(0, len(ffs) - 1, 5).astype(int))]

    biases = []
    for ff_name in picks:

        pp = pps[ff_name]
        ff = FFfile.read(dir_path, ff_name)
        if ff is None:
            continue

        ave = ff.avepixel.astype(float)
        w, h = pp.X_res, pp.Y_res
        date = FFfile.getMiddleTimeFF(ff_name, config.fps, ret_milliseconds=True)
        jd = date2JD(*date)

        step = 60
        cx, cy, adu = [], [], []
        for yy in range(step, h - step, step):
            for xx in range(step, w - step, step):
                if mask is not None and mask.img is not None and \
                        np.mean(mask.img[yy:yy + step, xx:xx + step] == 0) > 0.05:
                    continue
                cx.append(xx + step/2)
                cy.append(yy + step/2)
                adu.append(np.median(ave[yy:yy + step, xx:xx + step]))

        cx = np.array(cx, float)
        cy = np.array(cy, float)
        adu = np.array(adu)

        _, ra, dec, _ = xyToRaDecPP([date]*len(cx), list(cx), list(cy), [1]*len(cx), pp,
            extinction_correction=False, precompute_pointing_corr=True)
        az, alt = raDec2AltAz(np.array(ra), np.array(dec), jd, pp.lat, pp.lon)

        up = alt >= 8
        if np.sum(up) < 40:
            continue

        B = dome_model.skyBrightness(az[up], alt[up])
        vc = getattr(pp, "vignetting_coeff", 0.0) or 0.0
        r = np.hypot(cx[up] - w/2.0, cy[up] - h/2.0)
        x = B*np.cos(vc*r)**4
        y = adu[up]

        # Lever check: without contrast, bias and gain are inseparable
        if np.percentile(x, 95) < MIN_LEVER*np.percentile(x, 5):
            return None

        sel = np.ones(len(x), bool)
        for _ in range(3):
            c1, c0 = np.polyfit(x[sel], y[sel], 1)
            res = y - (c0 + c1*x)
            sel = np.abs(res) < 2.5*np.std(res[sel])

        if c1 > 0:
            biases.append(c0)

    if len(biases) < 3:
        return None

    return float(np.median(biases))


def _moonIsUp(jd, lat, lon):
    """ True if a moon brighter than MOON_PHASE_MAX percent is above the horizon. """

    try:
        import ephem
        from RMS.Astrometry.Conversions import jd2Date

        obs = ephem.Observer()
        obs.lat = str(lat)
        obs.lon = str(lon)
        obs.date = jd2Date(jd, dt_obj=True)

        moon = ephem.Moon()
        moon.compute(obs)

        return (np.degrees(float(moon.alt)) > 0) and (float(moon.phase) > MOON_PHASE_MAX)

    except Exception:
        return False


def _recordNight(config, dir_path, record, dome_model):
    """ Persist a night's outcome everywhere it belongs: the per-night JSON (archive), the
        station's long-term history, and the refreshed long-term plot (copied into the
        night directory so it travels with the archive). """

    night = record["night"]

    try:
        with open(os.path.join(dir_path, "{:s}_sky_quality.json".format(night)), "w") as f:
            json.dump(record, f, indent=1)
    except Exception:
        pass

    try:
        from Utils.PlotSkyQuality import appendSkyQualityHistory, plotStationSkyQuality
        import shutil

        appendSkyQualityHistory(config, night, record)
        png = plotStationSkyQuality(config, dome_model=dome_model)
        if png is not None:
            shutil.copy(png, os.path.join(dir_path,
                "{:s}_sky_quality.png".format(night)))
    except Exception as e:
        log.debug("Sky quality history/plot update failed: {}".format(e))


def _writeSkipRecord(config, dir_path, reason, dome_model=None):
    """ A skipped night still writes its record: an absent file must never be ambiguous
        between 'skipped by design' and 'broken'. """

    night = os.path.basename(os.path.normpath(dir_path))
    record = dict(stationID=str(config.stationID), night=night, status="skipped",
        reason=reason)

    _recordNight(config, dir_path, record, dome_model)

    log.info("Sky quality: skipped - {:s}".format(reason))

    return None


def measureSkyQuality(config, dir_path, dome_model, recalibrated_platepars, time_intervals,
        mask):
    """ Measure and record the night's sky quality. Never raises - guarded by the caller.
        A record is written even on skipped nights (with a status and reason).

    Arguments:
        config: [Config]
        dir_path: [str] Night directory (must contain FF files for measurements).
        dome_model: [LightDomeModel] May be None (tier 2 then unavailable).
        recalibrated_platepars: [dict] ff_name -> Platepar.
        time_intervals: [list] Clear intervals from detectClouds ([] = no measurement).
        mask: [Mask object]

    Return:
        result: [dict] The written sky quality record, or None if skipped.
    """

    from RMS.Formats import FFfile
    from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP
    from RMS.Astrometry.Conversions import date2JD, raDec2AltAz

    if not time_intervals:
        return _writeSkipRecord(config, dir_path, "no clear intervals (cloudy night)", dome_model)

    pps = {ff: pp for ff, pp in recalibrated_platepars.items()
           if getattr(pp, "auto_recalibrated", False)}
    if len(pps) < 3:
        return _writeSkipRecord(config, dir_path, "too few recalibrated frames", dome_model)

    # --- tonight's bias observation: attempted EVERY night (continuous tracking), gated
    # only by the physics (lever) inside the regression ---
    tonight_obs = None
    if dome_model is not None:
        tonight_obs = estimateBiasByRegression(config, dir_path, dome_model, pps, mask)

    # --- choose the measurement patch: highest-altitude unmasked cell ---
    ffs = sorted(pps.keys())
    pp0 = pps[ffs[len(ffs)//2]]
    w, h = pp0.X_res, pp0.Y_res
    date0 = FFfile.getMiddleTimeFF(ffs[len(ffs)//2], config.fps, ret_milliseconds=True)
    jd0 = date2JD(*date0)

    gx, gy = np.meshgrid(np.linspace(120, w - 120, 10), np.linspace(120, h - 120, 6))
    gx, gy = gx.ravel(), gy.ravel()
    _, ra, dec, _ = xyToRaDecPP([date0]*len(gx), list(gx), list(gy), [1]*len(gx), pp0,
        extinction_correction=False, precompute_pointing_corr=True)
    az, alt = raDec2AltAz(np.array(ra), np.array(dec), jd0, pp0.lat, pp0.lon)

    order = np.argsort(alt)[::-1]
    patch = None
    for i in order:
        x0, y0 = int(gx[i]), int(gy[i])
        if mask is not None and mask.img is not None and np.mean(
                mask.img[y0 - SKY_PATCH_HALF:y0 + SKY_PATCH_HALF,
                         x0 - SKY_PATCH_HALF:x0 + SKY_PATCH_HALF] == 0) > 0.02:
            continue
        patch = (x0, y0, float(az[i]), float(alt[i]))
        break

    if patch is None:
        return _writeSkipRecord(config, dir_path, "no unmasked measurement patch", dome_model)
    x0, y0, patch_az, patch_alt = patch

    # --- frames: inside clear intervals, no bright risen moon ---
    usable = []
    for ff_name in ffs:
        t = FFfile.filenameToDatetime(ff_name)
        if not any(beg <= t <= end for beg, end in time_intervals):
            continue
        date = FFfile.getMiddleTimeFF(ff_name, config.fps, ret_milliseconds=True)
        if _moonIsUp(date2JD(*date), pp0.lat, pp0.lon):
            continue
        usable.append(ff_name)

    if not usable:
        return _writeSkipRecord(config, dir_path, "no moonless clear frames", dome_model)

    usable = [usable[i] for i in np.unique(np.linspace(0, len(usable) - 1,
        min(MAX_FRAMES, len(usable))).astype(int))]

    levels = []
    aperture_samples = []
    for ff_name in usable:
        ff = FFfile.read(dir_path, ff_name)
        if ff is None:
            continue
        ave = ff.avepixel.astype(float)
        levels.append((ff_name,
            float(np.median(ave[y0 - SKY_PATCH_HALF:y0 + SKY_PATCH_HALF,
                                x0 - SKY_PATCH_HALF:x0 + SKY_PATCH_HALF]))))

        # Same frames, same avepixels: sample the PSF capture fraction from the
        # detected stars so tonight's aperture correction rides along for free
        try:
            aperture_samples += frameApertureSamples(ave,
                getattr(pps[ff_name], "star_list", None) or [],
                bit_depth=getattr(config, "bit_depth", 8))
        except Exception as e:
            log.debug("Aperture sampling failed on {:s}: {}".format(ff_name, e))

    if not levels:
        return _writeSkipRecord(config, dir_path, "no readable FF files in the night directory", dome_model)

    tonight_f = None
    if len(aperture_samples) >= APERTURE_MIN_STARS:
        tonight_f = float(np.median(aperture_samples))

    # --- continuous bias + aperture tracking: fold tonight's observations into the
    # history, get the working values, persist the updated history once ---
    tonight_floor = min(v for _, v in levels)
    cal = loadRadiometricCalibration(config)
    bias, source, cal = resolveWorkingBias(cal, tonight_obs, tonight_floor)
    f_aperture, f_nights, cal = resolveApertureCorrection(cal, tonight_f)

    try:
        with open(_radiometricPath(config), "w") as f:
            json.dump(cal, f, indent=1)
    except Exception as e:
        log.debug("Could not store radiometric history: {}".format(e))

    if bias is not None:
        tier = 2 if source.startswith(("tracked", "tonight", "tracking")) else 1
        method = source
        absolute = True
    else:
        # Limit: the darkest observed patch level bounds the bias from above
        bias = tonight_floor
        tier = 3
        method = "night-floor limit" + ("" if source is None else " ({:s})".format(source))
        absolute = False

    if source and "floor guard" in source:
        log.warning("Sky quality: floor guard tripped - pedestal appears to have moved "
                    "(floor {:.1f} ADU): {:s}".format(tonight_floor, source))

    sqm_values = []
    for ff_name, level in levels:
        sky = level - bias
        if sky < MIN_SKY_ADU:
            continue
        pp = pps[ff_name]
        # The pedestal is uniform but the sky signal is vignetted: undo the throughput
        # loss at the patch position (mag_lev calibrates vignetting-corrected counts,
        # and the bias regression fits against a devignetted model for the same reason)
        vc = getattr(pp, "vignetting_coeff", 0.0) or 0.0
        r = np.hypot(x0 - pp.X_res/2.0, y0 - pp.Y_res/2.0)
        sky /= np.cos(vc*r)**4
        # Aperture correction: mag_lev was fitted on windowed star sums that miss the
        # PSF wings; scaling the sky by the tracked capture fraction puts the extended
        # source on the same flux scale the zero point was calibrated on
        if f_aperture is not None:
            sky *= f_aperture
        area = (3600.0/pp.F_scale)**2
        sqm_values.append(pp.mag_lev - 2.5*np.log10(sky) + 2.5*np.log10(area))

    if not sqm_values:
        return _writeSkipRecord(config, dir_path, "sky signal below noise at the measurement patch", dome_model)

    sqm = float(np.median(sqm_values))

    # Bortle is a ZENITH scale: only class near-zenith measurements, otherwise the
    # naturally brighter low-altitude sky would be misread as light pollution class
    near_zenith = patch_alt >= 70.0

    result = dict(
        stationID=str(config.stationID),
        night=os.path.basename(os.path.normpath(dir_path)),
        patch=dict(x=x0, y=y0, az=round(patch_az, 1), alt=round(patch_alt, 1)),
        bias=dict(value=round(float(bias), 2), tier=tier, method=method),
        aperture=(dict(f=round(f_aperture, 3), n_stars=len(aperture_samples),
                       nights=f_nights) if f_aperture is not None
                  else dict(f=None, n_stars=len(aperture_samples), nights=0)),
        n_frames=len(sqm_values),
        sqm=round(sqm, 2),
        absolute=absolute,
        bortle=bortleClass(sqm) if (absolute and near_zenith) else None,
        note=("above-atmosphere magnitude convention (~+0.2 mag vs as-seen)" if absolute
              else "LIMIT only: bias unknown, sky is AT LEAST this bright"),
    )

    result["status"] = "ok"
    _recordNight(config, dir_path, result, dome_model)

    ap_str = ("aperture f={:.2f} ({:d} nights)".format(f_aperture, f_nights)
              if f_aperture is not None else "no aperture correction")
    if absolute:
        bortle_str = "Bortle {:s}, ".format(result["bortle"]) if result["bortle"] else ""
        log.info("Sky quality: {:.2f} mag/arcsec2 at alt {:.0f} deg ({:s}"
                 "tier {:d}, {:d} frames, {:s})".format(sqm, patch_alt, bortle_str,
                 tier, len(sqm_values), ap_str))
    else:
        log.info("Sky quality: brighter than {:.2f} mag/arcsec2 at alt {:.0f} deg "
                 "(limit only - no bias calibration, {:s})".format(sqm, patch_alt, ap_str))

    return result
