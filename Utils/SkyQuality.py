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
inside clear intervals with no significant moon. Results are written to
<night>_sky_quality.json in the night directory and one line is logged.

Convention note: mag_lev maps counts to above-atmosphere magnitudes, so the SQM values
read ~0.2 mag brighter than the as-seen convention at the zenith.
"""

from __future__ import absolute_import, division, print_function

import datetime
import json
import os

import numpy as np

from RMS.Logger import getLogger

log = getLogger("logger")


RADIOMETRIC_FILE_SUFFIX = "radiometric.json"
RADIOMETRIC_REFRESH_DAYS = 30    # auto (tier 2) calibrations older than this are redone
SKY_PATCH_HALF = 30              # px - half-size of the measurement patch
MAX_FRAMES = 10                  # frames sampled per night
MIN_SKY_ADU = 1.5                # below this the sky signal is in the noise - skip frame
MIN_LEVER = 3.0                  # min (p95/p05) model-brightness contrast for tier 2
MOON_PHASE_MAX = 25.0            # percent - frames with a brighter risen moon are skipped

# Approximate zenith SQM (mag/arcsec^2) to Bortle class mapping
BORTLE_SCALE = [(21.99, "1"), (21.89, "2"), (21.69, "3"), (21.25, "4"), (20.49, "5"),
                (19.50, "6"), (18.94, "7"), (18.38, "8"), (-99.0, "9")]


def bortleClass(sqm):
    """ Approximate Bortle class for a zenith sky brightness. """

    for limit, name in BORTLE_SCALE:
        if sqm >= limit:
            return name

    return "9"


def loadRadiometricCalibration(config):
    """ Load the station's stored radiometric calibration, or None.

    Return:
        cal: [dict] {bias, method, fit_date, ...} or None.
    """

    try:
        path = os.path.join(os.path.expanduser(config.data_dir),
            "{:s}_{:s}".format(str(config.stationID), RADIOMETRIC_FILE_SUFFIX))
        if os.path.isfile(path):
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass

    return None


def _calibrationIsUsable(cal, dome_model):
    """ Decide whether a stored calibration should be used as-is.

    Manual/host-level calibrations (anything except method "model-regression") are always
    trusted. Auto ones are refreshed when old or when the dome model they were regressed
    against has been refitted.
    """

    if cal is None or ("bias" not in cal):
        return False

    if cal.get("method") != "model-regression":
        return True

    try:
        age = (datetime.datetime.utcnow()
               - datetime.datetime.strptime(cal.get("fit_date", "1970-01-01"), "%Y-%m-%d")).days
        if age > RADIOMETRIC_REFRESH_DAYS:
            return False
    except ValueError:
        return False

    if dome_model is not None and \
            cal.get("model_fit_date") != dome_model.model.get("fit_date"):
        return False

    return True


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


def measureSkyQuality(config, dir_path, dome_model, recalibrated_platepars, time_intervals,
        mask):
    """ Measure and record the night's sky quality. Never raises - guarded by the caller.

    Arguments:
        config: [Config]
        dir_path: [str] Night directory (must contain FF files for measurements).
        dome_model: [LightDomeModel] May be None (tier 2 then unavailable).
        recalibrated_platepars: [dict] ff_name -> Platepar.
        time_intervals: [list] Clear intervals from detectClouds ([] = no measurement).
        mask: [Mask object]

    Return:
        result: [dict] The written sky quality record, or None.
    """

    from RMS.Formats import FFfile
    from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP
    from RMS.Astrometry.Conversions import date2JD, raDec2AltAz

    if not time_intervals:
        log.info("Sky quality: no clear intervals - skipped")
        return None

    pps = {ff: pp for ff, pp in recalibrated_platepars.items()
           if getattr(pp, "auto_recalibrated", False)}
    if len(pps) < 3:
        return None

    # --- resolve the bias through the tiers ---
    cal = loadRadiometricCalibration(config)
    tier = None

    if _calibrationIsUsable(cal, dome_model):
        bias = float(cal["bias"])
        tier = 1 if cal.get("method") != "model-regression" else 2
        method = cal.get("method", "stored")

    else:
        bias = None
        if dome_model is not None:
            bias = estimateBiasByRegression(config, dir_path, dome_model, pps, mask)

        if bias is not None:
            tier = 2
            method = "model-regression"
            try:
                path = os.path.join(os.path.expanduser(config.data_dir),
                    "{:s}_{:s}".format(str(config.stationID), RADIOMETRIC_FILE_SUFFIX))
                with open(path, "w") as f:
                    json.dump(dict(bias=round(bias, 2), method=method,
                        fit_date=datetime.datetime.utcnow().strftime("%Y-%m-%d"),
                        model_fit_date=dome_model.model.get("fit_date")), f, indent=1)
                log.info("Sky quality: self-primed bias {:.1f} ADU (model regression)".format(bias))
            except Exception as e:
                log.debug("Could not store radiometric calibration: {}".format(e))
        else:
            tier = 3
            method = "night-floor limit"

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
        return None
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
        log.info("Sky quality: no moonless clear frames - skipped")
        return None

    usable = [usable[i] for i in np.unique(np.linspace(0, len(usable) - 1,
        min(MAX_FRAMES, len(usable))).astype(int))]

    levels = []
    for ff_name in usable:
        ff = FFfile.read(dir_path, ff_name)
        if ff is None:
            continue
        ave = ff.avepixel.astype(float)
        levels.append((ff_name,
            float(np.median(ave[y0 - SKY_PATCH_HALF:y0 + SKY_PATCH_HALF,
                                x0 - SKY_PATCH_HALF:x0 + SKY_PATCH_HALF]))))

    if not levels:
        return None

    # Tier 3: the darkest observed patch level bounds the bias from above
    if tier == 3:
        bias = min(v for _, v in levels)

    sqm_values = []
    for ff_name, level in levels:
        sky = level - bias
        if sky < MIN_SKY_ADU:
            continue
        pp = pps[ff_name]
        area = (3600.0/pp.F_scale)**2
        sqm_values.append(pp.mag_lev - 2.5*np.log10(sky) + 2.5*np.log10(area))

    if not sqm_values:
        log.info("Sky quality: sky signal below noise at the measurement patch - skipped")
        return None

    sqm = float(np.median(sqm_values))
    absolute = tier in (1, 2)

    # Bortle is a ZENITH scale: only class near-zenith measurements, otherwise the
    # naturally brighter low-altitude sky would be misread as light pollution class
    near_zenith = patch_alt >= 70.0

    result = dict(
        stationID=str(config.stationID),
        night=os.path.basename(os.path.normpath(dir_path)),
        patch=dict(x=x0, y=y0, az=round(patch_az, 1), alt=round(patch_alt, 1)),
        bias=dict(value=round(float(bias), 2), tier=tier, method=method),
        n_frames=len(sqm_values),
        sqm=round(sqm, 2),
        absolute=absolute,
        bortle=bortleClass(sqm) if (absolute and near_zenith) else None,
        note=("above-atmosphere magnitude convention (~+0.2 mag vs as-seen)" if absolute
              else "LIMIT only: bias unknown, sky is AT LEAST this bright"),
    )

    try:
        out = os.path.join(dir_path, "{:s}_sky_quality.json".format(result["night"]))
        with open(out, "w") as f:
            json.dump(result, f, indent=1)
    except Exception as e:
        log.debug("Could not write sky quality file: {}".format(e))

    if absolute:
        bortle_str = "Bortle {:s}, ".format(result["bortle"]) if result["bortle"] else ""
        log.info("Sky quality: {:.2f} mag/arcsec2 at alt {:.0f} deg ({:s}"
                 "tier {:d}, {:d} frames)".format(sqm, patch_alt, bortle_str,
                 tier, len(sqm_values)))
    else:
        log.info("Sky quality: brighter than {:.2f} mag/arcsec2 at alt {:.0f} deg "
                 "(limit only - no bias calibration)".format(sqm, patch_alt))

    return result
