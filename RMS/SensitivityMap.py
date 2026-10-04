""" Per-camera image-plane sensitivity map.

The stellar limiting magnitude of a camera varies across its own image: vignetting, the growth
of the PSF toward the edges, defocus and the light-pollution gradient all change how deep the
star extractor reaches at a given pixel. The map stores the 50% detection limiting magnitude per
image block, fitted from catalog-star hit/miss trials on dark, moonless frames
(Utils.FitCameraSensitivityMap). It is the camera's detection sensitivity as the detector actually
sees it, which is what the flux collection area needs. The platepar's vignetting coefficient is
only the illumination part of that and says nothing about the sky background or the PSF.

File: <data_dir>/<stationID>_sensitivity_map.json. Its presence activates the map in the flux
collection area (Utils.Flux); delete it to fall back to the platepar vignetting and extinction.
The map used for a night is also archived in the night directory as
<night>_sensitivity_map.json, and a flux run on that night prefers the archived copy.

Two reference values make the map usable on nights other than the ones it was fitted on:
    - lm_ref: the best block's LM. Per-block losses are expressed relative to it, so the bin's
      stellar LM stands for the best block and every block is a loss (ratio <= 1), the same
      convention as the vignetting path.
    - mag_lev_ref: the photometric zero point of the frames the map was fitted on. The night's
      zero point measures transparency, so the bin LM is lm_ref moved by the zero-point LM
      model's response to (mag_lev_bin - mag_lev_ref). The zero point also moves when the
      platepar's vignetting coefficient is changed, so the stored coefficient is kept and the
      reference is shifted to the current one before comparing.

A map describes the camera as configured when it was fitted. It keeps itself honest in two ways
(see mapStaleness): a fingerprint of the configuration that sets the camera's intensity units and
detection (gamma, bit depth, camera settings file, extraction gate, resolution, pointing), and the
night's own photometry (zero point and matched-star depth) compared with the map's references. A
hard mismatch (units, geometry) makes the station map unusable until it is refitted; soft ones
(age, gate retune, a deeper camera) only ask for a refit. Utils.FitCameraSensitivityMap.
ensureSensitivityMap performs that refit nightly from the clearest recent nights of the same
intensity epoch, so nobody has to remember.
"""

from __future__ import absolute_import, division, print_function

import datetime
import hashlib
import json
import os

import numpy as np


SENSITIVITY_MAP_FILE_SUFFIX = "sensitivity_map.json"

# A block needs at least this many trials for its logistic fit to mean anything
MIN_TRIALS_PER_BLOCK = 500

# Plausible logistic width of the detection rolloff (mag)
LOGISTIC_WIDTH_RANGE = (0.05, 1.5)

# A block LM spread larger than this across one camera is a broken fit, not a camera
MAX_BLOCK_LM_SPREAD = 4.0

# Largest zero-point LM shift applied relative to the map's reference (mag). Transparency moves a
# photometric night's zero point by tenths; a shift beyond this means the camera's intensity units
# changed (firmware, gain, gamma) and the map must be refitted. It is clipped so a units change
# cannot silently add magnitudes of depth to the collection area
MAX_ZERO_POINT_SHIFT = 1.5

# Staleness thresholds (mapStaleness)
REFIT_AGE_DAYS = 30          # dead-man backstop: refit at least monthly (a refit is cheap)
ZP_EPOCH_TOL = 1.0           # mag - a night's zero point this far from the reference is a units change
DEPTH_GAIN_TOL = 0.5         # mag - the camera reaching this much deeper than at fit time
POINTING_TOL_DEG = 3.0       # deg - the map's light-pollution structure is tied to the sky it saw

# Matched stars a frame needs for its depth (90th percentile matched magnitude) to count
DEPTH_MIN_STARS = 20

# Configuration keys whose change invalidates a map outright: the camera's intensity units or
# its geometry are different, so the stored references no longer describe this camera. The
# remaining fingerprint keys (extraction gate, frame rate) only move the depth by tenths and are
# soft: they ask for a refit but the map stays in use until one succeeds
FINGERPRINT_KEYS = ("gamma", "bit_depth", "camera_settings", "X_res", "Y_res", "pointing", "fps",
                    "star_gate_factor", "segment_radius", "max_feature_ratio", "roundness_threshold")
HARD_FINGERPRINT_KEYS = ("gamma", "bit_depth", "camera_settings", "X_res", "Y_res", "pointing")


def sensitivityMapQualityIssues(map_dict):
    """ Blocking quality issues of a map dictionary; an empty list means the map is usable.

    Arguments:
        map_dict: [dict] Map as written by Utils.FitCameraSensitivityMap.

    Return:
        issues: [list of str]
    """

    issues = []

    try:
        nbx, nby = int(map_dict["nbx"]), int(map_dict["nby"])
        lm = np.array(map_dict["LM"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return ["missing or malformed nbx/nby/LM"]

    if (nbx < 1) or (nby < 1) or (lm.size != nbx*nby):
        issues.append("LM has {:d} values for {:d}x{:d} blocks".format(lm.size, nbx, nby))
        return issues

    if not np.all(np.isfinite(lm)):
        issues.append("non-finite block LM")

    try:
        s = float(map_dict.get("s", 0.0))
    except (TypeError, ValueError):
        s = 0.0
    if not (LOGISTIC_WIDTH_RANGE[0] <= s <= LOGISTIC_WIDTH_RANGE[1]):
        issues.append("logistic width {:.2f} mag outside {:.2f}-{:.2f}".format(s, *LOGISTIC_WIDTH_RANGE))

    try:
        n_trials = int(map_dict.get("n_trials", 0))
    except (TypeError, ValueError):
        n_trials = 0
    if n_trials < MIN_TRIALS_PER_BLOCK*nbx*nby:
        issues.append("only {:d} trials for {:d} blocks (need {:d} per block)".format(
            n_trials, nbx*nby, MIN_TRIALS_PER_BLOCK))

    if np.all(np.isfinite(lm)) and (lm.max() - lm.min() > MAX_BLOCK_LM_SPREAD):
        issues.append("block LM spread {:.1f} mag is implausible".format(lm.max() - lm.min()))

    return issues



def _areaMeanVignettingShift(x_res, y_res, k_old, k_new):
    """ Change of the photometric zero point implied by a change of the cos^4 vignetting
        coefficient, averaged uniformly over the image area (mag to add to mag_lev).

        mag = -2.5 log10(I) + 10 log10 cos(k r) + mag_lev, so the offset that keeps calibrated
        magnitudes unchanged moves by the mean of 10 log10(cos(k_new r)/cos(k_old r)) with the
        sign flipped.
    """

    xs = np.linspace(0.5, x_res - 0.5, 192)
    ys = np.linspace(0.5, y_res - 0.5, 108)
    gx, gy = np.meshgrid(xs, ys)
    r = np.hypot(gx - x_res/2.0, gy - y_res/2.0)

    loss_old = -10.0*np.log10(np.clip(np.cos(k_old*r), 1e-6, 1.0))
    loss_new = -10.0*np.log10(np.clip(np.cos(k_new*r), 1e-6, 1.0))

    return float(-np.mean(loss_old - loss_new))



def _fileDigest(path):
    """ Short content digest of a file, or None if it cannot be read. """

    try:
        with open(path, "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()[:10]
    except Exception:
        return None



def configFingerprint(config, platepar=None):
    """ The configuration that sets a camera's intensity units and detection depth, as a dict
        the map stores and later compares against (see fingerprintChanges).

    Arguments:
        config: [Config] Station config.

    Keyword arguments:
        platepar: [Platepar] Current platepar for the image size and pointing (skipped if None).

    Return:
        [dict] Fingerprint.
    """

    fp = {}

    for key in ("gamma", "bit_depth", "fps", "star_gate_factor", "segment_radius", "max_feature_ratio",
                "roundness_threshold"):
        value = getattr(config, key, None)
        if value is not None:
            try:
                fp[key] = round(float(value), 4)
            except (TypeError, ValueError):
                fp[key] = str(value)

    # The camera settings file (gain, exposure, gamma curve, noise reduction) is what the
    # station pushes into the camera; its content is the closest thing to a firmware-side
    # configuration record. The configured path may be relative to the working directory, so
    # it is resolved in a fixed order that does not depend on where the process was started:
    # absolute as given, next to the station config, in the RMS root, then relative to the
    # working directory. An unreadable file gives None, which fingerprintChanges treats as
    # unknown rather than as a change
    settings_path = getattr(config, "camera_settings_path", None)
    digest = None
    if settings_path:
        settings_path = os.path.expanduser(settings_path)
        candidates = []
        if os.path.isabs(settings_path):
            candidates.append(settings_path)
        for base in (getattr(config, "config_file_path", None), getattr(config, "rms_root_dir", None)):
            if base:
                candidates.append(os.path.join(os.path.expanduser(base), os.path.basename(settings_path)))
        candidates.append(settings_path)
        for cand in candidates:
            if os.path.isfile(cand):
                digest = _fileDigest(cand)
                break
    fp["camera_settings"] = digest

    if platepar is not None:
        fp["X_res"] = int(platepar.X_res)
        fp["Y_res"] = int(platepar.Y_res)
        try:
            fp["pointing"] = [round(float(platepar.az_centre), 2), round(float(platepar.alt_centre), 2)]
        except (AttributeError, TypeError, ValueError):
            pass

    return fp



def fingerprintChanges(old, new):
    """ Compare two fingerprints over the keys both carry.

    Return:
        (hard, soft): [list of str] Descriptions of the hard and soft changes.
    """

    hard, soft = [], []

    for key in FINGERPRINT_KEYS:

        if (key not in old) or (key not in new):
            continue

        a, b = old[key], new[key]

        # Unknown on either side (e.g. the settings file could not be read) is not a change
        if (a is None) or (b is None):
            continue

        if key == "pointing":
            try:
                d_az = abs((float(b[0]) - float(a[0]) + 180.0) % 360.0 - 180.0)
                d_alt = abs(float(b[1]) - float(a[1]))
            except (IndexError, TypeError, ValueError):
                continue
            if max(d_az, d_alt) > POINTING_TOL_DEG:
                hard.append("pointing moved {:.1f} deg".format(max(d_az, d_alt)))
            continue

        if a != b:
            msg = "{:s} changed {} -> {}".format(key, a, b)
            (hard if key in HARD_FINGERPRINT_KEYS else soft).append(msg)

    return hard, soft



def frameDepth(star_list):
    """ Matched-star depth of one frame: the 90th percentile catalog magnitude of its matched
        stars, or None with too few stars. Same statistic as Utils.Flux.measureNightMatchedDepth.
    """

    if (not star_list) or (len(star_list) < DEPTH_MIN_STARS):
        return None

    try:
        return float(np.percentile([row[6] for row in star_list], 90))
    except (IndexError, TypeError, ValueError):
        return None



def nightPhotometry(night_dir):
    """ A night's photometric state from its recalibrated platepars: zero point, matched-star
        depth and vignetting coefficient. Only successfully recalibrated frames count (failed ones
        carry the SkyFit calibration star list, a fossil).

    Arguments:
        night_dir: [str] Night directory.

    Return:
        [dict] zp (median mag_lev), depth (max frame depth, cloud-immune), k (median vignetting
            coefficient), n_frames, matched (median matched stars per frame, a clarity measure) -
            or None if the night has no usable recalibrations.
    """

    for name in ("platepars_flux_recalibrated.json", "platepars_all_recalibrated.json"):

        path = os.path.join(night_dir, name)
        if not os.path.isfile(path):
            continue

        try:
            with open(path) as f:
                ppr = json.load(f)
        except Exception:
            continue

        zps, depths, ks, matched = [], [], [], []
        for key, pp in ppr.items():
            if (not isinstance(pp, dict)) or (not pp.get("auto_recalibrated")):
                continue
            try:
                zps.append(float(pp["mag_lev"]))
            except (KeyError, TypeError, ValueError):
                continue
            k = pp.get("vignetting_coeff")
            ks.append(float(k) if k is not None else 0.0)
            star_list = pp.get("star_list") or []
            matched.append(len(star_list))
            depth = frameDepth(star_list)
            if depth is not None:
                depths.append(depth)

        if zps:
            return dict(zp=float(np.median(zps)), depth=(float(max(depths)) if depths else None),
                        k=float(np.median(ks)), n_frames=len(zps), matched=float(np.median(matched)))

    return None



def mapStaleness(map_dict, config, platepar=None, night_dir=None):
    """ Why a station's map no longer describes its camera.

    Arguments:
        map_dict: [dict] The map.
        config: [Config] Station config.

    Keyword arguments:
        platepar: [Platepar] Current platepar (image size, pointing, vignetting coefficient).
        night_dir: [str] A night directory whose own photometry is compared with the map's
            references: a zero point an intensity epoch away is a units change (hard), a camera
            reaching clearly deeper than at fit time wants a refit (soft).

    Return:
        (hard, soft): [list of str] Hard reasons make the map unusable; soft ones ask for a refit.
    """

    hard, soft = [], []

    fp_map = map_dict.get("fingerprint")
    if isinstance(fp_map, dict):
        h, s = fingerprintChanges(fp_map, configFingerprint(config, platepar))
        hard += h
        soft += s

    fit_date = map_dict.get("fit_date")
    if fit_date is not None:
        try:
            age = (datetime.datetime.utcnow() - datetime.datetime.strptime(str(fit_date), "%Y-%m-%d")).days
            if age > REFIT_AGE_DAYS:
                soft.append("age {:d} d > {:d} d".format(age, REFIT_AGE_DAYS))
        except ValueError:
            pass

    if night_dir is not None:

        night = nightPhotometry(night_dir)
        zp_ref = map_dict.get("mag_lev_ref")

        if (night is not None) and (zp_ref is not None):

            zp_ref = float(zp_ref)
            k_ref = map_dict.get("vignetting_coeff_ref")
            x_res, y_res = map_dict.get("X_res"), map_dict.get("Y_res")
            if (k_ref is not None) and (x_res is not None) and (y_res is not None) \
                    and (abs(float(k_ref) - night["k"]) > 1e-9):
                zp_ref += _areaMeanVignettingShift(int(x_res), int(y_res), float(k_ref), night["k"])

            if abs(night["zp"] - zp_ref) > ZP_EPOCH_TOL:
                hard.append("zero point {:.2f} is {:+.2f} mag from the map's reference: intensity units "
                            "changed".format(night["zp"], night["zp"] - zp_ref))

            depth_ref = map_dict.get("depth_ref")
            if (depth_ref is not None) and (night["depth"] is not None) \
                    and (night["depth"] > float(depth_ref) + DEPTH_GAIN_TOL):
                soft.append("camera reaches {:.2f} mag deeper than at fit time".format(
                    night["depth"] - float(depth_ref)))

    return hard, soft



class SensitivityMap(object):

    def __init__(self, map_dict, source_path=None):
        """ Per-camera block map of the stellar limiting magnitude.

        Arguments:
            map_dict: [dict] Map as written by Utils.FitCameraSensitivityMap. Required keys: nbx,
                nby, LM (nby*nbx values, row-major, image rows top to bottom). Optional: s,
                fit_date, nights, n_trials, X_res, Y_res, mag_lev_ref, vignetting_coeff_ref,
                depth_ref, fingerprint, pointing.

        Keyword arguments:
            source_path: [str] File the map was loaded from, if any.
        """

        self.model = dict(map_dict)
        self.source_path = source_path

        self.nbx = int(map_dict["nbx"])
        self.nby = int(map_dict["nby"])
        self.lm = np.array(map_dict["LM"], dtype=np.float64).reshape(self.nby, self.nbx)

        self.s = float(map_dict.get("s", 0.0))
        self.fit_date = map_dict.get("fit_date")
        self.nights = list(map_dict.get("nights", []))
        self.n_trials = int(map_dict.get("n_trials", 0))
        self.station_id = map_dict.get("stationID")

        self.x_res = map_dict.get("X_res")
        self.y_res = map_dict.get("Y_res")

        self.mag_lev_ref = map_dict.get("mag_lev_ref")
        self.vignetting_coeff_ref = map_dict.get("vignetting_coeff_ref")

        # Reference level: the best block. Every block's sensitivity is a loss relative to it
        self.lm_ref = float(np.max(self.lm))

        # Set once a bin's zero point shift had to be clipped (reported once per map)
        self.shift_clipped = False


    @staticmethod
    def stationMapPath(config):
        """ Path of the station's map file. """

        return os.path.join(os.path.expanduser(config.data_dir),
                            "{:s}_{:s}".format(str(config.stationID), SENSITIVITY_MAP_FILE_SUFFIX))


    @staticmethod
    def nightCopyPath(dir_path):
        """ Path of the map copy archived in a night directory. """

        night_name = os.path.basename(os.path.normpath(dir_path))

        return os.path.join(dir_path, "{:s}_{:s}".format(night_name, SENSITIVITY_MAP_FILE_SUFFIX))


    @classmethod
    def load(cls, config, dir_path=None, platepar=None, night_dir=None):
        """ Load the map to use, or None if there is no usable map.

        The copy archived in the night directory (dir_path) is preferred, since it is the map that
        described the camera on that night. Otherwise the station's map is used, unless it is
        hard-stale for the current configuration or for the night's own photometry (mapStaleness),
        in which case it is refused with a message and the platepar vignetting stays in effect
        until the map is refitted.

        Arguments:
            config: [Config] Station config (data_dir, stationID).

        Keyword arguments:
            dir_path: [str] Night directory whose archived copy is preferred.
            platepar: [Platepar] Current platepar for the staleness check of the station map.
            night_dir: [str] Night directory whose photometry the station map is checked against.

        Return:
            [SensitivityMap] or None.
        """

        candidates = []
        if dir_path:
            candidates.append((cls.nightCopyPath(dir_path), False))
        try:
            candidates.append((cls.stationMapPath(config), True))
        except Exception:
            pass

        for path, check_stale in candidates:

            if not os.path.isfile(path):
                continue

            try:
                with open(path) as f:
                    map_dict = json.load(f)
            except Exception:
                continue

            issues = sensitivityMapQualityIssues(map_dict)
            if issues:
                print("Sensitivity map {:s} is unusable - ignoring it:".format(os.path.basename(path)))
                for msg in issues:
                    print("  " + msg)
                continue

            if check_stale:
                hard, _ = mapStaleness(map_dict, config, platepar=platepar, night_dir=night_dir)
                if hard:
                    print("Sensitivity map {:s} no longer describes this camera - ignoring it, platepar "
                          "vignetting in effect until it is refitted:".format(os.path.basename(path)))
                    for msg in hard:
                        print("  " + msg)
                    continue

            return cls(map_dict, source_path=path)

        return None


    @property
    def tag(self):
        """ Short identity of this map for file names: the fit date plus a hash of the block values,
            so a refit or a different map changes every file computed from it. """

        digest = hashlib.sha1(json.dumps(self.model.get("LM"), sort_keys=True).encode("utf-8")).hexdigest()[:6]

        return "sensmap-{:s}-{:s}".format(str(self.fit_date or "nodate"), digest)


    def summary(self):
        """ One-line description for logs. """

        return "{:d}x{:d} blocks, LM {:.2f}-{:.2f} (best {:.2f}), s {:.2f}, {:d} trials, fit {:s} on {:d} night(s)".format(
            self.nbx, self.nby, float(self.lm.min()), float(self.lm.max()), self.lm_ref, self.s,
            self.n_trials, str(self.fit_date), len(self.nights))


    def _blockCentres(self, x_res, y_res):

        x_res = x_res if x_res is not None else self.x_res
        y_res = y_res if y_res is not None else self.y_res

        if (x_res is None) or (y_res is None):
            raise ValueError("Image size is needed to place the map blocks (map has no X_res/Y_res)")

        cx = (np.arange(self.nbx) + 0.5)*float(x_res)/self.nbx
        cy = (np.arange(self.nby) + 0.5)*float(y_res)/self.nby

        return cx, cy


    def lmAt(self, x, y, x_res=None, y_res=None):
        """ Limiting magnitude at image positions, bilinearly interpolated between block centres and
            held constant beyond the outer centres.

        Arguments:
            x, y: [float or ndarray] Image coordinates (px).

        Keyword arguments:
            x_res, y_res: [int] Image size. Taken from the map if not given.

        Return:
            [float or ndarray] Limiting magnitude.
        """

        cx, cy = self._blockCentres(x_res, y_res)

        x = np.clip(np.asarray(x, dtype=np.float64), cx[0], cx[-1])
        y = np.clip(np.asarray(y, dtype=np.float64), cy[0], cy[-1])

        if self.nbx > 1:
            ix = np.clip(np.searchsorted(cx, x, side="right") - 1, 0, self.nbx - 2)
            fx = (x - cx[ix])/(cx[ix + 1] - cx[ix])
        else:
            ix = np.zeros(np.shape(x), dtype=int)
            fx = np.zeros(np.shape(x))

        if self.nby > 1:
            iy = np.clip(np.searchsorted(cy, y, side="right") - 1, 0, self.nby - 2)
            fy = (y - cy[iy])/(cy[iy + 1] - cy[iy])
        else:
            iy = np.zeros(np.shape(y), dtype=int)
            fy = np.zeros(np.shape(y))

        ix1 = np.minimum(ix + 1, self.nbx - 1)
        iy1 = np.minimum(iy + 1, self.nby - 1)

        lm = ((1 - fx)*(1 - fy)*self.lm[iy, ix] + fx*(1 - fy)*self.lm[iy, ix1]
              + (1 - fx)*fy*self.lm[iy1, ix] + fx*fy*self.lm[iy1, ix1])

        if np.ndim(lm) == 0:
            return float(lm)

        return lm


    def sensitivityRatio(self, x, y, x_res=None, y_res=None):
        """ Flux ratio equivalent of the block's LM loss relative to the best block (<= 1), the
            quantity the flux collection area weights blocks with. """

        return 10**(-0.4*(self.lm_ref - self.lmAt(x, y, x_res=x_res, y_res=y_res)))


    def referenceZeroPoint(self, platepar=None):
        """ The map's reference photometric zero point, moved to the platepar's current vignetting
            coefficient so it compares with zero points fitted under that coefficient.

        Keyword arguments:
            platepar: [Platepar] Current platepar. None = no vignetting adjustment.

        Return:
            [float] or None if the map has no reference zero point.
        """

        if self.mag_lev_ref is None:
            return None

        ref = float(self.mag_lev_ref)

        if (platepar is not None) and (self.vignetting_coeff_ref is not None):

            k_now = platepar.vignetting_coeff if platepar.vignetting_coeff is not None else 0.0
            k_ref = float(self.vignetting_coeff_ref)

            if abs(k_now - k_ref) > 1e-9:
                ref += _areaMeanVignettingShift(platepar.X_res, platepar.Y_res, k_ref, k_now)

        return ref


    def binStellarLM(self, mag_lev_bin, platepar, lm_model):
        """ Stellar limiting magnitude of the best block for a time bin: the map's measured level,
            moved by the zero-point LM model's response to the bin's zero point relative to the
            map's reference. Without a reference zero point the dark-sky level is returned as is.

        Arguments:
            mag_lev_bin: [float] Mean photometric zero point of the bin.
            platepar: [Platepar] Current platepar (for the vignetting adjustment of the reference).
            lm_model: [callable] Zero point -> stellar LM (Utils.Flux.stellarLMModel).

        Return:
            [float] Stellar limiting magnitude of the best block.
        """

        ref = self.referenceZeroPoint(platepar)

        if (ref is None) or (mag_lev_bin is None) or (not np.isfinite(mag_lev_bin)):
            return self.lm_ref

        shift = float(lm_model(mag_lev_bin) - lm_model(ref))

        if abs(shift) > MAX_ZERO_POINT_SHIFT:

            if not self.shift_clipped:
                print("Sensitivity map: zero point {:.2f} is {:+.2f} mag of LM away from the map's reference "
                      "{:.2f} - clipping to {:+.1f}. The camera's intensity units have probably changed "
                      "(firmware, gain or gamma); refit the map.".format(mag_lev_bin, shift, ref,
                      np.sign(shift)*MAX_ZERO_POINT_SHIFT))
                self.shift_clipped = True

            shift = float(np.sign(shift)*MAX_ZERO_POINT_SHIFT)

        return self.lm_ref + shift
