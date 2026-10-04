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

Two reference values make the map usable on nights other than the ones it was fitted on:
    - lm_ref: the best block's LM. Per-block losses are expressed relative to it, so the bin's
      stellar LM stands for the best block and every block is a loss (ratio <= 1), the same
      convention as the vignetting path.
    - mag_lev_ref: the photometric zero point of the frames the map was fitted on. The night's
      zero point measures transparency, so the bin LM is lm_ref moved by the zero-point LM
      model's response to (mag_lev_bin - mag_lev_ref). The zero point also moves when the
      platepar's vignetting coefficient is changed, so the stored coefficient is kept and the
      reference is shifted to the current one before comparing.
"""

from __future__ import absolute_import, division, print_function

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



class SensitivityMap(object):

    def __init__(self, map_dict):
        """ Per-camera block map of the stellar limiting magnitude.

        Arguments:
            map_dict: [dict] Map as written by Utils.FitCameraSensitivityMap. Required keys: nbx,
                nby, LM (nby*nbx values, row-major, image rows top to bottom). Optional: s,
                fit_date, nights, n_trials, X_res, Y_res, mag_lev_ref, vignetting_coeff_ref.
        """

        self.model = dict(map_dict)

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


    @classmethod
    def load(cls, config):
        """ Load the station's map, or None if there is no usable map file.

        Searched in config.data_dir: <stationID>_sensitivity_map.json.

        Arguments:
            config: [Config] Station config (data_dir, stationID).

        Return:
            [SensitivityMap] or None.
        """

        try:
            data_dir = os.path.expanduser(config.data_dir)
        except Exception:
            return None

        path = os.path.join(data_dir, "{:s}_{:s}".format(str(config.stationID), SENSITIVITY_MAP_FILE_SUFFIX))

        if not os.path.isfile(path):
            return None

        try:
            with open(path) as f:
                map_dict = json.load(f)
        except Exception:
            return None

        issues = sensitivityMapQualityIssues(map_dict)
        if issues:
            print("Sensitivity map {:s} is unusable - ignoring it, platepar vignetting in effect:".format(
                os.path.basename(path)))
            for msg in issues:
                print("  " + msg)
            return None

        return cls(map_dict)


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
