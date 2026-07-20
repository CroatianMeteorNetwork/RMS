""" Site light-dome limiting magnitude model.

Evaluates the stellar limiting magnitude of a station anywhere on the sky from a fitted
site sky-brightness model. The sky brightness is composed in BRIGHTNESS space (sources
add in flux, not magnitudes):

    B(az, alt) = vanRhijn(alt) + amplitude*(LP_bowl(alt) + sum_k harmonic_k(az, alt))

where vanRhijn is the natural airglow slant-path brightening (no free parameters),
LP_bowl is the isotropic light-pollution term, and the harmonics are an azimuthal cosine
series (one altitude profile per order) - the natural basis for an observer EMBEDDED in
the glow field, where directional structure is a broad asymmetry gradient rather than
discrete distant-city domes. The limiting magnitude then follows the
background-limited SNR relation:

    LM(az, alt, station) = LM0_station - k*(airmass - 1) - 1.25*log10(B(az, alt))

LM0_station is the per-camera dark-sky potential (throughput, focus, threshold), fitted
jointly with the shared site terms from matched-star hit/miss trials (see
Utils.FitLightDome). The model file is per SITE: one JSON shared by all co-located
stations, each keyed by its stationID in the LM0 list.

The `amplitude` factor scales the light-pollution terms only (not the airglow) and is
the single nightly tunable: aerosols (e.g. monsoon haze) scatter more city light into
the beam, brightening the whole LP field together.

Detection is modeled as a soft logistic rolloff rather than a hard cutoff:

    P(detected | m) = 1 / (1 + exp(-(LM - m)/s))

so the expected matched-star count of a frame is the sum of per-star probabilities.
"""

from __future__ import absolute_import, division, print_function

import json
import os

import numpy as np


# Model file name: <stationID>_light_dome.json is searched first, then the site-generic
# light_dome.json, both in the station's data directory
LIGHT_DOME_FILE_SUFFIX = "light_dome.json"

# Default catalog depth for dome-model trials and expected counts. This is a STARTING
# depth, not a ceiling: dark stations reach LM 6+ and the fit adapts the depth so the
# logistic tail is fully sampled (see domeCatalogLimMag and Utils.FitLightDome). The
# depth actually used at fit time is stored in the model file (catalog_lim_mag) and
# scoring must read it from there (catalogLimMag) so expected counts stay calibrated.
DOME_CATALOG_LIM_MAG = 6.0

# Hard cap on the adaptive depth: bounded by the shipped catalog files and by the
# extractor's practical reach; nothing physical lives beyond LM + 4s anyway
DOME_CATALOG_MAX_LIM_MAG = 9.0

# The logistic tail is sampled to LM + this many s (P ~ 2% there)
DOME_CATALOG_TAIL_SIGMA = 4.0

# Lower LM0 fit bound (Utils.FitLightDome builds the optimizer bounds from this). An LM0
# pinned within FIT_BOUND_TOL of it means the fit nights carried no depth signal for that
# camera (overcast/fog/obstruction): the model predicts almost no stars, ratios inflate
# several-fold, and cloudy nights read as clear - so load() refuses such a model for the
# affected station and the scalar fallback takes over until a healthy refit lands
LM0_FIT_MIN = 4.0
FIT_BOUND_TOL = 0.1


def domeCatalogLimMag(lm0_list, s):
    """ Catalog depth required to fully sample the detection rolloff of a fitted model.

    Arguments:
        lm0_list: [list of float] Per-camera LM0 values.
        s: [float] Logistic rolloff width.

    Return:
        lim_mag: [float] Required catalog depth, clamped to
            [DOME_CATALOG_LIM_MAG, DOME_CATALOG_MAX_LIM_MAG].
    """

    if not lm0_list:
        return DOME_CATALOG_LIM_MAG

    wanted = max(lm0_list) + DOME_CATALOG_TAIL_SIGMA*float(s)

    return float(np.clip(wanted, DOME_CATALOG_LIM_MAG, DOME_CATALOG_MAX_LIM_MAG))

# (R/(R+h))^2 for an airglow emission layer at ~100 km
_VAN_RHIJN_C = 0.97


class LightDomeModel(object):
    def __init__(self, model_dict):
        """ Site light-dome model, evaluated from a fitted parameter dictionary.

        Arguments:
            model_dict: [dict] Fitted model, as written by Utils.FitLightDome. Fields:
                cams [list of stationID], LM0 [list, per cam], k, s, q0, h0,
                harmonics [list of {order, A, phi, h}].
        """

        self.model = model_dict

        self.cams = [str(c) for c in model_dict["cams"]]
        self.lm0_map = dict(zip(self.cams, [float(v) for v in model_dict["LM0"]]))
        self.lm0_default = float(np.mean(model_dict["LM0"]))

        self.k = float(model_dict["k"])
        self.s = float(model_dict["s"])
        self.q0 = float(model_dict.get("q0", -10.0))
        self.h0 = float(model_dict.get("h0", 20.0))

        # Azimuthal cosine series, one altitude profile per order
        self.harmonics = model_dict.get("harmonics", [])

        # Nightly aerosol scale on the light-pollution terms (1.0 = fit epoch)
        self.amplitude = 1.0


    def catalogLimMag(self):
        """ The catalog depth this model was FIT with - scoring must use the same depth
            so expected counts stay calibrated to the fit. Legacy models without the
            stored field were fit at the old fixed depth. """

        return float(self.model.get("catalog_lim_mag") or DOME_CATALOG_LIM_MAG)

    @classmethod
    def load(cls, config):
        """ Load the site model for a station, or None if no model file is present.

        Searched in config.data_dir: <stationID>_light_dome.json, then light_dome.json.

        Arguments:
            config: [Config] Station config (data_dir, stationID).

        Return:
            model: [LightDomeModel] or None.
        """

        try:
            data_dir = os.path.expanduser(config.data_dir)
        except Exception:
            return None

        candidates = [
            "{:s}_{:s}".format(str(config.stationID), LIGHT_DOME_FILE_SUFFIX),
            LIGHT_DOME_FILE_SUFFIX,
        ]

        for name in candidates:
            path = os.path.join(data_dir, name)

            if not os.path.isfile(path):
                continue

            try:
                with open(path) as f:
                    model_dict = json.load(f)
            except Exception:
                continue

            # Only the harmonic basis is supported - a file from the retired dome basis is
            # left in place for ensureLightDomeModel to detect and refit
            if model_dict.get("model") != "vanrhijn_harmonics":
                continue

            # Refuse a model whose LM0 for THIS station is pinned at the lower fit bound
            # (degenerate all-cloudy fit): the scalar fallback is strictly better than
            # several-fold inflated ratios. The file stays in place for
            # ensureLightDomeModel to diagnose and refit the site.
            cams = [str(c) for c in model_dict.get("cams", [])]
            lm0 = model_dict.get("LM0", [])
            station = str(config.stationID)
            if station in cams and lm0:
                lm0_here = float(lm0[cams.index(station)])
            else:
                lm0_here = float(np.mean(lm0)) if lm0 else LM0_FIT_MIN
            if lm0_here <= LM0_FIT_MIN + FIT_BOUND_TOL:
                print("Light-dome model {:s} has LM0={:.2f} pinned at the lower fit "
                      "bound (degenerate fit) - ignoring it, scalar fallback in "
                      "effect".format(name, lm0_here))
                continue

            return cls(model_dict)

        return None


    def skyBrightness(self, az, alt):
        """ Relative sky brightness (zenith natural sky = 1) at the given sky position.

        Arguments:
            az, alt: [ndarray/float] Horizontal coordinates (deg).

        Return:
            B: [ndarray/float] Sky brightness relative to the natural zenith sky.
        """

        alt_c = np.clip(alt, 5.0, 90.0)

        # Natural airglow slant-path brightening (van Rhijn)
        z2 = np.sin(np.radians(90.0 - alt_c))**2
        B = 1.0/np.sqrt(1.0 - _VAN_RHIJN_C*z2)

        # Light pollution: isotropic bowl term
        lp = (10.0**self.q0)*np.exp(-alt_c/self.h0)

        # Azimuthal cosine series, one altitude profile per order.
        # A_k*cos(k*(az - phi_k)) can be negative on the dark side; the total LP field is
        # clamped non-negative below.
        for h in self.harmonics:
            order = int(h["order"])
            lp = lp + h["A"]*np.cos(np.radians(order*(az - h["phi"])))*np.exp(-alt_c/h["h"])

        return B + self.amplitude*np.maximum(lp, 0.0)


    def limitingMagnitude(self, az, alt, station_id=None):
        """ Stellar limiting magnitude at the given sky position for the given station.

        Arguments:
            az, alt: [ndarray/float] Horizontal coordinates (deg).

        Keyword arguments:
            station_id: [str] Station whose LM0 to use. Site mean if None or unknown.

        Return:
            lm: [ndarray/float] Limiting magnitude (50% detection).
        """

        lm0 = self.lm0_map.get(str(station_id), self.lm0_default)

        alt_c = np.clip(alt, 5.0, 90.0)
        airmass = 1.0/np.sin(np.radians(alt_c))

        return lm0 - self.k*(airmass - 1.0) - 1.25*np.log10(self.skyBrightness(az, alt))


    def detectionProbability(self, mag, az, alt, station_id=None):
        """ Probability that a star of the given catalog magnitude is detected and matched
            at the given sky position, under clear skies.

        Arguments:
            mag: [ndarray/float] Catalog magnitude(s).
            az, alt: [ndarray/float] Horizontal coordinates (deg).

        Keyword arguments:
            station_id: [str] Station whose LM0 to use.

        Return:
            p: [ndarray/float] Detection probability in [0, 1].
        """

        lm = self.limitingMagnitude(az, alt, station_id=station_id)

        return 1.0/(1.0 + np.exp(-(lm - mag)/self.s))
