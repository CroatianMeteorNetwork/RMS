""" Site light-dome limiting magnitude model.

Evaluates the stellar limiting magnitude of a station anywhere on the sky from a fitted
site sky-brightness model. The sky brightness is composed in BRIGHTNESS space (sources
add in flux, not magnitudes):

    B(az, alt) = vanRhijn(alt) + amplitude*(LP_bowl(alt) + sum_i dome_i(az, alt))

where vanRhijn is the natural airglow slant-path brightening (no free parameters),
LP_bowl is the isotropic light-pollution term, and each dome_i is a localized city glow
(von Mises in azimuth, exponential in altitude). The limiting magnitude then follows the
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

# Catalog depth used when computing expected counts with the dome model. Fixed and
# deeper than any plausible LM so the logistic tail is always fully sampled.
DOME_CATALOG_LIM_MAG = 6.0

# (R/(R+h))^2 for an airglow emission layer at ~100 km
_VAN_RHIJN_C = 0.97


class LightDomeModel(object):
    def __init__(self, model_dict):
        """ Site light-dome model, evaluated from a fitted parameter dictionary.

        Arguments:
            model_dict: [dict] Fitted model, as written by Utils.FitLightDome. Fields:
                cams [list of stationID], LM0 [list, per cam], k, s, q0, h0,
                domes [list of {az, B, kappa, h}].
        """

        self.model = model_dict

        self.cams = [str(c) for c in model_dict["cams"]]
        self.lm0_map = dict(zip(self.cams, [float(v) for v in model_dict["LM0"]]))
        self.lm0_default = float(np.mean(model_dict["LM0"]))

        self.k = float(model_dict["k"])
        self.s = float(model_dict["s"])
        self.q0 = float(model_dict["q0"])
        self.h0 = float(model_dict["h0"])
        self.domes = model_dict.get("domes", [])

        # Nightly aerosol scale on the light-pollution terms (1.0 = fit epoch)
        self.amplitude = 1.0


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

            if os.path.isfile(path):
                try:
                    with open(path) as f:
                        return cls(json.load(f))

                except Exception:
                    return None

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

        # Light pollution: isotropic bowl + localized domes, scaled by the aerosol amplitude
        lp = (10.0**self.q0)*np.exp(-alt_c/self.h0)

        for d in self.domes:
            lp = lp + d["B"]*np.exp(d["kappa"]*(np.cos(np.radians(az - d["az"])) - 1.0)) \
                *np.exp(-alt_c/d["h"])

        return B + self.amplitude*lp


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
