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

# Upper logistic-width fit bound; s pinned here means the fit found no depth
# discrimination at all
S_FIT_MAX = 1.2

# Lunar scattered-light model (Krisciunas & Schaefer 1991, PASP 103, 1033):
# extinction coefficient for the moonlight path (a clean-atmosphere V value -
# NOT the fitted site k, which absorbs vignette and edge-PSF losses) and the
# dark-zenith reference brightness that converts nanoLamberts to this model's
# relative units (zenith natural sky = 1).
MOON_K_EXT = 0.25
MOON_B0_NL = 79.0
MOON_RHO_MIN = 5.0     # deg - clamp the star-moon separation; inside this the
                       # scattering function diverges and the sky is saturated
                       # glare anyway (detection there is hopeless regardless)


def fitQualityIssues(model_dict):
    """ Diagnose degenerate-fit symptoms in a fitted (or loaded) dome model.

    A degenerate fit is one whose optimizer solution sits on a fit bound: the data did
    not constrain the parameter, so the model reflects the bound, not the sky. Adopting
    such a model corrupts every downstream verdict (an LM0 at the lower bound predicts
    almost no stars, so cloudy nights score as clear).

    Arguments:
        model_dict: [dict] Fitted model dict (or LightDomeModel.model of a loaded one).

    Return:
        issues: [list of str] One message per symptom; empty for a healthy fit.
    """

    issues = []

    lim_mag = float(model_dict.get("catalog_lim_mag", DOME_CATALOG_LIM_MAG))
    lm0_max = lim_mag + 1.0

    for cam, lm0 in zip(model_dict.get("cams", []), model_dict.get("LM0", [])):

        if float(lm0) <= LM0_FIT_MIN + FIT_BOUND_TOL:
            issues.append("LM0[{:s}]={:.2f} pinned at the lower fit bound {:.1f} - no "
                "depth signal in that camera's training trials (cloudy/foggy/obstructed "
                "fit nights)".format(str(cam), float(lm0), LM0_FIT_MIN))

        elif float(lm0) >= lm0_max - FIT_BOUND_TOL:
            issues.append("LM0[{:s}]={:.2f} pinned at the upper fit bound {:.1f} - "
                "saturated against the catalog depth {:.1f}".format(
                str(cam), float(lm0), lm0_max, lim_mag))

    if float(model_dict.get("s", 0.0)) >= S_FIT_MAX - FIT_BOUND_TOL:
        issues.append("s={:.2f} pinned at the upper fit bound {:.1f} - the logistic "
            "has no depth discrimination".format(float(model_dict["s"]), S_FIT_MAX))

    return issues


def fitQualityWarnings(model_dict):
    """ Non-blocking degeneracy diagnostics: symptoms that corrupt the model's
    INTERPRETATION (renders, absolute brightness, extrapolation) but not its
    detection scoring, because a compensating parameter keeps the fitted
    combination calibrated where the trials are.

    The canonical case (observed on USC0K, a light-polluted site): a broad LP
    bowl is nearly flat across the FOV, collinear with a constant LM0 offset -
    the optimizer ran q0 to its upper bound and inflated every LM0 to
    compensate (LM0 ~8.3 at a polluted site is unphysical; the SPLIT between
    bowl and LM0 is arbitrary even when the total sky brightness is real).
    Scoring stays calibrated because only the fitted combination enters it;
    the render and any extrapolation inherit the arbitrary split.

    These must never block adoption or evict a model (that would push a
    working station to the scalar path over a cosmetic defect) - they travel
    in the model file for fleet visibility and print at fit time.

    Return:
        warnings: [list of str] One message per symptom; empty when clean.
    """

    warnings = []

    # Bounds mirror the optimizer bounds in Utils.FitLightDome (base_bounds /
    # order_bounds); q0 and log10(A) are log10 brightness amplitudes
    q0 = float(model_dict.get("q0", 0.0))
    if q0 >= 3.0 - FIT_BOUND_TOL:
        warnings.append("q0={:.2f} pinned at the upper fit bound 3.0 - the LP "
            "bowl ({:.0f}x zenith-natural, {:.1f} mag) is degenerate with a "
            "constant LM0 offset (broad bowl, h0={:.0f} deg); brightness "
            "renders and extrapolation are unreliable".format(
            q0, 10.0**q0, 1.25*np.log10(1.0 + 10.0**q0),
            float(model_dict.get("h0", 0.0))))
    # A q0 at the LOWER bound is the optimizer correctly turning the bowl off
    # at a dark site - expected, not flagged. The bowl-shape bounds are only
    # meaningful when the bowl itself carries brightness.
    h0 = float(model_dict.get("h0", 20.0))
    if (q0 > -2.0 + FIT_BOUND_TOL) and ((h0 <= 5.0 + 0.5) or (h0 >= 60.0 - 0.5)):
        warnings.append("h0={:.1f} deg at a fit bound - the bowl altitude "
            "profile is unconstrained".format(h0))

    for h in model_dict.get("harmonics", []):
        log_a = np.log10(max(float(h.get("A", 0.0)), 1e-12))
        if log_a >= 3.5 - FIT_BOUND_TOL:
            warnings.append("harmonic order {:d} amplitude at the fit bound "
                "(log10 A={:.2f}) - degenerate azimuthal term".format(
                int(h.get("order", 0)), log_a))
        hh = float(h.get("h", 20.0))
        if (hh <= 3.0 + 0.5) or (hh >= 60.0 - 0.5):
            warnings.append("harmonic order {:d} alt scale {:.1f} deg at a "
                "fit bound".format(int(h.get("order", 0)), hh))

    return warnings


def blockingQualityIssues(model_dict):
    """ The subset of quality issues that make a model UNUSABLE for scoring, for
    every camera in it - not just the flagged one. A joint site fit with any camera
    pinned at the LOWER bound (or s at its ceiling) was optimized against garbage
    data: the shared harmonics and logistic width are co-fit, so no station's slice
    of that model is trustworthy. (Observed: AUC0A6 scored a night against a fit
    whose siblings were pinned at 4.00 - its own LM0 of 4.31 passed a per-station
    check while the model predicted a ridiculous star count.) Upper-bound saturation
    is NOT blocking: the ratio normalization compensates until the refit lands.
    """

    return [i for i in fitQualityIssues(model_dict)
            if ("lower fit bound" in i) or i.startswith("s=")]


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

            # Refuse a degenerate model OUTRIGHT - for every station, not just the
            # camera whose parameter is pinned: the joint fit's shared terms were
            # optimized against that camera's garbage data, so no slice of the model
            # is trustworthy. The scalar fallback is strictly better than several-fold
            # inflated ratios. The file stays in place for ensureLightDomeModel to
            # diagnose and refit the site.
            blocking = blockingQualityIssues(model_dict)
            if blocking:
                print("Light-dome model {:s} is degenerate - ignoring it, scalar "
                      "fallback in effect:".format(name))
                for msg in blocking:
                    print("  " + msg)
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


    def moonBrightnessRatio(self, az, alt, moon_az, moon_alt, moon_phase):
        """ Scattered-moonlight brightness as a fraction of the model's site
        brightness at each sky position (Krisciunas & Schaefer 1991), BEFORE
        the per-site gain.

        The gain exists because the model's B_site carries the unidentifiable
        LM0/bowl split: its absolute scale is arbitrary without an anchor, so
        B_moon/B_site must be calibrated per site from archived moonlit-clear
        frames (see Utils.Flux.fitMoonGain). Measured with gain 1 the penalty
        over-corrected by 40-190% on three dark-site nights whose uncorrected
        expectations were already right - the fitted gains were ~0.

        Arguments:
            az, alt: [ndarray/float] Sky positions (deg).
            moon_az, moon_alt: [float] Moon horizontal coordinates (deg).
            moon_phase: [float] Illuminated fraction (percent, 0-100).

        Return:
            q: [ndarray] B_moon/B_site in the model's relative units (>= 0).
        """

        az = np.asarray(az, dtype=np.float64)
        alt = np.asarray(alt, dtype=np.float64)

        if (moon_alt is None) or (moon_alt <= 0.0) or (moon_phase is None):
            return np.zeros_like(az)

        # Phase angle from the illuminated fraction
        frac = np.clip(float(moon_phase)/100.0, 0.0, 1.0)
        alpha = np.degrees(np.arccos(np.clip(2.0*frac - 1.0, -1.0, 1.0)))

        # Moon magnitude and normalized illuminance
        m_moon = -12.73 + 0.026*abs(alpha) + 4e-9*alpha**4
        i_star = 10.0**(-0.4*(m_moon + 16.57))

        # Star-moon angular separation on the sky
        a1, a2 = np.radians(alt), np.radians(float(moon_alt))
        daz = np.radians(az - float(moon_az))
        cos_rho = np.sin(a1)*np.sin(a2) + np.cos(a1)*np.cos(a2)*np.cos(daz)
        rho = np.degrees(np.arccos(np.clip(cos_rho, -1.0, 1.0)))
        rho = np.maximum(rho, MOON_RHO_MIN)

        # Scattering function (Rayleigh + Mie), optical path lengths
        f_rho = 10.0**5.36*(1.06 + np.cos(np.radians(rho))**2)             + 10.0**(6.15 - rho/40.0)
        x_star = (1.0 - 0.96*np.sin(np.radians(90.0 - np.clip(alt, 5.0, 90.0)))**2)**-0.5
        x_moon = (1.0 - 0.96*np.sin(np.radians(90.0 - max(float(moon_alt), 5.0)))**2)**-0.5

        b_moon_nl = f_rho*i_star*10.0**(-0.4*MOON_K_EXT*x_moon)             *(1.0 - 10.0**(-0.4*MOON_K_EXT*x_star))

        b_site = np.asarray(self.skyBrightness(az, alt), dtype=np.float64)
        return (b_moon_nl/MOON_B0_NL)/np.maximum(b_site, 1e-6)


    def moonPenalty(self, az, alt, moon_az, moon_alt, moon_phase, gain=1.0):
        """ Limiting-magnitude penalty from scattered moonlight at the given sky
        positions: 1.25*log10(1 + gain*B_moon/B_site). gain scales the
        brightness RATIO (see moonBrightnessRatio) and is calibrated per site;
        gain 0 disables the correction (warmup, and the measured value at
        current dark-site model splits).

        Return:
            penalty: [ndarray/float] Magnitudes of LM lost to moonlight (>= 0).
        """

        q = self.moonBrightnessRatio(az, alt, moon_az, moon_alt, moon_phase)
        return 1.25*np.log10(1.0 + float(gain)*q)


    def detectionProbability(self, mag, az, alt, station_id=None, moon=None,
            moon_gain=1.0):
        """ Probability that a star of the given catalog magnitude is detected and matched
            at the given sky position, under clear skies.

        Arguments:
            mag: [ndarray/float] Catalog magnitude(s).
            az, alt: [ndarray/float] Horizontal coordinates (deg).

        Keyword arguments:
            station_id: [str] Station whose LM0 to use.
            moon: [tuple] (moon_az, moon_alt, moon_phase) of the frame - applies the
                scattered-moonlight LM penalty (moonPenalty). None = dark sky.

        Return:
            p: [ndarray/float] Detection probability in [0, 1].
        """

        lm = self.limitingMagnitude(az, alt, station_id=station_id)

        if moon is not None:
            lm = lm - self.moonPenalty(az, alt, *moon, gain=moon_gain)

        return 1.0/(1.0 + np.exp(-(lm - mag)/self.s))
