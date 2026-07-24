""" Forced aperture photometry at known star positions.

The shared primitive behind the star-evidence channels that do not depend on the
star extractor: the scoring product's forced-photometry match class (saturated
and extractor-culled stars) and the 5 s stills sampler. Unlike blind extraction
there is no candidate search and no PSF shape gate - the only question asked is
"how much flux sits in this small aperture above its local background", which
works on saturated stars, on single video frames, and through JPEG compression
(validated against FF products on LP and dark-site cameras: detection agreement
within a point of the extractor's own curve at 5 sigma).

Two floors are MANDATORY whenever detections are derived from these
measurements; both exist because a plain SNR threshold fails in the field:

- The frame noise floor is the MAD of fluxes at random positions. On a smooth
  overcast frame that statistic collapses to ~0 ADU, and any hot pixel or
  compression bump then passes an SNR cut with astronomical significance
  (observed: 55 false 5-sigma "detections" at SNR ~13000 on one overcast
  still, rendering false clear islands in solid cloud). Frame noise must be
  floored at a reference scale that cannot collapse - the night's median.
- A detection must also carry a physical fraction of the star's own expected
  flux (its clear-sky median). A 5-sigma blip at 1% of the star's normal
  brightness is not the star; this is the forced-photometry analog of the
  chance-match floor in the dome model.
"""

from __future__ import absolute_import, division, print_function

import numpy as np


# Aperture geometry (px). The 5x5 aperture tolerates ~2 px of pointing error;
# the background ring at +/-7 px clears the PSF of everything but the most
# bloomed stars while staying inside the same cloud/glare structure
APERTURE_HALF = 2
RING_OFFSETS = ((-7, -7), (-7, 0), (-7, 7), (0, -7), (0, 7), (7, -7), (7, 0),
                (7, 7))
EDGE_MARGIN = 8          # positions closer than this to the frame edge are not measured

N_NOISE_PATCHES = 400    # random patches per frame for the noise MAD

# Detection floors (see module docstring)
SNR_MIN = 5.0            # detection threshold in floored-noise units
CLEAR_FLUX_FRAC = 0.25   # minimum fraction of the star's clear-sky median flux
FLUX_ABS_MIN = 1.0       # ADU - guards the fraction test when the clear median
                         # itself is tiny


def measurePatchFluxes(img, x, y, rng=None):
    """ Aperture flux above local background at known positions, plus the frame's
        noise scale.

    Arguments:
        img: [ndarray HxW] Image (float or int; converted to float32).
        x, y: [ndarray] Pixel positions to measure.

    Keyword arguments:
        rng: [np.random.RandomState] Source for the noise-patch positions.
            A fixed default keeps repeated runs reproducible.

    Return:
        flux: [ndarray] Aperture-minus-ring flux per position; NaN where the
            position is closer than EDGE_MARGIN to the frame edge.
        frame_noise: [float] MAD-based noise scale of the same statistic at
            random positions. May be ~0 on smooth frames - callers deriving
            detections MUST floor it (see detectStars / nightNoiseFloor).
    """

    img = np.asarray(img, dtype=np.float32)
    h, w = img.shape

    x = np.asarray(x)
    y = np.asarray(y)
    xi = np.round(x).astype(np.int64)
    yi = np.round(y).astype(np.int64)

    ok = (np.isfinite(x) & np.isfinite(y)
          & (xi >= EDGE_MARGIN) & (xi < w - EDGE_MARGIN)
          & (yi >= EDGE_MARGIN) & (yi < h - EDGE_MARGIN))

    flux = np.full(len(xi), np.nan, dtype=np.float32)
    if np.any(ok):
        flux[ok] = _apertureFlux(img, xi[ok], yi[ok])

    if rng is None:
        rng = np.random.RandomState(7)
    rx = rng.randint(EDGE_MARGIN, w - EDGE_MARGIN, N_NOISE_PATCHES)
    ry = rng.randint(EDGE_MARGIN, h - EDGE_MARGIN, N_NOISE_PATCHES)
    rf = _apertureFlux(img, rx, ry)
    frame_noise = float(1.4826*np.median(np.abs(rf - np.median(rf))))

    return flux, frame_noise


def _apertureFlux(img, xi, yi):
    """ Vectorized aperture sum minus ring-median background at integer positions
        (all positions must be at least EDGE_MARGIN from the edge). """

    a = APERTURE_HALF
    dy, dx = np.mgrid[-a:a + 1, -a:a + 1]

    patches = img[(yi[:, None, None] + dy), (xi[:, None, None] + dx)]
    ring = np.stack([img[yi + oy, xi + ox] for oy, ox in RING_OFFSETS])
    bg = np.median(ring, axis=0)

    return (patches - bg[:, None, None]).sum(axis=(1, 2))


def nightNoiseFloor(frame_noises):
    """ The reference noise scale for a night: the median of the per-frame noise
        values measured so far, EXCLUDING collapsed ones - on a mostly-overcast
        night the collapsed frames would otherwise drag the floor itself to zero,
        defeating its purpose. Real 8-bit frames never measure below ~1 ADU;
        anything under half that is a collapse, not a quiet sky. """

    fn = np.asarray(frame_noises, dtype=np.float64)
    fn = fn[np.isfinite(fn) & (fn > 0.5)]

    return float(np.median(fn)) if len(fn) else 1.0


def detectStars(flux, frame_noise, noise_floor, clear_flux_median=None,
        snr_min=SNR_MIN, clear_flux_frac=CLEAR_FLUX_FRAC):
    """ Turn measured fluxes into detection bits, applying BOTH mandatory floors.

    Arguments:
        flux: [ndarray] Fluxes from measurePatchFluxes (NaN = not measured).
        frame_noise: [float] This frame's noise scale.
        noise_floor: [float] The night's reference noise (nightNoiseFloor) -
            the frame noise is floored at this so a smooth overcast frame
            cannot manufacture infinite SNR.

    Keyword arguments:
        clear_flux_median: [ndarray or None] Per-star clear-sky median flux.
            Where finite, a detection must also reach clear_flux_frac of it;
            where None/NaN (no calibration yet), only the SNR test applies -
            callers should treat such detections as provisional.
        snr_min, clear_flux_frac: [float] Thresholds.

    Return:
        detected: [ndarray bool]
        snr: [ndarray] Flux over floored noise (NaN where not measured).
    """

    nz = max(float(frame_noise), float(noise_floor), 1e-3)

    with np.errstate(invalid="ignore"):
        snr = flux/nz
        detected = np.isfinite(snr) & (snr >= snr_min)

        if clear_flux_median is not None:
            cfm = np.asarray(clear_flux_median, dtype=np.float64)
            has_cal = np.isfinite(cfm)
            detected &= ~has_cal | (flux >= clear_flux_frac*np.maximum(
                cfm, FLUX_ABS_MIN))

    return detected, snr
