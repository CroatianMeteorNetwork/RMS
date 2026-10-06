""" Rebuild the green channel of raw-saturated highlights in daytime colour frames.

On OpenIPC science cameras driven by podcontrol, the WB gains are scaled down (green gain
s < 1) to recover red/blue highlights. A pixel whose RAW green saturated then comes out at
green = f(s) * full scale while red and blue keep going, so clipped areas turn magenta. The
camera cannot fix it (per-channel clip in fixed ISP hardware, no 3D colour LUT on the
GK7205V200), so the saved day frames are fixed here, in linear light.

The clipped pixels are found by their colour, not by a clip level: a bright pixel whose green
is far below BOTH red and blue is magenta, a hue that sunlit scenes do not produce (sky is
blue, foliage green, clouds and the sun neutral, the reds have low blue). The clip level
cannot be predicted from the WB gains (green clips 17-36 codes below what the gains suggest,
and the gap grows as the gain is scaled down), and the plateau smears across a bright
gradient in the 4:2:0 video, so a level-based test leaves a pink fringe or misses the frame.

    ratio = G / min(R, B)     (linear light)
    weight = 0 above RATIO_HI (natural colour), 1 below RATIO_LO, smooth in between
    G += weight * (min(R, B) - G)

Green is only ever raised, to the lower of the other two channels (a neutral grey); where red and
blue are saturated too the pixel goes to white.

Frames are the camera's 8-bit full-range pure gamma 0.5 (linear = (v/255)^2). Frames taken
with normal WB gains (and night frames) have no pixels below RATIO_HI that matter and are
returned unchanged.
"""

from __future__ import print_function, division, absolute_import

import numpy as np


CLIP_CODE = 253          # 8-bit code at/above which red or blue counts as saturated
PLATEAU_MAX = 249        # a green ceiling at/above this is ordinary full-scale clipping: the frame is left alone
MIN_CODE = 150           # lower of red/blue (8-bit) below which a pixel is too dark to judge
RATIO_HI = 0.85          # G/min(R,B) (linear) above this is natural colour: untouched
RATIO_LO = 0.60          # ... below this, fully rebuilt; linear ramp in between
MIN_PIXELS = 200         # candidate pixels needed before the frame is touched

# G8 below this limit (indexed by min(R8, B8)) is a candidate: bright enough, and G/min < RATIO_HI in
# linear light. A lookup table, so the full-frame test needs no float arithmetic.
_CAND_LIMIT = np.where(np.arange(256) >= MIN_CODE, np.ceil(np.sqrt(RATIO_HI)*np.arange(256)), 0).astype(np.uint8)


def rebuildGreen(bgr, ceiling=None):
    """ Rebuild green where a bright pixel is magenta (green far below both red and blue).

    Arguments:
        bgr: [ndarray] HxWx3 uint8 frame, OpenCV channel order (B, G, R).

    Keyword arguments:
        ceiling: [float] 8-bit code at which green clips, from the frame's exposure record
            (RMS.FrameMetadata.greenCeiling), or None if unknown. Only used to skip frames
            taken at normal WB gains (ceiling >= PLATEAU_MAX), where magenta is not a clip artefact.

    Return:
        (frame, info): [ndarray] the frame (the input object when unchanged), [dict or None]
            green level of the saturated areas (0 if none), fraction of pixels changed, and
            the fraction made white; None when unchanged.
    """
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        return bgr, None
    if ceiling is not None and ceiling >= PLATEAU_MAX:
        return bgr, None

    B8, G8, R8 = bgr[..., 0], bgr[..., 1], bgr[..., 2]

    # Cheap test in 8 bit first: G/min < RATIO_HI in linear light is G8 < sqrt(RATIO_HI)*min8
    mn8 = np.minimum(R8, B8)
    cand = G8 < _CAND_LIMIT[mn8]
    if np.count_nonzero(cand) < MIN_PIXELS:
        return bgr, None

    rows, cols = np.flatnonzero(cand.any(axis=1)), np.flatnonzero(cand.any(axis=0))
    y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1      # work on the bounding box only

    sub = bgr[y0:y1, x0:x1]
    lin = (sub.astype(np.float32)/255.0)**2
    B, G, R = lin[..., 0], lin[..., 1], lin[..., 2]
    mn = np.minimum(R, B)
    ratio = G/np.maximum(mn, 1e-6)
    w = np.clip((RATIO_HI - ratio)/(RATIO_HI - RATIO_LO), 0.0, 1.0)
    w[mn8[y0:y1, x0:x1] < MIN_CODE] = 0.0
    if not (w > 0).any():
        return bgr, None

    g_new = G + w*(mn - G)

    out = bgr.copy()
    out[y0:y1, x0:x1, 1] = np.round(255.0*np.sqrt(np.clip(g_new, 0, 1))).astype(np.uint8)
    all_sat = (w >= 1.0) & (sub[..., 0] >= CLIP_CODE) & (sub[..., 2] >= CLIP_CODE)
    out[y0:y1, x0:x1][all_sat] = 255

    full = w >= 1.0
    level = int(np.median(sub[..., 1][full])) if full.any() else 0
    n = float(bgr.shape[0]*bgr.shape[1])
    return out, {"plateau": level, "clipped": np.count_nonzero(w)/n, "ratios": "chroma",
                 "ceiling": "chroma", "k_r": 1.0, "k_b": 1.0, "all_saturated": np.count_nonzero(all_sat)/n}
