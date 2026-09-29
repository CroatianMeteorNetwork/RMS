""" Rebuild the green channel of raw-saturated highlights in daytime colour frames.

On OpenIPC science cameras driven by podcontrol, the WB gains are scaled down (green gain
s < 1) to recover red/blue highlights. A pixel whose RAW green saturated then comes out at
green = s * full scale while red and blue keep going, so clipped areas turn magenta. The
camera cannot fix it (per-channel clip in fixed ISP hardware, no 3D colour LUT on the
GK7205V200), so the saved day frames are fixed here, in linear light:

    where green sits on its clip plateau: G = mean(kR * R, kB * B) over the channels that
    are not clipped themselves, never below the plateau; kR = G/R, kB = G/B are the scene's
    colour ratios from bright unclipped pixels (neutral 1, 1 when under 0.5% of the frame is
    unclipped); all three channels saturated -> white.

Frames are the camera's 8-bit full-range pure gamma 0.5 (linear = (v/255)^2). Tested against
a 4-16x darker frame on a CV300 and a GK7205V200 (2026-09-28), 21-99.8% of the frame clipped:
green error 1.8-6.5%, colour error 1.7-6.5% (as captured 12-43% / 14-81%).

Nothing happens unless a green plateau clearly stands out below full scale (code < 250), so
frames taken with normal WB gains, and night frames, are returned unchanged.
"""

from __future__ import print_function, division, absolute_import

import numpy as np


CLIP_CODE = 253          # 8-bit code at/above which red or blue counts as clipped
PLATEAU_MAX = 249        # a plateau at/above this is ordinary full-scale clipping, not the WB rung
MIN_REF = 0.005          # reference pixels needed (fraction of the frame) to trust the scene's ratios


def greenPlateau(g, lo=100):
    """ The green clip plateau (8-bit code) if one clearly stands out at the top of the green
        histogram below PLATEAU_MAX, else None. A few demosaic-edge pixels above it are normal.
    """
    v = g[g > lo]
    if v.size < 1000:
        return None
    h = np.bincount(v.ravel(), minlength=256)
    top = int(np.argmax(h[lo:])) + lo
    if top >= PLATEAU_MAX:
        return None
    others = np.median(h[max(lo, top - 20):top - 2]) if top - 2 > lo else 0
    if h[top] > 8*max(others, 1) and top >= np.percentile(v, 99.9) - 3:
        return top
    return None


CEILING_BAND = 12        # codes around the known green ceiling treated as clipped: the saved frames come
                         # from 4:2:0 video, and across a large bright gradient (the sun's halo) the
                         # plateau is smeared over ~20 codes (US05B1, 2026-09-29: 171-190 around 179)


def rebuildGreen(bgr, fallback=(1.0, 1.0), ceiling=None):
    """ Rebuild green where it sits on its clip plateau.

    Arguments:
        bgr: [ndarray] HxWx3 uint8 frame, OpenCV channel order (B, G, R).

    Keyword arguments:
        fallback: [tuple] (kR, kB) used when too little of the frame is unclipped.
        ceiling: [float] the known 8-bit code at which green clips (from the frame's own exposure
            record: RMS.FrameMetadata.greenCeiling); None = detect the plateau from the histogram.
            With it, pixels within CEILING_BAND of it count as green-clipped and green is only
            ever raised (never below its captured value).

    Return:
        (frame, info): [ndarray] the frame (the input object when unchanged), [dict or None]
            plateau code, clipped fraction, ratio source and values; None when unchanged.
    """
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        return bgr, None
    B8, G8, R8 = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    known = ceiling is not None and ceiling < PLATEAU_MAX
    if known:
        p = int(round(ceiling))
        if not (G8 >= p - CEILING_BAND).any():
            return bgr, None
    else:
        p = greenPlateau(G8)
        if p is None:
            return bgr, None

    lin = (bgr.astype(np.float32)/255.0)**2
    B, G, R = lin[..., 0], lin[..., 1], lin[..., 2]
    gp = (p/255.0)**2
    clip_g = G8 >= p - (CEILING_BAND if known else 1)
    clip_r, clip_b = R8 >= CLIP_CODE, B8 >= CLIP_CODE

    ref = (~clip_g) & (G > 0.5*gp) & ~clip_r & ~clip_b & (R > 1e-4) & (B > 1e-4)
    if ref.mean() >= MIN_REF:
        k_r, k_b = float(np.median(G[ref]/R[ref])), float(np.median(G[ref]/B[ref]))
        src = "scene"
    else:
        k_r, k_b = fallback
        src = "fallback"

    n = (~clip_r).astype(np.float32) + (~clip_b).astype(np.float32)
    est = (np.where(~clip_r, R*k_r, 0) + np.where(~clip_b, B*k_b, 0))/np.maximum(n, 1)
    est = np.where(n == 0, 1.0, est)
    # never below the captured green; with the histogram plateau also never below the plateau
    floor = G if known else np.maximum(G, gp)
    g_new = np.where(clip_g, np.clip(np.maximum(est, floor), 0, 1), G)

    out = bgr.copy()
    out[..., 1] = np.round(255.0*np.sqrt(g_new)).astype(np.uint8)
    all_sat = clip_g & clip_r & clip_b
    out[all_sat] = 255

    return out, {"plateau": p, "clipped": float(clip_g.mean()), "ratios": src, "ceiling": "record" if known else "histogram",
                 "k_r": k_r, "k_b": k_b, "all_saturated": float(all_sat.mean())}
