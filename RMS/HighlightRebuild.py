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

Two methods (config.day_highlight_method):

  chroma  rebuildGreen, above. Needs nothing but the frame; works on any camera.
  invert  rebuildGreenInvert: for cameras that run the ISP saturation at satu 255 (x1.99), which
          the chroma method cannot see past. Measured 2026-10-12 on .205 (GK7205V200/IMX307)
          from interleaved satu 128/255 frames: the stage is out = Y + s*(in - Y) in linear
          light, s = 255/128, Y = Rec.709 luma (fits to ~2 codes). The method undoes it, rebuilds
          green where it clipped at the level the frame's exposure record predicts, and re-applies
          it. Green estimate: a daylight scene is (nearly) a mix of two colours, sky and cloud,
          so in linear light every unclipped pixel lies close to one plane through black,
          G = a*R + b*B, fitted on the frame's own unclipped bright pixels. A channel the camera
          clipped at the output has no value of its own: B stays at full scale (measured: a
          clipped B is ~254.6 in the reference frames), R (and G where R and B both clipped)
          rises from its unclipped neighbours' level to full scale over ~INV_RAMP_PX, so a
          clipped patch does not jump to white next to much dimmer pixels. Against reference
          frames taken at unity green gain (same scene, exposure scaled 4-16x): core error 5-24
          codes, no magenta left. Calibrated on the Goke; the CV300 needs its own calibration.
"""

from __future__ import print_function, division, absolute_import

import cv2
import numpy as np

from RMS.FrameMetadata import BLC_FRACTION


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
    return out, {"method": "chroma", "plateau": level, "clipped": np.count_nonzero(w)/n, "ratios": "chroma",
                 "ceiling": "chroma", "k_r": 1.0, "k_b": 1.0, "all_saturated": np.count_nonzero(all_sat)/n}


SATU = 255/128.0            # the camera's ISP saturation at satu 255 (x1.99 about luma, linear light)
_LUMA = np.array([0.0722, 0.7152, 0.2126], np.float32)      # Rec.709 luma weights, B, G, R order
INV_NO_CLIP = 0.97          # green clip level (linear, from the exposure record) at/above which green clips
                            # at full scale (normal WB gains): the frame is left alone
INV_TOL = (0.85, 0.97)      # pre-saturation green / clip level: untouched below the first, rebuilt above the second
INV_OCLIP = 250             # 8-bit output code at/above which the camera clipped a channel
INV_RAMP_PX = 3.0           # a clipped R rises from its neighbours' level to full scale over ~this (px)
INV_NEIGH_SIGMA = 2.0       # Gaussian (px) over which the neighbours' level is taken
INV_MIN_REF = 2000          # unclipped bright pixels needed to fit the colour plane (else G = (R + B)/2)
INV_MIN_SUSP = 500          # magenta pixels needed to measure the clip level from the frame
_TILE = 16                  # px; the work is done on groups of tiles holding pixels that can change, grown by
                            # one tile: covers the neighbour Gaussian (radius 8), so the result is the full frame's


_LIN = ((np.arange(256, dtype=np.float32)/255.0)**2).astype(np.float32)   # 8-bit code -> linear (gamma 0.5)


def _saturate(b, g, r, s):
    """ The camera's saturation stage in linear light, Y + s*(x - Y) with Y = Rec.709 luma, on B, G, R planes
        (s = 1/SATU undoes it). """
    Yk = (b*_LUMA[0] + g*_LUMA[1] + r*_LUMA[2])*np.float32(1 - s)
    return b*np.float32(s) + Yk, g*np.float32(s) + Yk, r*np.float32(s) + Yk


def _tileMax(p):
    """ Maximum of an 8-bit plane over each _TILE x _TILE tile (edge tiles padded with 0). """
    p = np.pad(p, ((0, -p.shape[0] % _TILE), (0, -p.shape[1] % _TILE)))
    return p.reshape(p.shape[0]//_TILE, _TILE, p.shape[1]//_TILE, _TILE).max(axis=(1, 3))


def _regions(tiles, h, w):
    """ (rows, cols, own) of each group of tiles (a _TILE grid mask), grown by one tile: the group's
        bounding box in an h x w frame, and which of its pixels belong to the group. """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(tiles.astype(np.uint8), np.ones((3, 3), np.uint8)),
                                                        connectivity=8)
    out = []
    for i in range(1, n):
        x, y, tw, th = stats[i, :4]
        ys, xs = slice(y*_TILE, min((y + th)*_TILE, h)), slice(x*_TILE, min((x + tw)*_TILE, w))
        own = np.repeat(np.repeat(lab[y:y + th, x:x + tw] == i, _TILE, 0), _TILE, 1)
        out.append((ys, xs, own[:ys.stop - ys.start, :xs.stop - xs.start]))
    return out


def _clipped(b8, g8, r8, cG):
    """ Which pixels of one region are rebuilt and how much (w), with the clipped-R solve applied. """
    Lb, Lg, Lr = _LIN[b8], _LIN[g8], _LIN[r8]
    bp, gp, rp = _saturate(Lb, Lg, Lr, 1/SATU)                       # pre-saturation; exact where nothing clipped
    ocB, ocG, ocR = b8 >= INV_OCLIP, g8 >= INV_OCLIP, r8 >= INV_OCLIP   # output channels the camera clipped

    # Where R or B clipped at the output their true value is higher, so the true luma is higher and Gp
    # (which rises with luma) is a LOWER bound. Gp at the clip level => green clipped, whatever R and B did.
    w = np.clip((gp/cG - INV_TOL[0])/(INV_TOL[1] - INV_TOL[0]), 0, 1)

    # R clipped at the output (G not): Gp may sit under the clip only because R was cut. Solve with
    # G = cG and keep it if consistent (the solved pixel really drives R past full scale).
    Yg = (Lg - SATU*cG)/(1 - SATU)                                   # luma implied by the green output if G = cG
    B_from = np.where(ocB, bp, (Lb - (1 - SATU)*Yg)/SATU)
    R_lum = (Yg - _LUMA[1]*cG - _LUMA[0]*B_from)/_LUMA[2]
    solveR = ocR & ~ocG & (w < 1) & (gp > 0.8*cG) & ((Yg + SATU*(R_lum - Yg)) >= INV_NO_CLIP)
    changed = (w > 0) | solveR
    both = ocR & ocB & ~ocG & changed                                # R and B both cut, green under them
    return dict(rp=np.where(solveR, np.maximum(R_lum, rp), rp), bp=np.where(solveR & ~ocB, B_from, bp),
                gp=np.where(solveR, np.float32(cG), gp), w=np.where(solveR, np.float32(1), w), changed=changed,
                both=both, F=np.dstack([ocB, ocG, ocR]) & changed[..., None])


def _rebuilt(c, a, b):
    """ The rebuilt 8-bit BGR of one region (_clipped output c), colour plane G = a*R + b*B. """
    bp, gp, rp, w = c["bp"], c["gp"], c["rp"], c["w"]
    est = np.maximum(np.float32(a)*np.maximum(rp, 0) + np.float32(b)*np.maximum(bp, 0), gp)
    O = [np.clip(x, 0, 1) for x in _saturate(np.clip(bp, 0, None), gp + w*(est - gp), np.clip(rp, 0, None), SATU)]

    # Channels the camera clipped at the output (R and B both clipped: all three) have no value of their
    # own. B: full scale. R and G: from the level of the neighbouring pixels where that channel is known,
    # rising to full scale with the distance into the clipped patch. G never ends below R and B.
    F = c["F"]
    F[c["both"]] = True
    for ch in range(3):
        f = F[..., ch]
        if not f.any():
            continue
        if ch == 0:
            O[0][f] = 1.0
            continue
        m = (~f).astype(np.float32)
        den = cv2.GaussianBlur(m, (0, 0), INV_NEIGH_SIGMA)
        nb = np.where(den > 1e-3, cv2.GaussianBlur(O[ch]*m, (0, 0), INV_NEIGH_SIGMA)/np.maximum(den, 1e-6), 1.0)
        d = cv2.distanceTransform(f.astype(np.uint8), cv2.DIST_L2, 5)
        val = nb + (1 - nb)*(1 - np.exp(-np.maximum(d - 0.5, 0)/INV_RAMP_PX))
        O[ch] = np.where(f, np.maximum(val, O[ch]), O[ch])
    O[1] = np.where(F.any(-1), np.maximum(O[1], np.minimum(O[0], O[2])), O[1])
    return np.round(255*np.sqrt(cv2.merge(O))).astype(np.uint8)


def rebuildGreenInvert(bgr, meta):
    """ Rebuild green where it clipped below full scale, for a camera at satu 255: undo the
        saturation, rebuild green from the frame's colour plane, re-apply the saturation.

    Arguments:
        bgr: [ndarray] HxWx3 uint8 frame, OpenCV channel order (B, G, R).
        meta: [dict or None] the frame's exposure record (RMS.FrameMetadata.rowToDict); its
            ispdgain and wb_g give the green clip level. Without them the frame gets the chroma
            method (rebuildGreen).

    Return:
        (frame, info): [ndarray] the frame (the input object when unchanged), [dict or None]
            8-bit green clip level used ("plateau") and the one the record predicts ("ceiling"),
            fraction of pixels changed, colour plane G = k_r*R + k_b*B, fraction of R+B-clipped
            pixels; None when unchanged.
    """
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        return bgr, None
    try:
        cG = float(meta["ispdgain"])*BLC_FRACTION*float(meta["wb_g"])
    except (TypeError, KeyError, ValueError):
        cG = float("nan")
    if not (cG > 0 and np.isfinite(cG)):
        return rebuildGreen(bgr)
    if cG >= INV_NO_CLIP:
        return bgr, None
    ceiling = cG

    B8, G8, R8 = cv2.split(bgr)

    # The clip level itself, from the frame: green of the magenta-looking pixels (G output >12 codes
    # under both R and B, nothing clipped at the output) piles up at the clip. The record gives it to
    # ~1% on the Goke; the CV300 sits ~4% under it. Pixels near the R/B output clip are left out (the
    # camera lifts their green). The record's level when the frame has too few such pixels.
    susp = (cv2.subtract(cv2.min(B8, R8), G8) > 12) & (cv2.max(cv2.max(B8, G8), R8) < 240)
    gs = _saturate(_LIN[B8[susp]], _LIN[G8[susp]], _LIN[R8[susp]], 1/SATU)[1]
    gs = gs[gs > 0.7*cG]
    lvl = float(np.clip(np.median(gs)/cG, 0.8, 1.0)) if gs.size > INV_MIN_SUSP else 1.0
    cG = cG*lvl

    # Only pixels with pre-saturation green above 0.8 of the clip can change (below, w = 0 and the R
    # solve does not apply). It rises with each of B, G, R, so a tile whose channel maxima stay under
    # that holds none: the exact per-pixel work is done only on groups of the other tiles.
    tiles = _saturate(*[_LIN[_tileMax(p)] for p in (B8, G8, R8)], s=1/SATU)[1] > 0.8*cG
    regions = []
    for ys, xs, own in _regions(tiles, *B8.shape):
        c = _clipped(B8[ys, xs], G8[ys, xs], R8[ys, xs], cG)
        c["changed"] &= own
        if c["changed"].any():
            regions.append((ys, xs, c))
    if not regions:
        return bgr, None

    # The scene's colour plane G = a*R + b*B from pixels clearly below the clip (w = 0; clipped R is out
    # anyway, so no R solve), on every 2nd row and column; prefer fully unclipped ones. Weighted least
    # squares, weight = G.
    b8, g8, r8 = B8[::2, ::2], G8[::2, ::2], R8[::2, ::2]
    Bp, Gp, Rp = _saturate(_LIN[b8], _LIN[g8], _LIN[r8], 1/SATU)
    base = (g8 < INV_OCLIP) & (r8 < INV_OCLIP) & (Gp > 0.25*cG) & (Gp/cG <= INV_TOL[0]) & (Rp > 0) & (Bp > 0)
    ref = base & (b8 < INV_OCLIP)
    if np.count_nonzero(ref) < INV_MIN_REF/4:
        ref = base                                                   # blue clipped everywhere: its Bp is a lower bound
    if np.count_nonzero(ref) > INV_MIN_REF/4:
        g = Gp[ref].astype(np.float64)
        x, y = Rp[ref]*g, Bp[ref]*g
        a, b = np.linalg.lstsq([[np.dot(x, x), np.dot(x, y)], [np.dot(x, y), np.dot(y, y)]], [np.dot(x, g*g), np.dot(y, g*g)],
                               rcond=None)[0]                        # lstsq: a grey-only scene (R = B) is singular
    else:
        a = b = 0.5

    out = bgr.copy()
    n_changed = n_both = 0
    for ys, xs, c in regions:
        box, m = out[ys, xs], c["changed"]
        box[m] = _rebuilt(c, a, b)[m]
        n_changed += np.count_nonzero(m)
        n_both += np.count_nonzero(c["both"] & m)
    n = float(bgr.shape[0]*bgr.shape[1])
    return out, {"method": "invert", "plateau": int(round(255*cG**0.5)), "clipped": n_changed/n,
                 "ratios": "plane", "ceiling": int(round(255*ceiling**0.5)), "k_r": float(a), "k_b": float(b),
                 "all_saturated": n_both/n}
