""" Beam-blended effective source catalog (EXPERIMENTAL - not yet wired into the
pipeline).

Wide-field meteor cameras are confusion-limited in dense star fields: at ~4 arcmin/px
the detection beam (PSF core) spans ~10 arcminutes and routinely contains several
catalog stars, whose summed flux crosses the detection threshold together. Detection
statistics therefore apply to BEAM SOURCES, not individual catalog stars - measured on
USC0G4 (2026-07-18): mag 5-6 stars with a flux-contributing beam neighbor were detected
2.6x more often than isolated stars of the same magnitude, and every CALSTARS detection
matched a catalog star 1:1 once blending was accounted for.

This module pre-merges a star catalog into effective sources with a brightest-first
greedy assignment that mimics the extractor's one-detection-per-local-maximum behavior:
a star is a source seed if no brighter source lies within the beam; fainter stars inside
a source's beam contribute their flux to it. Unlike transitive (union-find) merging,
a chain A-B-C where C is inside B's beam but outside A's yields TWO sources (B absorbs
into A only if within A's beam; C seeds its own source) - matching how local maxima
behave in real images.

Intended consumers (see the fleet investigation notes): dome-model trials and expected
counts, nightly recalibration matching and photometry, SkyFit displays and band-ratio
fitting, and the SkyQuality aperture-correction isolation test. Consumers must all use
the SAME effective catalog per night so matched and expected stay comparable. The beam
radius should come from the night's measured median FWHM times the plate scale, so the
blend topology tracks focus.
"""

from __future__ import absolute_import, division, print_function

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None


# Default beam radius as a multiple of the PSF sigma (flux gathering within the core;
# the empirical P-boost vs separation curve on USC0G4 supports ~2 px at FWHM ~3 px)
BEAM_SIGMA_FACTOR = 1.6


def beamRadiusArcsec(fwhm_px, plate_scale_arcsec_px):
    """ Beam (blend) radius on the sky for a given PSF and plate scale.

    Arguments:
        fwhm_px: [float] Median stellar FWHM in pixels (e.g. the night's CALSTARS median).
        plate_scale_arcsec_px: [float] Plate scale in arcsec per pixel (3600/F_scale for
            an RMS platepar).

    Return:
        radius: [float] Beam radius in arcseconds.
    """

    sigma_px = fwhm_px/2.355

    return BEAM_SIGMA_FACTOR*sigma_px*plate_scale_arcsec_px


def buildEffectiveSources(ra, dec, mag, beam_radius_arcsec):
    """ Merge catalog stars into beam-blended effective sources.

    Greedy brightest-first: iterate stars from bright to faint; a star within the beam
    of an already-seeded source joins it (flux added, position stays flux-weighted),
    otherwise it seeds a new source. This reproduces local-maxima detection semantics
    and cannot daisy-chain distant stars together.

    Arguments:
        ra, dec: [ndarray] Catalog positions in degrees (J2000 or epoch-corrected;
            whatever the consumer projects with).
        mag: [ndarray] Catalog magnitudes in the calibration band.
        beam_radius_arcsec: [float] Blend radius on the sky.

    Return:
        sources: [dict of ndarray]
            ra, dec        - flux-weighted source positions (deg)
            mag            - effective (blended) magnitude
            mag_brightest  - magnitude of the brightest member
            n_members      - number of catalog stars merged into the source
            seed_index     - index into the input arrays of the seed (brightest) member
    """

    if cKDTree is None:
        raise ImportError("scipy is required for buildEffectiveSources")

    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    mag = np.asarray(mag, dtype=np.float64)

    n = len(ra)
    if n == 0:
        empty = np.array([])
        return dict(ra=empty, dec=empty, mag=empty, mag_brightest=empty,
                    n_members=np.array([], dtype=int), seed_index=np.array([], dtype=int))

    # Unit vectors; for small angles the chord length equals the angle in radians
    ra_r, dec_r = np.radians(ra), np.radians(dec)
    xyz = np.column_stack([np.cos(dec_r)*np.cos(ra_r),
                           np.cos(dec_r)*np.sin(ra_r),
                           np.sin(dec_r)])
    chord = 2.0*np.sin(np.radians(beam_radius_arcsec/3600.0)/2.0)

    tree = cKDTree(xyz)
    flux = 10.0**(-0.4*mag)

    # Precompute the within-beam neighbor graph once (symmetric), then assign greedily
    # in brightness order: a star joins the nearest already-seeded neighbor, else seeds
    pairs = tree.query_pairs(chord, output_type="ndarray")
    nbrs = [[] for _ in range(n)]
    for i, j in pairs:
        nbrs[i].append(j)
        nbrs[j].append(i)

    order = np.argsort(mag, kind="stable")       # brightest first
    source_of = np.full(n, -1, dtype=np.int64)   # star index -> seed index
    is_seed = np.zeros(n, dtype=bool)
    seeds = []

    for idx in order:
        best_d, best_s = np.inf, -1
        for j in nbrs[idx]:
            if is_seed[j]:
                d = np.linalg.norm(xyz[idx] - xyz[j])
                if d < best_d:
                    best_d, best_s = d, j
        if best_s >= 0:
            source_of[idx] = best_s
        else:
            source_of[idx] = idx
            is_seed[idx] = True
            seeds.append(idx)

    seeds = np.array(seeds, dtype=np.int64)

    # Aggregate flux and flux-weighted positions per source
    seed_pos = {int(s): k for k, s in enumerate(seeds)}
    src_flux = np.zeros(len(seeds))
    src_x = np.zeros(len(seeds))
    src_y = np.zeros(len(seeds))
    src_z = np.zeros(len(seeds))
    src_n = np.zeros(len(seeds), dtype=int)

    for idx in range(n):
        k = seed_pos[int(source_of[idx])]
        src_flux[k] += flux[idx]
        src_x[k] += flux[idx]*xyz[idx, 0]
        src_y[k] += flux[idx]*xyz[idx, 1]
        src_z[k] += flux[idx]*xyz[idx, 2]
        src_n[k] += 1

    norm = np.sqrt(src_x**2 + src_y**2 + src_z**2)
    src_dec = np.degrees(np.arcsin(np.clip(src_z/norm, -1.0, 1.0)))
    src_ra = np.degrees(np.arctan2(src_y, src_x))%360.0

    return dict(
        ra=src_ra,
        dec=src_dec,
        mag=-2.5*np.log10(src_flux),
        mag_brightest=mag[seeds],
        n_members=src_n,
        seed_index=seeds,
    )
