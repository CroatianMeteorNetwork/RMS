""" Nightly star-scoring product: per-star matched/expected records for ALL scored
frames, ungated (see the score-everything / gate-at-the-verdict design).

CALSTARS is the upstream, catalog-blind detection product with external consumers -
its format and content contract are frozen, and this product is derived FROM it (plus
platepars, the light-dome model and the star catalog), never the reverse. Matched
stars reference their CALSTARS row (calstars_row) instead of duplicating detection
data; CALSTARS remains the canonical detection store.

Consumers apply their own domain gates: the flux verdict path uses only frames with
in_flux_domain (dark, moonless - the same gate outcomes the verdict actually used,
recorded here as flags); SQM applies its own gates; a segmented cloud detector may
define its own domain and spatial cells from the per-star records.

Schema (npz, versioned):
    header          - JSON string: schema_version, stationID, night, match_radius_px,
                      dome model provenance (fit_date, catalog_lim_mag), gate factor.
    frame_names     - [n_frames] FF file names (unicode).
    frame_time_unix - [n_frames] float64, FF timestamp (UTC, from the file name).
    sun_alt         - [n_frames] float32, deg.
    moon_alt        - [n_frames] float32, deg.
    moon_phase      - [n_frames] float32, percent illuminated.
    n_detected      - [n_frames] int32, CALSTARS detections on the frame.
    in_flux_domain  - [n_frames] bool, True where the flux verdict path scored the
                      frame (dark + moonless, per the gates in effect that night).
    star_frame      - [n_stars] int32, index into the frame arrays.
    star_x, star_y  - [n_stars] float32, projected catalog position (px).
    star_mag        - [n_stars] float16, catalog magnitude.
    star_p          - [n_stars] float16, model detection probability (expected count
                      contribution).
    calstars_row    - [n_stars] int32, row index into that frame's CALSTARS entry for
                      the matched detection, -1 if unmatched.
"""

from __future__ import absolute_import, division, print_function

import json
import os

import numpy as np

# v2: per-FF cadence; frames gain p_chance (per-frame chance-match floor); header
#     gains dome_s and cadence.
# v3: star records are floored at header store_p_min - stars whose model detection
#     probability is below it are not persisted. Their matches are chance-dominated
#     (bias for consumers that treat a match as a real detection) and an
#     adaptive-depth fit can push the catalog deep enough that they dominate the
#     product (>10k stars/frame, hundreds of MB, an OOM-killed station). The cut is
#     independent of the match outcome, so observed and expected shrink consistently
#     and ratio consumers need no change.
# v4: stars gain a stable identity and a second evidence channel.
#     star_cat_id     - row index into the star catalog as read at
#                       header catalog_lim_mag: the star's identity across frames
#                       (and across nights while the catalog depth is unchanged).
#     star_flux_snr   - forced patch photometry SNR on the FF avepixel for the
#                       bright set (model p >= forced_p_bootstrap), NaN elsewhere.
#     cell_bg         - [n_frames, 5, 8] float16 per-cell median avepixel level
#                       (ADU, mask-excluded; NaN when the FF was unreadable).
#                       Jointly with the transparency map this makes every
#                       cloud a probe of the light-pollution field: cloud
#                       radiance contrast vs dm per cell encodes the upward
#                       flux at the cloud's position.
#     calstars_row -2 - matched by forced photometry only: the extractor missed
#                       the star (saturation, shape gates, max_stars culling) but
#                       its aperture flux passed both detection floors (see
#                       RMS.PatchPhotometry). Cloud/transparency consumers count
#                       -2 as a detection; astrometry and photometry consumers
#                       must use only rows >= 0 (a real CALSTARS reference).
SCHEMA_VERSION = 4

FILE_SUFFIX = "star_scoring.npz"


def scoringFileName(night_name):
    return "{:s}_{:s}".format(night_name, FILE_SUFFIX)


def saveStarScoring(dir_path, night_name, header, frames, stars):
    """ Write the nightly scoring product.

    Arguments:
        dir_path: [str] Night directory.
        night_name: [str] Night directory name (used in the file name).
        header: [dict] Provenance fields; schema_version is added automatically.
        frames: [dict of lists/arrays] frame_names, frame_time_unix, sun_alt, moon_alt,
            moon_phase, n_detected, in_flux_domain - equal lengths.
        stars: [dict of lists/arrays] star_frame, star_x, star_y, star_mag, star_p,
            calstars_row - equal lengths.

    Return:
        path: [str] Written file path.
    """

    header = dict(header)
    header["schema_version"] = SCHEMA_VERSION

    path = os.path.join(dir_path, scoringFileName(night_name))

    np.savez_compressed(
        path.replace(".npz", ""),
        header=json.dumps(header),
        frame_names=np.array([str(n) for n in frames["frame_names"]]),
        frame_time_unix=np.asarray(frames["frame_time_unix"], dtype=np.float64),
        sun_alt=np.asarray(frames["sun_alt"], dtype=np.float32),
        moon_alt=np.asarray(frames["moon_alt"], dtype=np.float32),
        moon_phase=np.asarray(frames["moon_phase"], dtype=np.float32),
        n_detected=np.asarray(frames["n_detected"], dtype=np.int32),
        in_flux_domain=np.asarray(frames["in_flux_domain"], dtype=bool),
        p_chance=np.asarray(frames.get("p_chance",
            np.zeros(len(frames["frame_names"]))), dtype=np.float32),
        cell_bg=np.asarray(frames.get("cell_bg",
            np.full((len(frames["frame_names"]), 5, 8), np.nan)),
            dtype=np.float16),
        star_frame=np.asarray(stars["star_frame"], dtype=np.int32),
        star_cat_id=np.asarray(stars.get("star_cat_id",
            np.full(len(stars["star_frame"]), -1)), dtype=np.int32),
        star_flux_snr=np.asarray(stars.get("star_flux_snr",
            np.full(len(stars["star_frame"]), np.nan)), dtype=np.float16),
        star_x=np.asarray(stars["star_x"], dtype=np.float32),
        star_y=np.asarray(stars["star_y"], dtype=np.float32),
        star_mag=np.asarray(stars["star_mag"], dtype=np.float16),
        star_p=np.asarray(stars["star_p"], dtype=np.float16),
        calstars_row=np.asarray(stars["calstars_row"], dtype=np.int32),
    )

    return path


def loadStarScoring(path):
    """ Load a scoring product.

    Return:
        (header, frames, stars): header dict and dicts of ndarrays as in the schema.
    """

    with np.load(path, allow_pickle=False) as z:
        header = json.loads(str(z["header"]))
        # Tolerant of schema growth in both directions: load the keys present
        frames = {k: z[k] for k in z.files
                  if not k.startswith("star_") and k not in ("header", "calstars_row")}
        stars = {k: z[k] for k in z.files
                 if k.startswith("star_") or k == "calstars_row"}

    return header, frames, stars
