""" Nightly transparency-map product: per-frame, per-sky-cell extinction estimates
derived from the star-scoring product, computed ON STATION so consumers (e.g. the
contrail associator) just read a file - no estimator knowledge needed.

For any timestamp, look up the nearest frame (the product is at FF cadence, ~10 s)
and index the cell grid. Semantics consumers MUST respect:

- dm is extinction in magnitudes RELATIVE to the night's own clear baseline (the
  per-cell normalization absorbs static sensitivity structure and the night's
  aerosol zero-point; see Utils.SegmentedCloudDetector).
- Night-only by construction: no frames exist where no stars were scored.
- flags per cell/frame:
    FLAG_OK          - measured value
    FLAG_NO_DATA     - too few stars to judge (dm is NaN)
    FLAG_MOON_DOMAIN - moon above horizon and bright; the count channel reads
                       moonlight as extinction, treat dm as an upper bound
    (INTERPOLATED is reserved for future spatial fill - v1 never interpolates.)

Schema (npz):
    header       - JSON: schema_version, stationID, night, grid (nx, ny),
                   image size, dome_s, cadence provenance.
    t_unix       - [n_frames] float64 frame times (UTC).
    dm           - [n_frames, ny, nx] float16 extinction (mag), NaN = no data.
    ratio        - [n_frames, ny, nx] float16 normalized matched/expected.
    flags        - [n_frames, ny, nx] uint8.
"""

from __future__ import absolute_import, division, print_function

import json
import os

import numpy as np

SCHEMA_VERSION = 1

FILE_SUFFIX = "transparency_map.npz"

FLAG_OK = 0
FLAG_NO_DATA = 1
FLAG_MOON_DOMAIN = 2
FLAG_CEILING = 4       # dm railed at the estimator ceiling: treat as a LOWER BOUND
                       # ("at least this opaque"), not a measurement

DM_CEILING = 5.0       # mag - beyond this the star-count inversion carries no
                       # information (every star is long gone)

MOON_PHASE_BRIGHT = 25.0   # percent illumination; matches the flux moon gate


def mapFileName(night_name):
    return "{:s}_{:s}".format(night_name, FILE_SUFFIX)


def computeTransparencyMap(header, frames, stars, nx=8, ny=5):
    """ Compute the map from a loaded star-scoring product.

    Arguments:
        header, frames, stars: [dicts] As returned by StarScoring.loadStarScoring.

    Keyword arguments:
        nx, ny: [int] Cell grid.

    Return:
        (t_unix, dm, ratio, flags): arrays per the schema.
    """

    from Utils.SegmentedCloudDetector import computeCellSeries, extinctionSeries

    result = computeCellSeries(frames, stars, nx=nx, ny=ny)

    dome_s = float(header.get("dome_s", 0.5))
    dm = extinctionSeries(frames, stars, result, dome_s)

    n_frames = dm.shape[0]
    flags = np.zeros(dm.shape, dtype=np.uint8)
    flags[~np.isfinite(dm)] = FLAG_NO_DATA

    # Railed inversions are lower bounds, not measurements
    ceiling = np.isfinite(dm) & (dm >= DM_CEILING)
    dm = np.where(ceiling, DM_CEILING, dm)
    flags[ceiling] |= FLAG_CEILING

    moon = (np.asarray(frames["moon_alt"]) > 0) \
        & (np.asarray(frames["moon_phase"]) > MOON_PHASE_BRIGHT)
    flags[moon, :, :] |= FLAG_MOON_DOMAIN

    return (np.asarray(frames["frame_time_unix"], dtype=np.float64),
            dm.astype(np.float16), result["ratio"].astype(np.float16), flags)


def saveTransparencyMap(dir_path, night_name, station_id, scoring_header,
        t_unix, dm, ratio, flags, nx, ny):
    """ Write the product. Returns the file path. """

    header = dict(
        schema_version=SCHEMA_VERSION,
        stationID=str(station_id),
        night=night_name,
        nx=int(nx), ny=int(ny),
        dome_s=float(scoring_header.get("dome_s", 0.5)),
        dome_fit_date=str(scoring_header.get("dome_fit_date")),
        cadence=str(scoring_header.get("cadence", "binned")),
        note="dm is extinction (mag) relative to the night's clear baseline; "
             "flags: 1=no_data 2=moon_domain 4=ceiling(dm is a lower bound)",
    )

    path = os.path.join(dir_path, mapFileName(night_name))
    np.savez_compressed(path.replace(".npz", ""),
        header=json.dumps(header),
        t_unix=t_unix,
        dm=dm,
        ratio=ratio,
        flags=flags)

    return path


def loadTransparencyMap(path):
    """ Load a map product.

    Return:
        (header, t_unix, dm, ratio, flags)
    """

    with np.load(path, allow_pickle=False) as z:
        header = json.loads(str(z["header"]))
        return header, z["t_unix"], z["dm"].astype(np.float32), \
            z["ratio"].astype(np.float32), z["flags"]


def transparencyAt(path_or_loaded, t_unix_query, max_gap=30.0):
    """ The consumer call: the cell map nearest a query time.

    Arguments:
        path_or_loaded: [str or tuple] Product path, or the loadTransparencyMap tuple.
        t_unix_query: [float] Query time (unix seconds, UTC).

    Keyword arguments:
        max_gap: [float] Seconds beyond which no frame is considered current.

    Return:
        (dm, flags, dt): cell arrays and the signed offset to the used frame,
        or (None, None, None) if no frame is within max_gap (e.g. daylight).
    """

    loaded = loadTransparencyMap(path_or_loaded) \
        if isinstance(path_or_loaded, str) else path_or_loaded
    _, t, dm, _, flags = loaded

    if not len(t):
        return None, None, None

    j = int(np.argmin(np.abs(t - t_unix_query)))
    dt = float(t[j] - t_unix_query)

    if abs(dt) > max_gap:
        return None, None, None

    return dm[j], flags[j], dt
