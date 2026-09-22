# RPi Meteor Station
# Copyright (C) 2025  Dino Grzinic
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

""" Sky coordinates for sprite detections.

    The conversion follows RMS/Astrometry/ApplyAstrometryECSV.py: xyToRaDecPP with measurement=True gives
    true J2000 RA/Dec, and trueRaDec2ApparentAltAz with refraction=False gives the geometric azimuth and
    altitude of date. The receiving server triangulates, so it needs true directions, not refracted ones.

    Refraction is always taken out: explicitly when the platepar was fitted with refraction on, and as part
    of the fitted distortion when it was fitted with refraction off. Either kind of platepar is used.

    Each detection has its own frame time, so the pointing correction is computed per point
    (precompute_pointing_corr=False); the shortcut is only valid when all Julian dates are identical.
"""

from __future__ import print_function, division, absolute_import

import hashlib
import json

import numpy as np

from RMS.Logger import getLogger

# Astrometry is optional, without it the detections are only marked as not calibrated
try:
    from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP
    from RMS.Astrometry.Conversions import trueRaDec2ApparentAltAz
    SPRITE_ASTROMETRY_AVAILABLE = True
except ImportError:
    xyToRaDecPP = None
    trueRaDec2ApparentAltAz = None
    SPRITE_ASTROMETRY_AVAILABLE = False


# Get the logger from the main module
log = getLogger("rmslogger")


# Key names of the five calibrated points: the centroid, then the box corners
# (x1,y1), (x1,y2), (x2,y1), (x2,y2)
POINT_KEYS = [
    ("ra_j2000", "dec_j2000", "azimuth", "altitude"),
    ("box_ra_j2000_1", "box_dec_j2000_1", "box_azimuth_1", "box_altitude_1"),
    ("box_ra_j2000_2", "box_dec_j2000_2", "box_azimuth_2", "box_altitude_2"),
    ("box_ra_j2000_3", "box_dec_j2000_3", "box_azimuth_3", "box_altitude_3"),
    ("box_ra_j2000_4", "box_dec_j2000_4", "box_azimuth_4", "box_altitude_4"),
]

N_POINTS = len(POINT_KEYS)

# Station codes a default or unset platepar carries
_UNSET_STATION_CODES = ("", "NONE")

# Platepar fields sent to the server: what maps pixels to the sky and back, what says where the calibration
#   came from, and the few photometry numbers without which RMS's xyToRaDecPP will not run. The star list and
#   the old duplicate fields are left out; they make up most of a platepar file.
PLATEPAR_SERVER_FIELDS = (
    "version", "station_code", "lat", "lon", "elev", "JD", "Ho", "UT_corr", "X_res", "Y_res",
    "RA_d", "dec_d", "pos_angle_ref", "F_scale", "az_centre", "alt_centre", "rotation_from_horiz",
    "x_poly_fwd", "y_poly_fwd", "x_poly_rev", "y_poly_rev", "distortion_type", "equal_aspect",
    "force_distortion_centre", "asymmetry_corr", "refraction", "measurement_apparent_to_true_refraction",
    "auto_check_fit_refined", "auto_recalibrated",
    "mag_0", "mag_lev", "vignetting_coeff", "vignetting_fixed", "gamma", "extinction_scale",
)



def plateparUsable(platepar, ncols, nrows, config):
    """ Check whether a platepar can calibrate the detections of a given FF.

    Arguments:
        platepar: [Platepar or None] Loaded platepar.
        ncols: [int] FF width.
        nrows: [int] FF height.
        config: [Config] Configuration, stationID is used.

    Return:
        (ok, reason): [tuple]
            ok: [bool] True if the platepar may be used.
            reason: [str or None] Why it is not usable, or a caveat if it is usable but was never refined
                by an automatic fit check. None if it is fully usable.
    """

    if platepar is None:
        return False, "no platepar"

    # A platepar copied from another camera would put the events in the wrong sky
    station_code = getattr(platepar, "station_code", None)
    if (station_code is not None) and (str(station_code).strip().upper() not in _UNSET_STATION_CODES):
        if str(station_code).strip().upper() != str(config.stationID).strip().upper():
            return False, "platepar station code {:s} does not match the station ID {:s}".format(
                str(station_code), str(config.stationID))

    # Never silently rescale, the distortion model is tied to the resolution it was fitted at
    if (int(platepar.X_res), int(platepar.Y_res)) != (int(ncols), int(nrows)):
        return False, "platepar resolution {:d}x{:d} does not match the FF resolution {:d}x{:d}".format(
            int(platepar.X_res), int(platepar.Y_res), int(ncols), int(nrows))

    # An unrefined platepar is still usable, the caller warns about it once
    refined = bool(getattr(platepar, "auto_check_fit_refined", False))
    recalibrated = bool(getattr(platepar, "auto_recalibrated", False))
    if not (refined or recalibrated):
        return True, "platepar was not refined by an automatic fit check " \
            "(auto_check_fit_refined and auto_recalibrated are both False)"

    return True, None



def plateparProvenance(platepar, platepar_path):
    """ Describe which platepar calibrated the detections, for the product metadata.

    Arguments:
        platepar: [Platepar or None] Platepar used.
        platepar_path: [str or None] Path the platepar was loaded from.

    Return:
        [dict] Keys JD, source_path, X_res, Y_res, refraction, measurement_apparent_to_true_refraction,
            auto_check_fit_refined, auto_recalibrated. All values except source_path are None if there is
            no platepar.
    """

    # Keep the key set fixed so the metadata schema does not depend on the platepar being present
    if platepar is None:
        return {
            "JD": None,
            "source_path": platepar_path,
            "X_res": None,
            "Y_res": None,
            "refraction": None,
            "measurement_apparent_to_true_refraction": None,
            "auto_check_fit_refined": None,
            "auto_recalibrated": None,
        }

    return {
        "JD": float(platepar.JD),
        "source_path": platepar_path,
        "X_res": int(platepar.X_res),
        "Y_res": int(platepar.Y_res),
        "refraction": bool(getattr(platepar, "refraction", True)),
        "measurement_apparent_to_true_refraction": bool(getattr(platepar,
            "measurement_apparent_to_true_refraction", False)),
        "auto_check_fit_refined": bool(getattr(platepar, "auto_check_fit_refined", False)),
        "auto_recalibrated": bool(getattr(platepar, "auto_recalibrated", False)),
    }



def plateparForServer(platepar):
    """ The platepar as it is sent to the server: trimmed to the mapping fields, with its key.

        The server stores each distinct platepar once, under the SHA-256 of its canonical JSON (sorted keys,
        no whitespace), and the frames refer to it. The same key is computed here, so a local record can be
        matched with the server's copy.

    Arguments:
        platepar: [Platepar or None]

    Return:
        (record, sha256): [tuple]
            record: [dict or None] Plain JSON types only. None if there is no platepar or it cannot be sent.
            sha256: [str or None] Hex digest of the canonical JSON of record.
    """

    if platepar is None:
        return None, None

    # RMS's own serialization turns the numpy arrays into lists, the same way the platepar file is written
    try:
        full = json.loads(platepar.jsonStr())

    except Exception as e:
        log.warning("The platepar could not be serialized for the sprite server: {:s}".format(repr(e)))
        return None, None

    record = dict((key, full[key]) for key in PLATEPAR_SERVER_FIELDS if key in full)

    # The server refuses a platepar with NaN or infinity anywhere, and with it the whole frame
    try:
        text = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)

    except ValueError:
        log.warning("The platepar has values that are not finite numbers, it is not sent to the server")
        return None, None

    return record, hashlib.sha256(text.encode("ascii")).hexdigest()



def _markNotOk(det, reason):
    """ Mark a detection as not calibrated and remove any stale sky coordinates.

    Arguments:
        det: [dict] Detection record, modified in place.
        reason: [str] Why it is not calibrated.
    """

    det["astrometry_ok"] = False
    det["astrometry_reason"] = reason

    for keys in POINT_KEYS:
        for key in keys:
            det.pop(key, None)



def calibrateSpriteDetections(detections, platepar, usable, reason):
    """ Add J2000 RA/Dec and geometric azimuth/altitude of date to the detections of one FF.

        All points of all detections are converted in one vectorized call. Each detection uses its own
        Julian date, repeated for its five points (centroid and four box corners).

    Arguments:
        detections: [list of dict] Detections of one FF, modified in place.
        platepar: [Platepar or None] Platepar.
        usable: [bool] Result of plateparUsable.
        reason: [str or None] Reason from plateparUsable, stored if the platepar is not usable.

    Return:
        None
    """

    if not detections:
        return

    # Every detection carries the outcome, so no detection is silently left without a status
    if not usable:
        for det in detections:
            _markNotOk(det, reason if reason else "platepar not usable")
        return

    if not SPRITE_ASTROMETRY_AVAILABLE:
        for det in detections:
            _markNotOk(det, "astrometry modules not available")
        return

    # Five points per detection, in the order of POINT_KEYS, each with the detection's own JD
    x_list = []
    y_list = []
    jd_list = []
    for det in detections:
        x1, y1 = float(det["box_x1"]), float(det["box_y1"])
        x2, y2 = float(det["box_x2"]), float(det["box_y2"])
        x_list.extend([float(det["centroid_x"]), x1, x1, x2, x2])
        y_list.extend([float(det["centroid_y"]), y1, y2, y1, y2])
        jd_list.extend([float(det["jd"])]*N_POINTS)

    x_arr = np.array(x_list, dtype=np.float64)
    y_arr = np.array(y_list, dtype=np.float64)
    jd_arr = np.array(jd_list, dtype=np.float64)
    level_arr = np.ones_like(x_arr)

    # A failure marks this FF's detections only, the caller goes on with the next FF
    try:
        jd_out, ra_arr, dec_arr, _ = xyToRaDecPP(jd_arr, x_arr, y_arr, level_arr, platepar,
            extinction_correction=False, measurement=True, jd_time=True, precompute_pointing_corr=False)

        ra_arr = np.asarray(ra_arr, dtype=np.float64)
        dec_arr = np.asarray(dec_arr, dtype=np.float64)

        azim_arr, alt_arr = trueRaDec2ApparentAltAz(ra_arr, dec_arr, jd_arr, platepar.lat, platepar.lon,
            refraction=False)

        azim_arr = np.asarray(azim_arr, dtype=np.float64)
        alt_arr = np.asarray(alt_arr, dtype=np.float64)

    except Exception as e:
        log.warning("Sprite astrometry failed for {:s}: {:s}".format(
            str(detections[0].get("ff_name", "?")), repr(e)))
        for det in detections:
            _markNotOk(det, "astrometry failed: {:s}".format(str(e) if str(e) else repr(e)))
        return

    # Distribute the results back, five consecutive points per detection
    for i, det in enumerate(detections):

        sl = slice(i*N_POINTS, (i + 1)*N_POINTS)
        ra_d, dec_d, az_d, alt_d = ra_arr[sl], dec_arr[sl], azim_arr[sl], alt_arr[sl]

        if not (np.all(np.isfinite(ra_d)) and np.all(np.isfinite(dec_d))
                and np.all(np.isfinite(az_d)) and np.all(np.isfinite(alt_d))):
            _markNotOk(det, "astrometry produced non-finite coordinates")
            continue

        for j, (ra_key, dec_key, az_key, alt_key) in enumerate(POINT_KEYS):
            det[ra_key] = round(float(ra_d[j]), 5)
            det[dec_key] = round(float(dec_d[j]), 5)
            det[az_key] = round(float(az_d[j]), 3)
            det[alt_key] = round(float(alt_d[j]), 3)

        det["astrometry_ok"] = True
        det["astrometry_reason"] = None
