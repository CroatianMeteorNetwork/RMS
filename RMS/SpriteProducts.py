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

""" Everything the sprite detector writes to disk: the per-night CSV and JSON Lines records, the marked and
    unmarked detection images, and the payload files that are sent to the sprite server.

    What goes where:

    - <night>_sprites.csv, one row per candidate including rejected ones, and the marked and unmarked JPEG
      images of confirmed detections are written to the top level of the night directory, so the nightly
      archive picks them up by itself (see RMS.ArchiveDetections.selectFiles).
    - <night>_sprites.json, one JSON record per confirmed FF, also goes in the night directory; the archive
      does not take .json files on its own, so Reprocess adds it explicitly.
    - The payloads that are posted to the sprite server go under data_dir/SpriteUploads/<night>/, because
      the upload queue keeps them for weeks (for instance while a station's key is not yet registered on
      the server) and night directories are pruned after a few days.
"""

from __future__ import print_function, division, absolute_import

import csv
import json
import os
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw

from RMS.Logger import getLogger
from RMS.Misc import RmsDateTime
from RMS.SpriteDetection import formatIsoTimestamp


# Get the logger from the main module
log = getLogger("rmslogger")


# Name of the directory in data_dir that holds the upload payloads, one sub-directory per night
UPLOAD_DIR_NAME = "SpriteUploads"

# Upload payload directories older than this are deleted. The upload queue gives up on an item after
#   21 days, so anything older is no longer referenced.
UPLOAD_DIR_DAYS_TO_KEEP = 30

# The server takes at most this many detections per frame
MAX_DETECTIONS_PER_FRAME = 20

# JPEG quality of the uploaded images. A 1280x720 maxpixel is 150-400 kB at this quality, well under the
#   server's 2 MiB limit, which a PNG of the same image routinely exceeds.
JPEG_QUALITY = 85

# Box colours in the marked image, by detection type
BOX_COLOURS = {"sprite": (0, 128, 255), "elve": (255, 64, 64)}
DEFAULT_BOX_COLOUR = (255, 255, 0)

# Columns of the per-night CSV. The first ones are what a person looks at; the sky coordinates follow.
CSV_COLUMNS = [
    "ff_name", "status", "timestamp", "frame_index", "detection_type", "model", "confidence",
    "artifact_share", "centroid_x", "centroid_y", "box_x1", "box_y1", "box_x2", "box_y2",
    "astrometry_ok", "astrometry_reason", "ra_j2000", "dec_j2000", "azimuth", "altitude",
    "box_ra_j2000_1", "box_dec_j2000_1", "box_azimuth_1", "box_altitude_1",
    "box_ra_j2000_2", "box_dec_j2000_2", "box_azimuth_2", "box_altitude_2",
    "box_ra_j2000_3", "box_dec_j2000_3", "box_azimuth_3", "box_altitude_3",
    "box_ra_j2000_4", "box_dec_j2000_4", "box_azimuth_4", "box_altitude_4",
]

# Detection fields sent to the server, in the server's own names (docs/API.md of spritemap-data-pipeline)
PAYLOAD_DETECTION_FIELDS = [
    "timestamp", "frame_index", "detection_type", "confidence", "centroid_x", "centroid_y",
    "box_x1", "box_y1", "box_x2", "box_y2", "azimuth", "altitude", "ra_j2000", "dec_j2000",
    "box_azimuth_1", "box_altitude_1", "box_azimuth_2", "box_altitude_2",
    "box_azimuth_3", "box_altitude_3", "box_azimuth_4", "box_altitude_4",
    "box_ra_j2000_1", "box_dec_j2000_1", "box_ra_j2000_2", "box_dec_j2000_2",
    "box_ra_j2000_3", "box_dec_j2000_3", "box_ra_j2000_4", "box_dec_j2000_4",
]


### File names ###


def ffStem(ff_name):
    """ FF file name without its extension, e.g. FF_US0001_20260828_031422_123_0000512.

    Arguments:
        ff_name: [str] FF file name.

    Return:
        [str]
    """

    return os.path.splitext(os.path.basename(ff_name))[0]


def csvPath(night_data_dir):
    """ Path of the per-night candidate CSV. """

    night_dir_name = os.path.basename(os.path.normpath(night_data_dir))

    return os.path.join(night_data_dir, "{:s}_sprites.csv".format(night_dir_name))


def jsonlPath(night_data_dir):
    """ Path of the per-night JSON Lines record of confirmed detections. """

    night_dir_name = os.path.basename(os.path.normpath(night_data_dir))

    return os.path.join(night_data_dir, "{:s}_sprites.json".format(night_dir_name))


def imagePaths(night_data_dir, ff_name):
    """ Paths of the marked and unmarked images of one FF.

        The names are exactly the ones the sprite server accepts, <FF stem>_marked.jpg and
        <FF stem>_unmarked.jpg, so the same files serve the nightly archive and the upload.

    Arguments:
        night_data_dir: [str] Night directory.
        ff_name: [str] FF file name.

    Return:
        [tuple] (marked_path, unmarked_path)
    """

    stem = ffStem(ff_name)

    return (os.path.join(night_data_dir, "{:s}_marked.jpg".format(stem)),
            os.path.join(night_data_dir, "{:s}_unmarked.jpg".format(stem)))


def uploadDir(config, night_data_dir):
    """ Directory for this night's upload payloads, created if needed.

    Arguments:
        config: [Config]
        night_data_dir: [str] Night directory.

    Return:
        [str]
    """

    night_dir_name = os.path.basename(os.path.normpath(night_data_dir))
    path = os.path.join(config.data_dir, UPLOAD_DIR_NAME, night_dir_name)

    # Python 2 has no exist_ok, so guard the creation instead
    if not os.path.isdir(path):
        os.makedirs(path)

    return path


def pruneUploadDirs(config, days_to_keep=UPLOAD_DIR_DAYS_TO_KEEP):
    """ Delete upload payload directories of nights older than days_to_keep.

    Arguments:
        config: [Config]

    Keyword arguments:
        days_to_keep: [float] Age limit in days. UPLOAD_DIR_DAYS_TO_KEEP by default.

    Return:
        [int] Number of directories removed.
    """

    root = os.path.join(config.data_dir, UPLOAD_DIR_NAME)

    if not os.path.isdir(root):
        return 0

    cutoff = time.time() - days_to_keep*86400
    removed = 0

    for name in os.listdir(root):

        path = os.path.join(root, name)

        # Judge a night by the last change of its directory, i.e. the last payload written into it
        try:
            if os.path.isdir(path) and (os.path.getmtime(path) < cutoff):
                shutil.rmtree(path)
                removed += 1

        except OSError as e:
            log.warning("Could not remove old sprite upload directory {:s}: {:s}".format(path, repr(e)))

    return removed


### Writers ###


def _cell(value):
    """ Format one CSV cell: empty for a missing value, the plain value otherwise. """

    if value is None:
        return ""

    return value


def appendSpriteCSV(csv_path, detections, status):
    """ Append candidate detections to the per-night CSV, writing the header on first use.

        Every candidate is recorded, whatever happened to it, so that the detector's behaviour over a night
        can be reviewed afterwards. The status says what happened.

    Arguments:
        csv_path: [str] Path of the CSV file.
        detections: [list] Detection dicts.
        status: [str] What happened to these detections, e.g. "confirmed", "false_positive_window",
            "artifact", "no_astrometry".

    Return:
        [bool] True if the rows were written.
    """

    try:
        write_header = (not os.path.isfile(csv_path)) or (os.path.getsize(csv_path) == 0)

        with open(csv_path, "a") as f:

            # Semicolons, like the rest of the RMS text products
            writer = csv.writer(f, delimiter=";", lineterminator="\n")

            if write_header:
                writer.writerow(CSV_COLUMNS)

            for det in detections:

                # The status is not part of the detection itself
                row = [det.get("ff_name", ""), status]
                row += [_cell(det.get(column)) for column in CSV_COLUMNS[2:]]

                writer.writerow(row)

        return True

    # A full disk must not stop detection; the record is lost, the detections are not
    except (IOError, OSError) as e:
        log.warning("Could not write sprite CSV {:s}: {:s}".format(csv_path, repr(e)))
        return False


def appendSpriteJSONL(jsonl_path, record):
    """ Append one record as a line of JSON to the per-night record of confirmed detections.

    Arguments:
        jsonl_path: [str] Path of the JSON Lines file.
        record: [dict] Anything JSON serialisable.

    Return:
        [bool] True if the record was written.
    """

    try:
        line = json.dumps(record, sort_keys=True, default=_jsonDefault)

        with open(jsonl_path, "a") as f:
            f.write(line + "\n")

        return True

    except (IOError, OSError, TypeError, ValueError) as e:
        log.warning("Could not write sprite record {:s}: {:s}".format(jsonl_path, repr(e)))
        return False


def _jsonDefault(value):
    """ Serialise the few non-JSON types that appear in detection records. """

    # Datetimes become ISO text, numpy scalars become plain Python numbers
    if hasattr(value, "isoformat"):
        return value.isoformat()

    if isinstance(value, np.generic):
        return value.item()

    raise TypeError("Not JSON serialisable: {:s}".format(type(value).__name__))


def writeDetectionImages(ff, detections, marked_path, unmarked_path):
    """ Write the maxpixel image of an FF with and without the detection boxes, as JPEG.

    Arguments:
        ff: [FFStruct] The FF the detections were found in.
        detections: [list] Detection dicts with box_x1..box_y2, detection_type and confidence.
        marked_path: [str] Where to write the image with the boxes drawn.
        unmarked_path: [str] Where to write the plain image.

    Return:
        [tuple] (marked_path, unmarked_path), with None for any image that could not be written. Always a
            pair, so the caller can unpack it whatever happens.
    """

    written = [None, None]

    try:

        # Grey maxpixel as the plain image; it is what the model looked at
        plain = Image.fromarray(np.asarray(ff.maxpixel).astype(np.uint8)).convert("L")
        plain.save(unmarked_path, quality=JPEG_QUALITY, optimize=True)
        written[1] = unmarked_path

        # The same image in colour, with a box and a label for every detection
        marked = plain.convert("RGB")
        draw = ImageDraw.Draw(marked)

        for det in detections:

            colour = BOX_COLOURS.get(det.get("detection_type"), DEFAULT_BOX_COLOUR)
            box = [det["box_x1"], det["box_y1"], det["box_x2"], det["box_y2"]]
            draw.rectangle(box, outline=colour, width=2)

            # Label above the box, or inside it when the box touches the top of the frame
            label = "{:s} {:.2f}".format(str(det.get("detection_type")), float(det.get("confidence", 0)))
            label_y = det["box_y1"] - 12 if det["box_y1"] >= 12 else det["box_y1"] + 2
            draw.text((det["box_x1"], label_y), label, fill=colour)

        marked.save(marked_path, quality=JPEG_QUALITY, optimize=True)
        written[0] = marked_path

    except Exception as e:
        log.warning("Could not write sprite images for {:s}: {:s}".format(
            os.path.basename(marked_path), repr(e)))

    return tuple(written)


### Server payload ###


def buildSpritePayload(config, ff_name, ff_start, fps, detections, night_dir_name):
    """ Build the frame object the sprite server takes, from the confirmed detections of one FF.

        Only detections with usable astrometry are included: the server triangulates from azimuth and
        altitude and refuses a detection without them. If none is usable, nothing is sent.

    Arguments:
        config: [Config] Station configuration (station id and position).
        ff_name: [str] FF file name.
        ff_start: [datetime] Time of the first frame of the FF block, naive UTC.
        fps: [float] Frame rate the frame indices refer to.
        detections: [list] Detection dicts after astrometry.
        night_dir_name: [str] Name of the night directory.

    Return:
        [tuple] (payload, reason): the payload dict, or None and a short reason when nothing can be sent.
    """

    # The server needs a position for every detection it takes
    usable = [det for det in detections if det.get("astrometry_ok")]

    if not usable:
        reasons = sorted(set(str(det.get("astrometry_reason")) for det in detections))
        return None, "no usable astrometry ({:s})".format(", ".join(reasons))

    # More than the server takes per frame cannot happen with the model's per-class caps, but make sure
    if len(usable) > MAX_DETECTIONS_PER_FRAME:
        log.warning("{:s}: sending the {:d} most confident of {:d} detections".format(
            ff_name, MAX_DETECTIONS_PER_FRAME, len(usable)))
        usable = sorted(usable, key=lambda det: det.get("confidence", 0), reverse=True)
        usable = usable[:MAX_DETECTIONS_PER_FRAME]

    payload_detections = []
    for det in usable:

        # Only the fields the server knows, and none that are missing
        entry = {}
        for field in PAYLOAD_DETECTION_FIELDS:
            value = det.get(field)
            if value is not None:
                entry[field] = _plain(value)

        payload_detections.append(entry)

    payload = {
        "station_id": config.stationID.upper(),
        "ff_name": ff_name,
        "night_dir": night_dir_name,
        "timestamp": formatIsoTimestamp(ff_start),
        "fps": float(fps),
        "latitude": float(config.latitude),
        "longitude": float(config.longitude),
        "elevation": float(config.elevation),
        "detections": payload_detections,
    }

    return payload, None


def _plain(value):
    """ Turn numpy scalars into plain Python numbers, so the JSON is exactly what was intended. """

    if isinstance(value, np.generic):
        return value.item()

    return value


def writeUploadPayload(payload, path):
    """ Write a payload file for the upload worker, atomically.

        The time of sending is stamped here, once. The worker posts these exact bytes on every attempt, so a
        retry after a failure or a reboot is recognised by the server as the same submission rather than
        stored twice.

    Arguments:
        payload: [dict] From buildSpritePayload().
        path: [str] Where to write it.

    Return:
        [bool] True if the file was written.
    """

    try:
        payload = dict(payload)
        payload["sent_at"] = formatIsoTimestamp(RmsDateTime.utcnow())

        body = json.dumps(payload, sort_keys=True, separators=(",", ":"))

        # Write next to the target and rename, so the worker never reads half a file
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            f.write(body)
            f.flush()
            os.fsync(f.fileno())

        os.rename(tmp_path, path)

        return True

    except (IOError, OSError, TypeError, ValueError) as e:
        log.warning("Could not write sprite upload payload {:s}: {:s}".format(path, repr(e)))
        return False
