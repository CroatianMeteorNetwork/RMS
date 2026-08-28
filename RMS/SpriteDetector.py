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

from __future__ import print_function, division, absolute_import

import base64
import collections
import csv
import json
import multiprocessing
import os
import queue
import shutil
import signal
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageDraw

from RMS.Formats.FFfits import read as readFFfile
from RMS.Logger import getLogger, getLoggingQueue, initChildProcess
from RMS.Misc import AtomicFlag
from RMS.Routines import MaskImage
from Utils.FalsePositiveFilter import FalsePositiveFilter

# Some functions were adapted from the yolov5 github repository (utils/general.py)


# --- TensorFlow-Lite import cascade -----------------------------------------
#
# 1.  ai-edge-litert      <- new LiteRT wheels
# 2.  tflite_runtime      <- legacy stand-alone wheels
# 3.  tensorflow          <- TF proper, last-ditch fallback

TFLITE_AVAILABLE = False
TFLITE_BACKEND = "none"

try:
    from ai_edge_litert.interpreter import Interpreter
    TFLITE_AVAILABLE = True
    TFLITE_BACKEND = "litert"
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
        TFLITE_AVAILABLE = True
        TFLITE_BACKEND = "tflite_runtime"
    except ImportError:
        try:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            from tensorflow.lite.python.interpreter import Interpreter
            TFLITE_AVAILABLE = True
            TFLITE_BACKEND = "tf_full"
        except ImportError:
            TFLITE_AVAILABLE = False


# --- Astrometry imports (optional — calibration disabled if not available) ---

ASTROMETRY_AVAILABLE = False

try:
    from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP
    from RMS.Astrometry.Conversions import trueRaDec2ApparentAltAz, datetime2JD
    from RMS.Formats.Platepar import Platepar
    ASTROMETRY_AVAILABLE = True
except ImportError:
    Platepar = None


# Get the logger from the main module
log = getLogger("rmslogger")


# Module-level caches (one per process, survive across calls)
_interpreter_cache = {}
_mask_cache = {}
_resized_mask_cache = {}


# Detection class names and colors
CLASS_NAMES = {0: "elf", 1: "sprite"}
CLASS_COLORS = {0: "red", 1: "blue"}
IOU_THRES = 0.1
MAX_DET = {0: 1, 1: 4}  # 0=elf: max 1, 1=sprite: max 4


# ###########################################################################
#                       Inference utility functions
# ###########################################################################

def _getInterpreter(model_path):
    """ Lazily create and cache a TFLite interpreter per worker process.

    Arguments:
        model_path: [str] Path to the TFLite model file.

    Return:
        [tuple] (interpreter, input_details, output_details)
    """

    if model_path not in _interpreter_cache:

        interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        log.debug("TFLite interpreter created (backend: {:s})".format(TFLITE_BACKEND))
        log.debug("Input: shape={:s}, dtype={:s}".format(
            str(input_details["shape"]), str(input_details["dtype"])))

        _interpreter_cache[model_path] = (interpreter, input_details, output_details)

    return _interpreter_cache[model_path]


def xywh2xyxy(x):
    """ Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2].

    Arguments:
        x: [ndarray] Boxes in xywh format, shape (N, 4).

    Return:
        [ndarray] Boxes in xyxy format, shape (N, 4).
    """

    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def boxIouBatch(boxes_a, boxes_b):
    """ Compute IoU between two sets of bounding boxes.

    Arguments:
        boxes_a: [ndarray] First set of boxes in xyxy format, shape (N, 4).
        boxes_b: [ndarray] Second set of boxes in xyxy format, shape (M, 4).

    Return:
        [ndarray] IoU matrix, shape (N, M).
    """

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area_a = box_area(boxes_a.T)
    area_b = box_area(boxes_b.T)

    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[:, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[:, 2:])

    area_inter = np.prod(np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)

    return area_inter / (area_a[:, None] + area_b - area_inter)


def nms(predictions, iou_threshold=0.45):
    """ Non-maximum suppression on detection predictions.

    Arguments:
        predictions: [ndarray] Predictions with columns [x1, y1, x2, y2, score], shape (N, 5).

    Keyword arguments:
        iou_threshold: [float] IoU threshold for suppression. Default 0.45.

    Return:
        [ndarray] Boolean mask of kept predictions, shape (N,).
    """

    rows, columns = predictions.shape

    sort_index = np.flip(predictions[:, 4].argsort())
    predictions = predictions[sort_index]

    boxes = predictions[:, :4]
    ious = boxIouBatch(boxes, boxes)
    ious = ious - np.eye(rows)

    keep = np.ones(rows, dtype=bool)

    for index, iou in enumerate(ious):
        if not keep[index]:
            continue
        condition = iou > iou_threshold
        keep = keep & ~condition

    return keep[sort_index.argsort()]


def loadMask(config):
    """ Load the station mask if available.

    Arguments:
        config: [Configuration object]

    Return:
        [MaskStructure or None] Loaded mask, or None if not found.
    """

    mask = None
    mask_path_default = os.path.join(config.config_file_path, config.mask_file)

    if os.path.exists(mask_path_default):
        mask_path = os.path.abspath(mask_path_default)
        mask = MaskImage.loadMask(mask_path)

    return mask


def getPrediction(frame, interpreter, input_details, mask=None, resized_mask_cache=None):
    """ Run TFLite inference on a single frame.

    Arguments:
        frame: [PIL.Image] Input image.
        interpreter: [Interpreter] TFLite interpreter.
        input_details: [dict] Model input details.

    Keyword arguments:
        mask: [MaskStructure or None] Station mask.
        resized_mask_cache: [list] Mutable list used as a cache for the resized mask. Pass a list with one
            element [None] to enable caching across calls within the same worker.

    Return:
        [tuple] (prediction, image) where prediction is the raw model output and image is the
            preprocessed PIL image.
    """

    # Apply mask if available
    if mask is not None:

        frame_arr = np.array(frame)

        # Check if we need to compute the resized mask
        if resized_mask_cache is not None and resized_mask_cache[0] is None \
                and frame_arr.shape[:2] != mask.img.shape[:2]:

            log.debug("Rescaling mask and caching it")
            mask_img = Image.fromarray(mask.img)
            mask_resized = mask_img.resize((frame.width, frame.height))
            resized_mask_cache[0] = np.array(mask_resized.convert("RGB"))

        if resized_mask_cache is not None and resized_mask_cache[0] is not None:
            image = Image.fromarray(
                MaskImage.maskImage(frame_arr, resized_mask_cache[0], image=True))
        else:
            image = Image.fromarray(MaskImage.maskImage(frame_arr, mask))
    else:
        image = frame

    input_shape = input_details["shape"]
    image = image.convert("RGB")

    if input_shape[1] == 3:  # channels-first: [1, 3, H, W]
        h, w = input_shape[2], input_shape[3]
        image = image.resize((w, h))
        input_data = np.array(image, dtype=np.float32)
        input_data /= 255
        input_data = np.transpose(input_data, (2, 0, 1))
        input_data = input_data[None]
    else:  # channels-last: [1, H, W, 3]
        h, w = input_shape[1], input_shape[2]
        image = image.resize((w, h))
        input_data = np.array(image, dtype=np.float32)
        input_data /= 255
        input_data = input_data[None]

    interpreter.set_tensor(input_details["index"], input_data)
    interpreter.invoke()

    output_details = interpreter.get_output_details()[0]
    prediction = interpreter.get_tensor(output_details["index"])

    return prediction, image


def processPredictions(prediction, conf_thres=0.386):
    """ Process raw model output into filtered detections with NMS.

    Arguments:
        prediction: [ndarray] Raw model output tensor.

    Keyword arguments:
        conf_thres: [float] Confidence threshold. Default 0.386.

    Return:
        [ndarray] Filtered detections with columns [x1, y1, x2, y2, conf, class_id], shape (N, 6).
            Returns empty array with shape (0, 6) if no detections.
    """

    x = prediction[0]
    x = x.T

    boxes = x[:, :4]
    class_scores = x[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    conf = np.max(class_scores, axis=1)

    mask = conf > conf_thres
    boxes, conf, class_ids = boxes[mask], conf[mask], class_ids[mask]

    if boxes.shape[0] == 0:
        return np.zeros((0, 6))

    boxes_xyxy = xywh2xyxy(boxes)
    x = np.concatenate([boxes_xyxy, conf[:, None], class_ids[:, None]], axis=1)

    kept_rows = []
    for cid in np.unique(class_ids):
        class_mask = x[:, 5] == cid
        x_cls = x[class_mask]
        x_cls = x_cls[np.argsort(x_cls[:, 4])[::-1]]
        keep = nms(x_cls[:, :5], IOU_THRES)
        x_cls = x_cls[keep]

        cls_max_det = MAX_DET.get(int(cid))
        if cls_max_det is not None and cls_max_det > 0:
            x_cls = x_cls[:cls_max_det]
        kept_rows.append(x_cls)

    output = np.concatenate(kept_rows, axis=0)
    output = output[np.argsort(output[:, 4])[::-1]]

    return output if output.shape[0] > 0 else np.zeros((0, 6))


# ###########################################################################
#                       Detection + calibration
# ###########################################################################

def detectSpritesInFF(data_dir, ff_name, model_path, config, platepar=None):
    """ Run sprite detection on a single FF file.

    Core detection function usable standalone or from the SpriteDetector process.
    Creates/caches the TFLite interpreter per process.

    Arguments:
        data_dir: [str] Path to the night data directory.
        ff_name: [str] Name of the FF file.
        model_path: [str] Path to the TFLite model file.
        config: [Configuration object]

    Keyword arguments:
        platepar: [Platepar or None] Platepar for alt/az calibration.

    Return:
        [tuple] (ff_name, detections_list, timestamp, jd) where each detection is a dict with
            image/sky coordinates. Returns (ff_name, [], timestamp, None) on error.
    """

    # Extract timestamp from FF filename
    basename = os.path.basename(ff_name)
    parts = basename.split("_")
    date_str, time_str = parts[2], parts[3]
    dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    timestamp = dt.timestamp()

    # Compute JD for astrometry
    jd = None
    if ASTROMETRY_AVAILABLE and platepar is not None:
        jd = datetime2JD(dt)

    # Load interpreter lazily in this process
    try:
        interpreter, input_details, output_details = _getInterpreter(model_path)
    except Exception as e:
        log.error("Failed to load TFLite model from {:s}: {:s}".format(model_path, repr(e)))
        return (ff_name, [], timestamp, jd)

    # Read FF file
    try:
        ff = readFFfile(data_dir, ff_name)
    except FileNotFoundError:
        log.error("File {:s} not found in {:s}. Skipping.".format(ff_name, data_dir))
        return (ff_name, [], timestamp, jd)

    # Use maxpixel as input (consider max-ave after model retraining)
    image = Image.fromarray(ff.maxpixel).convert("RGB")

    # Load mask once per process
    mask_key = config.config_file_path
    if mask_key not in _mask_cache:
        _mask_cache[mask_key] = loadMask(config)
        _resized_mask_cache[mask_key] = [None]
    mask = _mask_cache[mask_key]
    resized_mask = _resized_mask_cache[mask_key]

    prediction, image = getPrediction(image, interpreter, input_details, mask=mask,
                                      resized_mask_cache=resized_mask)

    # Process predictions
    output = processPredictions(prediction)

    if output.size == 0:
        log.debug("No detections in {:s}.".format(ff_name))
        return (ff_name, [], timestamp, jd)

    # Build detection result list
    detections = []
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    for i in output:
        class_id = int(i[5])
        detection_type = CLASS_NAMES.get(class_id, "unknown_{:d}".format(class_id))

        x1 = int(i[0] * config.width)
        y1 = int(i[1] * config.height)
        x2 = int(i[2] * config.width)
        y2 = int(i[3] * config.height)

        centroid_x = (x1 + x2) / 2.0
        centroid_y = (y1 + y2) / 2.0

        detections.append({
            "image_name": ff_name,
            "detection_type": detection_type,
            "model": model_name,
            "confidence": float(i[4]),
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "box_x1": x1,
            "box_y1": y1,
            "box_x2": x2,
            "box_y2": y2,
        })

    # Calibrate with platepar (add RA/Dec J2000 and alt/az)
    if jd is not None and platepar is not None:
        _calibrateDetections(detections, jd, platepar)

    log.info("Detection on {:s}! {:d} object(s) found.".format(ff_name, len(detections)))

    return (ff_name, detections, timestamp, jd)


def _calibrateDetections(detections, jd, platepar):
    """ Add RA/Dec J2000 and alt/az to each detection using the platepar.

    Converts centroid and all 4 bounding box corners from image coordinates to sky coordinates.
    Uses trueRaDec2ApparentAltAz which handles J2000-to-epoch-of-date precession and refraction.

    Arguments:
        detections: [list] List of detection dicts (modified in place).
        jd: [float] Julian date of the observation.
        platepar: [Platepar] Platepar for the astrometric solution.
    """

    # Key names for the 5 calibrated points: centroid + 4 box corners
    point_keys = [
        ("ra_j2000", "dec_j2000", "azimuth", "altitude"),
        ("box_ra_j2000_1", "box_dec_j2000_1", "box_azimuth_1", "box_altitude_1"),
        ("box_ra_j2000_2", "box_dec_j2000_2", "box_azimuth_2", "box_altitude_2"),
        ("box_ra_j2000_3", "box_dec_j2000_3", "box_azimuth_3", "box_altitude_3"),
        ("box_ra_j2000_4", "box_dec_j2000_4", "box_azimuth_4", "box_altitude_4"),
    ]

    for det in detections:

        cx, cy = det["centroid_x"], det["centroid_y"]
        x1, y1 = float(det["box_x1"]), float(det["box_y1"])
        x2, y2 = float(det["box_x2"]), float(det["box_y2"])

        # 5 points: centroid, top-left, bottom-left, top-right, bottom-right
        xs = [cx, x1, x1, x2, x2]
        ys = [cy, y1, y2, y1, y2]
        jds = [jd] * 5
        levels = [1] * 5

        try:
            _, ra_data, dec_data, _ = xyToRaDecPP(
                jds, xs, ys, levels, platepar,
                jd_time=True, extinction_correction=False,
                precompute_pointing_corr=True
            )
        except Exception as e:
            log.debug("Astrometry calibration failed for {:s}: {:s}".format(
                det.get("image_name", "?"), repr(e)))
            return

        for i, (ra, dec) in enumerate(zip(ra_data, dec_data)):

            ra_key, dec_key, az_key, alt_key = point_keys[i]

            az, alt = trueRaDec2ApparentAltAz(
                float(ra), float(dec), jd, platepar.lat, platepar.lon, refraction=True
            )

            det[ra_key] = round(float(ra), 5)
            det[dec_key] = round(float(dec), 5)
            det[az_key] = round(float(az), 3)
            det[alt_key] = round(float(alt), 3)


# ###########################################################################
#                       File I/O utility functions
# ###########################################################################

def _appendCSV(csv_path, ff_name, detections, timestamp):
    """ Append detection rows to the CSV file, creating the header if needed.

    Arguments:
        csv_path: [str] Path to the output CSV file.
        ff_name: [str] Name of the FF file.
        detections: [list] List of detection dicts.
        timestamp: [float] UTC timestamp of the detection.
    """

    file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0

    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=";", quotechar="|", quoting=csv.QUOTE_MINIMAL)

        if not file_exists:
            writer.writerow([
                "image name", "detection type", "model", "confidence",
                "centroid x", "centroid y", "box x1", "box y1", "box x2", "box y2",
                "ra_j2000", "dec_j2000", "azimuth", "altitude",
                "box_ra_j2000_1", "box_dec_j2000_1", "box_azimuth_1", "box_altitude_1",
                "box_ra_j2000_2", "box_dec_j2000_2", "box_azimuth_2", "box_altitude_2",
                "box_ra_j2000_3", "box_dec_j2000_3", "box_azimuth_3", "box_altitude_3",
                "box_ra_j2000_4", "box_dec_j2000_4", "box_azimuth_4", "box_altitude_4",
            ])

        for det in detections:
            writer.writerow([
                det.get("image_name", ff_name),
                det.get("detection_type", ""),
                det.get("model", ""),
                det.get("confidence", ""),
                det.get("centroid_x", ""),
                det.get("centroid_y", ""),
                det.get("box_x1", ""),
                det.get("box_y1", ""),
                det.get("box_x2", ""),
                det.get("box_y2", ""),
                det.get("ra_j2000", ""),
                det.get("dec_j2000", ""),
                det.get("azimuth", ""),
                det.get("altitude", ""),
                det.get("box_ra_j2000_1", ""),
                det.get("box_dec_j2000_1", ""),
                det.get("box_azimuth_1", ""),
                det.get("box_altitude_1", ""),
                det.get("box_ra_j2000_2", ""),
                det.get("box_dec_j2000_2", ""),
                det.get("box_azimuth_2", ""),
                det.get("box_altitude_2", ""),
                det.get("box_ra_j2000_3", ""),
                det.get("box_dec_j2000_3", ""),
                det.get("box_azimuth_3", ""),
                det.get("box_altitude_3", ""),
                det.get("box_ra_j2000_4", ""),
                det.get("box_dec_j2000_4", ""),
                det.get("box_azimuth_4", ""),
                det.get("box_altitude_4", ""),
            ])


def _markSprites(detections, data_dir, ff_name, save_dir, config):
    """ Draw detection boxes on the image and save marked/unmarked copies.

    Arguments:
        detections: [list] List of detection dicts.
        data_dir: [str] Path to the night data directory.
        ff_name: [str] Name of the FF file.
        save_dir: [str] Path to the output directory.
        config: [Configuration object]

    Return:
        [str or None] Path to the saved marked image, or None on error.
    """

    try:
        ff = readFFfile(data_dir, ff_name)
    except FileNotFoundError:
        log.error("Cannot mark {:s}: file not found.".format(ff_name))
        return None

    image = Image.fromarray(ff.maxpixel).convert("RGB")
    edit_image = image.copy()
    draw = ImageDraw.Draw(edit_image)

    for det in detections:
        class_name = det["detection_type"]
        confidence = det["confidence"]

        top_left = (det["box_x1"], det["box_y1"])
        bottom_right = (det["box_x2"], det["box_y2"])

        color = "blue" if "sprite" in class_name else "red"
        draw.rectangle([top_left, bottom_right], outline=color, width=1)

        text_position = (top_left[0], top_left[1] - 20)
        draw.text(text_position, "{:s}-{:.3f}".format(class_name, confidence), fill=color)

    marked_dir = os.path.join(save_dir, "marked")
    os.makedirs(marked_dir, exist_ok=True)
    marked_path = os.path.join(marked_dir, "{:s}_marked.png".format(ff_name))
    edit_image.save(marked_path)

    unmarked_dir = os.path.join(save_dir, "unmarked")
    os.makedirs(unmarked_dir, exist_ok=True)
    image.save(os.path.join(unmarked_dir, "{:s}_unmarked.png".format(ff_name)))

    return marked_path


def _copyFFFile(data_dir, ff_name, save_dir):
    """ Copy an FF file to the sprite output directory.

    Arguments:
        data_dir: [str] Path to the night data directory.
        ff_name: [str] Name of the FF file.
        save_dir: [str] Path to the sprite output directory.
    """

    try:
        src_path = os.path.join(data_dir, ff_name)
        ffs_dir = os.path.join(save_dir, "FFs")
        os.makedirs(ffs_dir, exist_ok=True)
        dst_path = os.path.join(ffs_dir, ff_name)

        shutil.copy2(src_path, dst_path)
        log.info("Copied {:s} to {:s}".format(ff_name, ffs_dir))
    except Exception as e:
        log.error("Failed to copy FF file {:s}: {:s}".format(ff_name, repr(e)))


# ###########################################################################
#                       HTTPS upload client
# ###########################################################################

class SpriteUploader(object):
    """ HTTPS client for the sprite detection REST API.

    Handles JSON detection uploads and optional FF file uploads with retry,
    exponential backoff, and disk-backed queue for network resilience.
    Uses urllib.request (stdlib) — no external HTTP library needed.
    All uploads require HTTPS for data security.

    Arguments:
        config: [Configuration object]
        pending_file_path: [str] Path to the JSONL file for crash-recovery persistence.
    """

    def __init__(self, config, pending_file_path):

        self.base_url = config.sprite_upload_url.rstrip("/") if config.sprite_upload_url else ""
        self.timeout = config.sprite_upload_timeout
        self.upload_ff = config.sprite_upload_ff
        self.station_id = config.stationID
        self.pending_file = pending_file_path

        # (payload_dict, attempt_count, next_retry_time) as mutable lists
        self._deque = collections.deque()
        # (ff_path, ff_name, attempt_count, next_retry_time)
        self._ff_deque = collections.deque()

    def queueDetection(self, payload):
        """ Queue a confirmed detection payload for HTTPS upload.

        Arguments:
            payload: [dict] Detection payload to upload.
        """

        self._deque.append([payload, 0, 0])
        self._savePending()

    def queueFFFile(self, ff_path, ff_name):
        """ Queue an FF file for HTTPS upload (when sprite_upload_ff is enabled).

        Arguments:
            ff_path: [str] Full path to the FF file.
            ff_name: [str] FF filename.
        """

        if self.upload_ff and os.path.isfile(ff_path):
            self._ff_deque.append([ff_path, ff_name, 0, 0])

    def processQueue(self):
        """ Try to upload one pending item. Non-blocking: processes one JSON and one FF per call. """

        if not self.base_url:
            return

        now = time.time()

        # Try one JSON detection upload
        if self._deque:
            item = self._deque[0]
            payload, attempts, next_retry = item[0], item[1], item[2]

            if now >= next_retry:
                if self._postJSON(payload):
                    self._deque.popleft()
                    self._savePending()
                    log.info("Sprite detection uploaded successfully.")
                elif attempts < 5:
                    backoff = min(15 * (2 ** attempts), 300)
                    item[1] = attempts + 1
                    item[2] = now + backoff
                    log.debug("Upload retry {:d}/5 in {:d}s".format(attempts + 1, int(backoff)))
                else:
                    log.warning("Dropping detection after 5 retries: {:s}".format(
                        payload.get("ff_name", "?")))
                    self._deque.popleft()
                    self._savePending()

        # Try one FF file upload (lower priority)
        if self._ff_deque:
            item = self._ff_deque[0]
            ff_path, ff_name, attempts, next_retry = item[0], item[1], item[2], item[3]

            if now >= next_retry:
                if self._postFile(ff_path, ff_name):
                    self._ff_deque.popleft()
                    log.info("FF file {:s} uploaded successfully.".format(ff_name))
                elif attempts < 3:
                    backoff = min(30 * (2 ** attempts), 300)
                    item[2] = attempts + 1
                    item[3] = now + backoff
                else:
                    log.warning("Dropping FF upload after 3 retries: {:s}".format(ff_name))
                    self._ff_deque.popleft()

    def _postJSON(self, payload):
        """ POST detection JSON to /api/v1/detections.

        Arguments:
            payload: [dict] Detection payload.

        Return:
            [bool] True on HTTP 2xx response.
        """

        url = self.base_url + "/api/v1/detections"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data,
            headers={"Content-Type": "application/json"}, method="POST")

        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
            return 200 <= resp.status < 300
        except Exception as e:
            log.debug("Sprite API upload failed: {:s}".format(repr(e)))
            return False

    def _postFile(self, ff_path, ff_name):
        """ POST FF file as multipart/form-data to /api/v1/files.

        Arguments:
            ff_path: [str] Full path to the FF file.
            ff_name: [str] FF filename.

        Return:
            [bool] True on HTTP 2xx response.
        """

        url = self.base_url + "/api/v1/files"
        boundary = "----RMSSpriteUpload"

        try:
            with open(ff_path, "rb") as f:
                file_data = f.read()
        except Exception as e:
            log.error("Failed to read FF file {:s}: {:s}".format(ff_path, repr(e)))
            return False

        # Build multipart body
        parts = []
        parts.append("--{}\r\n".format(boundary).encode())
        parts.append(b"Content-Disposition: form-data; name=\"station_id\"\r\n\r\n")
        parts.append("{}\r\n".format(self.station_id).encode())
        parts.append("--{}\r\n".format(boundary).encode())
        parts.append(b"Content-Disposition: form-data; name=\"ff_name\"\r\n\r\n")
        parts.append("{}\r\n".format(ff_name).encode())
        parts.append("--{}\r\n".format(boundary).encode())
        parts.append("Content-Disposition: form-data; name=\"file\"; filename=\"{}\"\r\n".format(
            ff_name).encode())
        parts.append(b"Content-Type: application/octet-stream\r\n\r\n")
        parts.append(file_data)
        parts.append("\r\n--{}--\r\n".format(boundary).encode())

        body = b"".join(parts)

        req = urllib.request.Request(url, data=body,
            headers={"Content-Type": "multipart/form-data; boundary={}".format(boundary)},
            method="POST")

        try:
            file_timeout = min(self.timeout * 10, 120)
            resp = urllib.request.urlopen(req, timeout=file_timeout)
            return 200 <= resp.status < 300
        except Exception as e:
            log.debug("Sprite FF upload failed: {:s}".format(repr(e)))
            return False

    def loadPending(self):
        """ Load pending uploads from disk for crash recovery. """

        if not os.path.isfile(self.pending_file):
            return

        try:
            with open(self.pending_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        payload = json.loads(line)
                        self._deque.append([payload, 0, 0])

            if self._deque:
                log.info("Loaded {:d} pending sprite uploads from disk.".format(len(self._deque)))

        except Exception as e:
            log.error("Failed to load pending uploads: {:s}".format(repr(e)))

    def _savePending(self):
        """ Write pending uploads to disk (JSONL). Atomic via tmp + replace. """

        tmp_path = self.pending_file + ".tmp"

        try:
            with open(tmp_path, "w") as f:
                for item in self._deque:
                    f.write(json.dumps(item[0]) + "\n")
            os.replace(tmp_path, self.pending_file)
        except Exception as e:
            log.error("Failed to save pending uploads: {:s}".format(repr(e)))

    def flush(self):
        """ Try to upload all pending items (best-effort, for shutdown). """

        max_attempts = len(self._deque) + len(self._ff_deque)
        for _ in range(max_attempts):
            self.processQueue()


# ###########################################################################
#               SpriteDetector — dedicated real-time process
# ###########################################################################

class SpriteDetector(multiprocessing.Process):
    """ Dedicated process for real-time sprite/elve detection and upload.

    Runs TFLite inference on every FF frame during capture, applies the FalsePositiveFilter in
    real-time, and uploads confirmed detections via HTTPS to a REST API. Follows the UploadManager
    and EventMonitor pattern: a long-running multiprocessing.Process with AtomicFlag signaling.

    The Compressor feeds FF filenames into input_queue; this process pulls them, runs inference,
    calibrates with the platepar, filters, saves outputs to SpriteData, and uploads.

    Arguments:
        night_data_dir: [str] Path to the night data directory.
        config: [Configuration object]
    """

    def __init__(self, night_data_dir, config):

        super(SpriteDetector, self).__init__()

        self.night_data_dir = night_data_dir
        self.config = config
        self.input_queue = multiprocessing.Queue()

        self.exit = AtomicFlag()
        self.run_exited = AtomicFlag()

        self.model_path = os.path.join(config.rms_root_dir, "share", "sprite_detector.tflite")
        self.logging_queue = getLoggingQueue()

    def stop(self):
        """ Signal exit, wait for queue drain and flush, then join the process. """

        self.exit.set()
        log.debug("Sprite detector exit flag set")

        t_beg = time.time()
        while not self.run_exited.is_set():
            time.sleep(0.01)
            if (time.time() - t_beg) > 60:
                log.debug("Waited 60s for sprite detector to finish, killing it...")
                break

        log.debug("Joining sprite detector...")

        if self.is_alive():
            log.info("Sprite detector still alive, sending interrupt...")
            try:
                if self.pid:
                    os.kill(self.pid, signal.SIGINT)
                self.join(5)

                if self.is_alive():
                    log.warning("Sprite detector still alive after interrupt, terminating...")
                    self.terminate()

            except ProcessLookupError:
                log.info("Sprite detector already terminated.")
            except Exception as e:
                log.error("Error during sprite detector shutdown: {:s}".format(repr(e)))
                self.terminate()

            self.join(5)

            if self.is_alive():
                log.warning("Sprite detector survived terminate, sending SIGKILL...")
                try:
                    os.kill(self.pid, signal.SIGKILL)
                except (OSError, AttributeError):
                    pass

            self.join()

        else:
            self.join(timeout=5)

        log.debug("Sprite detector stopped.")

    def run(self):
        """ Main loop: pull FF files, run inference, calibrate, filter, save, and upload. """

        initChildProcess(self.logging_queue, self.config)

        # Initialize TFLite interpreter
        try:
            _getInterpreter(self.model_path)
        except Exception as e:
            log.error("Failed to initialize TFLite interpreter: {:s}".format(repr(e)))
            self.run_exited.set()
            return

        # Load platepar for coordinate calibration
        platepar = self._loadPlatepar()

        # Set up output directory (SpriteData/<night>/)
        night_dir_name = os.path.basename(self.night_data_dir)
        save_dir = os.path.join(
            self.config.data_dir, self.config.sprite_dir, night_dir_name
        )

        # CSV file path
        csv_path = os.path.join(
            self.night_data_dir,
            "{:s}_sprite_detections.csv".format(night_dir_name),
        )

        # Upload client with disk-backed queue
        pending_path = os.path.join(self.config.data_dir, "SPRITE_UPLOADS_PENDING.jsonl")
        uploader = SpriteUploader(self.config, pending_path)
        uploader.loadPending()

        # FP filter with real-time confirmed-detection callback
        fp_filter = FalsePositiveFilter(60, 3,
            lambda det: self._onConfirmed(det, save_dir, csv_path, uploader, night_dir_name))

        log.info("Sprite detector process started.")

        while True:

            # Exit when flagged and queue is drained
            if self.exit.is_set() and self.input_queue.empty():
                break

            try:
                data_dir, ff_name = self.input_queue.get(timeout=1.0)
            except queue.Empty:
                fp_filter.tick(time.time())
                uploader.processQueue()
                continue

            # Run detection + calibration on this FF file
            _, detections, timestamp, jd = detectSpritesInFF(
                data_dir, ff_name, self.model_path, self.config, platepar=platepar)

            # Feed FP filter (even with no detections, tick advances the window)
            if detections:
                fp_filter.addDetection(timestamp, detections, ff_name)
            fp_filter.tick(timestamp)

            # Try uploads between detections (~7s of idle time per cycle)
            uploader.processQueue()

        # Final flush: expire all remaining buffered detections
        fp_filter.flush(time.time() + 120)
        uploader.flush()

        log.info("Sprite detector process exiting.")
        self.run_exited.set()

    def _loadPlatepar(self):
        """ Load the platepar for coordinate calibration.

        Return:
            [Platepar or None] Loaded platepar, or None if unavailable.
        """

        if not ASTROMETRY_AVAILABLE or Platepar is None:
            log.warning("Astrometry modules not available — alt/az calibration disabled.")
            return None

        pp = Platepar()

        # Try config directory first, then night data directory (same as getPlatepar in Reprocess.py)
        pp_path = os.path.join(os.path.dirname(self.config.config_file_path),
                               self.config.platepar_name)

        if not os.path.isfile(pp_path):
            pp_path = os.path.join(self.night_data_dir, self.config.platepar_name)

        if os.path.isfile(pp_path):
            try:
                pp.read(pp_path, use_flat=self.config.use_flat)
                log.info("Platepar loaded from {:s}".format(pp_path))
                return pp
            except Exception as e:
                log.error("Failed to read platepar: {:s}".format(repr(e)))
                return None

        log.warning("No platepar found — alt/az calibration disabled.")
        return None

    def _onConfirmed(self, det, save_dir, csv_path, uploader, night_dir_name):
        """ Handle a confirmed detection that passed the FalsePositiveFilter.

        Arguments:
            det: [Detection] Confirmed detection from the FP filter.
            save_dir: [str] Path to the SpriteData output directory.
            csv_path: [str] Path to the CSV file.
            uploader: [SpriteUploader] Upload client.
            night_dir_name: [str] Night directory basename.
        """

        ff_name = det.filename
        detections = det.data
        data_dir = self.night_data_dir

        log.info("Confirmed sprite detection in {:s}".format(ff_name))

        # Ensure output dirs exist
        os.makedirs(save_dir, exist_ok=True)
        json_dir = os.path.join(save_dir, "data")
        os.makedirs(json_dir, exist_ok=True)

        # Write CSV row(s) for this detection
        _appendCSV(csv_path, ff_name, detections, det.timestamp)

        # Save marked/unmarked images
        marked_image_path = _markSprites(detections, data_dir, ff_name, save_dir, self.config)

        # Copy FF file to sprite output directory
        _copyFFFile(data_dir, ff_name, save_dir)

        # Write per-detection JSON
        json_path = os.path.join(json_dir, "{:s}.json".format(ff_name))
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(detections, f, indent=4)
        except Exception as e:
            log.error("Failed to write JSON {:s}: {:s}".format(json_path, repr(e)))

        # Build and queue upload payload
        if uploader.base_url:

            # Format timestamp as ISO 8601
            ts_utc = time.gmtime(det.timestamp)
            micros = int((det.timestamp % 1) * 1000000)
            timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%S", ts_utc)
            timestamp_iso += ".{:06d}Z".format(micros)

            payload = {
                "station_id": self.config.stationID,
                "timestamp": timestamp_iso,
                "ff_name": ff_name,
                "night_dir": night_dir_name,
                "detections": detections,
            }

            # Encode marked image as base64 PNG
            if marked_image_path and os.path.isfile(marked_image_path):
                try:
                    with open(marked_image_path, "rb") as f:
                        payload["marked_image"] = base64.b64encode(f.read()).decode("ascii")
                except Exception:
                    pass

            uploader.queueDetection(payload)

            # Queue FF file for upload if enabled
            ff_path = os.path.join(data_dir, ff_name)
            uploader.queueFFFile(ff_path, ff_name)


# ###########################################################################
#                       Standalone CLI
# ###########################################################################

if __name__ == "__main__":

    import argparse
    import logging

    model_path_default = "share/sprite_detector.tflite"

    logger = logging.getLogger("rmslogger")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    parser = argparse.ArgumentParser(description="Run sprite detection on FF files")
    parser.add_argument("folder_path", help="Path to the folder containing FF files")
    parser.add_argument(
        "--model", "-m", default=model_path_default,
        help="Path to the TFLite model file (default: %(default)s)")
    parser.add_argument(
        "--confidence", "-c", type=float, default=0.386,
        help="Confidence threshold for detection (default: %(default)s)")

    args = parser.parse_args()

    if not TFLITE_AVAILABLE:
        logger.warning("TensorFlow Lite is not available. Sprite detection skipped.")
    else:
        import RMS.ConfigReader as cr

        config = cr.parse(".config")

        # Load platepar if available
        pp = None
        if ASTROMETRY_AVAILABLE and Platepar is not None:
            pp = Platepar()
            pp_path = os.path.join(os.path.dirname(config.config_file_path),
                                   config.platepar_name)
            if not os.path.isfile(pp_path):
                pp_path = os.path.join(args.folder_path, config.platepar_name)
            if os.path.isfile(pp_path):
                pp.read(pp_path, use_flat=config.use_flat)
                logger.info("Platepar loaded from {:s}".format(pp_path))
            else:
                pp = None
                logger.warning("No platepar found — alt/az calibration disabled.")

        files = sorted([
            f for f in os.listdir(args.folder_path)
            if f.startswith("FF_") and f.endswith(".fits")
        ])

        logger.info("Processing {:d} FF files...".format(len(files)))

        # Collect all detections
        all_detections = []
        for filename in files:
            ff_name, detections, timestamp, jd = detectSpritesInFF(
                args.folder_path, filename, args.model, config, platepar=pp)
            if detections:
                all_detections.append((ff_name, detections, timestamp))
            time.sleep(0.1)

        if not all_detections:
            logger.info("No sprite detections found.")
        else:
            # Sort by timestamp and run FP filter
            all_detections.sort(key=lambda x: x[2])

            save_dir = os.path.join(config.data_dir, config.sprite_dir,
                                    os.path.basename(args.folder_path))
            csv_path = os.path.join(
                args.folder_path,
                "{:s}_sprite_detections.csv".format(os.path.basename(args.folder_path)),
            )

            confirmed = []

            def on_confirmed(det):
                confirmed.append(det)

            fp_filter = FalsePositiveFilter(60, 3, on_confirmed)

            for ff_name, detections, timestamp in all_detections:
                fp_filter.addDetection(timestamp, detections, ff_name)

            # Flush remaining
            last_timestamp = all_detections[-1][2]
            fp_filter.flush(last_timestamp + 61)

            logger.info("{:d} detections, {:d} passed filter.".format(
                len(all_detections), len(confirmed)))

            # Write CSV for all detections
            for ff_name, detections, timestamp in all_detections:
                _appendCSV(csv_path, ff_name, detections, timestamp)

            # Process confirmed detections
            for det in confirmed:
                _markSprites(det.data, args.folder_path, det.filename, save_dir, config)
                _copyFFFile(args.folder_path, det.filename, save_dir)

                json_dir = os.path.join(save_dir, "data")
                os.makedirs(json_dir, exist_ok=True)
                json_path = os.path.join(json_dir, "{:s}.json".format(det.filename))
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(det.data, f, indent=4)

            logger.info("Processing complete. {:d} confirmed events saved to {:s}".format(
                len(confirmed), save_dir))
