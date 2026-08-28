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

import csv
import json
import os
import shutil
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageDraw

from RMS.Formats.FFfits import read as readFFfile
from RMS.Logger import getLogger
from RMS.Routines import MaskImage

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


# Get the logger from the main module
log = getLogger("rmslogger")


# Module-level caches (one per worker process, survive across calls)
_interpreter_cache = {}
_mask_cache = {}
_resized_mask_cache = {}


# Detection class names and colors
CLASS_NAMES = {0: "elf", 1: "sprite"}
CLASS_COLORS = {0: "red", 1: "blue"}
IOU_THRES = 0.1
MAX_DET = {0: 1, 1: 4}  # 0=elf: max 1, 1=sprite: max 4


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


def detectSprites(data_dir, ff_name, model_path, config):
    """ Detect sprites and elves in an FF frame using a TFLite model.

    This is the QueuedPool worker function. The TFLite interpreter is created lazily
    in the worker process and cached for reuse (never crosses process boundaries).

    Arguments:
        data_dir: [str] Path to the night data directory.
        ff_name: [str] Name of the FF file.
        model_path: [str] Path to the TFLite model file.
        config: [Configuration object]

    Return:
        [tuple] (ff_name, detections_list, timestamp) where each detection is a dict with keys:
            image_name, detection_type, model, confidence, centroid_x, centroid_y,
            box_x1, box_y1, box_x2, box_y2.
            Returns (ff_name, [], timestamp) if no detections or on error.
    """

    # Extract timestamp from FF filename
    basename = os.path.basename(ff_name)
    parts = basename.split("_")
    date_str, time_str = parts[2], parts[3]
    dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    timestamp = dt.timestamp()

    # Load interpreter lazily in worker process
    try:
        interpreter, input_details, output_details = _getInterpreter(model_path)
    except Exception as e:
        log.error("Failed to load TFLite model from {:s}: {:s}".format(model_path, repr(e)))
        return (ff_name, [], timestamp)

    # Read FF file
    try:
        ff = readFFfile(data_dir, ff_name)
    except FileNotFoundError:
        log.error("File {:s} not found in {:s}. Skipping.".format(ff_name, data_dir))
        return (ff_name, [], timestamp)

    # Use maxpixel as input (consider max-ave after model retraining)
    image = Image.fromarray(ff.maxpixel).convert("RGB")

    # Load mask once per worker process
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
        return (ff_name, [], timestamp)

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

        centroid_x = (x1 + x2) / 2
        centroid_y = (y1 + y2) / 2

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

    log.info("Detection on {:s}! {:d} object(s) found.".format(ff_name, len(detections)))

    return (ff_name, detections, timestamp)


def processResults(results, night_data_dir, config):
    """ Post-process sprite detection results after pool close.

    Applies the FalsePositiveFilter, writes CSV, marks images, and copies confirmed FF files.
    Runs in the parent process where state is not isolated.

    Arguments:
        results: [list] List of (ff_name, detections, timestamp) tuples from the pool.
        night_data_dir: [str] Path to the night data directory.
        config: [Configuration object]
    """

    from Utils.FalsePositiveFilter import FalsePositiveFilter

    # Collect all detections
    all_detections = []
    for result in results:
        if result is None:
            continue

        ff_name, detections, timestamp = result
        if detections:
            all_detections.append((ff_name, detections, timestamp))

    if not all_detections:
        log.info("No sprite detections this night.")
        return

    # Sort by timestamp
    all_detections.sort(key=lambda x: x[2])

    # Set up output directory
    save_dir = os.path.join(config.data_dir, config.sprite_dir, os.path.basename(night_data_dir))
    json_dir = os.path.join(save_dir, "data")
    os.makedirs(json_dir, exist_ok=True)

    # Track confirmed detections
    confirmed = []

    def onConfirmed(detection):
        """ Callback for confirmed detections that passed the FP filter. """
        confirmed.append(detection)

    # Run FP filter over sorted detections
    fp_filter = FalsePositiveFilter(
        window_seconds=60,
        max_detections=3,
        on_confirmed=onConfirmed,
    )

    for ff_name, detections, timestamp in all_detections:
        fp_filter.addDetection(timestamp=timestamp, data=detections, filename=ff_name)

    # Flush remaining buffered detections
    if all_detections:
        last_timestamp = all_detections[-1][2]
        fp_filter.flush(last_timestamp + 61)

    log.info("{:d} sprite detections, {:d} passed filter.".format(
        len(all_detections), len(confirmed)))

    # Write CSV for all detections (confirmed or not)
    csv_path = os.path.join(
        night_data_dir,
        "{:s}_sprite_detections.csv".format(os.path.basename(night_data_dir)),
    )
    _writeCSV(csv_path, all_detections)

    # Process confirmed detections
    for det in confirmed:
        ff_name = det.filename
        detections = det.data

        # Save marked/unmarked images
        _markSprites(detections, night_data_dir, ff_name, save_dir, config)

        # Copy FF file
        _copyFFFile(night_data_dir, ff_name, save_dir)

        # Save detection JSON
        json_path = os.path.join(json_dir, "{:s}.json".format(ff_name))
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(detections, f, indent=4)

    log.info("Sprite detection processing complete. {:d} confirmed events saved to {:s}".format(
        len(confirmed), save_dir))


def _writeCSV(csv_path, all_detections):
    """ Write detection results to a CSV file.

    Arguments:
        csv_path: [str] Path to the output CSV file.
        all_detections: [list] List of (ff_name, detections, timestamp) tuples.
    """

    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=";", quotechar="|", quoting=csv.QUOTE_MINIMAL)

        if not file_exists or os.stat(csv_path).st_size == 0:
            writer.writerow([
                "image name", "detection type", "model", "confidence",
                "centroid x", "centroid y", "box x1", "box y1", "box x2", "box y2",
            ])

        for ff_name, detections, timestamp in all_detections:
            for det in detections:
                writer.writerow([
                    det["image_name"], det["detection_type"], det["model"],
                    det["confidence"], det["centroid_x"], det["centroid_y"],
                    det["box_x1"], det["box_y1"], det["box_x2"], det["box_y2"],
                ])


def _markSprites(detections, data_dir, ff_name, save_dir, config):
    """ Draw detection boxes on the image and save marked/unmarked copies.

    Arguments:
        detections: [list] List of detection dicts.
        data_dir: [str] Path to the night data directory.
        ff_name: [str] Name of the FF file.
        save_dir: [str] Path to the output directory.
        config: [Configuration object]
    """

    try:
        ff = readFFfile(data_dir, ff_name)
    except FileNotFoundError:
        log.error("Cannot mark {:s}: file not found.".format(ff_name))
        return

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
    edit_image.save(os.path.join(marked_dir, "{:s}_marked.png".format(ff_name)))

    unmarked_dir = os.path.join(save_dir, "unmarked")
    os.makedirs(unmarked_dir, exist_ok=True)
    image.save(os.path.join(unmarked_dir, "{:s}_unmarked.png".format(ff_name)))


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


if __name__ == "__main__":

    import argparse
    import logging
    import time

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

        files = sorted([
            f for f in os.listdir(args.folder_path)
            if f.startswith("FF_") and f.endswith(".fits")
        ])

        logger.info("Processing {:d} FF files...".format(len(files)))

        results = []
        for filename in files:
            result = detectSprites(args.folder_path, filename, args.model, config)
            results.append(result)
            time.sleep(0.1)

        processResults(results, args.folder_path, config)
