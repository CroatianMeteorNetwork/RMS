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

""" Sprite and elve detection on a single FF file: TFLite inference, box scaling, the artifact filter and
    the timing of the peak frame.

    The inference preprocessing (RGB conversion, PIL resize with the default resampling, /255 float32) and
    the postprocessing (per-class NMS with IOU_THRES and MAX_DET) are exactly what the model was trained
    and validated with. Do not change them without retraining.

    Some functions were adapted from the yolov5 github repository (utils/general.py).
"""

from __future__ import print_function, division, absolute_import

import datetime
import os

import numpy as np
from PIL import Image

from RMS.Astrometry.Conversions import datetime2JD
from RMS.Formats.FFfile import filenameToDatetime
from RMS.Logger import getLogger
from RMS.Routines import MaskImage


# TFLite import cascade, newest wheels first:
#   1. ai-edge-litert  <- new LiteRT wheels
#   2. tflite_runtime  <- legacy stand-alone wheels
#   3. tensorflow      <- TF proper, last-ditch fallback
SPRITE_TFLITE_AVAILABLE = False
SPRITE_TFLITE_BACKEND = "none"

try:
    from ai_edge_litert.interpreter import Interpreter
    SPRITE_TFLITE_AVAILABLE = True
    SPRITE_TFLITE_BACKEND = "litert"
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
        SPRITE_TFLITE_AVAILABLE = True
        SPRITE_TFLITE_BACKEND = "tflite_runtime"
    except ImportError:
        try:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            from tensorflow.lite.python.interpreter import Interpreter
            SPRITE_TFLITE_AVAILABLE = True
            SPRITE_TFLITE_BACKEND = "tf_full"
        except ImportError:
            Interpreter = None
            SPRITE_TFLITE_AVAILABLE = False


# Get the logger from the main module
log = getLogger("rmslogger")


# Model class ids to the server's event type names (the model calls class 0 "elf")
CLASS_NAMES = {0: "elve", 1: "sprite"}

# Postprocessing constants the model was validated with
IOU_THRES = 0.1
MAX_DET = {0: 1, 1: 4}

# Default confidence threshold, used only when the config does not provide one
DEFAULT_CONFIDENCE = 0.386

# Artifact filter: share of the box light that the k brightest frames must carry
ARTIFACT_SHARE_THRES = 0.1
ARTIFACT_TOP_K = 1

# Raw box coordinates above this are taken to be in model input pixels, not normalized
BOX_PIXEL_UNITS_THRES = 1.5


# Per-process caches, they survive across calls within one worker
_interpreter_cache = {}
_resized_mask_cache = {"source": None, "shape": None, "resized": None}
_logged_once = set()



def _logOnce(key, message):
    """ Log an info message only the first time the given key is seen in this process.

    Arguments:
        key: [str] Deduplication key.
        message: [str] Message to log.
    """

    if key not in _logged_once:
        _logged_once.add(key)
        log.info(message)



# ###########################################################################
#                       Inference utility functions
# ###########################################################################

def getSpriteInterpreter(model_path):
    """ Lazily create and cache a TFLite interpreter, one per model path and process.

    Arguments:
        model_path: [str] Path to the TFLite model file.

    Return:
        (interpreter, input_details): [tuple]
            interpreter: [Interpreter] TFLite interpreter with allocated tensors.
            input_details: [dict] Details of the first model input.
    """

    if model_path not in _interpreter_cache:

        # Fail loudly here, the caller decides whether a missing backend is fatal
        if not SPRITE_TFLITE_AVAILABLE:
            raise ImportError("No TFLite backend available (ai-edge-litert, tflite_runtime or tensorflow)")

        interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()[0]

        log.debug("TFLite interpreter created (backend: {:s})".format(SPRITE_TFLITE_BACKEND))
        log.debug("Input: shape={:s}, dtype={:s}".format(
            str(input_details["shape"]), str(input_details["dtype"])))

        _interpreter_cache[model_path] = (interpreter, input_details)

    return _interpreter_cache[model_path]



def xywh2xyxy(x):
    """ Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2].

    Arguments:
        x: [ndarray] Boxes in xywh format, shape (N, 4).

    Return:
        [ndarray] Boxes in xyxy format, shape (N, 4).
    """

    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2]/2
    y[..., 1] = x[..., 1] - x[..., 3]/2
    y[..., 2] = x[..., 0] + x[..., 2]/2
    y[..., 3] = x[..., 1] + x[..., 3]/2

    return y



def boxIouBatch(boxes_a, boxes_b):
    """ Compute the IoU between two sets of bounding boxes.

    Arguments:
        boxes_a: [ndarray] First set of boxes in xyxy format, shape (N, 4).
        boxes_b: [ndarray] Second set of boxes in xyxy format, shape (M, 4).

    Return:
        [ndarray] IoU matrix, shape (N, M).
    """

    def boxArea(box):
        return (box[2] - box[0])*(box[3] - box[1])

    area_a = boxArea(boxes_a.T)
    area_b = boxArea(boxes_b.T)

    # Intersection rectangle of every pair
    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[:, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[:, 2:])

    area_inter = np.prod(np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)

    return area_inter/(area_a[:, None] + area_b - area_inter)



def nms(predictions, iou_threshold=0.45):
    """ Non-maximum suppression on detection predictions.

    Arguments:
        predictions: [ndarray] Predictions with columns [x1, y1, x2, y2, score], shape (N, 5).

    Keyword arguments:
        iou_threshold: [float] IoU threshold for suppression. 0.45 by default.

    Return:
        [ndarray] Boolean mask of kept predictions in the input order, shape (N,).
    """

    rows, columns = predictions.shape

    # Visit the boxes from the highest score down
    sort_index = np.flip(predictions[:, 4].argsort())
    predictions = predictions[sort_index]

    boxes = predictions[:, :4]
    ious = boxIouBatch(boxes, boxes)
    ious = ious - np.eye(rows)

    keep = np.ones(rows, dtype=bool)

    # A kept box suppresses every box that overlaps it too much
    for index, iou in enumerate(ious):
        if not keep[index]:
            continue
        condition = iou > iou_threshold
        keep = keep & ~condition

    return keep[sort_index.argsort()]



def getPrediction(frame, interpreter, input_details):
    """ Run TFLite inference on a single image.

    Arguments:
        frame: [PIL.Image] Input image, already masked if masking is wanted.
        interpreter: [Interpreter] TFLite interpreter.
        input_details: [dict] Model input details.

    Return:
        (prediction, image): [tuple]
            prediction: [ndarray] Raw model output.
            image: [PIL.Image] The preprocessed image at the model input size.
    """

    input_shape = input_details["shape"]
    image = frame.convert("RGB")

    # The model may be exported either channels-first or channels-last
    if input_shape[1] == 3:

        # Channels-first: [1, 3, H, W]
        h, w = input_shape[2], input_shape[3]
        image = image.resize((w, h))
        input_data = np.array(image, dtype=np.float32)
        input_data /= 255
        input_data = np.transpose(input_data, (2, 0, 1))
        input_data = input_data[None]

    else:

        # Channels-last: [1, H, W, 3]
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



def processPredictions(prediction, conf_thres=DEFAULT_CONFIDENCE):
    """ Turn the raw model output into filtered detections, with per-class NMS and a per-class cap.

    Arguments:
        prediction: [ndarray] Raw model output tensor, shape (1, 4 + n_classes, n_anchors).

    Keyword arguments:
        conf_thres: [float] Confidence threshold. 0.386 by default.

    Return:
        [ndarray] Detections with columns [x1, y1, x2, y2, conf, class_id], shape (N, 6), sorted by
            descending confidence. An array of shape (0, 6) if there are no detections.
    """

    x = prediction[0]
    x = x.T

    boxes = x[:, :4]
    class_scores = x[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    conf = np.max(class_scores, axis=1)

    # Drop the low-confidence anchors before the quadratic NMS
    mask = conf > conf_thres
    boxes, conf, class_ids = boxes[mask], conf[mask], class_ids[mask]

    if boxes.shape[0] == 0:
        return np.zeros((0, 6))

    boxes_xyxy = xywh2xyxy(boxes)
    x = np.concatenate([boxes_xyxy, conf[:, None], class_ids[:, None]], axis=1)

    # NMS and the detection cap are applied per class, so a sprite never suppresses an elve
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
#                       Artifact filter and timing
# ###########################################################################

def _clipBox(x1, y1, x2, y2, ncols, nrows):
    """ Order a box's corners and clip it to the frame, keeping at least one pixel.

    Arguments:
        x1: [int] First column.
        y1: [int] First row.
        x2: [int] Second column (exclusive).
        y2: [int] Second row (exclusive).
        ncols: [int] Frame width.
        nrows: [int] Frame height.

    Return:
        (x1, y1, x2, y2): [tuple of int] Clipped box with 0 <= x1 < x2 <= ncols, 0 <= y1 < y2 <= nrows.
    """

    x1, x2 = sorted((int(x1), int(x2)))
    y1, y2 = sorted((int(y1), int(y2)))

    x1 = min(max(x1, 0), ncols - 1)
    y1 = min(max(y1, 0), nrows - 1)
    x2 = min(max(x2, x1 + 1), ncols)
    y2 = min(max(y2, y1 + 1), nrows)

    return x1, y1, x2, y2



def boxLightPerFrame(ff, x1, y1, x2, y2, subtract_background=False):
    """ Light in a box per frame index, as the per-frame increments of the reconstructed frames' box sums.

        d[i] is the sum of maxpixel over the box pixels whose maxframe is i. This is exactly the per-frame
        increment of the cumulative box sum of reconstructFrame(ff, i), computed in one bincount instead of
        one frame reconstruction per frame.

    Arguments:
        ff: [FFStruct] Loaded FF file, maxframe must not be None.
        x1: [int] Left column of the box.
        y1: [int] Top row of the box.
        x2: [int] Right column of the box, exclusive.
        y2: [int] Bottom row of the box, exclusive.

    Keyword arguments:
        subtract_background: [bool] Weight by maxpixel - avepixel (clipped at 0) instead of raw maxpixel.
            False by default.

    Return:
        [ndarray] float64 light per frame index, ff.nframes long (or maxframe.max() + 1 if nframes <= 0).
    """

    nrows, ncols = ff.maxpixel.shape[:2]

    # The slice is exclusive at the upper edge, as in the original filter
    x1, y1, x2, y2 = _clipBox(x1, y1, x2, y2, ncols, nrows)
    maxframe_roi = ff.maxframe[y1:y2, x1:x2].astype(np.int64)
    maxpixel_roi = ff.maxpixel[y1:y2, x1:x2].astype(np.float64)

    # Optionally remove the static background so that bright stars and sky glow do not count
    weights = maxpixel_roi
    if subtract_background:
        if ff.avepixel is not None:
            avepixel_roi = ff.avepixel[y1:y2, x1:x2].astype(np.float64)
            weights = np.clip(maxpixel_roi - avepixel_roi, 0, None)
        else:
            log.debug("FF has no avepixel, the artifact filter uses raw maxpixel")

    # The histogram spans the FF's frames; without a frame count use the largest frame index seen
    nframes = int(ff.nframes)
    if nframes > 0:
        nbins = nframes
    else:
        nbins = int(ff.maxframe.max()) + 1

    return np.bincount(maxframe_roi.ravel(), weights=weights.ravel(), minlength=nbins)[:nbins]



def spriteArtifactFilter(ff, x1, y1, x2, y2, k=ARTIFACT_TOP_K, thres=ARTIFACT_SHARE_THRES,
                         subtract_background=False):
    """ Decide whether the light in a box arrived abruptly (a sprite) or spread over many frames (an
        artifact such as a cloud edge, a moving light or noise).

        The statistic is the share of the box light carried by the k brightest frames, see boxLightPerFrame.

    Arguments:
        ff: [FFStruct] Loaded FF file.
        x1: [int] Left column of the box.
        y1: [int] Top row of the box.
        x2: [int] Right column of the box, exclusive.
        y2: [int] Bottom row of the box, exclusive.

    Keyword arguments:
        k: [int] Number of brightest frames whose share is taken. ARTIFACT_TOP_K by default.
        thres: [float] The detection is a sprite if the share is above this. ARTIFACT_SHARE_THRES by default.
        subtract_background: [bool] Weight by maxpixel - avepixel (clipped at 0) instead of raw maxpixel.
            False by default, because the threshold was tuned on raw maxpixel.

    Return:
        (is_sprite, frame_index, share): [tuple]
            is_sprite: [bool] True if the box passes the filter.
            frame_index: [int] Frame with the largest light increment in the box.
            share: [float] Share of the box light in the k brightest frames.
    """

    # Without the frame map nothing can be said, so do not reject the detection
    if ff.maxframe is None:
        log.debug("FF has no maxframe, the artifact filter is skipped")
        return True, 0, 1.0

    d = boxLightPerFrame(ff, x1, y1, x2, y2, subtract_background=subtract_background)

    # An empty box has no light to judge, reject it instead of dividing by zero
    total = d.sum()
    if total <= 0:
        return False, 0, 0.0

    share = float(np.sort(d)[-k:].sum()/total)
    is_sprite = share > thres
    frame_index = int(np.argmax(d))

    return bool(is_sprite), frame_index, share



def spriteFrameTime(ff_name, frame_index, fps):
    """ Compute the time of a frame within an FF block.

    Arguments:
        ff_name: [str] FF file name, its start time (with milliseconds) is taken from it.
        frame_index: [int] Frame index within the FF block.
        fps: [float] Frames per second of the FF block.

    Return:
        [datetime] Naive UTC time of the frame start.
    """

    if fps <= 0:
        raise ValueError("fps must be positive, given: {:s}".format(str(fps)))

    start = filenameToDatetime(os.path.basename(ff_name))

    return start + datetime.timedelta(seconds=frame_index/float(fps))



def formatIsoTimestamp(dt):
    """ Format a UTC datetime as ISO 8601 with microseconds and a Z suffix.

    Arguments:
        dt: [datetime] Naive UTC datetime, or an aware datetime which is converted to UTC.

    Return:
        [str] E.g. "2026-08-28T03:14:27.473000Z".
    """

    # Aware datetimes are brought to naive UTC so the Z suffix is always true
    if dt.tzinfo is not None:
        dt = dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)

    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")



# ###########################################################################
#                       Detection on one FF
# ###########################################################################

def _maskForFrame(mask, nrows, ncols):
    """ Return the mask as a 2-D array of the frame's shape, resizing it once with nearest-neighbour.

    Arguments:
        mask: [ndarray or MaskStructure] Mask image, 0 where masked.
        nrows: [int] Frame height.
        ncols: [int] Frame width.

    Return:
        [ndarray] 2-D mask of shape (nrows, ncols).
    """

    # Accept a MaskStructure as well as a bare image
    mask_img = getattr(mask, "img", mask)
    mask_img = np.asarray(mask_img)
    if mask_img.ndim == 3:
        mask_img = mask_img[:, :, 0]

    if mask_img.shape == (nrows, ncols):
        return mask_img

    # Resize once per (mask, frame shape); the cache keeps a reference to the source so ids cannot be reused
    cache = _resized_mask_cache
    if (cache["source"] is not mask) or (cache["shape"] != (nrows, ncols)):

        import cv2

        log.info("Resizing the mask from {:s} to {:s} for sprite detection".format(
            str(mask_img.shape), str((nrows, ncols))))

        cache["resized"] = cv2.resize(mask_img, (ncols, nrows), interpolation=cv2.INTER_NEAREST)
        cache["source"] = mask
        cache["shape"] = (nrows, ncols)

    return cache["resized"]



def _applyMask(image, mask):
    """ Mask a 2-D image in place the RMS way, masked pixels get the mean of the unmasked ones.

    Arguments:
        image: [ndarray] 2-D image, modified in place.
        mask: [ndarray] 2-D mask of the same shape, 0 where masked.

    Return:
        [ndarray] The masked image.
    """

    # A fully masked frame has no unmasked mean, blank it instead of producing NaN
    if not np.any(mask > 0):
        log.warning("The mask covers the whole frame, the image fed to the sprite model is blank")
        image[:] = 0
        return image

    return MaskImage.maskImage(image, mask, image=True)



def _scaleBoxes(boxes, input_details, ncols, nrows):
    """ Scale model boxes to FF pixels.

    Arguments:
        boxes: [ndarray] Boxes in xyxy format from processPredictions, shape (N, 4).
        input_details: [dict] Model input details, used if the boxes are in model input pixels.
        ncols: [int] FF width.
        nrows: [int] FF height.

    Return:
        [list of tuple] (x1, y1, x2, y2) integer boxes, clipped to the frame, at least one pixel in size.
    """

    boxes = np.array(boxes, dtype=np.float64)

    # Some exports give boxes in model input pixels rather than normalized, bring them to 0..1 first
    if boxes.size and boxes.max() > BOX_PIXEL_UNITS_THRES:

        input_shape = input_details["shape"]
        if input_shape[1] == 3:
            in_h, in_w = input_shape[2], input_shape[3]
        else:
            in_h, in_w = input_shape[1], input_shape[2]

        _logOnce("box_pixel_units", "Sprite model boxes are in input pixels ({:d}x{:d}), normalizing".format(
            int(in_w), int(in_h)))

        boxes[:, [0, 2]] /= float(in_w)
        boxes[:, [1, 3]] /= float(in_h)

    scaled = []
    for box in boxes:
        x1 = int(box[0]*ncols)
        y1 = int(box[1]*nrows)
        x2 = int(box[2]*ncols)
        y2 = int(box[3]*nrows)
        scaled.append(_clipBox(x1, y1, x2, y2, ncols, nrows))

    return scaled



def detectSpritesInFF(ff, ff_name, config, model_path, mask=None):
    """ Run sprite and elve detection on one already loaded FF file.

        No file reading, no astrometry and no writing is done here.

    Arguments:
        ff: [FFStruct] Loaded FF file.
        ff_name: [str] FF file name, .fits included.
        config: [Config] Configuration, sprite_confidence and fps are used.
        model_path: [str] Path to the TFLite model file.

    Keyword arguments:
        mask: [ndarray or None] Loaded 2-D uint8 mask image, 0 where masked. Masking is done only if it is
            given; the caller decides from config.sprite_use_mask. None by default.

    Return:
        [list of dict] Detection records, see the module interface spec. Empty if nothing was found or the
            model could not be loaded.
    """

    # A missing model or backend is reported and treated as no detections, capture must go on
    try:
        interpreter, input_details = getSpriteInterpreter(model_path)
    except Exception as e:
        log.error("Failed to load the sprite model {:s}: {:s}".format(str(model_path), repr(e)))
        return []

    # The FF's own size is authoritative, the config resolution may differ (binning, reprocessing)
    nrows, ncols = ff.maxpixel.shape[:2]
    if (int(ff.nrows) > 0) and (int(ff.ncols) > 0):
        nrows, ncols = int(ff.nrows), int(ff.ncols)

    # Mask a 2-D copy before the RGB conversion, so the mask shape matches the image
    image_2d = np.array(ff.maxpixel, copy=True)
    if mask is not None:
        image_2d = _applyMask(image_2d, _maskForFrame(mask, image_2d.shape[0], image_2d.shape[1]))

    image = Image.fromarray(image_2d).convert("RGB")

    prediction, _ = getPrediction(image, interpreter, input_details)

    conf_thres = float(getattr(config, "sprite_confidence", DEFAULT_CONFIDENCE))
    output = processPredictions(prediction, conf_thres=conf_thres)

    if output.shape[0] == 0:
        log.debug("No sprite detections in {:s}".format(ff_name))
        return []

    # The FF header rate is the measured one, the config rate is the nominal fallback
    fps = float(ff.fps) if (ff.fps is not None) and (ff.fps > 0) else float(config.fps)

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    boxes = _scaleBoxes(output[:, :4], input_details, ncols, nrows)

    detections = []
    for row, (x1, y1, x2, y2) in zip(output, boxes):

        class_id = int(row[5])
        detection_type = CLASS_NAMES.get(class_id, "unknown_{:d}".format(class_id))

        # Reject the boxes whose light is spread over many frames
        is_sprite, frame_index, share = spriteArtifactFilter(ff, x1, y1, x2, y2)
        if not is_sprite:
            log.debug("{:s}: {:s} rejected by the artifact filter (share {:.3f})".format(
                ff_name, detection_type, share))
            continue

        event_time = spriteFrameTime(ff_name, frame_index, fps)

        detections.append({
            "ff_name": ff_name,
            "detection_index": len(detections),
            "detection_type": detection_type,
            "model": model_name,
            "confidence": float(row[4]),
            "centroid_x": (x1 + x2)/2.0,
            "centroid_y": (y1 + y2)/2.0,
            "box_x1": int(x1),
            "box_y1": int(y1),
            "box_x2": int(x2),
            "box_y2": int(y2),
            "frame_index": int(frame_index),
            "artifact_share": float(share),
            "event_time": event_time,
            "timestamp": formatIsoTimestamp(event_time),
            "jd": float(datetime2JD(event_time)),
        })

    if detections:
        log.info("Sprite detection on {:s}: {:d} object(s) found".format(ff_name, len(detections)))

    return detections
