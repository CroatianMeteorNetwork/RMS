import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

from RMS.Formats.CALSTARS import readCALSTARS
from RMS.Formats.FFfits import read as readFFfile
from RMS.QueuedPool import QueuedPool
from RMS.Routines import MaskImage
from Utils.FalsePositiveFilter import FalsePositiveFilter

try:
    from tflite_runtime.interpreter import Interpreter

    TFLITE_AVAILABLE = True
except ImportError:
    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        from tensorflow.lite.python.interpreter import Interpreter

        TFLITE_AVAILABLE = True
        USING_FULL_TF = True
    except ImportError:
        TFLITE_AVAILABLE = False
import csv

"""Some functions were adapted from the yolov5 github repository, mostly from the utils/general.py"""


# taken from ultralytics/yolov5/utils/general.py
def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


# adapted from https://blog.roboflow.com/how-to-code-non-maximum-suppression-nms-in-plain-numpy/
def box_iou_batch(boxes_a, boxes_b):

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    # determine surface of each box
    area_a = box_area(boxes_a.T)
    area_b = box_area(boxes_b.T)

    # determine the intersection box
    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[:, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[:, 2:])

    # calculate intersection area
    area_inter = np.prod(np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)

    # return iou
    return area_inter / (area_a[:, None] + area_b - area_inter)


def nms(predictions, iou_threshold=0.45):

    rows, columns = predictions.shape

    # sort predictions by descending score
    sort_index = np.flip(predictions[:, 4].argsort())
    predictions = predictions[sort_index]

    # prepare ious
    boxes = predictions[:, :4]
    ious = box_iou_batch(boxes, boxes)
    ious = ious - np.eye(rows)

    # start with accepting all boxes
    keep = np.ones(rows, dtype=bool)

    # iterate over ious in regard to each box
    for index, iou in enumerate(ious):
        # skip rejected boxes
        if not keep[index]:
            continue

        # discard boxes with high iou
        condition = iou > iou_threshold
        keep = keep & ~condition

    return keep[sort_index.argsort()]


def load_mask(config):
    mask = None
    mask_path_default = os.path.join(config.config_file_path, config.mask_file)
    if os.path.exists(mask_path_default):
        mask_path = os.path.abspath(mask_path_default)
        mask = MaskImage.loadMask(mask_path)
    return mask


class TLEDetector:
    def __init__(
        self,
        folder_path,
        model_path,
        config,
        log,
        conf_thres=0.386,
        disable_mask=False,
        min_stars=0,
        max_daily_detections=150,
        ff_filter_window_seconds=60,
        ff_filter_threshold=3,
    ):
        self.folder_path = folder_path
        self.model_path = model_path
        self.config = config
        self.log: logging.Logger = log

        self.conf_thres = conf_thres

        self.min_stars = min_stars
        if self.min_stars == 0:
            self.log.info("Filtering by number of stars disabled.")
        self.max_daily_detections = max_daily_detections
        self.detection_counter = 0

        self.fp_filter = FalsePositiveFilter(
            window_seconds=ff_filter_window_seconds,
            max_detections=ff_filter_threshold,
            on_confirmed=self.upload_detections,
            log=self.log,
        )

        self.CLASS_NAMES = {0: "elf", 1: "sprite"}
        self.CLASS_COLORS = {0: "red", 1: "blue"}
        self.IOU_THRES = 0.1
        self.MAX_DET = {0: 1, 1: 4}  # 0=elf: max 1, 1=sprite: max 4
        self.API_URL = ""  # todo

        self.save_dir = os.path.join(
            config.data_dir, config.tle_dir, os.path.basename(folder_path)
        )

        self.interpreter, self.input_details, self.output_details = (
            self.init_interpreter(model_path)
        )
        if disable_mask:
            self.mask = None
            self.log.info("Masking disabled, using original images")
        else:
            self.mask = load_mask(config)
            if self.mask is None:
                self.log.warning("No mask file found")
        self.resized_mask = None

        self.calstars = readCALSTARS(
            self.folder_path, "CALSTARS_" + os.path.basename(self.folder_path) + ".txt"
        )
        if self.calstars:
            self.calstars = self.calstars[0]
        else:
            self.log.warning("No CALSTARS file found")

        self.JSON_DIR = os.path.join(self.save_dir, "data")
        os.makedirs(self.JSON_DIR, exist_ok=True)

        self.pool = QueuedPool(self.find_tle, cores=1, log=log, low_priority=True)
        self.pool.startPool()

    def init_interpreter(self, model_path):
        try:
            interpreter = Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
        except Exception as e:
            self.log.error(f"Failed to load TFLite model from {model_path}: {e}")
            raise

        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        self.log.debug(
            f"Input details: shape={input_details['shape']}, dtype={input_details['dtype']}"
        )
        self.log.debug(f"Output details: shape={output_details['shape']}")
        return interpreter, input_details, output_details

    def get_prediction(
        self,
        frame,
    ):
        # remove known camera obstructions
        if self.mask is not None:
            self.log.info("Masking image")

            # Check if we need to compute the resized mask for the first time
            if (
                self.resized_mask is None
                and np.array(frame).shape != self.mask.img.shape
            ):
                self.log.debug("Rescaling mask and caching it")

                mask_img = Image.fromarray(self.mask.img)
                mask_resized = mask_img.resize((frame.width, frame.height))
                self.resized_mask = np.array(mask_resized.convert("RGB"))

            if self.resized_mask is not None:
                # Use the cached resized mask (requires 3rd argument True)
                image = Image.fromarray(
                    MaskImage.maskImage(np.array(frame), self.resized_mask, True)
                )
            else:
                image = Image.fromarray(MaskImage.maskImage(np.array(frame), self.mask))
        else:
            image = frame

        input_shape = self.input_details["shape"]
        image = image.convert("RGB")
        if input_shape[1] == 3:  # channels-first: [1, 3, H, W]
            h, w = input_shape[2], input_shape[3]
            image = image.resize((w, h))
            input_data = np.array(image, dtype=np.float32)
            input_data /= 255
            input_data = np.transpose(input_data, (2, 0, 1))  # HWC -> CHW
            input_data = input_data[None]  # add batch dim -> (1, 3, H, W)
        else:  # channels-last: [1, H, W, 3]
            h, w = input_shape[1], input_shape[2]
            image = image.resize((w, h))
            input_data = np.array(image, dtype=np.float32)
            input_data /= 255
            input_data = input_data[None]  # add batch dim -> (1, H, W, 3)

        self.interpreter.set_tensor(self.input_details["index"], input_data)

        # Run the inference
        self.interpreter.invoke()

        # Get the output tensor
        prediction = self.interpreter.get_tensor(self.output_details["index"])
        return prediction, image

    def process_predictions(self, prediction):
        x = prediction[0]  # (6, 2100)
        x = x.T  # -> (2100, 6)

        boxes = x[:, :4]
        class_scores = x[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        conf = np.max(class_scores, axis=1)

        mask = conf > self.conf_thres
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
            keep = nms(x_cls[:, :5], self.IOU_THRES)
            x_cls = x_cls[keep]

            cls_max_det = self.MAX_DET.get(int(cid))
            if cls_max_det is not None and cls_max_det > 0:
                x_cls = x_cls[:cls_max_det]
            kept_rows.append(x_cls)

        output = np.concatenate(kept_rows, axis=0)
        output = output[np.argsort(output[:, 4])[::-1]]

        return output[: output.shape[0]] if output.shape[0] > 0 else np.zeros((0, 6))

    def check_star_num(self, ff_name):
        if not self.calstars:
            return True
        for ff in self.calstars:
            if ff_name in ff[0]:
                return len(ff[1]) >= self.min_stars

    def find_tle(self, ff_name, save=True):
        self.log.debug(f"Processing {ff_name}.")

        # Extract timestamp from filename early
        basename = os.path.basename(ff_name)
        parts = basename.split("_")
        date_str, time_str = parts[2], parts[3]  # "YYYYmmDD", "HHMMSS"
        dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S").replace(
            tzinfo=timezone.utc
        )
        timestamp = dt.timestamp()

        if self.detection_counter >= self.max_daily_detections:
            self.fp_filter.tick(now=timestamp)
            self.log.warning("Too many detections in a single session. Stopping.")
            return 0
        if self.min_stars > 0 and not self.check_star_num(ff_name):
            self.fp_filter.tick(now=timestamp)
            self.log.info(f"Not enough stars in {ff_name}")
            return 0
        start_time = datetime.now()
        # dirname of folder_path is the main root folder of the night
        try:
            maxpixel = readFFfile(self.folder_path, ff_name).maxpixel
        except FileNotFoundError:
            self.log.error(f"File {ff_name} not found in {self.folder_path}. Skipping.")
            return 0
        prediction, image = self.get_prediction(
            Image.fromarray(maxpixel).convert("RGB")
        )
        output = self.process_predictions(
            prediction,
        )
        if output.size == 0:
            self.fp_filter.tick(now=timestamp)
            self.log.debug(f"No detections in {ff_name}.")
            return 0

        end_time = datetime.now()
        # Calculate elapsed time
        elapsed_time = end_time - start_time
        self.log.debug(f"Elapsed time in seconds: {elapsed_time.total_seconds()}")

        # for now we will store even detections that dont pass the filter 
        # so that we have at least some information about them, like how those false positives look like
        results = self.store_detections(
            image,
            output,
            save,
            ff_name,
        )

        self.log.info(f"Detection on {ff_name}! Adding to the filter.")
        self.fp_filter.add_detection(
            timestamp=timestamp, data=results, filename=ff_name
        )
        self.detection_counter += 1

        return len(output)

    def store_detections(self, image, output, save, imgname):
        self.mark_tles(output, image, imgname, save)

        csv_path = os.path.join(
            self.folder_path,
            f"{os.path.basename(self.folder_path)}_tle_detections.csv",
        )
        file_exists = os.path.isfile(csv_path)

        detections = []
        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.writer(
                csvfile, delimiter=";", quotechar="|", quoting=csv.QUOTE_MINIMAL
            )
            if not file_exists or os.stat(csv_path).st_size == 0:
                writer.writerow(
                    [
                        "image name",
                        "detection type",
                        "model",
                        "confidence",
                        "centroid x",
                        "centroid y",
                    ]
                )
            for i in output:
                class_id = int(i[5])
                detection_type = self.CLASS_NAMES.get(class_id, f"unknown_{class_id}")

                # Calculate the geometric centroid of the detection box
                x1 = int(i[0] * self.config.width)
                y1 = int(i[1] * self.config.height)
                x2 = int(i[2] * self.config.width)
                y2 = int(i[3] * self.config.height)

                centroid_x = (x1 + x2) / 2
                centroid_y = (y1 + y2) / 2

                detection_data = {
                    "image_name": imgname,
                    "detection_type": detection_type,
                    "model": os.path.splitext(os.path.basename(self.model_path))[0],
                    "confidence": float(i[4]),
                    "centroid_x": centroid_x,
                    "centroid_y": centroid_y,
                }

                detections.append(detection_data)
                writer.writerow(list(detection_data.values()))

        return detections

    def mark_tles(self, output, image, imgname, save=True):
        edit_image = image.copy()
        draw = ImageDraw.Draw(edit_image)
        # Draw the rectangle
        width, height = edit_image.size
        for i in range(output.shape[0]):
            class_id = int(output[i, 5])
            color = self.CLASS_COLORS.get(
                class_id, "yellow"
            )  # fallback for unexpected ids

            top_left = (output[i, 0] * width, output[i, 1] * height)
            bottom_right = (output[i, 2] * width, output[i, 3] * height)
            draw.rectangle([top_left, bottom_right], outline=color, width=1)
            # Display the number above the rectangle
            number = str(round(output[i, 4], 3))
            text_position = (
                top_left[0],
                top_left[1] - 20,
            )  # Adjust the position as needed
            draw.text(
                text_position,
                self.CLASS_NAMES.get(class_id, "?") + "-" + number,
                fill=color,
            )

        # Save the modified image
        if save:
            MARKED_DIR = os.path.join(self.save_dir, "marked")
            os.makedirs(MARKED_DIR, exist_ok=True)
            edit_image.save(f"{os.path.join(MARKED_DIR, imgname + '_marked')}.png")
            # its useful to ahve unmarked ones since they can be used in model training
            UNMARKED_DIR = os.path.join(self.save_dir, "unmarked")
            os.makedirs(UNMARKED_DIR, exist_ok=True)
            image.save(f"{os.path.join(UNMARKED_DIR, imgname + '_unmarked')}.png")

    def upload_detections(self, payload):
        try:
            response = requests.post(self.API_URL, json=payload.data)
            self.log.info(f"Upload status code {response.status_code}")
        except Exception as e:
            self.log.error(f"Failed to send TLE data to server: {e}")

        with open(
            os.path.join(self.JSON_DIR, f"{payload.filename}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(payload.data, f, indent=4)

    def close(self):
        self.pool.closePool()
        results = self.pool.getResults()
        results = [r for r in results if r is not None]
        self.log.info(
            f"{sum(results)} TLEs have been detected this night over {self.detection_counter} FF files."
        )
        self.pool.deleteBackupFiles()


if __name__ == "__main__":
    import argparse
    import time

    DEBUG_MODEL_PATH = "share/tle_detector.tflite"

    logger = logging.getLogger("TLE_Detector")

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    logger.info("Logger created successfully")

    parser = argparse.ArgumentParser(description="Run TLE detection on FITS files")
    parser.add_argument("folder_path", help="Path to the folder containing FITS files")
    parser.add_argument(
        "--model",
        "-m",
        default=DEBUG_MODEL_PATH,
        help="Path to the TFLite model file (default: %(default)s)",
    )
    parser.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=0.386,
        help="Confidence threshold for detection (default: %(default)s)",
    )
    parser.add_argument(
        "--star-threshold",
        "-s",
        type=int,
        default=0,
        help="Minimum number of stars on image to accept detection (default: %(default)s)",
    )
    parser.add_argument(
        "--disable-mask",
        "-d",
        action="store_true",
        help="Disable the use of mask even if available",
    )

    args = parser.parse_args()

    import RMS.ConfigReader as cr

    # Load the configuration file
    config = cr.parse(".config")

    if not TFLITE_AVAILABLE:
        logger.warning(
            "TensorFlow Lite is not available on this system. TLE detection skipped..."
        )
    else:
        detector = TLEDetector(
            folder_path=args.folder_path,
            model_path=args.model,
            conf_thres=args.confidence,
            config=config,
            log=logger,
            disable_mask=args.disable_mask,
            min_stars=args.star_threshold,
        )

        try:
            # Find all FF fits files in data_dir
            files = [
                f
                for f in os.listdir(args.folder_path)
                if f.startswith("FF_") and f.endswith(".fits")
            ]
            for filename in files:
                detector.pool.addJob([filename])
                time.sleep(3)

            logger.debug(f"Added {len(files)} jobs to the queue.")
        except KeyboardInterrupt:
            logger.info("Stopping TLE detector...")
        finally:
            detector.close()
    # python -m RMS.TLEDetector -m share/spritenet-maxpixel-v8-pretrained-best-fp16.tflite -c 0.432 -s 1 /home/pi/RMS_data/CapturedFiles/...
