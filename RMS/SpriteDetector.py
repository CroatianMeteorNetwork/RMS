import os
import shutil
from PIL import ImageDraw, Image
import numpy as np
import logging
from RMS.ExtractThumbnails import get_thumbnails, apply_vignetting
import tarfile
from datetime import datetime
import statistics
from RMS.Routines import MaskImage
from RMS.Formats.CALSTARS import readCALSTARS
from RMS.Formats.FFfits import read as readFFfile

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

DEBUG_MODEL_PATH = "/mnt/1tb/Documents/Astronomija/GMN/dev/SpriteNet/results/train/spriteNetv5-maxpix_pretrained/weights/best-fp16.tflite"

log = logging.getLogger("logger")


# taken from yolov5/utils/general.py
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


class SpriteDetector(object):
    def __init__(
        self,
        folder_path,
        model_path,
        conf_thres,
        config,
        disable_mask,
        min_stars,
        vignetting_parameter,
        thumbnails_only,
        max_fits_threshold,
    ):

        self.min_stars = min_stars
        self.folder_path = folder_path
        self.model_path = model_path
        self.config = config
        self.thumbnails_only = thumbnails_only
        self.vignetting_parameter = vignetting_parameter

        self.iou_thres = 0.1
        self.max_det = 4
        self.conf_thres = conf_thres
        self.max_fits_threshold = max_fits_threshold

        self.save_dir = os.path.join(
            config.data_dir, "SpriteFiles", os.path.basename(folder_path)
        )

        self.interpreter, self.input_details, self.output_details = (
            self.init_interpreter(model_path)
        )
        if disable_mask:
            self.mask = None
            print("Masking disabled, using original images")
        else:
            self.mask = load_mask(config)
            if self.mask is None:
                print("No mask file found")
        self.calstars = readCALSTARS(
            self.folder_path, "CALSTARS_" + os.path.basename(self.folder_path) + ".txt"
        )
        if self.calstars:
            self.calstars = self.calstars[0]
        else:
            print("No CALSTARS file found")
        thumbnail_file = os.path.join(
            self.folder_path,
            os.path.basename(self.folder_path) + "_CAPTURED_thumbs.jpg",
        )
        if os.path.exists(thumbnail_file):
            self.swipe_thumbnails(thumbnail_file, vignetting_parameter)
        else:
            print(
                f"Thumbnail file {thumbnail_file} not found. Skipping sprite detection."
            )

    def init_interpreter(self, model_path):
        interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        print(input_details["shape"], input_details["dtype"])
        print(output_details["shape"])
        return interpreter, input_details, output_details

    def swipe_thumbnails(self, thumbnail_file, vignetting_parameter):
        if os.path.exists(thumbnail_file):
            for thumbnail, thumbnail_name, subfolder_path in get_thumbnails(
                vignetting_parameter, thumbnail_file
            ):
                prediction, image = self.get_prediction(thumbnail)

                # process(prediction, image, subfolder_path, thumbnail_name, save=True)
                output = self.process_predictions(
                    prediction,
                    thumbnail_name,
                )
                # sprites candidates were found
                if output.shape[0] > 0:
                    self.filter_detections(
                        image,
                        thumbnail_name,
                        subfolder_path,
                        output,
                        True,
                    )

    def get_prediction(
        self,
        thumbnail,
    ):
        # remove known camera obstructions
        if self.mask is not None:
            if np.array(thumbnail).shape != self.mask.img.shape:
                # print(
                #    "Mask and image size do not match",
                #    np.array(thumbnail).shape,
                #    self.mask.img.shape,
                # )
                # mask_img = np.ascontiguousarray(self.mask.img)
                # mask = Image.fromarray(mask_img).resize((thumbnail.width, thumbnail.height))

                mask = np.resize(self.mask.img, (thumbnail.height, thumbnail.width, 3))
                # mask = mask.resize((thumbnail.width, thumbnail.height))

                # if len(np.array(thumbnail).shape) == 3:
                # mask = mask.convert("RGB")

                # print("Trying resize...",np.array(thumbnail).shape,
                #    mask.shape,)
                image = Image.fromarray(
                    MaskImage.maskImage(np.array(thumbnail), mask, True)
                )
            else:

                # print("Masking image")
                image = Image.fromarray(
                    MaskImage.maskImage(np.array(thumbnail), self.mask)
                )
        else:
            image = thumbnail  # .convert("RGB") already done in get_thumbnails

        input_shape = self.input_details["shape"]
        image = image.resize((input_shape[1], input_shape[2]))
        input_data = np.array(image, dtype=np.float32)

        # taken from run function in yolov5/detect.py
        input_data /= 255
        if len(input_data.shape) == 3:
            input_data = input_data[None]  # expand for batch dim

        # Set the tensor to point to the input data to be inferred
        self.interpreter.set_tensor(self.input_details["index"], input_data)

        # Run the inference
        self.interpreter.invoke()

        # Get the output tensor
        prediction = self.interpreter.get_tensor(self.output_details["index"])
        return prediction, image

    def process_predictions(
        self,
        prediction,
        imgname,
    ):
        max_nms = 30000  # upper limit for number of boxes before nms
        # max_wh = 7680 maximum box width/height

        xc = prediction[..., 4] > self.conf_thres  # detection candidates mask
        # output = np.zeros((max_det, 5))

        # we can later vectorize prediction so it processes all images at once
        for xi, x in enumerate(prediction):
            x = x[xc[xi]]  # detection candidates
            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
            # calculate box
            box = xywh2xyxy(x[:, :4])
            # find highest confidence among all classes
            conf = np.max(x[:, 5:6], 1, keepdims=True)
            # j=np.argmax(x[:, 5:6], 1, keepdims=True)
            # merge results into one array and filter candidates
            x = np.concatenate((box, conf), 1)[np.reshape(conf, -1) > self.conf_thres]
            if not x.shape[0]:
                continue

            print()
            print(imgname)
            # print("Number of initial boxes:", x.shape[0])
            # sort by confidence and remove excess boxes
            x = x[np.argsort(x[:, 4])[::-1][:max_nms]]
            # print("Pre-NMS:", x)
            # classes (only 1 used here),c=0
            # c = x[:, 5:6] * max_wh
            # boxes (offset by class), scores
            x = x[:, :5]
            # boxes, scores = x[:, :4] + c, x[:, 4]
            # non-max suppression
            i = nms(x, self.iou_thres)
            # print("Post-NMS:", i)

            # limit detections
            if self.max_det > 0:
                output = x[i][: self.max_det]
            else:
                output = x[i]

            # the output is in the format [x1, y1, x2, y2, conf]
            # x1, y1 is the top left corner, x2, y2 is the bottom right corner
            # values are normalized to the image size (0-1)
            # 0,0 is upper left corner
            print("Output:", output)
            return output
        # no detections found, return empty array
        return np.zeros((0))

    def filter_detections(
        self,
        image,
        imgname,
        folder_path,
        output,
        save,
    ):
        imgname, stack_files = self.get_timestamp(folder_path, imgname)
        if stack_files is None:  # cant determine image timestamp so were skipping it
            # probably empty part of the last thumb row
            print("Can't determine image timestamp")
            return
        print("Determined timestamp:", imgname)
        if self.calstars:
            ff_stars = []
            ff_to_process=[]
            # print(self.calstars[0][0],stack_files)
            print("Checking number of stars for", stack_files)
            print("Example FF name from CALSTARS:", self.calstars[0][0])
            for ff in self.calstars:
                # print(ff[0],len(ff[1]),stack_files)
                for file in stack_files:
                    if file in ff[0]:
                        ff_stars.append(len(ff[1]))
                        ff_to_process.append(ff[0])
                        break
            print("Number of stars per FF:", ff_stars)
            if ff_stars:
                print("Median stars", statistics.median(ff_stars))
            if not ff_stars or statistics.median(ff_stars) < self.min_stars:
                print("Not enough stars in the images")
                return
        #print("Keeping detection")
        if self.thumbnails_only:
            print("Storing thumbnail detection")
            self.store_detections(image, folder_path, output, save, imgname)
        else:
            print("Analyzing fits files")
            ff_found = self.analyze_fits(ff_to_process, save, folder_path)
            if not ff_found:
                print("Saving thumbnail since fits arent available.")
                self.store_detections(image, folder_path, output, save, imgname)

    def analyze_fits(self, stack_files, save, folder_path):
        detections = []  # here we store detections for each fits file
        ff_found = False
        ff_names_with_detections = []
        for ff_name in stack_files:
            # dirname of folder_path is the main root folder of the night
            try:
                maxpixel = readFFfile(self.folder_path, ff_name).maxpixel
                ff_found = True
            except FileNotFoundError:
                # print(f"File {ff_name} not found in {self.folder_path}. Skipping.")
                continue
            maxpixel_vignetting_corrected = apply_vignetting(
                maxpixel, self.vignetting_parameter
            ).convert("RGB")
            prediction, image = self.get_prediction(maxpixel_vignetting_corrected)
            output = self.process_predictions(
                prediction, ff_name  # os.path.splitext(ff_name)[0] + "_sprite"
            )
            if output.shape[0] > 0:
                ff_names_with_detections.append(ff_name)
                detections.append((output, image))
            else:
                print(f"No detections in {ff_name}.")

        print("Number of FFs with detections:", len(detections))
        if len(detections)== 0:
            print("Ignoring detection.")
            return ff_found
        if len(detections) <= self.max_fits_threshold:
            print("Saving FFs with detections")
            # we can save them
            for i in range(len(detections)):
                output, image = detections[i]
                ff_name = ff_names_with_detections[i]
                os.makedirs(os.path.join(self.save_dir, "FFs"), exist_ok=True)
                shutil.copy(
                    os.path.join(self.folder_path, ff_name),
                    os.path.join(self.save_dir, "FFs", ff_name),
                )
                self.store_detections(
                    image,
                    folder_path,
                    output,
                    save,
                    ff_name,
                    # os.path.splitext(ff_name)[0] + "_sprite",
                )
        else:
            print(f"Too many detections. Ditching the detections.")
            # we can skip saving them, too many detections
        return ff_found

    def store_detections(self, image, folder_path, output, save, imgname):
        self.mark_sprites(output, image, imgname, save)

        with open(
            os.path.join(self.save_dir, "detections.csv"), "a", newline=""
        ) as csvfile:
            writer = csv.writer(
                csvfile, delimiter=";", quotechar="|", quoting=csv.QUOTE_MINIMAL
            )
            writer.writerow(["image name", "detection type","upper left x","upper left y","bottom right x","bottom right y","confidence"])
            for i in output:
                writer.writerow([imgname,"sprite", i[0]*self.config.width, i[1]*self.config.height, i[2]*self.config.width, i[3]*self.config.height, i[4]])

        f = open(os.path.join(folder_path, "detections.txt"), "a")
        f.write(f"{imgname}\n")
        for i in output:
            f.write(f"{i[0]},{i[1]},{i[2]},{i[3]},{i[4]}\n")
        f.write("\n")
        f.close()

    def get_timestamp(self, folder_path, imgname):
        """
        Find the timestamp in a specific file from a FS_*.tar.bz2 archive

        Args:
            folder_path (str): Path to the folder with thumbnails
            imgname (str): Name of the image without extension
        """

        # Get parent folder (directory containing the folder_path)
        parent_folder = os.path.dirname(folder_path)

        # Extract info from imgname (assuming imgname format contains station code
        code_and_date = imgname[:15]
        thumb_index = int(imgname.split("_")[-1])

        # Find all .tar.bz2 files in parent folder that match pattern
        archive_file = ""
        for filename in sorted(os.listdir(parent_folder)):
            if filename.endswith(".tar.bz2") and filename.startswith(
                f"FS_{code_and_date}"
            ):
                archive_file = filename
                break

        if archive_file == "":
            print(f"No matching archives found for {code_and_date} in {parent_folder}")
            return imgname

        # Extract the timestamp from the appropriate file in the archive
        archive_path = os.path.join(parent_folder, archive_file)
        FF_FILES_IN_THUMB = 5  # config.thumb_stack
        try:
            with tarfile.open(archive_path, "r:bz2") as tar:
                # Look through archive files, first element is "." so it is ommitted
                files = sorted(
                    tar.getmembers()[1:],
                    key=lambda x: datetime.strptime(x.name[12:27], "%Y%m%d_%H%M%S"),
                )

                start_index = (thumb_index - 1) * FF_FILES_IN_THUMB
                if start_index < len(files):
                    stack_files = []
                    for j in range(
                        start_index, min(start_index + FF_FILES_IN_THUMB, len(files))
                    ):
                        stack_files.append("FF_" + files[j].name[5:31])
                    return (
                        files[start_index].name[5:27] + "_thumbnail" + str(thumb_index),
                        stack_files,
                    )

        except Exception as e:
            print(f"Error reading archive {archive_file}: {e}")
            return imgname, None

        print(f"No timestamp found for {imgname}")
        return imgname, None

    def mark_sprites(self, output, image, imgname, save=True):
        edit_image = image.copy()
        draw = ImageDraw.Draw(edit_image)
        # Draw the rectangle
        width, height = edit_image.size
        for i in range(output.shape[0]):
            top_left = (output[i, 0] * width, output[i, 1] * height)
            bottom_right = (output[i, 2] * width, output[i, 3] * height)
            draw.rectangle([top_left, bottom_right], outline="red", width=1)
            # Display the number above the rectangle
            number = str(round(output[i, 4], 3))
            text_position = (
                top_left[0],
                top_left[1] - 15,
            )  # Adjust the position as needed
            draw.text(text_position, number, fill="red")

        # Save the modified image
        if save:
            MARKED_DIR = os.path.join(self.save_dir, "marked")
            os.makedirs(MARKED_DIR, exist_ok=True)
            edit_image.save(f'{os.path.join(MARKED_DIR,imgname+"_marked")}.png')
            #its useful to ahve unmarked ones since they can be used in model training
            UNMARKED_DIR = os.path.join(self.save_dir, "unmarked")
            os.makedirs(UNMARKED_DIR, exist_ok=True)
            image.save(f'{os.path.join(UNMARKED_DIR,imgname+"_unmarked")}.png')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run sprite detection on FITS files")
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
        default=0.001,
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
    parser.add_argument(
        "--vignetting",
        "-v",
        type=float,
        default=0.0009,
        help="Vignetting coefficient (default: %(default)s)",
    )
    parser.add_argument(
        "--thumbnails-only",
        "-t",
        action="store_true",
        help="Processes only thumbnails and doesn't continue with fits files.",
    )
    parser.add_argument(
        "--max-fits-threshold",
        "-f",
        type=int,
        default=5,
        help="Maximum allowed number of .fits files (from a single thumbnail) with detections (default: %(default)s). If more than this number of .fits files is detected, the detections are considered false positives.",
    )

    args = parser.parse_args()

    import RMS.ConfigReader as cr

    # Load the configuration file
    config = cr.parse(".config")

    if not TFLITE_AVAILABLE:
        log.warning(
            "TensorFlow Lite is not available on this system. Sprite detection skipped..."
        )
    else:
        SpriteDetector(
            folder_path=args.folder_path,
            model_path=args.model,
            conf_thres=args.confidence,
            config=config,
            disable_mask=args.disable_mask,
            min_stars=args.star_threshold,
            vignetting_parameter=args.vignetting,
            thumbnails_only=args.thumbnails_only,
            max_fits_threshold=args.max_fits_threshold,
        )
    # example: python -m RMS.thumbs_detection -m /mnt/1tb/Documents/Astronomija/GMN/dev/SpriteNet/results/train/spritenet-maxpixel-v7-pretrained-yolov5/weights/best-fp16.tflite -c 0.455 -s 0 /mnt/1tb/Documents/Astronomija/GMN/dev/hr002k/HR002K_20250411_181455_674301
    # python -m RMS.SpriteDetector -m share/spritenet-maxpixel-v8-pretrained-best-fp16.tflite -c 0.432 -s 1 -v 0.001 -f 3 /home/pi/RMS_data/CapturedFiles/...
