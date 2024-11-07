""" 
    Filter records in FTDetectionInfo file by using machine learning, to avoid artefacts
    The old unfiltered file is renamed to _unfiltered'
    by Milan Kalina, 2022
    based on https://github.com/fiachraf/meteorml 
"""

from __future__ import print_function, division, absolute_import, unicode_literals

import argparse
import os
import math
import numpy as np
from PIL import Image
import traceback
#import time
import logging
import datetime
import shutil

TFLITE_AVAILABLE = False
USING_FULL_TF = False

try:
    from tflite_runtime.interpreter import Interpreter
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        from tensorflow.lite.python.interpreter import Interpreter
        TFLITE_AVAILABLE = True
        USING_FULL_TF = True
    except ImportError:
        TFLITE_AVAILABLE = False


from RMS.Formats import FFfile
from RMS.Formats import FTPdetectinfo
from RMS.Logger import initLogging
import RMS.ConfigReader as cr

# Get the logger from the main module
log = logging.getLogger("logger")


# Suffix for unfiltered FTPdetectinfo files
FTPDETECTINFO_UNFILTERED_SUFFIX = '_unfiltered.txt'


def standardize_me1(a, axis=None): 
    # axis param denotes axes along which mean & std reductions are to be performed
    mean = np.mean(a, axis=axis, keepdims=True)
    std = np.sqrt(((a - mean)**2).mean(axis=axis, keepdims=True))
    return (a - mean) / std


# make it between 0 and 1    
def rescale_me1(image):
    image = np.array(image / 255., dtype=np.float32)
    return image


# normalize between min and max    
def normalize_me1(image):
    #image = (image - np.amin(image)) / (np.amax(image) - np.amin(image))
    image *= 1/image.max()
    return image


def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    # fit the model input dimensions
    image = np.expand_dims(image, axis=2)
    input_tensor[:, :] = image    


# main classification part, for details see e.g.: https://www.tensorflow.org/lite/guide/inference
# currently we are using float32 Tensorflow Lite model with no quantization
def classify_image(interpreter, image, top_k=1):
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))
    scale, zero_point = output_details['quantization']
    #output = scale * (output.astype(np.float32) - zero_point)
    # no quantization therefore scale is not used
    output = output.astype(np.float32) - zero_point
    return output    


def classifyPNGs(file_dir, model_path):
    """ Given a directory with PNG files of meteors, classify them into meteors vs artefacts using the ML 
        model. 
    """

    # Load the model
    interpreter = Interpreter(model_path)
    #input_details = interpreter.get_input_details()
    #output_details = interpreter.get_output_details()

    # Define the input image parameters
    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']

    #print("Input Shape (", width, ",", height, ")")
    # PNG images are in the '1' subfolder
    #file_dir = file_dir + '/1/'


    # Run predictions for every meteor detection
    prediction_dict = {}
    for f in sorted(os.listdir(file_dir)):

        image = Image.open(os.path.join(file_dir, f))

        # rescale image size to fit the model 32x32 pixels input
        image = image.resize((width, height))

        # convert image to numpy array
        image = np.asarray(image, dtype=np.float32)

        # rescale values to (0;1)
        image = rescale_me1(image)

        # normalize min and max values
        image = normalize_me1(image)

        #image = standardize_me1(image)
        
        # Classify the image and measure the time
        #time1 = time.time()
        prob = classify_image(interpreter, image)
        #time2 = time.time()
        #classification_time = np.round(time2-time1, 3)
        #print("{:.3f}".format(prob) + "\t" + os.path.splitext(os.path.basename(f))[0])
        # + "\t" + str(classification_time), " seconds.")

        # Save the file name and the predicted classification probability
        prediction_dict[os.path.splitext(os.path.basename(f))[0]] = prob


    return prediction_dict



def add_zeros_row(image, top_or_bottom, num_rows_to_add):
    """ adds rows of zeros to either the top or bottom of the numpy array
        if performance is important the same effect can be achieved using numpy slicing which is faster
    """
    image_shape = np.shape(image)
    #shape returns (num_rows, num_cols)
    #num_rows = image_shape[0]
    num_cols = image_shape[1]

    zero_rows = np.zeros((num_rows_to_add, num_cols))

    if top_or_bottom == "top":
        new_image = np.vstack((zero_rows, image))
        return new_image
    elif top_or_bottom == "bottom":
        new_image = np.vstack((image, zero_rows))
        return new_image
    #return None which will cause an error if invalid inputs have been used
    return


def add_zeros_col(image, left_or_right, num__cols_to_add):
    """ adds columns of zeros to either the left or right of the numpy array
        if performance is important the same effect can be achieved using numpy slicing which is faster
    """
    image_shape = np.shape(image)
    #shape returns (num_rows, num_cols)
    num_rows = image_shape[0]
    #num_cols = image_shape[1]
    zero_cols = np.zeros((num_rows, num__cols_to_add))


    if left_or_right == "left":
        new_image = np.hstack((zero_cols, image))
        return new_image
    elif left_or_right == "right":
        new_image = np.hstack((image, zero_cols))
        return new_image
    #return None which will cause an error if invalid inputs have been used
    return


def blackfill(image, leftover_top=0, leftover_bottom=0, leftover_left=0, leftover_right=0):
    """ will make the image square by adding rows or columns of black pixels to the image, as the image needs to be square to be fed into a Convolutional Neural Network(CNN)

    As I am giving the cropped images +20 pixels on all sides from the detection square edges, any meteors that occur on the edge of the frame in the fits file will not be centered so I am using the leftover terms, to add columns or rows for the edges that were cut off so that the meteor is centered in the image. This might badly affect the data and thus the CNN so it might need to be changed. However I think it might help as the added space around the meteor I hope will make sure that it includes the entire meteor trail which can hopefully help differentiate true detections from false detections

    When squaring the image it will also keep the meteor detection in roughly the center

    Resizing of the images to be all the same size will be done in another script using the keras.preprocessing module
    Could also do squaring of images using keras.preprocessing module I just thought doing it this might yield better results as it wont be stretched or distorted in a potentially non-uniform way
    """

    if leftover_top > 0:
        image = add_zeros_row(image, "top", leftover_top)
    if leftover_bottom > 0:
        image = add_zeros_row(image, "bottom", leftover_bottom)
    if leftover_left > 0:
        image = add_zeros_col(image, "left", leftover_left)
    if leftover_right > 0:
        image = add_zeros_col(image, "right", leftover_right)

    if np.shape(image)[0] == np.shape(image)[1]:
        new_image = image[:,:]
        return new_image

    if np.shape(image)[0] < np.shape(image)[1]:
        rows_needed = np.shape(image)[1] - np.shape(image)[0]
        image = add_zeros_row(image, "top", math.floor(rows_needed/2))
        image = add_zeros_row(image, "bottom", math.ceil(rows_needed/2))
        new_image = image[:,:]
        return new_image

    if np.shape(image)[1] < np.shape(image)[0]:
        cols_needed = np.shape(image)[0] - np.shape(image)[1]
        image = add_zeros_col(image, "left", math.floor(cols_needed/2))
        image = add_zeros_col(image, "right", math.ceil(cols_needed/2))
        new_image = image[:,:]
        return new_image

    return


def crop_detections(detection_info, fits_dir):
    """
    crops the detection from the fits file using the information provided from the FTPdetectinfo files
    detection_info is a single element of the list returned by the RMS.RMS.Formats.FTPdetectinfo.readFTPdetectinfo() function. This list contains only information on a single detection
    fits_dir is the the directory where the fits file is located

    returns the cropped image as a Numpy array
    """

    fits_file_name = detection_info[0]
    #meteor_num = detection_info[2]
    #num_segments = detection_info[3]
    first_frame_info = detection_info[11][0]
    first_frame_no = first_frame_info[1]
    last_frame_info = detection_info[11][-1]
    last_frame_no = last_frame_info[1]

    try:
        # Read the fits_file
        # print("fits_dir:", fits_dir)
        # print("fits_file_name:", fits_file_name)
        fits_file = FFfile.read(fits_dir, fits_file_name, fmt="fits")

        # image array with background set to 0 so detections stand out more
        # TODO include code to use mask for the camera, currently masks not available on the data given to me, Fiachra Feehilly (2021)
        detect_only = fits_file.maxpixel - fits_file.avepixel

        # set image to only include frames where detection occurs, reduces likelihood that there will then be multiple detections in the same cropped image
        detect_only_frames = FFfile.selectFFFrames(detect_only, fits_file, first_frame_no, last_frame_no)

        # get size of the image
        row_size = detect_only_frames.shape[0]
        col_size = detect_only_frames.shape[1]

        # side 1, 2 are the left and right sides but still need to determine which is which
        # left side will be the lesser value as the value represents column number
        side_1 = first_frame_info[2]
        side_2 = last_frame_info[2]
        if side_1 > side_2:
            right_side = math.ceil(side_1) + 1 #rounds up and adds 1 to deal with Python slicing so that it includes everything rather than cutting off the last column
            left_side = math.floor(side_2)
        else:
            left_side = math.floor(side_1)
            right_side = math.ceil(side_2) + 1
        #side 3 and 4 are the top and bottom sides but still need to determine which is which
        # bottom side will be the higher value as the value represents the row number
        side_3 = first_frame_info[3]
        side_4 = last_frame_info[3]
        if side_3 > side_4:
            bottom_side = math.ceil(side_3) + 1
            top_side = math.floor(side_4)
        else:
            top_side = math.floor(side_3)
            bottom_side = math.ceil(side_4) + 1

        #add some space around the meteor detection so that its not touching the edges
        #leftover terms need to be set to 0 outside if statements otherwise they wont be set if there's nothing left over which will cause an error with the blackfill.blackfill() line
        left_side = left_side - 20
        leftover_left = 0
        if left_side < 0:
            #this will be used later to determine how to fill in the rest of the image to make it square but also have the meteor centered in the image
            leftover_left = 0 - left_side
            left_side = 0

        right_side = right_side + 20
        leftover_right = 0
        if right_side > col_size:
            leftover_right = right_side - col_size
            right_side = col_size

        top_side = top_side - 20
        leftover_top = 0
        if top_side < 0:
            leftover_top = 0 - top_side
            top_side = 0

        bottom_side = bottom_side + 20
        leftover_bottom = 0
        if bottom_side > row_size:
            leftover_bottom = bottom_side - row_size
            bottom_side = row_size


        #get cropped image of the meteor detection
        #first index set is for row selection, second index set is for column selection
        crop_image = detect_only_frames[top_side:bottom_side, left_side:right_side]
        square_crop_image = blackfill(crop_image, leftover_top, leftover_bottom, leftover_left, leftover_right)

        return square_crop_image

    except Exception:
        print("error: ", traceback.format_exc())
        return None



def makePNGname(fits_file_name, meteor_num):
    """ Make a PNG name unique for each detection. """

    return fits_file_name.strip('.fits').strip('.bin') + "_" + str(int(meteor_num))



def makePNGCrops(FTP_path, FF_dir_path):
    """ Take the FTPdetectinfo file and the FF files in the directory, and make PNG crops centered around
        the detection. These will be fed into the ML algorithm.
    """

    #os.chdir(FF_dir_path)

    #creating new directories for the png versions of ConfirmedFiles and RejectedFiles
    if "temp_png_dir" not in FF_dir_path:
        try:
            os.mkdir(os.path.join(FF_dir_path, "temp_png_dir"))
            os.mkdir(os.path.join(FF_dir_path, "temp_png_dir/1"))
        except:
            pass

    temp_png_dir = os.path.join(FF_dir_path, "temp_png_dir/1")
    #print(temp_png_dir)

    try:
        #read data from FTPdetectinfo file
        
        meteor_list = FTPdetectinfo.readFTPdetectinfo(os.path.dirname(FTP_path), os.path.basename(FTP_path))
        fits_file_list = os.listdir(FF_dir_path)

        #loop through each image entry in the FTPdetectinfo file and analyse each image
        for detection_entry in meteor_list:

            # Read FTPdetectinfo name and meteor number
            fits_file_name = detection_entry[0]
            meteor_num = detection_entry[2]

            # Make the output PNG name
            png_name = makePNGname(fits_file_name, meteor_num)

            # If the FF file is found in the directory, make a PNG cutout
            if fits_file_name in fits_file_list:

                square_crop_image = crop_detections(detection_entry, FF_dir_path)

                #save the Numpy array as a png using PIL
                im = Image.fromarray(square_crop_image)
                im = im.convert("L")    #converts to grayscale
                im.save(os.path.join(temp_png_dir, png_name + ".png"))

            else:
                print("file:", fits_file_name, " not found")

    except:
        print(traceback.format_exc())

    return temp_png_dir



def filterFTPdetectinfoML(config, ftpdetectinfo_path, threshold=0.85, keep_pngs=False, clear_prev_run=False):
    """ Using machine learning, reject false positives and only keep real meteors. An updated FTPdetectinfo
        file will be saved.  

    Arguments:  
        config: [object] RMS config object  
        ftpdetectinfo_path: [str] Path of FTPDetectinfo file.  

    Keyword arguments:  
        threshold: [float] Threshold meteor/non-meteor classification (0-1 range).  
        keep_pngs: [bool] Whether to keep or delete the temporary PNGs.  
        clear_prev_run: [bool] whether or not to remove any previous run's data - required when reprocessing a folder.   
    
    Return:  
        unfiltered count, filtered count  
    
    """


    # Get file name and dir path
    dir_path, file_name = os.path.split(ftpdetectinfo_path)


    log.info("ML filtering starting...")


    # Check if the module has already been run (the _unfiltered file already exists)
    unfiltered_name = os.path.splitext(os.path.basename(file_name))[0] + FTPDETECTINFO_UNFILTERED_SUFFIX
    orig_name = file_name

    if os.path.isfile(os.path.join(dir_path, unfiltered_name)):

        # if we're reprocessing a folder after config or plate changes we need to remove the 
        # unfiltered file so that the ML routine processes the new FTPdetect file.
        if clear_prev_run is True:
            os.remove(os.path.join(dir_path, unfiltered_name))
            log.info("Removing previous run data " + ftpdetectinfo_path)

        else:
            # If we are reprocessing with a different threshold 
            # we want to reprocess the original unfiltered data.
            ftpdetectinfo_path = os.path.join(dir_path, unfiltered_name)
            file_name = unfiltered_name

            log.info("Module was previously run, using the original unfiltered FTPdetect file: " \
                + ftpdetectinfo_path)
    
    
    # Load the appropriate FTPdetectioninfo file containing unfiltered detections
    cam_code, fps, meteor_list = FTPdetectinfo.readFTPdetectinfo(dir_path, file_name, ret_input_format=True)


    # Check if tflite is available
    if TFLITE_AVAILABLE:
        log.info("TensorFlow Lite is available.")
        if USING_FULL_TF:
            log.info("Using TensorFlow Lite from full TensorFlow package.")
        else:
            log.info("Using standalone tflite_runtime package.")
    else:
        log.warning("The package tflite_runtime is not installed! This package is not available on Python 2.")
        log.warning("ML filtering skipped...")

        # Return a full list of FF files
        return [meteor_entry[0] for meteor_entry in meteor_list]



    # Create cropped images from observations in FTPdetectinfo file
    log.info("Creating images for inference...")
    png_dir = makePNGCrops(ftpdetectinfo_path, dir_path)
    
    # Run inference and return probabilities along with file names
    log.info("Inference starting...")
    prediction_dict = classifyPNGs(png_dir, config.ml_model_path)
    

    # create list of PNG images to be moved into subdirs later on
    png_list = []
    for f in sorted(os.listdir(png_dir)):
        png_list.append(f)


    # Create meteor/artefact subdirs for easier manual confirmation
    if keep_pngs:

        meteors_dir = os.path.abspath(os.path.join(png_dir, os.pardir, 'meteors'))
        artefacts_dir = os.path.abspath(os.path.join(png_dir, os.pardir, 'artefacts'))

        if not os.path.exists(meteors_dir):
            os.mkdir(meteors_dir)

        if not os.path.exists(artefacts_dir):
            os.mkdir(artefacts_dir)

    # Otherwise remove any PNG directories that might already exist
    else:
        shutil.rmtree(os.path.abspath(os.path.join(png_dir, os.pardir)))


    # Generate a list of FF files in the data directory
    ff_list = [ff_name for ff_name in sorted(os.listdir(dir_path)) if FFfile.validFFName(ff_name)]

    # Main filtering code
    ff_filtered = []
    ftp_filtered = []
    for meteor_entry in meteor_list:


        # Get the PNG name of the meteor detection
        png_name = makePNGname(meteor_entry[0], meteor_entry[1])


        # If the detection doesn't have a local FF file, keep it
        if meteor_entry[0] not in ff_list:

            ftp_filtered.append(meteor_entry)
            ff_filtered.append(meteor_entry[0])

            log.info("A local FF file not found, keeping the detection {:s}".format(png_name))

            continue


        # If the FF file is not classified as a meteor list, skip it
        if png_name not in prediction_dict:
            continue


        # Extract the prediction score
        pred_score = prediction_dict[png_name]


        # If the ML model thinks the detection is a meteor, keep it
        if pred_score > threshold:

            status_str = "meteor"

            # Add the FF file to the filtered list
            ftp_filtered.append(meteor_entry)
            ff_filtered.append(meteor_entry[0])

            if keep_pngs:
                keep_png_dir = meteors_dir

        else:
            status_str = "artefact"

            if keep_pngs:
                keep_png_dir = artefacts_dir


        log.info(png_name + " - " + "Score: {:6.1%} - {:s}".format(pred_score, status_str))


        # Sort into PNG dir, if they are kept
        if keep_pngs:
            os.replace(
                os.path.join(png_dir, png_name + '.png'), 
                os.path.join(
                    keep_png_dir, 
                    os.path.splitext(os.path.basename(png_name))[0] \
                        + '_p-{:.3f}'.format(prediction_dict[png_name]) + '.png'
                    )
                )
    

    # Backup the original unfiltered file if it doesn't exist
    if not os.path.isfile(os.path.join(dir_path, unfiltered_name)):

        shutil.copy2(ftpdetectinfo_path, os.path.join(dir_path, unfiltered_name))


    log.info("Saving a filtered FTPdetectinfo file...")



    # Save a new FTPdetectinfo file containing meteors only
    FTPdetectinfo.writeFTPdetectinfo(
        meteor_list=ftp_filtered, ff_directory=dir_path, file_name=orig_name, cal_directory='', \
        cam_code=cam_code, fps=fps, 
        calibration="Filtered by RMS on: " + str(datetime.datetime.now()), celestial_coords_given=True)
        

    log.info("FTPdetectinfo filtered, {:d}/{:d} detections classified as real meteors".format(
                         len(ftp_filtered), len(meteor_list)
        )
    )

    
    return ff_filtered
     
    
if __name__ == "__main__":
      
    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Reads and filters meteors from FTPdetectInfo file.")

    arg_parser.add_argument('ftpdetectinfo_path', metavar='FILE_PATH', type=str,
        help='Path to the FTPDetectInfo file.')

    arg_parser.add_argument('--threshold', '-t', metavar='THRESHOLD', type=float,
        help='threshold for meteor/non-meteor classification', default=0.85)

    arg_parser.add_argument('--keep_pngs', '-p', action="store_true",
        help='Keep the temporary PNG crops on which the ML filter is run pngs. They will be deleted by default.')

    arg_parser.add_argument('--clear_prev_run', '-r', action="store_true",
        help='Remove the files created by the last run. Required if reprocessing after a plate change.\n' \
            'However, if you are reprocessing the same data with a different threshold, set to False')

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str,
                            help="Path to a config file which will be used instead of the default one."
                            " To load the .config file in the given data directory, write '.' (dot).")


    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    
    #########################

    # Read the config file
    dir_path = os.path.dirname(cml_args.ftpdetectinfo_path)
    config = cr.loadConfigFromDirectory(cml_args.config, dir_path)

    ### Init the logger
    initLogging(config, 'reprocess_')
    log = logging.getLogger("logger")

    #########################


    # Run ML filtering
    ftpdetectinfo_path = os.path.abspath(cml_args.ftpdetectinfo_path)
    filterFTPdetectinfoML(config, ftpdetectinfo_path, cml_args.threshold, cml_args.keep_pngs, cml_args.clear_prev_run)
