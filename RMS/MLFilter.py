""" Filter records in FTDetectionInfo file by using machine learning, to avoid artefacts """
""" The old unfiltered file is renamed to _unfiltered' """
""" by Milan Kalina, 2022 """
""" based on https://github.com/fiachraf/meteorml """

#from __future__ import print_function, division, absolute_import

import argparse
import os
import sys
import math
import numpy as np
from PIL import Image
import traceback
from pathlib import Path
import time
import logging
import datetime
import shutil
from tflite_runtime.interpreter import Interpreter
from RMS.Formats import FFfile
from RMS.Formats import FTPdetectinfo
from RMS.Logger import initLogging
import RMS.ConfigReader as cr


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


def cust_predict(file_dir, model_path):
    interpreter = Interpreter(model_path)
    #input_details = interpreter.get_input_details()
    #output_details = interpreter.get_output_details()
    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    #print("Input Shape (", width, ",", height, ")")
    # PNG images are in the '1' subfolder
    #file_dir = file_dir + '/1/'
    prediction_list = []
    
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
    	
      # Classify the image and mesaure the time
      time1 = time.time()
      prob = classify_image(interpreter, image)
      time2 = time.time()
      classification_time = np.round(time2-time1, 3)
      #print(f'{prob:.3f}' + "\t" + Path(f).stem)
      # + "\t" + str(classification_time), " seconds.")
      prediction_list.append((f'{prob:.3f}', Path(f).stem))

    return prediction_list



def add_zeros_row(image, top_or_bottom, num_rows_to_add):
    """ adds rows of zeros to either the top or bottom of the numpy array
        if performance is important the same effect can be achieved using numpy slicing which is faster
    """
    image_shape = np.shape(image)
    #shape returns (num_rows, num_cols)
    num_rows = image_shape[0]
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
    num_cols = image_shape[1]
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
    Could also do squaring of images using keras.preprocessing module I just thought doing it this might yield better results as it wont be stretched or distored in a potentially non-uniform way
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
    meteor_num = detection_info[2]
    num_segments = detection_info[3]
    first_frame_info = detection_info[11][0]
    first_frame_no = first_frame_info[1]
    last_frame_info = detection_info[11][-1]
    last_frame_no = last_frame_info[1]

    try:
        #read the fits_file
        # print(f"fits_dir: {fits_dir}\nfits_file_name: {fits_file_name}")
        fits_file = FFfile.read(fits_dir, fits_file_name, fmt="fits")
        #image array with background set to 0 so detections stand out more
        #TODO inlcude code to use mask for the camera, currently masks not available on the data given to me, Fiachra Feehilly (2021)
        detect_only = fits_file.maxpixel - fits_file.avepixel
        #set image to only include frames where detection occurs, reduces likelihood that there will then be multiple detections in the same cropped image
        detect_only_frames = FFfile.selectFFFrames(detect_only, fits_file, first_frame_no, last_frame_no)

        #get size of the image
        row_size = detect_only_frames.shape[0]
        col_size = detect_only_frames.shape[1]

        #side 1, 2 are the left and right sides but still need to determine which is which
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

    except Exception as error:
        print(f"error: {traceback.format_exc()}")
        return None

def gen_pngs(FTP_path, FF_dir_path):

    os.chdir(FF_dir_path)

    #creating new directories for the png versions of ConfirmedFiles and RejectedFiles
    if "temp_png_dir" not in FF_dir_path:
        try:
            os.mkdir("temp_png_dir")
            os.mkdir("temp_png_dir/1")
        except:
            pass

    temp_png_dir = os.path.join(FF_dir_path, "temp_png_dir/1")
    #print(temp_png_dir)

    try:
        #read data from FTPdetectinfo file
        
        FTP_file = FTPdetectinfo.readFTPdetectinfo(os.path.dirname(FTP_path), os.path.basename(FTP_path))
        fits_file_list = os.listdir(FF_dir_path)

        #loop through each image entry in the FTPdetectinfo file and analyse each image
        for detection_entry in FTP_file:
            fits_file_name = detection_entry[0]
            meteor_num = detection_entry[2]

            if fits_file_name in fits_file_list:
                square_crop_image = crop_detections(detection_entry, FF_dir_path)

                #save the Numpy array as a png using PIL
                im = Image.fromarray(square_crop_image)
                im = im.convert("L")    #converts to grescale
                im.save(temp_png_dir + "/" + fits_file_name[:-5] + "_" + str(int(meteor_num)) + ".png")

            else:
                print(f"file: {fits_file_name} not found")

    except:
        print(traceback.format_exc())

    return(temp_png_dir)


def filterFTPdetectinfo(file_path):
    """ filters meteors from artefacts

    ARGUMENTS:
        file_path: path of FTPDetectinfo file
    
    RETURNS:
        unfiltered count, filtered count
    
    """

    model_path = os.getcwd() + '/share/meteorml32.tflite'
    # threshold for filtering meteor/non-meteor detections, depends on particular model properties
    ML_threshold = 0.95

    # gets file naame and dir path
    dir_path = os.path.split(file_path)[0]
    file_name = os.path.split(file_path)[1]
    
    config = cr.loadConfigFromDirectory('.config', os.path.abspath(dir_path))
    initLogging(config)
    # Get the logger handle
    log = logging.getLogger("logger")
    log.info("ML filtering starting...")
    
    # create cropped images from observations in FTPdetectinfo file
    log.info("Creating images for inference...")
    png_dir = gen_pngs(file_path, dir_path)
    
    # run inference and return propabilities along with file names
    log.info("Inference starting...")
    pred_list = cust_predict(os.path.join(png_dir), model_path)
    
    # remove PNG temporary files
    #shutil.rmtree(os.path.abspath(os.path.join(png_dir, os.pardir)))
    
    # load FTPdetectioninfo file containing unfiltered detections
    FTP = FTPdetectinfo.readFTPdetectinfo(dir_path, file_name, True)
    
    # simple check if both counts fit, so we can use the indexing
    if (len(pred_list) != len(FTP[2])):
    	print("Error! Detection count check failed")
    else:
    	exit
    	
    i = 0
    FTPFiltered = []
    for obs in FTP[2]:
      #print(i, obs[0], Path(obs[0]).stem)
      if (float(pred_list[i][0]) > ML_threshold):
      	print(Path(obs[0]).stem + "  " + str("{0:.1%}".format(float(pred_list[i][0]))) + "  identified as meteor...")
      	FTPFiltered.append(obs)
      else:
      	print(Path(obs[0]).stem + "  " + str("{0:.1%}".format(float(pred_list[i][0]))) + " identified as artefact...")
      i += 1
      
    log.info("Modifying FTPdetectinfo file for meteors only...")
    os.rename(file_path, os.path.join(dir_path, Path(file_name).stem) + '_unfiltered.txt')
 
    # save a new FTPdetectinfo file containing meteors only
    FTPdetectinfo.writeFTPdetectinfo(meteor_list=FTPFiltered, ff_directory=dir_path, file_name=file_name, cal_directory='', cam_code=config.stationID, fps=config.fps, calibration="Filtered by RMS on: " + str(datetime.datetime.now()), celestial_coords_given=True)
    log.info("FTPdetectinfo modified, excluded " + str(len(FTP[2])-len(FTPFiltered)) + "/" + str(len(FTP[2])) + " records as artefacts")
    
    return(len(FTPFiltered), len(FTP[2]))
    
    
    
if __name__ == "__main__":
      
        # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Reads and filters meteors from FTPdetectInfo file.")
    arg_parser.add_argument('file_path', nargs=1, metavar='FILE_PATH', type=str, \
        help='Path to the FTPDetectInfo file.')

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()
    
    #########################
    
    # ML model file path
    model_path = os.getcwd() + '/share/meteorml32.tflite'
    # threshold for filtering meteor/non-meteor detections, depends on particular model properties
    ML_threshold = 0.95
    
    #########################
    
    file_path = os.path.abspath(cml_args.file_path[0])
    filterFTPdetectinfo(file_path)
    
