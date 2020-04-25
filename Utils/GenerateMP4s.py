""" Generate an High Quality MP4 movie from FF files. 
    Contributors: Tioga Gulon
"""

from __future__ import print_function, division, absolute_import

import sys
import os
import platform
import argparse
import subprocess
import shutil
import datetime
import cv2
import Utils.FFtoFrames as f2f

from PIL import ImageFont

from RMS.Formats.FFfile import read as readFF
from RMS.Formats.FFfile import validFFName, filenameToDatetime
from RMS.Misc import mkdirP

def GenerateMP4s(dir_path):
    t1 = datetime.datetime.utcnow()

    # Load the font for labeling
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 18)
    except:
        font = ImageFont.load_default()
    
    print("Preparing files for the timelapse...")

    ff_list = [ff_name for ff_name in sorted(os.listdir(dir_path)) if validFFName(ff_name)]
    for file_name in ff_list:

        # Read the FF file
        ff = readFF(dir_path, file_name)

        # Skip the file if it could not be read
        if ff is None:
            continue

        # Create temporary directory
        dir_tmp_path = os.path.join(dir_path, "temp_img_dir")

        if os.path.exists(dir_tmp_path):
            shutil.rmtree(dir_tmp_path)
            print("Deleted directory : " + dir_tmp_path)
            
        mkdirP(dir_tmp_path)
        print("Created directory : " + dir_tmp_path)

        # extract the individual frames
        print(file_name)
        f2f.FFtoFrames(dir_path+'/'+file_name, dir_tmp_path, 'jpg', -1)
        
        # Get the timestamp from the FF name
        timestamp = filenameToDatetime(file_name).strftime("%Y-%m-%d %H:%M:%S")
		
        # Get id cam from the file name
            # e.g.  FF499_20170626_020520_353_0005120.bin
            # or FF_CA0001_20170626_020520_353_0005120.fits

        file_split = file_name.split('_')

        # Check the number of list elements, and the new fits format has one more underscore
        i = 0
        if len(file_split[0]) == 2:
            i = 1
        camid = file_split[i]

        # add datestamp to each frame
        jpg_list = [jpg_name for jpg_name in sorted(os.listdir(dir_tmp_path))]
        for img_file_name in jpg_list:
            img=cv2.imread(os.path.join(dir_tmp_path, img_file_name))

            # Draw text to image
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = camid + " " + timestamp + " UTC"
            cv2.putText(img, text, (10, ff.nrows - 6), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            # Save the labelled image to disk
            cv2.imwrite(os.path.join(dir_tmp_path, img_file_name), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    
        ffbasename = os.path.splitext(file_name)[0]

        # If running on Windows, use ffmpeg.exe
        if platform.system() == 'Windows':

            # ffmpeg.exe path
            root = os.path.dirname(__file__)
            ffmpeg_path = os.path.join(root, "ffmpeg.exe")
        else:
            # lets hope its in the path
            ffmpeg_path = "ffmpeg"
        
        # Construct the ecommand for ffmpeg           
        mp4_path = ffbasename + ".mp4"
        temp_img_path = os.path.join(dir_tmp_path, ffbasename+"_%03d.jpg")
        com = ffmpeg_path + " -y -f image2 -pattern_type sequence -i " + temp_img_path +" " + mp4_path
        
        print("Creating timelapse using ffmpeg...")
        print(com)
        subprocess.call(com, shell=True, cwd=dir_path)
        
        #Delete temporary directory and files inside
        if os.path.exists(dir_tmp_path):
            shutil.rmtree(dir_tmp_path)
            print("Deleted temporary directory : " + dir_tmp_path)
		
    print("Total time:", datetime.datetime.utcnow() - t1)

if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Convert all FF files in a folder to animated GIFs")

    arg_parser.add_argument('dir_path', metavar='DIR_PATH', type=str, \
        help='Path to directory with FF files.')

#    arg_parser.add_argument('-x', '--nodel', action="store_true", \
#        help="""Do not delete generated JPG file.""")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    dir_path = os.path.normpath(cml_args.dir_path)

    GenerateMP4s(dir_path)
