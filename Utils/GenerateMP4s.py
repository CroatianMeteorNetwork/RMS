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
import time
import Utils.FFtoFrames as f2f

from RMS.Formats import FTPdetectinfo

from PIL import ImageFont

from RMS.Formats.FFfile import read as readFF
from RMS.Formats.FFfile import validFFName, filenameToDatetime
from RMS.Misc import mkdirP

def GenerateMP4s(dir_path, ftpfile_name):
    t1 = datetime.datetime.utcnow()

    # Load the font for labeling
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 18)
    except:
        font = ImageFont.load_default()
    
    print("Preparing files for the timelapse...")
    # load the ftpfile so we know which frames we want
    meteor_list = FTPdetectinfo.readFTPdetectinfo(dir_path, ftpfile_name)  
    for meteor in meteor_list:
        ff_name, _, _, n_segments, _, _, _, _, _, _, _, \
            meteor_meas = meteor
        # determine which frames we want

        first_frame=int(meteor_meas[0][1])-30
        last_frame=first_frame + 60
        if first_frame < 0:
            first_frame = 0
        if (n_segments > 1 ):
            lastseg=int(n_segments)-1
            last_frame = int(meteor_meas[lastseg][1])+30
        #if last_frame > 255 :
        #    last_frame = 255     
        if last_frame < first_frame+60:
            last_frame = first_frame+60

        print(ff_name, ' frames ', first_frame, last_frame)
        
        # Read the FF file
        ff = readFF(dir_path, ff_name)

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
        f2f.FFtoFrames(dir_path+'/'+ff_name, dir_tmp_path, 'jpg', -1, first_frame, last_frame)
        
        # Get the timestamp from the FF name
        timestamp = filenameToDatetime(ff_name).strftime("%Y-%m-%d %H:%M:%S")
		
        # Get id cam from the file name
            # e.g.  FF499_20170626_020520_353_0005120.bin
            # or FF_CA0001_20170626_020520_353_0005120.fits

        file_split = ff_name.split('_')

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
    
        ffbasename = os.path.splitext(ff_name)[0]
        mp4_path = ffbasename + ".mp4"
        temp_img_path = os.path.join(dir_tmp_path, ffbasename+"_%03d.jpg")

        # If running on Windows, use ffmpeg.exe
        if platform.system() == 'Windows':

            # ffmpeg.exe path
            root = os.path.dirname(__file__)
            ffmpeg_path = os.path.join(root, "ffmpeg.exe")
            # Construct the ecommand for ffmpeg           
            com = ffmpeg_path + " -y -f image2 -pattern_type sequence -i " + temp_img_path +" " + mp4_path
            print("Creating timelapse using ffmpeg...")
        else:
            # If avconv is not found, try using ffmpeg
            software_name = "avconv"
            print("Checking if avconv is available...")
            if os.system(software_name + " --help > /dev/null"):
                software_name = "ffmpeg"
                # Construct the ecommand for ffmpeg           
                com = software_name + " -y -f image2 -pattern_type sequence -i " + temp_img_path +" " + mp4_path
                print("Creating timelapse using ffmpeg...")
            else:
                print("Creating timelapse using avconv...")
                com = "cd " + dir_path + ";" \
                    + software_name + " -v quiet -r 30 -y -i " + temp_img_path \
                    + " -vcodec libx264 -pix_fmt yuv420p -crf 25 -movflags faststart -g 15 -vf \"hqdn3d=4:3:6:4.5,lutyuv=y=gammaval(0.97)\" " \
                    + mp4_path

        #print(com)
        subprocess.call(com, shell=True, cwd=dir_path)
        
        #Delete temporary directory and files inside
        if os.path.exists(dir_tmp_path):
            try:
                shutil.rmtree(dir_tmp_path)
            except:
                # may occasionally fail due to ffmpeg thread still terminating
                # so catch this and wait a bit
                time.sleep(2)
                shutil.rmtree(dir_tmp_path)

            print("Deleted temporary directory : " + dir_tmp_path)
		
    print("Total time:", datetime.datetime.utcnow() - t1)

if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Convert all FF files in a folder to animated GIFs")

    arg_parser.add_argument('dir_path', metavar='DIR_PATH', type=str, \
        help='Path to directory with FF files.')

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    dir_path = os.path.normpath(cml_args.dir_path)
    ftpdate=''
    if os.path.split(dir_path)[1] == '' :
        ftpdate=os.path.split(os.path.split(dir_path)[0])[1]
    else:
        ftpdate=os.path.split(dir_path)[1]
    ftpfile_name="FTPdetectinfo_"+ftpdate+'.txt'
    # print(ftpfile_name)

    GenerateMP4s(dir_path, ftpfile_name)
