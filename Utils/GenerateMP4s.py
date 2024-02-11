#!/usr/bin/env python

""" Generate an High Quality MP4 movie from FF files. 
    Contributors: Tioga Gulon
"""

from __future__ import print_function, division, absolute_import

import glob
import os
import platform
import argparse
import subprocess
import shutil
import datetime
import cv2
import time
import Utils.FFtoFrames as f2f
from Utils.ShowerAssociation import showerAssociation
import RMS.ConfigReader as cr
from RMS.Formats.FTPdetectinfo import validDefaultFTPdetectinfo

from RMS.Formats import FTPdetectinfo

from PIL import ImageFont

from RMS.Formats.FFfile import read as readFF
#from RMS.Formats.FFfile import filenameToDatetime
from RMS.Misc import mkdirP


def generateMP4s(dir_path, ftpfile_name, shower_code=None, min_mag=None, config=None):
    t1 = datetime.datetime.utcnow()

    # Load the font for labeling
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 18)
    except:
        font = ImageFont.load_default()
    
    if isinstance(min_mag,(int, float)):
        min_mag = float(min_mag)
    else:
        min_mag = None
    print('min_mag is {}'.format(min_mag))

    if shower_code is not None and config is not None:
        associations, _ = showerAssociation(config, [os.path.join(dir_path, ftpfile_name)])
    else:
        print('unable to determine shower associations, ignoring shower code')
        shower_code = None

    print("Preparing files for the timelapse...")
    # load the ftpfile so we know which frames we want
    meteor_list = FTPdetectinfo.readFTPdetectinfo(dir_path, ftpfile_name)  
    for meteor in meteor_list:
        ff_name, _, meteor_no, n_segments, _, _, _, _, _, _, _, \
            meteor_meas = meteor
        
        # checks on mag and shower        
        best_mag = 999
        if min_mag is not None:
            #print(meteor_meas)
            for meas in meteor_meas:
                best_mag = min(best_mag, meas[9])
            #print('best mag is {}'.format(best_mag))
            if best_mag > min_mag:
                print('rejecting {} as too dim'.format(ff_name))
                continue
        if shower_code is not None:
            if (ff_name, meteor_no) not in associations:
                print('rejecting {} as not in radiants data'.format(ff_name))
                continue
            shower = associations[(ff_name, meteor_no)][1]
            if shower is None: 
                print('rejecting {} as wrong shower'.format(ff_name))
                continue
            elif shower.name != shower_code:
                print('rejecting {} as wrong shower'.format(ff_name))
                continue
            print(shower.name)
            pass

        # determine which frames we want
        
        first_frame=int(meteor_meas[0][1])-30
        last_frame=first_frame + 60
        if first_frame < 0:
            first_frame = 0
        if (n_segments > 1):
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
        name_time_list = f2f.FFtoFrames(dir_path+'/'+ff_name, dir_tmp_path, 'jpg', -1, first_frame, last_frame)

        # Get id cam from the file name
        # e.g.  FF499_20170626_020520_353_0005120.bin
        # or FF_CA0001_20170626_020520_353_0005120.fits

        file_split = ff_name.split('_')

        # Check the number of list elements, and the new fits format has one more underscore
        i = 0
        if len(file_split[0]) == 2:
            i = 1
        camid = file_split[i]

        font = cv2.FONT_HERSHEY_SIMPLEX

        # add datestamp to each frame
        for img_file_name, timestamp in name_time_list:
            img=cv2.imread(os.path.join(dir_tmp_path, img_file_name))

            # Draw text to image
            text = camid + " " + timestamp.strftime("%Y-%m-%d %H:%M:%S") + " UTC"
            if min_mag is not None:
                text = text + ' Mag: {}'.format(best_mag)
            if shower_code:
                text = text + ' Shower: ' + shower_code
            cv2.putText(img, text, (10, ff.nrows - 6), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            # Save the labelled image to disk
            cv2.imwrite(os.path.join(dir_tmp_path, img_file_name), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    
        ffbasename = os.path.splitext(ff_name)[0]
        mp4_path = ffbasename + ".mp4"
        temp_img_path = os.path.abspath(os.path.join(dir_tmp_path, ffbasename+"_%03d.jpg"))

        # If running on Windows, use ffmpeg.exe
        if platform.system() == 'Windows':

            # ffmpeg.exe path
            root = os.path.dirname(__file__)
            ffmpeg_path = os.path.join(root, "ffmpeg.exe")
            # Construct the ecommand for ffmpeg           
            com = ffmpeg_path + " -hide_banner -loglevel error -pix_fmt yuv420p  -y -f image2 -pattern_type sequence -start_number " + str(first_frame) + " -i " + temp_img_path +" " + mp4_path
            print("Creating timelapse using ffmpeg...")
        else:
            # If avconv is not found, try using ffmpeg
            software_name = "avconv"
            print("Checking if avconv is available...")
            if os.system(software_name + " --help > /dev/null"):
                software_name = "ffmpeg"
                # Construct the ecommand for ffmpeg           
                com = software_name + " -hide_banner -loglevel error -pix_fmt yuv420p  -y -f image2 -pattern_type sequence -start_number " + str(first_frame) + " -i " + temp_img_path +" " + mp4_path
                print("Creating timelapse using ffmpeg...")
            else:
                print("Creating timelapse using avconv...")
                com = "cd " + dir_path + ";" \
                    + software_name + " -v quiet -r 30 -y -start_number " + str(first_frame) + " -i " + temp_img_path \
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

    # COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Convert all FF files in a folder to MP4s")

    arg_parser.add_argument('dir_path', metavar='DIR_PATH', type=str,
        help='Path to directory with FF files.')

    arg_parser.add_argument('-s', '--shower', metavar='SHOWER', type=str, \
        help="Process just this single shower given its code (e.g. PER, ORI, ETA).")

    arg_parser.add_argument('-m', '--minmag', metavar='MINMAG', type=float, \
        help="Process only detections brighter than this")

    arg_parser.add_argument('-c', '--config', metavar='CONFIG', type=str, \
        help="full path to config file. Only required if filtering by shower and no config file in the target folder")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    dir_path = os.path.normpath(cml_args.dir_path)
    try:
        ftps = glob.glob(os.path.join(dir_path,'FTPdetectinfo*.txt'))
    except Exception:
        print('unable to access target folder - check path')
        exit(1)

    # Load the config file
    config = None
    if cml_args.config:
        config = cr.loadConfigFromDirectory(cml_args.config, dir_path)
    else:
        if os.path.isfile(os.path.join(dir_path, '.config')):
            config = cr.loadConfigFromDirectory('.config', dir_path)

    if len(ftps) == 0:
        print('no ftpdetect files in target folder - unable to continue')

    else:
        ftps = [x for x in ftps if validDefaultFTPdetectinfo(os.path.basename(x))]
        if len(ftps) == 0:
            print('no usable ftpdetect file present')
        else:
            generateMP4s(dir_path, ftps[0], shower_code=cml_args.shower, min_mag=cml_args.minmag, config=config)
