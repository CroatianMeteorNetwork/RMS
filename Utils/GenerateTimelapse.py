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

from PIL import ImageFont

from RMS.Formats.FFfile import read as readFF
from RMS.Formats.FFfile import validFFName, filenameToDatetime
from RMS.Misc import mkdirP



def generateTimelapse(dir_path, keep_images=False, fps=None, output_file=None, hires=False):
    """ Generate an High Quality MP4 movie from FF files. 
    
    Arguments:
        dir_path: [str] Path to the directory containing the FF files.

    Keyword arguments:
        keep_images: [bool] Keep the temporary images. False by default.
        fps: [int] Frames per second. 30 by default.
        output_file: [str] Output file name. If None, the file name will be the same as the directory name.
        hires: [bool] Make a higher resolution timelapse. False by default.
    """

    # Set the default FPS if not given
    if fps is None:
        fps = 30


    # Make the name of the output file if not given
    if output_file is None:
        mp4_path = os.path.join(dir_path, os.path.basename(dir_path) + ".mp4")

    else:
        mp4_path = os.path.join(dir_path, os.path.basename(output_file))


    # Set the CRF value for the video compression depending on the resolution
    if hires:
        crf = 25

    else:
        crf = 29


    t1 = datetime.datetime.utcnow()

    # Load the font for labeling
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 18)
    except:
        font = ImageFont.load_default()

    # Create temporary directory
    dir_tmp_path = os.path.join(dir_path, "temp_img_dir")

    if os.path.exists(dir_tmp_path):
        shutil.rmtree(dir_tmp_path)
        print("Deleted directory : " + dir_tmp_path)
		
    mkdirP(dir_tmp_path)
    print("Created directory : " + dir_tmp_path)
    
    print("Preparing files for the timelapse...")
    c = 0


    ff_list = [ff_name for ff_name in sorted(os.listdir(dir_path)) if validFFName(ff_name)]

    for file_name in ff_list:

        # Read the FF file
        ff = readFF(dir_path, file_name)

        # Skip the file if it could not be read
        if ff is None:
            continue

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

        # Make a filename for the image, continuous count %04d
        img_file_name = 'temp_{:04d}.jpg'.format(c)

        img = ff.maxpixel

        # Draw text to image
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = camid + " " + timestamp + " UTC"
        cv2.putText(img, text, (10, ff.nrows - 6), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # Save the labelled image to disk
        cv2.imwrite(os.path.join(dir_tmp_path, img_file_name), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    
        c = c + 1

        # Print elapsed time
        if c % 30 == 0:
            print("{:>5d}/{:>5d}, Elapsed: {:s}".format(c + 1, len(ff_list), \
                str(datetime.datetime.utcnow() - t1)), end="\r")
            sys.stdout.flush()


    # If running on Linux, use avconv if available
    if platform.system() == 'Linux':

        # If avconv is not found, try using ffmpeg. In case of using ffmpeg,
        # use parameter -nostdin to avoid it being stuck waiting for user input
        software_name = "avconv"
        nostdin = ""
        print("Checking if avconv is available...")
        if os.system(software_name + " --help > /dev/null"):
            software_name = "ffmpeg"
            nostdin =  " -nostdin "
        
        # Construct the command for avconv            
        temp_img_path = os.path.basename(dir_tmp_path) + os.sep + "temp_%04d.jpg"
        com = "cd " + dir_path + ";" \
            + software_name + nostdin + " -v quiet -r "+ str(fps) +" -y -i " + temp_img_path \
            + " -vcodec libx264 -pix_fmt yuv420p -crf " + str(crf) \
            + " -movflags faststart -g 15 -vf \"hqdn3d=4:3:6:4.5,lutyuv=y=gammaval(0.77)\" " \
            + mp4_path

        print("Creating timelapse using {:s}...".format(software_name))
        print(com)
        subprocess.call([com], shell=True)


    # If running on Windows, use ffmpeg.exe
    elif platform.system() == 'Windows':
	
        # ffmpeg.exe path
        root = os.path.dirname(__file__)
        ffmpeg_path = os.path.join(root, "ffmpeg.exe")
	
        # Construct the ecommand for ffmpeg
        temp_img_path = os.path.join(os.path.basename(dir_tmp_path), "temp_%04d.jpg")
        com = ffmpeg_path + " -v quiet -r " + str(fps) + " -i " + temp_img_path \
            + " -c:v libx264 -pix_fmt yuv420p -an -crf " + str(crf) \
            + " -g 15 -vf \"hqdn3d=4:3:6:4.5,lutyuv=y=gammaval(0.77)\" -movflags faststart -y " \
            + mp4_path
		
        print("Creating timelapse using ffmpeg...")
        print(com)
        subprocess.call(com, shell=True, cwd=dir_path)
		
    else :
        print ("generateTimelapse only works on Linux or Windows the video could not be encoded")

    #Delete temporary directory and files inside
    if os.path.exists(dir_tmp_path) and not keep_images:
        shutil.rmtree(dir_tmp_path)
        print("Deleted temporary directory : " + dir_tmp_path)
		
    print("Total time:", datetime.datetime.utcnow() - t1)


if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Convert all FF files in a folder to a movie")

    arg_parser.add_argument('dir_path', metavar='DIR_PATH', type=str, \
        help='Path to directory with FF files.')

    arg_parser.add_argument('-fps', '--fps', metavar='FPS', type=int, \
        help='FPS to use for video.')

    arg_parser.add_argument('-x', '--nodel', action="store_true", \
        help="""Do not delete generated JPG file.""")
    
    arg_parser.add_argument('-o', '--output', metavar='OUTPUT', type=str, \
        help='Output video file name.')
    
    arg_parser.add_argument('--hires', action="store_true", \
                            help='Make a higher resolution timelapse. The video file will be larger.')

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################


    dir_path = os.path.normpath(cml_args.dir_path)

    # Generate the timelapse
    generateTimelapse(dir_path, keep_images=cml_args.nodel, fps=cml_args.fps, output_file=cml_args.output, hires=cml_args.hires)
