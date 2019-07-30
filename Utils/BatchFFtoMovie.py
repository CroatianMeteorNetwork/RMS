""" Batch convert FF files to MP4 movie """

import os
import os.path
import argparse
import subprocess

import scipy.misc

from RMS.Formats.FFfile import read as readFF
from RMS.Formats.FFfile import validFFName
from PIL import Image, ImageFont, ImageDraw
from datetime import datetime


if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Convert all FF files in a folder to a movie")

    arg_parser.add_argument('dir_path', nargs=1, metavar='DIR_PATH', type=str, \
        help='Path to directory with FF files.')

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    dir_path = cml_args.dir_path[0] + '/'

    myimages = [] #list of image filenames
    dirFiles = os.listdir(dir_path) #list of directory files
    dirFiles.sort() #good initial sort but doesnt sort numerically very well
    sorted(dirFiles) #sort numerically in ascending order 


    # Go through all files in the given folder
    c = 0 #image count
    # setting font for labelling
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 10)

    print ("Preparing files for the timelapse...")
    for file_name in dirFiles:

        # Check if the file is an FF file
        if validFFName(file_name):

            # Read the FF file
            ff = readFF(dir_path, file_name)

            # get the timestamp from the FITS file name
            label = file_name[16:18] + "-" + file_name[14:16] + "-" + file_name[10:14] + "  " + file_name[19:21] + ":" + file_name[21:23] + ":" + file_name[23:25]

            # Skip the file if it could not be read
            if ff is None:
                continue

            # Make a filename for the image, continuous count %04d
            img_file_name = 'temp_' + '%04d' % c + '.jpg'

            # convert scipy object to an image
            jpg = scipy.misc.toimage(ff.maxpixel)
            draw = ImageDraw.Draw(jpg)
            draw.text((0,0), label, 'rgb(255,255,255)', font=font)

            # draw text onto the image
            draw = ImageDraw.Draw(jpg)

            # Save the labelled image to disk
            scipy.misc.imsave(os.path.join(dir_path, img_file_name),jpg)

            c = c + 1

    # construct the ecommand for avconv            
    com = "cd " + dir_path + ";avconv -v quiet -r 30 -y -i temp_%04d.jpg -flags:0 gray -vcodec libx264 -vsync passthrough -pix_fmt yuv420p -crf 25 -r 30 " + dir_path + os.path.basename(os.path.dirname(dir_path)) + ".mp4"
    print ("Creating timelapse by avconv...")
    subprocess.call([com],shell=True)

    #remove temporary jpg files from the SD card to save space
    print ("Removing temporary files...")
    subprocess.call(['rm -f ' + dir_path + 'temp_*.jpg'],shell=True)
