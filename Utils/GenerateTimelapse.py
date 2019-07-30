""" Generate an MP4 movie from FF files. """

from __future__ import print_function, division, absolute_import

import os
import os.path
import argparse
import subprocess
import datetime

import scipy.misc

from RMS.Formats.FFfile import read as readFF
from RMS.Formats.FFfile import validFFName, filenameToDatetime
from PIL import ImageFont, ImageDraw


if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Convert all FF files in a folder to a movie")

    arg_parser.add_argument('dir_path', nargs=1, metavar='DIR_PATH', type=str, \
        help='Path to directory with FF files.')

    arg_parser.add_argument('-x', '--norm', action="store_true", \
        help="""Do not delete generated JPG file.""")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    dir_path = cml_args.dir_path[0] + '/'


    t1 = datetime.datetime.utcnow()

    # Load the font for labeling
    font = ImageFont.load_default()

    print ("Preparing files for the timelapse...")
    c = 0
    for file_name in sorted(os.listdir(dir_path)):

        # Check if the file is an FF file
        if validFFName(file_name):

            # Read the FF file
            ff = readFF(dir_path, file_name)

            # Skip the file if it could not be read
            if ff is None:
                continue

            # Get the timestamp from the FF name
            timestamp = filenameToDatetime(file_name).strftime("%Y-%m-%d %H:%M:%S")

            # Make a filename for the image, continuous count %04d
            img_file_name = 'temp_' + '%04d'%c + '.jpg'

            # convert scipy object to an image
            jpg = scipy.misc.toimage(ff.maxpixel)
            draw = ImageDraw.Draw(jpg)
            draw.text((0, 0), timestamp, 'rgb(255,255,255)', font=font)

            # draw text onto the image
            draw = ImageDraw.Draw(jpg)

            # Save the labelled image to disk
            scipy.misc.imsave(os.path.join(dir_path, img_file_name),jpg)

            c = c + 1

            # Print elapsed time
            print("Elapsed:", datetime.datetime.utcnow() - t1, end="\r")


    # Construct the ecommand for avconv            
    com = "cd " + dir_path + ";avconv -v quiet -r 30 -y -i temp_%04d.jpg -flags:0 gray -vcodec libx264 -vsync passthrough -pix_fmt yuv420p -crf 25 -r 30 " + dir_path + os.path.basename(os.path.dirname(dir_path)) + ".mp4"
    print ("Creating timelapse by avconv...")
    print(com)
    subprocess.call([com], shell=True)

    # Remove temporary jpg files from the SD card to save space
    print ("Removing temporary JPG files...")
    subprocess.call(['rm -f ' + dir_path + 'temp_*.jpg'], shell=True)


    print("Total time:", datetime.datetime.utcnow() - t1)
