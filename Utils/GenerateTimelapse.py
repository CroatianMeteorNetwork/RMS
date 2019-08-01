""" Generate an MP4 movie from FF files. """

from __future__ import print_function, division, absolute_import

import sys
import os
import argparse
import subprocess
import datetime

import scipy.misc

from RMS.Formats.FFfile import read as readFF
from RMS.Formats.FFfile import validFFName, filenameToDatetime
from RMS.Routines.Image import adjustLevels
from PIL import ImageFont, ImageDraw


if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Convert all FF files in a folder to a movie")

    arg_parser.add_argument('dir_path', metavar='DIR_PATH', type=str, \
        help='Path to directory with FF files.')

    arg_parser.add_argument('-x', '--nodel', action="store_true", \
        help="""Do not delete generated JPG file.""")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    dir_path = cml_args.dir_path


    t1 = datetime.datetime.utcnow()

    # Load the font for labeling
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 18)
    except:
        font = ImageFont.load_default()


    print ("Preparing files for the timelapse...")
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

        # Make a filename for the image, continuous count %04d
        img_file_name = 'temp_{:04d}.jpg'.format(c)

        img = ff.maxpixel

        # # Adjust image gamma to pretty up the image
        # img = adjustLevels(ff.maxpixel, 0, 1.3, 245)

        # convert scipy object to an image
        jpg = scipy.misc.toimage(img)
        draw = ImageDraw.Draw(jpg)
        draw.text((0, 0), timestamp, 'rgb(255,255,255)', font=font)

        # draw text onto the image
        draw = ImageDraw.Draw(jpg)

        # Save the labelled image to disk
        scipy.misc.imsave(os.path.join(dir_path, img_file_name), jpg)

        c = c + 1

        # Print elapsed time
        if c % 30 == 0:
            print("{:>5d}/{:>5d}, Elapsed: {:s}".format(c + 1, len(ff_list), \
                str(datetime.datetime.utcnow() - t1)), end="\r")
            sys.stdout.flush()


    # Construct the ecommand for avconv            
    mp4_path = os.path.join(dir_path, os.path.basename(dir_path).replace(os.sep, "") + ".mp4")
    com = "cd " + dir_path + ";" \
        + "avconv -v quiet -r 30 -y -i temp_%04d.jpg -flags:0 gray -vcodec libx264 -vsync passthrough -pix_fmt yuv420p -crf 25 -r 30 -vf lutyuv=\"y=gammaval=(0.77)\" " \
        + mp4_path

    print ("Creating timelapse by avconv...")
    print(com)
    subprocess.call([com], shell=True)

    # Remove temporary jpg files from the SD card to save space
    if not cml_args.nodel:
        print ("Removing temporary JPG files...")
        subprocess.call(['rm -f ' + dir_path + 'temp_*.jpg'], shell=True)


    print("Total time:", datetime.datetime.utcnow() - t1)
