""" Takes a video file and chops it into PNGs. """

from __future__ import print_function, division, absolute_import

import os
import sys
import math
import argparse

import cv2

from RMS.Misc import mkdirP


if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Takes a video file and chops it into PNGs.")

    arg_parser.add_argument('video_file', type=str, nargs=1, help='Path to a video file.')

    arg_parser.add_argument('output_dir', type=str, nargs=1, help='Path to the directory where the PNGs will be saved.')

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    video_file = cml_args.video_file[0]

    if not os.path.exists(video_file):
        print('The file does not exist:', video_file)
        sys.exit()

    
    # Open the video file
    cap = cv2.VideoCapture(video_file)

    # Get the total number of frames in a file
    try:
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    except:
        try:
            nframes = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        except:
            nframes = 10**6

    # Compute the number padding
    npad = int(math.log10(nframes)) + 1

    # PNG output dir
    out_dir = os.path.abspath(cml_args.output_dir[0])

    # Make a save directory
    mkdirP(out_dir)

    c = 0

    # Save all frames to disk
    while(cap.isOpened()):

        # Read a frame
        ret, frame = cap.read()

        # Break the loop if all frames were read
        if not ret:
            break

        # Convert a frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Save frame to disk
        frame_path = os.path.join(out_dir, ("{:0" + str(npad) + "d}.png").format(c))
        print('Writing:', frame_path)
        cv2.imwrite(frame_path, gray)

        c += 1


    print('Done!')


