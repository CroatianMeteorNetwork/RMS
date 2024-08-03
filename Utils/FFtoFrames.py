""" Convert the FF files into individual reconstructed video frames. """

import os
import sys
import argparse
import datetime

import numpy as np
import cv2

import RMS.ConfigReader as cr
from RMS.Formats.FFfile import read as readFF
from RMS.Formats.FFfile import reconstructFrame, filenameToDatetime
from RMS.Misc import mkdirP
from RMS.Routines.Image import deinterlaceEven, deinterlaceOdd


def saveFrame(frame, frame_no, out_dir, file_name, file_format, ff_dt, fps, half_frame=None):
    """ Save the file in an appropriate format. """

    # If the metal PNG format was given, convert the file to a 16-bit PNG and byteswap it
    if file_format == 'pngm':
        
        # Convert to 16 bit PNG
        frame = frame.astype(np.uint16)

        # Byteswap the PNG
        frame = frame.byteswap()

        file_format = 'png'


    # Construct the file name
    file_name_saving = file_name + '_{:03d}'.format(frame_no)

    if half_frame is not None:
        file_name_saving += ".{:d}".format(int(5*half_frame))

    file_name_saving = file_name_saving + '.' + file_format


    # Save the frame to file
    out_path = os.path.join(out_dir, file_name_saving)

    # Save the image to file
    cv2.imwrite(out_path, frame)

    print('Saved:', out_path)


    if half_frame is None:
        half_frame_time = 0
    else:
        half_frame_time = 0.5*half_frame

    # Compute the frame datetime
    frame_dt = ff_dt + datetime.timedelta(seconds=(frame_no + half_frame_time)/float(fps))

    return file_name_saving, frame_dt


def FFtoFrames(file_path, out_dir, file_format, deinterlace_mode, first_frame=0, last_frame=255, cml_fps=25, 
               avepixel_bg=True): 
    #########################

    # Read the deinterlace
    #   -1 - no deinterlace
    #    0 - odd first
    #    1 - even first

    if deinterlace_mode not in (-1, 0, 1):
        print('Unknown deinterlace mode:', deinterlace_mode)
        sys.exit()



    # Check if the file exists
    if not os.path.isfile(file_path):

        print('The file {:s} does not exist!'.format(file_path))
        sys.exit()


    # Check if the output directory exists, make it if it doesn't
    if not os.path.exists(out_dir):

        print('Making directory: out_dir')
        mkdirP(out_dir)



    # Open the FF file
    dir_path, file_name = os.path.split(file_path)
    ff = readFF(dir_path, file_name)




    # Take the FPS from the FF file, if available
    if hasattr(ff, 'fps'):
        fps = ff.fps

    # Take the FPS from the config file, if it was not given as an argument
    if fps is None:
        fps = cml_fps

    # Try to read the number of frames from the FF file itself
    if ff.nframes > 0:
        nframes = ff.nframes

    else:
        nframes = 256

    # Construct a file name for saving
    if file_format == 'pngm':

        # If the METAL type PNG file is given, make the file name 'dump'
        file_name_saving = 'dump'

    else:

        file_name_saving = file_name.replace('.fits', '').replace('.bin', '')


    frame_name_time_list = []

    # Get the initial time of the FF file
    ff_dt = filenameToDatetime(file_name)

    # Go through all frames
    for i in range(first_frame, last_frame+1):
        # Reconstruct individual frames

        frame = reconstructFrame(ff, i, avepixel=avepixel_bg)
        # Deinterlace the frame if necessary, odd first
        if deinterlace_mode == 0:

            frame_odd = deinterlaceOdd(frame)
            frame_name, frame_dt = saveFrame(frame_odd, i, out_dir, file_name_saving, file_format, ff_dt, fps, half_frame=0)
            frame_name_time_list.append([frame_name, frame_dt])

            frame_even = deinterlaceEven(frame)
            frame_name, frame_dt = saveFrame(frame_even, i, out_dir, file_name_saving, file_format, ff_dt, fps, half_frame=1)
            frame_name_time_list.append([frame_name, frame_dt])

        # Even first
        elif deinterlace_mode == 1:

            frame_even = deinterlaceEven(frame)
            frame_name, frame_dt = saveFrame(frame_even, i, out_dir, file_name_saving, file_format, ff_dt, fps, half_frame=0)
            frame_name_time_list.append([frame_name, frame_dt])

            frame_odd = deinterlaceOdd(frame)
            frame_name, frame_dt = saveFrame(frame_odd, i, out_dir, file_name_saving, file_format, ff_dt, fps, half_frame=1)
            frame_name_time_list.append([frame_name, frame_dt])


        # No deinterlace
        else:
            frame_name, frame_dt = saveFrame(frame, i, out_dir, file_name_saving, file_format, ff_dt, fps)
            frame_name_time_list.append([frame_name, frame_dt])

    # If the frames are saved for METAL, the times have to be given in a separate file
    if file_format == 'pngm':

        with open(os.path.join(out_dir, 'frtime.txt'), 'w') as f:

            # Write all frames and times in a file
            for frame_name, frame_dt in frame_name_time_list:
                # 20180117:01:08:29.8342
                f.write('{:s} {:s}\n'.format(frame_name, frame_dt.strftime("%Y%m%d:%H:%M:%S.%f")))

    return frame_name_time_list


if __name__ == "__main__":
    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Convert the given FF file to individual reconstructed FF video frames.")

    arg_parser.add_argument('file_path', nargs=1, metavar='FILE_PATH', type=str, \
        help='Path to the FF file.')

    arg_parser.add_argument('output_dir', nargs=1, metavar='OUTPUT_DIR', type=str, \
        help='Path to the output directory.')

    arg_parser.add_argument('file_format', nargs=1, metavar='FILE_FORMAT', type=str, \
        help='File format of images, e.g. jpg or png. Use pngm for METAL type png.')

    arg_parser.add_argument('-d', '--deinterlace', nargs='?', type=int, default=-1, help="Perform manual reduction on deinterlaced frames, even first by default. If odd first is desired, -d 1 should be used.")

    arg_parser.add_argument('-f', '--fps', metavar='FPS', type=float, help="Frames per second of the video. If not given, it will be read from a) the FF file if available, b) from the config file.")

    arg_parser.add_argument('--noavg', action='store_true', help="Do not use avepixel as the background.")

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
        help="Path to a config file which will be used instead of the default one.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    file_path = cml_args.file_path[0]
    out_dir = cml_args.output_dir[0]
    file_format = cml_args.file_format[0]
    deinterlace_mode = cml_args.deinterlace
    if cml_args.deinterlace is None:
        deinterlace_mode = 0

    config = cr.loadConfigFromDirectory(cml_args.config, 'notused')

    if cml_args.fps is None:
        cml_fps = config.fps
    else:
        cml_fps = cml_args.fps

    FFtoFrames(file_path, out_dir, file_format, deinterlace_mode, cml_fps=cml_fps, avepixel_bg=not cml_args.noavg)







