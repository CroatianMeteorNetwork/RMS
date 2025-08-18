from __future__ import print_function, division, absolute_import


import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

from RMS.Formats.FRbin import read as readFR
from RMS.Routines.Image import saveImage


def makeFRmosaic(dir_path, border=5, line=0, frame_beg=None, frame_end=None, every_nth=None, fps=25.0, dt=0):
    """ Make a mosaic out of an FR bin file. """

    dir_path, file_name = os.path.split(dir_path)

    # Load the FR file
    fr = readFR(dir_path, file_name)

    # # Go through all lines in the file
    # for i in range(fr.lines):

    # Select a given line
    i = line

    print("Line {:d}, of max index {:d}".format(line, fr.lines - 1))

    # Determine the maximum size of the window
    max_size = max([fr.size[i][z] for z in range(fr.frameNum[i])])


    # Get the frame range
    frame_range = []
    for z in range(fr.frameNum[i]):

        print("Frame:", z)

        # Skip all frames outside given frame range
        if frame_beg is not None:
            if z < frame_beg:
                print("... skipped")
                continue

        if frame_end is not None:
            if z > frame_end:
                print("... skipped")
                continue


        # Skip every Nth frame, if given
        if every_nth is not None:
            if not (z%every_nth == 0):
                print("... skipped")
                continue


        frame_range.append(z)

        
    total_frames = len(frame_range)

    if every_nth is None:
        every_nth = 1



    # Determine the width and the height in frame segments
    height = int(np.ceil(np.sqrt(total_frames)))
    width = int(np.ceil(total_frames/height))

    # Compute the image width and height
    w_img = int(np.ceil(width*(border + max_size)))
    h_img = int(np.ceil(height*(border + max_size)))

    print()
    print("Total frames:", total_frames)
    print("Frame segments (h, w):", height, width)
    print("Image dimensions (h, w) px:", h_img, w_img)

    # Create an empty mosaic image
    mosaic_img = np.zeros((h_img, w_img), dtype=np.uint8)

        
    # Cut image to max extent of frames
    #x_min = w_img
    # x_max = 0
    #y_min = h_img
    #y_max = 0

    # Don't cut the image
    x_min = 0
    x_max = w_img
    y_min = 0
    y_max = h_img

    # Go through all visible frames
    print()
    for tile_index, z in enumerate(frame_range):

        # # Get the center position of the detection on the current frame
        # yc = fr.yc[i][z]
        # xc = fr.xc[i][z]

        # Get the real frame number
        t = fr.t[i][z]

        # Get the size of the window
        size = fr.size[i][z]

        # Compute the position of the frame on the image (tiling)
        w_ind = int(tile_index%width)
        h_ind = int(tile_index/width)

        print("Plotting frame:", z, "position (h, w):", h_ind, w_ind)

        # Compute the position of the frame on the image in image coordinates
        frame_x = w_ind*max_size + border + (max_size - size)//2 + 1
        frame_y = h_ind*max_size + border + (max_size - size)//2 + 1

        # Get the frame size
        fr_y, fr_x = fr.frames[i][z].shape

        # Assign the frame to the mosaic
        print(frame_y, (frame_y + fr_y), frame_x, (frame_x + fr_x))
        mosaic_img[frame_y:(frame_y + fr_y), frame_x:(frame_x + fr_x)] = fr.frames[i][z]

        # Keep track of the min and max value of the extent of the frames
        #x_min = min(x_min, frame_x)
        x_max = max(x_max, frame_x + fr_x)
        #y_min = min(y_min, frame_y)
        y_max = max(y_max, frame_y + fr_y)

        # Add the frame time
        fr_time = t/fps + dt
        plt.text(w_ind*max_size + border, 10 + h_ind*max_size + border, '{:5.2f} s'.format(fr_time), \
            va='top', ha='left', size=10, color='w')


    # Draw a grid (skip edges)
    for h_ind in range(1, height):

        print(h_ind*max_size + border)
        
        # Draw horizontal lines
        mosaic_img[h_ind*max_size + border, :] = 255

    for v_ind in range(1, width):
        
        # Draw horizontal lines
        mosaic_img[:, v_ind*max_size + border] = 255


    # Cut the image to the size of the frames
    mosaic_img = mosaic_img[y_min:y_max, x_min:x_max]

    # Plot the image
    plt.imshow(mosaic_img, cmap='gray', vmin=0, vmax=255, aspect='equal')
    plt.gca().set_axis_off()

    # Save the image to disk
    img_file_name = ".".join(file_name.split('.')[:-1]) + '_line_{:d}_mosaic.png'.format(line)
    #saveImage(os.path.join(dir_path, img_file_name), mosaic_img)
    plt.savefig(os.path.join(dir_path, img_file_name), dpi=300, bbox_inches='tight',transparent=True, pad_inches=0)

    plt.show()










if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Make a mosaic image out of a FR bin file.")

    arg_parser.add_argument('dir_path', nargs=1, metavar='DIR_PATH', type=str, \
        help='Path to the FR bin file.')

    arg_parser.add_argument('-l', '--line', metavar='LINE', type=int,
        help="Select a given line.", default=0)

    arg_parser.add_argument('-b', '--framebeg', metavar='FRAMEBEG', type=int,
        help="First frame to take.")

    arg_parser.add_argument('-e', '--frameend', metavar='FRAMEEND', type=int,
        help="Last frame to take.")

    arg_parser.add_argument('-n', '--everyn', metavar='FRAMEEND', type=int,
        help="Take every Nth frame.")

    arg_parser.add_argument('--fps', metavar='FPS', type=float,
        help="FPS, used for the time stamp.", default=25.0)

    arg_parser.add_argument('--dt', metavar='DELTA_T', type=float,
        help="Timestamp offset (seconds).", default=0)

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################
    


    makeFRmosaic(cml_args.dir_path[0], line=cml_args.line, frame_beg=cml_args.framebeg, \
        frame_end=cml_args.frameend, every_nth=cml_args.everyn, fps=cml_args.fps, dt=cml_args.dt)