from __future__ import print_function, division, absolute_import


import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

from RMS.Formats.FRbin import read as readFR



def makeFRmosaic(dir_path, border=5):
    """ Make a mosaic out of an FR bin file. """

    dir_path, file_name = os.path.split(dir_path)

    # Load the FR file
    fr = readFR(dir_path, file_name)

    # Go through all lines in the file
    for i in range(fr.lines):

        # Determine the maximum size of the window
        max_size = max([fr.size[i][z] for z in range(fr.frameNum[i])])

        # Determine the width and the height in frame segments
        height = int(np.ceil(np.sqrt(fr.frameNum[i])))
        width = int(np.ceil(fr.frameNum[i]/height))

        # Compute the image width and height
        w_img = int(np.ceil(width*(border + max_size)))
        h_img = int(np.ceil(height*(border + max_size)))

        # Create an empty mosaic image
        mosaic_img = np.zeros((h_img, w_img), dtype=np.uint8)

        
        x_min = w_img
        x_max = 0
        y_min = h_img
        y_max = 0

        # Go through all frames
        for z in range(fr.frameNum[i]):

            # # Get the center position of the detection on the current frame
            # yc = fr.yc[i][z]
            # xc = fr.xc[i][z]

            # # Get the frame number
            # t = fr.t[i][z]

            # Get the size of the window
            size = fr.size[i][z]

            # Compute the position of the frame on the image (tiling)
            h_ind = int(z%height)
            v_ind = int(z/height)

            # Compute the position of the frame on the image in image coordinates
            frame_x = h_ind*max_size + border + (max_size - size)//2 + 1
            frame_y = v_ind*max_size + border + (max_size - size)//2 + 1

            # Get the frame size
            fr_y, fr_x = fr.frames[i][z].shape

            # Assign the frame to the mosaic
            mosaic_img[frame_y:(frame_y + fr_y), frame_x:(frame_x + fr_x)] = fr.frames[i][z]

            # Keep track of the min and max value of the extent of the frames
            x_min = min(x_min, frame_x)
            x_max = max(x_max, frame_x + fr_x)
            y_min = min(y_min, frame_y)
            y_max = max(y_max, frame_y + fr_y)


        # Draw a grid
        for h_ind in range(height+1):
        	
        	# Draw horizontal lines
        	mosaic_img[h_ind*max_size + border, :] = 255

        for v_ind in range(width+1):
            
            # Draw horizontal lines
            mosaic_img[:, v_ind*max_size + border] = 255


        # Cut the image to the size of the frames
        mosaic_img = mosaic_img[y_min:y_max, x_min:x_max]


        # Save the image to disk
        img_file_name = ".".join(file_name.split('.')[:-1]) + '_mosaic.png'
        scipy.misc.imsave(os.path.join(dir_path, img_file_name), mosaic_img)

        # Plot the image
        plt.imshow(mosaic_img, cmap='gray', vmin=0, vmax=255)
        plt.show()










if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Make a mosaic image out of a FR bin file.")

    arg_parser.add_argument('dir_path', nargs=1, metavar='DIR PATH', type=str, \
        help='Path to the FR bin file.')

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################
    


    makeFRmosaic(cml_args.dir_path[0])