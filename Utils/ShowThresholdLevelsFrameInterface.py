""" The script will open a valid file supported by FrameInterface and plot color-coded images where the color
    represents the threshold needed to detect individual features on the image.
"""

from __future__ import print_function, division, absolute_import

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import RMS.ConfigReader as cr
from RMS.Formats.FrameInterface import detectInputType


if __name__ == "__main__":

    ### PARSE INPUT ARGUMENTS ###

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description=""" Show threshold levels needed to detect certain meteors using FrameInterface (supports videos).""")

    arg_parser.add_argument('input_path', type=str, help="Path to the input file or directory.")

    arg_parser.add_argument('-s', '--start', type=int, default=0, help="Start frame number (default: 0).")

    arg_parser.add_argument('-n', '--nframes', type=int, default=256, help="Number of frames to stack (default: 256).")

    arg_parser.add_argument('-k', '--k1', type=float, help="Override k1 threshold.")

    arg_parser.add_argument('-j', '--j1', type=float, help="Override j1 threshold.")
    
    arg_parser.add_argument('-f', '--fireball', action="store_true", help="""Estimate threshold for fireball
        detection, not meteor detection. """)

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
        help="Path to a config file which will be used instead of the default one.")

    #############################

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    # Load the config file
    # We need to pass the directory of the input file to loadConfigFromDirectory if it's a file
    if os.path.isfile(cml_args.input_path):
        config_dir = os.path.dirname(cml_args.input_path)
    else:
        config_dir = cml_args.input_path
        
    config = cr.loadConfigFromDirectory(cml_args.config, config_dir)

    # Detect input type
    # We use detection=True to apply binning if configured, as we want to see what the detector sees
    img_handle = detectInputType(cml_args.input_path, config, detection=True)

    if img_handle is None:
        print("Could not detect input type for: {}".format(cml_args.input_path))
        sys.exit(1)

    print("Detected input type: {}".format(img_handle.input_type))
    print("Total frames: {}".format(img_handle.total_frames))

    # Determine thresholds
    if cml_args.k1 is not None:
        k1 = cml_args.k1
    else:
        if cml_args.fireball:
            k1 = config.k1
        else:
            k1 = config.k1_det

    if cml_args.j1 is not None:
        j1 = cml_args.j1
    else:
        if cml_args.fireball:
            j1 = config.j1
        else:
            j1 = config.j1_det

    print("Using thresholds: k1 = {:.2f}, j1 = {:.2f}".format(k1, j1))


    # Load the chunk
    print("Loading {} frames starting from frame {}...".format(cml_args.nframes, cml_args.start))
    ff = img_handle.loadChunk(first_frame=cml_args.start, read_nframes=cml_args.nframes)

    if ff is None or ff.nframes == 0:
        print("No frames loaded.")
        sys.exit(1)
    
    # Check if we have enough statistical data
    if not hasattr(ff, 'stdpixel') or ff.stdpixel is None:
         print("Error: stdpixel not available in the loaded chunk.")
         sys.exit(1)


    # Compute the threshold value
    # stdpixel can be 0, avoid division by zero
    stdpixel = ff.stdpixel.astype(np.float64)
    stdpixel[stdpixel == 0] = 1.0 # Avoid division by zero

    k1_vals = (ff.maxpixel.astype(np.float64) - ff.avepixel.astype(np.float64) \
        - j1)/stdpixel


    # Calculate max-ave
    max_ave = ff.maxpixel.astype(np.float64) - ff.avepixel.astype(np.float64)

    # Calculate figure size based on image dimensions
    # We want to display a 2x2 grid
    img_height, img_width = k1_vals.shape
    aspect_ratio = img_width / img_height
    
    # Base width on a reasonable screen size (e.g., 12 inches for 2 cols)
    fig_width = 12
    # Calculate height for 2 rows. 
    # Total width = 2 * img_width. Total height = 2 * img_height.
    # Aspect Ratio of the whole figure (excluding UI elements) should be AspectRatio
    fig_height = fig_width / aspect_ratio

    # Limit the height to something reasonable to avoid ultra-tall windows
    if fig_height > 15:
        scale = 15 / fig_height
        fig_width *= scale
        fig_height *= scale

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(fig_width, fig_height))

    # Plot max - ave (Signal)
    im1 = ax1.imshow(max_ave, cmap='gray', aspect='equal', vmin=0, vmax=np.percentile(max_ave, 99.5))
    ax1.set_title("Max - Ave (Signal)")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Plot k1 vals
    k1map = ax2.imshow(k1_vals, cmap='inferno', vmin=1, vmax=6,  aspect='equal')
    ax2.set_title("k1 values (Signal/Noise - j1)")
    
    if cml_args.fireball:
        plt.colorbar(k1map, ax=ax2, label='k1', fraction=0.046, pad=0.04)
    else:
        plt.colorbar(k1map, ax=ax2, label='k1_det', fraction=0.046, pad=0.04)

    # Plot stdpixel (Noise)
    im3 = ax3.imshow(stdpixel, cmap='gray', aspect='equal', vmin=0, vmax=np.percentile(stdpixel, 99.5))
    ax3.set_title("Stdpixel (Noise)")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)


    # Plot thresholded image
    threshld = ff.maxpixel > ff.avepixel + k1*ff.stdpixel + j1
    ax4.imshow(threshld, cmap='gray', aspect='equal')
    ax4.set_title("Thresholded")
    ax4.text(0, 0, "k1 = {:.2f}, j1 = {:.2f}".format(k1, j1), color='red', verticalalignment='top',
        weight='bold')

    # Main title
    fig.suptitle("Input: {:s} | Start: {:d} | Frames: {:d}".format(os.path.basename(cml_args.input_path), cml_args.start, ff.nframes))

    fig.tight_layout()

    plt.show()
