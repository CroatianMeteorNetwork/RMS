"""Stack FR*.bin fireball detections into a single max-value image."""

from __future__ import print_function, division, absolute_import, unicode_literals

import argparse
import math
import os

from RMS.Formats.FRbin import read as readFR
from RMS.Routines.Image import saveImage


def computeFrameExtents(fr):
    """Determine the bounding box that contains every FR frame cutout.

    Arguments:
        fr: [fr_struct] Parsed FR*.bin structure returned by RMS.Formats.FRbin.read.

    Returns:
        [tuple] (min_x, min_y, width, height) describing the extent of the stacked image in pixels.

    """

    # Track the extreme coordinates while iterating through every detection line and frame.
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')

    for line in range(fr.lines):
        for idx in range(fr.frameNum[line]):

            # Each cutout is square and stored using its centre coordinates together with the side length.
            half_size = fr.size[line][idx]//2

            x_start = fr.xc[line][idx] - half_size
            y_start = fr.yc[line][idx] - half_size

            # Update the extrema that mark the bounding box edges.
            min_x = min(min_x, x_start)
            min_y = min(min_y, y_start)

            max_x = max(max_x, x_start + fr.size[line][idx])
            max_y = max(max_y, y_start + fr.size[line][idx])

    if not math.isfinite(min_x) or not math.isfinite(min_y):
        raise ValueError('No frames were found in the supplied FR file.')

    width = int(math.ceil(max_x - min_x))
    height = int(math.ceil(max_y - min_y))

    if width <= 0 or height <= 0:
        raise ValueError('Computed stacked image dimensions are invalid.')

    return int(min_x), int(min_y), width, height


def stackFRbin(fr_path, output_path=None):
    """Stack every detection line from an FR*.bin file into a PNG image.

    Arguments:
        fr_path: [str] Full or relative path to an FR*.bin file.

    Keyword arguments:
        output_path: [str] Optional destination for the PNG file. The extension is enforced.

    Return:
        [str] Absolute path of the written PNG image.

    """

    # Normalise the path so the FR reader can locate the directory and filename separately.
    fr_path = os.path.abspath(fr_path)
    dir_path, file_name = os.path.split(fr_path)

    if not file_name:
        raise ValueError('The provided path does not point to an FR*.bin file.')

    # Load the FR structure and make sure there is at least one detection line to process.
    fr = readFR(dir_path, file_name)

    if fr.lines == 0:
        raise ValueError('The FR file does not contain any detection lines to stack.')

    min_x, min_y, width, height = computeFrameExtents(fr)

    # Shift the cutout centres so that the minimum coordinate starts at zero.
    shift_x = -min_x
    shift_y = -min_y

    if shift_x != 0 or shift_y != 0:
        for line in range(fr.lines):
            fr.xc[line] = [int(x + shift_x) for x in fr.xc[line]]
            fr.yc[line] = [int(y + shift_y) for y in fr.yc[line]]

    # Provide the canvas dimensions so fr.maxpixel can reconstruct the stacked image.
    fr.ncols = width
    fr.nrows = height

    stacked_image = fr.maxpixel

    # Resolve the output path and ensure the directory exists.
    if output_path is None:
        base_name = os.path.splitext(file_name)[0]
        output_path = os.path.join(dir_path, base_name + '_stack.png')
    else:
        output_path = os.path.abspath(output_path)
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        if not output_path.lower().endswith('.png'):
            output_path += '.png'

    saveImage(output_path, stacked_image)
    print('Saved FR stack to: {}'.format(output_path))

    return output_path


if __name__ == '__main__':

    # COMMAND LINE ARGUMENTS

    arg_parser = argparse.ArgumentParser(
        description='Create a stacked max-value image from an FR*.bin file.')

    arg_parser.add_argument('fr_file', nargs=1, metavar='FR_FILE', type=str,
        help='Path to the FR*.bin file to stack.')

    arg_parser.add_argument('-o', '--output', nargs='?', metavar='OUTPUT', type=str,
        help='Optional path for the stacked PNG image.')

    args = arg_parser.parse_args()

    stackFRbin(args.fr_file[0], output_path=args.output)
