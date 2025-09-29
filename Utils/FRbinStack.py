"""Stack FR*.bin fireball detections into a single max-value image."""

from __future__ import print_function, division, absolute_import, unicode_literals

import argparse
import math
import os

import cv2
import numpy as np

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


def loadStackedImage(fr_path):
    """Return the stacked image for a single FR*.bin file.

    Arguments:
        fr_path: [str] Full or relative path to an FR*.bin file.

    Returns:
        [tuple] (stacked_image, output_directory, file_name) where stacked_image is the numpy
        array generated from all detection lines.

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

    # Return the prepared stack alongside its source directory and file name.
    return fr.maxpixel, dir_path, file_name


def extractCameraCode(file_name):
    """Extract the camera code from an FR file name if present."""

    # Split the stem using underscores as separators to isolate the camera token.
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    parts = base_name.split('_')

    if len(parts) >= 2:
        return parts[1]

    return None


def annotateCameraCode(image, camera_code):
    """Render the camera code in the upper-left corner of a stacked inset."""

    if not camera_code:
        return

    # Configure the font style so the annotation matches other RMS utilities.
    font = cv2.FONT_HERSHEY_SIMPLEX

    height, width = image.shape[:2]

    # Adapt the text layout to the image dimensions so the annotation remains legible.
    scale = max(0.3, min(0.8, width / 320.0))
    thickness = max(1, int(round(scale * 2)))
    margin = max(3, int(round(min(height, width) * 0.02)))

    text_size, baseline = cv2.getTextSize(camera_code, font, scale, thickness)
    y_pos = margin + text_size[1]

    if y_pos + baseline > height:
        y_pos = max(text_size[1], height - baseline - 1)

    x_pos = margin

    # Select colour values that contrast well regardless of the image depth or channels.
    if np.issubdtype(image.dtype, np.integer):
        white_value = np.iinfo(image.dtype).max
        black_value = np.iinfo(image.dtype).min
    else:
        white_value = 1.0
        black_value = 0.0

    if image.ndim == 2 or image.shape[2] == 1:
        white_colour = white_value
        black_colour = black_value
    else:
        white_colour = tuple([white_value] * image.shape[2])
        black_colour = tuple([black_value] * image.shape[2])

    # Draw a black outline first and then the white text to improve visibility.
    cv2.putText(image, camera_code, (x_pos, y_pos), font, scale,
        black_colour, thickness + 2, cv2.LINE_AA)
    cv2.putText(image, camera_code, (x_pos, y_pos), font, scale,
        white_colour, thickness, cv2.LINE_AA)


def stackFRbin(fr_path, output_path=None):
    """Stack every detection line from an FR*.bin file into a PNG image.

    Arguments:
        fr_path: [str] Full or relative path to an FR*.bin file.

    Keyword arguments:
        output_path: [str] Optional destination for the PNG file. The extension is enforced.

    Return:
        [str] Absolute path of the written PNG image.

    """

    stacked_image, dir_path, file_name = loadStackedImage(fr_path)

    # Resolve the output path and ensure the directory exists.
    if output_path is None:
        base_name = os.path.splitext(file_name)[0]
        output_path = os.path.join(dir_path, base_name + '_stack.png')
    else:
        # Normalise explicit output paths and ensure the destination exists.
        output_path = os.path.abspath(output_path)
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        if not output_path.lower().endswith('.png'):
            output_path += '.png'

    # Persist the stacked image using the standard RMS save helper.
    saveImage(output_path, stacked_image)
    print('Saved FR stack to: {}'.format(output_path))

    return output_path


def stackFRbins(fr_paths, output_path=None, columns=None, output_dir=None):
    """Create a mosaic composed of stacks from multiple FR*.bin files.

    Arguments:
        fr_paths: [Iterable[str]] Collection of FR*.bin file paths to include in the mosaic.

    Keyword arguments:
        output_path: [str] Optional destination for the PNG file. The extension is enforced.
        columns: [int] Optional number of mosaic columns. If omitted the layout is square-ish.
        output_dir: [str] Optional directory used when deriving the mosaic output name.

    Returns:
        [str] Absolute path of the written PNG mosaic image.

    """

    # Convert any incoming iterable to a list so it can be traversed repeatedly below.
    fr_paths = list(fr_paths)

    # Ensure there is at least one FR file to include in the mosaic composition.
    if not fr_paths:
        raise ValueError('At least one FR*.bin file must be provided for the mosaic.')

    # Collect the stacked images together with their directory and file name metadata.
    stacks = []
    base_dirs = []
    file_names = []

    # Store the camera code for each FR stack so the mosaic tiles can be annotated later.
    camera_codes = []

    for path in fr_paths:
        stacked_image, dir_path, file_name = loadStackedImage(path)
        stacks.append(stacked_image)
        base_dirs.append(dir_path)
        file_names.append(file_name)
        camera_codes.append(extractCameraCode(file_name))

    # Determine the mosaic grid dimensions. Default to a square layout when possible.
    image_count = len(stacks)
    if columns is None or columns <= 0:
        columns = int(math.ceil(math.sqrt(image_count)))

    rows = int(math.ceil(float(image_count) / float(columns)))

    # Track the maximum width per column and height per row to build a tight canvas.
    column_widths = [0] * columns
    row_heights = [0] * rows

    for index, image in enumerate(stacks):
        row_idx = index // columns
        col_idx = index % columns

        height, width = image.shape[:2]
        column_widths[col_idx] = max(column_widths[col_idx], width)
        row_heights[row_idx] = max(row_heights[row_idx], height)

    mosaic_height = sum(row_heights)
    mosaic_width = sum(column_widths)

    if mosaic_height == 0 or mosaic_width == 0:
        raise ValueError('Could not determine valid mosaic dimensions from the provided files.')

    # Build the mosaic canvas using the dtype/channel count of the first stack.
    template = stacks[0]
    if template.ndim == 2:
        mosaic = np.zeros((mosaic_height, mosaic_width), dtype=template.dtype)
    else:
        mosaic = np.zeros((mosaic_height, mosaic_width, template.shape[2]), dtype=template.dtype)

    # Lay out each stacked image while keeping track of the current offsets.
    y_offset = 0
    for row_idx in range(rows):
        x_offset = 0
        for col_idx in range(columns):
            image_index = row_idx * columns + col_idx
            if image_index >= image_count:
                break

            image = stacks[image_index]
            height, width = image.shape[:2]

            if template.ndim == 2:
                mosaic[y_offset:y_offset + height, x_offset:x_offset + width] = image
                region = mosaic[y_offset:y_offset + height, x_offset:x_offset + width]
            else:
                mosaic[y_offset:y_offset + height, x_offset:x_offset + width, :] = image
                region = mosaic[y_offset:y_offset + height, x_offset:x_offset + width, :]

            # Tag each inset with the originating camera code for quick identification.
            annotateCameraCode(region, camera_codes[image_index])
            x_offset += column_widths[col_idx]

        y_offset += row_heights[row_idx]

    # Choose a sensible output location when one is not supplied.
    if output_path is None:
        base_name = os.path.splitext(file_names[0])[0]
        output_dir_path = output_dir if output_dir else base_dirs[0]
        if output_dir_path:
            output_dir_path = os.path.abspath(output_dir_path)
            if not os.path.isdir(output_dir_path):
                os.makedirs(output_dir_path)
        else:
            output_dir_path = ''
        output_path = os.path.join(output_dir_path, base_name + '_mosaic.png')
    else:
        output_path = os.path.abspath(output_path)
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        if not output_path.lower().endswith('.png'):
            output_path += '.png'

    # Persist the combined mosaic to disk so it can be inspected later.
    saveImage(output_path, mosaic)
    print('Saved FR mosaic to: {}'.format(output_path))

    return output_path


if __name__ == '__main__':

    # COMMAND LINE ARGUMENTS

    arg_parser = argparse.ArgumentParser(
        description='Create a stacked max-value image from an FR*.bin file.')

    # Source FR*.bin files can either be stacked individually or combined into a mosaic.
    arg_parser.add_argument('fr_file', nargs='+', metavar='FR_FILE', type=str,
        help='Path to one or more FR*.bin files to stack.')

    # Allow overriding the default file path when producing a single stacked image.
    arg_parser.add_argument('-o', '--output', nargs='?', metavar='OUTPUT', type=str,
        help='Optional path for the stacked PNG image.')

    arg_parser.add_argument('--mosaic', action='store_true',
        help='Combine all provided FR files into a single mosaic image.')

    # Accept a directory for the mosaic output so users do not need to specify the full path.
    arg_parser.add_argument('--mosaic-dir', nargs='?', metavar='DIR', type=str,
        help='Optional directory where the mosaic PNG will be stored when --mosaic is used.')

    arg_parser.add_argument('--columns', nargs='?', type=int, metavar='COLS',
        help='Optional number of columns to use when building a mosaic image.')

    args = arg_parser.parse_args()

    # When requested, build a combined mosaic instead of individual stack images.
    if args.mosaic:
        stackFRbins(args.fr_file, output_path=args.output, columns=args.columns,
            output_dir=args.mosaic_dir)
    else:
        if len(args.fr_file) == 1:
            stackFRbin(args.fr_file[0], output_path=args.output)
        else:
            # Prepare a target directory when saving several individual stack outputs.
            if args.output:
                output_dir = os.path.abspath(args.output)

                if os.path.exists(output_dir) and not os.path.isdir(output_dir):
                    raise ValueError('When stacking multiple FR files without a mosaic the output '
                        'path must point to a directory.')

                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)
            else:
                output_dir = None

            for fr_path in args.fr_file:
                if output_dir is None:
                    stackFRbin(fr_path)
                else:
                    base_name = os.path.splitext(os.path.basename(fr_path))[0]
                    target_path = os.path.join(output_dir, base_name + '_stack.png')
                    stackFRbin(fr_path, output_path=target_path)
