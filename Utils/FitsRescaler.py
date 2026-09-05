from __future__ import print_function  # Python 2.7 compatibility for print()
import os
import sys
import argparse
import json
import numpy as np
import cv2
from astropy.io import fits


def rescale_fits(input_path, output_path, target_width=1280, target_height=720):
    """
    Rescales Multi-Extension FITS files from RMS, preserving structure and headers.
    Compatible with both Python 2.7 and Python 3.x.
    """
    print("Processing: {0}...".format(os.path.basename(input_path)))

    # Safe open for both Python 2 and 3 environments
    hdulist = fits.open(input_path)
    try:
        # In older astropy (Py2), the primary header is accessed via hdulist[0].header
        # In newer astropy (Py3), it can be accessed directly from hdulist.header
        try:
            header = hdulist[0].header
        except Exception:
            header = hdulist.header

        # Check current resolution to avoid rescaling already downsampled files
        if header.get('NCOLS') == target_width and header.get('NROWS') == target_height:
            print("File is already {0}x{1}. Skipping.".format(target_width, target_height))
            return False

        # Update dimension metadata in the header
        header['NROWS'] = target_height
        header['NCOLS'] = target_width

        # Initialize a new HDU list with the updated primary header
        new_hdulist = fits.HDUList([fits.PrimaryHDU(header=header)])

        # Rescale each of the 4 image extensions (1: maxpixel, 2: maxframe, 3: avepixel, 4: stdpixel)
        for i in range(1, 5):
            data = hdulist[i].data
            if data is None:
                continue

            # INTER_AREA interpolation preserves photometric integrity
            rescaled_data = cv2.resize(data, (target_width, target_height), interpolation=cv2.INTER_AREA)

            # Preserve the original data type of the layer
            rescaled_data = rescaled_data.astype(data.dtype)

            # Create a new ImageHDU and copy the extension-specific header
            img_hdu = fits.ImageHDU(data=rescaled_data, header=hdulist[i].header)
            new_hdulist.append(img_hdu)

        # Handle API differences between old and new astropy versions
        # Python 2 / old astropy uses 'clobber', Python 3 / modern astropy uses 'overwrite'
        try:
            new_hdulist.writeto(output_path, overwrite=True)
        except TypeError:
            new_hdulist.writeto(output_path, clobber=True)

        print("Successfully saved FITS to: {0}".format(output_path))
        return True

    finally:
        hdulist.close()


def rescale_bin(input_path, output_path, scale_factor=1.5):
    """
    Reads an RMS FR_*.bin file, downscales crop coordinates, crop sizes,
    and resizes the crop images, then writes a new compatible .bin file.
    Works on Python 2.7 and 3.x.
    """
    print("Processing BIN: {0}...".format(os.path.basename(input_path)))

    # Open input binary file
    with open(input_path, "rb") as fid_in:
        # Read total number of lines/tracks
        lines = np.fromfile(fid_in, dtype=np.uint32, count=1)
        if len(lines) == 0:
            print("Empty or corrupted BIN file.")
            return False

        total_lines = lines[0]

        # Open output binary file
        with open(output_path, "wb") as fid_out:
            # Write total lines first
            np.array([total_lines], dtype=np.uint32).tofile(fid_out)

            for i in range(total_lines):
                frame_num_arr = np.fromfile(fid_in, dtype=np.uint32, count=1)
                if len(frame_num_arr) == 0:
                    break
                frame_num = frame_num_arr[0]

                # Write frame number for current line
                np.array([frame_num], dtype=np.uint32).tofile(fid_out)

                for z in range(frame_num):
                    # Read coordinates, timestamp, and crop size
                    yc = int(np.fromfile(fid_in, dtype=np.uint32, count=1)[0])
                    xc = int(np.fromfile(fid_in, dtype=np.uint32, count=1)[0])
                    t = int(np.fromfile(fid_in, dtype=np.uint32, count=1)[0])
                    size = int(np.fromfile(fid_in, dtype=np.uint32, count=1)[0])

                    # Read the raw pixel data for the crop
                    raw_pixels = np.fromfile(fid_in, dtype=np.uint8, count=size**2)
                    frame = np.reshape(raw_pixels, (size, size))

                    # --- RESCALE LOGIC ---
                    # Scale coordinates and size (1080p -> 720p means dividing by 1.5)
                    new_yc = int(round(yc / scale_factor))
                    new_xc = int(round(xc / scale_factor))
                    new_size = int(round(size / scale_factor))

                    # Resize the crop image itself
                    # (Ensure new_size is at least 1x1 to prevent openCV errors on tiny crops)
                    if new_size < 1:
                        new_size = 1
                    rescaled_frame = cv2.resize(frame, (new_size, new_size), interpolation=cv2.INTER_AREA)

                    # Write updated records back to the new binary file
                    np.array([new_yc], dtype=np.uint32).tofile(fid_out)
                    np.array([new_xc], dtype=np.uint32).tofile(fid_out)
                    np.array([t], dtype=np.uint32).tofile(fid_out)
                    np.array([new_size], dtype=np.uint32).tofile(fid_out)
                    rescaled_frame.astype(np.uint8).tofile(fid_out)

    print("Successfully saved BIN to: {0}".format(output_path))
    return True

def rescale_config(input_path, output_path, target_width=1280, target_height=720):
    print("Processing CONFIG: {0}...".format(os.path.basename(input_path)))
    try:
        with open(input_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            cleaned = line.strip().lower()

            # Check for width parameters
            if cleaned.startswith('width') or cleaned.startswith('resolution_width'):
                # Determine separator (":" or "=")
                sep = ":" if ":" in line else "="
                new_lines.append("width{0} {1}\n".format(sep, target_width))

            # Check for height parameters
            elif cleaned.startswith('height') or cleaned.startswith('resolution_height'):
                sep = ":" if ":" in line else "="
                new_lines.append("height{0} {1}\n".format(sep, target_height))

            else:
                new_lines.append(line)

        with open(output_path, 'w') as f:
            f.writelines(new_lines)

        print("Successfully updated CONFIG to 720p.")
        return True
    except Exception as e:
        print("Failed to process CONFIG: {0}".format(e))
        return False

def rescale_calibration(input_path, output_path, scale_factor=1.5, target_width=1280, target_height=720):
    """
    Parses platepar_cmn2010.cal JSON file, modifies focal scale and dimensions,
    and clears 1080p star_list to align with 720p frames.
    """
    print("Processing CALIBRATION: {0}...".format(os.path.basename(input_path)))
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)

        # 1. Update resolution references
        data['X_res'] = target_width
        data['Y_res'] = target_height

        # 2. Rescale focal parameters to preserve true FOV
        if 'F_scale' in data:
            data['F_scale'] = data['F_scale'] / scale_factor

        # # 3. Clear or reset star list because it contains old 1080p absolute pixel positions
        # if 'star_list' in data:
        #     data['star_list'] = []

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)

        print("Successfully updated CALIBRATION (.cal) to 720p.")
        return True
    except Exception as e:
        print("Failed to process CALIBRATION: {0}".format(e))
        return False

def process_path(input_path, output_dir=None):
    """
    Handles the logic for processing either a single file or an entire directory.
    """
    # If no output directory is provided, create a 'rescaled_720p' subdirectory
    if not output_dir:
        if os.path.isdir(input_path):
            output_dir = os.path.join(input_path, "rescaled_720p")
        else:
            output_dir = os.path.join(os.path.dirname(input_path), "rescaled_720p")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process directory
    if os.path.isdir(input_path):
        all_files = os.listdir(input_path)

        # Independent scan for FF_ fits files
        fits_files = [f for f in all_files if f.lower().startswith('ff_') and f.lower().endswith('.fits')]
        # Independent scan for FR_ bin files
        bin_files = [f for f in all_files if f.lower().startswith('fr_') and f.lower().endswith('.bin')]
        config_files = [f for f in all_files if f.lower() == '.config']
        cal_files = [f for f in all_files if f.lower().endswith('.cal')]

        print("Found {0} FITS, {1} BIN, {2} CONFIG, and {3} CAL files.".format(
            len(fits_files), len(bin_files), len(config_files), len(cal_files)))

        for file in fits_files: rescale_fits(os.path.join(input_path, file), os.path.join(output_dir, file))
        for file in bin_files:  rescale_bin(os.path.join(input_path, file), os.path.join(output_dir, file))
        for file in config_files: rescale_config(os.path.join(input_path, file), os.path.join(output_dir, file))
        for file in cal_files: rescale_calibration(os.path.join(input_path, file), os.path.join(output_dir, file))

    elif os.path.isfile(input_path):
        filename = os.path.basename(input_path)
        full_output_path = os.path.join(output_dir, filename)
        ext = os.path.splitext(filename).lower()

        if filename.lower().startswith('ff_') and ext in ['.fits', '.fit']: rescale_fits(input_path, full_output_path)
        elif filename.lower().startswith('fr_') and ext == '.bin': rescale_bin(input_path, full_output_path)
        elif filename.lower() == '.config': rescale_config(input_path, full_output_path)
        elif ext == '.cal': rescale_calibration(input_path, full_output_path)
        else: print("Unknown file type. Processing stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RMS FITS & BIN Rescaler 1080p -> 720p")
    parser.add_argument(
        "path",
        type=str,
        help="Path to a single file (FITS/BIN) or a directory containing RMS products"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to the output directory (defaults to 'rescaled_720p')"
    )

    args = parser.parse_args()
    process_path(args.path, args.output)
