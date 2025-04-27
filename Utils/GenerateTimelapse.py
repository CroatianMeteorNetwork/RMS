""" Generate an High Quality MP4 movie from FF files. 
    Contributors: Tioga Gulon
"""

from __future__ import print_function, division, absolute_import

import sys
import os
import re
import platform
import subprocess
import shutil
import cv2
import json
from datetime import datetime, timedelta

from PIL import ImageFont

from RMS.Formats.FFfile import read as readFF
from RMS.Formats.FFfile import validFFName, filenameToDatetime
from RMS.Misc import mkdirP, RmsDateTime, tarWithProgress
from RMS.Logger import getLogger

log = getLogger("logger")

IMAGE_PATTERN = re.compile(
    r"""
    ^(?P<station>.+?)_
      (?P<date>\d{8})_            # YYYYMMDD
      (?P<time>\d{6})_            # HHMMSS
      (?P<msec>\d{3})             # milliseconds
    (?:_(?P<suffix>[dn]))?        # optional _d / _n
    \.(?P<ext>jpg|png)$
    """,
    re.VERBOSE | re.IGNORECASE,
)

ONE_DAY = timedelta(days=1)


def _timestampFromName(fname):
    """Parse the UTC timestamp in the filename and return a *naïve* datetime."""
    m = IMAGE_PATTERN.match(os.path.basename(fname))
    if not m:
        raise ValueError("Bad filename: {}".format(fname))
    stamp = "{}{}{}".format(m.group("date"), m.group("time"), m.group("msec"))
    return datetime.strptime(stamp, "%Y%m%d%H%M%S%f")  # no tzinfo


def _modeFromName(fname):
    m = IMAGE_PATTERN.match(os.path.basename(fname))
    return (m.group("suffix") or '').lower()           # '', 'd', or 'n'


def listImageBlocksBefore(cutoff, dir_path):
    """
    cutoff   : naïve datetime assumed to be in UTC
    dir_path : directory tree to walk

    Returns  : list[list[str]] - blocks that are
               • consecutive in time
               • homogeneous in mode ('', 'd', 'n')
               • never longer than 24 h (split only if >24 h)
    """
    # 1. collect .jpg / .png whose embedded time < cutoff -------------------
    paths = []
    for root, _, files in os.walk(dir_path):
        for fname in files:
            if not IMAGE_PATTERN.match(fname):
                continue
            ts = _timestampFromName(fname)
            if ts >= cutoff:
                continue
            paths.append(os.path.join(root, fname))

    if not paths:
        return []

    # 2. chronological sort -------------------------------------------------
    paths.sort(key=_timestampFromName)

    # 3. first pass – break on mode changes ---------------------------------
    prelim_blocks, cur_block, cur_mode = [], [], None
    for path in paths:
        mode = _modeFromName(path)
        if cur_block and mode != cur_mode:          # mode switch ⇒ new block
            prelim_blocks.append(cur_block)
            cur_block = []
        cur_block.append(path)
        cur_mode = mode
    if cur_block:
        prelim_blocks.append(cur_block)

    # 4. second pass – split blocks that run > 24 h at each UTC midnight ----
    final_blocks = []
    for block in prelim_blocks:
        start_ts = _timestampFromName(block[0])
        end_ts = _timestampFromName(block[-1])

        if end_ts - start_ts < ONE_DAY:             # already ≤24 h
            final_blocks.append(block)
            continue

        next_midnight = (start_ts + ONE_DAY).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        chunk = []
        for path in block:
            ts = _timestampFromName(path)
            if ts >= next_midnight:                 # crossed midnight
                final_blocks.append(chunk)
                chunk = []
                while ts >= next_midnight:
                    next_midnight += ONE_DAY
            chunk.append(path)
        if chunk:
            final_blocks.append(chunk)

    return final_blocks


_name_re = re.compile(r"""
    (?P<station>[^_]+)_          # station  (e.g. US005C)
    (?P<date>\d{8})_             # YYYYMMDD
    (?P<time>\d{6})_             # HHMMSS
    """, re.VERBOSE)


def _parse(fname):
    """Return (station, datetime) extracted from an image filename."""
    m = _name_re.match(os.path.basename(fname))
    if not m:
        raise ValueError(f"Cannot parse name: {fname}")
    dt = datetime.strptime(m["date"] + m["time"], "%Y%m%d%H%M%S")
    return m["station"], dt


def _video_path(frames_root, first_img, last_img):
    """Build .../STN_YYYYMMDD_[doy_hhmm-to-doy_hhmm]_frames_timelapse.mp4"""
    station, t0 = _parse(first_img)
    _,       t1 = _parse(last_img)

    doy0, doy1 = t0.timetuple().tm_yday, t1.timetuple().tm_yday
    name = (f"{station}_{t0:%Y%m%d}_[{doy0:03d}_{t0:%H%M}-"
            f"to-{doy1:03d}_{t1:%H%M}]_frames_timelapse.mp4")
    return os.path.join(frames_root, name)


def generateTimelapseFromFrameBlocks(frame_blocks,
                                     frames_root,
                                     fps=30,
                                     base_crf=25,
                                     cleanup_mode='none',
                                     compression='bz2',
                                     use_color=True):
    """
    Create one timelapse per block and collect the resulting paths.

    Parameters
    ----------
    frame_blocks : list[list[str]]
        Each sub-list is a chronologically ordered set of image paths.
    frames_root  : str
        Directory where the timelapse files will be written.

    Returns
    -------
    list[tuple[str, str] | None]
        (mp4_path, json_path) per block, or None if that block failed.
    """
    results = []
    for block in frame_blocks:
        if not block:                       # empty block → skip
            results.append(None)
            continue

        first_img, last_img = block[0], block[-1]
        mp4_path = _video_path(frames_root, first_img, last_img)

        mp4_path, json_path = generateTimelapseFromFrames(
            image_files=block,
            video_path=mp4_path,
            fps=fps,
            base_crf=base_crf,
            cleanup_mode=cleanup_mode,
            compression=compression,
            use_color=use_color
        )

        results.append((mp4_path, json_path) if mp4_path else None)

    return results


def deleteFilesAndEmptyDirs(file_paths):
    """Delete files and empty directories.
    
    Arguments:
        file_paths: [list] List of file paths to delete.
    """
    
    for file_path in file_paths:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                log.error("Error removing file {}: {}".format(file_path, e))
    
    # Remove empty directories
    for dir_path in set(os.path.dirname(p) for p in file_paths):
        if os.path.exists(dir_path) and not os.listdir(dir_path):
            try:
                os.rmdir(dir_path)
                # log.info("Removed empty directory: {}".format(dir_path))
            except Exception as e:
                log.error("Error removing directory {}: {}".format(dir_path, e))


def generateTimelapseFromFrames(image_files,
                                video_path,
                                fps=30,
                                base_crf=25,
                                cleanup_mode='none',
                                compression='bz2',
                                use_color=True):
    """
    Generate a timelapse video using streaming to avoid temporary storage.
    
    Parameters:
        image_files: [list] List of image file paths.
        video_path: [str] Output path for the generated video.
        fps: [int] Frames per second for the output video.
        base_crf: [int] Base Constant Rate Factor for video compression. Used as is for color video and increased
                    by 2 for grayscale.
        cleanup_mode: [str] Cleanup mode after video creation ('none', 'delete', 'tar').
        compression: [str] Compression method for tar ('bz2', 'gz').
        use_color: [bool] Whether to create a color video (True) or grayscale video (False).
    Returns
        (video_path, json_path) on success, (None, None) on failure.
    """
    # Validate input parameters
    image_files = list(image_files)
    total_frames = len(image_files)
    if total_frames == 0:
        log.warning('generateTimelapseFromFrames: no images supplied')
        return None, None
    
    # Create a temporary output path
    output_dir = os.path.dirname(video_path)
    output_filename = os.path.basename(video_path)
    output_name, output_ext = os.path.splitext(output_filename)
    temp_video_path = os.path.join(output_dir, "{}_temp{}".format(output_name, output_ext))
    
    # Process a valid first image to get dimensions
    # Try up to 10 images to find a valid one for dimensions
    # Necessary for the streaming method
    sample_size = min(10, len(image_files))
    found_valid_image = False
    width, height = 0, 0
    
    for i in range(sample_size):
        sample_image = cv2.imread(image_files[i], cv2.IMREAD_UNCHANGED)
        if sample_image is not None:
            # Get dimensions from the first valid image
            height, width = sample_image.shape[:2]

            # Determine if the image is grayscale or color
            channels = 1 if sample_image.ndim == 2 else sample_image.shape[2]

            is_color = channels > 1

            found_valid_image = True
            break
    
    if not found_valid_image:
        log.error("Error: Could not find any valid images to determine dimensions.")
        return None, None
    
    # Set up ffmpeg command based on color mode
    if platform.system() in ['Linux', 'Darwin']:
        ffmpeg_path = "ffmpeg"
    elif platform.system() == 'Windows':
        ffmpeg_path = os.path.join(os.path.dirname(__file__), "ffmpeg.exe")
    else:
        log.warning("Unsupported platform.")
        return None, None

    # Configure ffmpeg command based on color mode
    if use_color and is_color:
        # If the image is color, use the base CRF value
        crf = base_crf

        # Set pixel format for color
        pix_fmt = "bgr24"
    else:
        # If the image is grayscale, increase CRF by 2
        crf = base_crf + 2

        # Set pixel format for grayscale
        pix_fmt = "gray"

    ffmpeg_cmd = [ffmpeg_path, "-y", "-nostdin",
                  "-f", "rawvideo", 
                  "-vcodec", "rawvideo",
                  "-s", "{}x{}".format(width, height),
                  "-pix_fmt", pix_fmt,
                  "-r", str(fps),
                  "-i", "-",
                  "-c:v", "libx264",
                  "-crf", str(crf),
                  "-profile:v", "high", # baseline, main, high
                  "-preset", "medium",  # fast, medium, slow to change size/processing speed
                  "-pix_fmt", "yuv420p",
                  "-movflags", "faststart",  # Optimize for streaming
                  "-threads", "1",
                  "-g", "120",
                  temp_video_path]  # Use temporary path
    
    log.info("Starting ffmpeg process for {}...".format(video_path))
    log.info("Video mode: {}".format('Color' if use_color else 'Grayscale'))
    ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    
    # Initialize timestamp JSON data
    timestamp_data = {}
    
    # Process frames
    log.info("Processing {} frames...".format(len(image_files)))
    processed_count = 0
    skipped_count = 0
    
    for index, img_path in enumerate(image_files):
        if index % 100 == 0:
            print("Processing frame {}/{} ({:.1f}%)"
                  .format(index, len(image_files), (index/len(image_files)*100.0)))
        
        # Load image with error handling
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            log.warning("Warning: Skipping corrupted or unreadable image: {}".format(img_path))
            skipped_count += 1
            continue  # Skip this frame
        
        # Extract timestamp from filename
        try:
            image_name_parts = os.path.basename(img_path).split('_')
            station_id = image_name_parts[0]
            extracted_time = datetime.strptime(image_name_parts[1] + image_name_parts[2], "%Y%m%d%H%M%S")
            
            # Record timestamp data (only for frames that are successfully processed)
            timestamp_data[str(processed_count)] = "_".join(os.path.basename(img_path).split('.')[0].split('_')[1:])

            # Add timestamp with outline for better visibility
            text = station_id + " " + extracted_time.strftime("%Y-%m-%d %H:%M:%S") + " UTC"
            position = (10, image.shape[0] - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            
            # Create text with outline (black background, white text)
            cv2.putText(image, text, position, font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(image, text, position, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            
        except Exception as e:
            log.warning("Warning: Error processing metadata for {}: {}".format(img_path, e))
        
        # Handle color conversion based on target mode
        try:
            if not use_color and is_color:
                # Convert to grayscale if color
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                     
            # Write frame to ffmpeg
            ffmpeg_process.stdin.write(image.tobytes())
            processed_count += 1
            
        except Exception as e:
            log.error("Warning: Error processing image {}: {}".format(img_path, e))
            skipped_count += 1
    
    # Create a temporary timestamp JSON file
    timestamp_path = video_path.replace('frames_timelapse.mp4', 'frametimes.json')
    temp_timestamp_path = os.path.join(output_dir, "{}_temp_timestamps.json".format(output_name))
    
    try:
        with open(temp_timestamp_path, 'w') as f:
            json.dump(timestamp_data, f, indent=2)
    except Exception as e:
        log.warning("Warning: Error saving timestamp data: {}".format(e))
    
    # Finalize video
    log.info("All frames processed. Successfully processed: {}, Skipped: {}"
             .format(processed_count, skipped_count))
    log.info("Finalizing video...")
    
    try:
        ffmpeg_process.stdin.close()
        return_code = ffmpeg_process.wait(timeout=300)  # Wait up to 5 minutes
        
        if return_code != 0:
            log.warning("Warning: ffmpeg process exited with code {}".format(return_code))
    except subprocess.TimeoutExpired:
        log.warning("Warning: ffmpeg process did not complete within timeout, terminating...")
        ffmpeg_process.terminate()
        try:
            ffmpeg_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            ffmpeg_process.kill()
    
    # Check result and handle cleanup
    if os.path.exists(temp_video_path) and os.path.getsize(temp_video_path) > 0:
        video_size_mb = os.path.getsize(temp_video_path) / (1024 * 1024)
        
        # Rename temporary files to final paths
        try:
            # Remove destination file if it exists
            if os.path.exists(video_path):
                os.remove(video_path)
            
            # Rename video file
            os.rename(temp_video_path, video_path)
            log.info("Video created successfully: {} ({:.2f} MB)".format(video_path, video_size_mb))
            
            # Rename timestamp file if it exists
            if os.path.exists(temp_timestamp_path):
                if os.path.exists(timestamp_path):
                    os.remove(timestamp_path)
                os.rename(temp_timestamp_path, timestamp_path)
                log.info("Timestamp data saved to: {}".format(timestamp_path))
        
            # Handle cleanup based on specified mode
            if cleanup_mode == 'delete':
                deleteFilesAndEmptyDirs(image_files)

            elif cleanup_mode == 'tar':
                try:
                    ext = '.tar.bz2' if compression == 'bz2' else '.tar.gz'

                    # Derive the tar file name from video_path, stripping '_timelapse.mp4'
                    base_name = os.path.basename(video_path).replace('_timelapse.mp4', '')

                    # Construct the full path for the tar file
                    tar_path = os.path.join(os.path.dirname(video_path), base_name + ext)

                    # Create a temporary tar file
                    temp_tar_path = tar_path + ".tmp"
                    
                    log.info("Creating {} archive of {}...".format(compression, base_name))
                    
                    # Determine if we should remove the source files based on cleanup_mode
                    remove_source = cleanup_mode == 'tar'
                    
                    # Create tar with progress reporting, verification, and optional source removal
                    archive_success = tarWithProgress(None,
                                                      temp_tar_path,
                                                      compression,
                                                      remove_source, 
                                                      file_list=image_files)
                    
                    if archive_success:
                        # Rename to final tar path
                        if os.path.exists(tar_path):
                            os.remove(tar_path)
                        os.rename(temp_tar_path, tar_path)
                        log.info("Archive created successfully at: {}".format(tar_path))
                        if remove_source:
                            # Remove source files if archive creation was successful
                            deleteFilesAndEmptyDirs(image_files)
                            log.info("Removed source files after archiving.")
                    else:
                        log.warning("Archive creation or verification failed. Keeping original directory.")
                        # Clean up temporary tar file if it exists
                        if os.path.exists(temp_tar_path):
                            try:
                                os.remove(temp_tar_path)
                                log.info("Removed incomplete archive file")
                            except:
                                pass
                    
                except Exception as e:
                    log.error("Error in archiving process: {}".format(e))
                    # Clean up temporary tar file if it exists
                    if os.path.exists(temp_tar_path):
                        try:
                            os.remove(temp_tar_path)
                        except:
                            pass

        except Exception as e:
            log.error("Error finalizing files: {}".format(e))
            log.info("Temporary file remains at: {}".format(temp_video_path))
        
        return video_path, timestamp_path
   
    else:
        log.warning("Video creation failed or resulted in an empty file.")
        # Clean up temporary files
        for temp_file in [temp_video_path, temp_timestamp_path]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    log.info("Removed incomplete temporary file: {}".format(temp_file))
                except:
                    pass
        return None, None
    


def generateTimelapse(dir_path, keep_images=False, fps=None, output_file=None, hires=False):
    """ Generate an High Quality MP4 movie from FF files. 
    
    Arguments:
        dir_path: [str] Path to the directory containing the FF files.

    Keyword arguments:
        keep_images: [bool] Keep the temporary images. False by default.
        fps: [int] Frames per second. 30 by default.
        output_file: [str] Output file name. If None, the file name will be the same as the directory name.
        hires: [bool] Make a higher resolution timelapse. False by default.
    """

    # Set the default FPS if not given
    if fps is None:
        fps = 30


    # Make the name of the output file if not given
    if output_file is None:
        mp4_path = os.path.join(dir_path, os.path.basename(dir_path) + ".mp4")

    else:
        mp4_path = os.path.join(dir_path, os.path.basename(output_file))


    # Set the CRF value for the video compression depending on the resolution
    if hires:
        crf = 25

    else:
        crf = 29


    t1 = RmsDateTime.utcnow()

    # Load the font for labeling
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 18)
    except:
        font = ImageFont.load_default()

    # Create temporary directory
    dir_tmp_path = os.path.join(dir_path, "temp_img_dir")

    if os.path.exists(dir_tmp_path):
        shutil.rmtree(dir_tmp_path)
        log.info("Directory removal complete: {}".format(dir_tmp_path))
		
    mkdirP(dir_tmp_path)
    log.info("Created directory : " + dir_tmp_path)
    
    log.info("Preparing files for the timelapse...")
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
		
        # Get id cam from the file name
        # e.g.  FF499_20170626_020520_353_0005120.bin
        # or FF_CA0001_20170626_020520_353_0005120.fits

        file_split = file_name.split('_')

        # Check the number of list elements, and the new fits format has one more underscore
        i = 0
        if len(file_split[0]) == 2:
            i = 1
        camid = file_split[i]

        # Make a filename for the image, continuous count %04d
        img_file_name = 'temp_{:04d}.jpg'.format(c)

        img = ff.maxpixel

        # Draw text to image
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = camid + " " + timestamp + " UTC"
        cv2.putText(img, text, (10, ff.nrows - 6), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # Save the labelled image to disk
        cv2.imwrite(os.path.join(dir_tmp_path, img_file_name), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    
        c = c + 1

        # Print elapsed time
        if c % 30 == 0:
            print("{:>5d}/{:>5d}, Elapsed: {:s}".format(c + 1, len(ff_list), \
                  str(RmsDateTime.utcnow() - t1)), end="\r")
            sys.stdout.flush()


    # If running on Linux, use avconv if available
    if platform.system() == 'Linux':

        # If avconv is not found, try using ffmpeg. In case of using ffmpeg,
        # use parameter -nostdin to avoid it being stuck waiting for user input
        software_name = "avconv"
        nostdin = ""
        log.info("Checking if avconv is available...")
        if os.system(software_name + " --help > /dev/null"):
            software_name = "ffmpeg"
            nostdin =  " -nostdin "
        
        # Construct the command for avconv            
        temp_img_path = os.path.basename(dir_tmp_path) + os.sep + "temp_%04d.jpg"
        com = "cd " + dir_path + ";" \
            + software_name + nostdin + " -v quiet -r "+ str(fps) +" -y -i " + temp_img_path \
            + " -vcodec libx264 -pix_fmt yuv420p -crf " + str(crf) \
            + " -movflags faststart -threads 2 -g 15 -vf \"hqdn3d=4:3:6:4.5,lutyuv=y=gammaval(0.77)\" " \
            + mp4_path

        log.info("Creating timelapse using {:s}...".format(software_name))
        log.info(com)
        subprocess.call([com], shell=True)


    # If running on Windows, use ffmpeg.exe
    elif platform.system() == 'Windows':
	
        # ffmpeg.exe path
        root = os.path.dirname(__file__)
        ffmpeg_path = os.path.join(root, "ffmpeg.exe")
	
        # Construct the ecommand for ffmpeg
        temp_img_path = os.path.join(os.path.basename(dir_tmp_path), "temp_%04d.jpg")
        com = ffmpeg_path + " -v quiet -r " + str(fps) + " -i " + temp_img_path \
            + " -c:v libx264 -pix_fmt yuv420p -an -crf " + str(crf) \
            + " -g 15 -vf \"hqdn3d=4:3:6:4.5,lutyuv=y=gammaval(0.77)\" -movflags faststart -y " \
            + mp4_path
		
        log.info("Creating timelapse using ffmpeg...")
        log.info(com)
        subprocess.call(com, shell=True, cwd=dir_path)
		
    else :
        log.warning("generateTimelapse only works on Linux or Windows the video could not be encoded")

    #Delete temporary directory and files inside
    if os.path.exists(dir_tmp_path) and not keep_images:
        shutil.rmtree(dir_tmp_path)
        log.info("Directory removal complete: {}".format(dir_tmp_path))
		
    log.info("Total time: %s", RmsDateTime.utcnow() - t1)



def main():
    """
    Test function for the generateTimelapseFromFrames function.
    This allows testing the function with different parameters from the command line.
    """
    import argparse
    import os
    from datetime import datetime
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Generate a timelapse video from image frames.')
    parser.add_argument('input_dir', help='Directory containing image frames organized in hour subdirectories')
    parser.add_argument('--output', '-o', help='Output video path. Default: [input_dir]_timelapse.mp4')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second (default: 30)')
    parser.add_argument('--crf', type=int, default=25, help='Constant Rate Factor for compression (default: 27)')
    parser.add_argument('--cleanup', choices=['none', 'delete', 'tar'], default='none',
                      help='Cleanup mode after processing (default: none)')
    parser.add_argument('--compression', choices=['bz2', 'gz'], default='bz2',
                      help='Compression method for tar (default: bz2)')
    parser.add_argument('--grayscale', action='store_true', 
                      help='Create grayscale video instead of color')
    
    args = parser.parse_args()
    
    # Set default output path if not specified
    if not args.output:
        input_dir_name = os.path.basename(os.path.normpath(args.input_dir))
        args.output = os.path.join(os.path.dirname(args.input_dir), 
                                  f"{input_dir_name}_timelapse.mp4")
    
    # Print configuration
    print("Timelapse Generator Configuration:")
    print(f"  Input directory: {args.input_dir}")
    print(f"  Output video: {args.output}")
    print(f"  FPS: {args.fps}")
    print(f"  CRF value: {args.crf}")
    print(f"  Cleanup mode: {args.cleanup}")
    if args.cleanup == 'tar':
        print(f"  Compression: {args.compression}")
    print(f"  Video mode: {'Grayscale' if args.grayscale else 'Color'}")
    
    # Record start time
    start_time = datetime.now()
    print(f"Starting process at: {start_time}")
    
    # Generate the timelapse
    try:
        generateTimelapseFromFrames(
            day_dir=args.input_dir,
            video_path=args.output,
            fps=args.fps,
            base_crf=args.crf,
            cleanup_mode=args.cleanup,
            compression=args.compression,
            use_color=not args.grayscale
        )
        
        # Record and print completion time and duration
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"Process completed at: {end_time}")
        print(f"Total processing time: {duration}")
        
        # Print file sizes if successful
        if os.path.exists(args.output):
            video_size = os.path.getsize(args.output) / (1024 * 1024)  # Convert to MB
            print(f"Output video size: {video_size:.2f} MB")
            
            # Check if tar was created
            if args.cleanup == 'tar':
                ext = '.tar.bz2' if args.compression == 'bz2' else '.tar.gz'
                base_name = os.path.basename(args.output).replace('_timelapse.mp4', '')
                tar_path = os.path.join(os.path.dirname(args.output), base_name + ext)
                
                if os.path.exists(tar_path):
                    tar_size = os.path.getsize(tar_path) / (1024 * 1024)  # Convert to MB
                    print(f"Archive size: {tar_size:.2f} MB")
        
    except Exception as e:
        print(f"Error generating timelapse: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    
    # Run the main function
    exit(main())
