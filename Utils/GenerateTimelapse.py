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


# --------------------------------------------------------------------
#  Timelapse generation from FF files (meteors)
# --------------------------------------------------------------------
def generateTimelapse(dir_path, keep_images=False, fps=None, output_file=None, hires=False):
    """Generate a high-quality MP4 movie from FF files.

    Arguments:
        dir_path: [str] Directory that contains the FF files.

    Keyword arguments:
        keep_images: [bool] Retain the temporary JPGs after encoding.
            False by default.
        fps: [int] Frames per second; 30 by default.
        output_file: [str] Custom output filename. If None, the basename of
            *dir_path* is used. None by default.
        hires: [bool] Produce a higher-resolution movie (lower CRF). False by
            default.

    Return:
        None
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
    log.info("Created directory : {}".format(dir_tmp_path))
    
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
        position = (10, ff.nrows - 6)
        font_scale = 0.4
        thickness = 1

        cv2.putText(img, text, position, font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img, text, position, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Save the labelled image to disk
        cv2.imwrite(os.path.join(dir_tmp_path, img_file_name), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    
        c = c + 1

        # Print elapsed time
        if c % 30 == 0:
            print("{:>5d}/{:>5d}, Elapsed: {:s}".format(c, len(ff_list), \
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

        log.info("Creating timelapse using {}...".format(software_name))
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


# --------------------------------------------------------------------
#  Timelapse generation from image files (Contrails)
# --------------------------------------------------------------------

#  Output-naming helpers - one place to tweak naming scheme
FNAME_TEMPLATE = (
    "{station}_{start:%Y%m%d-%H%M%S}_to_{end:%Y%m%d-%H%M%S}_{suffix}"
)

MP4_SUFFIX = "frames_timelapse.mp4"
TS_JSON_SUFFIX = "frametimes.json"
TAR_SUFFIX_BZ2 = "frames_timelapse.tar.bz2"
TAR_SUFFIX_GZ = "frames_timelapse.tar.gz"

# Source image filename pattern
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


def _buildName(station, t0, t1, suffix):
    """Return the timelapse base filename derived from start/end timestamps.

    Arguments:
        station: [str] Station identifier.
        t0: [datetime] UTC timestamp of the first frame in the block.
        t1: [datetime] UTC timestamp of the last frame in the block.
        suffix: [str] Filename suffix such as ``frames_timelapse.mp4``.

    Return:
        fname: [str] Basename (no directory) built from *FNAME_TEMPLATE*.
    """
    return FNAME_TEMPLATE.format(
        station=station,
        doy_start=t0.timetuple().tm_yday,
        # doy_end=t1.timetuple().tm_yday,
        start=t0,
        end=t1,
        suffix=suffix,
    )


def _timestampFromName(fname):
    """Extract the UTC timestamp embedded in the filename.

    Arguments:
        fname: [str] Image filename.

    Return:
        ts: [datetime] Naive datetime parsed from the name (no timezone).
    """
    m = IMAGE_PATTERN.match(os.path.basename(fname))
    if not m:
        raise ValueError("Bad filename: {}".format(fname))
    stamp = "{}{}{}".format(m.group("date"), m.group("time"), m.group("msec"))
    return datetime.strptime(stamp, "%Y%m%d%H%M%S%f")  # no tzinfo


def _modeFromName(fname):
    """Return '' (unknown), 'd' (day) or 'n' (night) encoded in the filename.

    Arguments:
        fname: [str] Image filename.

    Return:
        mode: [str] One of '', 'd', 'n' in lower-case.
    """
    m = IMAGE_PATTERN.match(os.path.basename(fname))
    return (m.group("suffix") or '').lower()           # '', 'd', or 'n'


def _parse(fname):
    """Return *(station, datetime)* extracted from an image filename.

    Arguments:
        fname: [str] Image filename.

    Return:
        station: [str] Station identifier.
        dt: [datetime] UTC timestamp (naive) truncated to whole seconds.
    """
    m = IMAGE_PATTERN.match(os.path.basename(fname))
    if not m:
        raise ValueError("Cannot parse name: {}".format(fname))

    station = m.group("station")
    # Ignore milliseconds here; keep them if you ever need sub-second precision
    dt = datetime.strptime(m.group("date") + m.group("time"),
                           "%Y%m%d%H%M%S")
    return station, dt


def listImageBlocksBefore(cutoff, dir_path):
    """Group images into chronological, same-mode blocks before a cutoff.

    Arguments:
        cutoff: [datetime] Naive UTC timestamp; images >= cutoff are ignored.
        dir_path: [str] Root directory to search (walks sub-dirs recursively).

    Return:
        blocks: [list[list[str]]] Each sub-list is a consecutive sequence of
            image paths that share capture mode and never span more than
            24 h.
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

    # 3. first pass - break on mode changes ---------------------------------
    prelim_blocks, cur_block, cur_mode = [], [], None
    for path in paths:
        mode = _modeFromName(path)
        if cur_block and mode != cur_mode:          # mode switch - new block
            prelim_blocks.append(cur_block)
            cur_block = []
        cur_block.append(path)
        cur_mode = mode
    if cur_block:
        prelim_blocks.append(cur_block)

    # 4. second pass - split blocks that run > 24 h at each UTC midnight ----
    final_blocks = []
    for block in prelim_blocks:
        start_ts = _timestampFromName(block[0])
        end_ts = _timestampFromName(block[-1])

        if end_ts - start_ts < ONE_DAY:             # already <=24 h
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


def deleteFilesAndEmptyDirs(file_paths, stop_at=None, max_up=3):
    """Delete files then prune empty parent directories.

    Arguments:
        file_paths: [list[str]] Paths to delete.

    Keyword arguments:
        stop_at: [str | None] Directory that acts as an upper boundary when
            pruning. Defaults to the deepest common parent of *file_paths*.
        max_up: [int] Maximum directory levels to climb above each file before
            stopping the prune. 3 by default.

    Return:
        None
    """

    if not file_paths:
        return

    if stop_at is None:
        # deepest common parent of all files
        stop_at = os.path.commonpath(file_paths)

    stop_at = os.path.abspath(stop_at)

    # ---------- pass 1: delete & record candidate dirs ----------
    prune_these = set()
    for file_path in file_paths:
        try:
            os.remove(file_path)
        except OSError:
            pass

        # climb up to max_up collecting dirs beneath stop_at
        current = os.path.dirname(file_path)
        hops = 0
        while hops < max_up and os.path.commonpath([stop_at, current]) == stop_at:
            prune_these.add(current)
            if current == stop_at:
                break
            current = os.path.dirname(current)
            hops += 1

    # ---------- pass 2: bottom-up prune just those dirs ----------
    for dir in sorted(prune_these, key=lambda p: p.count(os.sep), reverse=True):
        if os.path.samefile(dir, stop_at):
            continue            # never delete the anchor
        try:
            os.rmdir(dir)         # succeeds only if empty
        except OSError:
            pass                # not empty, leave it


def generateTimelapseFromFrameBlocks(frame_blocks,
                                     frames_root,
                                     fps=30,
                                     base_crf=25,
                                     cleanup_mode='none',
                                     compression='bz2',
                                     use_color=True):
    """Create one timelapse per block and return their paths.

    Arguments:
        frame_blocks: [list[list[str]]] Chronologically ordered image sets.
        frames_root: [str] Directory where output files will be written.

    Keyword arguments:
        fps: [int] Frames per second; 30 by default.
        base_crf: [int] Base CRF for H.264 encoding; 25 by default.
        cleanup_mode: [str] Post-encode cleanup: 'none', 'delete', or 'tar'.
            'none' by default.
        compression: [str] Tar compression when *cleanup_mode* == 'tar';
            'bz2' or 'gz'. 'bz2' by default.
        use_color: [bool] Encode colour if possible. True by default.

    Return:
        results: [list[tuple[str, str] | None]] One *(mp4_path, json_path)*
            per block, or *None* if that block failed.
    """
    results = []
    for block in frame_blocks:
        if not block:
            results.append(None)
            continue

        first_img, last_img = block[0], block[-1]

        station, t0 = _parse(first_img)
        _,       t1 = _parse(last_img)

        video_name = _buildName(station, t0, t1, MP4_SUFFIX)

        mp4_path_in =  os.path.join(frames_root, video_name)

        log.info("Generating timelapse for %s (%d frames)", video_name, len(block))

        mp4_path, json_path = generateTimelapseFromFrames(
            image_files=block,
            frames_root=frames_root,
            video_path=mp4_path_in,
            fps=fps,
            base_crf=base_crf,
            cleanup_mode=cleanup_mode,
            compression=compression,
            use_color=use_color
        )

        results.append((mp4_path, json_path) if mp4_path else None)

    return results


def generateTimelapseFromDir(dir_path,
                             frames_root=None,
                             video_path=None,
                             fps=30,
                             base_crf=25,
                             cleanup_mode="none",
                             compression="bz2",
                             use_color=True,
                             ):
    """Build a single timelapse from every image under *dir_path*.

    Arguments:
        dir_path: [str] Root directory containing the images.

    Keyword arguments:
        frames_root: [str | None] Override for the output directory.
        video_path: [str | None] Explicit output path; constructed if None.
        fps: [int] Frames per second; 30 by default.
        base_crf: [int] Base CRF for H.264; 25 by default.
        cleanup_mode: [str] Post-encode action: 'none', 'delete', or 'tar'.
        compression: [str] Tar compression when *cleanup_mode* == 'tar'.
        use_color: [bool] Encode colour if possible. True by default.

    Return:
        (video_path, json_path): [tuple[str, str] | (None, None)]
            Paths on success, (None, None) on failure.
    """
    # 1 - gather images -----------------------------------------------------
    img_paths = []
    for root, _, files in os.walk(dir_path):         # use os.listdir(dir_path) if
        for f in files:                              # you *don't* want recursion
            if IMAGE_PATTERN.match(f):
                img_paths.append(os.path.join(root, f))

    if not img_paths:
        log.warning("generateTimelapseFromDir: no images found in %s", dir_path)
        return None, None

    img_paths.sort(key=_timestampFromName)

    # 2 - build output paths ------------------------------------------------
    station, t0 = _parse(img_paths[0])
    _,       t1 = _parse(img_paths[-1])

    if video_path is None:
        video_name = _buildName(station, t0, t1, MP4_SUFFIX)
        video_path = os.path.join(dir_path, video_name)

    # 3 - delegate to the frame-streamer ------------------------------------
    return generateTimelapseFromFrames(
        image_files=img_paths,
        frames_root=frames_root,
        video_path=video_path,
        fps=fps,
        base_crf=base_crf,
        cleanup_mode=cleanup_mode,
        compression=compression,
        use_color=use_color,
    )

def isFfmpegWorking(ffmpeg_path="ffmpeg"):
    """ Check if ffmpeg is available and working. """

    try:
        subprocess.check_call([ffmpeg_path, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    
    except Exception:
        log.warning("ffmpeg is not available or not working.")
        return False


def generateTimelapseFromFrames(image_files,
                                frames_root,
                                video_path,
                                fps=30,
                                base_crf=25,
                                cleanup_mode='none',
                                compression='bz2',
                                use_color=True):
    """Stream images into ffmpeg and write a timelapse without temp JPGs.

    Arguments:
        image_files: [list[str]] Ordered list of image paths.
        frames_root: [str] Directory used when archiving or deleting frames.
        video_path: [str] Destination .mp4 path.

    Keyword arguments:
        fps: [int] Frames per second; 30 by default.
        base_crf: [int] Base CRF for H.264; 25 by default.
        cleanup_mode: [str] What to do with source frames: 'none', 'delete',
            or 'tar'. 'none' by default.
        compression: [str] Tar compression when cleanup_mode == 'tar';
            'bz2' or 'gz'. 'bz2' by default.
        use_color: [bool] Encode colour if possible. True by default.

    Return:
        (video_path, json_path): [tuple[str, str] | (None, None)]
            Output paths on success, (None, None) on failure.
    """

    # Validate input parameters
    image_files = list(image_files)
    total_frames = len(image_files)

    if total_frames < 10:
        log.warning('Fewer than 10 images found, cannot create timelapse.')
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

        # work backward from the last image in the sample window as the first images are sometimes not
        # representative in case of multiple captures per camera
        # (e.g. main capture controls camera mode and passive auxiliary captures)
        idx = sample_size - 1 - i

        sample_image = cv2.imread(image_files[idx], cv2.IMREAD_UNCHANGED)

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
        
        # Check if ffmpeg is available and working
        if not isFfmpegWorking(ffmpeg_path):
            log.warning("ffmpeg is not available or not working.")
            return None, None

    elif platform.system() == 'Windows':
        ffmpeg_path = os.path.join(os.path.dirname(__file__), "ffmpeg.exe")

        if not os.path.exists(ffmpeg_path):
            log.warning("ffmpeg.exe not found in the expected location: {}".format(ffmpeg_path))
            return None, None

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

        # Don't use color if grayscale
        use_color = False

    # Set maxrate and bufsize based on height to prevent very large video files with noisy source images.
    # With normal images, maxrate and bufsize are not limiting - crf is the main factor.
    if height <= 720:
        maxrate, bufsize = "2M", "4M"
    else:
        maxrate, bufsize = "4M", "8M"

    ffmpeg_cmd = [ffmpeg_path, "-y", "-nostdin",
                  "-f", "rawvideo", 
                  "-vcodec", "rawvideo",
                  "-s", "{}x{}".format(width, height),
                  "-pix_fmt", pix_fmt,
                  "-r", str(fps),
                  "-i", "-",
                  "-c:v", "libx264",
                  "-crf", str(crf),
                  "-maxrate", maxrate,       # cap bursts
                  "-bufsize", bufsize,       # smoothing window
                  "-profile:v", "high",      # baseline, main, high
                  "-preset", "medium",       # fast, medium, slow to change size/processing speed
                  "-pix_fmt", "yuv420p",
                  "-movflags", "faststart",  # Optimize for streaming
                  "-threads", "1",
                  "-g", "120",
                  temp_video_path]           # Use temporary path
    
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
            position = (10, image.shape[0] - 6)
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
            if use_color:
                # we promised ffmpeg bgr24
                if image.ndim == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                elif image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            else:
                # we promised ffmpeg gray
                if image.ndim == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Write frame to ffmpeg
            ffmpeg_process.stdin.write(image.tobytes())
            processed_count += 1
            
        except Exception as e:
            log.error("Warning: Error processing image {}: {}".format(img_path, e))
            skipped_count += 1
    
    # Create a temporary timestamp JSON file
    timestamp_path = video_path.replace(MP4_SUFFIX, TS_JSON_SUFFIX)
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
                deleteFilesAndEmptyDirs(image_files, stop_at=frames_root)

            elif cleanup_mode == 'tar':
                try:
                    tar_suffix = TAR_SUFFIX_BZ2 if compression == 'bz2' else TAR_SUFFIX_GZ

                    # Derive the tar file name from video_path, stripping the MP4 suffix
                    base_name = os.path.basename(video_path).replace(MP4_SUFFIX, tar_suffix)

                    # Construct the full path for the tar file
                    tar_path = os.path.join(os.path.dirname(video_path), base_name)

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
                            deleteFilesAndEmptyDirs(image_files, stop_at=frames_root)
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


def main():
    """CLI harness for manual timelapse generation.

    Arguments:
        None

    Return:
        exit_code: [int] 0 on success, non-zero on failure.
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
        args.output = os.path.join(os.path.dirname(args.input_dir), "{}_timelapse.mp4".format(input_dir_name))
    
    # Print configuration
    print("Timelapse Generator Configuration:")
    print("  Input directory: {}".format(args.input_dir))
    print("  Output video: {}".format(args.output))
    print("  FPS: {}".format(args.fps))
    print("  CRF value: {}".format(args.crf))
    print("  Cleanup mode: {}".format(args.cleanup))
    if args.cleanup == 'tar':
        print("  Compression: {}".format(args.compression))
    print("  Video mode: {}".format('Grayscale' if args.grayscale else 'Color'))
    
    # Record start time
    start_time = datetime.now()
    print("Starting process at: {}".format(start_time))
    
    # Generate the timelapse
    try:
        video_path, json_path = generateTimelapseFromDir(
            dir_path=args.input_dir,
            video_path=args.output,
            fps=args.fps,
            base_crf=args.crf,
            cleanup_mode=args.cleanup,
            compression=args.compression,
            use_color=not args.grayscale,
        )

        # Record and print completion time and duration
        end_time = datetime.now()
        duration = end_time - start_time
        print("Process completed at: {}".format(end_time))
        print("Total processing time: {}".format(duration))
        
        # Print file sizes if successful
        if video_path and os.path.exists(video_path):
            video_size = os.path.getsize(video_path) / (1024 * 1024)
            
            print("Output video size: {:.2f} MB".format(video_size))
            
            # Check if tar was created
            if args.cleanup == 'tar':
                ext = '.tar.bz2' if args.compression == 'bz2' else '.tar.gz'
                base_name = os.path.basename(args.output).replace('_timelapse.mp4', '')
                tar_path = os.path.join(os.path.dirname(args.output), base_name + ext)
                
                if os.path.exists(tar_path):
                    tar_size = os.path.getsize(tar_path) / (1024 * 1024)  # Convert to MB
                    print("Archive size: {:.2f} MB".format(tar_size))
        
    except Exception as e:
        print("Error generating timelapse: {}".format(e))
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    
    # Run the main function
    exit(main())
