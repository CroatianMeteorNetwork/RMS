""" Generate an High Quality MP4 movie from FF files. 
    Contributors: Tioga Gulon
"""

from __future__ import print_function, division, absolute_import

import sys
import re
import subprocess
import shutil
import traceback
import argparse
import os
import tempfile

import cv2
import numpy as np
import json
from datetime import datetime, timedelta


from RMS.Formats.FFfile import read as readFF
from RMS.Formats.FFfile import validFFName, filenameToDatetime
from RMS.Misc import mkdirP, RmsDateTime, tarWithProgress, getRmsRootDir
from RMS.Logger import getLogger
from RMS.FrameStack import STACK_SCALE

log = getLogger("rmslogger")


# --------------------------------------------------------------------
#  Timelapse generation from FF files (meteors)
# --------------------------------------------------------------------
class TimelapseWriter(object):
    def __init__(self, dir_path, mp4_path, fps=30, hires=False, keep_images=False):
        """ Streams stamped FF maxpixels into ffmpeg as raw grayscale video, one frame at a time, so the
            images can come from any reader loop. ffmpeg is started on the first frame, which fixes the
            frame geometry; frames of any other size are skipped.

        Arguments:
            dir_path: [str] Directory the movie belongs to; ffmpeg runs there and keep_images writes there.
            mp4_path: [str] Path of the movie to write.

        Keyword arguments:
            fps: [int] Frames per second; 30 by default.
            hires: [bool] Produce a higher-resolution movie (lower CRF). False by default.
            keep_images: [bool] Also write the stamped frames as JPGs into a temp_img_dir subdirectory.
                False by default.
        """

        self.dir_path = dir_path
        self.mp4_path = mp4_path
        self.fps = fps
        self.crf = 25 if hires else 29
        self.keep_images = keep_images

        self.t_start = RmsDateTime.utcnow()

        self.ffmpeg_path = isFfmpegWorking()
        self.available = bool(self.ffmpeg_path)
        if not self.available:
            log.warning('No video convertor available - tried ffmpeg')

        # Encode into a temporary file and rename on success, so a half-written movie never gets
        # picked up as a finished product
        mp4_dir, mp4_name = os.path.split(mp4_path)
        self.tmp_mp4_path = os.path.join(mp4_dir, "temp_" + mp4_name)

        self.dir_tmp_path = os.path.join(dir_path, "temp_img_dir")
        if self.available and keep_images:
            if os.path.exists(self.dir_tmp_path):
                shutil.rmtree(self.dir_tmp_path, ignore_errors=True)
            mkdirP(self.dir_tmp_path)
            log.info("Stamped frames will be kept in: {}".format(self.dir_tmp_path))

        self.process = None
        self.frame_shape = None
        self.failed = False
        self.n_written = 0
        self.n_skipped = 0


    def _start(self, nrows, ncols):
        """ Start ffmpeg for the given frame geometry. """

        com = [self.ffmpeg_path, '-nostdin', '-y', '-hide_banner',
                '-loglevel', 'error',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', '{:d}x{:d}'.format(ncols, nrows),
                '-pix_fmt', 'gray',
                '-r', str(self.fps),
                '-i', '-',
                '-vcodec', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-an',
                '-crf', str(self.crf),
                '-g', '15',
                '-movflags', 'faststart',
                '-vf', 'hqdn3d=4:3:6:4.5,lutyuv=y=gammaval(0.77)',
                self.tmp_mp4_path]

        log.info('Creating timelapse using...')
        log.info(' '.join(com))

        self.process = subprocess.Popen(com, stdin=subprocess.PIPE, cwd=self.dir_path)
        self.frame_shape = (nrows, ncols)


    def add(self, ff_name, img):
        """ Stamp the timestamp on the frame and send it to ffmpeg. The array is modified in place.

        Arguments:
            ff_name: [str] Name of the FF file, used for the timestamp and camera ID.
            img: [ndarray] Maxpixel image, uint8.
        """

        if (not self.available) or self.failed:
            return

        if img is None:
            self.n_skipped += 1
            return

        if self.process is None:
            self._start(*img.shape)

        elif img.shape != self.frame_shape:
            log.warning("Skipping {} - frame size {} differs from {}".format(ff_name, img.shape,
                self.frame_shape))
            self.n_skipped += 1
            return

        nrows = self.frame_shape[0]

        # Stamp the timestamp on the frame
        timestamp = filenameToDatetime(ff_name).strftime("%Y-%m-%d %H:%M:%S")

        file_split = ff_name.split('_')
        i = 1 if len(file_split[0]) == 2 else 0
        camid = file_split[i]

        text = camid + " " + timestamp + " UTC"
        position = (10, nrows - 6)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1

        cv2.putText(img, text, position, font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img, text, position, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        if self.keep_images:
            cv2.imwrite(os.path.join(self.dir_tmp_path, 'temp_{:04d}.jpg'.format(self.n_written)), img,
                [cv2.IMWRITE_JPEG_QUALITY, 100])

        try:
            self.process.stdin.write(np.ascontiguousarray(img).tobytes())

        except BrokenPipeError:
            log.error("ffmpeg closed its input early while writing frame {:d}".format(self.n_written + 1))
            self.failed = True
            return

        self.n_written += 1


    def close(self):
        """ Finish the movie.

        Return:
            mp4_path: [str] Path of the movie, or None if it could not be made.
        """

        return_code = -1

        if self.process is not None:

            try:
                self.process.stdin.close()
            except OSError:
                pass

            try:
                return_code = self.process.wait(timeout=600)

            except subprocess.TimeoutExpired:
                log.warning("ffmpeg did not finish within the timeout, terminating...")
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                return_code = -1

            self.process = None

        if not self.available:
            return None

        log.info("Timelapse frames written: {:d}, skipped: {:d}".format(self.n_written, self.n_skipped))

        if (return_code == 0) and (not self.failed) and os.path.exists(self.tmp_mp4_path) \
                and (os.path.getsize(self.tmp_mp4_path) > 0):

            os.replace(self.tmp_mp4_path, self.mp4_path)
            log.info("Timelapse saved to: {} ({:.2f} MB)".format(self.mp4_path,
                os.path.getsize(self.mp4_path)/(1024*1024)))
            result = self.mp4_path

        else:
            log.error("Timelapse encoding failed (ffmpeg return code {})".format(return_code))
            if os.path.exists(self.tmp_mp4_path):
                os.remove(self.tmp_mp4_path)
            result = None

        log.info("Timelapse total time: %s", RmsDateTime.utcnow() - self.t_start)

        return result



def generateTimelapse(dir_path, keep_images=False, fps=None, output_file=None, hires=False):
    """Generate a high-quality MP4 movie from FF files.

    The maxpixel of every FF is stamped with its timestamp and streamed straight into ffmpeg as raw
    grayscale video. Nothing is written to disk except the movie itself, unless keep_images asks for
    the stamped frames as JPGs as well.

    Arguments:
        dir_path: [str] Directory that contains the FF files.

    Keyword arguments:
        keep_images: [bool] Also write the stamped frames as JPGs into a temp_img_dir subdirectory.
            False by default.
        fps: [int] Frames per second; 30 by default.
        output_file: [str] Custom output filename. If None, the basename of
            *dir_path* is used. None by default.
        hires: [bool] Produce a higher-resolution movie (lower CRF). False by
            default.

    Return:
        mp4_path: [str] Path of the movie, or None if it could not be made.
    """

    # Set the default FPS if not given
    if fps is None:
        fps = 30


    # Make the name of the output file if not given
    if output_file is None:
        mp4_path = os.path.join(dir_path, os.path.basename(dir_path) + ".mp4")

    else:
        mp4_path = os.path.join(dir_path, os.path.basename(output_file))


    ff_list = [ff_name for ff_name in sorted(os.listdir(dir_path)) if validFFName(ff_name)]

    if not ff_list:
        log.warning('No FF files found in {}, cannot create timelapse.'.format(dir_path))
        return None

    writer = TimelapseWriter(dir_path, mp4_path, fps=fps, hires=hires, keep_images=keep_images)

    if not writer.available:
        return None

    log.info("Streaming {:d} FF files into ffmpeg...".format(len(ff_list)))

    try:
        for file_name in ff_list:

            # Only the maxpixel is needed, skip the other three planes on disk
            ff = readFF(dir_path, file_name, planes=('maxpixel',))

            writer.add(file_name, None if ff is None else ff.maxpixel)

            if writer.n_written % 30 == 0:
                print("{:>5d}/{:>5d}, Elapsed: {:s}".format(writer.n_written, len(ff_list),
                    str(RmsDateTime.utcnow() - writer.t_start)), end="\r")
                sys.stdout.flush()

    finally:
        mp4_path = writer.close()

    return mp4_path



# --------------------------------------------------------------------
#  Timelapse generation from image files (Contrails)
# --------------------------------------------------------------------

#  Output-naming helpers - one place to tweak naming scheme
FNAME_TEMPLATE = (
    "{station}_{start:%Y%m%d-%H%M%S}_to_{end:%Y%m%d-%H%M%S}_{suffix}"
)

MP4_SUFFIX = "frames_timelapse.mp4"
TS_JSON_SUFFIX = "frametimes.json"
# Per-frame exposure metadata of the timelapse (see frameMetaPath). Meant to supersede frametimes.json
FRAME_META_JSON_SUFFIX = "framemeta.json"
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


def findLastNightFrameInWindow(dir_path, window_start, window_end):
    """Find the timestamp of the last night-mode frame within a time window.

    This is used to determine the correct cutoff for night timelapse generation.
    After sunrise, there may be a few night frames captured during the inertia
    period before the mode switch completes. We want to include those frames
    in the night timelapse, but exclude any day frames that follow.

    Arguments:
        dir_path: [str] Root directory to search (walks sub-dirs recursively).
        window_start: [datetime] Start of the search window (e.g., sunrise).
        window_end: [datetime] End of the search window (e.g., sunrise + inertia).

    Return:
        last_n_timestamp: [datetime | None] Timestamp of the last _n frame in
            the window, or None if no night frames were found.
    """
    last_n_timestamp = None

    for root, _, files in os.walk(dir_path):
        for fname in files:
            if not IMAGE_PATTERN.match(fname):
                continue

            # Check if it's a night frame
            mode = _modeFromName(fname)
            if mode != 'n':
                continue

            # Get the timestamp
            try:
                ts = _timestampFromName(fname)
            except ValueError:
                continue

            # Check if it's within the window
            if ts < window_start or ts >= window_end:
                continue

            # Track the latest night frame
            if last_n_timestamp is None or ts > last_n_timestamp:
                last_n_timestamp = ts

    return last_n_timestamp


def listImageBlocksBefore(cutoff, dir_path, max_gap_hours=2):
    """Group images into chronological, same-mode blocks before a cutoff.

    Arguments:
        cutoff: [datetime] Naive UTC timestamp; images >= cutoff are ignored.
        dir_path: [str] Root directory to search (walks sub-dirs recursively).

    Keyword arguments:
        max_gap_hours: [float] Maximum time gap in hours between consecutive
            frames before starting a new block. 2 hours by default.

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

    # 3. first pass - break on mode changes AND time gaps ------------------
    MAX_GAP = timedelta(hours=max_gap_hours)
    prelim_blocks, cur_block, cur_mode = [], [], None
    prev_ts = None

    for path in paths:
        mode = _modeFromName(path)
        ts = _timestampFromName(path)

        # Check if we should start a new block
        should_split = False

        if cur_block:
            # Split on mode change
            if mode != cur_mode:
                should_split = True
            # Split on significant time gap
            elif prev_ts and (ts - prev_ts) > MAX_GAP:
                should_split = True

        if should_split:
            prelim_blocks.append(cur_block)
            cur_block = []

        cur_block.append(path)
        cur_mode = mode
        prev_ts = ts

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
                                     use_color=True,
                                     threads=2,
                                     dejump_sigma=120.0,
                                     gamma=0.5,
                                     high_bit=False,
                                     crf_day=16.0,
                                     crf_night=22.0,
                                     crf_max_day=20.0,
                                     crf_max_night=25.0,
                                     budget_mb=300.0,
                                     frame_interval=5.0):
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
        threads: [int] x264 encoder threads. 2 by default.
        dejump_sigma: [float] Exposure de-jump time scale (s), 0 disables. 120 by default.
        gamma: [float] Camera gamma for the de-jump. 0.5 by default.
        high_bit: [bool] 10-bit encoder (see generateTimelapseFromFrames), with the CRFs planned
            against the size budget (planTimelapseCrfs). False by default.
        crf_day, crf_night, crf_max_day, crf_max_night: [float] 10-bit target and maximum CRF per mode.
        budget_mb: [float] 10-bit size budget, MB per camera-day; 0 = no budget. 300 by default.
        frame_interval: [float] Seconds between saved frames (the budget's frames per day). 5 by default.

    Return:
        results: [list[tuple[str, str] | None]] One *(mp4_path, json_path)*
            per block, or *None* if that block failed.
    """
    # 10-bit: one CRF per block, planned for all blocks together against the size budget
    plan = [None]*len(frame_blocks)
    if high_bit:
        ffmpeg_path = isFfmpegWorking()
        if ffmpeg_path and x264HighBitWorks(ffmpeg_path):
            try:
                plan = planTimelapseCrfs(frame_blocks, ffmpeg_path, crf_day=crf_day, crf_night=crf_night,
                                         crf_max_day=crf_max_day, crf_max_night=crf_max_night,
                                         budget_mb=budget_mb, frame_interval=frame_interval, threads=threads,
                                         use_color=use_color, tmp_dir=frames_root)
            except Exception as e:
                log.warning("Timelapse CRF plan failed, using the target CRFs: {}".format(e))

    results = []
    for block, block_plan in zip(frame_blocks, plan):
        if not block:
            results.append(None)
            continue

        first_img, last_img = block[0], block[-1]

        station, t0 = _parse(first_img)
        _, t1 = _parse(last_img)

        video_name = _buildName(station, t0, t1, MP4_SUFFIX)

        mp4_path_in = os.path.join(frames_root, video_name)

        log.info("Generating timelapse for %s (%d frames)", video_name, len(block))

        mp4_path, json_path = generateTimelapseFromFrames(
            image_files=block,
            frames_root=frames_root,
            video_path=mp4_path_in,
            fps=fps,
            base_crf=base_crf,
            cleanup_mode=cleanup_mode,
            compression=compression,
            use_color=use_color,
            threads=threads,
            dejump_sigma=dejump_sigma,
            gamma=gamma,
            high_bit=high_bit,
            crf=block_plan[0] if block_plan else None,
            night=block_plan[1] if block_plan else None,
            crf_day=crf_day,
            crf_night=crf_night
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
                             threads=2,
                             dejump_sigma=120.0,
                             gamma=0.5,
                             high_bit=False,
                             crf_day=16.0,
                             crf_night=22.0,
                             crf_max_day=20.0,
                             crf_max_night=25.0,
                             budget_mb=300.0,
                             frame_interval=5.0,
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
        threads: [int] x264 encoder threads. 2 by default.
        dejump_sigma: [float] Exposure de-jump time scale (s), 0 disables. 120 by default.
        gamma: [float] Camera gamma for the de-jump. 0.5 by default.
        high_bit, crf_day, crf_night, crf_max_day, crf_max_night, budget_mb, frame_interval: the 10-bit
            encoder and its CRF plan, as in generateTimelapseFromFrameBlocks.

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
    _, t1 = _parse(img_paths[-1])

    if video_path is None:
        video_name = _buildName(station, t0, t1, MP4_SUFFIX)
        video_path = os.path.join(dir_path, video_name)

    # 3 - 10-bit: the CRF against the size budget -----------------------------
    plan = None
    if high_bit:
        ffmpeg_path = isFfmpegWorking()
        if ffmpeg_path and x264HighBitWorks(ffmpeg_path):
            plan = planTimelapseCrfs([img_paths], ffmpeg_path, crf_day=crf_day, crf_night=crf_night,
                                     crf_max_day=crf_max_day, crf_max_night=crf_max_night, budget_mb=budget_mb,
                                     frame_interval=frame_interval, threads=threads, use_color=use_color,
                                     tmp_dir=os.path.dirname(os.path.abspath(video_path)))[0]

    # 4 - delegate to the frame-streamer ------------------------------------
    return generateTimelapseFromFrames(
        image_files=img_paths,
        frames_root=frames_root,
        video_path=video_path,
        fps=fps,
        base_crf=base_crf,
        cleanup_mode=cleanup_mode,
        compression=compression,
        use_color=use_color,
        threads=threads,
        dejump_sigma=dejump_sigma,
        gamma=gamma,
        high_bit=high_bit,
        crf=plan[0] if plan else None,
        night=plan[1] if plan else None,
        crf_day=crf_day,
        crf_night=crf_night,
    )


def isFfmpegWorking(ffmpeg_path='ffmpeg', rms_root=getRmsRootDir()):
    """ Check if ffmpeg is available and working. """
    for ffmpeg in [os.path.join(rms_root, 'Utils', 'ffmpeg'), ffmpeg_path, 'ffmpeg']:
        try:
            subprocess.check_call([ffmpeg, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return ffmpeg
        except Exception:
            pass
        # if we get this far, we cant find ffmpeg
    log.warning("ffmpeg is not available or not working.")
    return False


def frameMetaPath(video_path):
    """ The per-frame metadata sidecar of a frames timelapse, next to its frametimes.json.

    It maps each video frame index (the same keys as frametimes.json) to the exposure record RMS
    embedded in that saved frame (RMS.FrameMetadata, config.save_frame_metadata: capture_utc,
    exp_us, again, dgain, ispdgain, wb_r, wb_g, wb_b, temp_c, frame_seq, and "highlight_rebuild" when the
    highlight rebuild changed the frame), or null for a frame without one. Only written when at
    least one frame carries a record, so stations without the metadata ship no file. With the
    exposure de-jump on, each record also has "dejump_gain": the multiplier applied to that frame's
    codes in the video, never above 1 (divide by it to get the saved frame's codes back, to
    within rounding).

    Arguments:
        video_path: [str] Path of the timelapse .mp4.

    Return:
        [str] Path of the sidecar.
    """
    if MP4_SUFFIX in video_path:
        return video_path.replace(MP4_SUFFIX, FRAME_META_JSON_SUFFIX)
    return video_path.replace('.mp4', '_' + FRAME_META_JSON_SUFFIX)


# Clamp of the exposure de-jump gains (see dejumpGains). Never above 1: a gain above 1 pushes the
# frame's highlights into 255 (up to 12 % of a dawn frame on US05 2026-10-11).
DEJUMP_GAIN_RANGE = (0.5, 1.0)


def frameExposure(meta):
    """ Effective linear exposure of a frame from its metadata record.

    E = exp_us*again*dgain*ispdgain*wb_g. podcontrol's white balance rung scales all three WB gains
    by one factor, so a wb_g change is an exposure change. Validated on US05C1 2026-10-11: the square
    root of the exposure ratio predicts the code ratio of consecutive frames across 50 steps to 0.011
    stop, at zero lag.

    Arguments:
        meta: [dict | None] Frame metadata record (RMS.FrameMetadata).

    Return:
        exposure: [float | None] None when the record is missing or incomplete.
    """
    try:
        exposure = float(meta["exp_us"])*float(meta["again"])
        for key in ("dgain", "ispdgain", "wb_g"):
            if meta.get(key) is not None:
                exposure *= float(meta[key])
    except (KeyError, TypeError, ValueError):
        return None
    return exposure if exposure > 0 else None


def dejumpGains(times, exposures, modes, sigma_s, gamma, gain_range=DEJUMP_GAIN_RANGE):
    """ Per-frame code multipliers that turn the camera's exposure steps into smooth ramps.

    The auto exposure moves in steps (and podcontrol's WB rung snaps back by up to 1.5 stops in one
    frame), which makes the timelapse jump in brightness. The frames are scaled to a virtual
    exposure instead, built with hindsight from the frames on both sides:

    1. The trend: the log exposure smoothed by a Gaussian of sigma_s seconds of capture time, with
       the kernel renormalised at block edges and gaps. Slow changes of the light pass through.
    2. Never brighten: a frame exposed below the trend would need a gain above 1, which pushes its
       highlights into 255, and the auto exposure usually stepped down because highlights were
       clipping. Where a frame sits below the trend, the virtual exposure dips to that frame's own
       exposure instead, entering and leaving the dip at no more than 1 stop per 5*sigma_s (the
       tightest such curve below the trend and every exposure, two-pass). The neighbours are darkened
       to meet the dip, so every gain is <= 1.

    Day and night frames are handled separately, so the mode switch is left alone.

    On US05C1 2026-10-11 (sigma 120 s) this removed every exposure-caused jump (16 jumps over 0.1
    stop -> 0, largest 0.73 -> 0.062 stop) and left the scene's own changes alone; gains spanned
    0.61-1. The trend alone (gains 0.71-1.29) removed the same jumps but pushed up to 12 % of a
    frame into 255 on US05A1-D1 at dawn.

    Arguments:
        times: [list[float]] Capture times (s), any order.
        exposures: [list[float | None]] Frame exposures (frameExposure); None = gain 1.
        modes: [list[str]] Day/night mode of each frame ('d', 'n' or '').
        sigma_s: [float] Smoothing time scale (s).
        gamma: [float] Camera gamma: codes are proportional to light**gamma.

    Keyword arguments:
        gain_range: [tuple[float, float]] Clamp of the gains.

    Return:
        gains: [ndarray] One multiplier per frame.
    """
    times = np.array([np.nan if t is None else t for t in times], dtype=np.float64)
    log_exp = np.array([np.log2(e) if e else np.nan for e in exposures], dtype=np.float64)
    modes = np.asarray(modes)
    gains = np.ones(len(times))
    half_width = 4*sigma_s
    max_slope = 1.0/(5*sigma_s)

    for mode in set(modes.tolist()):
        idx = np.flatnonzero((modes == mode) & np.isfinite(times) & np.isfinite(log_exp))
        if len(idx) < 2:
            continue
        idx = idx[np.argsort(times[idx], kind="stable")]
        t, log_e = times[idx], log_exp[idx]
        lo = np.searchsorted(t, t - half_width)
        hi = np.searchsorted(t, t + half_width, side="right")

        # Trend
        trend = np.empty(len(idx))
        for i in range(len(idx)):
            w = np.exp(-0.5*((t[lo[i]:hi[i]] - t[i])/sigma_s)**2)
            trend[i] = np.dot(w, log_e[lo[i]:hi[i]])/w.sum()

        # Dips: the highest slope-limited curve below every frame's offset from the trend
        dip = np.minimum(log_e - trend, 0.0)
        for i in range(1, len(dip)):
            dip[i] = min(dip[i], dip[i - 1] + max_slope*(t[i] - t[i - 1]))
        for i in range(len(dip) - 2, -1, -1):
            dip[i] = min(dip[i], dip[i + 1] + max_slope*(t[i + 1] - t[i]))

        gains[idx] = 2**(gamma*(trend + dip - log_e))

    return np.clip(gains, gain_range[0], gain_range[1])


# x264 parameters of the frames timelapse (the reasons are in generateTimelapseFromFrames)
X264_PARAMS = "psy-rd=0.0,0.0:aq-strength=0.5"

# The 10-bit encoder adds (benchmarks 2026-10-15..20, US05E1/F1 8-frame stacks, synthetic contrails):
# - qcomp=1: one quantizer for every frame. x264's default 0.6 raises the quantizer of busy frames, and
#   night sky noise is busy to x264: a night "CRF 27" was coded at QP 30, in flat 16x16 blocks.
# - trellis=0: trellis quantization zeroes faint coefficients. Off, the faintest detectable contrail is
#   23 % fainter at night at 5 KB/frame (11 % at 8-12 KB/frame), for 7 % less CPU; neutral by day.
# - 8x8dct=0 at night only: in heavily compressed dark sky the 8x8 intra vertical prediction chains rows
#   into flickering vertical stripes; without the 8x8 transform none appear down to 2.8 KB/frame. By day
#   it codes contrails and cloud edges (off: 3 px contrails keep 1-12 % less) and no stripes were seen.
X264_PARAMS_10BIT = X264_PARAMS + ":qcomp=1:trellis=0"
X264_PARAMS_10BIT_NIGHT = X264_PARAMS_10BIT + ":8x8dct=0"

# Frames sampled per block to predict the size of a 10-bit timelapse (planTimelapseCrfs)
BUDGET_SAMPLE_FRAMES = 60

_X264_10BIT_OK = {}


def x264HighBitWorks(ffmpeg_path):
    """ True if this ffmpeg's libx264 encodes 10-bit H.264 (High 10). Cached per ffmpeg.

    Arguments:
        ffmpeg_path: [str] ffmpeg executable.

    Return:
        [bool]
    """
    if ffmpeg_path not in _X264_10BIT_OK:
        try:
            r = subprocess.run([ffmpeg_path, "-nostdin", "-v", "error", "-f", "lavfi", "-i", "color=c=gray:s=64x64:d=0.04",
                                "-c:v", "libx264", "-profile:v", "high10", "-pix_fmt", "yuv420p10le", "-f", "null", "-"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
            _X264_10BIT_OK[ffmpeg_path] = (r.returncode == 0)
        except Exception:
            _X264_10BIT_OK[ffmpeg_path] = False
    return _X264_10BIT_OK[ffmpeg_path]


def x264Args(crf, threads, high_bit=False, night=False):
    """ ffmpeg output options of the frames timelapse encoder, without the container options.

    Arguments:
        crf: [float] x264 CRF.
        threads: [int] Encoder threads.

    Keyword arguments:
        high_bit: [bool] 10-bit (High 10) encoder. False by default (8-bit High).
        night: [bool] Night frames (10-bit only: no 8x8 transform). False by default.

    Return:
        [list[str]]
    """
    if high_bit:
        return ["-c:v", "libx264",
                "-crf", "{:.2f}".format(crf),
                "-x264-params", X264_PARAMS_10BIT_NIGHT if night else X264_PARAMS_10BIT,
                "-profile:v", "high10",
                "-preset", "medium",
                "-pix_fmt", "yuv420p10le",
                "-threads", str(int(threads)),
                "-g", "1",
                "-bf", "0"]
    return ["-c:v", "libx264",
            "-crf", str(crf),
            "-x264-params", X264_PARAMS,
            "-profile:v", "high",      # baseline, main, high
            "-preset", "medium",       # fast, medium, slow to change size/processing speed
            "-pix_fmt", "yuv420p",
            "-threads", str(int(threads)),
            "-g", "1",
            "-bf", "0"]


def blockIsNight(image_files, is_color):
    """ True for a block of night frames: most frames carry the night suffix (_n), or, when no frame
        carries a mode suffix, the frames are grayscale.

    Arguments:
        image_files: [list[str]] Image paths.
        is_color: [bool] The frames are colour.

    Return:
        [bool]
    """
    modes = [_modeFromName(p) for p in image_files]
    n_night, n_day = modes.count('n'), modes.count('d')
    if n_night or n_day:
        return n_night > n_day
    return not is_color


def toEncoderDepth(image, high_bit):
    """ A saved frame at the encoder's input depth.

    Stacked frames (RMS.FrameStack) are 16-bit with STACK_SCALE per 8-bit code, single frames 8-bit.
    The 10-bit encoder takes 16-bit input (8-bit frames are scaled up, exactly); the 8-bit encoder
    takes 8-bit (16-bit frames are rounded).

    Arguments:
        image: [ndarray] uint8 or uint16 image.
        high_bit: [bool] 16-bit output.

    Return:
        [ndarray] uint16 if high_bit else uint8.
    """
    if high_bit:
        if image.dtype == np.uint8:
            return np.left_shift(image.astype(np.uint16), int(np.log2(STACK_SCALE)))
        return image
    if image.dtype != np.uint8:
        return cv2.addWeighted(image, 1.0/STACK_SCALE, image, 0.0, 0.0, dtype=cv2.CV_8U)
    return image


def stampTime(image, img_path):
    """ Burn the station and UTC time from a frame's file name into its bottom-left corner, white text
        with a black outline for visibility, in place.

    Arguments:
        image: [ndarray] uint8 or uint16 frame.
        img_path: [str] The frame's path.

    Return:
        image: [ndarray] The same array.
    """
    image_name_parts = os.path.basename(img_path).split('_')
    station_id = image_name_parts[0]
    extracted_time = datetime.strptime(image_name_parts[1] + image_name_parts[2], "%Y%m%d%H%M%S")

    text = station_id + " " + extracted_time.strftime("%Y-%m-%d %H:%M:%S") + " UTC"
    position = (10, image.shape[0] - 6)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1

    white = 65535 if image.dtype == np.uint16 else 255
    cv2.putText(image, text, position, font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, position, font, font_scale, (white, white, white), thickness, cv2.LINE_AA)
    return image


def toEncoderChannels(image, use_color):
    """ A frame with the channels the encoder was promised: BGR when use_color, else grayscale. """
    if use_color:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def probeFrames(image_files, sample_size=10):
    """ Size and colour of a block's frames, from the first readable of its first frames.

    Works backward from the last image in the sample window, as the first images are sometimes not
    representative in case of multiple captures per camera (e.g. main capture controls camera mode and
    passive auxiliary captures).

    Arguments:
        image_files: [list[str]] Image paths.

    Keyword arguments:
        sample_size: [int] How many of the first frames to try. 10 by default.

    Return:
        (width, height, is_color): [tuple] or None if none of them could be read.
    """
    n = min(sample_size, len(image_files))
    for i in range(n):
        image = cv2.imread(image_files[n - 1 - i], cv2.IMREAD_UNCHANGED)
        if image is not None:
            height, width = image.shape[:2]
            return width, height, (image.ndim == 3) and (image.shape[2] > 1)
    return None


def sampleBytesPerFrame(ffmpeg_path, image_files, width, height, use_color, high_bit, encoder_args, tmp_dir=None):
    """ Mean coded size of some frames under several encoder settings, in one read of the frames.

    Arguments:
        ffmpeg_path: [str] ffmpeg executable.
        image_files: [list[str]] The frames to encode.
        width, height: [int] Frame size; frames of another size are skipped.
        use_color: [bool] Colour encode.
        high_bit: [bool] 16-bit input (the 10-bit encoder).
        encoder_args: [list[list[str]]] x264Args of each setting.

    Keyword arguments:
        tmp_dir: [str] Where the temporary H.264 streams go. System temp by default.

    Return:
        [list[float | None]] Bytes per frame of each setting (raw H.264, no container), None if it failed.
    """
    pix_fmt = ("bgr48le" if use_color else "gray16le") if high_bit else ("bgr24" if use_color else "gray")
    procs, paths = [], []
    for args in encoder_args:
        fd, path = tempfile.mkstemp(suffix=".h264", dir=tmp_dir)
        os.close(fd)
        paths.append(path)
        procs.append(subprocess.Popen([ffmpeg_path, "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
                                       "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "{}x{}".format(width, height),
                                       "-pix_fmt", pix_fmt, "-r", "30", "-i", "-"] + args + ["-f", "h264", path],
                                      stdin=subprocess.PIPE))
    n_frames = 0
    try:
        for img_path in image_files:
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if (image is None) or (image.shape[:2] != (height, width)):
                continue
            image = toEncoderDepth(image, high_bit)
            try:
                stampTime(image, img_path)              # the time stamp costs bytes too (~2 KB/frame)
            except Exception:
                pass
            data = toEncoderChannels(image, use_color).tobytes()
            for proc in procs:
                proc.stdin.write(data)
            n_frames += 1
        sizes = []
        for proc, path in zip(procs, paths):
            proc.stdin.close()
            proc.wait()
            ok = (proc.returncode == 0) and (n_frames > 0) and (os.path.getsize(path) > 0)
            sizes.append(os.path.getsize(path)/float(n_frames) if ok else None)
        return sizes
    except Exception as e:
        log.warning("Timelapse size sample failed: {}".format(e))
        return [None]*len(encoder_args)
    finally:
        for proc, path in zip(procs, paths):
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            if os.path.exists(path):
                os.remove(path)


def planTimelapseCrfs(frame_blocks, ffmpeg_path, crf_day=16.0, crf_night=22.0, crf_max_day=20.0, crf_max_night=25.0,
                      budget_mb=300.0, frame_interval=5.0, threads=2, use_color=True, tmp_dir=None):
    """ CRF of each block's 10-bit timelapse under a size budget per camera-day.

    Each mode has a target CRF (constant quantizer, so constant quality) and a maximum. The size at a
    fixed CRF follows the sky (day blocks 2.7-5x between clear and cloudy), so the size of every block
    is predicted from BUDGET_SAMPLE_FRAMES frames spread over it, encoded at its target and at its
    maximum CRF (log-linear in between). If the blocks are predicted over the budget, every block's CRF
    is raised by the same step until they fit, but no block past its maximum: quality wins over the
    budget, and that is logged as a warning.

    Arguments:
        frame_blocks: [list[list[str]]] The blocks to encode (each block one mode).
        ffmpeg_path: [str] ffmpeg executable.

    Keyword arguments:
        crf_day, crf_night: [float] Target CRF per mode.
        crf_max_day, crf_max_night: [float] Maximum CRF per mode.
        budget_mb: [float] MB per camera-day (86400 s of frames at frame_interval); 0 = no budget.
        frame_interval: [float] Seconds between saved frames.
        threads: [int] Encoder threads of the sample encodes.
        use_color: [bool] Colour encode where the frames are colour.
        tmp_dir: [str] Directory of the temporary sample streams.

    Return:
        plan: [list[tuple[float, bool] | None]] (CRF, night) per block, None for a block too short to encode.
    """
    blocks = []
    for block in frame_blocks:
        probe = probeFrames(block) if (block and len(block) >= 10) else None
        if probe is None:
            blocks.append(None)
            continue
        width, height, is_color = probe
        night = blockIsNight(block, use_color and is_color)
        target, top = (crf_night, crf_max_night) if night else (crf_day, crf_max_day)
        blocks.append(dict(files=block, n=len(block), night=night, target=float(target), top=float(max(top, target)),
                           width=width, height=height, color=use_color and is_color))

    live = [b for b in blocks if b is not None]
    plan = [None if b is None else (round(b["target"], 2), b["night"]) for b in blocks]
    if (not live) or (budget_mb <= 0):
        return plan

    # Predicted bytes per frame at the target and at the maximum CRF
    for b in live:
        step = max(1, b["n"]//BUDGET_SAMPLE_FRAMES)
        settings = [x264Args(b["target"], threads, high_bit=True, night=b["night"])]
        if b["top"] > b["target"]:
            settings.append(x264Args(b["top"], threads, high_bit=True, night=b["night"]))
        sizes = sampleBytesPerFrame(ffmpeg_path, b["files"][step//2::step], b["width"], b["height"], b["color"], True,
                                    settings, tmp_dir=tmp_dir)
        if any(x is None for x in sizes):
            log.warning("Timelapse size could not be predicted, using the target CRFs")
            return plan
        b["bpf_target"], b["bpf_top"] = sizes[0], sizes[-1]

    def bytesAt(b, crf):
        if b["top"] <= b["target"]:
            return b["n"]*b["bpf_target"]
        f = (crf - b["target"])/(b["top"] - b["target"])
        return b["n"]*np.exp((1 - f)*np.log(b["bpf_target"]) + f*np.log(b["bpf_top"]))

    def total(step):
        return sum(bytesAt(b, min(b["target"] + step, b["top"])) for b in live)

    n_frames = sum(b["n"] for b in live)
    budget = budget_mb*1e6*n_frames/(86400.0/frame_interval)
    step_max = max(b["top"] - b["target"] for b in live)
    if total(0) <= budget:
        step = 0.0
    elif total(step_max) > budget:
        step = step_max
    else:
        lo, hi = 0.0, step_max
        for _ in range(30):
            mid = (lo + hi)/2
            lo, hi = (mid, hi) if total(mid) > budget else (lo, mid)
        step = hi

    for b in live:
        crf = min(b["target"] + step, b["top"])
        log.info("Timelapse plan: {:s} block of {:d} frames: CRF {:.2f} (target {:.1f}, max {:.1f}), predicted "
                 "{:.1f} KB/frame, {:.1f} MB".format("night" if b["night"] else "day", b["n"], crf, b["target"],
                                                       b["top"], bytesAt(b, crf)/b["n"]/1e3, bytesAt(b, crf)/1e6))
    log.info("Timelapse budget {:.0f} MB per camera-day = {:.1f} MB for these {:d} frames: predicted {:.1f} MB at the "
             "target CRFs, {:.1f} MB as planned (CRF +{:.2f})".format(budget_mb, budget/1e6, n_frames, total(0)/1e6,
                                                                       total(step)/1e6, step))
    if total(step) > budget:
        log.warning("Timelapse over budget at the maximum CRFs: predicted {:.1f} MB > {:.1f} MB. Quality wins: "
                    "encoding at the maximum CRFs".format(total(step)/1e6, budget/1e6))

    return [None if b is None else (round(min(b["target"] + step, b["top"]), 2), b["night"]) for b in blocks]


def generateTimelapseFromFrames(image_files,
                                frames_root,
                                video_path,
                                fps=30,
                                base_crf=25,
                                cleanup_mode='none',
                                compression='bz2',
                                use_color=True,
                                threads=2,
                                dejump_sigma=120.0,
                                gamma=0.5,
                                high_bit=False,
                                crf=None,
                                night=None,
                                crf_day=16.0,
                                crf_night=22.0):
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
        threads: [int] x264 encoder threads. 2 by default.
        dejump_sigma: [float] Time scale (s) of the exposure de-jump (dejumpGains) applied to frames
            with exposure metadata; 0 disables it. 120 by default.
        gamma: [float] Camera gamma (config.gamma) used by the de-jump. 0.5 by default.
        high_bit: [bool] 10-bit encoder (H.264 High 10, X264_PARAMS_10BIT): codes 16-bit stacked frames
            (RMS.FrameStack) at their precision. Falls back to the 8-bit encoder when this ffmpeg's
            libx264 cannot do 10-bit. False by default.
        crf: [float | None] 10-bit: the CRF (planTimelapseCrfs); None = crf_day or crf_night by mode.
            The 8-bit encoder always uses base_crf (+2 for grayscale).
        night: [bool | None] 10-bit: night frames (no 8x8 transform); None = from the file names
            (blockIsNight).
        crf_day, crf_night: [float] 10-bit CRF per mode when crf is None. 16 and 22 by default.

    Return:
        (video_path, json_path): [tuple[str, str] | (None, None)]
            Output paths on success, (None, None) on failure. The per-frame metadata sidecar,
            when written, is at frameMetaPath(video_path).
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
    probe = probeFrames(image_files)
    if probe is None:
        log.error("Error: Could not find any valid images to determine dimensions.")
        return None, None
    width, height, is_color = probe
    
    # Set up ffmpeg command 
    ffmpeg_path = isFfmpegWorking()
    if not ffmpeg_path:
        log.warning("ffmpeg not available on this installation.")
        return None, None

    if high_bit and not x264HighBitWorks(ffmpeg_path):
        log.warning("This ffmpeg's libx264 cannot encode 10-bit: using the 8-bit timelapse encoder")
        high_bit = False

    # Don't use color if grayscale
    use_color = use_color and is_color

    if high_bit:
        # 10-bit: CRF by mode (or as planned against the size budget), 16-bit input
        if night is None:
            night = blockIsNight(image_files, use_color)
        if crf is None:
            crf = crf_night if night else crf_day
        pix_fmt = "bgr48le" if use_color else "gray16le"

    elif use_color:
        # If the image is color, use the base CRF value
        crf = base_crf

        # Set pixel format for color
        pix_fmt = "bgr24"

    else:
        # If the image is grayscale, increase CRF by 2
        crf = base_crf + 2

        # Set pixel format for grayscale
        pix_fmt = "gray"

    # Encoder settings, from benchmarks on NR-off (grainy) science-camera stills, 2026-09 and 2026-10:
    # - No maxrate/bufsize cap. The old 2M/4M cap was pinned on 74% of a real night's frames and
    #   flattened every busy frame to the same SSIM regardless of CRF; the movie is a science
    #   product (contrail detection), so CRF alone sets the quality.
    # - psy-rd 0 and aq-strength 0.5: x264 stops spending bits reproducing grain texture, which a
    #   detector does not need. -41% size at equal SSIM on the grainy source.
    # - All-intra (GOP 1). Predicted frames skip the faint (1-4 code) change a new or vanished
    #   contrail makes, so it shows up only at the next keyframe and lingers until the one after:
    #   the detector saw contrails from neighbouring frames. Synthetic contrails on real skies, CRF 25:
    #   GOP 10 kept 0.48 of a contrail's amplitude and left +0.09 of it in the next frame; all-intra
    #   keeps 0.65 and leaves nothing, for 1.86x the size of a day block and half the CPU. Every
    #   frame is also a keyframe when the server fetches one by timestamp.
    # x264-params: ':' separates options, and the two psy-rd values need a ',' (a ':' there makes
    # ffmpeg 6.1 silently drop everything after it).
    # The 10-bit encoder's additions are explained at X264_PARAMS_10BIT.
    ffmpeg_cmd = [ffmpeg_path, "-y", "-nostdin", '-hide_banner', 
                  '-loglevel', 'error',
                  "-f", "rawvideo", 
                  "-vcodec", "rawvideo",
                  "-s", "{}x{}".format(width, height),
                  "-pix_fmt", pix_fmt,
                  "-r", str(fps),
                  "-i", "-"] \
        + x264Args(crf, threads, high_bit=high_bit, night=bool(night)) \
        + ["-movflags", "faststart",  # Optimize for streaming
           temp_video_path]           # Use temporary path
    
    # Per-frame metadata embedded in the saved frames, read before encoding: the de-jump needs every
    # frame's exposure up front, and cleanup may delete the frames afterwards
    try:
        from RMS.FrameMetadata import readImageMeta
    except Exception:
        readImageMeta = None
    image_meta = []
    for img_path in image_files:
        meta = None
        if readImageMeta is not None:
            try:
                meta = readImageMeta(img_path, max_bytes=65536)
            except Exception:
                meta = None
        image_meta.append(meta)

    # Exposure de-jump: one code multiplier per frame
    gains = None
    exposures = [frameExposure(meta) for meta in image_meta]
    n_exposures = sum(1 for e in exposures if e)
    if dejump_sigma and (dejump_sigma > 0) and (n_exposures >= 2):
        times, modes = [], []
        for img_path, meta in zip(image_files, image_meta):
            try:
                times.append(float(meta["capture_utc"]))
            except (KeyError, TypeError, ValueError):
                times.append(None)
            m = IMAGE_PATTERN.match(os.path.basename(img_path))
            modes.append((m.group("suffix") or '').lower() if m else '')
        gains = dejumpGains(times, exposures, modes, dejump_sigma, gamma)
        log.info("Exposure de-jump (sigma {:.0f} s, gamma {:.2f}) on {:d} frames: gains {:.3f} to {:.3f}"
                 .format(dejump_sigma, gamma, n_exposures, gains.min(), gains.max()))

    log.info("Starting ffmpeg process for {}...".format(video_path))
    log.info("Video mode: {}".format('Color' if use_color else 'Grayscale'))
    if high_bit:
        log.info("10-bit encoder, {} frames, CRF {:.2f}".format('night' if night else 'day', crf))
    ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    
    # Initialize timestamp JSON data
    timestamp_data = {}
    frame_meta = {}
    
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

        # The encoder's input depth: 16-bit for the 10-bit encoder, else 8-bit
        image = toEncoderDepth(image, high_bit)

        # Exposure de-jump, before the timestamp is stamped so the text keeps its levels
        gain = 1.0
        if (gains is not None) and (gains[index] != 1.0):
            gain = float(gains[index])
            if image.dtype == np.uint8:
                image = cv2.LUT(image, np.clip(np.round(np.arange(256)*gain), 0, 255).astype(np.uint8))
            else:
                image = cv2.addWeighted(image, gain, image, 0.0, 0.0, dtype=cv2.CV_16U)
        
        # Stamp the time from the filename
        try:
            stampTime(image, img_path)

            # Record timestamp data (only for frames that are successfully processed)
            timestamp_data[str(processed_count)] = "_".join(os.path.basename(img_path).split('.')[0].split('_')[1:])

        except Exception as e:
            log.warning("Warning: Error processing metadata for {}: {}".format(img_path, e))
        
        # Handle color conversion based on target mode (we promised ffmpeg colour or gray)
        try:
            image = toEncoderChannels(image, use_color)

            # Write frame to ffmpeg
            ffmpeg_process.stdin.write(image.tobytes())

            meta = image_meta[index]
            if (meta is not None) and (gains is not None):
                meta = dict(meta, dejump_gain=round(gain, 4))
            frame_meta[str(processed_count)] = meta
            processed_count += 1
            
        except Exception as e:
            log.error("Warning: Error processing image {}: {}".format(img_path, e))
            skipped_count += 1
    
    # Create a temporary timestamp JSON file
    # note that if an output video filename was provided, then it may not contain MP4_SUFFIX
    if MP4_SUFFIX in video_path:
        timestamp_path = video_path.replace(MP4_SUFFIX, TS_JSON_SUFFIX)
    else: 
        timestamp_path = video_path.replace('.mp4', '_' + TS_JSON_SUFFIX)
    temp_timestamp_path = os.path.join(output_dir, "{}_temp_timestamps.json".format(output_name))
    
    try:
        with open(temp_timestamp_path, 'w') as f:
            json.dump(timestamp_data, f, indent=2)
    except Exception as e:
        log.warning("Warning: Error saving timestamp data: {}".format(e))

    # The metadata sidecar: one compact JSON line per frame keeps it readable and diffable
    meta_path = frameMetaPath(video_path)
    temp_meta_path = os.path.join(output_dir, "{}_temp_framemeta.json".format(output_name))
    n_meta = sum(1 for v in frame_meta.values() if v)
    if n_meta:
        try:
            with open(temp_meta_path, 'w') as f:
                f.write("{\n" + ",\n".join('"{}": {}'.format(k, json.dumps(frame_meta[k], separators=(",", ":")))
                                            for k in sorted(frame_meta, key=int)) + "\n}\n")
            log.info("Frame metadata for {:d} of {:d} frames".format(n_meta, processed_count))
        except Exception as e:
            log.warning("Warning: Error saving frame metadata: {}".format(e))
    
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

            # Rename the frame metadata file if it exists
            if os.path.exists(temp_meta_path):
                if os.path.exists(meta_path):
                    os.remove(meta_path)
                os.rename(temp_meta_path, meta_path)
                log.info("Frame metadata saved to: {}".format(meta_path))
        
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
        for temp_file in [temp_video_path, temp_timestamp_path, temp_meta_path]:
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

    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Generate a timelapse video from image frames.')
    parser.add_argument('input_dir', help='Directory containing image frames organized in hour subdirectories')
    parser.add_argument('--output', '-o', help='Output video path. Default: [input_dir]_timelapse.mp4')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second (default: 30)')
    parser.add_argument('--crf', type=int, default=25, help='Constant Rate Factor for compression (default: 27)')
    parser.add_argument('--cleanup', choices=['none', 'delete', 'tar', 'keep-jpg'], default='none',
                      help='Cleanup mode after processing (default: none)')
    parser.add_argument('--compression', choices=['bz2', 'gz'], default='bz2',
                      help='Compression method for tar (default: bz2)')
    parser.add_argument('--image-files', action="store_true",
                        help='Use image files instead of fits files.')
    parser.add_argument('--grayscale', action='store_true', 
                      help='When using image files create grayscale video instead of color')
    parser.add_argument('--hires', action="store_true",
                            help='Make a higher resolution timelapse. The video file will be larger.')
    parser.add_argument('--dejump', type=float, default=120.0,
                        help='Image files: exposure de-jump time scale in seconds, 0 disables (default: 120)')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='Image files: camera gamma used by the de-jump (default: 0.5)')
    parser.add_argument('--10bit', dest='high_bit', action='store_true',
                        help='Image files: 10-bit encoder (H.264 High 10), CRF by mode (--crf is not used)')
    parser.add_argument('--crf-day', type=float, default=16.0, help='10-bit: target CRF of day frames (default: 16)')
    parser.add_argument('--crf-night', type=float, default=22.0, help='10-bit: target CRF of night frames (default: 22)')
    parser.add_argument('--crf-max-day', type=float, default=20.0, help='10-bit: maximum CRF of day frames (default: 20)')
    parser.add_argument('--crf-max-night', type=float, default=25.0,
                        help='10-bit: maximum CRF of night frames (default: 25)')
    parser.add_argument('--budget-mb', type=float, default=300.0,
                        help='10-bit: size budget in MB per camera-day, 0 = none (default: 300)')
    parser.add_argument('--interval', type=float, default=5.0,
                        help='10-bit: seconds between frames, for the budget (default: 5)')


    
    args = parser.parse_args()
    
    input_dir = os.path.abspath(args.input_dir)
    # Set default output path if not specified
    if not args.output:
        input_dir_name = os.path.basename(os.path.normpath(input_dir))
        args.output = os.path.join(os.path.dirname(input_dir), "{}_timelapse.mp4".format(input_dir_name))
    
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
    if not args.image_files:

        try:
            keep_images = True if args.cleanup == 'keep-jpg' else False

            generateTimelapse(dir_path=input_dir,
                              keep_images=keep_images,
                              fps=args.fps,
                              output_file=args.output,
                              hires=args.hires)

        except Exception as e:
            print("Error generating timelapse: {}".format(e))
            traceback.print_exc()
            return 1

    else:

        try:
            video_path, _ = generateTimelapseFromDir(
                dir_path=input_dir,
                video_path=args.output,
                fps=args.fps,
                base_crf=args.crf,
                cleanup_mode=args.cleanup,
                compression=args.compression,
                use_color=not args.grayscale,
                dejump_sigma=args.dejump,
                gamma=args.gamma,
                high_bit=args.high_bit,
                crf_day=args.crf_day,
                crf_night=args.crf_night,
                crf_max_day=args.crf_max_day,
                crf_max_night=args.crf_max_night,
                budget_mb=args.budget_mb,
                frame_interval=args.interval,
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
            traceback.print_exc()
            return 1
    
    return 0


if __name__ == "__main__":
    
    # Run the main function
    exit(main())
