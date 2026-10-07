# Continuous processing with the monitor

`RMS.MonitorProcessFrameInterface` watches a directory for recordings made by some other program, processes
every new file as it appears, and at the end of each night produces the same products as a normal RMS
station: calibration report, stacks, thumbnails, timelapse, archive and, optionally, an upload. It is meant
for cameras that are not run by RMS capture, e.g. high frame rate or scientific cameras with their own
recording software.

This guide assumes you know how a normal RMS station works (config file, platepar, mask, FF files,
CALSTARS/FTPdetectinfo, night directories) and are comfortable on the Linux command line.


## What the monitor does

For every new input file, a worker process:

1. Waits until the file has stopped growing (it is still being recorded otherwise).
2. Opens the file and reads the time of its first frame.
3. Applies the dark and the flat, if they were given.
4. Extracts the stars in chunks of frames (128 by default), and saves the max pixel and average pixel image of
   every chunk as an FF-equivalent image pair into the night directory.
5. Detects meteors.
6. Recalibrates the astrometry and photometry of every chunk on the detected stars, starting from the
   platepar.
7. Writes CALSTARS, FTPdetectinfo, the recalibrated platepars and, optionally, ECSV files into a results
   directory for the file, plus a `done.flag` which marks the file as processed.

When a night is over (by default after sunrise), the monitor merges the results of all files of the night and
generates the night report in a separate process, while it keeps processing new files.


## Supported input

| `file_type` | Input | Time of the first frame |
|---|---|---|
| `vid` | UWO `.vid` files, 8 or 16 bit | from the file |
| `mkv`, `mp4`, `avi`, `mov`, `wmv` | video files, decoded with GStreamer (or OpenCV as a fallback) | from the file name |
| `ff` | RMS FF files (`.fits`, `.bin`), see the note below | from the file name |
| `fitsdirs` | directories of FITS frames, one directory per recording | from the FITS headers |
| any other value | treated as a file extension, opened as a video file | from the file name |

Video file names have to contain the time of the first frame, in UTC, either as the whole name
(`20261007_031500.mkv`, `20261007_031500.123456.mkv`, `20261007-031500.mkv`) or in the RMS style
(`CAWE01_20261007_031500_123456_video.mkv`). A video whose time can't be read fails and is given up (see
[Failed files](#failed-files)).

Things to know about the input:

- **One recording per file, written once.** The monitor processes each file once. A file is considered
  complete when its size and time didn't change for 5 s, or right away if it was last modified more than
  30 s ago (so the clocks of the recording and the processing machine should agree, e.g. on a network
  share). If the recording pauses longer in the middle of a file, the worker notices that the file changed
  and the file is processed again in full, also if the incomplete file couldn't be read. A recording program
  which appends to an old file later (e.g. hours later) is not supported. Renaming a file after it was
  processed makes it a new file, which is processed again.
- **File names must be unique.** A file is identified by its name without the extension, which also names
  its results directory. Files in subdirectories (`--recursive`) also get a short hash of the subdirectory
  (e.g. `2026-10-07/22-00-00.mkv` becomes `22-00-00_<hash>`), so the same name in different subdirectories
  is fine. Don't reuse a name for a new recording in the same directory.
- **Workers need a lot of memory.** MKV/MP4/AVI files are read completely into memory, and the meteor
  detection works on all frames of the file at once: a worker processing a 30 s 1080p MKV peaks at about
  7 GB. `.vid` files and FITS directories are read as needed; a 10-minute 512x512 16-bit `.vid` file needs
  about 2 GB. Keep video files short (a few minutes at most) and set `--nproc` to the free RAM divided by
  the peak of one worker. A worker killed by the system for lack of memory counts as a failure of the file.
- **Video decoding.** For video files the decoder is set by `media_backend` and `gst_decoder` in the config.
  `nvh264dec` uses the GPU; if it can't be used, the monitor falls back to the software decoder
  automatically.
- **FF input** is processed (stars, detections, recalibration), but the FF files stay in the input
  directory and no image pairs are made from them, so the night report only has the merged results, the
  calibration variation plots and the archive, without images (stacks, thumbnails, calibration report image,
  timelapse). For FF files of a normal RMS camera, the normal RMS night processing is the better tool.


## Setting up

### Files

You need:

- **A config file** of the camera (`.config`), with the correct station ID, coordinates, elevation, image
  size and frame rate. The detection and star extraction settings are taken from it. If the monitor is
  started without `--config`, it looks for exactly one `.config` file in the input directory.
- **A platepar** fitted on data of this camera (e.g. with SkyFit2 on a few image pairs or FF files). It only
  needs to be roughly right, as every chunk is recalibrated. Without `--platepar`, the monitor looks for
  the file named by `platepar_name` in the config (`platepar_cmn2010.cal` by default) in the input
  directory.
- **A mask** (optional), given with `--mask` (or `mask` in the multicam file). If it's not given, the file
  named by `mask` in the config (`mask.bmp` by default) is used from the input directory; if it's not there,
  each file uses the mask next to it or next to the config file, if there is one.
- **A dark (bias) and a flat** (optional). They are applied only if given with `--dark`/`--flat` (or `dark`
  and `flat` in the multicam file); `use_dark`/`use_flat` in the config are ignored. With detection binning
  the dark, flat and mask are binned automatically. The dark and flat are applied once, to the frames, and
  the saved image pairs are already corrected.

All given files are checked when the monitor starts, and it refuses to start if one is missing.

### Directories

- **Input directory**: where the recording program writes the files. The monitor never modifies or deletes
  input files.
- **Output directory** (`-o`): where the results go. It defaults to the input directory, but use a separate
  directory, ideally on its own disk or partition, so the cleanup of the monitor's data (see
  [Disk space](#disk-space-and-cleanup)) can't be confused by the recordings. Output directories can't be
  nested: the monitor refuses to start in a directory inside the output directory of another monitor (one
  with a `.monitor.lock` file), and the cameras of a multicam file need separate, non-nested directories.

### Starting the monitor

```
python -m RMS.MonitorProcessFrameInterface vid /data/recordings -o /data/monitor \
    -c /data/cal/camera.config -p /data/cal/platepar_cmn2010.cal \
    --dark /data/cal/bias.png --flat /data/cal/flat.png --nproc 4
```

Command line options:

| Option | Default | |
|---|---|---|
| `file_type`, `input_dir` | | what to watch and where (see above) |
| `-o`, `--output` | input dir | output directory |
| `-c`, `--config` | `.config` in the input dir | camera config |
| `-p`, `--platepar` | `platepar_name` in the input dir | platepar |
| `--dark`, `--flat` | none | dark (bias) and flat, applied only if given |
| `--mask` | `mask` in the input dir | mask |
| `-n`, `--nproc` | 2 | parallel worker processes |
| `--chunk_frames` | 128 | frames per star extraction chunk (and per saved image pair) |
| `-r`, `--recursive` | off | also watch subdirectories |
| `-s`, `--start_time` | none | only process files which begin at or after this UTC time, e.g. `20261007_000000` |
| `--report_mode` | from the config | when to make the night reports (see [Night reports](#night-reports)) |
| `--worker_timeout` | 3600 | seconds after which a worker which is still processing a file is stopped |
| `--retry_failed` | off | process the files again which failed twice in previous runs |
| `-f`, `--force` | off | process all files again, also the ones which are done |
| `-m`, `--multicam` | none | run several cameras from an INI file (see below) |

Only one monitor can work on an output directory: a second one (e.g. started twice by mistake) exits with an
error naming the PID of the running one.

### Several cameras

One monitor process can serve several cameras which record the same file type, with a shared pool of
workers:

```
python -m RMS.MonitorProcessFrameInterface --multicam /data/cameras.ini
```

```ini
[Global]
# Parallel workers for all cameras together
nproc = 8
# File type of all cameras
file_type = mkv
recursive = False
chunk_frames = 128
poll_interval = 2
# Seconds before a failed file is tried again
fail_wait_time = 300
# Seconds after which a worker which is still processing a file is stopped
worker_timeout = 3600
# Process the files again which failed twice in previous runs
retry_failed = False
force = False
# start_time = 20261007_000000

[CAM1]
input_dir = /data/cam1/recordings
output_dir = /data/cam1/monitor
config = /data/cam1/cam1.config
platepar = /data/cam1/platepar_cmn2010.cal
dark = /data/cam1/bias.png
flat = /data/cam1/flat.png
mask = /data/cam1/mask.bmp

[CAM2]
input_dir = /data/cam2/recordings
output_dir = /data/cam2/monitor
config = /data/cam2/cam2.config
platepar = /data/cam2/platepar_cmn2010.cal
```

Every camera needs its own output directory. The night reports of the cameras are made one at a time.
Cameras with different file types need separate monitor processes. With `--multicam`, the other command line
options except `--start_time`, `--report_mode` and `--retry_failed` are ignored; they are set in the
`[Global]` section.

### Running as a service

Run the monitor as a systemd service, so it starts with the machine and is restarted if it fails:

```ini
# /etc/systemd/system/rms-monitor.service
[Unit]
Description=RMS monitor processing
Wants=network-online.target
After=network-online.target local-fs.target

[Service]
User=rms
WorkingDirectory=/home/rms/source/RMS
ExecStart=/home/rms/vRMS/bin/python -m RMS.MonitorProcessFrameInterface vid /data/recordings -o /data/monitor -c /data/cal/camera.config -p /data/cal/platepar_cmn2010.cal --nproc 4 --dark /data/cal/bias.png --flat /data/cal/flat.png
Restart=on-failure
RestartSec=30
# Stop only the main process, which stops its workers and gives a running report time to finish
KillMode=mixed
TimeoutStopSec=120

[Install]
WantedBy=multi-user.target
```

```
sudo systemctl daemon-reload
sudo systemctl enable --now rms-monitor
journalctl -u rms-monitor -f
```

Stopping the monitor (`systemctl stop`, `kill`, Ctrl+C) is safe at any time:

- The workers are stopped at once. A file which was being processed has no `done.flag` and is processed again
  after the restart.
- A running night report gets 30 s to finish, otherwise it is stopped and made again after the restart.
- The upload queue is kept on disk, an interrupted upload continues after the restart.
- If the monitor itself is killed (`kill -9`, out of memory), its workers notice it within 2 s and end too.
  The upload process doesn't; with uploading on, stop it by hand before starting the monitor again.


## Settings

The monitor uses the `[MonitorProcessing]` section of the camera config. All options are optional, the
defaults are shown.

```ini
[MonitorProcessing]
monitor_save_images: true
monitor_report_mode: sunrise
monitor_report_quiet_min: 15
monitor_partial_report_time:
monitor_night_cutoff_hours: 0
monitor_upload: false
monitor_delete_images_days: 0
monitor_delete_old_data: true
monitor_update_platepar: true
monitor_save_ecsv: false
monitor_shower_association: false
monitor_fov_kml: false
monitor_flux: false
monitor_observation_summary: false
```

| Option | What it does |
|---|---|
| `monitor_save_images` | Save the max pixel and average pixel image of every chunk as an image pair. The night reports need them (stacks, thumbnails, timelapse, calibration report), so keep it on unless you only want the detections. |
| `monitor_report_mode` | When the night reports are made: `sunrise`, `idle`, `external` or `none`, see [Night reports](#night-reports). `--report_mode` overrides it. |
| `monitor_report_quiet_min` | Minutes without new results of the night before its report is made, so a night isn't reported while its files are still coming in. |
| `monitor_partial_report_time` | Time of day in UTC (`HH:MM`) for a partial report of the night with what was processed until then, e.g. to have the products in the morning while the processing is still catching up. Empty: no partial report. |
| `monitor_night_cutoff_hours` | Hours after the end of the night (sunrise) after which the files of the night which were not processed yet are skipped. 0: never skip. |
| `monitor_upload` | Upload the night archive after the report. It also needs `upload_enabled` and the usual RMS upload settings (`hostname`, `host_port`, `remote_dir`, `rsa_private_key`, `upload_mode`). |
| `monitor_delete_images_days` | Delete the image pairs of reported nights after this many days, keeping the reports and products. 0: keep them until the night directory is deleted. |
| `monitor_delete_old_data` | Delete old data from the output directory the way RMS manages its data directory, see [Disk space](#disk-space-and-cleanup). |
| `monitor_update_platepar` | Carry the best platepar of each night forward to the following data, see [Platepar](#platepar). |
| `monitor_save_ecsv` | Save every calibrated detection as an ECSV file; the night report collects them into the `ECSV` directory of the night. |
| `monitor_shower_association`, `monitor_fov_kml`, `monitor_flux`, `monitor_observation_summary` | Additional night products: single station shower association, FOV KML files (25, 70 and 100 km), flux, and the observation summary. |

The usual RMS options for the night products also apply, e.g. `timelapse_generate_captured`, `thumb_stack`,
`upload_mode`, `capt_dirs_to_keep`, `arch_dirs_to_keep`, `bz2_files_to_keep`, `logdays_to_keep` and
`extra_space_gb`.


## Nights

A night runs from one local solar noon to the next, and is named after the station and the time of the sunset
(at the capture horizon of -5.43°), e.g. `CAWE01_20261006_213008_000000`, like a normal RMS night directory.
Every file belongs to the night which contains the time of its first frame. In polar day or night (no sunset
or sunrise), the night starts at noon and lasts 24 hours.

Files recorded during the day (between sunrise and noon) belong to the night that just ended, files from
the afternoon to the coming night.


## Night reports

### When a night is reported

| `monitor_report_mode` | The night is reported when |
|---|---|
| `sunrise` (default) | the night is over (sunrise) and no new results of the night came in for `monitor_report_quiet_min`. The monitor doesn't have to be idle, so the reports run while the next night is already being processed. |
| `idle` | nothing is being processed or waiting, and no new results came in for `monitor_report_quiet_min`, also during the night. Useful for processing archived data. |
| `external` | only on request, with the trigger file (below). |
| `none` | never automatically; use the command line report. |

On top of that:

- **Partial report**: with `monitor_partial_report_time`, the night is reported at that time with the data
  processed until then, once per night, unless it was already reported after that time. It is not made for
  nights older than a day (e.g. a backlog after a downtime). The final report follows as usual and replaces the
  partial products.
- **Trigger file**: creating the file `.report_now` in the output directory reports all nights with new data
  at once (in `sunrise`, `idle` and `external` mode). It is deleted when picked up.
  ```
  touch /data/monitor/.report_now
  ```
- **Late files**: a file which arrives after its night was reported, or a file processed again (e.g. with
  `--force`), makes the night be reported again.
- **Retries**: a failed report is tried once more after 5 minutes (`fail_wait_time`). A report which takes
  longer than 4 hours is stopped and counts as failed.

### What a report produces

In the night directory `OUTPUT/CapturedFiles/<night>/`, next to the image pairs:

- the merged `CALSTARS_<night>.txt` and `FTPdetectinfo_<night>.txt` (with the per-meteor frame rate and
  RA/Dec), and `platepars_all_recalibrated.json`
- the calibration report (`<night>_calib_report_astrometry.jpg`) and the calibration and photometry variation
  plots of the night
- the captured stack, the detected stack (all images covering detected meteors), and the CAPTURED and
  DETECTED thumbnails
- the timelapse, if `timelapse_generate_captured` is set
- the optional products (shower association, KML, flux, observation summary) and the `ECSV` directory
- the platepar, config and mask that were used

The archive goes into `OUTPUT/ArchivedFiles/<night>/` and the `_metadata`/`_imgdata` `.tar.bz2` files, as
on a normal station (`upload_mode` decides which images are archived), and is uploaded if
`monitor_upload` is on.

A report counts as done when the merge and the archive (or the stacks and thumbnails) succeed; the optional
products are only logged if they fail, like in a normal RMS night.

As a rough guide: a 5-hour night of 1080p video (590 files, 3500 image pairs) takes about 7 minutes to
report and up to about 1 GB of RAM.

### Reporting from the command line

```
python -m RMS.MonitorNightReport /data/monitor                 # all nights with new data
python -m RMS.MonitorNightReport /data/monitor --night CAWE01_20261006_213008_000000
python -m RMS.MonitorNightReport /data/monitor --all --no_archive
```

The command line report never uploads. It refuses to run while a monitor is using the output directory;
use the trigger file then.


## Platepar

Every chunk is recalibrated on its stars, starting from the platepar. With `monitor_update_platepar`, the
best recalibrated platepar of each reported night (the most matched stars, then the lowest residual) is saved
as `OUTPUT/latest_<platepar name>` and used for the following files, so slow drifts of the pointing are
followed. The platepar you gave is never overwritten, and if it is newer than the latest one (e.g. you fitted
it again in SkyFit2), it is used instead. "Newer" is the modification time of the file, so `touch` a platepar
copied with `cp -p`, `rsync` or `scp`, which keep the old time.

If fewer than half of the chunks of a night could be recalibrated, the report logs a warning ("Only N of M
chunks ... could be recalibrated"): the platepar doesn't fit the data, e.g. the platepar of another camera or
a camera which moved. `.night_reports.json` records the numbers for every night (`recalibration`).


## Failed files

- A file which fails is tried once more after `fail_wait_time` (5 minutes). After the second failure it is
  given up, and stays given up after restarts. The failures are recorded in `OUTPUT/.failed_files.json`.
- Typical causes: a corrupt or truncated file, a video whose name has no time, a file that can't be decoded.
- To process the given up files again (e.g. after fixing a problem), start the monitor once with
  `--retry_failed`.
- A worker which takes longer than `--worker_timeout` (1 hour) is stopped, and the file counts as failed. If
  your files take longer than that (very long recordings, slow machine), raise it.
- A worker killed from outside the monitor, usually by the system when the memory runs out, doesn't count as a
  failure: the file is retried, and given up only after it was killed 5 times. Repeated "was killed" warnings
  mean that `--nproc` is too high for the RAM.
- A dark or flat whose size doesn't match the frames fails the file ("The dark is ... px, but the frames are
  ... px"), as it would otherwise silently not be applied.
- `--force` also forgets the failed files.
- If the free space on the output disk falls below `extra_space_gb`, the monitor pauses the processing (it
  doesn't start new files) and runs the cleanup, instead of letting every file fail. It continues when there
  is space again.


## Overflow: when processing can't keep up

The processing should be faster than real time on average; check the logs for how long the files take
(about 40 to 60 s per worker for 30 s of 1080p video, 3.5 to 4 minutes for 10 minutes of 16-bit 512x512
`.vid` data on a modern machine). If it falls behind, the backlog is processed in the order of the files, also
after sunrise, and the night is reported when it's done. Two settings help when that takes too long:

- `monitor_partial_report_time` gives you the products of the night at a fixed time anyway.
- `monitor_night_cutoff_hours` skips the files of a night which were not processed within that many hours
  after sunrise, so a slow night never delays the next one. The files which are being processed at the cutoff
  are finished, and the night is reported with what was processed. Skipped files get a `done.flag` with
  `"skipped": true` in their results directory; delete that directory to process the file later. Note that
  a downtime longer than the cutoff skips the whole backlog of the nights before it.


## Disk space and cleanup

With `monitor_delete_old_data` (on by default), the monitor manages its output directory like RMS manages
its data directory, at startup, after every report, every 12 hours, and when the disk is full. The cleanup
runs independently of the reports, also while a report runs (the night being reported is kept), so the free
space is kept up during long reports:

- Old night directories and archives are deleted by `capt_dirs_to_keep`, `arch_dirs_to_keep`,
  `bz2_files_to_keep` and the quotas, and old logs by `logdays_to_keep`. Nights which were not reported yet
  are kept, unless the quotas need the space.
- If the free space is less than the size of the largest of the last three nights plus `extra_space_gb`,
  whole old nights are deleted, oldest first, but never the latest night. If deleting the old nights couldn't
  free enough space because other data fills the disk, nothing is deleted and a warning is logged.
- The results of the files of deleted nights are reduced to their `done.flag`, which keeps marking the files
  as processed, and the nights are not reported again (also not if a late file of the night arrives).
- `monitor_delete_images_days` deletes the image pairs of reported nights earlier, keeping the products.
  These nights are not reported again either, as a report without the images would replace the products.
- The cleanup uses the RMS data management, which also deletes old `VideoFiles`, `FramesFiles` and
  `TimeFiles` directories in the output directory, so don't put the recordings there.

As a rough guide for the space: image pairs of 1080p 8-bit video take about 1 GB per hour of recording,
512x512 16-bit `.vid` data about 0.6 GB per hour.


## The output directory

```
OUTPUT/
    CapturedFiles/<night>/          image pairs, merged results and products of each night
    ArchivedFiles/<night>/          archive of each night, plus the .tar.bz2 files
    YYYY/YYYYMM/YYYYMMDD/<file>/    results of every input file, with its done.flag
    logs/                           monitor_log_* (main process), monitor_multicam_log_* (multicam),
                                    monitor_<file>_log_* (one per file), report_<night>_log_*, cleanup_log_*
    latest_<platepar name>          best platepar of the latest reported night
    observation.db                  observation summary database (monitor_observation_summary)
    FILES_TO_UPLOAD.inf             upload queue (monitor_upload)
    .night_reports.json             which files of every night were reported, and how
    .failed_files.json              failed and given up files
    .monitor.lock                   lock of the running monitor
    .report_now                     create it to request the reports
```

The night directory also gets `<night>_logs.tar.bz2` with the logs of the night, and a results directory
keeps a backup of an FTPdetectinfo file which is written again (when a file is processed again).

A damaged state file (e.g. after a disk error) is moved aside to `<name>.corrupt`, and the monitor starts
over: the nights are reported again and the failed files are retried.

### Checking on the monitor

- `journalctl -u rms-monitor -f` or the latest `logs/monitor_log_*` shows the files being queued, processed,
  failed and the reports. The log of a single file is in `logs/monitor_<file>_log_*`; errors before the file
  could be opened (e.g. an empty file) are only in the main log.
- After the first nights, check that `latest_<platepar name>` exists and that the reports don't warn about
  the recalibration.
- The monitor reads its config at the start; restart it after changing the config.
- "Processing failed ... giving up" means a file was given up, see [Failed files](#failed-files).
- "pausing the processing until there is space" means the output disk is full.
- "was still being written while it was processed" means the recording paused in the middle of a file, and the
  file is processed again; harmless if rare.
- `.night_reports.json` shows when each night was reported, why (`partial`, `sunrise`, `idle`, `trigger`), and
  which steps failed. The trigger file also retries nights whose report failed.


## Upgrading from the old monitor

Results of the old monitor have an empty `done.flag`. They count as processed, so their files are not
processed again, but they don't belong to a night and get no night reports. Start the new monitor with a new
output directory; to get reports for old data, process it with `--force` into a new output directory.


## What the monitor does not do

These are up to you or other software:

- **Recording**, and **deleting the input files**. The monitor never deletes recordings, so the recording
  software has to remove old files, after they were processed (look for the results directory with a
  `done.flag`, or keep the files longer than the processing can lag behind).
- **Disk space of the input**. The monitor only manages its output directory.
- **Time synchronization**. The night assignment, the cutoff and the reports use the file times and the
  system clock, so keep both the recording and the processing machine on NTP.
- **Calibration files**: making the platepar, mask, dark and flat. Make the first platepar with SkyFit2;
  after that the monitor keeps it up to date.
- **Starting on boot and restarting**: use systemd (see above).
- **Monitoring and alerts**: nothing notifies you if a camera stops recording or files keep failing; watch the
  logs or the reports.
- **The upload server**: the monitor only uploads with the usual RMS upload settings.
- **Multi-station work**: trajectories and coincidences are done by the usual GMN/WMPL tools on the uploaded
  data.
- **Files recorded in daylight** are processed like any other file. Have the recording software stop during the
  day, or use `--start_time`/`monitor_night_cutoff_hours`, if you don't want them.
