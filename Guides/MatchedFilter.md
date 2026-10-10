# Matched-filter detection of faint moving objects

`RMS.MatchedFilterDetection` finds moving objects which are too faint to be seen in single frames of
frame-based video (e.g. UWO `.vid` files). The normal detection needs an object to stand out in a single
frame, at least about 5 times the noise. The matched filter instead adds the frames along every possible
motion of an object, so a faint object adds up while the noise averages out, and finds objects down to about
1.5 times the noise per frame (about 2 magnitudes fainter). It runs as a separate pass after the normal
detection, as a replacement of it, or on its own from the command line.

This guide assumes you know the monitor (see [MonitorProcessing.md](MonitorProcessing.md)).


## How it works

1. **Background.** The per-pixel median and noise of every block of 256 frames, smoothed over the neighbouring
   blocks and interpolated in time. The brightness of the stars changes with the transparency of the sky (by
   several times within a minute in thin clouds), so the best fitting scale of the star template is
   subtracted from every frame. Bright stars, the mask and the image border are masked.
   - On some sensors a very bright moving object leaves a faint trail along its whole column and along its row.
     The trails move with the object and would be found as objects themselves, so in every frame the rows and
     columns of the sources brighter than `mf_trail_level` (25 times the noise) are left out of the search,
     except at the source itself.
   - Columns and rows brighter than their neighbours (the trails of very bright stars, bad columns) flicker as a
     whole and are masked.
   - The PSF sigma is measured on the stars of the first block (or set with `mf_psf_sigma`).
2. **Search.** The frames are clipped to +-`mf_clip` (10) times the noise (single-pixel outliers), binned 2x2,
   smoothed with the PSF (at least 1 px), and summed along a grid of motions over runs of 8 and 16 frames (all
   motions up to `mf_ang_vel_max`) and of 32 frames (small motions). Peaks above `mf_threshold` are the hits.
   Pixels above the threshold in a large part of the whole input are flickering or variable sources and are
   removed.
   - A bright object which stays on the same pixels for most of a block is part of the background of the
     block and masked as a star. For these, the median of every block is compared with the medians of blocks
     a few blocks away, shifted by the motion of the stars (e.g. the diurnal motion of a fixed camera): the
     stars cancel, and an object which moves relative to them is linked from block to block. Its background
     is estimated again without it before it is measured.
3. **Linking.** The hits of consecutive runs are linked into tracks, predicting the next hit from the motion of
   the track. A strong hit can come from a single bright frame of its run (a flash of an object whose brightness
   changes quickly), so strong hits are linked by their positions, also across gaps of up to
   `mf_link_max_gap` (96) frames between flashes. Tracks which only follow a much stronger one (the wings and
   trails of a bright object) are dropped, and at most `mf_max_tracks` (300) tracks are measured, the strongest
   ones, so a file in bad conditions can't take much longer than usual.
4. **Measurement.** A moving PSF is fitted jointly to short runs of frames of every track. Each track uses as
   few frames per position as keep the position error below `mf_max_pos_error` (0.5 px): every frame for
   bright objects, up to `mf_max_measure_frames` (8) at the faint limit. Tracks are followed with a local
   motion model, so curved tracks (lens distortion) are measured too. The positions are then combined over
   +-`mf_smooth_frames` (64) frames with a weighted local quadratic fit of the track, which reduces their
   errors 2 to 5 times (positions of nearby measurements are then correlated; 0 disables it).
5. **Verification.** The signal along a smooth track through the measurements, minus the signal at the same
   positions at times when the object is elsewhere, and leaving out the pixels on stars, has to be at least
   `mf_track_significance` (12) times its noise. This rejects tracks made of noise and along the residuals of
   faint stars. The frames are also stacked along the track: the light of an object is concentrated at the
   centre, while the structures of thin clouds drifting across the field (lit from below) are extended and move
   together, and are rejected.
6. **Photometry.** The intensity of every frame along the track is measured, also of the frames whose position
   was not measured on its own (e.g. between the flashes of an object), so the light curve has the full time
   resolution. It is the sum of the pixels within 3 PSF sigmas of the segment the object moved along during the
   frame, put on the scale of the stars of the photometric calibration (the stars are measured with the same
   aperture; the factor is in the done file).
   - Saturated pixels (above `mf_saturation_level`) are left out of the position fits and counted.
   - A very bright object spills its charge along its row and column before its pixels saturate: where the sum
     within 12 px is more than 1.3 times the normal sum, the wider sum is the intensity and the frame counts as
     saturated.


## Output

The detections are written as an FTPdetectinfo (the suffix `mf`) with a row for every frame of a detection:

| Column | Meaning |
|---|---|
| Frame | Frame number. The positions of faint objects are measured on several frames together and interpolated |
| Col, Row | Position (px), combined along the track (see Measurement) |
| RA, Dec, Azim, Elev, Mag | From the recalibrated platepar, as for the normal detection |
| Inten | Background-subtracted sum of the pixels of the object in this frame, on the scale of the CALSTARS intensities |
| Bcknd | Background level at the object |
| SNR | Intensity over its noise in this frame |
| NSatPx | Number of saturated pixels in the frame; at least 1 for spilled charge (the magnitude is then a lower limit of the brightness) |

`matched_filter_done.json` next to it lists every candidate track with its significance, the PSF sigma, the
aperture correction of the intensities, the number of candidate tracks which were not measured
(`tracks_dropped`, see `mf_max_tracks`), and the processing time of every step.


## Running it

### In the monitor, after the normal detection

Add to the config:

```
[MatchedFilter]
mf_enable: true
```

After the normal detection of a file, the worker runs the matched filter on the same frames and writes its
results to a `matched_filter` directory in the results directory of the file (like the normal detection, it
is skipped when the file has fewer than `ff_min_stars` stars):

- `FTPdetectinfo_<...>_mf.txt`, recalibrated with the platepar (RA/Dec and magnitudes),
- `CALSTARS_<...>_mf.txt`, the platepar and the recalibrated platepars,
- `matched_filter_done.json`.

The night report merges the detections of all files into
`<night directory>/matched_filter/FTPdetectinfo_<night>_mf.txt`. The rest of the report (stacks, archive,
upload) only uses the normal detections. A failure of the matched filter is logged and doesn't fail the
file.

### In the monitor, instead of the normal detection

With `mf_replace_detection: true` (and `mf_enable: true`), the normal detection is not run: the stars are
extracted, and the detections of the matched filter are the results of the file (its FTPdetectinfo, CALSTARS,
recalibration, night report, archive and upload), with the done file `matched_filter_done.json` in the results
directory and no `matched_filter` directory. Objects faster than `mf_ang_vel_max` are then not detected.

### On files or directories, e.g. during the day

```
python -m RMS.MatchedFilterDetection /path/to/night/ -c .config -p platepar_cmn2010.cal \
    --dark bias.png --flat flat.png -o /path/to/output
```

The inputs are video files (e.g. `.vid`, `.mkv`) and directories of FITS frames (one frame per file, with the
time of the frame in `DATE-OBS`; the frame rate is taken from the config), each directory being one input.
Every input file gets a directory in the output directory with the same files as above. Files which already
have a `matched_filter_done.json` are skipped, so an interrupted run can simply be started again (`--force`
processes them again). A file which fails is logged and the others are processed; the exit code is 1 if any
file failed. At the end, the detections of all files are merged into
`FTPdetectinfo_<output directory name>_mf.txt` in the output directory.

Options: `--gpu auto|on|off`, `--threads N`, `--no-velocity-search` (only the runs of 32 frames), `--no-stars`
(no star extraction and no recalibration, faster).

### Processing time and the GPU

The search runs on an NVIDIA GPU when numba can use CUDA (`mf_gpu: auto`, the default). This needs the
`numba-cuda` package for the installed CUDA version, e.g. `pip install "numba-cuda[cu12]"`; the log says
`(GPU)` next to the number of velocities when it is used. The results are the same on the CPU and the GPU.

Measured on 10-minute 512x512 `.vid` files (32 FPS, 19,280 frames):

| | Matched filter per file |
|---|---|
| GPU (RTX 6000 Ada), one file at a time, without star extraction | 2.1 min |
| CPU only, 8 threads (`mf_threads: 8`), one file at a time | 3.9 min |
| GPU, three files at a time (6 threads each), with star extraction and recalibration | 3-4.5 min |

The CPU search is limited by memory access, so more than about 8 threads per worker doesn't help. With several
workers, set `mf_threads` to the number of CPU cores divided by the number of workers. Memory: about 2 GB per
worker in addition to the normal processing.


## Tests

On recorded 512x512 frames (9° field of view, 32 FPS) with objects added to them, with the PSF of the camera:

- Found from about 1.5 times the noise per frame in the brightest pixel; about 2 magnitudes fainter than the
  normal detection at all speeds of the search.
- Position errors: 0.03-0.1 px for bright objects, 0.1-0.25 px at the faint limit (larger for faint objects
  which move very little).
- Photometry: within a few percent of the flux, also in the frames of short flashes and between them.
- Objects whose brightness changes (smoothly, in short flashes, or only flashes with nothing between them):
  found down to flashes every 0.5 s.
- Crossing objects, crowded fields, objects passing bright stars, saturated objects: found and measured.
  Parallel objects closer than about 6 px are confused.
- No false detections in these tests; on recorded data the thresholds keep noise and star residuals below the
  significance limit, and the structures of thin drifting clouds are rejected by their shape.

On recorded data the matched filter finds nearly all the detections of the normal detection within its speed
range (except very short ones and ones at the image border), and a comparable number of fainter objects.


## Settings

All settings are in the `[MatchedFilter]` section, see the comments in `.config`. The ones to know:

| Setting | Default | Meaning |
|---|---|---|
| `mf_enable` | false | Run in the monitor after the normal detection |
| `mf_replace_detection` | false | With mf_enable, the matched filter replaces the normal detection in the monitor |
| `mf_ang_vel_min`, `mf_ang_vel_max` | 0.01, 2.0 | Range of the motion of the objects (deg/s). The search cost grows with the square of the largest |
| `mf_threshold` | 5.0 | Threshold of the search (sigma of the best motion) |
| `mf_track_significance` | 12.0 | Minimum significance of a detection |
| `mf_min_frames` | 20 | Minimum duration of a detection (frames) |
| `mf_max_pos_error` | 0.5 | Largest position error of a measurement (px) |
| `mf_max_measure_frames` | 8 | Most frames combined into one position |
| `mf_smooth_frames` | 64 | Positions combined along the track over +- this many frames, 0 disables it |
| `mf_clip` | 10 | The normalized frames are clipped to +- this before the search |
| `mf_link_max_gap` | 96 | Largest gap (frames) between strong hits of a track, e.g. between flashes |
| `mf_trail_level` | 25 | Rows and columns of sources brighter than this in a frame (in the noise of a pixel) are left out of the search, 0 disables it |
| `mf_max_tracks` | 300 | Most candidate tracks measured per file (the strongest), bounds the time in bad conditions |
| `mf_psf_sigma` | 0 | PSF sigma (px), 0 measures it on the stars |
| `mf_saturation_level` | 0 | Saturation level of the raw frames (ADU), 0 = 98% of the bit range |
| `mf_gpu` | auto | Use the GPU: auto, on, off |
| `mf_threads` | 0 | CPU threads, 0 = all cores |


## Limits

- Objects faster than `mf_ang_vel_max` or shorter than `mf_min_frames` are not detected; the normal
  detection finds the bright ones.
- Objects within `mf_edge_margin` (8 px) of the image border are not measured.
- Objects closer than about 6 px moving in parallel are confused.
- Single-frame flashes more than about 2 s apart, with nothing visible between them, are not linked.
- FF files are skipped: the frames themselves are needed.
