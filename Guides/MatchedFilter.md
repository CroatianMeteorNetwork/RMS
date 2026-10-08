# Matched-filter detection of faint moving objects

`RMS.MatchedFilterDetection` finds moving objects which are too faint to be seen in single frames of
frame-based video (e.g. UWO `.vid` files). The normal detection needs an object to stand out in a single
frame, at least about 5 times the noise. The matched filter instead adds the frames along every possible
motion of an object, so a faint object adds up while the noise averages out, and finds objects down to about
1.5 times the noise per frame (about 2 magnitudes fainter). It runs as a separate pass, after or apart from
the normal detection, and its results are kept apart from the normal results.

This guide assumes you know the monitor (see [MonitorProcessing.md](MonitorProcessing.md)).


## How it works

1. **Background.** The per-pixel median and noise of every block of 256 frames, smoothed over the neighbouring
   blocks and interpolated in time. The brightness of the stars changes with the transparency of the sky (by
   several times within a minute in thin clouds), so the best fitting scale of the star template is
   subtracted from every frame. Bright stars, the mask and the image border are masked.
2. **Search.** The frames are binned 2x2, smoothed with the point spread function (PSF), and summed along a
   grid of velocities over runs of 8 and 16 frames (all speeds up to `mf_ang_vel_max`), and of 32 frames (slow
   objects). Peaks above `mf_threshold` are the hits. Pixels above the threshold in a large part of the whole
   input are flickering or variable sources and are removed.
3. **Linking.** The hits of consecutive runs are linked into tracks, predicting the next hit from the velocity
   of each hit.
4. **Measurement.** A moving PSF is fitted jointly to short runs of frames of every track. Each track uses as
   few frames per position as keep the position error below `mf_max_pos_error` (0.5 px): every frame for
   bright objects, up to `mf_max_measure_frames` (8) at the faint limit. Tracks are followed with a local
   motion model, so curved tracks (lens distortion) are measured too. The positions are then combined over
   +-`mf_smooth_frames` (64) frames with a weighted local quadratic fit of the track, which reduces their
   errors 2 to 5 times (positions of nearby measurements are then correlated; 0 disables it). On objects added
   to real frames, the errors are 0.1-0.2 px from 1.5 times the noise per frame and 0.03-0.1 px for bright
   objects.
5. **Verification.** The signal along a smooth track through the measurements, minus the signal at the same
   positions at times when the object is elsewhere, and leaving out the pixels on stars, has to be at least
   `mf_track_significance` (12) times its noise. This rejects tracks made of noise, and slow tracks along the
   residuals of faint stars.


## Running it

### In the monitor, on every file

Add to the config:

```
[MatchedFilter]
mf_enable: true
```

After the normal detection of a file, the worker runs the matched filter on the same frames and writes its
results to a `matched_filter` directory in the results directory of the file:

- `FTPdetectinfo_<...>_mf.txt`, recalibrated with the platepar (RA/Dec and magnitudes),
- `CALSTARS_<...>_mf.txt`, the platepar and the recalibrated platepars,
- `matched_filter_done.json`: every candidate track with its significance, and the timing.

The night report merges the detections of all files into
`<night directory>/matched_filter/FTPdetectinfo_<night>_mf.txt`. The rest of the report (stacks, archive,
upload) only uses the normal detections. A failure of the matched filter is logged and doesn't fail the
file.

The matched filter roughly doubles the processing time of a file without a GPU (see below), so check that the
monitor still keeps up. With several workers, set `mf_threads` to the number of CPU cores divided by the
number of workers, so the workers don't compete for the CPU.

### On files or directories, e.g. during the day

```
python -m RMS.MatchedFilterDetection /path/to/night/ -c .config -p platepar_cmn2010.cal \
    --dark bias.png --flat flat.png -o /path/to/output
```

Every input file gets a directory in the output directory with the same files as above. Files which already
have a `matched_filter_done.json` are skipped, so an interrupted run can simply be started again (`--force`
processes them again). A file which fails is logged and the others are processed; the exit code is 1 if any
file failed. At the end, the detections of all files are merged into
`FTPdetectinfo_<output directory name>_mf.txt` in the output directory.

Options: `--gpu auto|on|off`, `--threads N`, `--no-velocity-search` (only slow objects), `--no-stars` (no star
extraction and no recalibration, faster).


## GPU

The velocity search runs on an NVIDIA GPU when numba can use CUDA (`mf_gpu: auto`, the default). This needs
the `numba-cuda` package for the installed CUDA version, e.g. `pip install "numba-cuda[cu12]"`; the log
says `(GPU)` next to the number of velocities when it is used. The results are the same on the CPU and the
GPU.

Measured on a 10-minute 512x512 `.vid` file (32 FPS, 19,280 frames), one file at a time, without star
extraction:

| | Matched filter per file |
|---|---|
| GPU (RTX 6000 Ada) | 2.1 min |
| CPU only, 8 threads (`mf_threads: 8`) | 3.9 min |
| CPU only, all 24 threads | 4.7 min |

The CPU search is limited by memory access, so more than about 8 threads per worker doesn't help. Memory:
about 2 GB per worker in addition to the normal processing.


## Settings

All settings are in the `[MatchedFilter]` section, see the comments in `.config`. The ones to know:

| Setting | Default | Meaning |
|---|---|---|
| `mf_enable` | false | Run in the monitor after the normal detection |
| `mf_ang_vel_min`, `mf_ang_vel_max` | 0.01, 2.0 | Speed range of the objects (deg/s). The search cost grows with the square of the largest speed |
| `mf_threshold` | 5.0 | Threshold of the search (sigma of the best velocity) |
| `mf_track_significance` | 12.0 | Minimum significance of a detection |
| `mf_min_frames` | 20 | Minimum duration of a detection (frames) |
| `mf_max_pos_error` | 0.5 | Largest position error of a measurement (px) |
| `mf_max_measure_frames` | 8 | Most frames combined into one position |
| `mf_smooth_frames` | 64 | Positions combined along the track over +- this many frames, 0 disables it |
| `mf_gpu` | auto | Use the GPU: auto, on, off |
| `mf_threads` | 0 | CPU threads, 0 = all cores |

The thresholds were set on recorded data from a 512x512 camera with a 9° field of view at 32 FPS: on 55
stretches of 1024 frames without injected objects, no false candidate exceeded a significance of 8, and
every object added to the frames that was found had a significance above 12.


## Limits

- Objects faster than `mf_ang_vel_max` or shorter than `mf_min_frames` are not detected; the normal
  detection finds the bright ones.
- Objects within `mf_edge_margin` (8 px) of the image border are not measured.
- FF files are skipped: the frames themselves are needed.
