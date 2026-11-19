# Working on RMS: rules and code map

These are the instructions for anyone changing RMS, people and AI coding agents alike (Claude Code reads them
through `CLAUDE.md`, Codex and other agents read this file directly). They describe how the code is organised,
the rules that keep thousands of unattended stations running, the code style, and what a pull request needs.

RMS is the software of the Global Meteor Network (GMN) stations. Each night a station captures video from its
camera, compresses it into FF files, detects meteors and stars, calibrates its pointing and photometry, reduces
the meteors, and uploads the results to the GMN server. Stations run unattended on Raspberry Pis and Linux PCs,
often for years, and many operators never look at them. A change that crashes, hangs or fills the disk on one
night costs data from every station that updates.


## Pull requests

- Base every pull request on the `prerelease` branch. Never target `master`: the maintainers merge `prerelease`
  into `master` for a release.
- One concern per pull request. Do not mix in reformatting, renames, import reordering or style changes of code
  you did not otherwise change; style clean-ups go in their own pull request.
- The description has three parts: **Problem** (what goes wrong, for whom, how it shows up, with a concrete
  example), **Fix** (what the change does and why it is right), **Testing** (the new tests, the full suite, and
  any check on real data with its result).
- Commit messages are plain sentences, usually with the area first, e.g. "SkyFit2: keep the startup mask when
  the data directory has no mask.bmp" or "Upload resume: a failed sync on the server keeps the upload for the
  next try". The body says what was wrong and why the change fixes it.
- Before a merge, each pull request has its tests run in full and is reviewed, including by the Codex review
  (see "Review guidelines" below) until it reports no issues.


## Setting up, building and testing

- Install the Python packages from `requirements.txt` into a virtual or conda environment.
- Build the Cython modules (`*.pyx` in `RMS/`, `RMS/Astrometry/`, `RMS/Routines/` and `Utils/`) after checking
  out and after every change to a `.pyx` file:

  ```
  python setup.py build_ext --inplace
  ```

- Run the tests from the repository root:

  ```
  QT_QPA_PLATFORM=offscreen python -m pytest Tests -q
  ```

  pytest collects `Tests/test_*.py` and `Tests/Test*.py` (set in `pyproject.toml`). The suite takes about 10
  minutes. `QT_QPA_PLATFORM=offscreen` lets the SkyFit2 tests run without a display. Some files in `Tests/` are
  scripts or examples rather than tests (e.g. `CompressionTimings.py`).
- New tests go into `Tests/test_<Area>.py`, next to the tests of the same area. They must be deterministic and
  fast, need no network and no station data, build small synthetic inputs, and write only into pytest's
  `tmp_path` (never into the repository or the working directory).
- A bug fix comes with a test that fails without the fix. Check it by putting the old file back
  (`git show origin/prerelease:<path> > <path>`), running the test, and restoring your version.
- Where the multiprocessing start method matters, test both `fork` and `forkserver` (the default on Python 3.14).
- Changes to detection, calibration, astrometry or photometry also need a check on real data: run the changed
  step on a real night before and after the change, compare the outputs, and explain every difference in the
  pull request.


## Platforms

- Python: the oldest station images (Raspbian Buster) run Python 3.7, current ones run up to 3.14. Station
  code must stay compatible with all of them (no `:=`, no APIs newer than 3.7, check the version markers in
  `requirements.txt` before adding or raising a requirement, and that the package has builds for 32-bit ARM,
  e.g. on piwheels).
- NumPy 1.21 to 2.x must all work. Under NumPy 2, arithmetic on unsigned image arrays wraps around instead of
  being promoted, so subtract frames in a signed type.
- Both Cython 0.29 and 3.x must build the `.pyx` modules.
- Linux (Raspberry Pi OS 32 and 64 bit, Ubuntu and Debian) runs stations; SkyFit2 and the tools also run on
  Windows and macOS.


## Rules that keep stations running

- **Nothing may block forever.** Every network call has a timeout, every `join()` and lock wait is bounded,
  every child process exits when its parent is gone, and a retry loop has a limit.
- **One failure must not stop the night.** Each optional night product (calibration report, shower
  association, flux, FOV KML, flat, timelapse, ...) is wrapped in `try`/`except` that logs the failure with
  `log.warning("... failed", exc_info=True)` and carries on with the rest of the processing.
- **Log through the RMS logger** (`from RMS.Logger import getLogger; log = getLogger("rmslogger")`), so messages
  reach the night log that is uploaded with the data. Station code does not use `print()` for anything an
  operator or the maintainers need to see.
- **Stay compatible with what is already out there:**
  - old `.config` files: a missing option takes its default;
  - old platepars (JSON, and pickled in older data): a new attribute needs a default when an old platepar is
    loaded, and loading must not change the results of an old platepar;
  - file formats read by the GMN server and by other tools (FF, FR, FTPdetectinfo, CALSTARS, platepar, the
    observation summary): extend them only in a backward-compatible way.
- **New config options** are added in three places: the default in `Config.__init__` (`RMS/ConfigReader.py`),
  the parsing and validation in the matching `parse*` function, and the `.config` template with a `;` comment
  that explains the option. The default keeps today's behaviour unless changing it is the point of the change.
- **Resources:** stations are small computers that run for months. Avoid copying full frame arrays, close
  files and processes, and keep memory flat over a night.
- **Line endings:** `RMS/BufferedCapture.py`, `RMS/CaptureDuration.py`, `RMS/Compression.py`,
  `RMS/MLFilter.py` and `RMS/StartCapture.py` use CRLF line endings. Keep them: an editor or a script that
  writes LF turns the whole file into a conflict. After editing, `git diff --stat` must show only your lines.
- **Scripts that run on stations** (`Scripts/RMS_Update.sh`, `Scripts/RMS_StartCapture.sh`,
  `Scripts/MultiCamLinux/GRMSUpdater.sh`) update and restart every station. Test them with `bash -n` and on a
  test machine; a broken update script cannot fix itself.


## Code style

Follow the code around your change. In particular:

- Names: functions and methods `camelCase` (`findPartialUpload`), classes `CapWords`, variables `snake_case`,
  constants `UPPER_CASE`.
- Docstrings in the RMS form, also for nested helpers and test helpers:

  ```python
  def binImage(img, bin_factor, method='avg'):
      """ Bin the image by the given factor.

      Arguments:
          img: [ndarray] Image to bin.
          bin_factor: [int] Binning factor, a power of 2.

      Keyword arguments:
          method: [str] 'avg' to average the binned pixels, 'sum' to add them. 'avg' by default.

      Return:
          [ndarray] The binned image.
      """
  ```

- Comments: a short comment before nearly every small block, saying what the block does conceptually, not
  repeating the code. Long functions are split into sections marked with `### Section name ###`. A comment that
  continues on the next line is indented by two extra spaces after the `#`.
- Operators: one space around `=`, comparisons and the binary `+` and `-`; **no spaces around `*`, `/` and `**`**,
  so the spacing shows the order of evaluation, e.g. `ecef_z = ((1 - EARTH.E**2)*N + h)*sin(lat)` and
  `cos_zenith = (sx*dx + sy*dy + sz*dz)/np.sqrt(...)` (RMS/Astrometry/Conversions.py).
- Strings are formatted with `.format()` like most of the code base.
- Lines are kept under about 110 characters.
- Prefer the existing helpers (`RMS/Misc.py`, `RMS/Routines/`, `RMS/Astrometry/Conversions.py`) over new ones.
- No new dependency unless it is really needed; it has to install on every station.


## Code map

### Entry points

| Command | What it does |
| --- | --- |
| `python -m RMS.StartCapture` | The station: waits for night, captures, compresses, detects, processes the night, uploads, repeats. |
| `python -m RMS.Reprocess <night dir>` | Processes a captured night again (detection, calibration, products, upload). |
| `python -m Utils.SkyFit2 <night dir>` | GUI for calibrating a camera (platepar), manual meteor reduction and mask editing. |
| `python -m Utils.<Tool>` | Tools for camera control, image stacks, flux, timelapses and more; see `Utils/README.md`. |

### Station pipeline (`RMS/`)

- **Capture:** `BufferedCapture.py` reads the camera (GStreamer or OpenCV, `Routines/GstreamerCapture.py`)
  into a shared-memory ring buffer and writes the frame times into FT files. `CaptureDuration.py` computes
  when the night starts and how long it lasts (with the polar day and night cases). `CaptureModeSwitcher.py`
  switches between night and day in continuous mode. `RawFrameSave.py` saves raw frames when enabled.
- **Compression:** `Compression.py` (with `CompressionCy.pyx`) turns every block of 256 frames into an FF
  file (maximum, average and standard deviation images, plus the frame of the maximum of each pixel).
- **Detection:** `DetectStarsAndMeteors.py` runs `ExtractStars.py` (stars, written to CALSTARS) and
  `Detection.py` (meteors: thresholding, Hough/KHT line finding, 3D grouping in
  `Routines/Grouping3Dcy.pyx`) on every FF file, in a pool of workers (`QueuedPool.py`). `MLFilter.py` filters
  false detections with a neural network. `VideoExtraction.py` extracts bright events (FR files).
  `EventMonitor.py` uploads data of events requested by the server.
- **Night processing:** `Reprocess.py` (`processNight`): detection results, automatic calibration
  (`Astrometry/NightCalibration.py`), the pointing check (`Astrometry/CheckFit.py`), recalibration of every FF
  file and the astrometry of the meteors (`Astrometry/ApplyRecalibrate.py`, `Astrometry/ApplyAstrometry.py`),
  the night products, the archive (`ArchiveDetections.py`) and the upload.
- **Upload:** `UploadManager.py` keeps a queue of files and uploads it to the GMN server over SFTP, resuming
  interrupted uploads. `DownloadPlatepar.py` and `DownloadMask.py` fetch files from the server.
- **Configuration:** the `.config` template (INI-like, `;` comments) and `ConfigReader.py` (`Config` defaults,
  `parse()` and its validation).
- **Logging:** `Logger.py`. The main process runs a listener process that writes the night log; child processes
  send their records through a queue (`initChildProcess`). Use `getLogger`.
- **Multiprocessing:** `Misc.setMultiprocessingStartMethod()` is called once in each entry point: `fork` on Linux
  up to Python 3.13, `forkserver` from 3.14, `spawn` elsewhere. Code handed to a child must be picklable for
  `forkserver`/`spawn`.
- **Shared helpers:** `Misc.py` (time, paths, network checks with timeouts), `Math.py`, `Routines/` (image
  operations, masks, binning, morphology, celestial grids, solar longitude, ...).

### Calibration and astrometry (`RMS/Astrometry/`, `RMS/Formats/Platepar.py`)

- `Formats/Platepar.py`: the calibration model of a camera: pointing, lens distortion (polynomial and radial
  models), photometric zero point, and the fits of all of these to matched stars.
- `Conversions.py`: time and coordinate conversions; `CyFunctions.pyx`: the fast pixel to sky conversions.
- `ApplyAstrometry.py` (pixels to sky, FTPdetectinfo), `ApplyAstrometryECSV.py` (ECSV output).
- `CheckFit.py`: the nightly refinement of the pointing on the stars of the night.
- `ApplyRecalibrate.py`: recalibration of every FF file before the meteors are reduced.
- `AutoPlatepar.py`, `AstrometryNet.py`, `AstrometryNetNova.py`, `NNalign.py`: blind solving and automatic
  platepars for new cameras.
- `NightCalibration.py`, `AutoCalibration.py`, `JointDistortionFit.py`: the automatic calibration at the end of
  the night (joint fit of the distortion over the night).
- `MatchStars.py`, `StarFilters.py`, `StarClasses.py`, `AtmosphericExtinction.py`: star matching, filtering and
  photometry helpers.

### File formats (`RMS/Formats/`)

FF files (`FFfile.py`, `FFbin.py`, `FFfits.py`), FR files (`FRbin.py`), FT files, `FTPdetectinfo.py` (meteor
detections and their astrometry), `CALSTARS.py` (stars of each FF file), `Platepar.py`, `ObservationSummary.py`
(the night's JSON summary), `FrameInterface.py` (one interface to FF files, videos, image sequences and `.vid`
files), `StarCatalog.py`, `Showers.py`, and readers for other networks' formats.

### Tools and GUI (`Utils/`)

`SkyFit2.py` is the calibration and manual reduction GUI (pyqtgraph, about 18,000 lines; its widgets are in
`RMS/Routines/CustomPyqtgraphClasses.py` and its help in `RMS/Routines/SkyFitHelp.py`). The other modules are
command-line tools, listed with a short description in `Utils/README.md`.

### Other folders

- `Scripts/`: install, first-run, start, update and watchdog scripts for stations; `Scripts/MultiCamLinux/` for
  computers running several cameras.
- `Catalogs/`: star catalogues; `share/`: shower tables, the machine-learning models, platepar templates and
  other data files.
- `Tests/`: the tests (see above); `Tests/ExampleStationData/` holds example data for a few tests.
- `Guides/`: setup and hardware guides; `Native/`: native code (the Hough transform).


## Review guidelines

Reviews (by people and by Codex) look first for:

- crashes, hangs and unbounded waits in station code: a missing timeout, an unbounded `join()` or lock, a child
  that survives its parent, a retry without a limit;
- data loss or wrong results: files deleted before they are safely stored, wrong times, wrong astrometry or
  photometry, a detection silently dropped;
- broken compatibility: old `.config` files, old platepars, file formats read by the server, Python 3.7 on old
  stations, NumPy 1 and 2, Windows for SkyFit2 and the tools;
- a failing optional step that stops the night processing instead of being logged and skipped;
- CRLF files rewritten with LF endings;
- tests that do not test the change, or would still succeed without it;
- unrelated reformatting or renames mixed into a fix.
