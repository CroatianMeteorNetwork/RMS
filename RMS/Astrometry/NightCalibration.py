""" Automatic calibration of a classic camera (StartCapture): a platepar during the night when there is none, and the
joint fit of the lens distortion on the whole night at its end.

During the night (NightCalibration, run by runNightCalibration in a low-priority process next to the capture): the
stars which the detection measures on every FF file are read from its backup files in the night directory. As soon
as there are enough clear chunks and the twilight is over, a platepar is fitted from scratch
(AutoCalibration.fitCandidate) and, if it is good on chunks it wasn't fitted on, installed at once. After auto_calibration_check_min minutes it is checked on
new stars (clear chunks taken after it was installed, not used in any fit), and a new one is fitted on all the
clear chunks so far, starting from the installed one (a joint fit of the night so far, see
JointDistortionFit). The two are scored on the same new chunks and the better one is installed. If the installed
platepar is then good, the fitting of the night is done; otherwise, or if no platepar could be fitted, the next
attempt comes auto_calibration_retry_min minutes later with more stars, up to auto_calibration_max_fits attempts a
night. The installed platepar is replaced only by a better one, so it never gets worse during the night.

At the end of the night (endOfNightCalibration, called by Reprocess.processNight before the night is reduced): the
platepar installed during the night (or a platepar fitted from scratch on the night's stars, if none was, or an
existing platepar which fails on the night's stars) is refitted with the joint fit on all clear chunks of the night,
every few chunks held out. It is installed only if it meets the quality control of the joint fit; the replaced
platepar is kept. The night is then reduced with the installed platepar as usual.
"""

from __future__ import print_function, division, absolute_import

import datetime
import json
import os
import shutil

import numpy as np

from RMS.Astrometry import AutoCalibration as ac
from RMS.Logger import getLogger


log = getLogger("rmslogger")


# Name of the state of the calibration of a night, in the night directory
STATE_NAME = 'auto_calibration_night.json'

# Directory of the star lists of the fits, in the night directory
WORK_DIR = 'auto_calibration'

# Seconds between the reads of the new stars during the night
POLL_SECONDS = 60

# At most this many clear chunks in the joint fit of the night so far (during the night), and in the joint fit of
#   the whole night (at its end), and at most this many held out, spread evenly in time; this bounds the time of the
#   fits on a Raspberry Pi
NIGHT_REFIT_MAX_CHUNKS = 60
END_MAX_CHUNKS = 150

# At the end of the night every N-th clear chunk is held out of the joint fit, which needs this many hours of clear
#   chunks, spread over this share of the night (from its first to its last chunk)
END_HOLDOUT_CHUNKS = 4
END_MIN_HOURS = 2.0
END_MIN_SPAN_SHARE = 0.5

# The first fit waits until the number of stars per chunk stops growing (the end of the twilight, or clouds
#   clearing): the median of the last STABLE_MINUTES must not exceed that of the STABLE_MINUTES before by more than
#   STABLE_GROWTH. A fit while the sky is still darkening has few stars and wastes one of the attempts of the night
STABLE_MINUTES = 5.0
STABLE_GROWTH = 1.1

# Number of fits of the distortion in a joint fit
JOINT_ITERATIONS = 3

# Minimum SNR of the stars of a joint fit: not applied, the SNR of the CALSTARS files of the classic stations is on
#   another scale than the default of the joint fit (the clear chunks and the matching select the stars)
JOINT_SNR_MIN = 0



def installPath(config):
    """ Path of the platepar of a classic camera, which the processing reads (Reprocess.getPlatepar).

    Arguments:
        config: [Config] Configuration.

    Return:
        [str] Path of the platepar.
    """

    return os.path.join(config.config_file_path, config.platepar_name)


def installPlatepar(platepar, path, reason, dry_run=False):
    """ Install a platepar: written atomically to its path, keeping the previous one with the time it was replaced
        in its name (replaced_<UTC time>_<name>). In a dry run it is only saved as a candidate next to it
        (candidate_<name>).

    Arguments:
        platepar: [Platepar] The platepar.
        path: [str] Path of the installed platepar.
        reason: [str] Why it is installed (logged).

    Keyword arguments:
        dry_run: [bool] Only save the platepar as a candidate. False by default.

    Return:
        [str] Path of the written platepar.
    """

    dir_path, name = os.path.split(path)

    # In a dry run only a candidate, which nothing uses
    if dry_run:
        path = os.path.join(dir_path, 'candidate_' + name)

    # The replaced platepar is kept, copied so the installed path never goes missing
    elif os.path.isfile(path):
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
        shutil.copy2(path, os.path.join(dir_path, 'replaced_{:s}_{:s}'.format(stamp, name)))

    ac.writePlatepar(platepar, path)
    log.info("Automatic calibration: {:s} platepar saved to {:s} ({:s})".format(
        'candidate' if dry_run else 'new', path, reason))

    return path


def readState(night_dir):
    """ The state of the calibration of a night.

    Arguments:
        night_dir: [str] Night directory.

    Return:
        [dict] The state, a new one if there is none or it can't be read.
    """

    state = {'attempts': 0, 'next_attempt_jd': None, 'installed': None, 'installed_jd': None,
             'check_from_jd': None, 'done': False, 'history': []}

    path = os.path.join(night_dir, STATE_NAME)
    if os.path.isfile(path):
        try:
            with open(path) as f:
                state.update(json.load(f))
        except Exception as e:
            log.warning("The state of the automatic calibration of the night can't be read ({:s}), starting "
                        "over".format(repr(e)))

    return state


def writeState(night_dir, state):
    """ Write the state of the calibration of a night atomically.

    Arguments:
        night_dir: [str] Night directory.
        state: [dict] The state.
    """

    path = os.path.join(night_dir, STATE_NAME)
    with open(path + '.tmp', 'w') as f:
        json.dump(state, f, indent=4, sort_keys=True, default=ac._jsonDefault)
    os.replace(path + '.tmp', path)


def writeChunks(chunks, dir_path, config, ff_frames=256):
    """ Write chunks to the only CALSTARS file of a directory (which the plate solver and the joint fit read).

    Arguments:
        chunks: [list] Chunks as (name, jd, stars).
        dir_path: [str] Directory, emptied of other CALSTARS files.
        config: [Config] Configuration.

    Keyword arguments:
        ff_frames: [int] Number of frames of a chunk. 256 by default.
    """

    from RMS.Formats import CALSTARS

    # The directory has only this CALSTARS file
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    for name in os.listdir(dir_path):
        if name.startswith('CALSTARS'):
            os.remove(os.path.join(dir_path, name))

    # The chunks in time order
    star_list = [[name, [tuple(row) for row in stars]] for name, _, stars in sorted(chunks, key=lambda c: c[1])]
    CALSTARS.writeCALSTARS(star_list, dir_path, 'CALSTARS_{:s}_auto.txt'.format(config.stationID),
                           config.stationID, config.height, config.width, chunk_frames=ff_frames)


def skyStable(chunks, now_jd):
    """ Whether the number of stars per chunk has stopped growing (see STABLE_MINUTES, STABLE_GROWTH).

    Arguments:
        chunks: [list] Chunks as (name, jd, stars), sorted in time.
        now_jd: [float] Julian date of now.

    Return:
        [bool] True if the last STABLE_MINUTES have no more stars than the STABLE_MINUTES before, within the
            growth (False without chunks in both).
    """

    step = STABLE_MINUTES/1440.0
    recent = [len(stars) for _, jd, stars in chunks if jd > now_jd - step]
    before = [len(stars) for _, jd, stars in chunks if now_jd - 2*step < jd <= now_jd - step]
    if (not recent) or (not before):
        return False

    return np.median(recent) <= STABLE_GROWTH*max(np.median(before), 1)


def _mask(config, night_dir):
    """ The mask of the camera.

    Arguments:
        config: [Config] Configuration.
        night_dir: [str] Night directory, where the mask is looked for first (then in the config directory).

    Return:
        [MaskStructure] The mask, None without one.
    """

    # A mask which can't be loaded is no mask
    try:
        from RMS.Routines.MaskImage import getMaskFile
        return getMaskFile(night_dir, config, default_as_backup=True)

    except Exception as e:
        log.warning("Automatic calibration: the mask can't be loaded ({:s})".format(repr(e)))
        return None


def _quietJointFit(dir_paths, platepar, config, **kwargs):
    """ The joint fit of the distortion (JointDistortionFit.jointDistortionFit) on the given directories, its
        errors logged and returned as None.

    Arguments:
        dir_paths: [list] Directories with CALSTARS files.
        platepar: [Platepar] Starting platepar.
        config: [Config] Configuration.

    Keyword arguments:
        The keyword arguments of jointDistortionFit (the minimum SNR is JOINT_SNR_MIN).

    Return:
        (platepar, report): [tuple] The fitted platepar and the report, (None, None) if the fit failed.
    """

    from RMS.Astrometry.JointDistortionFit import jointDistortionFit

    try:
        return jointDistortionFit(dir_paths, platepar, config, snr_min=JOINT_SNR_MIN, **kwargs)

    except Exception as e:
        log.warning("Automatic calibration: the joint fit failed: {:s}".format(repr(e)))
        return None, None



class NightCalibration(object):
    """ The calibration of a classic camera without a platepar during the night (see the module docstring).

    Arguments:
        config: [Config] Configuration.
        night_dir: [str] Night directory (the state and the star lists of the fits are kept there).

    Keyword arguments:
        platepar_path: [str] Path of the installed platepar. installPath(config) by default.
        mask: [MaskStructure] Mask of the camera. None by default, in which case it is loaded (see _mask).
        dry_run: [bool] Only save the fitted platepars as candidates. config.auto_calibration_dry_run by default.
    """

    def __init__(self, config, night_dir, platepar_path=None, mask=None, dry_run=None):

        self.config = config
        self.night_dir = night_dir
        self.platepar_path = platepar_path if platepar_path is not None else installPath(config)
        self.mask = mask if mask is not None else _mask(config, night_dir)
        self.mask_img = getattr(self.mask, 'img', None)
        self.dry_run = config.auto_calibration_dry_run if dry_run is None else dry_run
        self.work_dir = os.path.join(night_dir, WORK_DIR)
        self.state = readState(night_dir)


    def _installed(self):
        """ The platepar installed during this night.

        Return:
            [Platepar] The platepar, None if there is none.
        """

        path = self.state.get('installed')
        if path is None:
            return None

        return ac._platepar(path, self.config.use_flat)


    def _record(self, jd, action, **details):
        """ Record an action of the calibration in the state (saved) and the log.

        Arguments:
            jd: [float] Julian date of the action (of the last chunk).
            action: [str] What was done.

        Keyword arguments:
            Details saved with the action (e.g. the score).

        Return:
            [str] The action.
        """

        entry = dict(jd=jd, action=action, **details)
        self.state['history'].append(entry)
        writeState(self.night_dir, self.state)
        log.info("Automatic calibration of the night: {:s}".format(action))

        return action


    def update(self, chunks, ff_frames=256):
        """ Take the next step of the calibration with the chunks of the night so far.

        Arguments:
            chunks: [list] All chunks of the night so far as (name, jd, stars), the stars as CALSTARS rows.

        Keyword arguments:
            ff_frames: [int] Number of frames of a chunk. 256 by default.

        Return:
            [str] What was done, None if nothing.
        """

        config, state = self.config, self.state
        if state['done'] or (not chunks):
            return None

        chunks = sorted(chunks, key=lambda chunk: chunk[1])
        now_jd = chunks[-1][1]

        # After the last fit of the night nothing more is fitted, but a platepar installed by it is still checked
        #   on new stars
        current = self._installed()
        no_more_fits = state['attempts'] >= config.auto_calibration_max_fits
        if no_more_fits and ((current is None) or (state['check_from_jd'] != state['installed_jd'])):
            state['done'] = True
            return self._record(now_jd, 'no more attempts tonight ({:d})'.format(state['attempts']))

        # Nothing until the time of the next attempt
        if (state['next_attempt_jd'] is not None) and (now_jd < state['next_attempt_jd'] - 1.0/86400):
            return None

        clear = ac.clearChunks(chunks, int(config.width), int(config.height), mask=self.mask_img)
        self._chunks = chunks

        if current is None:
            return self._firstFit(clear, now_jd, ff_frames)

        return self._checkAndRefit(current, clear, now_jd, ff_frames, refit=(not no_more_fits))


    def _firstFit(self, clear, now_jd, ff_frames):
        """ Fit the first platepar of the night from scratch on the clear chunks so far, install it if it is good on
            held-out chunks.

        Arguments:
            clear: [list] Clear chunks so far.
            now_jd: [float] Julian date of the last chunk.
            ff_frames: [int] Number of frames of a chunk.

        Return:
            [str] What was done, None if nothing.
        """

        config, state = self.config, self.state

        # Enough clear sky to fit and judge, with no more stars coming as the sky darkens
        if (len(clear) < ac.MIN_CLEAR_CHUNKS) or (not skyStable(self._chunks, now_jd)):
            return None

        # Every other clear chunk can be held out, the platepar is fitted on sections of the others
        held_out = ac.spreadChunks(clear[1::2], ac.MAX_HELD_OUT)
        held_names = set(name for name, _, _ in held_out)
        pool = [chunk for chunk in clear if chunk[0] not in held_names]

        all_dir = os.path.join(self.work_dir, 'all')
        writeChunks(clear, all_dir, config, ff_frames=ff_frames)
        catalog_stars = ac._loadCatalog(config, now_jd)

        state['attempts'] += 1
        candidate, info = ac.fitCandidate({name: all_dir for name, _, _ in clear}, pool, held_out, catalog_stars,
                                          config, mask=self.mask, use_flat=config.use_flat,
                                          fit_vignetting=(not config.use_flat), ff_frames=ff_frames)

        score = None
        if candidate is not None:
            score = ac.scorePlatepar(candidate, held_out, catalog_stars, config, ff_frames=ff_frames,
                                     mask=self.mask_img)
            ok, reasons = ac.judgeScore(score)
        else:
            ok, reasons = False, [info.get('reason', 'no candidate')]

        # A platepar which fails waits for more stars
        if not ok:
            state['next_attempt_jd'] = now_jd + config.auto_calibration_retry_min/1440.0
            return self._record(now_jd, 'first fit {:d} failed: {:s}'.format(state['attempts'], '; '.join(reasons)),
                                fit=info)

        # Installed at once, and checked on new stars later
        path = installPlatepar(candidate, self.platepar_path, 'first platepar of the night', dry_run=self.dry_run)
        state['installed'] = path
        state['installed_jd'] = state['check_from_jd'] = now_jd
        state['next_attempt_jd'] = now_jd + config.auto_calibration_check_min/1440.0

        return self._record(now_jd, 'installed the first platepar (fit {:d})'.format(state['attempts']),
                            score=score.toDict())


    def _checkAndRefit(self, current, clear, now_jd, ff_frames, refit=True):
        """ Check the installed platepar on new stars, fit a new one on all clear chunks so far starting from it,
            and install the better of the two.

        Arguments:
            current: [Platepar] The installed platepar.
            clear: [list] Clear chunks so far.
            now_jd: [float] Julian date of the last chunk.
            ff_frames: [int] Number of frames of a chunk.

        Keyword arguments:
            refit: [bool] Also fit a new platepar (an attempt of the night). False only checks the installed one
                (after the last attempt), which ends the night's calibration. True by default.

        Return:
            [str] What was done, None if nothing.
        """

        config, state = self.config, self.state

        # The new stars: clear chunks taken after the installed platepar was last checked or installed
        new = [chunk for chunk in clear if chunk[1] > state['check_from_jd']]
        if len(new) < ac.MIN_CLEAR_CHUNKS:
            return None
        check = ac.spreadChunks(new, ac.MAX_HELD_OUT)
        check_names = set(name for name, _, _ in check)

        catalog_stars = ac._loadCatalog(config, now_jd)

        # A new platepar from all clear chunks so far except the new ones it is checked on, starting from the
        #   installed one (an attempt of the night)
        candidate = None
        if refit:
            state['attempts'] += 1
            fit_dir = os.path.join(self.work_dir, 'fit')
            writeChunks([chunk for chunk in clear if chunk[0] not in check_names], fit_dir, config,
                        ff_frames=ff_frames)
            candidate, _ = _quietJointFit([fit_dir], current, config, n_iter=JOINT_ITERATIONS, chunk_step=1,
                                          max_chunks=NIGHT_REFIT_MAX_CHUNKS, nproc=1, qc=False)

        # Both scored on the same new chunks, the better one is installed
        current_score = ac.scorePlatepar(current, check, catalog_stars, config, ff_frames=ff_frames,
                                         mask=self.mask_img)
        best, best_score, action = current, current_score, 'kept the installed platepar'
        if candidate is not None:
            candidate_score = ac.scorePlatepar(candidate, check, catalog_stars, config, ff_frames=ff_frames,
                                               mask=self.mask_img)
            better, reason = ac.candidateIsBetter(candidate_score, current_score)
            if better:
                path = installPlatepar(candidate, self.platepar_path, 'refit: ' + reason, dry_run=self.dry_run)
                state['installed'] = path
                state['installed_jd'] = now_jd
                best, best_score, action = candidate, candidate_score, 'installed the refit ({:s})'.format(reason)
            else:
                action += ' (refit: {:s})'.format(reason)
        elif refit:
            action += ' (the refit failed)'
        else:
            action = 'checked the platepar of the last attempt'

        # The next new stars are those after this attempt
        state['check_from_jd'] = now_jd

        # Done if the installed platepar is good on the new stars (or after the check of the last attempt),
        #   otherwise the next attempt comes later
        ok, reasons = ac.judgeScore(best_score)
        if not refit:
            state['done'] = True
            return self._record(now_jd, '{:s}: it is {:s} on new stars; done for the night'.format(
                action, 'good' if ok else 'not good ({:s})'.format('; '.join(reasons))), score=best_score.toDict())
        if ok:
            state['done'] = True
            return self._record(now_jd, 'attempt {:d}: {:s}, it is good on new stars; done for the night'.format(
                state['attempts'], action), score=best_score.toDict())

        state['next_attempt_jd'] = now_jd + config.auto_calibration_retry_min/1440.0
        return self._record(now_jd, 'attempt {:d}: {:s}, it fails on new stars ({:s})'.format(
            state['attempts'], action, '; '.join(reasons)), score=best_score.toDict())



def readDetectionBackups(night_dir, seen):
    """ The stars of the FF files the detection finished since the last read, from its backup files in the night
        directory (see QueuedPool.saveBackupFile).

    Arguments:
        night_dir: [str] Night directory.
        seen: [set] Names of the backup files already read, updated in place.

    Return:
        [list] New chunks as (FF name, stars), the stars as CALSTARS rows.
    """

    from RMS.DetectStarsAndMeteors import starRows
    from RMS.Pickling import loadPickle

    chunks = []
    for name in sorted(os.listdir(night_dir)):
        if (name in seen) or (not name.startswith('rms_queue_bkup_')) or (not name.endswith('.pickle')):
            continue

        # A file which is still being written is read the next time
        try:
            backup = loadPickle(night_dir, name)
            ff_name, star_data = backup.outputs[0], backup.outputs[1]
        except Exception:
            continue

        seen.add(name)
        rows = starRows(star_data)
        if rows:
            chunks.append((ff_name, np.array(rows, dtype=np.float64)))

    return chunks


def _chunkJD(ff_name, config, ff_frames=256):
    """ The Julian date of the middle of a chunk from its FF name.

    Arguments:
        ff_name: [str] FF name.
        config: [Config] Configuration.

    Keyword arguments:
        ff_frames: [int] Number of frames of a chunk. 256 by default.

    Return:
        [float] The Julian date, None if the name has no time.
    """

    # The middle of the chunk, as the recalibration uses it
    from RMS.Astrometry.Conversions import date2JD
    from RMS.Formats.FFfile import getMiddleTimeFF

    try:
        return date2JD(*getMiddleTimeFF(ff_name, config.fps, ret_milliseconds=True, ff_frames=ff_frames))
    except Exception:
        return None


def runNightCalibration(config, night_dir, stop_event, poll_seconds=POLL_SECONDS):
    """ Calibrate a classic camera during the night (see NightCalibration), reading the stars of the detection as it
        runs, until the stop event is set or the night's fitting is done. Runs at a low priority.

    Arguments:
        config: [Config] Configuration.
        night_dir: [str] Night directory.
        stop_event: [multiprocessing.Event] Set at the end of the capture.

    Keyword arguments:
        poll_seconds: [float] Seconds between the reads of the new stars. POLL_SECONDS by default.
    """

    # The capture and the detection go first
    try:
        os.nice(19)
    except (AttributeError, OSError):
        pass

    calibration = NightCalibration(config, night_dir)
    seen, chunks = set(), []
    while not stop_event.is_set():

        # The new stars, then the next step
        try:
            for ff_name, stars in readDetectionBackups(night_dir, seen):
                jd = _chunkJD(ff_name, config)
                if jd is not None:
                    chunks.append((ff_name, jd, stars))
            calibration.update(chunks)

        # The calibration never stops the capture
        except Exception as e:
            log.warning("Automatic calibration of the night failed: {:s}".format(repr(e)))

        if calibration.state['done']:
            break

        stop_event.wait(poll_seconds)


class NightCalibrationProcess(object):
    """ The calibration of the night in its own process, next to the capture (see runNightCalibration). """

    def __init__(self, config, night_dir):
        """ Start the calibration of the night in its own process.

        Arguments:
            config: [Config] Configuration.
            night_dir: [str] Night directory.
        """

        import multiprocessing

        # The process stops when the event is set (or at the end of the main process)
        self.stop_event = multiprocessing.Event()
        self.process = multiprocessing.Process(target=runNightCalibration, args=(config, night_dir, self.stop_event),
                                               name='NightCalibration')
        self.process.daemon = True
        self.process.start()
        log.info("Automatic calibration of the night started (no platepar)")


    def stop(self, timeout=600):
        """ Stop the calibration at the end of the capture, waiting for a fit in progress up to the timeout.

        Keyword arguments:
            timeout: [float] Seconds to wait. 600 by default.
        """

        # A fit in progress finishes, a hung one is ended
        self.stop_event.set()
        self.process.join(timeout)
        if self.process.is_alive():
            log.warning("Automatic calibration of the night didn't stop in {:d} s, it is terminated".format(timeout))
            self.process.terminate()
            self.process.join(10)



def jointFitNight(clear, start, config, night_dir, night_hours, ff_frames=256, nproc=1,
                  max_chunks=END_MAX_CHUNKS, mask=None):
    """ The joint fit of the lens distortion on the clear chunks of a night, every END_HOLDOUT_CHUNKS-th chunk held
        out, with the quality control of a night (END_MIN_HOURS of clear chunks over END_MIN_SPAN_SHARE of the
        night). The report of the fit is written to the night directory (auto_calibration_joint_report.txt).

    Arguments:
        clear: [list] Clear chunks of the night as (name, jd, stars).
        start: [Platepar] Starting platepar.
        config: [Config] Configuration.
        night_dir: [str] Night directory (the star lists of the fit go to its WORK_DIR).
        night_hours: [float] Hours from the first to the last chunk of the night.

    Keyword arguments:
        ff_frames: [int] Number of frames of a chunk. 256 by default.
        nproc: [int] Workers of the fit. 1 by default.
        max_chunks: [int] At most this many chunks in the fit and held out. END_MAX_CHUNKS by default.
        mask: [ndarray] Mask image of the camera (0 is masked), for the quality control. None by default.

    Return:
        (fitted, report): [tuple] The fitted platepar and the report with the verdict of the quality control
            (report['qc']['ok']), (None, None) if the fit failed.
    """

    from RMS.Astrometry.JointDistortionFit import formatReport

    # The clear chunks of the night in the only CALSTARS file of a directory of the fit
    fit_dir = os.path.join(night_dir, WORK_DIR, 'night')
    writeChunks(clear, fit_dir, config, ff_frames=ff_frames)

    # Every few clear chunks held out, with enough clear sky over the night
    fitted, report = _quietJointFit([fit_dir], start, config, n_iter=JOINT_ITERATIONS, chunk_step=1,
                                    holdout_chunks=END_HOLDOUT_CHUNKS, max_chunks=max_chunks, nproc=nproc,
                                    qc_options={'min_hours': END_MIN_HOURS,
                                                'min_span_hours': END_MIN_SPAN_SHARE*night_hours}, mask=mask)

    # The report of the fit is kept with the night
    if report is not None:
        with open(os.path.join(night_dir, 'auto_calibration_joint_report.txt'), 'w') as f:
            f.write(formatReport(report))

    return fitted, report


def endOfNightCalibration(config, night_dir, calstars_name, platepar, platepar_path, platepar_fmt, dry_run=None,
                          nproc=None):
    """ The joint fit of the lens distortion on the clear chunks of the whole night (see the module docstring),
        before the night is reduced.

    Arguments:
        config: [Config] Configuration.
        night_dir: [str] Night directory.
        calstars_name: [str] Name of the CALSTARS file of the night.
        platepar: [Platepar] Platepar of the night (Reprocess.getPlatepar), None if there is none.
        platepar_path: [str] Its path, the platepar is installed there (installPath(config) if there is none).
        platepar_fmt: [str] Its format.

    Keyword arguments:
        dry_run: [bool] Only save the fitted platepars as candidates. config.auto_calibration_dry_run by default.
        nproc: [int] Workers of the joint fit. config.auto_calibration_nproc by default.

    Return:
        (platepar, platepar_path, platepar_fmt): [tuple] The platepar to reduce the night with.
    """

    from RMS.Astrometry.AutoCalibration import readChunks

    dry_run = config.auto_calibration_dry_run if dry_run is None else dry_run
    nproc = config.auto_calibration_nproc if nproc is None else nproc
    if platepar_path is None:
        platepar_path = installPath(config)

    state = readState(night_dir)
    mask = _mask(config, night_dir)
    mask_img = getattr(mask, 'img', None)

    # The clear chunks of the night
    chunks, ff_frames = readChunks(night_dir, config, file_name=calstars_name)
    clear = ac.clearChunks(chunks, int(config.width), int(config.height), mask=mask_img)
    if len(clear) < ac.MIN_CLEAR_CHUNKS:
        log.info("Automatic calibration at the end of the night: not enough clear sky ({:d} clear chunks)".format(
            len(clear)))
        return platepar, platepar_path, platepar_fmt

    held_out = ac.spreadChunks(clear[1::2], ac.MAX_HELD_OUT)
    catalog_stars = ac._loadCatalog(config, clear[len(clear)//2][1])
    fitted_tonight = state.get('installed') is not None

    # The start of the joint fit: the platepar installed tonight, a platepar fitted from scratch on the night if
    #   there is none, or an existing platepar which fails on the night's stars. An existing platepar which is good
    #   is kept as it is
    start = platepar
    if start is None:
        held_names = set(name for name, _, _ in held_out)
        pool = [chunk for chunk in clear if chunk[0] not in held_names]
        all_dir = os.path.join(night_dir, WORK_DIR, 'all')
        writeChunks(clear, all_dir, config, ff_frames=ff_frames)
        start, info = ac.fitCandidate({name: all_dir for name, _, _ in clear}, pool, held_out, catalog_stars,
                                      config, mask=mask, use_flat=config.use_flat,
                                      fit_vignetting=(not config.use_flat), ff_frames=ff_frames)
        if start is None:
            log.info("Automatic calibration at the end of the night: no platepar could be fitted ({:s})".format(
                info.get('reason', '')))
            return platepar, platepar_path, platepar_fmt
        ok, reasons = ac.judgeScore(ac.scorePlatepar(start, held_out, catalog_stars, config, ff_frames=ff_frames,
                                                     mask=mask_img))
        if not ok:
            log.info("Automatic calibration at the end of the night: the platepar fitted from scratch fails "
                     "({:s})".format('; '.join(reasons)))
            return platepar, platepar_path, platepar_fmt

    elif not fitted_tonight:
        ok, reasons = ac.judgeScore(ac.scorePlatepar(platepar, held_out, catalog_stars, config,
                                                     ff_frames=ff_frames, mask=mask_img))
        if ok:
            log.info("Automatic calibration at the end of the night: the platepar is good, it is kept")
            return platepar, platepar_path, platepar_fmt
        log.info("Automatic calibration at the end of the night: the platepar fails ({:s}), it is refitted".format(
            '; '.join(reasons)))

    # The joint fit of the whole night
    fitted, report = jointFitNight(clear, start, config, night_dir, (chunks[-1][1] - chunks[0][1])*24.0,
                                   ff_frames=ff_frames, nproc=nproc, mask=mask_img)

    # Installed if it meets the quality control; otherwise the start is installed if it isn't yet (it was good on
    #   held-out chunks)
    if (fitted is not None) and report['qc']['ok']:
        installed, reason = fitted, 'joint fit of the night'
    elif start is not platepar:
        installed, reason = start, 'fitted from scratch on the night (the joint fit {:s})'.format(
            'failed' if report is None else 'fails: ' + ', '.join(report['qc']['failed']))
    else:
        log.info("Automatic calibration at the end of the night: the joint fit {:s}, the platepar is kept".format(
            'failed' if report is None else 'fails its quality control: ' + ', '.join(report['qc']['failed'])))
        return platepar, platepar_path, platepar_fmt

    path = installPlatepar(installed, platepar_path, reason, dry_run=dry_run)
    if dry_run:
        return platepar, platepar_path, platepar_fmt

    from RMS.Formats.Platepar import Platepar
    new_platepar = Platepar()
    fmt = new_platepar.read(path, use_flat=config.use_flat)

    return new_platepar, path, fmt




### Periodic refit over many nights ###

# Name of the state of the periodic refit, in the config directory
REFIT_STATE_NAME = 'auto_calibration_refit.json'

# Directory of the star lists of the refit, in the data directory
REFIT_WORK_DIR = 'auto_calibration_refit'

# A night is used in the refit only with this many hours of clear chunks, and the refit needs this many nights
REFIT_MIN_CLEAR_HOURS = 1.0
REFIT_MIN_NIGHTS = 4

# Every N-th night from the K-th is held out of the refit, and at most this many chunks are in the fit (and held out)
REFIT_HOLDOUT = (4, 1)
REFIT_MAX_CHUNKS = 400

# The refit replaces the platepar only if the worst zone or corner of the held-out nights gets better by at least
#   this many standard errors of its mean residual (besides the quality control of the joint fit). A share of the
#   width of the stars was tried first: 0.1 FWHM is 50" on the 1280x720 cameras of the network, more than the mean
#   residuals of their corners, so a refit which made the worst corner 18" better (57" -> 39") was not used
REFIT_MIN_GAIN_SE = 3.0


def _worstZone(stats, min_stars):
    """ The largest mean residual of the zones and corners with enough stars.

    Arguments:
        stats: [dict] Residual statistics (see JointDistortionFit.residualStats).
        min_stars: [int] Minimum number of stars of a zone.

    Return:
        (size, n): [tuple] The largest mean residual (arcsec) and the number of stars of its zone, (0, 0) without
            a zone with enough stars.
    """

    from RMS.Astrometry.JointDistortionFit import CORNER_NAMES

    zones = [zone for row in stats['zones'] for zone in row] + [stats['corners'][name] for name in CORNER_NAMES]
    sizes = [(np.hypot(dx, dy), n) for dx, dy, n in zones
             if (n >= max(min_stars, 1)) and np.isfinite(dx) and np.isfinite(dy)]

    return max(sizes) if sizes else (0.0, 0)


def refitNights(night_dirs, platepar, config, work_dir, max_nights=None, nproc=1):
    """ Refit the lens distortion of a platepar on the clear chunks of many nights with the joint fit (every few
        nights held out), and decide whether it replaces the platepar: the fit has to meet its quality control,
        and the worst zone or corner of the held-out nights has to get better by REFIT_MIN_GAIN_SE standard errors
        of its mean residual, both compared with the given platepar (its distortion with the pointing of every
        night). The fitted distortion gets the pointing of the given platepar.

    Arguments:
        night_dirs: [list] Directories of the nights with their CALSTARS files (and platepars), the newest first.
        platepar: [Platepar] The platepar of the camera.
        config: [Config] Configuration.
        work_dir: [str] Directory of the star lists of the fit (emptied first).

    Keyword arguments:
        max_nights: [int] At most this many nights (the newest with enough clear sky). None by default (all).
        nproc: [int] Workers of the fit. 1 by default.

    Return:
        (platepar, report, reason): [tuple] The platepar which replaces the given one (None if it isn't
            replaced), the report of the joint fit (None if there was no fit) and why.
    """

    from RMS.Astrometry.JointDistortionFit import transferPointing

    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)

    # The newest nights with enough clear sky, each with its clear chunks (and its platepar, the start of its
    #   pointing) in a directory of the fit
    selected = []
    for night_dir in night_dirs:
        if (max_nights is not None) and (len(selected) >= max_nights):
            break

        # The clear chunks judged with the mask of the night (or of the camera), as during the night
        chunks, ff_frames = ac.readChunks(night_dir, config)
        mask = _mask(config, night_dir)
        clear = ac.clearChunks(chunks, int(config.width), int(config.height), mask=getattr(mask, 'img', None))
        if len(clear)*ff_frames/config.fps/3600.0 < REFIT_MIN_CLEAR_HOURS:
            continue

        fit_dir = os.path.join(work_dir, os.path.basename(os.path.normpath(night_dir)))
        writeChunks(clear, fit_dir, config, ff_frames=ff_frames)
        night_platepar = os.path.join(night_dir, config.platepar_name)
        if os.path.isfile(night_platepar):
            shutil.copy2(night_platepar, os.path.join(fit_dir, config.platepar_name))
        selected.append(fit_dir)

    if len(selected) < REFIT_MIN_NIGHTS:
        return None, None, 'only {:d} nights with {:.0f} h of clear sky'.format(len(selected), REFIT_MIN_CLEAR_HOURS)

    # The joint fit on the nights, every few nights held out, which are compared with the platepar of the camera
    #   (its distortion with the pointing of every night carried over), the one the fit would replace
    fitted, report = _quietJointFit(selected, platepar, config, n_iter=JOINT_ITERATIONS, chunk_step=1,
                                    holdout=REFIT_HOLDOUT, max_chunks=REFIT_MAX_CHUNKS, nproc=nproc,
                                    holdout_baseline='start')
    if fitted is None:
        return None, None, 'the joint fit failed'
    if not report['qc']['ok']:
        return None, report, 'it fails the quality control ({:s})'.format(', '.join(report['qc']['failed']))

    # The worst zone of the held-out nights has to get clearly better: by a few standard errors of its mean
    #   residual (the rms residual of the held-out stars over the square root of the stars of the zone)
    from RMS.Astrometry.JointDistortionFit import QC_DEFAULTS
    held = report['held_out_stats']
    min_stars = QC_DEFAULTS['min_zone_stars']
    worst_before, n_before = _worstZone(held['start']['stats'], min_stars)
    worst_after, _ = _worstZone(held['fitted']['stats'], min_stars)
    gain = worst_before - worst_after
    needed = REFIT_MIN_GAIN_SE*held['start']['stats']['rms']/np.sqrt(max(n_before, 1))
    if gain < needed:
        return None, report, 'not clearly better (worst zone {:+.2f}", needed {:.2f}")'.format(gain, needed)

    # The fitted distortion with the pointing of the platepar of the camera
    fitted, _ = transferPointing(fitted, platepar)

    return fitted, report, 'the worst zone {:.2f}" better on {:d} nights'.format(gain, len(selected))


def periodicRefit(config, nproc=None, now=None):
    """ The periodic refit of the lens distortion of a classic camera on its recent nights (see refitNights):
        every auto_calibration_refit_days days, on the newest auto_calibration_refit_nights archived nights with
        enough clear sky. The state (the time of the last refit and its outcome) is kept in the config directory,
        the report next to it.

    Arguments:
        config: [Config] Configuration.

    Keyword arguments:
        nproc: [int] Workers of the fit. config.auto_calibration_nproc by default.
        now: [datetime] The time now (UTC). None by default (the clock).

    Return:
        [str] What was done, None if it wasn't time for a refit.
    """

    from RMS.Astrometry.JointDistortionFit import formatReport

    if config.auto_calibration_refit_days <= 0:
        return None

    nproc = config.auto_calibration_nproc if nproc is None else nproc
    now = now if now is not None else datetime.datetime.now(datetime.timezone.utc)

    # Only every few days
    state_path = os.path.join(config.config_file_path, REFIT_STATE_NAME)
    state = {}
    if os.path.isfile(state_path):
        try:
            with open(state_path) as f:
                state = json.load(f)
        except Exception:
            state = {}
    last = state.get('last_refit_utc')
    if last is not None:
        last = datetime.datetime.strptime(last, '%Y-%m-%d %H:%M:%S').replace(tzinfo=datetime.timezone.utc)
        if now - last < datetime.timedelta(days=config.auto_calibration_refit_days):
            return None

    # The platepar of the camera
    platepar_path = installPath(config)
    platepar = ac._platepar(platepar_path, config.use_flat)
    if platepar is None:
        return None

    # The archived nights of the camera, the newest first
    archived = os.path.join(config.data_dir, config.archived_dir)
    night_dirs = []
    if os.path.isdir(archived):
        night_dirs = sorted((os.path.join(archived, name) for name in os.listdir(archived)
                             if name.startswith(config.stationID) and os.path.isdir(os.path.join(archived, name))),
                            reverse=True)

    work_dir = os.path.join(config.data_dir, REFIT_WORK_DIR)
    fitted, report, reason = refitNights(night_dirs, platepar, config, work_dir,
                                         max_nights=config.auto_calibration_refit_nights, nproc=nproc)

    # The report next to the star lists of the fit
    if report is not None:
        with open(os.path.join(work_dir, 'auto_calibration_refit_report.txt'), 'w') as f:
            f.write(formatReport(report))

    # Installed if it is clearly better
    if fitted is not None:
        installPlatepar(fitted, platepar_path, 'refit on the recent nights: ' + reason,
                        dry_run=config.auto_calibration_dry_run)
        action = 'installed the refit ({:s})'.format(reason)
    else:
        action = 'kept the platepar ({:s})'.format(reason)
    log.info("Automatic calibration, periodic refit: {:s}".format(action))

    # The time of this refit, also if nothing was installed
    state['last_refit_utc'] = now.strftime('%Y-%m-%d %H:%M:%S')
    state['last_action'] = action
    with open(state_path + '.tmp', 'w') as f:
        json.dump(state, f, indent=4, sort_keys=True)
    os.replace(state_path + '.tmp', state_path)

    return action


def replayNight(night_dir, config, out_dir, step_minutes=1.0, nproc=None):
    """ Replay the automatic calibration of a recorded night: its chunks (from its CALSTARS file) are given to the
        calibration of the night in time order, as the detection would give them, a step every few minutes of data;
        then the end of the night follows. Nothing is changed in the night directory: the platepars, the state and
        the star lists of the fits are written to the output directory, which stands for both the config
        directory and the night directory. The time the fits take is not simulated.

    Arguments:
        night_dir: [str] Directory of the recorded night, with its CALSTARS file (and a mask, if any).
        config: [Config] Configuration of the camera.
        out_dir: [str] Output directory.

    Keyword arguments:
        step_minutes: [float] Minutes of data between the steps of the calibration. 1 by default.
        nproc: [int] Workers of the joint fit at the end of the night. config.auto_calibration_nproc by default.

    Return:
        (state, platepar): [tuple] The state of the calibration of the night, and the platepar installed at the
            end (None if there is none).
    """

    from RMS.Astrometry.AutoCalibration import readChunks

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # The output directory is the config directory (where the platepar is installed) and the night directory
    config.config_file_path = out_dir
    calstars_names = sorted(name for name in os.listdir(night_dir) if name.startswith('CALSTARS')
                            and name.endswith('.txt'))
    if not calstars_names:
        raise ValueError("No CALSTARS file in {:s}".format(night_dir))
    shutil.copy2(os.path.join(night_dir, calstars_names[0]), os.path.join(out_dir, calstars_names[0]))
    if os.path.isfile(os.path.join(night_dir, config.mask_file)):
        shutil.copy2(os.path.join(night_dir, config.mask_file), os.path.join(out_dir, config.mask_file))

    chunks, ff_frames = readChunks(out_dir, config, file_name=calstars_names[0])
    if not chunks:
        raise ValueError("No chunks in {:s}".format(calstars_names[0]))

    # The night step by step, as the stars come in
    calibration = NightCalibration(config, out_dir, platepar_path=installPath(config), dry_run=False)
    step = step_minutes/1440.0
    t = chunks[0][1] + step
    while (t <= chunks[-1][1] + step) and (not calibration.state['done']):
        calibration.update([chunk for chunk in chunks if chunk[1] <= t], ff_frames=ff_frames)
        t += step

    # The end of the night
    platepar = ac._platepar(installPath(config), config.use_flat)
    platepar, _, _ = endOfNightCalibration(config, out_dir, calstars_names[0], platepar, installPath(config),
                                           'json', dry_run=False, nproc=nproc)

    return readState(out_dir), platepar


def main(argv=None):
    """ The command line interface: replay the automatic calibration of a recorded night (see replayNight).

    Keyword arguments:
        argv: [list] Command line arguments. None by default, in which case those of the program are used.

    Return:
        [int] Exit code: 0 if a platepar was installed, 1 otherwise.
    """

    import argparse

    import RMS.ConfigReader as cr

    arg_parser = argparse.ArgumentParser(description="Replay the automatic calibration of a recorded night of a "
                                         "camera without a platepar: during the night and at its end.")
    arg_parser.add_argument('night_dir', metavar='NIGHT_DIR', type=str,
                            help="Directory of the night, with its CALSTARS file.")
    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str,
                            help="Path to the config file of the camera.")
    arg_parser.add_argument('-o', '--out', metavar='OUT_DIR', type=str, required=True,
                            help="Output directory (platepars, state, reports).")
    arg_parser.add_argument('--nproc', metavar='N_PROC', type=int,
                            help="Workers of the joint fit at the end of the night.")
    cml_args = arg_parser.parse_args(argv)

    config = cr.loadConfigFromDirectory(cml_args.config, cml_args.night_dir)
    getLogger("rmslogger", level="INFO")

    state, platepar = replayNight(cml_args.night_dir, config, cml_args.out, nproc=cml_args.nproc)
    for entry in state['history']:
        print('{:s}  {:s}'.format(str(ac.jd2Date(entry['jd'], dt_obj=True))[:19], entry['action']))
    print('Installed platepar: {:s}'.format(installPath(config) if platepar is not None else 'none'))

    return 0 if platepar is not None else 1


if __name__ == "__main__":

    import sys

    sys.exit(main())
