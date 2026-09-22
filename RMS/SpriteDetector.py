# RPi Meteor Station
# Copyright (C) 2025  Dino Grzinic
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

""" Live sprite and elve detection during capture, and the same detection run offline over a night directory.

    During capture, the Compressor hands every new FF file to a SpriteDetector process through a small
    bounded queue. For each FF the process runs the model, keeps detections that pass the artifact filter,
    computes their sky coordinates with the station's platepar, and feeds them to a sliding-window filter that
    throws away bursts (clouds lit by a storm, an aircraft). A detection confirmed by that filter, about a
    minute after it happened, is recorded in the night directory and handed to the upload worker, which sends
    it to the sprite server in the background.

    The pieces live in their own modules: RMS.SpriteDetection (model and artifact filter),
    RMS.SpriteAstrometry, RMS.SpriteFilter, RMS.SpriteProducts (everything written to disk) and
    RMS.SpriteUpload. This module only runs them, live or offline:

        python -m RMS.SpriteDetector ~/RMS_data/CapturedFiles/XX0001_20260827_230000_123456
"""

from __future__ import print_function, division, absolute_import

import datetime
import multiprocessing
import os
import signal
import sys
import time
import traceback

try:
    import queue
except ImportError:
    import Queue as queue

from RMS.DownloadSpriteModel import downloadSpriteModel, spriteModelReady
from RMS.Formats import FFfile
from RMS.Formats.Platepar import Platepar
from RMS.Logger import getLogger, getLoggingQueue, initChildProcess
from RMS.Misc import AtomicFlag
from RMS.Routines import MaskImage
from RMS.SpriteAstrometry import calibrateSpriteDetections, plateparForServer, plateparProvenance, \
    plateparUsable
from RMS.SpriteDetection import SPRITE_TFLITE_AVAILABLE, SPRITE_TFLITE_BACKEND, detectSpritesInFF, \
    getSpriteInterpreter
from RMS.SpriteFilter import SpriteFalsePositiveFilter
from RMS import SpriteProducts
from RMS.SpriteUpload import SpriteUploadWorker


# Get the logger from the main module
log = getLogger("rmslogger")


# FF files waiting for the detector. An FF block is 10 s long, so this is over a minute of backlog; when it is
#   full the Compressor skips files for sprite detection rather than wait (see RMS.Compression)
INPUT_QUEUE_MAXSIZE = 8

# How long the upload worker may keep sending at the end of the night before it gives up for now. Anything
#   unsent stays queued on disk for the next night.
UPLOAD_DRAIN_TIMEOUT = 60

# How long stop() waits for the process to finish its last minute of filtering and the upload drain
SHUTDOWN_TIMEOUT = UPLOAD_DRAIN_TIMEOUT + 60


def unixTime(dt):
    """ Seconds since the epoch of a naive UTC datetime.

        datetime.timestamp() would take a naive value to be local time, which is wrong on any station that
        is not set to UTC.

    Arguments:
        dt: [datetime] Naive, in UTC.

    Return:
        [float]
    """

    return (dt - datetime.datetime(1970, 1, 1)).total_seconds()


def loadStationPlatepar(config, night_data_dir):
    """ Load the platepar used for sprite astrometry, and say where it came from.

        Same order as RMS.Formats.Platepar.findBestPlatepar: the night directory first, then the directory of
        the configuration file. During capture the night directory has no platepar yet, so this is normally
        the station's standing platepar.

    Arguments:
        config: [Config]
        night_data_dir: [str] Night directory.

    Return:
        [tuple] (platepar, path), or (None, None) if there is no readable platepar.
    """

    candidates = [os.path.join(night_data_dir, config.platepar_name),
                  os.path.join(config.config_file_path, config.platepar_name)]

    for path in candidates:

        if not os.path.isfile(path):
            continue

        # Platepar.read returns False instead of raising when the file is missing
        try:
            platepar = Platepar()
            if platepar.read(path, use_flat=config.use_flat) is not False:
                return platepar, path

        except Exception as e:
            log.warning("Could not read the platepar {:s}: {:s}".format(path, repr(e)))

    return None, None


class NightRecorder(object):
    """ Everything that happens to the confirmed and rejected detections of one night: the CSV, the JSON
        record, the images and the hand-over to the upload worker.

        Shared by the live process and the offline run, so that both write exactly the same products.
    """

    def __init__(self, config, night_data_dir, uploader, platepar=None, platepar_path=None):
        """
        Arguments:
            config: [Config]
            night_data_dir: [str] Night directory, where the FF files are and the products go.
            uploader: [SpriteUploadWorker or None] Where confirmed payloads are sent. None to send nothing.

        Keyword arguments:
            platepar: [Platepar or None] Platepar the detections are calibrated with. None by default.
            platepar_path: [str or None] Where it was loaded from. None by default.
        """

        self.config = config
        self.night_data_dir = night_data_dir
        self.night_dir_name = os.path.basename(os.path.normpath(night_data_dir))
        self.uploader = uploader

        # The trimmed platepar goes with every payload; the server keeps one copy under its SHA-256, which
        #   the local record also carries so the two can be matched
        self.server_platepar, platepar_sha256 = plateparForServer(platepar)
        self.platepar_info = plateparProvenance(platepar, platepar_path)
        self.platepar_info["sha256"] = platepar_sha256

        self.csv_path = SpriteProducts.csvPath(night_data_dir)
        self.jsonl_path = SpriteProducts.jsonlPath(night_data_dir)

        # Counts reported at the end of the night
        self.n_confirmed = 0
        self.n_rejected = 0
        self.n_images = 0
        self.n_not_sent = 0
        self.not_sent_logged = False


    def onRejected(self, candidate):
        """ A burst of detections was thrown away by the false-positive filter; record it and move on. """

        self.n_rejected += 1

        SpriteProducts.appendSpriteCSV(self.csv_path, candidate.detections, "burst_rejected")


    def onConfirmed(self, candidate):
        """ A detection passed the false-positive filter: record it, draw it and hand it to the uploader.

        Arguments:
            candidate: [SpriteCandidate] From the false-positive filter.
        """

        ff_name = candidate.ff_name
        detections = candidate.detections

        self.n_confirmed += 1
        log.info("Sprite detection confirmed in {:s}: {:s}".format(
            ff_name, ", ".join("{:s} {:.2f}".format(det["detection_type"], det["confidence"])
                               for det in detections)))

        SpriteProducts.appendSpriteCSV(self.csv_path, detections, "confirmed")

        # The images need the FF, which is still in the night directory. It is read again rather than kept
        #   in memory for the minute it takes the filter to confirm it.
        marked_path, unmarked_path = None, None
        if self.n_images < self.config.sprite_max_images:
            marked_path, unmarked_path = self.writeImages(ff_name, detections)

        # FF start time and frame rate, as the frame indices refer to them
        ff_start = FFfile.filenameToDatetime(ff_name)
        fps = detections[0].get("fps", self.config.fps)

        # Full record of what was found and how it was calibrated, for the nightly archive
        SpriteProducts.appendSpriteJSONL(self.jsonl_path, {
            "ff_name": ff_name,
            "ff_start": SpriteProducts.formatIsoTimestamp(ff_start),
            "fps": fps,
            "platepar": self.platepar_info,
            "detections": detections,
        })

        self.sendToServer(ff_name, ff_start, fps, detections, marked_path, unmarked_path)


    def writeImages(self, ff_name, detections):
        """ Write the marked and unmarked images of a confirmed FF into the night directory.

        Return:
            [tuple] (marked_path, unmarked_path), None for any that could not be written.
        """

        try:
            ff = FFfile.read(self.night_data_dir, ff_name, verbose=False)

        except Exception as e:
            log.warning("Could not read {:s} to draw the sprite images: {:s}".format(ff_name, repr(e)))
            return None, None

        if ff is None:
            return None, None

        marked_path, unmarked_path = SpriteProducts.imagePaths(self.night_data_dir, ff_name)
        written = SpriteProducts.writeDetectionImages(ff, detections, marked_path, unmarked_path)

        # Count the FF once, however many of its two images were written
        if any(written):
            self.n_images += 1

        # Say once that the rest of the night goes without images
        if self.n_images == self.config.sprite_max_images:
            log.info("Written the maximum of {:d} sprite images for this night".format(self.n_images))

        return written


    def sendToServer(self, ff_name, ff_start, fps, detections, marked_path, unmarked_path):
        """ Build the payload for the sprite server and queue it, unless there is nothing it could use. """

        if (self.uploader is None) or (not self.uploader.isEnabled()):
            return

        payload, reason = SpriteProducts.buildSpritePayload(
            self.config, ff_name, ff_start, fps, detections, self.night_dir_name,
            platepar=self.server_platepar)

        # The server triangulates from azimuth and altitude; without them it has no use for the detection
        if payload is None:
            self.n_not_sent += 1

            if not self.not_sent_logged:
                log.warning("Sprite detection in {:s} not sent to the server: {:s}. Further ones for the "
                            "same reason are counted, not logged.".format(ff_name, reason))
                self.not_sent_logged = True

            return

        payload_path = os.path.join(SpriteProducts.uploadDir(self.config, self.night_data_dir),
                                    SpriteProducts.ffStem(ff_name) + ".json")

        if not SpriteProducts.writeUploadPayload(payload, payload_path):
            return

        # The files the server may ask for later, once a second station has seen the same event
        files = {"ff": os.path.join(self.night_data_dir, ff_name)}
        if marked_path:
            files["marked"] = marked_path
        if unmarked_path:
            files["unmarked"] = unmarked_path

        self.uploader.enqueueDetections(ff_name, payload_path, files)


    def summary(self):
        """ One line describing the night, for the log. """

        text = "{:d} confirmed, {:d} rejected as bursts, {:d} with images".format(
            self.n_confirmed, self.n_rejected, self.n_images)

        if self.n_not_sent:
            text += ", {:d} not sent for lack of astrometry".format(self.n_not_sent)

        return text


class NightDetector(object):
    """ Runs the detection of one night, one FF at a time: model, artifact filter, astrometry and the
        false-positive filter. Used by the live process and by the offline run.
    """

    def __init__(self, config, night_data_dir, recorder, platepar, platepar_reason):
        """
        Arguments:
            config: [Config]
            night_data_dir: [str] Night directory.
            recorder: [NightRecorder] Receives the confirmed and rejected detections.
            platepar: [Platepar or None] Platepar for the sky coordinates.
            platepar_reason: [str or None] Why the platepar cannot be used, if it was already known.
        """

        self.config = config
        self.night_data_dir = night_data_dir
        self.recorder = recorder

        self.platepar = platepar
        self.platepar_reason = platepar_reason

        # The platepar check depends on the frame size, so it is done on the first FF and repeated only if
        #   the size ever changes
        self.gate_size = None
        self.gate_ok = False
        self.gate_reason = platepar_reason
        self.gate_warned = False

        # The station mask, loaded once, only when it is to be used
        self.mask = self.loadMask() if config.sprite_use_mask else None

        self.fp_filter = SpriteFalsePositiveFilter(recorder.onConfirmed, on_rejected=recorder.onRejected)

        # Per-night cap on FF blocks with detections
        self.n_detected_ffs = 0
        self.capped = False

        self.n_processed = 0


    def loadMask(self):
        """ Load the station mask image, or None if there is none. """

        mask_path = os.path.join(self.config.config_file_path, self.config.mask_file)

        if not os.path.isfile(mask_path):
            log.warning("sprite_use_mask is set but there is no mask at {:s}; running unmasked".format(
                mask_path))
            return None

        try:
            mask = MaskImage.loadMask(mask_path)
            return mask.img if mask is not None else None

        except Exception as e:
            log.warning("Could not load the mask {:s}, running unmasked: {:s}".format(mask_path, repr(e)))
            return None


    def checkPlatepar(self, ff):
        """ Decide once per frame size whether the platepar can be used for this night's detections. """

        size = (ff.ncols, ff.nrows)

        if size == self.gate_size:
            return

        self.gate_size = size

        if self.platepar is None:
            self.gate_ok = False
            self.gate_reason = self.platepar_reason or "no platepar"

        else:
            self.gate_ok, self.gate_reason = plateparUsable(self.platepar, ff.ncols, ff.nrows, self.config)

        # Once a night is enough to say that the directions are unavailable or may be off
        if (not self.gate_ok) and (not self.gate_warned):
            log.warning("Sprite detections will have no sky coordinates: {:s}".format(self.gate_reason))
            self.gate_warned = True

        elif self.gate_ok and self.gate_reason and (not self.gate_warned):
            log.warning("Sprite astrometry: {:s}".format(self.gate_reason))
            self.gate_warned = True


    def processFF(self, ff_name, model_path):
        """ Run everything on one FF file.

        Arguments:
            ff_name: [str] FF file name, in the night directory.
            model_path: [str] Path to the model.
        """

        self.n_processed += 1

        # The FF start time drives the false-positive filter; it comes from the name, without reading the file
        ff_start = FFfile.filenameToDatetime(ff_name)
        t_ff = unixTime(ff_start)

        # Past the night's cap, only the filter's clock moves on, so the pending detections still get decided
        if self.capped:
            self.fp_filter.tick(t_ff)
            return

        ff = FFfile.read(self.night_data_dir, ff_name, verbose=False)

        if ff is None:
            log.warning("Could not read {:s} for sprite detection".format(ff_name))
            self.fp_filter.tick(t_ff)
            return

        detections = detectSpritesInFF(ff, ff_name, self.config, model_path, mask=self.mask)

        if detections:

            # Sky coordinates, with this FF's own frame times
            self.checkPlatepar(ff)
            calibrateSpriteDetections(detections, self.platepar, self.gate_ok, self.gate_reason)

            # Remember the frame rate the indices refer to, for the payload
            for det in detections:
                det["fps"] = ff.fps if ff.fps > 0 else self.config.fps

            self.fp_filter.addCandidate(t_ff, ff_name, detections)
            self.n_detected_ffs += 1

            # A night with this many hits is a night of false positives; stop spending CPU on it
            cap = self.config.sprite_max_detections_per_night
            if (cap > 0) and (self.n_detected_ffs >= cap):
                self.capped = True
                log.warning("{:d} FF files with sprite detections tonight, the most allowed by "
                            "sprite_max_detections_per_night; sprite detection is off for the rest of the "
                            "night".format(self.n_detected_ffs))

        self.fp_filter.tick(t_ff)


    def finish(self):
        """ Decide every detection still waiting in the false-positive filter. """

        try:
            self.fp_filter.flush()

        except Exception as e:
            log.error("Flushing the sprite false-positive filter failed: {:s}".format(repr(e)))


class SpriteDetector(multiprocessing.Process):
    """ Process that runs sprite detection on every FF file of the night, as the Compressor writes them.

        The Compressor puts (directory, FF name) on input_queue. The process stops when stop() is called and
        the queue is empty; it then confirms the last minute of detections and gives the upload worker a
        bounded time to send what is pending.
    """

    def __init__(self, night_data_dir, config):
        """
        Arguments:
            night_data_dir: [str] Night directory the Compressor writes into.
            config: [Config]
        """

        super(SpriteDetector, self).__init__()

        self.night_data_dir = night_data_dir
        self.config = config

        # Bounded, so a slow machine drops files for sprite detection instead of queueing all night
        self.input_queue = multiprocessing.Queue(maxsize=INPUT_QUEUE_MAXSIZE)

        # Lock-free flags, as in the other capture processes
        self.exit = AtomicFlag()
        self.run_exited = AtomicFlag()

        self.logging_queue = getLoggingQueue()


    def stop(self):
        """ Ask the process to finish, wait for it, and make sure it is gone. Never raises. """

        self.exit.set()

        # Wait for the process to finish its last filtering and the upload drain, but not for a process that
        #   has already died, whose flag would never be set
        t_beg = time.time()
        while (not self.run_exited.is_set()) and self.is_alive():

            if (time.time() - t_beg) > SHUTDOWN_TIMEOUT:
                log.warning("Sprite detector did not finish within {:d} s, stopping it".format(
                    SHUTDOWN_TIMEOUT))
                break

            time.sleep(0.1)

        self.join(5)

        # Escalate only if it is still there. SIGINT would not help: child processes ignore it.
        if self.is_alive():
            self.terminate()
            self.join(5)

        if self.is_alive() and self.pid:
            try:
                os.kill(self.pid, signal.SIGKILL)

            except (OSError, AttributeError):
                pass

            self.join(5)


    def run(self):
        """ Process entry point. Everything is guarded, so a failure can never leave stop() waiting. """

        initChildProcess(self.logging_queue, self.config)

        try:
            self.runDetection()

        except Exception as e:
            log.error("Sprite detector failed: {:s}".format(repr(e)))
            log.error(repr(traceback.format_exception(*sys.exc_info())))

        finally:
            self.run_exited.set()


    def runDetection(self):
        """ The night's work: prepare, then detect on every FF until told to stop. """

        model_path = self.config.sprite_model_path

        # The model is downloaded here rather than before capture starts, so a slow link cannot delay capture
        if not self.prepareModel(model_path):
            self.discardQueue()
            return

        # Old upload payloads are cleared once a night
        SpriteProducts.pruneUploadDirs(self.config)

        platepar, platepar_path = loadStationPlatepar(self.config, self.night_data_dir)
        platepar_reason = None if platepar is not None else "no platepar found"

        # The upload worker sends in the background; it does nothing when uploading is not configured
        uploader = SpriteUploadWorker(self.config)
        uploader.start()

        recorder = NightRecorder(self.config, self.night_data_dir, uploader, platepar, platepar_path)
        detector = NightDetector(self.config, self.night_data_dir, recorder, platepar, platepar_reason)

        log.info("Sprite detector started ({:s} backend, platepar: {:s}, uploads: {:s})".format(
            SPRITE_TFLITE_BACKEND, platepar_path or "none", "on" if uploader.isEnabled() else "off"))

        try:
            while True:

                # Finish when asked to and nothing is left in the queue
                if self.exit.is_set() and self.input_queue.empty():
                    break

                try:
                    _, ff_name = self.input_queue.get(timeout=1.0)

                except queue.Empty:
                    continue

                # One bad FF must not end the night's detection
                try:
                    detector.processFF(ff_name, model_path)

                except Exception as e:
                    log.error("Sprite detection failed on {:s}: {:s}".format(ff_name, repr(e)))
                    log.debug(repr(traceback.format_exception(*sys.exc_info())))

        finally:

            # Decide the last minute of detections, then let the uploads catch up for a while
            detector.finish()
            uploader.stop(drain_timeout=UPLOAD_DRAIN_TIMEOUT)

            log.info("Sprite detector finished: {:d} FF file(s) processed, {:s}".format(
                detector.n_processed, recorder.summary()))


    def prepareModel(self, model_path):
        """ Make sure the model is there and loads.

        Return:
            [bool] True if detection can run.
        """

        if not spriteModelReady(self.config):
            if not downloadSpriteModel(self.config):
                log.warning("Sprite detection is off tonight: the model is not available")
                return False

        # Load it now, so a broken model is reported once rather than on every FF
        try:
            getSpriteInterpreter(model_path)

        except Exception as e:
            log.error("Sprite detection is off tonight: the model {:s} does not load: {:s}".format(
                model_path, repr(e)))
            return False

        return True


    def discardQueue(self):
        """ Keep emptying the queue until stopped, when detection cannot run.

            Otherwise the queue fills up and the Compressor keeps reporting that the detector is behind.
        """

        while not (self.exit.is_set() and self.input_queue.empty()):
            try:
                self.input_queue.get(timeout=1.0)

            except queue.Empty:
                pass


def startSpriteDetector(night_data_dir, config):
    """ Start the sprite detector for a night of capture, if it can run.

    Arguments:
        night_data_dir: [str] Night directory.
        config: [Config]

    Return:
        [SpriteDetector or None] The running process, or None if sprite detection cannot run here.
    """

    if not SPRITE_TFLITE_AVAILABLE:
        log.warning("Sprite detection is enabled but no TFLite backend is installed; skipping it")
        return None

    try:
        sprite_detector = SpriteDetector(night_data_dir, config)
        sprite_detector.start()

    except Exception as e:
        log.error("Could not start the sprite detector: {:s}".format(repr(e)))
        return None

    log.info("Sprite detector started")

    return sprite_detector


def stopSpriteDetector(sprite_detector):
    """ Stop the sprite detector. Never raises, so capture shutdown always goes on.

    Arguments:
        sprite_detector: [SpriteDetector]
    """

    try:
        log.info("Stopping the sprite detector...")
        sprite_detector.stop()
        log.info("Sprite detector stopped")

    except Exception as e:
        log.error("Error while stopping the sprite detector: {:s}".format(repr(e)))


### Offline run over a night directory ###


def runSpriteDetectionDirectory(dir_path, config, upload=False, overwrite=False):
    """ Run sprite detection over every FF file of a night directory, as the live detector would have.

        The FF files are processed in time order through the same false-positive filter as during capture, so
        the result is what the live detector would have produced. It is the way to add sprite detections to a
        night that was captured without them.

    Arguments:
        dir_path: [str] Night directory.
        config: [Config]

    Keyword arguments:
        upload: [bool] Send confirmed detections to the sprite server. False by default.
        overwrite: [bool] Replace this night's sprite products instead of refusing to run. False by default.

    Return:
        [NightRecorder or None] With the night's counts, or None if nothing ran.
    """

    if not SPRITE_TFLITE_AVAILABLE:
        log.error("No TFLite backend is installed, sprite detection cannot run")
        return None

    model_path = config.sprite_model_path

    if (not spriteModelReady(config)) and (not downloadSpriteModel(config)):
        log.error("The sprite model {:s} is not available".format(model_path))
        return None

    # Running twice would record every detection twice; replacing is an explicit choice
    csv_path = SpriteProducts.csvPath(dir_path)
    jsonl_path = SpriteProducts.jsonlPath(dir_path)
    if os.path.isfile(csv_path) or os.path.isfile(jsonl_path):

        if not overwrite:
            log.error("{:s} already has sprite products; run with --overwrite to replace them".format(
                dir_path))
            return None

        for path in (csv_path, jsonl_path):
            if os.path.isfile(path):
                os.remove(path)

    # Time order, as the live detector sees them
    ff_names = sorted([name for name in os.listdir(dir_path) if FFfile.validFFName(name)],
                      key=FFfile.filenameToDatetime)

    platepar, platepar_path = loadStationPlatepar(config, dir_path)
    platepar_reason = None if platepar is not None else "no platepar found"

    uploader = None
    if upload:
        uploader = SpriteUploadWorker(config)
        uploader.start()

    recorder = NightRecorder(config, dir_path, uploader, platepar, platepar_path)
    detector = NightDetector(config, dir_path, recorder, platepar, platepar_reason)

    log.info("Running sprite detection on {:d} FF file(s) in {:s}".format(len(ff_names), dir_path))

    for ff_name in ff_names:
        try:
            detector.processFF(ff_name, model_path)

        except Exception as e:
            log.error("Sprite detection failed on {:s}: {:s}".format(ff_name, repr(e)))

    detector.finish()

    if uploader is not None:
        uploader.stop(drain_timeout=UPLOAD_DRAIN_TIMEOUT)

    log.info("Sprite detection finished: {:s}".format(recorder.summary()))

    return recorder


if __name__ == "__main__":

    import argparse

    import RMS.ConfigReader as cr
    from RMS.Logger import LoggingManager


    arg_parser = argparse.ArgumentParser(
        description="Run sprite detection on the FF files of a night directory.")
    arg_parser.add_argument("dir_path", type=str, help="Night directory with FF files.")
    arg_parser.add_argument("-c", "--config", nargs=1, metavar="CONFIG_PATH", type=str,
                            help="Path to a config file which will be used instead of the default one.")
    arg_parser.add_argument("--model", type=str, help="Model file to use instead of the configured one.")
    arg_parser.add_argument("--confidence", type=float,
                            help="Minimum model confidence, instead of the configured sprite_confidence.")
    arg_parser.add_argument("--upload", action="store_true",
                            help="Send confirmed detections to the configured sprite server.")
    arg_parser.add_argument("--overwrite", action="store_true",
                            help="Replace sprite products already in the directory.")
    cml_args = arg_parser.parse_args()

    dir_path = os.path.abspath(os.path.expanduser(cml_args.dir_path))
    config = cr.loadConfigFromDirectory(cml_args.config, dir_path)

    # Command line overrides of the configuration
    if cml_args.model:
        config.sprite_model_path = os.path.abspath(cml_args.model)
        config.sprite_model_file = os.path.basename(config.sprite_model_path)

    if cml_args.confidence is not None:
        config.sprite_confidence = cml_args.confidence

    # Log to the console and to the usual log directory
    log_manager = LoggingManager()
    log_manager.initLogging(config, "sprites_")
    log = getLogger("rmslogger")

    recorder = runSpriteDetectionDirectory(dir_path, config, upload=cml_args.upload,
                                           overwrite=cml_args.overwrite)

    sys.exit(0 if recorder is not None else 1)
