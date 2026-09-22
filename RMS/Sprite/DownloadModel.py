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

""" Download the sprite detection model from the GMN web server when it is missing or outdated.

    The model is about 35 MB, three hundred times the size of the meteor ML model, so it is not kept in the
    repository. Stations fetch it over HTTPS from config.sprite_model_base_url; no key or account is needed.
    It is a YOLOv5su model trained with Ultralytics and is licensed under AGPL-3.0; the README.md next to it
    on the server describes it.

    The expected SHA-256 of every known model file is listed in KNOWN_MODEL_SHA256, which ties a model to the
    code that knows how to run it. A downloaded file that does not match is discarded, so the web server only
    has to deliver the bytes, it does not have to be trusted.
"""

from __future__ import print_function, division, absolute_import

import hashlib
import os
import ssl

from RMS.Logger import getLogger

# Python 2/3 compatibility for urllib
try:
    import urllib2 as urllib_request
    import urllib2 as urllib_error

except ImportError:
    import urllib.request as urllib_request
    import urllib.error as urllib_error


# Get the logger from the main module
log = getLogger("rmslogger")


# SHA-256 of the model files this code knows how to run. A new model gets a new entry here, together with
#   whatever change to the detection code it needs.
KNOWN_MODEL_SHA256 = {
    "sprite_detector.tflite": "a73da954ab97a7b68683edc249eafe457f1085bfd8d619f05c9e0d94cdad831a",
}

# Size of the pieces the file is read and hashed in, to keep memory low on a Raspberry Pi
CHUNK_BYTES = 1024*1024

# A download larger than this is not a model; stop instead of filling the SD card
MAX_MODEL_BYTES = 200*1024*1024

# Seconds without any data before the download is given up
DOWNLOAD_TIMEOUT = 60


def fileSha256(file_path):
    """ SHA-256 of a file, as lower case hex.

    Arguments:
        file_path: [str] Path to the file.

    Return:
        [str] Hex digest, or None if the file cannot be read.
    """

    digest = hashlib.sha256()

    try:
        with open(file_path, "rb") as f:

            # Hash in pieces rather than reading 35 MB into memory at once
            while True:
                chunk = f.read(CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)

    except (IOError, OSError):
        return None

    return digest.hexdigest()


def expectedModelSha256(config):
    """ SHA-256 the configured model file must have, or None if the file is not one this code knows.

    Arguments:
        config: [Config]

    Return:
        [str or None]
    """

    return KNOWN_MODEL_SHA256.get(config.sprite_model_file)


def spriteModelReady(config):
    """ Whether the configured model file is present and, for a known model, intact.

    Arguments:
        config: [Config]

    Return:
        [bool]
    """

    if not os.path.isfile(config.sprite_model_path):
        return False

    expected = expectedModelSha256(config)

    # A model this code does not know about was put there on purpose; take it as it is
    if expected is None:
        return True

    return fileSha256(config.sprite_model_path) == expected


def spriteModelUrl(config):
    """ URL the configured model file is downloaded from.

    Arguments:
        config: [Config]

    Return:
        [str]
    """

    return config.sprite_model_base_url.rstrip("/") + "/" + config.sprite_model_file


def _fetchToFile(url, file_path):
    """ Stream a URL into a file, with a verified TLS connection and a size limit.

    Arguments:
        url: [str] https URL.
        file_path: [str] Where to write it.

    Return:
        [str or None] None on success, otherwise why it failed.
    """

    # Only HTTPS with a checked certificate; never fall back to an unverified connection
    if not url.lower().startswith("https://"):
        return "the model URL must start with https://"

    if not hasattr(ssl, "create_default_context"):
        return "this Python cannot verify HTTPS certificates"

    context = ssl.create_default_context()
    request = urllib_request.Request(url, headers={"User-Agent": "RMS-sprite-model-download"})

    response = urllib_request.urlopen(request, timeout=DOWNLOAD_TIMEOUT, context=context)

    try:

        # Refuse early when the server already says the file is too big
        length = response.headers.get("Content-Length")
        if length is not None and length.isdigit() and int(length) > MAX_MODEL_BYTES:
            return "the server reports {:s} bytes, more than the {:d} allowed".format(length, MAX_MODEL_BYTES)

        written = 0

        with open(file_path, "wb") as f:

            # Copy in pieces, stopping if the file grows past the limit
            while True:
                chunk = response.read(CHUNK_BYTES)
                if not chunk:
                    break

                written += len(chunk)
                if written > MAX_MODEL_BYTES:
                    return "the download exceeded {:d} bytes".format(MAX_MODEL_BYTES)

                f.write(chunk)

        # A connection cut short can end the stream early without an error
        if length is not None and length.isdigit() and written != int(length):
            return "received {:d} of {:s} bytes".format(written, length)

    finally:
        response.close()

    return None


def downloadSpriteModel(config):
    """ Make sure the sprite detection model is present, downloading it if needed.

        Nothing is downloaded when the local file is already correct. A partial or corrupt download never
        replaces a good file: the file is fetched to a temporary name, checked and only then moved into place.

    Arguments:
        config: [Config]

    Return:
        [bool] True if a usable model is present afterwards.
    """

    # Nothing to do when the model is already there and intact
    if spriteModelReady(config):
        return True

    if os.path.isfile(config.sprite_model_path):
        log.warning("Sprite model {:s} does not match the expected checksum, downloading it again".format(
            config.sprite_model_path))

    url = spriteModelUrl(config)
    tmp_path = config.sprite_model_path + ".download"

    log.info("Downloading the sprite model from {:s} ...".format(url))

    try:

        # The share directory normally exists, but not in every installation
        model_dir = os.path.dirname(config.sprite_model_path)
        if model_dir and not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        reason = _fetchToFile(url, tmp_path)

    except urllib_error.HTTPError as e:
        reason = "HTTP {:d}".format(e.code)

    except (urllib_error.URLError, ssl.SSLError, EOFError, IOError, OSError, ValueError) as e:
        reason = repr(e)

    if reason is not None:
        log.warning("Could not download the sprite model: {:s}".format(reason))
        _removeQuietly(tmp_path)
        return False

    # Check the download before it can replace anything
    expected = expectedModelSha256(config)
    actual = fileSha256(tmp_path)

    if (expected is not None) and (actual != expected):
        log.error("Downloaded sprite model has SHA-256 {:s}, expected {:s}; discarding it".format(
            str(actual), expected))
        _removeQuietly(tmp_path)
        return False

    # An unknown model cannot be checked, which is worth saying once
    if expected is None:
        log.warning("Sprite model {:s} is not a known model, its integrity was not checked".format(
            config.sprite_model_file))

    try:

        # A file already there failed its checksum; renaming over it would fail on Windows
        _removeQuietly(config.sprite_model_path)
        os.rename(tmp_path, config.sprite_model_path)

    except OSError as e:
        log.error("Could not move the sprite model into place: {:s}".format(repr(e)))
        _removeQuietly(tmp_path)
        return False

    log.info("Sprite model downloaded to {:s}".format(config.sprite_model_path))

    return True


def _removeQuietly(path):
    """ Delete a file if it exists, ignoring errors. """

    try:
        if os.path.isfile(path):
            os.remove(path)

    except OSError:
        pass


if __name__ == "__main__":

    import argparse

    import RMS.ConfigReader as cr
    from RMS.Logger import LoggingManager


    # Command line: an optional config file, like the other download scripts
    arg_parser = argparse.ArgumentParser(
        description="Download the sprite detection model from the GMN server.")
    arg_parser.add_argument("-c", "--config", nargs=1, metavar="CONFIG_PATH", type=str,
                            help="Path to a config file which will be used instead of the default one.")
    cml_args = arg_parser.parse_args()

    config = cr.loadConfigFromDirectory(cml_args.config, os.path.abspath("."))

    # Log to the console and the usual log directory
    log_manager = LoggingManager()
    log_manager.initLogging(config, "sprite_model_")
    log = getLogger("rmslogger")

    ok = downloadSpriteModel(config)

    print("Sprite model {:s}".format("ready at " + config.sprite_model_path if ok else "not available"))
