""" Setting up the logger. """

# RPi Meteor Station
# Copyright (C) 2017 Denis Vida
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

import os
import sys
import logging
import logging.handlers

from RMS.Misc import mkdirP, RmsDateTime

# Initialize variables for GStreamer import status
GST_IMPORTED = False
GST_IMPORT_ERROR = None

# Attempt to import GStreamer
try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    GST_IMPORTED = True

except ImportError as e:
    GST_IMPORT_ERROR = f"Could not import gi: {e}"

except ValueError as e:
    GST_IMPORT_ERROR = f"Could not import Gst: {e}"


class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():  # Avoid logging empty lines
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass  # No need to flush anything for logging


def gstDebugLogger(category, level, file, function, line, object, message, user_data):
    """
    The function maps GStreamer debug levels to Python logging levels and logs
    the message using the 'gstreamer' logger. If a GStreamer debug level
    doesn't have a direct mapping, it defaults to the Python DEBUG level.

    Args:
        category (Gst.DebugCategory): The debug category of the message.
        level (Gst.DebugLevel): The debug level of the message.
        file (str): The file where the message originated.
        function (str): The function where the message originated.
        line (int): The line number where the message originated.
        object (GObject.Object): The object that emitted the message, or None.
        message (Gst.DebugMessage): The debug message.
        user_data: User data passed to the log function.
    """

    # Get or create a logger specifically for GStreamer messages
    logger = logging.getLogger('gstreamer')

    # Map GStreamer debug levels to Python logging levels
    level_map = {
        Gst.DebugLevel.ERROR: logging.ERROR,
        Gst.DebugLevel.WARNING: logging.WARNING,
        Gst.DebugLevel.INFO: logging.INFO,
        Gst.DebugLevel.DEBUG: logging.DEBUG
    }

    # Convert GStreamer level to Python logging level, defaulting to DEBUG
    py_level = level_map.get(level, logging.DEBUG)

    # Log the message with the appropriate level
    logger.log(py_level, f"GStreamer: {category.get_name()}: {message.get()}")


def initLogging(config, log_file_prefix="", safedir=None):
    """ Initializes the logger. 
    
    Arguments:
        config: [Config] Config object.
    
    Keyword arguments:
        log_file_prefix: [str] String which will be prefixed to the log file. Empty string by default.
        safedir: [str] Path to the directory where the log files will always be able to be written to. It will
            be used if the default log directory is not writable. None by default.
    """

    # Path to the directory with log files
    log_path = os.path.join(config.data_dir, config.log_dir)

    # Make directories
    print("Creating directory: " + config.data_dir)
    data_dir_status = mkdirP(config.data_dir)
    print("   Success: {}".format(data_dir_status))
    print("Creating directory: " + log_path)
    log_path_status = mkdirP(log_path)
    print("   Sucess: {}".format(log_path_status))

    # If the log directory doesn't exist or is not writable, use the safe directory
    if safedir is not None:
        if not os.path.exists(log_path) or not os.access(log_path, os.W_OK):
            print("Log directory not writable, using safe directory: " + safedir)
            log_path = safedir

    # Generate a file name for the log file
    log_file_name = log_file_prefix + "log_" + str(config.stationID) + "_" + RmsDateTime.utcnow().strftime('%Y%m%d_%H%M%S.%f') + ".log"
        
    # Init logging
    log = logging.getLogger('logger')
    log.setLevel(logging.INFO)
    log.setLevel(logging.DEBUG)

    # Make a new log file each day
    handler = logging.handlers.TimedRotatingFileHandler(os.path.join(log_path, log_file_name), when='D', \
        interval=1, utc=True)
    handler.setLevel(logging.INFO)
    handler.setLevel(logging.DEBUG)

    # Set the log formatting
    formatter = logging.Formatter(fmt='%(asctime)s-%(levelname)s-%(module)s-line:%(lineno)d - %(message)s', 
        datefmt='%Y/%m/%d %H:%M:%S')
    handler.setFormatter(formatter)
    log.addHandler(handler)

    # Stream all logs to stdout as well
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(asctime)s-%(levelname)s-%(module)s-line:%(lineno)d - %(message)s', 
        datefmt='%Y/%m/%d %H:%M:%S')
    ch.setFormatter(formatter)
    log.addHandler(ch)

    # Optionally redirect stdout to the logger
    if config.log_stdout:
        sys.stdout = LoggerWriter(log, logging.INFO)

    # Redirect stderr to the logger
    sys.stderr = LoggerWriter(log, logging.INFO)

    # Log GStreamer import status
    if GST_IMPORTED:
        log.info("GStreamer successfully imported")
    else:
        log.warning(f"GStreamer import failed: {GST_IMPORT_ERROR}. GStreamer-specific logging is disabled.")

    # Set up GStreamer logging
    if GST_IMPORTED:
        Gst.init(None)
        Gst.debug_remove_log_function(None)
        Gst.debug_add_log_function(gstDebugLogger, None)
        Gst.debug_set_default_threshold(Gst.DebugLevel.WARNING)
        log.info("GStreamer logging successfully initialized")
