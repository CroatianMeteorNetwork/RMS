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
import datetime

from RMS.Misc import mkdirP


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
    mkdirP(config.data_dir)
    mkdirP(log_path)

    # If the log directory doesn't exist or is not writable, use the safe directory
    if safedir is not None:
        if not os.path.exists(log_path) or not os.access(log_path, os.W_OK):
            log_path = safedir

    # Generate a file name for the log file
    log_file_name = log_file_prefix + "log_" + str(config.stationID) + "_" + datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S.%f') + ".log"
        
    # Init logging
    log = logging.getLogger('logger')
    log.setLevel(logging.INFO)
    log.setLevel(logging.DEBUG)

    # Make a new log file each day
    handler = logging.handlers.TimedRotatingFileHandler(os.path.join(log_path, log_file_name), when='D', \
        interval=1) 
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



