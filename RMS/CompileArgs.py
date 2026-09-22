""" Platform-specific compile arguments for the Cython extensions, shared by setup.py,
the .pyxbld pyximport fallback recipes, and ConfigReader.

This module must only depend on the standard library: setup.py imports it at build
time, before any of the RMS dependencies (numpy, matplotlib, ...) are installed.
"""

# The MIT License

# Copyright (c) 2016 Denis Vida

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from __future__ import print_function, division, absolute_import

import os
import sys

try:
    from configparser import RawConfigParser
except ImportError:
    from ConfigParser import RawConfigParser  # Python 2


# Defaults, used when the .config file is missing or does not set the [Build] options.
# These mirror the values shipped in the repository .config. Windows gets no extra
# arguments: distutils already passes /O2 /W3 to MSVC, which is the right baseline.
DEFAULT_WIN_PC_ARGS = []
DEFAULT_LINUX_PC_ARGS = ["-O3"]
DEFAULT_RPI_ARGS = ["-O3", "-mfpu=neon", "-funsafe-loop-optimizations",
                    "-ftree-loop-if-convert-stores"]


def getCompileArgs(win_args=None, rpi_args=None, linux_pc_args=None):
    """ Choose the compile arguments for the current platform, falling back to the
        defaults for any that are not given.

    Keyword arguments:
        win_args: [list] Arguments for Windows. None by default.
        rpi_args: [list] Arguments for 32-bit ARM Linux (Raspberry Pi). None by default.
        linux_pc_args: [list] Arguments for Linux/macOS PCs. None by default.

    Return:
        [list] Compile arguments for the current platform.
    """

    # Windows - use the given arguments or the defaults
    if sys.platform.startswith('win'):
        args = win_args if win_args is not None else DEFAULT_WIN_PC_ARGS

        # MSVC reads -Wall as /Wall, which enables every warning there is (including
        # ones from system headers and Cython-generated code), unlike gcc's -Wall.
        # Old .config files ship it, so strip the gcc spelling; distutils already
        # passes /W3. An explicit /Wall in the config is left alone.
        return [arg for arg in args if arg != "-Wall"]

    # The RPi flags (e.g. -mfpu=neon) only apply to 32-bit ARM Linux with gcc; macOS
    # on Apple Silicon also reports an 'arm' machine but clang rejects those flags
    if ('arm' in os.uname()[4]) and (sys.platform != 'darwin'):
        return rpi_args if rpi_args is not None else DEFAULT_RPI_ARGS

    # Linux/macOS PCs
    return linux_pc_args if linux_pc_args is not None else DEFAULT_LINUX_PC_ARGS


def getExtraCompileArgs(config_path=None):
    """ Read the [Build] section of the .config file and return the compile arguments
        for the current platform. Falls back to the defaults if the file or the options
        are missing, so it is always safe to call.

    Keyword arguments:
        config_path: [str] Path to the .config file. None by default, in which case the
            .config in the repository root (next to setup.py) is used.

    Return:
        [list] Compile arguments for the current platform.
    """

    # Default to the .config in the repository root
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   '.config')

    # None means "use the built-in default" for that platform
    win_args, rpi_args, linux_pc_args = None, None, None

    if os.path.isfile(config_path):

        # Allow inline ';' comments and duplicate options, as the RMS ConfigReader does
        try:
            parser = RawConfigParser(inline_comment_prefixes=(';',), strict=False)
        except TypeError:
            parser = RawConfigParser()  # Python 2

        # A broken .config must never stop the build, so fall back to the defaults on any error
        try:
            parser.read(config_path)

            # Read the per-platform arguments from the [Build] section, if present
            section = "Build"
            if parser.has_section(section):

                if parser.has_option(section, "win_pc_weave"):
                    win_args = parser.get(section, "win_pc_weave").split()

                if parser.has_option(section, "rpi_weave"):
                    rpi_args = parser.get(section, "rpi_weave").split()

                if parser.has_option(section, "linux_pc_weave"):
                    linux_pc_args = parser.get(section, "linux_pc_weave").split()

        except Exception as e:
            print("Warning: could not read [Build] options from {:s}: {:s}".format(
                config_path, str(e)))

    return getCompileArgs(win_args, rpi_args, linux_pc_args)
