
from __future__ import print_function, division, absolute_import

import platform
import os
import sys
import traceback
import shutil
import errno
import subprocess
import random
import re
import string
import inspect
import datetime
import tarfile

# tkinter import that works on both Python 2 and 3
if sys.version_info[0] < 3:
    import Tkinter as tkinter
    import tkFileDialog as filedialog
    import pkgutil
else:
    import tkinter
    from tkinter import filedialog
    import importlib.util


import numpy as np

from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import FixedLocator

from RMS.Logger import getLogger

# Map FileNotFoundError to IOError in Python 2 as it does not exist
if sys.version_info[0] < 3:
    FileNotFoundError = IOError

# Get the logger from the main module
log = getLogger("logger")


def mkdirP(path):
    """ Makes a directory and handles all errors.
    """

    # Try to make a directory
    try:
        os.makedirs(path)
        return True

    # If it already exist, do nothing
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            return True
        else:
            log.error("Error creating directory: " + str(exc))
            return False

    # Raise all other errors
    except Exception as e:
        log.error("Error creating directory: " + str(e))
        return False

    return False


def walkDirsToDepth(dir_path, depth=-1):
    """ Mimic os.walk, but define the maximum depth. 
    
    Arguments:
        dir_path: [str] Path to the directory.

    Keyword arguments:
        depth: [int] Maximum depth. Use -1 for no limit, in which case the function behaves the same as
            os.walk.

    Return:
        file_list: [list] List where the elements are:
            - dir_path - path to the directory
            - dir_list - list of directories in the path
            - file_list - list of files in the path
    """
    
    final_list = []
    dir_list = []
    file_list = []

    # Find all files and directories in the given path and sort them accordingly
    for entry in sorted(os.listdir(dir_path)):

        entry_path = os.path.join(dir_path, entry)

        if os.path.isdir(entry_path):
            dir_list.append(entry)

        else:
            file_list.append(entry)


    # Mimic the output of os.walk
    final_list.append([dir_path, dir_list, file_list])


    # Decrement depth for the next recursion
    depth -= 1

    # Stop the recursion if the final depth has been reached
    if depth != 0:

        # Do this recursively for all directories up to a certain depth
        for dir_name in dir_list:

            final_list_rec = walkDirsToDepth(os.path.join(dir_path, dir_name), depth=depth)

            # Add the list to the total list
            final_list += final_list_rec


    return final_list


def archiveDir(source_dir, file_list, dest_dir, compress_file, delete_dest_dir=False, extra_files=None):
    """ Move the given file list from the source directory to the destination directory, compress the 
        destination directory and save it as a .bz2 file. BZ2 compression is used as ZIP files have a limit
        of 2GB in size.

    Arguments:
        source_dir: [str] Path to the directory from which the files will be taken and archived.
        file_list: [list] A list of files from the source_dir which will be archived.
        dest_dir: [str] Path to the archive directory which will be compressed.
        compress_file: [str] Name of the compressed file which will be created.

    Keyword arguments:
        delete_dest_dir: [bool] Delete the destination directory after compression. False by default.
        extra_files: [list] A list of extra files (with fill paths) which will be be saved to the night 
            archive.

    Return:
        archive_name: [str] Full name of the archive.

    """

    # Make the archive directory
    mkdirP(dest_dir)

    # Copy the files from the source to the archive directory
    for file_name in file_list:

        if hasattr(shutil, "SameFileError"):
            try:
                if os.path.isfile(os.path.join(source_dir, file_name)):
                    shutil.copy2(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))
            except shutil.SameFileError:
                pass
            except FileNotFoundError:
                log.warning('file {} not found '.format(os.path.join(source_dir, file_name)))
        else:
            try:
                shutil.copy2(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))
            except Exception as e:
                log.warning(e)


    # Copy the additional files to the archive directory
    if extra_files is not None:
        for file_path in extra_files:

            if hasattr(shutil, "SameFileError"):
                try:
                    if os.path.isfile(file_path):
                        shutil.copy2(file_path, os.path.join(dest_dir, os.path.basename(file_path)))
                except shutil.SameFileError:
                    pass
                except FileNotFoundError:
                    log.warning('file {} not found'.format(file_path))
            else:
                try:
                    shutil.copy2(file_path, os.path.join(dest_dir, os.path.basename(file_path)))
                except Exception as e:
                    log.warning(e)


    # Compress the archive directory
    archive_name = shutil.make_archive(os.path.join(dest_dir, compress_file), 'bztar', dest_dir, logger=log)

    # Delete the archive directory after compression
    if delete_dest_dir:
        shutil.rmtree(dest_dir)


    return archive_name




def openFileDialog(dir_path, initialfile, title, mpl, filetypes=()):
    """ Open the file dialog and close it properly, depending on the backend used. 
    
    Arguments:
        dir_path: [str] Initial path of the directory.
        initialfile: [str] Initial file to load.
        title: [str] Title of the file dialog window.
        mpl: [matplotlib instance] Instace of matplotlib import which is used to determine the used backend.
        filetypes: [list of tuples] A tuple with file type pairs to filter (label, pattern)

    Return:
        file_name: [str] Path to the chosen file.
    """

    root = tkinter.Tk()
    root.withdraw()
    root.update()

    # Open the file dialog
    file_name = filedialog.askopenfilename(initialdir=dir_path,
        initialfile=initialfile, title=title, filetypes=filetypes)
    root.update()

    if (mpl.get_backend() != 'TkAgg') and (mpl.get_backend() != 'WXAgg'):
        root.quit()
    else:
        root.destroy()


    return file_name


class SegmentedScale(mscale.ScaleBase):
    """ Segmented scale used to defining flux and ZHR on the same graph. """
    name = 'segmented'

    def __init__(self, axis, **kwargs):

        # Handle different matplotlib versions
        try:
            mscale.ScaleBase.__init__(self, axis)
        except TypeError:
            mscale.ScaleBase.__init__(axis)

        self.points = kwargs.get('points', [0, 1])
        self.lb = self.points[0]
        self.ub = self.points[-1]

    def get_transform(self):
        return self.SegTrans(self.lb, self.ub, self.points)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(FixedLocator(self.points))

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return max(vmin, self.lb), min(vmax, self.ub)

    class SegTrans(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, lb, ub, points):
            mtransforms.Transform.__init__(self)
            self.lb = lb
            self.ub = ub
            self.points = points

        def transform_non_affine(self, a):
            masked = a # ma.masked_where((a < self.lb) | (a > self.ub), a)
            return np.interp(masked, self.points, np.arange(len(self.points)))

        def inverted(self):
            return SegmentedScale.InvertedSegTrans(self.lb, self.ub, self.points)

    class InvertedSegTrans(SegTrans):

        def transform_non_affine(self, a):
            return np.interp(a, np.arange(len(self.points)), self.points)
        
        def inverted(self):
            return SegmentedScale.SegTrans(self.lb, self.ub, self.points)


def openFolderDialog(initialdir, title, mpl):
    """ Open the file dialog and close it properly, depending on the backend used.

    Arguments:
        initialdir: [str] Initial path of the directory.
        title: [str] Title of the file dialog window.
        mpl: [matplotlib instance] Instace of matplotlib import which is used to determine the used backend.

    Return:
        file_name: [str] Path to the chosen file.
    """

    root = tkinter.Tk()
    root.withdraw()
    root.update()

    # Open the file dialog
    file_name = filedialog.askdirectory(initialdir=initialdir, title=title, mustexist=True)

    root.update()

    if (mpl.get_backend() != 'TkAgg') and (mpl.get_backend() != 'WXAgg'):
        root.quit()
    else:
        root.destroy()


    return file_name



def ping(host, count=1):
    """ Ping the host and return True if reachable. 
        Remember that a host may not respond to a ping (ICMP) request even if the host name is valid.

    Source: https://stackoverflow.com/a/32684938/6002120

    Arguments:
        host: [str] Host name or IP address.

    Return:
        [bool] True if host (str) responds to a ping request.
    """

    # Ping command count option as function of OS
    param = '-n' if platform.system().lower()=='windows' else '-c'

    # Building the command. Ex: "ping -c 1 google.com"
    command = ['ping', param, str(count), host]

    # Pinging
    return subprocess.call(command) == 0





def randomCharacters(length):
    """ Returns a random string of alphanumeric characters. """

    letters = string.ascii_lowercase + string.digits

    return ''.join(random.choice(letters) for i in range(length))




def checkListEquality(t1, t2):
    """ Given two lists or tuples, compare every element and make sure they are the same. This function takes care
        of comparing two objects attribute-wise.
    
    Arguments:
        t1: [tuple] First list or tuple.
        t2: [tuple] Second list or tuple.

    Return:
        [bool] True if equal, False otherwise.
    """

    # Check if they are tuples or lists
    if not (type(t1) is list) and not (type(t1) is tuple):
        return False

    if not (type(t2) is list) and not (type(t2) is tuple):
        return False


    # Check if they have the same length
    if len(t1) != len(t2):
        return False


    # Check them element-wise
    for e1, e2 in zip(t1, t2):

        # If they are tuples or lists, recursively check equality
        if (type(e1) is list) or (type(e1) is tuple):
            if not checkListEquality(e1, e2):
                return False


        # If the elements are instances of objects, compare their attributes
        elif hasattr(e1, '__dict__') and hasattr(e2, '__dict__'):

            # Check if they are functions or classes, compare them directly
            if (inspect.isroutine(e1) and inspect.isroutine(e2)) or \
                (inspect.isclass(e1) and inspect.isclass(e2)):

                if e1 == e2: 
                    return True
                else: 
                    return False


            # Check the dictionaries
            else:

                # If they are instances of objects, compare their attributes
                for key1 in e1.__dict__:

                    # If the other dictionary doesn't have the same keys, it's obviously not the same dict
                    if key1 not in e2.__dict__:
                        return False

                    val1 = e1.__dict__[key1]
                    val2 = e2.__dict__[key1]

                    # Check if the value is a numpy array, and check if they are the same
                    if isinstance(val1, np.ndarray):
                        if not np.array_equal(val1, val2):
                            return False

                    else:
                        # Check if the values are the same
                        if val1 != val2:
                            return False

        else:

            # If the elements are something else, compare them directly
            if e1 != e2:
                return False


    # If all checks have passes, return True
    return True



def isListKeyInDict(lst, dct):
    """ Given the list or a tuple, check if it is a key in the dictionary. Difference instances of same 
        classes are handled as well.

    Arguments:
        lst: [list or tuple] Input list.
        dct: [dict] Dictionary to be checked.

    Return:
        (status, key): [bool, obj] True if list/tuple in dictionary, False otherwise. The key for value
            retrieval will be returned as well.
    """

    # Go though the sict
    for key in dct:

        # If the list was found, return True
        if checkListEquality(lst, key):
            return True, key


    # If list was not found in the dictionary, return False
    return False, None



def listToTupleRecursive(lst):
    """ Given the list, convert it to tuples, and convert all sublists to tuples. """

    out = []

    for elem in lst:

        if (type(elem) is list) or (type(elem) is tuple):
            elem = listToTupleRecursive(elem)

        out.append(elem)

    return tuple(out)



def decimalDegreesToSexHours(val):
    """ Convert a value in decimal degrees DDD.ddd into the sexadecimal format in hours, HH, MM, SS.ss.
    
    Arguments:
        val: [float] Value in decimal degrees

    Return:
        (hh, mm, ss):
            - hh [int] Hours
            - mm [int] Minutes
            - ss [float] Seconds
    """

    val = val/15.0

    sign = np.sign(val)
    val = abs(val)
    hh = int(val)
    mm = int((val - hh)*60)
    ss = ((val - hh)*60 - mm)*60

    return sign, hh, mm, ss



def formatScientific(val, dec_places):
    """ Format a given number in the scientific notation such that it looks like e.g. 2.1 x 10^3. The
        string is returned in the LaTex format that can be given to matplotlib.

    Source: https://stackoverflow.com/questions/31453422/displaying-numbers-with-x-instead-of-e-scientific-notation-in-matplotlib/31453961

    Arguments:
        val: [float] Input value.
        dec_places: [int] Number of decimal places.

    Return:
        [str] Formatted Latex string.

    """
    
    s = '{val:0.{dec_places:d}e}'.format(val=val, dec_places=dec_places)

    # Handle NaN values
    if 'nan' in s:
        return 'NaN'

    m, e = s.split('e')

    return r'{m:s}\times 10^{{{e:d}}}'.format(m=m, e=int(e))



def roundToSignificantDigits(x, n=2):
    """ Round the number to N significant digits. """

    def _decimalPlace(x, n):
        return -int(np.floor(np.log10(x))) + (n - 1)

    def _round(x, n, dec_place=None):

        # Don't try to round zeros
        if x == 0:
            return x

        # Compute the decimal place if not given
        if dec_place is None:
            dec_place = _decimalPlace(x, n)


        return np.round(x, dec_place)


    ### Compute the decimal place to round to ###

    # Run on only one number
    if np.isscalar(x):
        out = _round(x, n)

    else:
        out = []

        # If a list is given, determine the smallest decimal place for all numbers
        for num in x:
            dec_places = [_decimalPlace(num, n) for num in x if num != 0]

        # Handle the cases when all numbers are 0
        if len(dec_places):
            # Compute the smallest decimal place
            common_dec_place = np.max(dec_places)
        else:
            common_dec_place = 0

        # Compute the rounded numbers
        for num in x:
            out.append(_round(num, n, dec_place=common_dec_place))

        out = np.array(out)

    ### ###

    return out


def isRaspberryPi():
    """ Check if the code is running on a Raspberry Pi. 
    
    Return:
        [bool] True if the code is running on a Raspberry Pi, False otherwise.
    """

    try:
        # Open a file with the RPi model name
        with open('/sys/firmware/devicetree/base/model', 'r') as m:

            if 'raspberry pi' in m.read().lower(): 
                return True               

    except FileNotFoundError:
        pass

    return False

def sanitise(unsanitised, lower = False, space_substitution = "", log_changes = False):
    """ Strictly sanitise an input string

        Return:
            [string] Sanitised string
        """

    permitted = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890_-"
    sanitised = ""
    for c in unsanitised:
        if c == " ":
            sanitised += space_substitution
        else:
            if c in permitted:
                sanitised += c
    sanitised = sanitised.lower() if lower else sanitised
    if unsanitised != sanitised and log_changes:
        log.info("String {} was sanitised to {}".format(unsanitised, sanitised))

    return sanitised



class RmsDateTime:
    """ Class to hold utcnow() wrapper function definition.
        Select the best approach to retrieve current UTC time according to Python version.
    """
    if sys.version_info[0] < 3:
        @staticmethod
        def utcnow():
            # Python 2: Use the existing utcnow, which is not timezone-aware.
            return datetime.datetime.utcnow()
    else:
        @staticmethod
        def utcnow():
            # Python 3: Get timezone-aware UTC time and then make it naive.
            return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


class UTCFromTimestamp:
    """Cross-version helper to convert Unix timestamps to naive UTC datetime objects.

    - Python 2.7-3.11: uses datetime.utcfromtimestamp()
    - Python 3.12+: uses datetime.fromtimestamp(..., tz=timezone.utc).replace(tzinfo=None)
    """

    @staticmethod
    def utcfromtimestamp(timestamp):
        if sys.version_info >= (3, 12):
            # Use aware datetime then strip tzinfo to make it naive
            return datetime.datetime.fromtimestamp(
                timestamp, tz=UTCFromTimestamp._get_utc_timezone()
            ).replace(tzinfo=None)
        else:
            return datetime.datetime.utcfromtimestamp(timestamp)

    @staticmethod
    def _get_utc_timezone():
        """Safely provide UTC tzinfo across Python versions."""
        try:
            # Python 3.2+
            from datetime import timezone
            return timezone.utc
        except ImportError:
            # Python 2: no timezone support
            raise NotImplementedError(
                "timezone-aware fromtimestamp() is not supported in Python < 3.2. "
                "Use Python >= 3.12 or fallback to utcfromtimestamp()."
            )

def niceFormat(string, delim=":", extra_space=5):

    """
    Takes a string of lines such as

    key : value
    key2 : value1
    alongerkey : value2

    and formats the string so that the delimiters are all in the same column

    key        : value
    key2       : value1
    alongerkey : value2

    Args:
        string: takes a string, possibly including \n, each line of format key delimiter value
        delim: delimited between key and value default :
        extra_space: number of extra spaces between the key and the delimiter

    Returns:
        a string
    """

    max_to_delim = 0
    for line in string.splitlines():
        max_to_delim = line.find(delim) if line.find(delim) > max_to_delim else max_to_delim

    formatted_string = ""
    for line in string.splitlines():
        field_name = line.split(delim)[0].strip()
        value = line[len(field_name) + 1:]
        padding = " " * (extra_space + max_to_delim - len(field_name))
        formatted_string += "{:s}{:s}{:s} {:s}\n".format(field_name, padding, delim, value)

    return formatted_string



def getRMSStyleFileName(night_data_dir, name_suffix):

    """ Given path to a night_data_dir and a name suffix generate an RMS style file name

        e.g

        night_data_dir  :    /home/david/RMS_data/ArchivedFiles/AU0006_20240811_101142_903530
        name_suffix     :    observation_summary.txt

        yields          :   /home/david/RMS_data/ArchivedFiles/AU0006_20240811_101142_903530/
                                                    AU0006_20240811_101142_903530_observation_summary.txt


        arguments:
            night_data_dir: path to the night data directory
            suffix: suffix of the file to be created

        returns:
            full path and filename of the file to be created

    """

    return os.path.join(night_data_dir, os.path.split(night_data_dir.strip(os.sep))[1] + "_" + name_suffix)


def maxDistBetweenPoints(points_x, points_y):
    """
    Routine to calculate the maximum cartesian distance between any two points in a
    list of points

    Args:
        points_x: list of points
        points_y: list of points

    Returns:
        maximum cartesian distance between points
    """

    max_separation = 0
    for ref_x, ref_y in zip(points_x, points_y):
        min_separation = np.inf
        for x, y in zip(points_x, points_y):
            pixel_separation = ((ref_x - x) ** 2 + (ref_y - y) ** 2) ** 0.5
            if pixel_separation < min_separation and pixel_separation != 0:
                min_separation = pixel_separation
        if min_separation > max_separation:
            max_separation = min_separation

    return max_separation


def getRmsRootDir():
    """
        Return the path to the RMS root directory without importing the whole
        codebase
    """
    if sys.version_info[0] == 3:
        # Python 3.x: Use importlib to find the RMS module
        rms_spec = importlib.util.find_spec('RMS')
        if rms_spec is None or rms_spec.origin is None:
            raise ImportError("RMS module not found.")

        # Get the absolute path to the RMS root directory
        return os.path.abspath(os.path.dirname(os.path.dirname(rms_spec.origin)))
    else:
        # Python 2.7: Use pkgutil (deprecated) to locate the RMS module
        loader = pkgutil.get_loader('RMS')
        if loader is None:
            raise ImportError("RMS module not found.")

        # Get the filename associated with the loader
        rms_file = loader.get_filename()

        # Get the absolute path to the RMS root directory
        return os.path.abspath(os.path.dirname(os.path.dirname(rms_file)))


def obfuscatePassword(url):
    """
    Obfuscate the password in a given URL string if it's not empty or the default.

    This function attempts to find the password in RTSP URLs or in URL parameters. 
    If a non-empty and non-default password is found, it is replaced with '****'.

    Arguments:
        url: [str] The URL string that may contain a password.

    Returns:
        str: The URL with the password obfuscated, if present and not empty or default.
             If the password is empty or 'password', the original URL is returned.
             If an error occurs during obfuscation, returns "[URL_REDACTED_DUE_TO_ERROR]".
    """
    try:
        pattern = r'(rtsp://.*?:.*?@|user=.*?&password=)(.*?)(&|$)'
        match = re.search(pattern, url)
        if match:
            password = match.group(2)
            if password and password != 'password':
                return re.sub(pattern, r'\1****\3', url)
        return url
    except Exception as e:
        log.error("Error in obfuscate_password: %s", str(e))
        return "[URL_REDACTED_DUE_TO_ERROR]"


def _portableCommonpath(paths):
    """Return the longest common sub-path shared by all given paths.

    Arguments:
        paths: [list[str]] Sequence (list, tuple, etc.) of file-system
            paths to compare.

    Return:
        common_path: [str] The directory prefix common to every element in
            *paths*, or an empty string if none exists.
    """
    try:
        return os.path.commonpath(paths)          # Py 3.5+
    
    except AttributeError:
        if not paths:
            return ''
        
        split_paths = [os.path.normpath(p).split(os.sep) for p in paths]
        prefix_parts = os.path.commonprefix(split_paths)

        if not prefix_parts:
            return os.path.dirname(paths[0])      # diff drives (Windows)
        
        prefix = os.sep.join(prefix_parts)

        if not os.path.isdir(prefix):
            prefix = os.path.dirname(prefix)

        return prefix or os.path.dirname(paths[0])


def tarWithProgress(source_dir, tar_path, compression='bz2', remove_source=False, file_list=None):
    """Create a tar archive with progress feedback, verify it, and (optionally) delete the sources.

    Arguments:
        source_dir: [str | None] Directory whose entire contents will be
            archived. Ignored when *file_list* is supplied.
        tar_path: [str] Full path (including extension) where the archive
            will be written.

    Keyword arguments:
        compression: [str] Compression algorithm: 'bz2' or 'gz'.
            'bz2' by default.
        remove_source: [bool] If *True* and *file_list* is *None*, delete
            *source_dir* after the archive verifies correctly. False by
            default.
        file_list: [list[str] | None] Explicit list of file paths to
            archive. When given, the directory walk is skipped and each
            file is stored relative to their deepest common parent
            directory. None by default.

    Return:
        success: [bool] True if the archive was created **and** verified
            successfully, False otherwise.
    """
    try:
        # 1. Build list of files ------------------------------------------------
        if file_list is not None:
            files_to_archive = list(file_list)
            if not files_to_archive:
                log.error("No files given in file_list - nothing to archive")
                return False
            base_dir = _portableCommonpath(files_to_archive)
        else:
            files_to_archive = [os.path.join(r, f)
                                for r, _, fs in os.walk(source_dir)
                                for f in fs]
            base_dir = source_dir

        total_files = len(files_to_archive)
        if not total_files:
            log.info("Nothing to archive")
            return False

        log.info("Found {:d} files to archive".format(total_files))
                
        # 2. Create tarball -----------------------------------------------------
        mode = 'w:bz2' if compression == 'bz2' else 'w:gz'
        with tarfile.open(tar_path, mode) as tar:
            processed = 0
            last_pct = 0
            for fpath in files_to_archive:
                rel = os.path.relpath(fpath, base_dir)
                
                # Check if the file is outside the base directory
                if rel.startswith(os.pardir):
                    raise ValueError("{} is outside {}".format(fpath, base_dir))
                
                arcname = os.path.join(os.path.basename(base_dir), rel)
                tar.add(fpath, arcname=arcname)

                processed += 1
                pct = int(processed * 100.0/total_files)
                if pct >= last_pct + 5:
                    last_pct = (pct//5)*5
                    print("Archiving progress: {}% ({}/{})".format(
                          last_pct, processed, total_files))
        
        # 3. Verify -------------------------------------------------------------
        log.info("Verifying archive integrity...")
        read_mode = 'r:bz2' if compression == 'bz2' else 'r:gz'

        if not (os.path.exists(tar_path) and os.path.getsize(tar_path) > 0):
            log.error("Archive verification failed: file is empty or missing")
            return False

        with tarfile.open(tar_path, read_mode) as tst:
            archive_files = len(tst.getnames())
            if archive_files < total_files:
                log.error("Archive verification failed: wanted >={} files, found {}".format(
                          total_files, archive_files))
                return False
            log.info("Archive verified successfully: contains {} files".format(
                     archive_files))
            print("Archive verified successfully: contains {} files".format(
                  archive_files))

        # 4. Optional cleanup ---------------------------------------------------
        if remove_source and file_list is None and source_dir:
            log.info("Removing source directory {} ...".format(source_dir))
            shutil.rmtree(source_dir)
            log.info("Source directory removed")

        return True

    except Exception as e:
        log.error("Error creating archive: {}".format(e))
        log.error("".join(traceback.format_exception(*sys.exc_info())))
        return False