
from __future__ import print_function, division, absolute_import

import platform
import os
import shutil
import errno
import logging
import subprocess
import random
import string
import inspect
import sys


# tkinter import that works on both Python 2 and 3
if sys.version_info[0] >= 3:
    import tkinter
    from tkinter import filedialog
else:
    import Tkinter as tkinter
    import tkFileDialog as filedialog


import numpy as np


# Get the logger from the main module
log = logging.getLogger("logger")


def mkdirP(path):
    """ Makes a directory and handles all errors.
    """

    # Try to make a directory
    try:
        os.makedirs(path)

    # If it already exist, do nothing
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            pass

    # Raise all other errors
    except:
        raise 



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
                shutil.copy2(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))
            except shutil.SameFileError:
                pass
        else:
            shutil.copy2(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))


    # Copy the additional files to the archive directory
    if extra_files is not None:
        for file_name in extra_files:

            if hasattr(shutil, "SameFileError"):
                try:
                    shutil.copy2(file_name, os.path.join(dest_dir, os.path.basename(file_name)))
                except shutil.SameFileError:
                    pass
            else:
                shutil.copy2(file_name, os.path.join(dest_dir, os.path.basename(file_name)))


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



def ping(host):
    """ Ping the host and return True if reachable. 
        Remember that a host may not respond to a ping (ICMP) request even if the host name is valid.

    Source: https://stackoverflow.com/a/32684938/6002120

    Arguments:
        host: [str] Host name or IP address.

    Return:
        [bool] True if host (str) responds to a ping request.
    """

    # Ping command count option as function of OS
    if platform.system().lower()=='windows':
        command = ['ping', '-n', '1', host]
    else:
        command = ['ping', '-c', '1', host]

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
    if not (type(t1) is list) and not(type(t1) is tuple):
        return False

    if not (type(t2) is list) and not(type(t2) is tuple):
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

            # If the elements are someting else, compare them directly
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
