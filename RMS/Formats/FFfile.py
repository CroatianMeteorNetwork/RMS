""" Functions for reading/writing FF files with respet to the rheir format (.bin for .fits). """


from __future__ import print_function, division, absolute_import

import os
import datetime

import numpy as np

from RMS.Formats.FFbin import read as readFFbin
from RMS.Formats.FFbin import write as writeFFbin
from RMS.Formats.FFfits import read as readFFfits
from RMS.Formats.FFfits import write as writeFFfits



def read(directory, filename, fmt=None, array=False, full_filename=False):
    """ Read FF file from the specified directory and choose the proper format for reading.
    
    Arguments:
        directory: [str] Path to directory containing file
        filename: [str] Name of FF file (either with FF and extension or without)

    Keyword arguments:
        fmt: [str] Format for reading the file. It should either be 'bin' or 'fits'. If it is not given,
            the format will be guessed.
        array: [ndarray] True in order to populate structure's array element (default is False)
        full_filename: [bool] True if full file name is given explicitly, a name which may differ from the
            usual FF*.fits format. False by default.
    
    Return:
        [ff structure]

    """


    # If the reading format was not given, try to guess the proper format
    if fmt is None:

        # Try to guess the format from the file extension
        extens = os.path.splitext(filename)[1]
        if (extens.lower() == '.bin') or (extens.lower() == '.fits'):
            fmt = extens.replace('.', '')

        else:
            # If the extension is not given, try to load the file in one of the formats

            # Try reading the file as FITS
            try:
                ff = readFFfits(directory, filename, array=array)
                fmt = 'fits'

            except IOError:

                # Try reading the file as a .bin file
                try:
                    ff = readFFbin(directory, filename, array=array)
                    fmt = 'bin'

                except:
                    ff = None
                    fmt = None


    if fmt == 'bin':

        # Read the file as bin
        ff = readFFbin(directory, filename, array=array, full_filename=full_filename)


    elif fmt == 'fits':

        # Read the file as FITS
        ff = readFFfits(directory, filename, array=array, full_filename=full_filename)

    else:
        ff = None


    return ff





def write(ff, directory, filename, fmt=None):
    """ Write a FF structure to a FITS file in specified directory.
    
    Arguments:
        ff: [ff bin struct] FF file loaded in the FF structure.
        directory: [str] Path to the directory where the file will be written.
        filename: [str] Name of the file which will be written.

    Keyword arguments:
        fmt: [str] Format for writing the file. It should either be 'bin' or 'fits'. If it is not given,
            the format will be guessed. By default, the format will be 'fits'.
    
    Return:
        None

    """

    # If the reading format was not given, try to guess the proper format
    if fmt is None:

        # Try to guess the format from the file extension
        extens = os.path.splitext(filename)[1]
        if (extens.lower() == '.bin') or (extens.lower() == '.fits'):
            fmt = extens.replace('.', '')

        else:

            # Set the defualt format to FITS
            fmt = 'fits'


    if fmt == 'bin':
        writeFFbin(ff, directory, filename)

    elif fmt == 'fits':
        writeFFfits(ff, directory, filename)


def reconstructFrame(ff, frame_no, avepixel=False):
    """ Reconstruct just a given frame from an FF file.
    
    Arguments:
        ff: [ff bin struct] FF file loaded in the FF structure.
        frame_no: [int] Frame index, from 0 to 255 (inclusive).

    Keyword arguments:
        avepixel: [bool] If True, the average pixel image will be the background. False by default, which
            will use a black background.
    
    Return:
        frame: [ndarray] Reconstructed frame.
    """

    # Init the empty frame
    if avepixel:
        frame = np.copy(ff.avepixel)
    else:
        frame = np.zeros_like(ff.maxpixel)

    # Find where the max values occured for this frame
    indices = np.where(ff.maxframe == frame_no)
    frame[indices] = ff.maxpixel[indices]

    return frame


def reconstruct(ff):
    """ Reconstruct video frames from the FF file. 
    
    Arguments:
        ff: [ff bin struct] FF file loaded in the FF structure.
    
    Return:
        frames: [ndarray] an array of reconstructed video frames (255 x nrows x ncols)
    """

    # Try to read the number of frames from the FF file itself
    if ff.nframes > 0:
        nframes = ff.nframes

    else:
        nframes = 256
    
    frames = np.zeros((nframes, ff.nrows, ff.ncols), np.uint8)
    
    if ff.array is not None:
        ff.maxpixel = ff.array[0]
        ff.maxframe = ff.array[1]
    
    for i in range(nframes):
        indices = np.where(ff.maxframe == i)
        frames[i][indices] = ff.maxpixel[indices]
    
    return frames



def filenameToDatetime(file_name):
    """ Converts FF bin file name to a datetime object.

    Arguments:
        file_name: [str] Name of a FF file.

    Return:
        [datetime object] Date and time of the first frame in the FF file.

    """

    # e.g.  FF499_20170626_020520_353_0005120.bin
    # or FF_CA0001_20170626_020520_353_0005120.fits

    file_name = file_name.split('_')

    # Check the number of list elements, and the new fits format has one more underscore
    i = 0
    if len(file_name[0]) == 2:
        i = 1

    date = file_name[i + 1]
    year = int(date[:4])
    month = int(date[4:6])
    day = int(date[6:8])

    time = file_name[i + 2]
    hour = int(time[:2])
    minute = int(time[2:4])
    seconds = int(time[4:6])

    ms = int(file_name[i + 3])


    return datetime.datetime(year, month, day, hour, minute, seconds, ms*1000)



def getMiddleTimeFF(ff_name, fps, ret_milliseconds=True, ff_frames=256):
    """ Converts a CAMS format FF file name to datetime object of its recording time. 

    Arguments:
        ff_name: [str] name of the FF file
        fps: [float] Frames per second of the video compressed in the FF file.

    Keyword arguments:
        ret_milliseconds: [bool] If True, the last number returned will be in milliseconds. Otverwise, it will
            be in microseconds.
        ff_frames: [int] Number of frames that were compressed in the FF file.
        UT_corr: [float] UT correction (hours).
    
    Return:
        [tuple] (year, month, day, hour, minute, second, microsecond/millisecond)

    """

    # Extract date and time of the FF file from its name
    dt_obj = filenameToDatetime(ff_name)

    # Time in seconds from the middle of the FF file
    middle_diff = datetime.timedelta(seconds=ff_frames/2.0/fps)

    # Add the difference in time
    dt_obj = dt_obj + middle_diff

    # Unpack datetime to individual values
    year, month, day, hour, minute, second, microsecond = (dt_obj.year, dt_obj.month, dt_obj.day, dt_obj.hour, 
        dt_obj.minute, dt_obj.second, dt_obj.microsecond)

    if ret_milliseconds:
        return (year, month, day, hour, minute, second, microsecond/1000)
    else:
        return (year, month, day, hour, minute, second, microsecond)



def validFFName(ff_file, fmt=None):
    """ Checks if the given file is an FF file. 
    
    Arguments:
        ff_file: [str] Name of the FF file

    Keyword arguments:
        fmt: [str] Format of the FF file. If not given, it will tried to be determined from the file name.
    """

    if fmt is None:
        if '.bin' in ff_file:
            fmt = 'bin'

        else:
            fmt = 'fits'

    # Make sure the file starts with FF
    if ('FF' in ff_file[:2]):

        candidate = False

        # Check that the format coresponds to the given file
        if (fmt == 'bin') and ('.bin' in ff_file):
            candidate = True

        elif (fmt == 'fits') and ('.fits' in ff_file):
            candidate = True


        # Make sure it's not an image file
        if candidate and (('.jpg' in ff_file) or ('.bmp' in ff_file) or ('.png' in ff_file)):
            return False
            
        else:
            return True

        
    return False




if __name__ == '__main__':

    ### TEST ###

    # Load a .bin file
    dir_path = 'D:\\Dropbox\\RPi Meteor Station\\samples\\sample_bins'

    file_name = 'FF453_20150620_201239_920_0058880.bin'

    # Read the FF file
    ff = read(dir_path, file_name)

    print(ff)

    file_name_fits = file_name.replace('.bin', '.fits')

    # Write the read FF file as FITS
    write(ff, dir_path, file_name_fits, fmt='fits')

    # Read the FFfits
    ff = read(dir_path, file_name_fits)


    import matplotlib.pyplot as plt

    plt.imshow(ff.maxpixel, cmap='gray', vmin=0, vmax=255)
    plt.show()

    print(ff)
