

from __future__ import print_function, division, absolute_import

import os

import numpy as np
from astropy.io import fits

from RMS.Formats.FFStruct import FFStruct
import datetime



def filenameToDatetimeStr(file_name):
    """ Converts FS and FF bin file name to a datetime object.

    Arguments:
        file_name: [str] Name of a FF or FS file.

    Return:
        [datetime object] Date and time of the first frame in the FS or FF file.

    """

    # e.g.  FF499_20170626_020520_353_0005120.bin
    # or FF_CA0001_20170626_020520_353_0005120.fits
    # or FS_US9999_20240318_011731_867370_1054720_fieldsum.bin

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

    # Determine if the time fraction is in milliseconds or microseconds
    time_fraction_str = file_name[i + 3]
    if len(time_fraction_str) == 3:  # Milliseconds, need to convert to microseconds
        microseconds = int(time_fraction_str) * 1000
    else:  # Assuming microseconds directly
        microseconds = int(time_fraction_str)


    return datetime.datetime(year, month, day, hour, minute, seconds, microseconds).strftime("%Y-%m-%d %H:%M:%S.%f")



def read(directory, filename, array=False, full_filename=False):
    """ Read a FF structure from a FITS file. 
    
    Arguments:
        directory: [str] Path to directory containing file
        filename: [str] Name of FF*.fits file (either with FF and extension or without)

    Keyword arguments:
        array: [ndarray] True in order to populate structure's array element (default is False)
        full_filename: [bool] True if full file name is given explicitly, a name which may differ from the
            usual FF*.fits format. False by default.
    
    Return:
        [ff structure]

    """

    # Make sure the file starts with "FF_"
    if (filename.startswith('FF') and ('.fits' in filename)) or full_filename:
        fid = open(os.path.join(directory, filename), "rb")
    else:
        fid = open(os.path.join(directory, "FF_" + filename + ".fits"), "rb")

    # Init an empty FF structure
    ff = FFStruct()

    # Read in the FITS
    hdulist = fits.open(fid)

    # Read the header
    head = hdulist[0].header

    # Read in the data from the header
    ff.nrows = head['NROWS']
    ff.ncols = head['NCOLS']
    ff.nbits = head['NBITS']
    ff.nframes = head['NFRAMES']
    ff.first = head['FIRST']
    ff.camno = head['CAMNO']
    ff.fps = head['FPS']

    # Check for the EXPSTART field and read datetime from filename it if it doesn't exist
    if 'EXPSTART' in head:
        ff.starttime = head['EXPSTART']
    else:
        ff.starttime = filenameToDatetimeStr(filename)

    # Read in the image data
    ff.maxpixel = hdulist[1].data
    ff.maxframe = hdulist[2].data
    ff.avepixel = hdulist[3].data
    ff.stdpixel = hdulist[4].data

    if array:
        ff.array = np.dstack([ff.maxpixel, ff.maxframe, ff.avepixel, ff.stdpixel])

        ff.array = np.swapaxes(ff.array, 0, 1)
        ff.array = np.swapaxes(ff.array, 0, 2)

    # CLose the FITS file
    hdulist.close()

    return ff



def write(ff, directory, filename):
    """ Write a FF structure to a FITS file in specified directory.
    
    Arguments:
        ff: [ff bin struct] FF bin file loaded in the FF structure
        directory: [str] path to the directory where the file will be written
        filename: [str] name of the file which will be written
    
    Return:
        None

    """

    # Make sure the file starts with "FF"
    if filename[:3] == "FF_":
        file_path = os.path.join(directory, filename)

    else:
        file_path = os.path.join(directory, "FF_" + filename + ".fits")

    # Create a new FITS file
    
    # Create the header
    head = fits.Header()
    head['NROWS'] = ff.nrows
    head['NCOLS'] = ff.ncols
    head['NBITS'] = ff.nbits
    head['NFRAMES'] = ff.nframes
    head['FIRST'] = ff.first
    head['CAMNO'] = ff.camno
    head['FPS'] = ff.fps
    head['EXPSTART'] = ff.starttime

    # Deconstruct the 3D array into individual images
    if ff.array is not None:
        ff.maxpixel, ff.maxframe, ff.avepixel, ff.stdpixel = np.split(ff.array, 4, axis=0)
        ff.maxpixel = ff.maxpixel[0]
        ff.maxframe = ff.maxframe[0]
        ff.avepixel = ff.avepixel[0]
        ff.stdpixel = ff.stdpixel[0]

    # Add the maxpixle to the list
    maxpixel_hdu = fits.ImageHDU(ff.maxpixel, name='MAXPIXEL')
    maxframe_hdu = fits.ImageHDU(ff.maxframe, name='MAXFRAME')
    avepixel_hdu = fits.ImageHDU(ff.avepixel, name='AVEPIXEL')
    stdpixel_hdu = fits.ImageHDU(ff.stdpixel, name='STDPIXEL')
    
    # Create the primary part
    prim = fits.PrimaryHDU(header=head)
    
    # Combine everything into into FITS
    hdulist = fits.HDUList([prim, maxpixel_hdu, maxframe_hdu, avepixel_hdu, stdpixel_hdu])

    # Save the FITS
    hdulist.writeto(file_path, overwrite=True)






if __name__ == "__main__":

    dir_path = '.'
    file_name = 'FF_test.fits'

    wid = 720
    ht = 576


    ff = FFStruct()

    ff.ncols = wid
    ff.nrows = ht

    # ff.maxpixel = np.zeros((ht, wid), dtype=np.uint8)
    # ff.avepixel = np.zeros((ht, wid), dtype=np.uint8) + 10
    # ff.stdpixel = np.zeros((ht, wid), dtype=np.uint8) + 20
    # ff.maxframe = np.zeros((ht, wid), dtype=np.uint8) + 30

    maxpixel = np.zeros((ht, wid), dtype=np.uint8)
    avepixel = np.zeros((ht, wid), dtype=np.uint8) + 10
    stdpixel = np.zeros((ht, wid), dtype=np.uint8) + 20
    maxframe = np.zeros((ht, wid), dtype=np.uint8) + 30

    ff.array = np.stack([maxpixel, maxframe, avepixel, stdpixel], axis=0)

    # Write the FF to FITS
    write(ff, dir_path, file_name)

    # Read the FITS
    ff = read(dir_path, file_name)

    print(ff)
    print(ff.maxpixel)
    print(ff.maxframe)
    print(ff.avepixel)
    print(ff.stdpixel)
