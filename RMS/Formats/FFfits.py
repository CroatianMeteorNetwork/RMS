

from __future__ import print_function, division, absolute_import

import os

import numpy as np
from astropy.io import fits

from RMS.Formats.FFStruct import FFStruct
import datetime



def filenameToDatetimeStr(file_name, iso8601=False):
    """ Converts FS and FF bin file name to a datetime object.

    Arguments:
        file_name: [str] Name of a FF or FS file.

    Keyword arguments:
        iso8601: [bool] True if the output should be in ISO 8601 format. False by default.

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


    if iso8601:
        dt_str = datetime.datetime(year, month, day, hour, minute, seconds, microseconds).isoformat(timespec='microseconds')

    else:
        dt_str = datetime.datetime(year, month, day, hour, minute, seconds, microseconds).strftime("%Y-%m-%d %H:%M:%S.%f")

    return dt_str



# Index of the optional HDU carrying the sub-ADU residual of the average (after the four legacy planes)
AVEFRAC_HDU = 5


def splitAvepixel16(avepixel16):
    """ Split the 8.8 fixed-point average into the two planes stored in the FF file.

    Arguments:
        avepixel16: [2D ndarray uint16] Average in units of 1/256 ADU.

    Return:
        (avepixel, averesid):
            - avepixel: [2D ndarray uint8] The average rounded to whole ADU, i.e. the legacy plane.
            - averesid: [2D ndarray uint8] Signed residual avepixel16 - 256*avepixel, in 1/256 ADU
                and stored as its two's complement byte (equivalently, the low byte of avepixel16).
    """

    avepixel16 = np.asarray(avepixel16)

    avepixel = np.clip((avepixel16.astype(np.uint32) + 128) >> 8, 0, 255).astype(np.uint8)
    averesid = (avepixel16 & 0xFF).astype(np.uint8)

    return avepixel, averesid


def joinAvepixel16(avepixel, averesid):
    """ Reassemble the 8.8 fixed-point average from the planes stored in the FF file, the exact
        inverse of splitAvepixel16.

    Arguments:
        avepixel: [2D ndarray uint8] Average rounded to whole ADU.
        averesid: [2D ndarray uint8] Two's complement byte of the sub-ADU residual.

    Return:
        [2D ndarray uint16] Average in units of 1/256 ADU.
    """

    avepixel16 = (np.asarray(avepixel).astype(np.int32) << 8) \
        + np.asarray(averesid).astype(np.uint8).view(np.int8).astype(np.int32)

    return np.clip(avepixel16, 0, 65535).astype(np.uint16)



def read(directory, filename, array=False, full_filename=False, memmap=True):
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
        file_path = os.path.join(directory, filename)
    else:
        file_path = os.path.join(directory, "FF_" + filename + ".fits")

    # Init an empty FF structure
    ff = FFStruct()

    # Unsigned 16-bit planes (native 16-bit camera files) carry BZERO scaling, which astropy
    # refuses to read lazily from an explicitly requested memory map. Fall back to a plain read
    # for such files - all planes are copied out below anyway, so nothing is lost. 8-bit files
    # keep the memmap path
    if memmap:
        with fits.open(file_path, memmap=True) as hdulist:
            if any(('BZERO' in hdu.header) or ('BSCALE' in hdu.header) or ('BLANK' in hdu.header)
                    for hdu in hdulist[1:]):
                memmap = False

    # Read in the FITS. Pass the path (not a pre-opened handle) so astropy owns and closes
    # the file, and use a context manager so the file/memmap is always released, even on
    # error. The image data is copied out of the HDUs (below) so it stays valid after the
    # file is closed. Without the copy, returning memmap-backed views keeps a file handle
    # open per FF; reading many FFs in sequence and retaining the arrays then exhausts the
    # process file descriptor limit ("too many open files"). See issue #406.
    with fits.open(file_path, memmap=memmap) as hdulist:

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

        # Check for the DATE-OBS field and read datetime from filename it if it doesn't exist
        if 'DATE-OBS' in head:
            ff.starttime = head['DATE-OBS']
        else:
            ff.starttime = filenameToDatetimeStr(filename, iso8601=True)

        # Read in the image data, copying it so it remains valid (and detached from the
        # memmap) after the file is closed
        ff.maxpixel = hdulist[1].data.copy()
        ff.maxframe = hdulist[2].data.copy()
        ff.avepixel = hdulist[3].data.copy()
        ff.stdpixel = hdulist[4].data.copy()

        # Full-precision average. The AVEPIXEL plane is the legacy 8-bit average (rounded), and
        # the optional fifth HDU carries the sub-ADU residual of the 8.8 fixed-point mean, so
        # avepixel16 = 256*avepixel + residual. Readers that only know the four legacy planes
        # never see the extra HDU
        avefrac = head.get('AVEFRAC', 0)
        if avefrac and (len(hdulist) > AVEFRAC_HDU) and (hdulist[AVEFRAC_HDU].name == 'AVEFRAC'):
            ff.avepixel16 = joinAvepixel16(ff.avepixel, hdulist[AVEFRAC_HDU].data)
            ff.avegamma = head.get('AVEGAMMA', 1.0)

        # Transitional: files from the first draft of this format stored the 8.8 fixed-point mean
        # as a uint16 AVEPIXEL plane. Derive the legacy 8-bit view from it. This path only exists
        # for the test-station files written by that draft
        elif avefrac and (ff.avepixel.dtype.itemsize == 2) and (ff.nbits <= 8):
            ff.avepixel16 = ff.avepixel
            ff.avegamma = head.get('AVEGAMMA', 1.0)
            ff.avepixel = splitAvepixel16(ff.avepixel16)[0]

    if array:
        ff.array = np.dstack([ff.maxpixel, ff.maxframe, ff.avepixel, ff.stdpixel])

        ff.array = np.swapaxes(ff.array, 0, 1)
        ff.array = np.swapaxes(ff.array, 0, 2)

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
    head['DATE-OBS'] = ff.starttime

    # Deconstruct the 3D array into individual images
    if ff.array is not None:
        ff.maxpixel, ff.maxframe, ff.avepixel, ff.stdpixel = np.split(ff.array, 4, axis=0)
        ff.maxpixel = ff.maxpixel[0]
        ff.maxframe = ff.maxframe[0]
        ff.avepixel = ff.avepixel[0]
        ff.stdpixel = ff.stdpixel[0]

    # Full-precision average: the AVEPIXEL plane stays the legacy 8-bit average, and the sub-ADU
    # residual of the 8.8 fixed-point mean goes into a fifth HDU. The four legacy planes keep
    # their position and dtype, so readers that predate the residual plane read the file as before
    # and simply never look past the fourth HDU. Both planes are derived from avepixel16 so they
    # always agree
    avepixel = ff.avepixel
    averesid = None
    if getattr(ff, 'avepixel16', None) is not None:
        avepixel, averesid = splitAvepixel16(ff.avepixel16)
        head['AVEFRAC'] = (8, 'sub-ADU bits of the mean in the AVEFRAC HDU')
        head['AVEGAMMA'] = (float(getattr(ff, 'avegamma', 1.0) or 1.0),
            'gamma used for linear-domain averaging')

    # Create the primary part
    prim = fits.PrimaryHDU(header=head)

    # Combine everything into into FITS
    hdulist = fits.HDUList([prim,
        fits.ImageHDU(ff.maxpixel, name='MAXPIXEL'),
        fits.ImageHDU(ff.maxframe, name='MAXFRAME'),
        fits.ImageHDU(avepixel, name='AVEPIXEL'),
        fits.ImageHDU(ff.stdpixel, name='STDPIXEL')])

    if averesid is not None:
        hdulist.append(fits.ImageHDU(averesid, name='AVEFRAC'))

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
    ff.nbits = 8
    ff.nframes = 256
    ff.first = 0
    ff.camno = 1
    ff.fps = 25.0
    ff.starttime = "2022-02-26T18:17:11.737000"

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
