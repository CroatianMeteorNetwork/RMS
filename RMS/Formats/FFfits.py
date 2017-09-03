

from __future__ import print_function, division, absolute_import

import os

import numpy as np
from astropy.io import fits

from RMS.Formats.FFStruct import FFStruct



def read(directory, filename, array=False):
    """ Read a FF structure from a FITS file. 
    
    Arguments:
        directory: [str] Path to directory containing file
        filename: [str] Name of FF*.fits file (either with FF and extension or without)

    Keyword arguments:
        _array: [ndarray] True in order to populate structure's array element (default is False)
    
    Return:
        [ff structure]

    """

    # Make sure the file starts with "FF"
    if filename[:2] == "FF":
        fid = open(os.path.join(directory, filename), "rb")
    else:
        fid = open(os.path.join(directory, "FF" + filename + ".fits"), "rb")

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

    # Read in the image data
    ff.maxpixel = hdulist[1].data
    ff.maxframe = hdulist[2].data
    ff.avepixel = hdulist[3].data
    ff.stdpixel = hdulist[4].data

    if array:
        ff.array = np.dstack([ff.maxpixel, ff.maxframe, ff.avepixel, ff.stdpixel])

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

    file_path = os.path.join(directory, filename)

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
    hdulist.writeto(file_path, clobber=True)






if __name__ == "__main__":

    dir_path = '.'
    file_name = 'FFtest.fits'

    wid = 720
    ht = 576


    ff = FFStruct()

    ff.ncols = wid
    ff.nrows = ht

    ff.maxpixel = np.zeros((ht, wid), dtype=np.uint8)
    ff.avepixel = np.zeros((ht, wid), dtype=np.uint8) + 10
    ff.stdpixel = np.zeros((ht, wid), dtype=np.uint8) + 20
    ff.maxframe = np.zeros((ht, wid), dtype=np.uint8) + 30

    # Write the FF to FITS
    write(ff, dir_path, file_name)

    # Read the FITS
    ff = read(dir_path, file_name)

    print(ff)