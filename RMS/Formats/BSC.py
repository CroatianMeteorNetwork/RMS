import os
import numpy as np

def readBSC(file_path, file_name, years_from_J2000=0):
    """ Import the Bright Star Catalog in a numpy array. 

    @param file_path: [str] path to the catalog file
    @param file_name: [str] name of the catalog file
    @param years_from_J2000: [float] decimal years elapsed from the J2000 epoch (for applying poper motion 
        correction, leave at 0 to read non-corrected coordinates)

    @return BSC_data: [ndarray] an array of (RA, dec, mag) parameters for each star in the BSC corrected for
        proper motion, coordinates are in degrees
    """

    bsc_path = os.path.join(file_path, file_name)

    # Check if the BSC file exits
    if not os.path.isfile(bsc_path):
        return False

    with open(os.path.join(file_path, file_name), 'rb') as fid:

        ## Define data types for reading the file
        # 32-bit integer
        int_32d = np.dtype('<i4')
        # 8-bit integer
        int_8d = np.dtype('<i2')
        # 32-bit float
        float_32d = np.dtype('<f4')
        # 64-bit float
        float_64d = np.dtype('<f8')
        # 8-bit char
        char_8d = np.dtype('<a2')

        # Read the header
        star_seq_offset = np.fromfile(fid, dtype=int_32d, count = 1)[0]
        star_first = np.fromfile(fid, dtype=int_32d, count = 1)[0]
        star_num = -np.fromfile(fid, dtype=int_32d, count = 1)[0]
        star_id_status = np.fromfile(fid, dtype=int_32d, count = 1)[0]
        star_proper_motion = np.fromfile(fid, dtype=int_32d, count = 1)[0]
        magnitudes = np.fromfile(fid, dtype=int_32d, count = 1)[0]
        bytes_per_entry = np.fromfile(fid, dtype=int_32d, count = 1)[0]

        # Make an array for storing the star vaues (RA, dec, mag)
        BSC_data = np.zeros(shape=(star_num, 3), dtype=float_64d)

        # Read entries
        for i in range(star_num):

            # Read the entry for each star
            catalog_No = np.fromfile(fid, dtype=float_32d, count=1)[0]
            RA = np.fromfile(fid, dtype=float_64d, count=1)[0]
            dec = np.fromfile(fid, dtype=float_64d, count=1)[0]
            spectral = np.fromfile(fid, dtype=char_8d, count=1)[0]
            mag = np.fromfile(fid, dtype=int_8d, count=1)[0].astype(np.float)/100
            RA_proper = np.fromfile(fid, dtype=float_32d, count=1)[0]
            dec_proper = np.fromfile(fid, dtype=float_32d, count=1)[0]

            # print catalog_No, RA, dec, spectral, mag, RA_proper, dec_proper

            # Assign data to array and apply the proper motion correction
            BSC_data[i][0] = np.degrees(RA + RA_proper*years_from_J2000)
            BSC_data[i][1] = np.degrees(dec + dec_proper*years_from_J2000)
            BSC_data[i][2] = mag


    # Sort stars by descending declination
    BSC_data = BSC_data[BSC_data[:,1].argsort()[::-1]]

    return BSC_data



# Test open the file
# print readBSC('../../Catalogs', 'BSC5', 16.5)