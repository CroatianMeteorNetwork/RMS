""" Reading the cusom star catalog. """

from __future__ import print_function, division, absolute_import

import os
import numpy as np

from RMS.Decorators import memoizeSingle

@memoizeSingle
def readBSC(file_path, file_name, years_from_J2000=0):
    """ Import the Bright Star Catalog in a numpy array. 
    
    Arguments:
        file_path: [str] Path to the catalog file.
        file_name: [str] Name of the catalog file.

    Keyword arguments:
        years_from_J2000: [float] Decimal years elapsed from the J2000 epoch (for applying poper motion 
            correction, leave at 0 to read non-corrected coordinates).
    
    Return:
        BSC_data: [ndarray] Array of (RA, dec, mag) parameters for each star in the BSC corrected for
            proper motion, coordinates are in degrees.
    """
    
    bsc_path = os.path.join(file_path, file_name)

    # Check if the BSC file exits
    if not os.path.isfile(bsc_path):
        return False

    with open(os.path.join(file_path, file_name), 'rb') as fid:

        ### Define data types for reading the file
        
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

        ###

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
        c = 0
        for _ in range(star_num):

            # Read the entry for each star
            catalog_No = np.fromfile(fid, dtype=float_32d, count=1)[0]
            RA = np.fromfile(fid, dtype=float_64d, count=1)[0]
            dec = np.fromfile(fid, dtype=float_64d, count=1)[0]
            spectral = np.fromfile(fid, dtype=char_8d, count=1)[0]
            mag = np.fromfile(fid, dtype=int_8d, count=1)[0].astype(float)/100
            RA_proper = np.fromfile(fid, dtype=float_32d, count=1)[0]
            dec_proper = np.fromfile(fid, dtype=float_32d, count=1)[0]

            # Skip RA/Dec = (zero, zero) entries
            if (RA == 0) and (dec == 0):
                continue

            # print(catalog_No, np.degrees(RA), np.degrees(dec), spectral, mag, RA_proper, dec_proper)

            # Assign data to array and apply the proper motion correction
            BSC_data[c][0] = np.degrees(RA + RA_proper*years_from_J2000)
            BSC_data[c][1] = np.degrees(dec + dec_proper*years_from_J2000)
            BSC_data[c][2] = mag

            c += 1


    # Cut the list to the number of stars actually added
    BSC_data = BSC_data[:c]

    # Sort stars by descending declination
    BSC_data = BSC_data[BSC_data[:,1].argsort()[::-1]]

    return BSC_data



def loadGaiaCatalog(dir_path, file_name, lim_mag=None):
    """ Read star data from the GAIA catalog in the .npy format. 
    
    Arguments:
        dir_path: [str] Path to the directory where the catalog file is located.
        file_name: [str] Name of the catalog file.

    Keyword arguments:
        lim_mag: [float] Faintest magnitude to return. None by default, which will return all stars.

    Return:
        results: [2d ndarray] Rows of (ra, dec, mag), angular values are in degrees.
    """

    file_path = os.path.join(dir_path, file_name)

    # Read the catalog
    results = np.load(str(file_path), allow_pickle=False)


    # Filter by limiting magnitude
    if lim_mag is not None:

        results = results[results[:, 2] <= lim_mag]


    # Sort stars by descending declination
    results = results[results[:,1].argsort()[::-1]]


    return results



def readStarCatalog(dir_path, file_name, lim_mag=None, mag_band_ratios=None):
    """ Import the star catalog into a numpy array.
    
    Arguments:
        dir_path: [str] Path to the directory where the catalog file is located.
        file_name: [str] Name of the catalog file.

    Keyword arguments:
        lim_mag: [float] Limiting magnitude. Stars fainter than this magnitude will be filtered out. None by
            default.
        mag_band_ratios: [list] A list of relative contributions of every photometric band (BVRI) to the 
            final camera-bandpass magnitude. The list should contain 4 numbers, one for every band: 
                [B, V, R, I].
    
    Return:
        (star_data, mag_band_string): 
            star_data: [ndarray] Array of (RA, dec, mag) parameters for each star, coordinates are in degrees.
            mag_band_string: [str] Text describing the magnitude band of the catalog.
            mag_band_ratios: [list] A list of BVRI magnitude band ratios for the given catalog.
    """

    # Use the BSC star catalog if BSC is given
    if 'BSC' in file_name:

        # Load all BSC stars
        BSC_data = readBSC(dir_path, file_name)

        # Filter out stars fainter than the limiting magnitude, if it was given
        if lim_mag is not None:
            BSC_data = BSC_data[BSC_data[:, 2] < lim_mag]

        return BSC_data, 'BSC5 V band', [0.0, 1.0, 0.0, 0.0]


    # Use the GAIA star catalog
    if 'gaia' in file_name.lower():
        return loadGaiaCatalog(dir_path, file_name, lim_mag=lim_mag), 'GAIA G band', [0.45, 0.70, 0.72, 0.50]



    ### Load the SKY2000 catalog ###

    file_path = os.path.join(dir_path, file_name)

    # Check if the star catalog exits
    if not os.path.isfile(file_path):
        return False


    with open(file_path) as f:

        star_data = []

        for line in f:

            line = line.replace('\n', '')

            if not line:
                continue

            # Skip lines which do not begin with a number
            try:
                float(line[0:4])

            except:
                continue


            # Unpack star parameters
            ra, dec, mag_v, mag_bv, mag_r, mag_i = list(map(float, line.split()))


            # Use visual magnitude by defualt
            mag_spectrum = mag_v

            # Calculate the camera-bandpass magnitude if given
            if mag_band_ratios is not None:

                if len(mag_band_ratios) == 4:
                    
                    # Calculate the B band magnitude
                    mag_b = mag_v + mag_bv

                    rb, rv, rr, ri = mag_band_ratios

                    ratio_sum = sum(mag_band_ratios)

                    # Make sure the ratios are normalized to 1.0
                    rb /= ratio_sum
                    rv /= ratio_sum
                    rr /= ratio_sum
                    ri /= ratio_sum

                    # Calcualte the camera-band magnitude
                    mag_spectrum = rb*mag_b + rv*mag_v + rr*mag_r + ri*mag_i


                else:
                    mag_band_ratios = [0, 1.0, 0, 0]


            else:
                mag_band_ratios = [0, 1.0, 0, 0]


            # Skip the star if it fainter then the given limiting magnitude
            if lim_mag is not None:
                if mag_spectrum > lim_mag:
                    continue

            star_data.append([ra, dec, mag_spectrum])


    # Convert the data to a numpy array
    star_data = np.array(star_data).astype(np.float64)

    # Sort stars by descending declination
    star_data = star_data[star_data[:,1].argsort()[::-1]]


    mag_band_string = "Sky2000 {:.2f}B + {:.2f}V + {:.2f}R + {:.2f}I".format(*mag_band_ratios)

    return star_data, mag_band_string, mag_band_ratios




if __name__ == "__main__":

    import RMS.ConfigReader as cr

    # Load the configuration file
    config = cr.parse(".config")

    # Test open the file
    print(readStarCatalog(config.star_catalog_path, config.star_catalog_file, \
        mag_band_ratios=config.star_catalog_band_ratios))