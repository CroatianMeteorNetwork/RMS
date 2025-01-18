""" Reading the custom star catalog. """

from __future__ import print_function, division, absolute_import

import os
import zlib
import sys

# Import the requests library for downloading the GMN star catalog
try:
    from urllib.request import urlopen  # Python 3
    from urllib.error import URLError, HTTPError
except ImportError:
    from urllib2 import urlopen  # Python 2
    from urllib2 import URLError, HTTPError

import numpy as np

from RMS.Decorators import memoizeSingle


def downloadCatalog(url, dir_path, file_name):
    """ Download a catalog file from a given URL and save it to the specified directory. """

    try:
        response = urlopen(url)
        total_size = int(response.info().get('Content-Length', 0))
        block_size = 1024 * 1024  # 1 MB
        downloaded_size = 0

        with open(os.path.join(dir_path, file_name), 'wb') as f:
            while True:
                data = response.read(block_size)
                if not data:
                    break
                downloaded_size += len(data)
                f.write(data)
                print("\rDownloading: {:.2f}%".format(100 * float(downloaded_size) / total_size), end='')
                sys.stdout.flush()

        print(" - Done!")  # Move to the next line after download completes

        return True
    
    except HTTPError as e:
        print("HTTP Error: ", e.code, url)
        return False
    
    except URLError as e:
        print("URL Error: ", e.reason, url)
        return False

@memoizeSingle
def readBSC(file_path, file_name, years_from_J2000=0):
    """ Import the Bright Star Catalog in a numpy array. 
    
    Arguments:
        file_path: [str] Path to the catalog file.
        file_name: [str] Name of the catalog file.

    Keyword arguments:
        years_from_J2000: [float] Decimal years elapsed from the J2000 epoch (for applying proper motion 
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

        # Make an array for storing the star values (RA, dec, mag)
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


@memoizeSingle
def loadGMNStarCatalog(file_path, years_from_J2000=0, lim_mag=None, mag_band_ratios=None, catalog_file=''):
    """
    Reads in the GMN Star Catalog from a compressed binary file, applying proper motion correction,
    magnitude limiting, and synthetic magnitude computation. Adjusts the RA/Dec positions to the J2000 epoch.

    Arguments:
        file_path: [str] Path to the binary file.

    Keyword arguments:
        years_from_J2000: [float] Years elapsed since J2000 for proper motion correction (default: No correction added).
        lim_mag: [float] Limiting magnitude for filtering stars (default: None).
        mag_band_ratios: [list] Relative contributions of photometric bands [B, V, R, I]
            to compute synthetic magnitudes (default: None).
        catalog_file: [str] Name of the catalog file (default: ''). Used for caching purposes.

    Returns:
        filtered_data: [ndarray] A filtered and corrected catalog contained as a structured NumPy array 
            (currently outputs only: ra, dec, mag)
        mag_band_string: [str] A string describing the magnitude band of the catalog.
        mag_band_ratios: [list] A list of BVRI magnitude band ratios for the given catalog.
    """

    # Catalog data used for caching
    cache_name = "_catalog_data_{:s}".format(catalog_file.replace(".", "_"))

    # Step 1: Cache the catalog data to avoid repeated decompression
    if not hasattr(loadGMNStarCatalog, cache_name):

        # Define the data structure for the catalog
        data_types = [
            ('designation', 'S30'),
            ('ra', 'f8'),
            ('dec', 'f8'),
            ('pmra', 'f8'),
            ('pmdec', 'f8'),
            ('phot_g_mean_mag', 'f4'),
            ('phot_bp_mean_mag', 'f4'),
            ('phot_rp_mean_mag', 'f4'),
            ('classprob_dsc_specmod_star', 'f4'),
            ('classprob_dsc_specmod_binarystar', 'f4'),
            ('spectraltype_esphs', 'S8'),
            ('B', 'f4'),
            ('V', 'f4'),
            ('R', 'f4'),
            ('Ic', 'f4'),
            ('oid', 'i4'),
            ('preferred_name', 'S30'),
            ('Simbad_OType', 'S30')
        ]

        with open(file_path, 'rb') as fid:

            # Read the catalog header
            declared_header_size = int(np.fromfile(fid, dtype=np.uint32, count=1)[0])
            num_rows = int(np.fromfile(fid, dtype=np.uint32, count=1)[0])
            num_columns = int(np.fromfile(fid, dtype=np.uint32, count=1)[0])
            fid.read(declared_header_size - 12)  # Skip column names

            # Read and decompress the catalog data
            compressed_data = fid.read()
            decompressed_data = zlib.decompress(compressed_data)
            catalog_data = np.frombuffer(decompressed_data, dtype=data_types, count=num_rows)

        # Cache the catalog data for future use
        setattr(loadGMNStarCatalog, cache_name, catalog_data)
    
    else:
        catalog_data = getattr(loadGMNStarCatalog, cache_name)


    # Step 2: Compute synthetic magnitudes if required
    if mag_band_ratios is not None:
        
        # Compute synthetic magnitudes if band ratios are provided
        total_ratio = sum(mag_band_ratios)
        rb, rv, rr, ri, rg, rbp, rrp = [x/total_ratio for x in mag_band_ratios]
        synthetic_mag = (
            rb*catalog_data['B'] +
            rv*catalog_data['V'] +
            rr*catalog_data['R'] +
            ri*catalog_data['Ic'] +
            rg*catalog_data['phot_g_mean_mag'] +
            rbp*catalog_data['phot_bp_mean_mag'] +
            rrp*catalog_data['phot_rp_mean_mag']
        )
        mag_mask = synthetic_mag <= lim_mag

    else:
        synthetic_mag = catalog_data['V']

    # Step 3: Filter stars based on limiting magnitude
    if lim_mag is not None:
        
        # Generate a mask for stars fainter than the limiting magnitude
        mag_mask = synthetic_mag <= lim_mag

        # Apply the magnitude filter
        catalog_data = catalog_data[mag_mask]
        synthetic_mag = synthetic_mag[mag_mask]
        

    # Step 4: Apply proper motion correction
    mas_to_deg = 1/(3.6e6)  # Conversion factor for mas/yr to degrees/year
    
    # GMN catalog is relative to the J2016 epoch (from GAIA DR3)
    time_elapsed = years_from_J2000 - 16

    # Correct the RA and Dec relative to the years_from_J2000 argument
    corrected_ra = catalog_data['ra'] + catalog_data['pmra']*time_elapsed*mas_to_deg
    corrected_dec = catalog_data['dec'] + catalog_data['pmdec']*time_elapsed*mas_to_deg

    # Step 5: Prepare the filtered data for output
    filtered_data = np.zeros((len(catalog_data), 3), dtype=np.float64)
    filtered_data[:, 0] = corrected_ra  # RA
    filtered_data[:, 1] = corrected_dec  # Dec
    filtered_data[:, 2] = synthetic_mag  # Magnitude

    # Step 6: Sort the filtered data by descending declination
    filtered_data = filtered_data[np.argsort(filtered_data[:, 1])[::-1]]

    # Step 7: Generate the magnitude band string
    if mag_band_ratios is None:
        mag_band_string = "GMN V band"

    else:

        # Generate the magnitude band string
        bands = ['B', 'V', 'R', 'I', 'G', 'BP', 'RP']
        mag_band_string = "GMN "
        count = 0
        for i, band in enumerate(bands):
            if mag_band_ratios[i] > 0:
                if count > 0:
                    mag_band_string += "+ "
                mag_band_string += "{:.2f}{} ".format(mag_band_ratios[i], band)
                count += 1

        mag_band_string = mag_band_string.strip()

    # Step 8: Return the filtered data, magnitude band string, and band ratios
    return filtered_data, mag_band_string, tuple(mag_band_ratios or [0.0, 1.0, 0.0, 0.0])


def readStarCatalog(dir_path, file_name, years_from_J2000=0, lim_mag=None, mag_band_ratios=None):
    """ Import the star catalog into a numpy array.
    
    Arguments:
        dir_path: [str] Path to the directory where the catalog file is located.
        file_name: [str] Name of the catalog file.

    Keyword arguments:
        years_from_J2000: [float] Decimal years elapsed from the J2000 epoch. Used for proper motion 
            correction.
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
        BSC_data = readBSC(dir_path, file_name, years_from_J2000=years_from_J2000)

        # Filter out stars fainter than the limiting magnitude, if it was given
        if lim_mag is not None:
            BSC_data = BSC_data[BSC_data[:, 2] < lim_mag]

        return BSC_data, 'BSC5 V band', [0.0, 1.0, 0.0, 0.0]


    # Use the GAIA star catalog
    if 'gaia' in file_name.lower():
        return loadGaiaCatalog(dir_path, file_name, lim_mag=lim_mag), 'GAIA G band', [0.45, 0.70, 0.72, 0.50]

    # Use the GMN star catalog
    if "GMN_StarCatalog".lower() in file_name.lower():

        # Define catalog names for the bright and faint stars
        gmn_starcat_lm9 = "GMN_StarCatalog_LM9.0.bin"
        gmn_starcat_lm12 = "GMN_StarCatalog_LM12.0.bin"

        # Check the existence of the LM 12.0 catalog file
        gmn_starcat_lm12_exists = os.path.exists(os.path.join(dir_path, gmn_starcat_lm12))

        # Ensure mag_band_ratios is a tuple for caching
        if (mag_band_ratios is not None) and isinstance(mag_band_ratios, list):
            mag_band_ratios = tuple(mag_band_ratios)

        # Determine which catalog file to use based on the limiting magnitude
        if (lim_mag is not None) and (lim_mag <= 9.0):
            catalog_to_load = gmn_starcat_lm9

        else:

            # If the full catalog is missing, post a notification and load the LM9.0 catalog
            if gmn_starcat_lm12_exists:

                catalog_to_load = gmn_starcat_lm12

            else:

                # URL to the LM+12.0 catalog
                gmn_starcat_lm12_url = "https://globalmeteornetwork.org/projects/gmn_star_catalog/GMN_StarCatalog_LM12.0.bin"

                # Display a warning message that the catalog will be downloaded
                print("The full catalog (LM+12.0) is beind downloaded from the GMN server... ")

                # Download the full catalog from the GMN server
                download_status = downloadCatalog(gmn_starcat_lm12_url, dir_path, gmn_starcat_lm12)

                if download_status:
                    catalog_to_load = gmn_starcat_lm12

                else:
                    print("Error downloading the full catalog, loading the LM+9.0 catalog instead. ")
                    catalog_to_load = gmn_starcat_lm9


        file_path = os.path.join(dir_path, catalog_to_load)

        return loadGMNStarCatalog(
            file_path, 
            years_from_J2000=years_from_J2000, 
            lim_mag=lim_mag, 
            mag_band_ratios=mag_band_ratios,
            catalog_file=catalog_to_load
        )


    ### Default to loading the SKY2000 catalog ###

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


            # Use visual magnitude by default
            mag_spectrum = mag_v

            # Calculate the camera-bandpass magnitude if given
            if mag_band_ratios is not None:

                # Only take the first 4 ratios
                if len(mag_band_ratios) > 4:
                    mag_band_ratios = mag_band_ratios[:4]


                if len(mag_band_ratios) == 4:

                    # If all ratios are zero, use the visual magnitude
                    if sum(mag_band_ratios) == 0:
                        mag_band_ratios = [0, 1.0, 0, 0]
                    
                    # Calculate the B band magnitude
                    mag_b = mag_v + mag_bv

                    rb, rv, rr, ri = mag_band_ratios

                    ratio_sum = sum(mag_band_ratios)

                    # Make sure the ratios are normalized to 1.0
                    rb /= ratio_sum
                    rv /= ratio_sum
                    rr /= ratio_sum
                    ri /= ratio_sum

                    # Calculate the camera-band magnitude
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