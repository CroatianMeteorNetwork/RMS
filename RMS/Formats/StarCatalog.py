""" Reading the custom star catalog. """

from __future__ import print_function, division, absolute_import

import os
import zlib
import sys
import argparse
import time

from RMS.Math import angularSeparationDeg
from RMS.Logger import LoggingManager, getLogger

# Import the requests library for downloading the GMN star catalog
try:
    from urllib.request import urlopen  # Python 3
    from urllib.error import URLError, HTTPError
except ImportError:
    from urllib2 import urlopen  # Python 2
    from urllib2 import URLError, HTTPError

import numpy as np
from scipy.spatial import cKDTree

from RMS.Decorators import memoizeSingle
from RMS.Misc import RmsDateTime
from datetime import datetime, timezone

J2000 = datetime(2000, 1, 1, 12, 0, 0).replace(tzinfo=timezone.utc)

class Catalog:
    """
    Load a star catalogue, build a spherical KD-tree, and expose query methods.
    """

    def __init__(self, config, catalogue_time=None, ra_col=0, dec_col=1, mag_col=2, name_col=0, lim_mag=None):

        """Initialise a catalog in a spherical tree object/

        Arguments:
            config[config]: RMS config instance.

        Keyword Arguments:
            catalogue_time: Time point for the catalogue generation if none build for now.
            ra_col: Optional, default 0 - array column with ra data in degrees.
            dec_col: Optional, default 1 - array column with dec data in degrees.
            mag_col: Optional, default 2 - array column of magnitude data.
            name_col: Optional, default 3 - array column of star names.

        """

        self.ra_col, self.dec_col, self.mag_col = ra_col, dec_col, mag_col
        self.name_col = name_col

        if catalogue_time is None:
            catalogue_time = datetime.now(timezone.utc)

        # Compute the number of years from J2000
        years_from_J2000 = (catalogue_time - J2000).total_seconds() / (365.25 * 24 * 3600)


        star_catalog_status = readStarCatalog(
            config.star_catalog_path,
            config.star_catalog_file,
            lim_mag=np.inf,
            years_from_J2000=years_from_J2000,
            mag_band_ratios=config.star_catalog_band_ratios,
            additional_fields=['preferred_name', 'common_name', 'bayer_name'])

        pass

        if not star_catalog_status:
            print("Could not load star catalogue")

        catalog_stars, _, config.star_catalog_band_ratios, extras = star_catalog_status


        # Do some cleaning on the data - this is not required at present
        maskFinite = np.isfinite(catalog_stars[:, 0:3]).all(axis=1)
        maskRange = (
                (catalog_stars[:, 0] >= 0.0) & (catalog_stars[:, 0] < 360.0) &  # RA
                (catalog_stars[:, 1] >= -90.0) & (catalog_stars[:, 1] <= 90.0)  # Dec
        )

        mask = maskFinite & maskRange

        self.cat = catalog_stars[mask]
        self.names = extras['preferred_name'][mask]

        # Convert to arrays of radians
        ra, dec = np.radians(self.cat[:, ra_col]), np.radians(self.cat[:, dec_col])

        # Build tree of spherical unit vectors
        self.tree = cKDTree(np.column_stack((np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec))))

    def queryRaDec(self, ra_deg, dec_deg, radius_deg=0.1, n_brightest=1):
        """
        Tree search for ra dec coordinates, search is in a wrapped Euclidean space

        Arguments:
            ra_deg: [float] right ascension degrees
            dec_deg: [float] declination degrees

        Keyword Arguments:
            radius_deg:[float] search radius degrees default 0.1
            n_brightest: [int] number of stars to return, ordered by increasing magnitude, default 1

        Returns:
            [list of arrays]: [names, ra ,dec ,mag ,theta] theta is angular separation (degrees)
        """

        # Normalise inputs to arrays of radians
        ra_deg, dec_deg = np.atleast_1d(ra_deg), np.atleast_1d(dec_deg)
        ra, dec = np.radians(ra_deg), np.radians(dec_deg)

        # Build query vectors
        query_vectors = np.column_stack((np.cos(dec) * np.cos(ra),  np.cos(dec) * np.sin(ra), np.sin(dec)))

        # Euclidean chord distance for spherical radius
        ecd = 2 * np.sin(np.radians(radius_deg) / 2)

        results = []
        for i, (qvec, ra0, dec0) in enumerate(zip(query_vectors, ra_deg, dec_deg)):

            # KD-tree search
            result_index_on_full_catalogue = np.array(self.tree.query_ball_point(qvec, ecd), dtype=int)

            if len(result_index_on_full_catalogue) == 0:
                results.append(np.empty((0, 5), dtype=object))
                continue

            # Sort by magnitude ascending - brightest stars first
            mags = self.cat[result_index_on_full_catalogue, self.mag_col]
            chosen = result_index_on_full_catalogue[np.argsort(mags)[:n_brightest]]

            # Extract fields
            names = self.names[chosen].astype(str)
            ras, decs = self.cat[chosen, self.ra_col].astype(float), self.cat[chosen, self.dec_col].astype(float)
            mags = self.cat[chosen, self.mag_col].astype(float)


            # Angular separation
            thetas = angularSeparationDeg(ra0, dec0, ras, decs)

            # Stack result for this query
            row = np.column_stack((names, ras, decs, mags, thetas))

            results.append(row)

        # If input was scalar, return a list with a single entry of an array
        if len(ra_deg) == 1:
            return [results[0]]

        return results



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
def loadGMNStarCatalog(file_path,
                       years_from_J2000=0,
                       lim_mag=None,
                       mag_band_ratios=None,
                       catalog_file='',
                       additional_fields=False
                       ):
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
        additional_fields: [str | list | bool]  
            - False (default) - return only the basic three columns (RA, Dec, Mag).  
            - 'all' - include every extra column present in the catalog.  
            - list/tuple of column names - include exactly those extras.  
            In either non-False case the extras are returned in a dict as the 4th value.

    Returns:
        filtered_data: [ndarray] A filtered and corrected catalog contained as a structured NumPy array 
            (currently outputs only: ra, dec, mag)
        mag_band_string: [str] A string describing the magnitude band of the catalog.
        mag_band_ratios: [list] A list of BVRI magnitude band ratios for the given catalog.
        additional_fields: [dict - optional] A dictionary of additional fields requested by the user.
    """

    # Catalog data used for caching
    cache_name = "_catalog_data_{:s}".format(catalog_file.replace(".", "_"))

    # Step 1: Cache the catalog data to avoid repeated decompression
    if not hasattr(loadGMNStarCatalog, cache_name):

        # Define the data structure for the catalog (v1 - 18 columns, legacy format)
        data_types_v1 = [
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

        # Define the data structure for the catalog (v2 - 20 columns, with common_name and bayer_name)
        data_types_v2 = [
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
            ('common_name', 'S30'),
            ('bayer_name', 'S30'),
            ('Simbad_OType', 'S30')
        ]

        with open(file_path, 'rb') as fid:

            # Read the catalog header
            declared_header_size = int(np.fromfile(fid, dtype=np.uint32, count=1)[0])
            num_rows = int(np.fromfile(fid, dtype=np.uint32, count=1)[0])
            num_columns = int(np.fromfile(fid, dtype=np.uint32, count=1)[0])
            fid.read(declared_header_size - 12)  # Skip column names

            # Select data types based on number of columns (v1=18, v2=20)
            if num_columns >= 20:
                data_types = data_types_v2
            else:
                data_types = data_types_v1

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

        # Validate band_ratios length - GMN catalog expects 7 bands [B, V, R, I, G, BP, RP]
        if len(mag_band_ratios) != 7:
            # If wrong length, fall back to V band only
            print("Warning: GMN catalog expects 7 band ratios (B,V,R,I,G,BP,RP), "
                  "got {}. Using V band only.".format(len(mag_band_ratios)))
            mag_band_ratios = None

    if mag_band_ratios is not None:
        # Compute synthetic magnitudes by combining fluxes (not magnitudes).
        # The camera integrates photon flux across its bandpass, so the correct
        # combination is: m = -2.5*log10(sum(r_i * 10^(-0.4*m_i)))
        total_ratio = sum(mag_band_ratios)
        rb, rv, rr, ri, rg, rbp, rrp = [x/total_ratio for x in mag_band_ratios]

        band_mags = [
            (rb,  catalog_data['B']),
            (rv,  catalog_data['V']),
            (rr,  catalog_data['R']),
            (ri,  catalog_data['Ic']),
            (rg,  catalog_data['phot_g_mean_mag']),
            (rbp, catalog_data['phot_bp_mean_mag']),
            (rrp, catalog_data['phot_rp_mean_mag']),
        ]

        # Sum weighted fluxes from all bands with nonzero ratios.
        # Skip bands where magnitude is 0.0 (old sentinel for missing data) or NaN,
        # and renormalize the remaining ratios per-star so that missing bands don't
        # artificially brighten or dim the synthetic magnitude.
        total_flux = np.zeros(len(catalog_data), dtype=np.float64)
        valid_ratio_sum = np.zeros(len(catalog_data), dtype=np.float64)

        for ratio, mags in band_mags:
            if ratio > 0:
                valid = np.isfinite(mags) & (mags != 0.0)
                total_flux += np.where(valid, ratio * np.power(10, -0.4 * mags), 0.0)
                valid_ratio_sum += np.where(valid, ratio, 0.0)

        # Renormalize for stars with missing bands
        valid_ratio_sum = np.maximum(valid_ratio_sum, 1e-30)
        total_flux /= valid_ratio_sum

        # Convert combined flux back to magnitude
        # Stars where ALL requested bands are missing get ~75 mag and are filtered by LM cut
        total_flux = np.maximum(total_flux, 1e-30)
        synthetic_mag = -2.5 * np.log10(total_flux)
        if lim_mag is None:
            lim_mag = np.inf
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

    # Gaia DR3 stores pmra* (pmra already multiplied by cos_dec).
    # Convert it back to true delta RA by dividing by cos_dec before
    # turning mas/yr into degrees.
    cos_dec = np.cos(np.deg2rad(catalog_data['dec']))
    
    # Numerical safety: guard against |cos_dec| so small that 1/cos dec blows up.
    eps = 1e-6
    mask = np.abs(cos_dec) < eps
    cos_dec[mask] = eps
    corrected_ra = catalog_data['ra'] + (catalog_data['pmra']/cos_dec)*time_elapsed*mas_to_deg
    corrected_dec = catalog_data['dec'] + catalog_data['pmdec']*time_elapsed*mas_to_deg
    
    # Ensure RA stays within [0, 360) after proper-motion shift
    corrected_ra = np.mod(corrected_ra, 360.0)

    # Step 5: build core numeric arrays & optional extras dict ----------------
    ra_arr  = corrected_ra.astype(np.float64)
    dec_arr = corrected_dec.astype(np.float64)
    mag_arr = synthetic_mag.astype(np.float32)

    extras_dict = {}

    if additional_fields:
        # Determine which extra columns to keep
        if additional_fields == 'all':
            requested = [n for n in catalog_data.dtype.names
                         if n not in ('ra', 'dec')]
        else:
            requested = list(additional_fields)

        # Filter to only fields that exist in this catalog version (backward compatibility)
        valid = set(catalog_data.dtype.names)
        available = [n for n in requested if n in valid]

        # Populate dict with available fields only
        for name in available:
            extras_dict[name] = catalog_data[name]

    # Stack core fields for legacy callers
    core_data = np.column_stack((ra_arr, dec_arr, mag_arr))
    # Sort by descending declination
    sort_idx = np.argsort(core_data[:, 1])[::-1]
    core_data = core_data[sort_idx]
    for k in extras_dict:
        extras_dict[k] = extras_dict[k][sort_idx]

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
    # GMN catalog uses 7 bands: B, V, R, I, G, BP, RP - default to V band only
    default_gmn_ratios = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if additional_fields:
        return core_data, mag_band_string, tuple(mag_band_ratios or default_gmn_ratios), extras_dict
    else:
        return core_data, mag_band_string, tuple(mag_band_ratios or default_gmn_ratios)


def readStarCatalog(dir_path, file_name, years_from_J2000=0, lim_mag=None,
                   mag_band_ratios=None, additional_fields=None):
    """ Import the star catalog into a numpy array.
    
    Arguments:
        dir_path: [str] Path to the directory where the catalog file is located.
        file_name: [str] Name of the catalog file.

    Keyword arguments:
        years_from_J2000: [float] Decimal years elapsed from the J2000 epoch. Used for proper motion 
            correction.
        lim_mag: [float] Limiting magnitude. Stars fainter than this magnitude will be filtered out. None by
            default.
        mag_band_ratios: [list] A list of relative contributions of every photometric band to the
            final camera-bandpass magnitude. For the GMN catalog, 7 numbers: [B, V, R, I, G, BP, RP].
            Legacy catalogs use only the first 4 elements [B, V, R, I].
        additional_fields: [list | str | None] Extra GMN column names to return, or "all".
            Passed straight through to `loadGMNStarCatalog`. Ignored for other catalog types.
    
    Return:
        If additional_fields is False  
            star_data          : [ndarray]  shape (N,3) - columns (RA, Dec, Mag) in degrees.  
            mag_band_string    : [str]      description of the magnitude band.  
            mag_band_ratios    : [list]     BVRI band ratios used.
        
        If additional_fields is not False  
            star_data          : [ndarray]  shape (N,3) - (RA, Dec, Mag).  
            mag_band_string    : [str]  
            mag_band_ratios    : [list]  
            extras_dict        : [dict]     mapping {column_name: ndarray}.
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

        # --- make additional_fields hashable so memoizeSingle works ----------
        if (additional_fields is not None) and isinstance(additional_fields, list):
            additional_fields = tuple(additional_fields)

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
            catalog_file=catalog_to_load,
            additional_fields=additional_fields
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

                    # Calculate the camera-band magnitude by combining fluxes
                    total_flux = 0
                    for ratio, mag in [(rb, mag_b), (rv, mag_v), (rr, mag_r), (ri, mag_i)]:
                        if ratio > 0:
                            total_flux += ratio * 10**(-0.4 * mag)
                    mag_spectrum = -2.5 * np.log10(max(total_flux, 1e-30))


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

def testCatQueryRaDec():


    test_star_names = np.array([
        "Sirius", "Canopus", "Alpha Centauri", "Arcturus", "Vega",
        "Capella", "Rigel", "Procyon", "Achernar", "Betelgeuse",
        "Altair", "Aldebaran", "Antares", "Spica", "Fomalhaut",
        "Deneb", "Pollux", "Castor", "Regulus", "Bellatrix"
    ])

    test_star_mag = np.array([
        -1.46, -0.74, -0.27, -0.05, 0.03,
        0.08, 0.18, 0.38, 0.46, 0.50,
        0.77, 0.85, 1.06, 1.04, 1.16,
        1.25, 1.14, 1.58, 1.35, 1.64
    ])


    test_star_ra_deg = np.array([
        101.25, 96.00, 220.00, 214.00, 279.25,
        79.00, 78.75, 114.75, 24.50, 88.75,
        297.75, 69.00, 247.25, 201.25, 344.50,
        310.25, 116.25, 113.75, 152.00, 81.25
    ])

    test_star_dec_deg = np.array([
        -16.7, -52.7, -60.8, 19.2, 38.8,
        45.9, -8.2, 5.2, -57.2, 7.4,
        8.9, 16.5, -26.4, -11.2, -29.6,
        45.3, 28.0, 31.9, 12.0, 6.3
    ])

    cat = Catalog(config, lim_mag=6)
    log.info(serializeQueryResults(cat.queryRaDec(test_star_ra_deg, test_star_dec_deg, radius_deg=2, n_brightest=3), test_star_names, test_star_mag))


def serializeQueryResults(results, star_names=None, star_mag=None):

    output = []
    if len(results) == 0:
        output.append("\tNo results returned")
        return output

    last_searched_name = None
    for i, r in enumerate(results):
        for name, ra, dec, mag, theta in r:
            if star_mag is not None:
                searched_mag = star_mag[i]
            if star_names is not None:
                searched_name = star_names[i]
                if last_searched_name != searched_name:
                    output.append(f"\n\tNew search for {searched_name:20} of magnitude {float(searched_mag):4.2f}")
                    last_searched_name = searched_name


            output.append(
                f"\t\tReturned name: {name:20s} RA={float(ra):8.3f}  Dec={float(dec):8.3f}  Mag={float(mag):5.2f}  Sep={float(theta):6.3f}")

    return "\n".join(output) + "\n\n"


if __name__ == "__main__":

    import RMS.ConfigReader as cr


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description=""" Test routines for catalogue""")


    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
        help="Path to a config file which will be used instead of the default one.")

    arg_parser.add_argument('--radec', metavar='radec', nargs=4, help="""Search in radec space (degrees) RA DEC Radius Lim Mag""")

    arg_parser.add_argument('-t', '--test', action="store_true", help="""Run tests""")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, os.path.abspath('.'))

    #Initialize the logger
    log_manager = LoggingManager()
    log_manager.initLogging(config)


    #Get the logger handle
    log = getLogger("rmslogger")

    if cml_args.test:
        test = testCatQueryRaDec()


    if cml_args.radec is not None:
        ra, dec, radius = float(cml_args.radec[0]), float(cml_args.radec[1]), float(cml_args.radec[2])
        lim_mag = float(cml_args.radec[3])
        log.info(f"Querying RA={ra:.3f} DEC={dec:.3f} Radius={radius:.3f} degrees Limiting mag={lim_mag:.1f}")
        cat = Catalog(config, lim_mag=float(cml_args.radec[2]))
        results = cat.queryRaDec(ra, dec, lim_mag)
        log.info(serializeQueryResults(results))
        # Allow logger time to write
        time.sleep(1)
