""" Reading the custom star catalog. """

from __future__ import print_function, division, absolute_import

import os
import zlib
import sys
import tempfile

# Import the requests library for downloading the GMN star catalog
try:
    from urllib.request import urlopen  # Python 3
    from urllib.error import URLError, HTTPError
except ImportError:
    from urllib2 import urlopen  # Python 2
    from urllib2 import URLError, HTTPError

import numpy as np

from RMS.Decorators import memoizeSingle
from RMS.Misc import RmsDateTime
from datetime import datetime


# Data structure for the GMN catalog (v1 - 18 columns, legacy format)
GMN_CATALOG_DTYPE_V1 = np.dtype([
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
])

# Data structure for the GMN catalog (v2 - 20 columns, with common_name and bayer_name)
GMN_CATALOG_DTYPE_V2 = np.dtype([
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
])


def gmnCatalogDtype(num_columns):
    """ Select the GMN catalog dtype based on the number of columns declared in the header. """
    if num_columns >= 20:
        return GMN_CATALOG_DTYPE_V2
    return GMN_CATALOG_DTYPE_V1


def removeFileSilently(path):
    """ Remove a file, ignoring the error if it does not exist. """
    try:
        os.remove(path)
    except OSError:
        pass


def downloadCatalog(url, dir_path, file_name):
    """ Download a catalog file from a given URL and save it to the specified directory.

    The file is first written to a unique temporary ".part" file in the same directory and only
    moved into its final location once the download has completed (and, when the server reports a
    Content-Length, once the full size has been received). This guarantees that an interrupted
    download can never leave a partial/corrupt catalog at the destination path. The temp name is
    made unique (via tempfile.mkstemp) so concurrent processes downloading the same catalog do not
    clobber each other's temp file.

    Note: if the server reports no Content-Length and closes the connection cleanly but early, the
    truncation cannot be detected here (there is no expected size to check against).
    """

    dest_path = os.path.join(dir_path, file_name)

    # Create a unique temp file in the same directory (so os.replace stays atomic on one filesystem).
    # Close the descriptor immediately and reopen by path below, so an early failure cannot leak the fd.
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=file_name + ".", suffix=".part", dir=dir_path)
    os.close(tmp_fd)

    try:
        response = urlopen(url)
        total_size = int(response.info().get('Content-Length', 0))
        block_size = 1024 * 1024  # 1 MB
        downloaded_size = 0

        with open(tmp_path, 'wb') as f:
            while True:
                data = response.read(block_size)
                if not data:
                    break
                downloaded_size += len(data)
                f.write(data)
                if total_size > 0:
                    print("\rDownloading: {:.2f}%".format(100 * float(downloaded_size) / total_size), end='')
                    sys.stdout.flush()

        # If the server declared the size, make sure the whole file was received
        if (total_size > 0) and (downloaded_size != total_size):
            print("\nIncomplete download: received {} of {} bytes.".format(downloaded_size, total_size))
            removeFileSilently(tmp_path)
            return False

        # Atomically move the completed file into place (atomic when on the same filesystem)
        try:
            os.replace(tmp_path, dest_path)
        except AttributeError:
            # Python 2 fallback: os.replace is unavailable, emulate the overwrite
            removeFileSilently(dest_path)
            os.rename(tmp_path, dest_path)

        print(" - Done!")  # Move to the next line after download completes

        return True

    except HTTPError as e:
        print("HTTP Error: ", e.code, url)
        removeFileSilently(tmp_path)
        return False

    except URLError as e:
        print("URL Error: ", e.reason, url)
        removeFileSilently(tmp_path)
        return False

    except Exception as e:
        # Any other failure (e.g. a connection reset mid-stream) must not leave a partial file
        print("\nError downloading catalog: {}".format(e))
        removeFileSilently(tmp_path)
        return False


# Exceptions raised by loadGMNStarCatalog when the catalog file is truncated or corrupt
# (bad header, failed zlib decompression, buffer smaller than the declared record count, etc.).
CORRUPT_CATALOG_ERRORS = (zlib.error, ValueError, OSError, EOFError, IndexError)


def loadGMNCatalog(dir_path, use_full_catalog, full_name, full_url, fallback_name, load_kwargs):
    """ Load the GMN star catalog, lazily downloading and repairing the full catalog as needed.

    The full (LM+12.0) catalog is downloaded only if it is missing. Integrity is checked lazily:
    the file is simply loaded, and only if the load fails (a truncated/corrupt file, e.g. from an
    interrupted download) is it deleted and downloaded once more. This avoids reading and
    decompressing the catalog twice on every start-up, which matters on worn SD cards. If the full
    catalog still cannot be loaded, the bundled LM+9.0 catalog is used instead.

    Arguments:
        dir_path: [str] Directory holding the catalog files.
        use_full_catalog: [bool] Whether the full (LM+12.0) catalog is required.
        full_name: [str] File name of the full (LM+12.0) catalog.
        full_url: [str] URL to download the full catalog from.
        fallback_name: [str] File name of the catalog to use if the full one is unavailable.
        load_kwargs: [dict] Keyword arguments passed through to loadGMNStarCatalog.

    Return:
        Whatever loadGMNStarCatalog returns for the catalog that was successfully loaded.
    """

    if use_full_catalog:

        full_path = os.path.join(dir_path, full_name)

        # Download the full catalog if it is not present yet
        if not os.path.exists(full_path):
            print("The full catalog ({}) is being downloaded from the GMN server...".format(full_name))
            downloadCatalog(full_url, dir_path, full_name)

        # Try to load the full catalog; repair it once if the load reveals corruption
        if os.path.exists(full_path):

            try:
                return loadGMNStarCatalog(full_path, catalog_file=full_name, **load_kwargs)

            except CORRUPT_CATALOG_ERRORS as e:
                print("Full star catalog '{}' is corrupt ({}) - re-downloading.".format(full_name, e))
                removeFileSilently(full_path)

                if downloadCatalog(full_url, dir_path, full_name):
                    try:
                        return loadGMNStarCatalog(full_path, catalog_file=full_name, **load_kwargs)
                    except CORRUPT_CATALOG_ERRORS as e2:
                        print("Re-downloaded catalog still unreadable ({}).".format(e2))
                        removeFileSilently(full_path)

        print("Could not obtain the full catalog, loading '{}' instead.".format(fallback_name))

    # Load the fallback (LM+9.0) catalog
    fallback_path = os.path.join(dir_path, fallback_name)
    return loadGMNStarCatalog(fallback_path, catalog_file=fallback_name, **load_kwargs)

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

        with open(file_path, 'rb') as fid:

            # Read the catalog header
            declared_header_size = int(np.fromfile(fid, dtype=np.uint32, count=1)[0])
            num_rows = int(np.fromfile(fid, dtype=np.uint32, count=1)[0])
            num_columns = int(np.fromfile(fid, dtype=np.uint32, count=1)[0])
            fid.read(declared_header_size - 12)  # Skip column names

            # Select data types based on number of columns (v1=18, v2=20)
            data_types = gmnCatalogDtype(num_columns)

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

        # Ensure mag_band_ratios is a tuple for caching
        if (mag_band_ratios is not None) and isinstance(mag_band_ratios, list):
            mag_band_ratios = tuple(mag_band_ratios)

        # --- make additional_fields hashable so memoizeSingle works ----------
        if (additional_fields is not None) and isinstance(additional_fields, list):
            additional_fields = tuple(additional_fields)

        # URL to the LM+12.0 catalog
        gmn_starcat_lm12_url = "https://globalmeteornetwork.org/projects/gmn_star_catalog/GMN_StarCatalog_LM12.0.bin"

        # The full (LM+12.0) catalog is only needed when stars fainter than mag 9.0 are requested
        use_full_catalog = (lim_mag is None) or (lim_mag > 9.0)

        # Lazily load the catalog: the full catalog is downloaded if missing, and re-downloaded
        # only if a load attempt reveals it is corrupt (e.g. an interrupted download), falling
        # back to the LM+9.0 catalog if it cannot be obtained.
        load_kwargs = dict(
            years_from_J2000=years_from_J2000,
            lim_mag=lim_mag,
            mag_band_ratios=mag_band_ratios,
            additional_fields=additional_fields,
        )

        return loadGMNCatalog(dir_path, use_full_catalog, gmn_starcat_lm12,
                              gmn_starcat_lm12_url, gmn_starcat_lm9, load_kwargs)


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




if __name__ == "__main__":

    import RMS.ConfigReader as cr

    # Load the configuration file
    config = cr.parse(".config")

    # Test open the file
    print(readStarCatalog(config.star_catalog_path, config.star_catalog_file, \
        mag_band_ratios=config.star_catalog_band_ratios))
