""" Writing meteor point observations in the GDEF ECSV standard.

This produces the same point-observation ECSV format that SkyFit2 and ASTRA read/write
(see Utils/SkyFit2.py saveECSV and Utils/Astra.py loadECSV), so RMS output can be consumed
directly by the ASTRA refinement pipeline.
"""

from __future__ import print_function, division, absolute_import

import os

import numpy as np


# Datetime format used in the data rows (microsecond precision). ASTRA's reader parses this exact format.
ECSV_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"


def writeECSV(dir_path, file_name, station_id, platepar, picks, mag_band_string='V', n_stars=0,
    image_file='', isodate_start_obs=''):
    """ Write a list of calibrated meteor picks into a GDEF ECSV file.

    Arguments:
        dir_path: [str] Directory where the file will be written.
        file_name: [str] Name of the ECSV file.
        station_id: [str] Station/camera code.
        platepar: [Platepar instance] Used for the meta header (station location, FOV pointing/size,
            resolution) and the photometric magnitude uncertainty (mag_lev_stddev).
        picks: [list] A list of per-pick dicts with the keys:
            datetime [datetime] - UTC time of the pick (full microsecond precision),
            ra, dec, azim, alt [float] - calibrated celestial/horizontal coordinates (deg),
            x, y [float] - image coordinates (pix),
            intensity [float] - integrated (background-subtracted) pixel value,
            background [float] - background pixel value,
            saturated [bool] - whether the centroid contains saturated pixels,
            mag [float] - apparent magnitude,
            snr [float] - signal-to-noise ratio.
            Picks with NaN x/y coordinates are skipped.

    Keyword arguments:
        mag_band_string: [str] Photometric band of the star catalogue. 'V' by default.
        n_stars: [int] Number of stars used in the astrometric calibration. 0 by default.
        image_file: [str] Name of the original image/video (e.g. the FF file name). Empty by default.
        isodate_start_obs: [str] ISO date/time of the start of the observation. Empty by default.

    Return:
        [str] Full path to the written ECSV file.
    """

    # Reuse the existing astrometry helpers for the FOV pointing and size meta fields
    from RMS.Astrometry.ApplyAstrometry import computeFOVSize, rotationWrtHorizon
    from RMS.Astrometry.Conversions import trueRaDec2ApparentAltAz

    # Compute the alt/az pointing of the FOV centre (topocentric, i.e. without refraction)
    azim_centre, elev_centre = trueRaDec2ApparentAltAz(platepar.RA_d, platepar.dec_d, platepar.JD,
        platepar.lat, platepar.lon, refraction=False)

    # Compute the FOV size
    fov_horiz, fov_vert = computeFOVSize(platepar)

    # Assemble the meta header (order preserved, written as an ordered map)
    meta_dict = {
        'obs_latitude': platepar.lat,                  # Decimal signed latitude (-90 S to +90 N)
        'obs_longitude': platepar.lon,                 # Decimal signed longitude (-180 W to +180 E)
        'obs_elevation': platepar.elev,                # Altitude in metres above MSL. Note not WGS84
        'origin': 'RMS',                               # The software which produced the data file
        'camera_id': station_id,                       # The code name of the camera
        'cx': platepar.X_res,                          # Horizontal camera resolution in pixels
        'cy': platepar.Y_res,                          # Vertical camera resolution in pixels
        'photometric_band': mag_band_string,           # The photometric band of the star catalogue
        'image_file': image_file,                      # The name of the original image or video
        'isodate_start_obs': isodate_start_obs,        # The date and time of the start of the observation
        'astrometry_number_stars': n_stars,            # Number of stars used in the astrometric calibration
        'mag_label': 'mag_data',                       # The label of the Magnitude column
        'no_frags': 1,                                 # The number of meteoroid fragments
        'obs_az': azim_centre,                         # Azimuth of the FOV centre (deg, North=0, E positive)
        'obs_ev': elev_centre,                         # Elevation of the FOV centre (deg, Horizon=0, Zenith=90)
        'obs_rot': rotationWrtHorizon(platepar),       # Rotation of the FOV from horizontal (deg, CW positive)
        'fov_horiz': fov_horiz,                        # Horizontal extent of the FOV (deg)
        'fov_vert': fov_vert,                          # Vertical extent of the FOV (deg)
    }

    # Write the fixed ECSV header
    out_str = """# %ECSV 0.9
# ---
# datatype:
# - {name: datetime, datatype: string}
# - {name: ra, unit: deg, datatype: float64}
# - {name: dec, unit: deg, datatype: float64}
# - {name: azimuth, datatype: float64}
# - {name: altitude, datatype: float64}
# - {name: x_image, unit: pix, datatype: float64}
# - {name: y_image, unit: pix, datatype: float64}
# - {name: integrated_pixel_value, datatype: int64}
# - {name: background_pixel_value, datatype: int64}
# - {name: saturated_pixels, datatype: bool}
# - {name: mag_data, datatype: float64}
# - {name: err_minus_mag, datatype: float64}
# - {name: err_plus_mag, datatype: float64}
# - {name: snr, datatype: float64}
# delimiter: ','
# meta: !!omap
"""

    # Add the meta information
    for key in meta_dict:

        value = meta_dict[key]

        if isinstance(value, str):
            value_str = "'{:s}'".format(value)
        else:
            value_str = str(value)

        out_str += "# - {" + "{:s}: {:s}".format(key, value_str) + "}\n"

    out_str += "# schema: astropy-2.0\n"
    out_str += "datetime,ra,dec,azimuth,altitude,x_image,y_image,integrated_pixel_value," \
        "background_pixel_value,saturated_pixels,mag_data,err_minus_mag,err_plus_mag,snr\n"

    # Add the data rows
    for pick in picks:

        # Skip picks without a valid centroid
        if np.isnan(pick['x']) or np.isnan(pick['y']):
            continue

        # Read the SNR and compute the magnitude uncertainty
        snr = pick['snr']
        if (snr is None) or np.isnan(snr) or (snr <= 0):

            # If the SNR is unusable, set it and the random magnitude error to zero
            snr = 0.0
            mag_err_random = 0.0

        else:

            # Compute the random error based on the SNR
            mag_err_random = 2.5*np.log10(1 + 1/snr)

        # Combine the random and systematic photometric errors
        mag_err_total = np.sqrt(mag_err_random**2 + platepar.mag_lev_stddev**2)

        entry = [
            pick['datetime'].strftime(ECSV_DATETIME_FORMAT),
            "{:10.6f}".format(pick['ra']), "{:+10.6f}".format(pick['dec']),
            "{:10.6f}".format(pick['azim']), "{:+10.6f}".format(pick['alt']),
            "{:9.3f}".format(pick['x']), "{:9.3f}".format(pick['y']),
            "{:10d}".format(int(pick['intensity'])),
            "{:10d}".format(int(pick['background'])),
            "{:5s}".format(str(bool(pick['saturated']))),
            "{:+7.2f}".format(pick['mag']), "{:+6.2f}".format(-mag_err_total), "{:+6.2f}".format(mag_err_total),
            "{:10.2f}".format(snr),
        ]

        out_str += ",".join(entry) + "\n"

    # Write the file to disk
    ecsv_file_path = os.path.join(dir_path, file_name)
    with open(ecsv_file_path, 'w') as f:
        f.write(out_str)

    return ecsv_file_path
