""" Convert UWO-style event files to FTPdetectinfo files."""

import os
import glob
import datetime

import numpy as np
import scipy.interpolate

from RMS.Astrometry.Conversions import raDec2AltAz, altAz2RADec, unixTime2JD, jd2Date
from RMS.Formats.FTPdetectinfo import writeFTPdetectinfo


def readEvFile(dir_path, file_name):
    """ Given the path of the UWO-style event file, read it into the StationData object. 

    Arguments:
        dir_path: [str] Path to the directory with the ev file.
        file_name: [str] Name of the ev file.

    Return:
        [StationData instance]
    """


    with open(os.path.join(dir_path, file_name)) as f:

        lat = None
        lon = None
        elev = None
        site = None
        stream = None


        time_data = []
        level_data = []
        x_data = []
        y_data = []
        theta_data = []
        phi_data = []
        mag_data = []

        for line in f:

            if not line:
                continue

            # Read metadata
            if line.startswith("#"):

                entry = line[1:].split()

                if not entry:
                    continue

                # Read reference time
                if entry[0] == "unix":
                    ts, tu = list(map(int, entry[2].split(".")))

                elif entry[0] == "site":
                    site = entry[2]

                elif entry[0] == "latlon":
                    lat, lon, elev = list(map(float, entry[2:5]))

                elif entry[0] == "stream":
                    stream = entry[2]


            # Read data
            else:

                line = line.split()

                time_data.append(float(line[1]))
                level_data.append(int(line[2]))
                x_data.append(float(line[4]))
                y_data.append(float(line[5]))
                theta_data.append(float(line[6]))
                phi_data.append(float(line[7]))

                # Check if the magnitude is NaN and set None instead
                mag = line[9]
                if 'nan' in mag:
                    mag = None
                else:
                    mag = float(mag)

                mag_data.append(mag)


        # If there is a NaN in the magnitude data, interpolate it
        if None in mag_data:

            # Get a list of clean data
            mag_data_clean = [entry for entry in enumerate(mag_data) if entry[1] is not None]
            clean_indices, clean_mags = np.array(mag_data_clean).T

            # If there aren't at least 2 good points, return None
            if len(clean_indices) < 2:
                return None

            # Interpolate in linear units
            intens_interpol = scipy.interpolate.PchipInterpolator(clean_indices, 10**(clean_mags/(-2.5)))


            # Interpolate missing magnitudes
            for i, mag in enumerate(mag_data):

                # Don't interpolate at the edges if there are NaNs
                if (i < np.min(clean_indices)) or (i > np.max(clean_indices)):
                    mag_data[i] = np.nan
                    continue

                if mag is None:
                    mag_data[i] = -2.5*np.log10(intens_interpol(i))
        

        # Change the relative time to 0 and update the reference Julian date
        time_data = np.array(time_data)
        time_data -= time_data[0]

        # Compute the julian date
        jd_ref = unixTime2JD(ts, tu)

        # Convert theta and phi to RA/Dec
        ra_data, dec_data = altAz2RADec(90 - np.array(phi_data), 90 - np.array(theta_data), jd_ref, lat, lon)

        return jd_ref, lat, lon, elev, site, stream, time_data, x_data, y_data, ra_data, dec_data, level_data, mag_data



if __name__ == "__main__":

    import argparse

    argp = argparse.ArgumentParser(description="Convert UWO-style event files to FTP-style detectinfo files.")

    argp.add_argument("ev_files", help="Path to the UWO-style event files.", type=str, nargs="+")
    argp.add_argument("station_code", help="Station code of the station.", type=str)
    argp.add_argument("fps", help="Frames per second of the video.", type=float)

    args = argp.parse_args()


    # Unpack wildcards, e.g. /path/to/ev/files/ev_*.txt
    ev_files = []
    for ev_file in args.ev_files:
        ev_files.extend(glob.glob(ev_file))

    args.ev_files = ev_files


    # Loop over the event files
    meteor_list = []
    for ev_file in args.ev_files:

        # Read the Ev file
        (
            jd_ref, lat, lon, elev, site, stream, 
            time_data, x_data, y_data, ra_data, dec_data, 
            level_data, mag_data    

        ) = readEvFile(*os.path.split(ev_file))
        
        meteor_dt = jd2Date(jd_ref, dt_obj=True)

        # Make a name for the FF file
        ff_name = "FF_{:>06s}_{:s}_{:>03d}_0000000.txt".format(
            args.station_code, meteor_dt.strftime("%Y%m%d_%H%M%S"), int(meteor_dt.strftime("%f"))//1000)

        # Add meteor centroids
        centroids = []
        t0 = time_data[0]
        for i, (t, x, y, ra, dec, lvl, mag) in enumerate(zip(time_data, x_data, y_data, ra_data, dec_data, level_data, mag_data)):

            # Compute frame number
            frame = (t - t0)*args.fps

            # Compute azimuth and elevation
            azim, elev = raDec2AltAz(ra, dec, jd_ref, lat, lon)

            centroids.append([frame, x, y, ra, dec, azim, elev, lvl, mag])

        meteor_list.append([ff_name, i + 1, 0, 0, centroids])


    # Make an FTPdetectinfo file name
    ftpdetectinfo_file_name = "FTPdetectinfo_{:>06s}_{:s}.txt".format(args.station_code, os.path.basename(ev_file).split(".")[0])

    # Write an FTPdetectinfo file
    writeFTPdetectinfo(meteor_list, os.path.dirname(ev_file), ftpdetectinfo_file_name, 
                       os.path.dirname(ev_file), args.station_code, args.fps, celestial_coords_given=True)