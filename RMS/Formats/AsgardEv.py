""" UWO ASGARD event file format. """

from __future__ import print_function, division, absolute_import


import os

import numpy as np

from RMS.Astrometry.Conversions import jd2UnixTime, jd2Date


def writeEv(dir_path, file_name, ev_array, platepar):
    """ Write an UWO ASGARD style event file. """


    with open(os.path.join(dir_path, file_name), 'w') as f:


        frame_array, jd_array, intensity_array, x_array, y_array, azim_array, alt_array, \
            mag_array = ev_array.T

        # Get the Julian date of the peak
        jd_peak = jd_array[mag_array.argmin()]

        # Get the frame of the peak
        frame_peak = int(frame_array[mag_array.argmin()])


        # Extract the site number and stream
        if len(platepar.station_code) == 3:
            site = platepar.station_code[:2]
            stream = platepar.station_code[2]

        else:
            site = platepar.station_code
            stream = 'A'

        ### Write the header

        f.write('#\n')
        f.write('#   version : RMS Detection\n')
        f.write("#    num_fr : {:d}\n".format(len(ev_array)))
        f.write("#    num_tr : 0\n")
        f.write("#      time : {:s} UTC\n".format(jd2Date(jd_peak, dt_obj=True).strftime('%Y%m%d %H:%M:%S.%f')[:-3]))
        f.write("#      unix : {:.6f}\n".format(jd2UnixTime(jd_peak)))
        f.write("#       ntp : LOCK 0 0 0\n")
        f.write("#       seq : {:d}\n".format(frame_peak))
        f.write("#       mul : 0 [A]\n")
        f.write("#      site : {:s}\n".format(site))
        f.write("#    latlon : {:.4f} {:.4f} {:.1f}\n".format(platepar.lat, platepar.lon, platepar.elev))
        f.write("#      text : \n")
        f.write("#    stream : {:s}\n".format(stream))
        f.write("#     plate : RMS_SkyFit\n")
        f.write("#      geom : {:d} {:d}\n".format(platepar.X_res, platepar.Y_res))
        f.write("#    filter : 0\n")
        f.write("#\n")
        f.write("#  fr    time    sum     seq       cx       cy      th      phi     lsp    mag  flag   bak    max\n")


        ###

        # Go through all centroids and write them to file
        for i, entry in enumerate(ev_array):

            frame, jd, intensity, x, y, azim, alt, mag = entry

            # Compute the relative time in seconds
            t_rel = (jd - jd_peak)*86400

            # Compute theta and phi
            theta = 90 - alt
            phi = (90 - azim)%360

            f.write("{:5d} {:7.3f} {:6d} {:7d} {:8.3f} {:8.3f} {:7.3f} {:8.3f} {:7.3f} {:6.2f}  0000   0.0    0.0\n".format(31 + i, \
                t_rel, int(intensity), int(frame), x, y, theta, phi, -2.5*np.log10(intensity), mag))
