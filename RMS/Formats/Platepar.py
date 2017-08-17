""" This module contains functions for parsing CMN format data.
"""

# The MIT License

# Copyright (c) 2016 Denis Vida

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from __future__ import print_function, division, absolute_import

import os
import datetime
import numpy as np

from RMS.Astrometry.Conversions import date2JD, jd2Date

class stationData(object):
    """ Holds information about one meteor station (location) and observed points.
    """

    def __init__(self, file_name):
        self.file_name = file_name
        self.station_code = ''
        self.lon = 0
        self.lat = 0
        self.h = 0
        self.points = []

    def __str__(self):
        return 'Station: ' + self.station_code + ' data points: ' + str(len(self.points))


def parseInf(file_name):
    """ Parse information from an INF file to a stationData object.
    """

    station_data_obj = stationData(file_name)

    with open(file_name) as f:
        for line in f.readlines()[2:]:
            
            line = line.split()

            if 'Station_Code' in line[0]:
                station_data_obj.station_code = line[1]

            elif 'Long' in line[0]:
                station_data_obj.lon = float(line[1])

            elif 'Lati' in line[0]:
                station_data_obj.lat = float(line[1])

            elif 'Height' in line[0]:
                station_data_obj.h = int(line[1])

            else:
                station_data_obj.points.append(map(float, line))


    return station_data_obj


class PlateparCMN(object):
    """ Load calibration parameters from a platepar file.
    """

    def __init__(self):
        """ Read platepar and return object to access the data externally. 

        @param file_name: [string] path to the platepar file

        @return self: [object] instance of this class with loaded platepar parameters
        """

        # Station coordinates
        self.lat = self.lon = self.elev = 0

        # Referent time and date
        self.time = 0
        self.JD = 0

        self.Ho = 0
        self.X_res = self.Y_res = self.focal_length = 0

        # FOV centre
        self.RA_d = self.RA_H = self.RA_M = self.RA_S = 0
        self.dec_d = self.dec_D = self.dec_M = self.dec_S = 0
        self.pos_angle_ref = 0

        self.az_centre = self.alt_centre = 0

        # FOV scale
        self.F_scale = 0
        self.w_pix = 0

        self.mag_0 = self.mag_lev = 0

        # Distortion fit
        self.x_poly = np.zeros(shape=(12,), dtype=np.float64)
        self.y_poly = np.zeros(shape=(12,), dtype=np.float64)

        self.station_code = None




    def parseLine(self, f):
        """ Read next line, split the line and convert parameters to float.

        @param f: [file handle] file we want to read

        @return (a1, a2, ...): [tuple of floats] parsed data from the line
        """

        return map(float, f.readline().split())



    def read(self, file_name):
        """ Read the platepar. """


        # Check if platepar exists
        if not os.path.isfile(file_name):
            return False

        with open(file_name) as f:

            # Parse latitude, longitude, elevation
            self.lon, self.lat, self.elev = self.parseLine(f)

            # Parse date and time as int
            D, M, Y, h, m, s = map(int, f.readline().split())

            # Calculate the datetime of the platepar time
            self.time = datetime.datetime(Y, M, D, h, m, s)

            # Convert time to JD
            self.JD = date2JD(Y, M, D, h, m, s)

            # Calculate the referent hour angle
            T = (self.JD - 2451545.0)/36525.0
            self.Ho = (280.46061837 + 360.98564736629*(self.JD - 2451545.0) + 0.000387933*T**2 - 
                T**3/38710000.0)%360

            # Parse camera parameters
            self.X_res, self.Y_res, self.focal_length = self.parseLine(f)

            # Parse the right ascension of the image centre
            self.RA_d, self.RA_H, self.RA_M, self.RA_S = self.parseLine(f)

            # Parse the declination of the image centre
            self.dec_d, self.dec_D, self.dec_M, self.dec_S = self.parseLine(f)

            # Parse the rotation parameter
            self.pos_angle_ref = self.parseLine(f)[0]

            # Parse the sum of image scales per each image axis (arcsec per px)
            self.F_scale = self.parseLine(f)[0]
            self.w_pix = 50*self.F_scale/3600
            self.F_scale = 3600/self.F_scale

            # Load magnitude slope parameters
            self.mag_0, self.mag_lev = self.parseLine(f)

            # Load X axis polynomial parameters
            self.x_poly = np.zeros(shape=(12,), dtype=np.float64)
            for i in range(12):
                self.x_poly[i] = self.parseLine(f)[0]

            # Load Y axis polynomial parameters
            self.y_poly = np.zeros(shape=(12,), dtype=np.float64)
            for i in range(12):
                self.y_poly[i] = self.parseLine(f)[0]

            # Read station code
            self.station_code = f.readline().replace('\r', '').replace('\n', '')


    def write(self, file_name):
        """ Write platepar to file. """

        with open(file_name, 'w') as f:
            
            # Write geo coords
            f.write('{:9.6f} {:9.6f} {:04d}\n'.format(self.lon, self.lat, int(self.elev)))

            # Calculate referent time from referent JD
            Y, M, D, h, m, s, ms = list(map(int, jd2Date(self.JD)))

            # Write the referent time
            f.write('{:02d} {:02d} {:04d} {:02d} {:02d} {:02d}\n'.format(D, M, Y, h, m, s))

            # Write resolution and focal length
            f.write('{:d} {:d} {:f}\n'.format(int(self.X_res), int(self.Y_res), self.focal_length))

            # Write referent RA
            self.RA_H = int(self.RA_d/15)
            self.RA_M = int((self.RA_d/15 - self.RA_H)*60)
            self.RA_S = int(((self.RA_d/15 - self.RA_H)*60 - self.RA_M)*60)

            f.write("{:7.3f} {:02d} {:02d} {:02d}\n".format(self.RA_d, self.RA_H, self.RA_M, self.RA_S))

            # Write referent Dec
            self.dec_D = int(self.dec_d)
            self.dec_M = int((self.dec_d - self.dec_D)*60)
            self.dec_S = int(((self.dec_d - self.dec_D)*60 - self.dec_M)*60)

            f.write("{:+7.3f} {:02d} {:02d} {:02d}\n".format(self.dec_d, self.dec_D, self.dec_M, self.dec_S))

            # Write rotation parameter
            f.write('{:<7.3f}\n'.format(self.pos_angle_ref))

            # Write F scale
            f.write('{:<5.1f}\n'.format(3600/self.F_scale))

            # Write magnitude fit
            f.write("{:.3f} {:.3f}\n".format(self.mag_0, self.mag_lev))

            # Write X distorsion polynomial
            for x_elem in self.x_poly:
                f.write('{:+E}\n'.format(x_elem))

            # Write y distorsion polynomial
            for y_elem in self.y_poly:
                f.write('{:+E}\n'.format(y_elem))

            # Write station code
            f.write(str(self.station_code) + '\n')









    

        