""" CMN-style astrometric calibration.
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
import json
import copy
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



class Platepar(object):
    """ Load calibration parameters from a platepar file.
    """

    def __init__(self):
        """ Read platepar and return object to access the data externally. 
    
        Arguments:
            file_name: [string] Path to the platepar file.
    
        Return:
            self: [object] Instance of this class with loaded platepar parameters.

        """

        # Station coordinates
        self.lat = self.lon = self.elev = 0

        # Referent time and date
        self.time = 0
        self.JD = 0

        # UT correction
        self.UT_corr = 0

        self.Ho = 0
        self.X_res = 0
        self.Y_res = 0
        self.focal_length = 0

        # FOV centre
        self.RA_d = self.RA_H = self.RA_M = self.RA_S = 0
        self.dec_d = self.dec_D = self.dec_M = self.dec_S = 0
        self.pos_angle_ref = 0

        self.az_centre = 0
        self.alt_centre = 0

        # FOV scale
        self.F_scale = 1.0
        self.w_pix = 0

        # Photometry calibration
        self.mag_0 = -2.5
        self.mag_lev = 1.0
        self.mag_lev_stddev = 0.0

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



    def read(self, file_name, fmt=None):
        """ Read the platepar. 
            
        Arguments:
            file_name: [str] Path and the name of the platepar to read.

        Keyword arguments:
            fmt: [str] Format of the platepar file. 'json' for JSON format and 'txt' for the usual CMN textual
                format.

        Return:
            fmt: [str]
        """


        # Check if platepar exists
        if not os.path.isfile(file_name):
            return False


        # Determine the type of the platepar if it is not given
        if fmt is None:

            with open(file_name) as f:
                data = " ".join(f.readlines())

                # Try parsing the file as JSON
                try:
                    json.loads(data)
                    fmt = 'json'

                except:
                    fmt = 'txt'


        # Load the file as JSON
        if fmt == 'json':

            # Load the JSON file
            with open(file_name) as f:
                data = " ".join(f.readlines())

            # Parse JSON into an object with attributes corresponding to dict keys
            self.__dict__ = json.loads(data)

            # Add UT correction if it was not in the platepar
            if not 'UT_corr' in self.__dict__:
                self.UT_corr = 0

            # Convert lists to numpy arrays
            self.x_poly = np.array(self.x_poly)
            self.y_poly = np.array(self.y_poly)

            # Calculate the datetime
            self.time = jd2Date(self.JD, dt_obj=True)


        # Load the file as TXT
        else:

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


        return fmt



    def write(self, file_path, fmt=None):
        """ Write platepar to file. 
        
        Arguments:
            file_path: [str] Path and the name of the platepar to write.

        Keyword arguments:
            fmt: [str] Format of the platepar file. 'json' for JSON format and 'txt' for the usual CMN textual
                format. The format is JSON by default.

        Return:
            fmt: [str]

        """

        # Set JSON to be the defualt format
        if fmt is None:
            fmt = 'json'


        # If the format is JSON, write a JSON file
        if fmt == 'json':

            # Make a copy of the platepar object, which will be modified for writing
            self2 = copy.deepcopy(self)

            # Convert numpy arrays to list, which can be serialized
            self2.x_poly = self.x_poly.tolist()
            self2.y_poly = self.y_poly.tolist()
            del self2.time

            out_str = json.dumps(self2, default=lambda o: o.__dict__, indent=4, sort_keys=True)

            with open(file_path, 'w') as f:
                f.write(out_str)

        else:

            with open(file_path, 'w') as f:
                
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


        return fmt




if __name__ == "__main__":


    # Platepar file
    pp_file = "/home/dvida/Desktop/platepar_cmn2010.cal"

    pp = Platepar()
    pp.read(pp_file)

    print(pp.x_poly)
    print(pp.station_code)

    #txt = json.dumps(pp, default=lambda o: o.__dict__)

    pp.write(pp_file+'.json')