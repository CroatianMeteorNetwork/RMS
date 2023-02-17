""" UWO ASGARD event file format. """

from __future__ import print_function, division, absolute_import


import os
import datetime

import numpy as np

from RMS.Astrometry.Conversions import jd2UnixTime, jd2Date


class AsgardEv(object):
    def __init__(self):

        self.version = ''
        self.num_fr = 0
        self.num_tr = 0
        self.time = ''
        self.unix = 0.0
        self.ntp = ''
        self.seq = 0
        self.mul = ''
        self.site = ''
        self.latlon = []
        self.text = ''
        self.stream = ''
        self.plate = ''
        self.geom = []
        self.filter = 0

        self.data = None


    def __repr__(self):
        """ Returns a string formatted as the ASGARD ev file. """

        out_list = []

        out_list.append(['version', self.version])
        out_list.append(['num_fr', str(self.num_fr)])
        out_list.append(['num_tr', str(self.num_tr)])
        out_list.append(['time', self.time.strftime("%Y%m%d %H:%M:%S.%f")[:-3] + " UTC"])
        out_list.append(['unix', "{:.6f}".format(self.unix)])
        out_list.append(['ntp', self.ntp])
        out_list.append(['seq', str(self.seq)])
        out_list.append(['mul', self.mul])
        out_list.append(['site', str(self.site)])
        out_list.append(['latlon', "{:.6f} {:.6f} {:.1f}".format(*self.latlon)])
        out_list.append(['text', self.text])
        out_list.append(['stream', self.stream])
        out_list.append(['plate', self.plate])
        out_list.append(['geom', "{:d} {:d}".format(*self.geom)])
        out_list.append(['filter', str(self.filter)])

        out_str =  "#\n"

        for key, value in out_list:
            out_str += "# {:>9s} : {:s}\n".format(key, value)

        out_str += "#\n"
        out_str += "#  fr       time    sum     seq          cx          cy         th         phi     lsp    mag  flag   bak    max\n"
        
        fr_arr, time_arr, sum_arr, seq_arr, cx_arr, cy_arr, th_arr, phi_arr, lsp_arr, mag_arr, flag_arr, \
            bak_arr, max_arr = self.data.T

        fr_arr = fr_arr.astype(int)
        sum_arr = sum_arr.astype(int)
        seq_arr = seq_arr.astype(int)
        flag_arr = flag_arr.astype(int)

        for i in range(len(self.data)):
            out_str += "{:5d} {:10.6f} {:6d} {:7d} {:11.6f} {:11.6f} {:10.6f} {:11.6f} {:7.3f} {:6.2f}  {:04d} {:5.1f} {:6.1f}\n".format(\
                fr_arr[i], time_arr[i], sum_arr[i], seq_arr[i], cx_arr[i], cy_arr[i], th_arr[i], \
                phi_arr[i], lsp_arr[i], mag_arr[i], flag_arr[i], bak_arr[i], max_arr[i])


        return out_str


    def write(self, file_path):
        """ Write the ASGARD ev object to ev file. 
        
        Arguments:
            file_path: [str] Path to the file which will be saved.

        """

        with open(file_path, 'w') as f:

            for line in self.__repr__():
                f.write(line)


    def read(self, file_path):
        """ Read an ASGARD ev file. 
    
        Arguments:
            file_path: [str] Path to the ASGARD ev file.
        """

        readEv(*os.path.split(file_path), ev=self)

        



def readEv(dir_path, file_name, ev=None):
    """ Read the UWO ASGARD style event file. 
    
    Arguments:
        dir_path: [str] Directory where the file is.
        file_name: [str] Name of the event file.

    Keyword arguments:
        ev: [AsgardEv instance] AsgardEv object to use. None by default, in which case a new object will be
            inited.

    Return:
        ev: [AsgardEv instance] Instance of the AsgardEv object filled with data read from the file.

    """

    with open(os.path.join(dir_path, file_name)) as f:

        if ev is None:
            
            # Init the new ev structure
            ev = AsgardEv()


        data_list = []

        for line in f:

            line = line.replace('\n', '').replace('\r', '')

            # Read the header
            if line.startswith('#'):

                line = line[1:]

                line = line.split(':', 1)

                if len(line) != 2:
                    continue

                key, value = line
                key = key.strip()
                value = value.strip()

                if key == 'version':
                    ev.version = value

                elif key == 'num_fr':
                    ev.num_fr = int(value)

                elif key == 'num_tr':
                    ev.num_tr = int(value)

                elif key == 'time':
                    ev.time = datetime.datetime.strptime(value, "%Y%m%d %H:%M:%S.%f UTC")

                elif key == 'unix':
                    ev.unix = float(value)

                elif key == 'ntp':
                    ev.ntp = value

                elif key == 'seq':
                    ev.seq = int(value)

                elif key == 'mul':
                    ev.mul = value

                elif key == 'site':
                    ev.site = value

                elif key == 'latlon':
                    ev.latlon = list(map(float, value.split()))

                elif key == 'text':
                    ev.text = value

                elif key == 'stream':
                    ev.stream = value

                elif key == 'plate':
                    ev.plate = value

                elif key == 'geom':
                    ev.geom = list(map(int, value.split()))

                elif key == 'filter':
                    ev.filter = value

            else:
                data_list.append(line.split())


        ev.data = np.array(data_list).astype(np.float64)

        return ev



def writeEv(dir_path, file_name, ev_array, plate, multi, ast_input=False, vidinfo=None):
    """ Write an UWO ASGARD style event file. 
    
    Arguments:
        dir_path: [str] Path to directory where the file will be saved to.
        file_name: [str] Name of the ev file.
        ev_array: [ndarray] Array where columns are: frame number, sequence number, JD, intensity, x, y, 
            azimuth (deg), altitude (deg), magnitude
        plate: [?] Platepar or AST plate.
        multi: identifier for simultaneous detections, 0 = 'A', 1 = 'B', etc.

    Keyword arguments:
        ast_input: [bool] True if AST plate if given, False if platepar is given (default).
        vidinfo:   [?] metadata from a UWO .vid file, if available

    """

    # ASGARD default site if we can't get a matching site number from another source
    site = 0
    stream = 'Z'

    # AST plate used for input
    if ast_input:
        # AST files don't populate the 'st' field so we can't get site info from here
        # station_code = plate.sitename
        lat = np.degrees(plate.lat)
        lon = np.degrees(plate.lon)
        elev = plate.elev
        X_res = plate.wid
        Y_res = plate.ht
        text = plate.sitename
        plate_text = plate.text

    # Platepar used for input
    else:
        station_code = plate.station_code
        lat = plate.lat
        lon = plate.lon
        elev = plate.elev
        X_res = plate.X_res
        Y_res = plate.Y_res
        text = ''
        plate_text = 'RMS_SkyFit'

        # valid  ASGARD site ids can only be exactly two digits followed by one letter (ex. 02A)
        # if the first two characters are digits, use that as the site number
        # if the third character is a letter, use that as the stream, otherwise fall back on 'A'
        # if the station code doesn't match at all, use the defaults
        if len(station_code) == 3:
            if station_code[:2].isdigit() and station_code[2].isalpha():
                site = station_code[:2]
                stream = station_code[2].upper()

        elif len(station_code) == 2:
            if station_code[:2].isdigit():
                site = station_code[:2]
                stream = 'A'


    # If a .vid file was used and we have vidinfo, prefer site info and descriptive text from
    # the .vid file itself
    if vidinfo is not None:
        site = vidinfo.station_id
        stream = chr(ord('A') + vidinfo.str_num)
        text = vidinfo.text

    with open(os.path.join(dir_path, file_name), 'w') as f:


        frame_array, seq_array, jd_array, intensity_array, x_array, y_array, azim_array, alt_array, \
            mag_array = ev_array.T

        seq_array = seq_array.astype(np.uint32)

        # Get the Julian date of the peak
        jd_peak = jd_array[mag_array.argmin()]

        # Get the sequence number of the peak
        seq_peak = int(seq_array[mag_array.argmin()])


        ### Write the header

        f.write('#\n')
        f.write('#   version : RMS_Detection\n')
        f.write("#    num_fr : {:d}\n".format(len(ev_array)))
        f.write("#    num_tr : 0\n")
        f.write("#      time : {:s} UTC\n".format(jd2Date(jd_peak, dt_obj=True).strftime('%Y%m%d %H:%M:%S.%f')[:-3]))
        f.write("#      unix : {:.6f}\n".format(jd2UnixTime(jd_peak)))
        f.write("#       ntp : LOCK 0 0 0\n")
        f.write("#       seq : {:d}\n".format(seq_peak))
        f.write("#       mul : {:d} [{:c}]\n".format(multi, ord('A') + multi))
        f.write("#      site : {:02d}\n".format(site))
        f.write("#    latlon : {:.4f} {:.4f} {:.1f}\n".format(lat, lon, elev))
        f.write("#      text : {:s}\n".format(text))
        f.write("#    stream : {:s}\n".format(stream))
        f.write("#     plate : {:s}\n".format(plate_text))
        f.write("#      geom : {:d} {:d}\n".format(X_res, Y_res))
        f.write("#    filter : 0\n")
        f.write("#\n")
        f.write("#  fr       time    sum     seq          cx          cy         th         phi     lsp    mag  flag   bak    max\n")


        ###

        # Go through all centroids and write them to file
        for i, entry in enumerate(ev_array):

            frame, seq_num, jd, intensity, x, y, azim, alt, mag = entry

            # Compute the relative time in seconds
            t_rel = (jd - jd_peak)*86400

            # Compute theta and phi
            theta = 90 - alt
            phi = (90 - azim)%360

            f.write("{:5d} {:10.6f} {:6d} {:7d} {:11.6f} {:11.6f} {:10.6f} {:11.6f} {:7.3f} {:6.2f}  0000   0.0    0.0\n".format(int(31 + int(seq_num) - seq_array[0]), \
                t_rel, int(intensity), int(seq_num), x, y, theta, phi, -2.5*np.log10(intensity), mag))



def batchRecomputeMagnitudes(dir_path, photom_offset, site=None):
    """ Recompute the magnitudes in all ev files in the given directory. The files will be read in, the 
        magnitudes recomputed, and the files will be saved back.

    Arguments:
        dir_path: [str] Path to the directory with ev files.
        photom_offset: [float] The new photometric offset.

    Keyword arguments:
        site: [str] Recompute the magnitudes only on files from a given site, e.g. 02F. None by default,
            in which case all ev files will be used.

    """

    for file_name in os.listdir(dir_path):

        if file_name.startswith('ev_') and file_name.endswith('.txt'):

            if site is not None:
                tmp_name = file_name.strip('.txt')
                if not tmp_name.endswith(site):
                    print('Skipping:', file_name)
                    continue


            print('Fixing magnitudes:', file_name)

            # Read the ev file
            ev = readEv(dir_path, file_name)

            # Recompute the magnitudes
            lsp_arr = ev.data[:, 8]
            mag_arr = lsp_arr + photom_offset

            # Save the new magnitudes
            ev.data[:, 9] = mag_arr

            # Save the ev file
            ev.write(os.path.join(dir_path, file_name))




if __name__ == "__main__":

    dir_path = '/mnt/bulk/2018Perseids/ev_files'

    print('Recomputing magnitudes...')
    batchRecomputeMagnitudes(dir_path, 16.89, site='01F')
    batchRecomputeMagnitudes(dir_path, 16.48, site='01G')
    batchRecomputeMagnitudes(dir_path, 16.05, site='02F')
    batchRecomputeMagnitudes(dir_path, 16.32, site='02G')