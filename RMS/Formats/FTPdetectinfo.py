# RPi Meteor Station
# Copyright (C) 2016  Denis Vida
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function, division, absolute_import

import os
import datetime
import git

import numpy as np


def writeFTPdetectinfo(meteor_list, ff_directory, file_name, cal_directory, cam_code, fps, calibration=None,
    celestial_coords_given=False):
    """ Writes a FTPdetectinfo file from the list of detected meteors. 
    
    Arguments:
        meteor_list: [list] a list of meteor data, entries: 
            ff_name, meteor_No, rho, theta, centroids
        ff_directory: [str] path to the directory in which the file will be written
        file_name: [str] file name of the file in which the data will be written
        cal_directory: [str] path to the CAL directory (optional, used only in CAMS processing)
        cam_code: [str] camera code
        fps: [float] frames per second of the camera

    Keyword arguments:
        calibration: [str] String to write when the data is calibrated. None by default, which will write 
            'Uncalibrated' in the file.
        celestial_coords_given: [bool] If True, meteor picks in meteor_list should contain (frame, x, y, ra, 
            dec, azim, elev, intens), if False it should contain (frame, x, y, intens).

    Return:
        None
    """


    # Open a file
    with open(os.path.join(ff_directory, file_name), 'w') as ftpdetect_file:

        try:
            # Get latest version's commit hash and time of commit
            repo = git.Repo(search_parent_directories=True)
            commit_unix_time = repo.head.object.committed_date
            sha = repo.head.object.hexsha
            commit_time = datetime.datetime.fromtimestamp(commit_unix_time).strftime('%Y%m%d_%H%M%S')

        except:
            commit_time = ""
            sha = ""

        # Write the number of meteors on the beginning fo the file
        total_meteors = len(meteor_list)
        ftpdetect_file.write("Meteor Count = "+str(total_meteors).zfill(6)+ "\n")
        ftpdetect_file.write("-----------------------------------------------------\n")
        ftpdetect_file.write("Processed with RMS " + commit_time + " " + str(sha) + " on " \
            + str(datetime.datetime.utcnow()) + " UTC\n")
        ftpdetect_file.write("-----------------------------------------------------\n")
        ftpdetect_file.write("FF  folder = " + ff_directory + "\n")
        ftpdetect_file.write("CAL folder = " + cal_directory + "\n")
        ftpdetect_file.write("-----------------------------------------------------\n")
        ftpdetect_file.write("FF  file processed\n")
        ftpdetect_file.write("CAL file processed\n")
        ftpdetect_file.write("Cam# Meteor# #Segments fps hnr mle bin Pix/fm Rho Phi\n")
        
        ftpdetect_file.write("Per segment:  Frame# Col Row RA Dec Azim Elev Inten Mag\n")

        # Write info for all meteors
        for meteor in meteor_list:

            # Unpack the meteor data
            ff_name, meteor_No, rho, theta, centroids = meteor

            ftpdetect_file.write("-------------------------------------------------------\n")
            ftpdetect_file.write(ff_name + "\n")
            if calibration is not None:
                ftpdetect_file.write(calibration + "\n")
            else:
                ftpdetect_file.write("Uncalibrated" + "\n")

            # Calculate meteor's angular velocity
            first_centroid = centroids[0]
            last_centroid  = centroids[-1]
            frame1, x1, y1 = first_centroid[:3]
            frame2, x2, y2 = last_centroid[:3]

            ang_vel = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)/float(frame2 - frame1)

            # Write detection header
            ftpdetect_file.write(str(cam_code).zfill(4) + " " + str(int(meteor_No)).zfill(4) + " " + 
                str(int(len(centroids))).zfill(4) + " " + "{:07.2f}".format(round(float(fps), 2)) + 
                " 000.0 000.0  00.0 " + str(round(ang_vel, 1)).zfill(5) + " " + 
                "{:06.1f} {:06.1f}".format(round(rho, 1), round(theta, 1)) + "\n")

            # Write individual detection points
            for line in centroids:

                if celestial_coords_given:

                    frame, x, y, ra, dec, azim, elev, level, mag = line

                    # If the coordinates or the magnitude are NaN, skip this centroid
                    if np.isnan(x) or np.isnan(y) or np.isnan(mag):
                        continue

                    detection_line_str = "{:06.4f} {:07.2f} {:07.2f} {:08.4f} {:+08.4f} {:08.4f} {:+08.4f} {:06d} {:.2f}"

                    ftpdetect_file.write(detection_line_str.format(round(frame, 4), round(x, 2), \
                        round(y, 2), round(ra, 4), round(dec, 4), round(azim, 4), round(elev, 4), \
                        int(level), round(mag, 2)) + "\n")

                else:
                    frame, x, y, level = line

                    # If the coordinates are NaN, skip this centroid
                    if np.isnan(x) or np.isnan(y):
                        continue

                    ftpdetect_file.write("{:06.4f} {:07.2f} {:07.2f}".format(round(frame, 4), round(x, 2), \
                        round(y, 2)) + " 000.00 000.00 000.00 000.00 " + "{:06d}".format(int(level)) \
                        + " 0.00\n")



def readFTPdetectinfo(ff_directory, file_name, ret_input_format=False):
    """ Read the CAMS format FTPdetectinfo file. 

    Arguments:
        ff_directory: [str] Directory where the FTPdetectinfo file is.
        file_name: [str] Name of the FTPdetectinfo file.

    Keyword arguments:
        ret_input_format: [bool] If True, the list that can be written back using writeFTPdetectinfo is 
            returned. False returnes the expanded list containing everyting that was read from the file (this
            is the default behavious, thus it's False by default)

    Return:
        [tuple]: Two options, see ret_input_format.
    """

    ff_name = ''


    # Open the FTPdetectinfo file
    with open(os.path.join(ff_directory, file_name)) as f:

        entry_counter = 0
        meteor_list = []
        meteor_meas = []
        cam_code = meteor_No = n_segments = fps = hnr = mle = binn = px_fm = rho = phi = None
        calib_status = 0

        # Skip the header
        for i in range(11):
            next(f)


        for line in f:
            
            line = line.replace('\n', '').replace('\r', '')


            # The separator marks the beginning of a new meteor
            if "-------------------------------------------------------" in line:

                # Add the read meteor info to the final list
                if meteor_meas:
                    meteor_list.append([ff_name, cam_code, meteor_No, n_segments, fps, hnr, mle, binn, \
                        px_fm, rho, phi, meteor_meas])

                # Reset the line counter to 0
                entry_counter = 0
                meteor_meas = []


            # Read the calibration status
            if entry_counter == 0:

                if 'Uncalibrated' in line:
                    calib_status = 0

                else:
                    calib_status = 1


            # Read the name of the FF file
            if entry_counter == 1:
                ff_name = line

            # Read the meteor parameters
            if entry_counter == 3:
                cam_code, meteor_No, n_segments, fps, hnr, mle, binn, px_fm, rho, phi = line.split()
                meteor_No, n_segments, fps, hnr, mle, binn, px_fm, rho, phi = list(map(float, [meteor_No, \
                    n_segments, fps, hnr, mle, binn, px_fm, rho, phi]))

            # Read meteor measurements
            if entry_counter > 3:
                
                # Skip lines with NaNs for centroids
                if '000nan' in line:
                    continue

                mag = np.nan

                # Read magnitude if it is in the file
                if len(line.split()) > 8:

                    line_sp = line.split()

                    mag = float(line_sp[8])

                # Read meteor frame-by-frame measurements
                frame_n, x, y, ra, dec, azim, elev, inten = list(map(float, line.split()[:8]))

                meteor_meas.append([calib_status, frame_n, x, y, ra, dec, azim, elev, inten, mag])


            entry_counter += 1


        # Add the last entry to the list
        if meteor_meas:
            meteor_list.append([ff_name, cam_code, meteor_No, n_segments, fps, hnr, mle, binn, px_fm, 
                rho, phi, meteor_meas])


        # If the return in the format suitable for the writeFTPdetectinfo function, reformat the output list
        if ret_input_format:

            output_list = []

            for entry in meteor_list:
                ff_name, cam_code, meteor_No, n_segments, fps, hnr, mle, binn, px_fm, rho, phi, \
                    meteor_meas = entry

                # Remove the calibration status from the list of centroids
                meteor_meas = [line[1:] for line in meteor_meas]

                output_list.append([ff_name, meteor_No, rho, phi, meteor_meas])

            return cam_code, fps, output_list

        else:
            return meteor_list




# Test
if __name__ == '__main__':
    # meteor_list = [["FF453_20160419_184117_248_0020992.bin", 1, 271.953268044, 8.13010235416,
    # [[ 124.5      ,   665.44095949,  235.00000979,  101.        ],
    #  [ 128.       ,   665.60121632,  235.9999914 ,  119.        ],
    #  [ 137.5      ,   666.54497978,  237.00000934,  195.        ],
    #  [ 151.5      ,   664.52378186,  238.99999005,  120.        ],
    #  [ 152.5      ,   666.        ,  239.        ,  47.        ]]]]

    # file_name = 'FTPdetect_test.txt'
    # ff_directory = 'here'
    # cal_directory = 'there'
    # cam_code = 450
    # fps = 25

    # writeFTPdetectinfo(meteor_list, ff_directory, file_name, cal_directory, cam_code, fps)


    dir_path = "C:\\Users\\delorayn1\\Desktop\\20170813_213506_620678"
    file_name = "FTPdetectinfo_20170813_213506_620678.txt"

    meteor_list = readFTPdetectinfo(dir_path, file_name)

    print(meteor_list)

            


