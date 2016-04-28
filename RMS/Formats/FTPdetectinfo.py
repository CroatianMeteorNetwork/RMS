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


import datetime
import numpy as np


def makeFTPdetectinfo(meteor_list, file_name, ff_directory, cal_directory, cam_code, fps):
    """ Writes a FTPdetectinfo file from the list of detected meteors. 

    @param meteor_list: [list] a list of meteor data, entries: 
        ff_name, meteor_No, rho, theta, centroids
    @param file_name: [str] file name of the file in which the data will be written
    @param ff_directory: [str] path to the directory in which the file will be written
    @param cal_directory: [str] path to the CAL directory (optional, used only in CAMS processing)
    @param cam_code: [int] camera number
    @param fps: [float] frames per second of the camera

    @return None
    """


    # Open a file
    with open(file_name, 'w') as ftpdetect_file:

        # Write the number of meteors on the beginning fo the file
        total_meteors = len(meteor_list)
        ftpdetect_file.write("Meteor Count = "+str(total_meteors).zfill(6)+ "\n")
        ftpdetect_file.write("-----------------------------------------------------\n")
        ftpdetect_file.write("Processed with Asteria on " + str(datetime.datetime.now()) + "\n")
        ftpdetect_file.write("-----------------------------------------------------\n")
        ftpdetect_file.write("FF  folder = " + ff_directory + "\n")
        ftpdetect_file.write("CAL folder = " + cal_directory + "\n")
        ftpdetect_file.write("-----------------------------------------------------\n")
        ftpdetect_file.write("FF  file processed\n")
        ftpdetect_file.write("CAL file processed\n")
        ftpdetect_file.write("Cam# Meteor# #Segments fps hnr mle bin Pix/fm Rho Phi\n")
        ftpdetect_file.write("Per segment:  Frame# Col Row RA Dec Azim Elev Inten\n")

        # Write info for all meteors
        for meteor in meteor_list:

            # Unpack the meteor data
            ff_name, meteor_No, rho, theta, centroids = meteor

            ftpdetect_file.write("-------------------------------------------------------\n")
            ftpdetect_file.write(ff_name + "\n")
            ftpdetect_file.write("Uncalibrated" + "\n")

            # Calculate meteor's angular velocity
            first_centroid = centroids[0]
            last_centroid = centroids[-1]
            frame1, x1, y1, _ = first_centroid
            frame2, x2, y2, _ = last_centroid

            ang_vel = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) / float(frame2 - frame1)

            # Write detection header
            ftpdetect_file.write(str(cam_code).zfill(4) + " " + str(meteor_No).zfill(4) + " " + 
                str(len(centroids)).zfill(4) + " " + "{:07.2f}".format(round(float(fps), 2)) + 
                " 000.0 000.0  00.0 " + str(round(ang_vel, 1)).zfill(5) + " " + 
                "{:06.1f} {:06.1f}".format(round(rho, 1), round(theta, 1)) + "\n")

            # Write individual detection points
            for line in centroids:
                frame, x, y, level = line

                ftpdetect_file.write("{:06.1f} {:07.2f} {:07.2f}".format(frame, round(x, 2), round(y, 2)) + 
                    " 000.00 000.00 000.00 000.00 " + "{:06d}".format(int(level)) + "\n")



# Test
if __name__ == '__main__':
    meteor_list = [["FF453_20160419_184117_248_0020992.bin", 1, 271.953268044, 8.13010235416,
    [[ 124.5      ,   665.44095949,  235.00000979,  101.        ],
     [ 128.       ,   665.60121632,  235.9999914 ,  119.        ],
     [ 137.5      ,   666.54497978,  237.00000934,  195.        ],
     [ 151.5      ,   664.52378186,  238.99999005,  120.        ],
     [ 152.5      ,   666.        ,  239.        ,  47.        ]]]]

    file_name = 'FTPdetect_test.txt'
    ff_directory = 'here'
    cal_directory = 'there'
    cam_code = 450
    fps = 25

    makeFTPdetectinfo(meteor_list, file_name, ff_directory, cal_directory, cam_code, fps)

            


