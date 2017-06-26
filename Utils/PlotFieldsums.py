# RPi Meteor Station
# Copyright (C) 2017  Denis Vida
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

""" Generates a plot with all intensities from fieldsum files. """


import os
import datetime
import shutil

import numpy as np
import matplotlib.pyplot as plt

import RMS.ConfigReader as cr
from RMS.Formats.FieldIntensities import readFieldIntensitiesBin
from RMS.Formats.FFbin import filenameToDatetime


def plotFieldsums(dir_path, config):
    """ Plots a graph of all intensity sums from FS*.bin files in the given directory. 
    
    Arguments:
        dir_path: [str] Path to the directory which containes the FS*.bin files.
        config: [Config structure] Configuration.

    Return:
        None
    """


    time_data = []
    intensity_data = []

    # Get all fieldsum files in the directory
    for file_name in sorted(os.listdir(dir_path)):

        # Check if it is the fieldsum file
        if ('FS' in file_name) and ('_fieldsum.bin' in file_name):

            # Read the field sums
            half_frames, intensity_array = readFieldIntensitiesBin(dir_path, file_name)

            # Extract the date and time from the 
            dt = filenameToDatetime(file_name)

            # Calculate the exact time of every half frame
            for half_frame, intensity in zip(half_frames, intensity_array):

                frame_time = dt + datetime.timedelta(seconds=float(half_frame)/config.fps)

                time_data.append(frame_time)
                intensity_data.append(intensity)




    # Plot the intensity over time
    #plt.scatter(time_data, intensity_data, s=0.05)
    plt.plot(time_data, intensity_data, color='k', linewidth=0.01, zorder=3)

    plt.gca().set_yscale('log')

    plt.xlim(np.min(time_data), np.max(time_data))

    plt.ylim(np.min(intensity_data), np.max(intensity_data))

    plt.xlabel('Time')
    plt.ylabel('ADU')

    plt.grid(color='0.9', which='both')

    plt.title('Field sums for ' + os.path.basename(dir_path))

    plt.tight_layout()


    plt.savefig(os.path.join(dir_path, os.path.basename(dir_path) + '_fieldsums.png'), dpi=300)





if __name__ == "__main__":


    # Load config file
    config = cr.parse(".config")


    dir_path = '/home/dvida/Dropbox/Apps/Elginfield RPi RMS data/ArchivedFiles/20170626_020228_442736_detected'

    plotFieldsums(dir_path, config)

