# RPi Meteor Station
# Copyright (C) 2016  Dario Zubovic, Denis Vida
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

import sys
import os
import time
import datetime
import logging

# RMS imports
import RMS.ConfigReader as cr
from RMS.Formats import FFbin
from RMS.Formats import FTPdetectinfo
from RMS.Formats import CALSTARS
from RMS.ExtractStars import extractStars
from RMS.Detection import detectMeteors


# Get the logger from the main module
log = logging.getLogger("logger")


def detectStarsAndMeteors(ff_directory, ff_name, config):
    """ Run the star extraction and subsequently runs meteor detection on the FF bin file if there are enough
        stars on the image.

    Arguments:
        ff_directory: [str] path to the directory where the FF files are located
        ff_name: [str] name of the FF file
        config: [Configuration object] configuration object

    Return:
        [ff_name, star_list, meteor_list] detected stars and meteors

    """

    # Run star extraction on the FF bin
    star_list = extractStars(ff_directory, ff_name, config)

    log.info('Detected stars: ' + str(len(star_list[0])))

    # Run meteor detection if there are enough stars on the image
    if len(star_list[0]) >= config.ff_min_stars:

        log.debug('More than ' + str(config.ff_min_stars) + ' stars, detecting meteors...')

        meteor_list = detectMeteors(ff_directory, ff_name, config)

        log.debug(ff_name + ' detected meteors: ' + str(len(meteor_list)))

    else:
        meteor_list = []


    return ff_name, star_list, meteor_list





if __name__ == "__main__":

    time_start = datetime.datetime.now()

    # Load config file
    config = cr.parse(".config")

    if not len(sys.argv) == 2:
        print "Usage: python -m RMS.ExtractStars /path/to/bin/files/"
        sys.exit()
    
    # Get paths to every FF bin file in a directory 
    ff_dir = os.path.abspath(sys.argv[1])
    ff_list = [ff_name for ff_name in os.listdir(ff_dir) if ff_name[0:2]=="FF" and ff_name[-3:]=="bin"]

    # Check if there are any file in the directory
    if(len(ff_list) == None):
        print "No files found!"
        sys.exit()


    # Init data lists
    star_list = []
    meteor_list = []

    # Go through all files in the directory
    for ff_name in sorted(ff_list):

        print ff_name

        t1 = time.clock()

        # Run star and meteor detection
        _, star_data, meteor_data = detectStarsAndMeteors(ff_dir, ff_name, config)

        print 'Time for processing: ', time.clock() - t1

        x2, y2, background, intensity = star_data

        # Skip if no stars were found
        if not x2:
            continue

        # Construct the table of the star parameters
        star_data = zip(x2, y2, background, intensity)

        # Add star info to the star list
        star_list.append([ff_name, star_data])

        # Print found stars
        print '   ROW    COL intensity'
        for x, y, bg_level, level in star_data:
            print ' {:06.2f} {:06.2f} {:6d} {:6d}'.format(round(y, 2), round(x, 2), int(bg_level), int(level))


        # # Show stars if there are only more then 20 of them
        # if len(x2) < 20:
        #     continue

        # # Load the FF bin file
        # ff = FFbin.read(ff_dir, ff_name)

        # plotStars(ff, x2, y2)

        # Handle the detected meteors
        meteor_No = 1
        for meteor in meteor_data:

            rho, theta, centroids = meteor

            # Append to the results list
            meteor_list.append([ff_name, meteor_No, rho, theta, centroids])
            meteor_No += 1


    # Load data about the image
    ff = FFbin.read(ff_dir, ff_name)

    # Generate the name for the CALSTARS file
    calstars_name = 'CALSTARS' + "{:04d}".format(int(ff.camno)) + os.path.basename(ff_dir) + '.txt'

    # Write detected stars to the CALSTARS file
    CALSTARS.writeCALSTARS(star_list, ff_dir, calstars_name, ff.camno, ff.nrows, ff.ncols)

    # Generate FTPdetectinfo file name
    ftpdetectinfo_name = os.path.join(ff_dir, 'FTPdetectinfo_' + os.path.basename(ff_dir) + '.txt')

    # Write FTPdetectinfo file
    FTPdetectinfo.writeFTPdetectinfo(meteor_list, ff_dir, ftpdetectinfo_name, ff_dir, 
        config.stationID, config.fps)

    print 'Total time taken: ', datetime.datetime.now() - time_start


