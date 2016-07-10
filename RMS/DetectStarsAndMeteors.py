import sys
import os
import time

# RMS imports
import RMS.ConfigReader as cr
from RMS.Formats import FFbin
from RMS.Formats import FTPdetectinfo
from RMS.Formats import CALSTARS
from RMS.ExtractStars import extractStars
from RMS.Detection import detectMeteors


def detectStarsAndMeteors(ff_directory, ff_name, config):
    """ Run the star extraction and subsequently runs meteor detection on the FF bin file if there are enough
        stars on the image.
    """


    # Add to config!!!!
    ff_min_stars = 5

    # Run star extraction on the FF bin
    star_list = extractStars(ff_directory, ff_name, config)

    print 'N STARS:', len(star_list[0])

    # Run meteor detection if there are enough stars on the image
    if len(star_list[0]) >= ff_min_stars:
        meteor_list = detectMeteors(ff_directory, ff_name, config)
    else:
        meteor_list = []

    return star_list, meteor_list





if __name__ == "__main__":

    time_start = time.clock()

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
        star_data, meteor_data = detectStarsAndMeteors(ff_dir, ff_name, config)

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


        # # Show stars if there are only more then 10 of them
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
    calstars_name = 'CALSTARS' + "{:04d}".format(int(ff.camno)) + ff_dir.split(os.sep)[-1] + '.txt'

    # Write detected stars to the CALSTARS file
    CALSTARS.writeCALSTARS(star_list, ff_dir, calstars_name, ff.camno, ff.nrows, ff.ncols)

    # Generate FTPdetectinfo file name
    ftpdetectinfo_name = os.path.join(ff_dir, 'FTPdetectinfo_' + os.path.basename(ff_dir) + '.txt')

    # Write FTPdetectinfo file
    FTPdetectinfo.writeFTPdetectinfo(meteor_list, ff_dir, ftpdetectinfo_name, ff_dir, 
        config.stationID, config.fps)

    print 'Total time taken: ', time.clock() - time_start


