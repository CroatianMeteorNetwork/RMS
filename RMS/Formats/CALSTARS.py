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


import os


def writeCALSTARS(star_list, ff_directory, file_name, cam_code, nrows, ncols):
    """ Writes the star list into the CAMS CALSTARS format. 

    @param star_list: [list] a list of star data, entries:
        ff_name, star_data
        star_data entries:
            x, y, bg_level, level

    @param ff_directory: [str] path to the directory in which the file will be written
    @param file_name: [str] file name in which the data will be written
    @param cam_code: [str] camera code
    @param nrows: [int] number of rows in the image
    @param ncols: [int] number of columns in the image

    @return None
    """

    with open(os.path.join(ff_directory, file_name), 'w') as star_file:

        # Write the header
        star_file.write("==========================================================================\n")
        star_file.write("RMS star extractor" + "\n")
        star_file.write("Cal time = FF header time plus 255/(2*framerate_Hz) seconds" + "\n")
        star_file.write("Row  Column  Intensity-Backgnd  Amplitude  (integrated values) FWHM" + "\n")
        star_file.write("==========================================================================\n")
        star_file.write("FF folder = " + ff_directory + "\n")
        star_file.write("Cam #  = " + str(cam_code) + "\n")
        star_file.write("Nrows  = " + str(nrows) + "\n")
        star_file.write("Ncols  = " + str(ncols) + "\n")
        star_file.write("Nstars = -1" + "\n")

        # Write all stars in the CALSTARS file
        for star in star_list:

            # Unpack star data
            ff_name, star_data = star

            # Write star header per image
            star_file.write("==========================================================================\n")
            star_file.write(ff_name + "\n")
            star_file.write("Star area dim = -1" + "\n")
            star_file.write("Integ pixels  = -1" + "\n")

            # Write every star to file
            for y, x, amplitude, level, fwhm in list(star_data):
                star_file.write("{:7.2f} {:7.2f} {:6d} {:6d} {:5.2f}".format(round(y, 2), round(x, 2), 
                    int(level), int(amplitude), fwhm) + "\n")

        # Write the end separator
        star_file.write("##########################################################################\n")



def readCALSTARS(file_path, file_name):
    """ Reads a list of detected stars from a CAMS CALSTARS format. 

    @param file_path: [string] path to the directory where the CALSTARS file is located
    @param file_name: [string] name of the CALSTARS file

    @return star_list: [list] a list of star data, entries:
        ff_name, star_data
        star_data entries:
            x, y, bg_level, level, fwhm
    """

    
    calstars_path = os.path.join(file_path, file_name)

    # Check if the CALSTARS file exits
    if not os.path.isfile(calstars_path):
        print('The CALSTARS file: {:s} does not exist!'.format(calstars_path))
        return False

    # Open the CALSTARS file for reading
    with open(calstars_path) as star_file:

        calibrationstars_list = []

        ff_name = ''
        star_data = []
        skip_lines = 0
        for line in star_file.readlines()[11:]:

            # Skip lines if neccessary
            if skip_lines > 0:
                skip_lines -= 1
                continue

            # Check for end of star entry
            if ("===" in line) or ("###" in line):

                # Add the star list to the main list
                calibrationstars_list.append([ff_name, star_data])

                # Reset the star list
                star_data = []
                
                continue

            # Remove newline
            line = line.replace('\n', '').replace('\r', '')

            if 'FF' in line:
                ff_name = line
                skip_lines = 2
                continue

            # Split the line
            line = line.split()

            if len(line) < 4:
                continue

            try:
                float(line[0])
                float(line[1])
                int(line[2])
                int(line[3])

            except:
                continue

            # Read FWHM if given
            if len(line) == 5:
                fwhm = float(line[4])
            else:
                fwhm = -1.0

            # Save star data
            star_data.append([float(line[0]), float(line[1]), int(line[2]), int(line[3]), fwhm])

    
    return calibrationstars_list