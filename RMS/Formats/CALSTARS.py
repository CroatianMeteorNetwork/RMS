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


def writeCALSTARS(star_list, ff_directory, file_name, cam_code, nrows, ncols, chunk_frames=256):
    """ Writes the star list into the CAMS CALSTARS format. 

    Arguments:
        star_list: [list] a list of star data, entries:
            ff_name, star_data
            star_data entries:
                x, y, bg_level, level
        ff_directory: [str] path to the directory in which the file will be written
        file_name: [str] file name in which the data will be written
        cam_code: [str] camera code
        nrows: [int] number of rows in the image
        ncols: [int] number of columns in the image

    Keyword arguments:
        chunk_frames: [int] Number of frames in the FF file or frame chunk. Default is 256.

    Return:
        None
    """

    with open(os.path.join(ff_directory, file_name), 'w') as star_file:

        # Write the header
        star_file.write("==========================================================================\n")
        star_file.write("RMS star extractor" + "\n")
        star_file.write("Cal time = FF header time plus 255/(2*framerate_Hz) seconds" + "\n")
        star_file.write("      Y       X IntensSum Ampltd  FWHM  BgLvl   SNR NSatPx" + "\n")
        star_file.write("==========================================================================\n")
        star_file.write("FF folder = " + ff_directory + "\n")
        star_file.write("Cam #   = " + str(cam_code) + "\n")
        star_file.write("Nrows   = " + str(nrows) + "\n")
        star_file.write("Ncols   = " + str(ncols) + "\n")
        star_file.write("Nframes = " + str(chunk_frames) + "\n")
        star_file.write("Nstars  = -1" + "\n")

        # Write all stars in the CALSTARS file
        for star in star_list:

            # Skip empty star lists
            if len(star) < 2:
                continue

            # Unpack star data
            ff_name, star_data = star

            # Write star header per image
            star_file.write("==========================================================================\n")
            star_file.write(ff_name + "\n")
            star_file.write("Star area dim = -1" + "\n")
            star_file.write("Integ pixels  = -1" + "\n")

            # Write every star to file
            for y, x, amplitude, level, fwhm, background, snr, saturated_count in list(star_data):

                # Limit the saturation count to 999999
                if saturated_count > 999999:
                    saturated_count = 999999

                # Limit the SNR to 99.99
                if snr > 99.99:
                    snr = 99.99

                star_file.write("{:7.2f} {:7.2f} {:9d} {:6d} {:5.2f} {:6d} {:5.2f} {:6d}".format(
                    round(y, 2), round(x, 2), 
                    int(level), int(amplitude), fwhm, int(background), snr, int(saturated_count)) + "\n")

        # Write the end separator
        star_file.write("##########################################################################\n")



def readCALSTARS(file_path, file_name, chunk_frames=256):
    """ Reads a list of detected stars from a CAMS CALSTARS format. 

    Arguments:
        file_path: [str] Path to the directory where the CALSTARS file is located.
        file_name: [str] Name of the CALSTARS file.

    Keyword arguments:
        chunk_frames: [int] Number of frames in the FF file or frame chunk. Default is 256.
            Will be overwritten by a number in the CALSTARS file if present.

    Return:
        star_list, chunk_frames: 
            - star_list [list] a list of star data, entries:
                ff_name, star_data
                star_data entries:
                    x, y, bg_level, level, fwhm
            - chunk_frames [int] Number of frames in the FF file or frame chunk.
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

            # Skip lines if necessary
            if skip_lines > 0:
                skip_lines -= 1
                continue

            # Read the number of frames if given (Nframes = ...)
            if "Nframes" in line:
                chunk_frames = int(line.split('=')[-1])
                continue

            # Check for end of star entry
            if (("===" in line) or ("###" in line)) and len(ff_name):

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

            # Read the star data
            y, x, level, amplitude = float(line[0]), float(line[1]), int(line[2]), int(line[3])

            # Read FWHM if given
            if len(line) >= 5:
                fwhm = float(line[4])
            else:
                fwhm = -1.0

            # Read the background level
            if len(line) >= 6:
                background = int(line[5])
            else:
                background = -1

            # Read the SNR
            if len(line) >= 7:
                snr = float(line[6])
            else:
                snr = -1.0

            # Read the number of saturated pixels
            if len(line) >= 8:
                saturated_count = int(line[7])
            else:
                saturated_count = -1

            # Save star data
            star_data.append([y, x, level, amplitude, fwhm, background, snr, saturated_count])

    
    return calibrationstars_list, chunk_frames