""" Given the FTPdetectinfo file (assuming FF files are available) and the stddev of Gaussian PSF of the image,
    correct the magnitudes and levels in the file for saturation. """

from __future__ import absolute_import, division, print_function

import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
from RMS.Formats.FFfile import read as readFF
from RMS.Formats.FFfile import validFFName
from RMS.Formats.FTPdetectinfo import (findFTPdetectinfoFile,
                                       readFTPdetectinfo, writeFTPdetectinfo)
from RMS.Routines.Image import applyFlat, loadFlat, thickLine

from Utils.SaturationSimulation import findUnsaturatedMagnitude

if __name__ == "__main__":

    import argparse

    ### COMMAND LINE ARGUMENTS
    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Correct the magnitudes in the FTPdetectinfo file for saturation.")

    arg_parser.add_argument('ftpdetectinfo_path', nargs=1, metavar='DIR_PATH', type=str, \
        help='Path to the FTPdetectinfo file.')

    arg_parser.add_argument('psf_sigma', nargs=1, metavar='PSF_SIGMA', type=float, \
        help='Standard deviation of the Gaussian PSF in pixels.')

    arg_parser.add_argument('-s', '--satlvl', metavar='SATURATION_LEVEL', type=int, \
        help="Saturation level. 255 by default.", default=255)

    arg_parser.add_argument('-f', '--flat', metavar='FLAT', type=str, \
        help="Path to the flat frame.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################


    # Read command line arguments
    ftpdetectinfo_path = cml_args.ftpdetectinfo_path[0]
    ftpdetectinfo_path = findFTPdetectinfoFile(ftpdetectinfo_path)
    dir_path, ftpdetectinfo_name = os.path.split(ftpdetectinfo_path)

    gauss_sigma = cml_args.psf_sigma[0]

    saturation_lvl = cml_args.satlvl


    # Load meteor data from FTPdetecinfo
    cam_code, fps, meteor_list = readFTPdetectinfo(dir_path, ftpdetectinfo_name, ret_input_format=True)


    # Load the flat, if given
    flat = None
    if cml_args.flat:
        flat = loadFlat(*os.path.split(cml_args.flat))

    corrected_meteor_list = []

    # Find matching FF files in the directory
    for entry in meteor_list:

        ftp_ff_name, meteor_No, rho, phi, meteor_meas = entry

        # Find the matching FTPdetectinfo file in the directory
        for ff_name in sorted(os.listdir(dir_path)):

            # Reject all non-FF files
            if not validFFName(ff_name):
                continue

            # Reject all FF files which do not match the name in the FTPdetecinfo
            if ff_name != ftp_ff_name:
                continue


            print('Correcting for saturation:', ff_name)

            # Load the FF file
            ff = readFF(dir_path, ff_name)

            # Apply the flat to avepixel
            if flat:
                avepixel = applyFlat(ff.avepixel, flat)

            else:
                avepixel = ff.avepixel


            # Compute angular velocity
            first_centroid = meteor_meas[0]
            last_centroid  = meteor_meas[-1]
            frame1, x1, y1 = first_centroid[:3]
            frame2, x2, y2 = last_centroid[:3]

            px_fm = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)/float(frame2 - frame1)

            print('Ang vel:', px_fm*fps, 'px/s')

            corrected_meteor_meas = []


            print('Frame, App mag, Corr mag, Background')

            # Go though all meteor centroids
            for line in meteor_meas:

                frame_n, x, y, ra, dec, azim, elev, inten, mag = line

                # Compute the photometric offset
                photom_offset = mag + 2.5*np.log10(inten)


                ### Compute the background intensity value behind the meteor ###

                # Get the mask for the background as a 3 sigma streak around the meteor, but using avepixel
                mask = thickLine(avepixel.shape[0], avepixel.shape[1], x, y, px_fm, phi - 90, \
                    3*gauss_sigma).astype(bool)

                img = np.ma.masked_array(avepixel, ~mask)
                    
                bg_val = np.ma.median(img)


                ### ###

                # Find the unsaturated magnitude
                unsaturated_mag = findUnsaturatedMagnitude(mag, photom_offset, bg_val, fps, px_fm*fps,
                    gauss_sigma, saturation_point=saturation_lvl)


                print("{:5.1f}, {:7.2f}, {:8.2f}, {:10.1f}".format(frame_n, mag, unsaturated_mag, bg_val))


                # Compute the intensity from unsaturated magnitude
                unsaturated_inten = round(10**((photom_offset - mag)/2.5), 0)

                corrected_meteor_meas.append([frame_n, x, y, ra, dec, azim, elev, unsaturated_inten, 
                    unsaturated_mag])



            if not corrected_meteor_meas:
                corrected_meteor_meas = meteor_meas


        corrected_meteor_list.append([ftp_ff_name, meteor_No, rho, phi, corrected_meteor_meas])


    # Calibration string to be written to the FTPdetectinfo file
    calib_str = "RMS - Saturation corrected on {:s} UTC".format(str(datetime.datetime.utcnow()))

    # Write a corrected FTPdetectinfo file
    corrected_ftpdetectinfo_name = ftpdetectinfo_name.strip('.txt') + '_saturation_corrected.txt'

    print('Saving to:', os.path.join(dir_path, corrected_ftpdetectinfo_name))

    writeFTPdetectinfo(corrected_meteor_list, dir_path, corrected_ftpdetectinfo_name, dir_path, cam_code, \
        fps, calibration=calib_str, celestial_coords_given=True)




