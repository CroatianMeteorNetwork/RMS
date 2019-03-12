""" Plot the meteor PSF profile from the detection and FF file. """

import os


import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import scipy.optimize


from RMS.Formats.FFfile import read as readFF
from RMS.Formats.FTPdetectinfo import readFTPdetectinfo



def mapCoordinatesToRotatedImage(img, x, y, rot):
    """ Map the given coordinates to rotated image coordinates. """

    img = np.zeros_like(img)
    img[int(y), int(x)] = 1

    # Rotate the image
    img = scipy.ndimage.interpolation.rotate(img, rot)

    # Find the coordinates of the white pixel
    return np.unravel_index(img.argmax(), img.shape)




def gauss1D(params, A, mu, sigma, bg, saturation=255):


    x, force_sigma = params

    if force_sigma:
        sigma = force_sigma
        
    # Compute values of a gaussian
    values = A*np.exp(-(x - mu)**2/(2.0*sigma**2)) + bg

    if saturation > 0:
        
        # Clip the values to saturation limit
        values[values > saturation] = saturation

    return values



if __name__ == "__main__":


    import argparse

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="""Plot the meteor PSF profile from the detection and FF file.""", \
        formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('ff_file', metavar='FF_FILE', type=str, nargs=1, \
                    help='Path to an FF file which will be used for meteor profile plotting.')

    arg_parser.add_argument('ftpdetectinfo', metavar='FTP_FILE', type=str, nargs=1, \
                    help='FTPdetectinfo file with meteor detections.')

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    # Width of the strip around the meteor
    strip_width = 10

    # Number of meteor profiles to plot
    n_profiles = 20

    # Difference in Y coordinates between every profile
    vertical_step_offset = 70

    # Force sigma for fitting the Gaussian PSF (disable with -1)
    force_sigma = 1.3


    dir_path, ff_name = os.path.split(cml_args.ff_file[0])

    # Load the FF file
    ff = readFF(dir_path, ff_name)


    # Load the FTPdetectinfo file
    meteor_list = readFTPdetectinfo(*os.path.split(cml_args.ftpdetectinfo[0]))


    # Find the FF file among the detections
    for entry in meteor_list:

        ftp_ff_name, cam_code, meteor_No, n_segments, fps, hnr, mle, binn, px_fm, rho, phi, \
            meteor_meas = entry


        # Take only the FF file with the detection
        if ff_name == ftp_ff_name:

            img = ff.maxpixel

            x_beg = meteor_meas[0][2]
            y_beg = meteor_meas[0][3]
            x_end = meteor_meas[-1][2]
            y_end = meteor_meas[-1][3]

            rot_angle = (-phi + 180)%360

            # Get the coordinates of the beginning and the end in the rotated image
            y_beg_rot, x_beg_rot = mapCoordinatesToRotatedImage(img, x_beg, y_beg, rot_angle)
            y_end_rot, x_end_rot = mapCoordinatesToRotatedImage(img, x_end, y_end, rot_angle)

            # Rotate the meteor image so it's straight
            img = scipy.ndimage.interpolation.rotate(img, rot_angle)

            # Crop the meteor from the image
            x_mid = int((x_beg_rot + x_end_rot)/2)
            x_min = int(x_mid - strip_width)
            x_max = int(x_mid + strip_width)
            y_min = int(y_beg_rot - strip_width)
            y_max = int(y_end_rot + strip_width)

            x_min, x_max = sorted([x_min, x_max])
            y_min, y_max = sorted([y_min, y_max])

            img_crop = img[y_min:y_max, x_min + 1:x_max + 1]


            #fig, (ax1, ax2) = plt.subplots(ncols=2, gridspec_kw={'width_ratios': (0.1, 0.9)})
            ax1 = plt.subplot2grid((4,8), (1,0), rowspan=3)
            ax2 = plt.subplot2grid((4,8), (0,1), rowspan=4, colspan=7)

            ax1.imshow(img_crop, cmap='gray', vmin=0, vmax=255)
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)


            # Compute which profiles to take
            if n_profiles > 0:
                step = int(np.ceil(img_crop.shape[0]/n_profiles))
                if step < 1:
                    step = 1

            else:
                step = 1
                n_profiles = img_crop.shape[0]

            print('Step:', step)
            print('Rot angle:', rot_angle)



            color_list = plt.cm.inferno(np.linspace(0.8, 0.1, n_profiles + 1))

            # Plot PSF profiles
            count = 0
            sigma_list = []
            max_row = 0
            for i, row in enumerate(np.flipud(img_crop)):

                if i%step == 0:


                    x_arr = np.linspace(0, len(row) - 1, len(row))

                    # Fit a Gaussian to the meteor profile
                    p0 = [255., len(row)/2., 1.0, np.min(row)]


                    try:
                        popt, _ = scipy.optimize.curve_fit(gauss1D, (x_arr, force_sigma), row, p0=p0)
                    except:
                        # If the profile can't be fitted, skip it
                        count += 1
                        continue

                    
                    sigma_list.append(popt[2])

                    offset = vertical_step_offset*count
                    row = row.astype(np.float64)

                    # Plot the PSF profile
                    ax2.plot(x_arr, row + offset, color=color_list[count])


                    max_row = max([max_row, np.median(row + offset)])


                    # Plot the saturation region
                    x_arr_saturation = x_arr[np.where(row >= 250)]
                    row_saturation = row[np.where(row >= 250)]

                    if len(x_arr_saturation) > 1:
                        ax2.plot(x_arr_saturation, row_saturation + offset, color='r')


                    # Plot the fitted Gaussian
                    x_arr_plot = np.linspace(0, len(row) - 1, 10*len(row))
                    ax2.plot(x_arr_plot, gauss1D((x_arr_plot, force_sigma), *popt, saturation=-1) + offset, color=color_list[count], linestyle='dotted')

                    count += 1

                    if count > n_profiles:
                        break


            ax2.get_yaxis().set_visible(False)

            ax2.set_xlim([0, np.max(x_arr)])
            ax2.set_ylim([0, 4/3*max_row])

            ax2.set_xlabel('X (px)')

            plt.tight_layout()
            plt.show()



            if not force_sigma:

                plt.hist(sigma_list)
                plt.show()




