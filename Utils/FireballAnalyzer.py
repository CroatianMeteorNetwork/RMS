""" Automatically makes position picks for fireball detections. """

import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

from RMS.Formats.FRbin import read as readFR
from RMS.Formats.FFfile import read as readFF
from RMS.Formats.FFfile import validFFName, reconstructFrame
from RMS.Formats.FTPdetectinfo import readFTPdetectinfo


if __name__ == "__main__":

    import argparse
    import RMS.ConfigReader as cr

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Automatically makes position picks on FR files.")

    arg_parser.add_argument('input_path', metavar='INPUT_PATH', type=str,
                            help="Path to an FR file.")

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str,
                            help="Path to a config file which will be used instead of the default one."
                                 " To load the .config file in the given data directory, write '.' (dot).")



    # Parse the command line arguments
    cml_args = arg_parser.parse_args()


    dir_path = os.path.dirname(cml_args.input_path)

    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, dir_path)
    
    # Load the FR file
    fr = readFR(*os.path.split(cml_args.input_path))

    # Load the FF file
    for file_name in os.listdir(dir_path):
        if validFFName(file_name):
            ff_name = file_name
    ff = readFF(dir_path, ff_name)


    # Load the FTPdetectinfo file
    for file_name in os.listdir(dir_path):
        if file_name.startswith("FTPdetectinfo") and file_name.endswith(".txt") and not "uncalibrated" in file_name:
            ftpdetectinfo_name = file_name


    meteor_data = readFTPdetectinfo(dir_path, ftpdetectinfo_name)


    print(fr)


    manual_picks = {}
    automated_picks = {}

    for line_no in range(fr.lines):

        print("Line {:d}".format(line_no))


        # Construct a thumbnail of all FR frames
        max_size = max(fr.size[line_no])
        n_frames = fr.frameNum[line_no]

        graph_dimension = int(np.ceil(np.sqrt(n_frames)))

        # Init the summary image
        crop_thumb = np.zeros((max_size*graph_dimension, max_size*graph_dimension), dtype=np.uint8)

        for frame_no in range(fr.frameNum[line_no]):

            frame_index = fr.t[line_no][frame_no]
            xc = fr.xc[line_no][frame_no]
            yc = fr.yc[line_no][frame_no]
            size = fr.size[line_no][frame_no]

            # Update size so it's an even number
            size = 2*(size//2)

            crop = fr.frames[line_no][frame_no][0:size, 0:size]

            print(frame_index, xc, yc, size)

            # Extract the FF segment
            ff_xmin = max([xc - size//2, 0])
            ff_xmax = min([xc + size//2, ff.ncols - 1])
            ff_ymin = max([yc - size//2, 0])
            ff_ymax = min([yc + size//2, ff.nrows - 1])
            # ff_frame = reconstructFrame(ff, frame_index, avepixel=True)
            # crop_ff = ff_frame[ff_ymin:ff_ymax, ff_xmin:ff_xmax]
            crop_ave = ff.avepixel[ff_ymin:ff_ymax, ff_xmin:ff_xmax]
            crop_std = ff.stdpixel[ff_ymin:ff_ymax, ff_xmin:ff_xmax]

            # Apply thresholding to the crop image
            threshold_mask = crop_ave + config.k1*crop_std + config.j1
            threshold_mask[threshold_mask > 250] = 250
            crop_masked = np.array(crop)
            crop_masked[(crop < threshold_mask) & (crop < 250)] = 0

            # Add the image to the summary image
            sum_y0 = max_size*(frame_no//graph_dimension)
            sum_x0 = max_size*(frame_no%graph_dimension)

            crop_thumb[sum_y0:sum_y0 + 2*(size//2), sum_x0:sum_x0 + 2*(size//2)] = crop_masked


            # Apply a median filter on the image
            crop_median = scipy.ndimage.median_filter(crop, size=10)
            crop_median_masked = scipy.ndimage.median_filter(crop_masked, size=10)

            # Compute the centre of mass and plot it
            #center_mass = scipy.ndimage.measurements.center_of_mass(img_median, labels=(1))
            #y_cmass, x_cmass = center_mass

            # Find the centroid as the peak of the image sum
            x_sum = np.sum(crop_median_masked, axis=0)
            y_sum = np.sum(crop_median_masked, axis=1)
            x_centroid = np.argmax(x_sum)
            y_centroid = np.argmax(y_sum)


            automated_x = sum_x0 + x_centroid
            automated_y = sum_y0 + y_centroid

            automated_picks[frame_index] = (automated_x, automated_y)
            
            if frame_no > 40:
                plt.clf()
                plt.plot(x_sum)
                plt.plot(y_sum)
                plt.show()
                print(x_centroid, y_centroid)

                plt.clf()
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4)
                ax1.imshow(crop_median, cmap='gray', vmin=0, vmax=255)
                ax1.scatter(x_centroid, y_centroid, marker='x', c='blue')
                ax2.imshow(crop_masked, cmap='gray', vmin=0, vmax=255)
                ax2.scatter(x_centroid, y_centroid, marker='x', c='blue')

                ax3.imshow(crop_ave, cmap='gray', vmin=0, vmax=255)
                ax4.imshow(crop_std, cmap='gray', vmin=0, vmax=255)
                plt.show()

            plt.scatter(sum_x0 + x_centroid, sum_y0 + y_centroid, marker='x', c='blue')


            # Find FTPdetectinfo pick and plot it
            for pick in meteor_data[0][11]:
                calib_status, frame_n, x, y, ra, dec, azim, elev, inten, mag = pick

                manual_x = sum_x0 + x - xc + size/2
                manual_y = sum_y0 + y - yc + size/2

                manual_picks[frame_index] = (manual_x, manual_y)

                if int(frame_index) == int(frame_n):
                    plt.scatter(manual_x, manual_y, marker='+', c='red')




        plt.imshow(crop_thumb, cmap='gray')

        plt.show()