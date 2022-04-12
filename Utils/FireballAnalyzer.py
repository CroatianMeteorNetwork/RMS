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
            ff_xmax = min([xc + size//2, ff.ncols])
            ff_ymin = max([yc - size//2, 0])
            ff_ymax = min([yc + size//2, ff.nrows])
            # ff_frame = reconstructFrame(ff, frame_index, avepixel=True)
            # crop_ff = ff_frame[ff_ymin:ff_ymax, ff_xmin:ff_xmax]
            crop_ave = ff.avepixel[ff_ymin:ff_ymax, ff_xmin:ff_xmax]
            crop_std = ff.stdpixel[ff_ymin:ff_ymax, ff_xmin:ff_xmax]

            # Apply thresholding to the crop image
            threshold_mask = crop_ave + config.k1*crop_std + config.j1
            threshold_mask[threshold_mask > 255] = 255
            threshold_mask = (crop <= threshold_mask) & (crop < 250)
            crop_masked = np.array(crop)
            crop_masked[threshold_mask] = 0

            # Add the image to the summary image
            sum_y0 = max_size*(frame_no//graph_dimension)
            sum_x0 = max_size*(frame_no%graph_dimension)

            crop_thumb[sum_y0:sum_y0 + 2*(size//2), sum_x0:sum_x0 + 2*(size//2)] = crop_masked

            # Add frame index
            plt.text(sum_x0, sum_y0, str(frame_index), va='top', ha='left', size=5, color='0.5')


            # Apply a median filter on the image
            crop_median = scipy.ndimage.median_filter(crop, size=10)
            crop_median_masked = scipy.ndimage.median_filter(crop_masked, size=5)

            # Compute the centre of mass and plot it
            #center_mass = scipy.ndimage.measurements.center_of_mass(img_median, labels=(1))
            #y_cmass, x_cmass = center_mass

            if np.any(crop_median_masked):

                # Find the centroid as the peak of the image sum
                x_sum = np.sum(crop_median_masked, axis=0)
                y_sum = np.sum(crop_median_masked, axis=1)
                # x_centroid = np.argmax(x_sum)
                # y_centroid = np.argmax(y_sum)
                x_max = np.argwhere(x_sum == np.amax(x_sum)).flatten()
                #x_centroid = x_max[int(round(len(x_max)/2))]
                x_centroid = np.mean(x_max)
                
                y_max = np.argwhere(y_sum == np.amax(y_sum)).flatten()
                #y_centroid = y_max[int(round(len(y_max)/2))]
                y_centroid = np.mean(y_max)

                print(x_max, x_centroid)
                print(y_max, y_centroid)

                automated_x = x_centroid + xc - size/2
                automated_y = y_centroid + yc - size/2

                automated_picks[frame_index] = (automated_x, automated_y)

                plt.scatter(sum_x0 + x_centroid, sum_y0 + y_centroid, marker='x', c='blue')

            else:
                x_centroid = None
                y_centroid = None
                automated_x = None
                automated_y = None




            
            if frame_index > 144:
                plt.clf()
                plt.plot(x_sum, color='b')
                plt.plot(y_sum, color='r')
                plt.vlines(x_centroid, np.min(x_sum), np.max(x_sum), color='b')
                plt.vlines(y_centroid, np.min(y_sum), np.max(y_sum), color='r')
                plt.show()
                print(x_centroid, y_centroid)

                plt.clf()
                fig, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(ncols=6, sharex=True, sharey=True)
                ax0.imshow(crop, cmap='gray', vmin=0, vmax=255)
                ax0.scatter(x_centroid, y_centroid, marker='x', c='blue')
                ax1.imshow(crop_median, cmap='gray', vmin=0, vmax=255)
                ax1.scatter(x_centroid, y_centroid, marker='x', c='blue')
                ax2.imshow(crop_masked, cmap='gray', vmin=0, vmax=255)
                ax2.scatter(x_centroid, y_centroid, marker='x', c='blue')
                ax3.imshow(crop_median_masked, cmap='gray', vmin=0, vmax=255)
                ax3.scatter(x_centroid, y_centroid, marker='x', c='blue')

                ax4.imshow(crop_ave, cmap='gray', vmin=0, vmax=255)
                ax5.imshow(crop_std, cmap='gray', vmin=0, vmax=255)
                plt.show()


            # Find FTPdetectinfo pick and plot it
            for pick in meteor_data[0][11]:
                calib_status, frame_n, manual_x, manual_y, ra, dec, azim, elev, inten, mag = pick

                manual_thumb_x = sum_x0 + manual_x - xc + size/2
                manual_thumb_y = sum_y0 + manual_y - yc + size/2

                manual_picks[frame_index] = (manual_x, manual_y)

                if int(frame_index) == int(frame_n):
                    plt.scatter(manual_thumb_x, manual_thumb_y, marker='+', c='red')


                    print("automated:", automated_x, automated_y)
                    print("manual:", manual_x, manual_y)

                    break




        plt.imshow(crop_thumb, cmap='gray')

        plt.show()


        # Plot deviations per frame
        for frame_index in sorted(automated_picks.keys()):

            automated_x, automated_y = automated_picks[frame_index]
            manual_x, manual_y = manual_picks[frame_index]

            if automated_x is not None:
                x_diff = manual_x - automated_x
                y_diff = manual_y - automated_y
                plt.scatter(frame_index, x_diff, marker='x', color='b')
                plt.scatter(frame_index, y_diff, marker='+', color='r')

                print(frame_index)
                print("    automated:", automated_x, automated_y)
                print("    manual:", manual_x, manual_y)


        plt.show()
