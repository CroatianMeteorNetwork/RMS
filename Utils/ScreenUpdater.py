from __future__ import print_function, division, absolute_import


import os
import time
import platform
import multiprocessing
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from RMS.Formats.FFfile import read as readFF
from RMS.Formats.FFfile import validFFName



def myPause(interval):
    """ Modify the pause function so that it doesn't re-focus the plot on update. """

    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return None


class LiveViewer(multiprocessing.Process):
    def __init__(self, dir_path, pause_time=1):

        self.dir_path = dir_path
        self.pause_time = pause_time

        self.exit = multiprocessing.Event()


        # Create a list of FF files in the given directory
        ff_list = []


        ### PLOTTING

        # Disable navbar
        matplotlib.rcParams['toolbar'] = 'None'

        # Remove bells and whistles
        plt.gca().set_axis_off()
        plt.gcf().patch.set_facecolor('k')
        plt.gcf().canvas.set_window_title('LiveViewer')

        # Open the window full screen
        mng = plt.get_current_fig_manager()
        if platform.system() == "Windows":
            mng.window.state('zoomed')
        else:
            mng.resize(*mng.window.maxsize())


        #plt.axis([-50,50,0,10000])

        # Enable interactive plotting
        plt.ion()

        plt.show()

        showing_empty = False

        # Repeat until the process is killed from the outside
        while not self.exit.is_set():

            # Monitor the given folder for new FF files
            new_ffs = [file_name for file_name in sorted(os.listdir(self.dir_path)) \
                if validFFName(file_name) and (file_name not in ff_list)]


            # If there are no FF files in the directory, show an empty image
            if (not len(ff_list)) and (not len(new_ffs)) and (not showing_empty):
                text = "No FF files found in the given directory as of yet: {:s}".format(self.dir_path)
                img = np.zeros((720, 1280))
                showing_empty = None


            # If there are new FF files, update the image
            if len(new_ffs):

                new_ff = new_ffs[-1]
                text = new_ff

                # Load the new FF
                ff = readFF(self.dir_path, new_ff)

                if ff is not None:
                    img = ff.maxpixel

                else:
                    time.sleep(self.pause_time)
                    continue

                plt.clf()
                showing_empty = False

                # Add new FF files to the list
                ff_list += new_ffs

            # If there are no FF files, wait
            else:
                if showing_empty is not None:
                    time.sleep(self.pause_time)
                    continue


            if showing_empty is not True:
                plt.imshow(img, cmap='gray', vmin=0, vmax=255)
                plt.text(0, 0, text, color='g')

                plt.tight_layout()

                plt.draw()

                myPause(self.pause_time)


            # Set the proper flag if not showing any FF files
            if showing_empty is None:
                showing_empty = True


    def stop(self):
        self.exit.set()




if __name__ == "__main__":

    import argparse

        ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Monitor the given folder for new FF files and show them on the screen.")

    arg_parser.add_argument('dir_path', metavar='DIR_PATH', type=str, \
        help="Directory to monitor for FF files.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    lv = LiveViewer(cml_args.dir_path)
    lv.start()