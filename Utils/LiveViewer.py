""" Monitors a given directory for FF files, and shows them on the screen as new ones get created. It can
    also do slideshows.
"""

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
    def __init__(self, dir_path, slideshow=False, slideshow_pause=2.0, banner_text="", \
        update_interval=5.0):
        """ Monitors a given directory for FF files, and shows them on the screen as new ones get created. It can
            also do slideshows. 

        Arguments:
            dir_path: [str] Directory to monitor for new FF files.
            
        Keyword arguments:
            slideshow: [bool] Start a slide show instead of monitoring the folder for new files.
            slideshow_pause: [float] Number of seconds between slideshow updated. 2 by default.
            banner_text: [str] Banner text that will be shown on the screen.
            update_interval: [float] Number of seconds for checking the given directory for new files.
        """

        super(LiveViewer, self).__init__()

        self.dir_path = dir_path

        self.slideshow = slideshow
        self.slideshow_pause = slideshow_pause
        self.banner_text = banner_text
        self.update_interval = update_interval

        self.exit = multiprocessing.Event()



    def initPlot(self):
        """ Init the plot. """

        # Force backend
        plt.switch_backend("TkAgg")

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


        # Enable interactive plotting
        plt.ion()

        plt.show()


    def updateImage(self, img, text, banner_text=""):
        """ Update the image on the screen. 
        
        Arguments:
            img: [2D ndarray] Image to show on the screen as a numpy array.
            text: [str] Text that will be printed on the image.
        """

        # Show the image
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)

        # Plot the text (i.e. FF file name)
        plt.text(0, 0, text, color='g')

        # Plot the banner
        plt.text(img.shape[1]//2, 0, banner_text, ha='center', va='top', color='r', size=20)

        plt.tight_layout()

        plt.draw()

        myPause(self.update_interval)


    def startSlideshow(self):
        """ Start a slideshow. 
        """

        # Make a list of FF files in the given directory
        ff_list = [file_name for file_name in sorted(os.listdir(self.dir_path)) if validFFName(file_name)]

        # Exit if no FF files were found
        if not ff_list:
            print("No FF files in the given directory to use for a slideshow!")
            self.exit.set()
            return None

        # Go through the list of FF files and show them on the screen
        first_run = True
        while not self.exit.is_set():
            for ff_name in ff_list:

                # Stop the loop if the slideshow should stop
                if self.exit.is_set():
                    break

                # Load the FF file
                ff = readFF(self.dir_path, ff_name, verbose=False)
                text = ff_name

                # If the FF files was loaded, show the maxpixel
                if ff is not None:
                    img = ff.maxpixel

                else:

                    # If an FF files could not be loaded on the first run, show an empty image
                    if first_run:
                        img = np.zeros((720, 1280))
                        text = "The FF file {:s} could not be loaded.".format(ff_name)

                    # Otherwise, just wait one more pause interval
                    else:
                        time.sleep(self.slideshow_pause)
                        continue


                # Clear the plot
                plt.clf()

                # Update the image on the screen
                self.updateImage(img, text, banner_text=self.banner_text)

                first_run = False



    def monitorDir(self):
        """ Monitor the given directory and show new FF files on the screen. """

        # Create a list of FF files in the given directory
        ff_list = []


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
                ff = readFF(self.dir_path, new_ff, verbose=False)

                if ff is not None:
                    img = ff.maxpixel

                else:
                    time.sleep(self.update_interval)
                    continue

                # Clear the plot
                plt.clf()
                showing_empty = False

                # Add new FF files to the list
                ff_list += new_ffs

            # If there are no FF files, wait
            else:
                if showing_empty is not None:
                    time.sleep(self.update_interval)
                    continue


            if showing_empty is not True:
                self.updateImage(img, text, banner_text=self.banner_text)


            # Set the proper flag if not showing any FF files
            if showing_empty is None:
                showing_empty = True



    def run(self):
        """ Main processing loop. """


        # Init the plot
        self.initPlot()


        if self.slideshow:
            self.startSlideshow()

        else:
            self.monitorDir()



    def stop(self):
        self.exit.set()




if __name__ == "__main__":

    import argparse

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Monitor the given folder for new FF files and show them on the screen.")

    arg_parser.add_argument('dir_path', metavar='DIR_PATH', type=str, \
        help="Directory to monitor for FF files.")

    arg_parser.add_argument('-s', '--slideshow', action="store_true", \
        help="Start a slide show (infinite repeat) of FF files in the given folder.")

    arg_parser.add_argument('-p', '--pause', metavar='SLIDESHOW_PAUSE', default=2, type=float, \
        help="Pause between frames in slideshow. 2 seconds by deault.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    lv = LiveViewer(cml_args.dir_path, slideshow=cml_args.slideshow, slideshow_pause=cml_args.pause)
    lv.start()