""" Monitors a given directory for FF files, and shows them on the screen as new ones get created. It can
    also do slideshows.
"""

from __future__ import print_function, division, absolute_import


import os
import time
import platform
import multiprocessing

import cv2
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
# tkinter import that works on both Python 2 and 3
try:
    import tkinter
except:
    import Tkinter as tkinter

from RMS.Formats.FFfile import read as readFF
from RMS.Formats.FFfile import validFFName
from RMS.Routines.Image import loadImage


# Load the default font
PIL_FONT = ImageFont.load_default()



def drawText(img_arr, img_text):
    """ Draws text on the image represented as a numpy array.
    """

    # Convert the array to PIL image
    im = Image.fromarray(np.uint8(img_arr))
    im = im.convert('RGB')
    draw = ImageDraw.Draw(im)

    # Draw the text on the image, in the upper left corent
    draw.text((0, 0), img_text, (255,255,0), font=PIL_FONT)
    draw = ImageDraw.Draw(im)

    # Convert the type of the image to grayscale, with one color
    try:
        if len(img_arr[0][0]) != 3:
            im = im.convert('L')
    except:
        im = im.convert('L')

    im = np.array(im)
    del draw

    return im


class LiveViewer(multiprocessing.Process):
    def __init__(self, dir_path, image=False, slideshow=False, slideshow_pause=2.0, banner_text="", \
        update_interval=5.0):
        """ Monitors a given directory for FF files, and shows them on the screen as new ones get created. It can
            also do slideshows. 

        Arguments:
            dir_path: [str] Directory to monitor for new FF files.
            
        Keyword arguments:
            image: [bool] Monitor a single image file and show on the screen as it updates.
            slideshow: [bool] Start a slide show instead of monitoring the folder for new files.
            slideshow_pause: [float] Number of seconds between slideshow updated. 2 by default.
            banner_text: [str] Banner text that will be shown on the screen.
            update_interval: [float] Number of seconds for checking the given directory for new files.
        """

        super(LiveViewer, self).__init__()

        self.dir_path = dir_path

        self.image = image
        self.slideshow = slideshow
        self.slideshow_pause = slideshow_pause
        self.banner_text = banner_text
        self.update_interval = update_interval

        self.exit = multiprocessing.Event()

        self.first_image = True



    def updateImage(self, img, text, pause_time, banner_text=""):
        """ Update the image on the screen. 
        
        Arguments:
            img: [2D ndarray] Image to show on the screen as a numpy array.
            text: [str] Text that will be printed on the image.
        """

        img = drawText(img, text)

        if not banner_text:
            banner_text = "LiveViewer"

        # Update the image on the screen
        cv2.imshow(banner_text, img)

        # If this is the first image, move it to the upper left corner
        if self.first_image:
            
            cv2.moveWindow(banner_text, 0, 0)

            self.first_image = False


        cv2.waitKey(int(1000*pause_time))


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


                # Update the image on the screen
                self.updateImage(img, text, self.slideshow_pause, banner_text=self.banner_text)

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

                showing_empty = False

                # Add new FF files to the list
                ff_list += new_ffs

            # If there are no FF files, wait
            else:
                if showing_empty is not None:
                    time.sleep(self.update_interval)
                    continue


            if showing_empty is not True:
                self.updateImage(img, text, self.update_interval, banner_text=self.banner_text)


            # Set the proper flag if not showing any FF files
            if showing_empty is None:
                showing_empty = True


    def showImage(self):
        """ Show one image file on the screen and refresh it in a given interval. """

        # Repeat until the process is killed from the outside
        first_run = True
        while not self.exit.is_set():

            if os.path.isfile(self.dir_path):
                
                # Load the image
                try:
                    img = loadImage(self.dir_path)
                    text = ""
                    
                except:
                    img = np.zeros((720, 1280), dtype='uint8')
                    text = "The image {:s} could not be loaded.".format(self.dir_path)

            else:
                # If an FF files could not be loaded on the first run, show an empty image
                if first_run:
                    img = np.zeros((720, 1280), dtype='uint8')
                    text = "The image {:s} could not be loaded.".format(self.dir_path)

                # Otherwise, just wait one more pause interval
                else:
                    time.sleep(self.slideshow_pause)
                    continue

            self.updateImage(img, text, self.update_interval)

            first_run = False


    def run(self):
        """ Main processing loop. """

        # Try setting the process niceness (available only on Unix systems)
        try:
            os.nice(20)
            print('Set low priority for the LiveViewer thread!')
        except Exception as e:
            print('Setting niceness failed with message:\n' + repr(e))


        # Show image if a file is given
        if os.path.isfile(self.dir_path) or self.image:
            self.showImage()

        elif os.path.isdir(self.dir_path):

            if self.slideshow:
                self.startSlideshow()

            else:
                self.monitorDir()

        else:
            self.exit.set()
            return None



    def stop(self):
        self.exit.set()




if __name__ == "__main__":

    import argparse

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Monitor the given folder for new FF files and show them on the screen.")

    arg_parser.add_argument('dir_path', metavar='DIR_PATH', type=str, \
        help="Directory to monitor for FF files.")

    arg_parser.add_argument('-i', '--image', action="store_true", \
        help="Monitor an image file and show on the screen.")

    arg_parser.add_argument('-s', '--slideshow', action="store_true", \
        help="Start a slide show (infinite repeat) of FF files in the given folder.")

    arg_parser.add_argument('-p', '--pause', metavar='SLIDESHOW_PAUSE', default=2, type=float, \
        help="Pause between frames in slideshow. 2 seconds by default.")

    arg_parser.add_argument('-u', '--update', metavar='UPDATE_INTERVAL', default=5, type=float, \
        help="Time between image refreshes. 5 seconds by default.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    lv = LiveViewer(cml_args.dir_path, image=cml_args.image, slideshow=cml_args.slideshow, \
        slideshow_pause=cml_args.pause, update_interval=cml_args.update)
    lv.start()