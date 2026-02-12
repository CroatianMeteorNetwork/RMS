""" Monitors a given directory for FF files, and shows them on the screen as new ones get created. It can
    also do slideshows.
"""

from __future__ import print_function, division, absolute_import

import fnmatch
import os
import time
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

import datetime
import RMS.ConfigReader as cr

# Load the default font
PIL_FONT = ImageFont.load_default()



def drawText(img_arr, img_text, color=(255,255,0)):
    """ Draws text on the image represented as a numpy array.
    """

    # Convert the array to PIL image
    im = Image.fromarray(np.uint8(img_arr))
    im = im.convert('RGB')
    draw = ImageDraw.Draw(im)

    # Draw the text on the image, in the upper left corner
    draw.text((0, 0), img_text, color, font=PIL_FONT)
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
    def __init__(self, dir_path=None, config=None, image=False, capturing=False, slideshow=False,
                 slideshow_pause=2.0, banner_text="", update_interval=5.0):
        """ Monitors a given directory for FF files, and shows them on the screen as new ones get created. It can
            also do slideshows. 


        Arguments:

            
        Keyword arguments:
            dir_path: [str] Directory to monitor for new FF files, optional, default None.
            config: [config] RMS config instance, optional, default None.
            image: [bool] Monitor a single image file and show on the screen as it updates, default False.
            slideshow_pause: [float] Number of seconds between slideshow updated. 2 by default.
            banner_text: [str] Banner text that will be shown on the screen.
            update_interval: [float] Number of seconds for checking the given directory for new files.
        """

        super(LiveViewer, self).__init__()

        if dir_path is None:
            self.dir_path = dir_path
        else:
            self.dir_path = os.path.expanduser(dir_path)

        self.image = image
        self.slideshow = slideshow
        self.slideshow_pause = slideshow_pause
        self.banner_text = banner_text
        self.update_interval = update_interval

        self.exit = multiprocessing.Event()

        self.first_image = True
        self.config = config
        self.capturing = capturing



    def updateImage(self, img, text, pause_time, banner_text="", color=(255,255,0)):
        """ Update the image on the screen. 
        
        Arguments:
            img: [2D ndarray] Image to show on the screen as a numpy array.
            text: [str] Text that will be printed on the image.
            pause_time: [float] Time to wait on the image - seconds

        Keyword Arguments:
            banner_text: [str] Banner text that will be shown at the top of the window, optional default ""

        Return:
            Nothing
        """

        img = drawText(img, text, color=color)

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
        if os.path.isdir(self.dir_path):
            ff_list = [file_name for file_name in sorted(os.listdir(self.dir_path)) if validFFName(file_name)]
        elif os.path.isfile(self.dir_path):
            ff_list = [self.dir_path]
        else:
            self.exit.set()

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

    def monitorFramesDirAndSlideshow(self):
        """ Show the latest frames files, and display a slideshow last 48 hours of detections. """

        frame_interval = self.config.frame_save_aligned_interval


        # Get screen resolution

        root = tkinter.Tk()

        width, height = root.winfo_screenwidth(), root.winfo_screenheight()
        print(width, height)

        # Initialise two windows

        if self.config.live_maxpixel_enable:
            cc_w_handle = "Continuous Capture"
            cv2.namedWindow(cc_w_handle)

        if self.config.slideshow_enable and not self.monitorFramesDirOnly:
            ss_w_handle = "Slideshow of detections from past 48 hours"
            cv2.namedWindow(ss_w_handle)

        _cc_file_to_show, slideshow_index = None, 0

        ff_file_list = []
        while not self.exit.is_set():

            # Get the time now
            dt_now = datetime.datetime.now(tz=datetime.timezone.utc)

            # When both are enabled, split the pause time between the two images
            if self.config.slideshow_enable and (self.config.live_maxpixel_enable and not self.monitorFramesDirOnly):
                pause = self.slideshow_pause * 0.5
                cv2.moveWindow(cc_w_handle, int(width * 0.05), int(height * 0.25))
                cv2.moveWindow(ss_w_handle, int(width * 0.55), int(height * 0.25))

            # If they are different, then only one must be enabled, so give both the full pause time
            elif self.config.slideshow_enable != self.config.live_maxpixel_enable:
                pause = self.slideshow_pause
            # Otherwise
            else:
                pause = self.slideshow_pause

            if self.config.slideshow_enable and not self.monitorFramesDirOnly:
                #### Slideshow work start

                # Build a new slideshow only after iterating through all the previous slides, or on first iteration
                if slideshow_index == 0:
                    ff_file_list, dir_list = [], []
                    archived_dir_path = os.path.join(self.config.data_dir, self.config.archived_dir)
                    archived_dir_contents = sorted(os.listdir(archived_dir_path), reverse=True)

                    # Iterate through in reverse order, adding before checking, so that we always get one extra dir to work with
                    for archived_dir in fnmatch.filter(archived_dir_contents, f"{self.config.stationID.upper()}_*_*_*"):
                        full_path_to_matched_dir = os.path.join(archived_dir_path, archived_dir)
                        if not os.path.isdir(full_path_to_matched_dir):
                            continue

                        dir_date, dir_time = archived_dir.split("_")[1], archived_dir.split("_")[2]
                        dir_time_object = datetime.datetime.strptime(f"{dir_date}_{dir_time}",
                                                                      "%Y%m%d_%H%M%S").replace(tzinfo=datetime.timezone.utc)
                        if (dt_now - dir_time_object).total_seconds() < 48 * 60 * 60:
                            dir_list.append(full_path_to_matched_dir)

                    dir_list.sort()

                    for dir in dir_list:
                        for root, dirs, files in os.walk(os.path.join(dir)):
                            for ff_file in fnmatch.filter(files, f"FF_{self.config.stationID.upper()}_*_*_*_*.fits"):
                                file_date, file_time = ff_file.split("_")[2], ff_file.split("_")[3]
                                file_time_object = datetime.datetime.strptime(f"{file_date}_{file_time}", "%Y%m%d_%H%M%S").replace(
                                    tzinfo=datetime.timezone.utc)
                                if (dt_now - file_time_object).total_seconds() < 48 * 60 * 60:
                                    ff_file_list.append(os.path.join(root, ff_file))
                    ff_file_list.sort()

                if not len(ff_file_list):
                    print("No FF files found in the previous 48 hours, waiting one hour")
                    time.sleep(3600)

                # This will guard against iterating over an empty list
                if slideshow_index < len(ff_file_list):
                    ff_file_to_show = ff_file_list[slideshow_index]
                    slideshow_index += 1
                    image_annotation = f"{os.path.basename(ff_file_to_show)}  #{slideshow_index}/{len(ff_file_list)}"

                    # Now plot the detection.maxpixel
                    ff_data = readFF(os.path.dirname(ff_file_to_show),
                                 os.path.basename(ff_file_to_show),
                                 verbose=False)

                    if ff_data is None or not hasattr(ff_data, "maxpixel"):
                        # Remove path to invalid file from list
                        ff_file_list.pop(slideshow_index)
                        continue
                    else:
                        # Extract the maxpixel
                        img = ff_data.maxpixel

                    if self.config.live_maxpixel_enable:
                        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
                    self.updateImage(img, image_annotation, pause, ss_w_handle)
                else:
                    # This will trigger rebuilding the slideshow on the next iteration
                    slideshow_index = 0

                #### Slideshow work end

            if self.config.live_maxpixel_enable:

                #### Continuous capture live image work start

                # Compute target_dt, which is the datetime object of the target image
                # Pushing time back 240 seconds to cope with the delay in continuous capture
                target_dt = dt_now - datetime.timedelta(seconds=240)

                # Handle all the file paths
                frame_dir_root = os.path.join(self.config.data_dir, self.config.frame_dir)
                l1_dir = str(target_dt.year)
                l2_dir = str(target_dt.strftime("%Y%m%d-%j"))
                l3_dir = str(target_dt.strftime("%Y%m%d-%j_%H"))
                target_dir = os.path.join(frame_dir_root, l1_dir, l2_dir, l3_dir)

                if not os.path.exists(target_dir):
                    time.sleep(5)
                    continue

                latest_file_list = sorted(os.listdir(target_dir))

                # Find the image which is closest to the target time
                time_deviation_list = []
                for file_name in latest_file_list:
                    file_date, file_time = file_name.split("_")[1], file_name.split("_")[2]
                    file_time_object = datetime.datetime.strptime(f"{file_date}_{file_time}","%Y%m%d_%H%M%S").replace(tzinfo=datetime.timezone.utc)
                    time_deviation_list.append(abs((file_time_object - target_dt).total_seconds()))
                min_deviation = min(time_deviation_list)
                min_deviation_index = time_deviation_list.index(min_deviation)
                cc_file_to_show = os.path.join(target_dir, latest_file_list[min_deviation_index])

                if os.path.exists(cc_file_to_show):
                    if os.path.isfile(cc_file_to_show):

                        # Check file is not still being written
                        _size = None
                        size = os.path.getsize(cc_file_to_show)
                        while _size != size:
                            _size = size
                            time.sleep(1)
                            size = os.path.getsize(cc_file_to_show)

                        name, _ = os.path.splitext(os.path.basename(cc_file_to_show))
                        last_char = name[-1]

                        # Plot in black by day, and white by night
                        if last_char == 'd':
                            color = (0,0,0)
                        else:
                            color = (255,255,255)

                        # Show the file if it is the first iteration
                        if _cc_file_to_show is None:
                            if cc_file_to_show != _cc_file_to_show:
                                img = np.array(Image.open(cc_file_to_show))
                                if self.config.slideshow_enable:
                                    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
                                self.updateImage(img, os.path.basename(cc_file_to_show), pause, cc_w_handle, color=color)

                        # Or if it is different from the last iteration
                        elif _cc_file_to_show != cc_file_to_show:
                            img = np.array(Image.open(cc_file_to_show))
                            if self.config.slideshow_enable:
                                img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
                            self.updateImage(img, os.path.basename(cc_file_to_show), pause , cc_w_handle, color=color)
                        _cc_file_to_show = cc_file_to_show

                #### Continuous capture live image work end

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

        self.monitorFramesDirOnly = False
        if self.dir_path is not None:
            if os.path.exists(self.dir_path):

                if os.path.expanduser(self.dir_path) == (os.path.expanduser(os.path.join(self.config.data_dir, self.config.frame_dir))):
                    self.monitorFramesDirOnly = True
                    self.monitorFramesDirAndSlideshow()
                if self.slideshow or os.path.isdir(self.dir_path):
                    self.startSlideshow()
                elif self.image or os.path.isfile(self.dir_path):
                    self.showImage()

        if self.config.continuous_capture:
            # Work with frames directory

            self.monitorFramesDirAndSlideshow()

        else:

            if self.capturing and self.config.live_maxpixel_enable:
                captured_dir_path = os.path.join(self.config.data_dir, self.config.captured_dir)
                if os.path.exists(captured_dir_path):
                    if os.path.isdir(captured_dir_path):
                        captured_dir_list = os.listdir(captured_dir_path)
                        if len(captured_dir_list):
                            latest_captured_dir = sorted(fnmatch.filter(captured_dir_list, f"{self.config.stationID.upper()}_*_*_*"))[-1]
                            self.dir_path = os.path.join(captured_dir_path, latest_captured_dir)
                            if os.path.exists(self.dir_path):
                                if os.path.isdir(self.dir_path):
                                    self.monitorDir()


            elif not self.capturing and self.config.slideshow_enable:
                archived_dir_path = os.path.join(self.config.data_dir, self.config.archived_dir)
                if os.path.exists(archived_dir_path):
                    if os.path.isdir(archived_dir_path):
                        archived_dir_list = [d for d in os.listdir(archived_dir_path) if os.path.isdir(os.path.join(archived_dir_path,d))]
                        if len(archived_dir_list):
                            latest_archive_dir = sorted(fnmatch.filter(archived_dir_list, f"{self.config.stationID.upper()}_*_*_*"))[-1]
                            self.dir_path = os.path.join(archived_dir_path, latest_archive_dir)
                            if os.path.exists(self.dir_path):
                                if os.path.isdir(self.dir_path):
                                    self.startSlideshow()

            else:
                self.exit.set()
                return None



    def stop(self):
        self.exit.set()
        self.join()




if __name__ == "__main__":

    import argparse

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Monitor the given folder for new FF files and show them on the screen.")

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
        help="Path to a config file which will be used instead of the default one.")

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

    # Load the config file
    if cml_args.config is None:
        config = None
    else:
        config = cr.loadConfigFromDirectory(cml_args.config, os.path.abspath('.'))


    lv = LiveViewer(cml_args.dir_path, config=config, image=cml_args.image, slideshow=cml_args.slideshow, \
        slideshow_pause=cml_args.pause, update_interval=cml_args.update)
    lv.start()

