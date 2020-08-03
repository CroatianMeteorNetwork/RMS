"""Tool for manual astrometry and photometry of FF and FR files. """

from __future__ import print_function, division, absolute_import

import os
import sys
import math
import argparse
import datetime
import pytz
import json

# tkinter import that works on both Python 2 and 3
try:
    from tkinter import messagebox
except:
    import tkMessageBox as messagebox


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties

import RMS.ConfigReader as cr
from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP, applyAstrometryFTPdetectinfo
from RMS.Astrometry.Conversions import J2000_JD, datetime2JD, jd2Date, raDec2AltAz
from RMS.Formats.FFfile import filenameToDatetime
from RMS.Formats.FRbin import read as readFR
from RMS.Formats.FRbin import validFRName
from RMS.Formats.FTPdetectinfo import writeFTPdetectinfo
from RMS.Formats.FrameInterface import detectInputType
from RMS.Formats.Platepar import Platepar
from RMS.Misc import openFileDialog
from RMS.Pickling import loadPickle, savePickle
from RMS.Routines import Image
from RMS.Routines import RollingShutterCorrection

# Import Cython functions
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
from RMS.Astrometry.CyFunctions import equatorialCoordPrecession


# TkAgg has issues when opening an external file prompt, so other backends are forced if available
if matplotlib.get_backend() == 'TkAgg':

    backends = ['Qt5Agg', 'Qt4Agg', 'WXAgg']

    for bk in backends:

        # Try setting backend
        try:
            plt.switch_backend(bk)

        except:
            pass


print('Using backend: ', matplotlib.get_backend())


class Pick(object):
    def __init__(self):
        """ Container for picks per every frame. """

        self.frame = None
        self.x_centroid = None
        self.y_centroid = None

        self.photometry_pixels = None
        self.intensity_sum = None



class ManualReductionTool(object):
    def __init__(self, config, img_handle, fr_file, first_frame=None, fps=None, deinterlace_mode=-1, \
        station_name=None):
        """ Tool for manually picking meteor centroids and photometry.

        Arguments:
            config: [config] Configuration structure.
            img_handle: [object] Handle with provides common interface to various input formats.
            fr_file: [str] Path to the FR file, if given (can be None).
        Keyword Arguments:
            first_frame: [int] First frame to start with. None by default, which will start with the first one.
            fps: [float] Frames per second. None by default, which will read the fps from the config file.
            deinterlace_mode: [int]
                -1 - no deinterlace
                 0 - odd first
                 1 - even first
            station_name: [str] Station name. None by default, then 'manual' will be used.
        """


        self.config = config

        self.fps = fps

        self.deinterlace_mode = deinterlace_mode

        self.fr_file = fr_file


        # Compute the frame step
        if self.deinterlace_mode > -1:
            self.frame_step = 0.5
        else:
            self.frame_step = 1


        self.img_handle = img_handle

        self.ff = None
        self.fr = None


        self.station_name = station_name

        # If the station name was not given and the FF file is used, read it from the FF file name
        if self.img_handle is not None:
            if (self.station_name is None) and (self.img_handle.input_type == 'ff'):

                # Extract the station name from the FF file
                self.station_name = self.img_handle.current_ff_file.split("_")[1]

        # Otherwise extract the station code from the FR file
        else:
            self.station_name = os.path.basename(self.fr_file).split("_")[1]


        if self.station_name is None:
            self.station_name = 'manual'


        # If the image handle was given, load the first chunk as the FF file
        if self.img_handle is not None:
            self.ff = self.img_handle.loadChunk(read_nframes=-1)
            self.nframes = self.ff.nframes

            self.dir_path = self.img_handle.dir_path


        # Each FR bin can have multiple detections, the first one is by default
        self.current_line = 0

        self.fr = None


        # Take the FPS from the FF file, if available
        if self.ff is not None:
            if hasattr(self.ff, 'fps'):
                self.fps = self.ff.fps

        if self.fps is None:

            # Try reading FPS from image handle
            if self.img_handle is not None:
                self.fps = self.img_handle.fps

            else:
                # Otherwise, read FPS from config
                self.fps = self.config.fps


        print('Using FPS:', self.fps)


        # If there is only one frame, assume it's a static image, and enable adding more picks on the same
        #   image
        self.single_image_mode = False
        if self.img_handle is not None:
            if self.img_handle.total_frames == 1:
                self.single_image_mode = True


        ###########

        self.flat_struct = None
        self.dark = None


        # Load platepar
        _, self.platepar = self.loadPlatepar()


        # Image gamma and levels
        self.auto_levels = False
        self.bit_depth = self.config.bit_depth
        self.img_gamma = 1.0
        self.img_level_min = 0
        self.img_level_max = 2**self.bit_depth - 1


        self.show_maxpixel = False
        self.subtract_avepixel = False

        self.show_key_help = True

        self.current_image = None
        self.current_image_viewing = None

        self.fr_xmin = None
        self.fr_xmax = None
        self.fr_ymin = None
        self.fr_ymax = None

        # Previous zoom
        self.prev_xlim = None
        self.prev_ylim = None

        self.circle_aperture = None
        self.circle_aperture_outer = None
        self.aperture_radius = 5
        self.scroll_counter = 0

        self.mouse_x = None
        self.mouse_x_press = None
        self.mouse_y = None
        self.mouse_y_press = None

        self.centroid_handle = None

        self.photometry_coloring_mode = False
        self.photometry_coloring_color = False
        self.photometry_aperture_radius = 3
        self.photometry_add = True
        self.photometry_coloring_handle = None

        self.pick_list = []

        ###########


        if first_frame is not None:
            self.current_frame = first_frame%self.nframes

        else:
            self.current_frame = 0


        # Set the current frame in the image handle
        if self.img_handle is not None:
            self.img_handle.current_frame = self.current_frame

        # Only one image is used, start at frame 100, so some previous frames can be added
        if self.single_image_mode:
            self.current_frame = 100


        # Initialize matplotlib config
        self.initImage()



    def initImage(self):
        """ Initialize matplotlib configuration. """

        ### INIT IMAGE ###

        plt.figure(facecolor='black')

        # Init the first image
        self.updateImage(first_update=True)
        self.printStatus()

        self.ax = plt.gca()

        # Register keys with matplotlib
        self.registerEventHandling()


    def registerEventHandling(self):
        """ Register mouse button and key pressess with matplotlib. """

        # Set window title
        plt.gcf().canvas.set_window_title('RMS Manual Reduction')

        # Set the bacground color to black
        #matplotlib.rcParams['axes.facecolor'] = 'k'

        # Disable standard matplotlib keyboard shortcuts
        plt.rcParams['keymap.save'] = ''
        plt.rcParams['keymap.fullscreen'] = ''
        plt.rcParams['keymap.all_axes'] = ''
        plt.rcParams['keymap.quit'] = ''
        plt.rcParams['keymap.pan'] = ''
        plt.rcParams['keymap.xscale'] = ''


        # Register event handlers
        self.ax.figure.canvas.mpl_connect('key_press_event', self.onKeyPress)
        self.ax.figure.canvas.mpl_connect('key_release_event', self.onKeyRelease)
        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.onMouseMotion)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.onMousePress)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.onMouseRelease)
        self.ax.figure.canvas.mpl_connect('scroll_event', self.onScroll)


        # Set the status update formatter
        plt.gca().format_coord = self.mouseOverStatus





    def printStatus(self):
        """ Print the status message. """

        print('----------------------------')
        print('Frame:', self.current_frame)


        if self.img_handle is not None:

            # Image mode
            if self.img_handle.input_type == 'images':
                print('File:', self.img_handle.current_img_file)




    def loadDark(self):
        """ Open a file dialog and ask user to load a dark frame. """

        dark_file = openFileDialog(self.dir_path, None, 'Select the dark frame file', matplotlib)

        if not dark_file:
            return False, None


        try:

            # Byteswap the dark if vid file is used or UWO png
            byteswap = False
            if self.img_handle is not None:
                if self.img_handle.byteswap:
                    byteswap = True

            # Load the dark
            dark = Image.loadDark(*os.path.split(dark_file), dtype=self.current_image.dtype,
                byteswap=byteswap)

        except:
            messagebox.showerror(title='Dark frame error', \
                message='Dark frame could not be loaded!')
            return False, None


        dark = dark.astype(self.current_image.dtype)


        # Check if the size of the file matches
        if self.current_image.shape != dark.shape:
            messagebox.showerror(title='Dark frame file error', \
                message='The size of the dark frame does not match the size of the image!')

            dark = None

        # Check if the dark frame was successfuly loaded
        if dark is None:
            messagebox.showerror(title='Dark frame file error', \
                message='The file you selected could not be loaded as a dark frame!')


        return dark_file, dark


    def loadFlat(self):
        """ Open a file dialog and ask user to load a flat field. """


        # Check if flat exists in the folder, and set it as the defualt file name if it does
        if self.config.flat_file in os.listdir(self.dir_path):
            initialfile = self.config.flat_file
        else:
            initialfile = ''


        flat_file = openFileDialog(self.dir_path, initialfile, 'Select the flat field file', matplotlib)

        if not flat_file:
            return False, None


        # Byteswap the flat if vid file is used or UWO png
        byteswap = False
        if self.img_handle is not None:
            if self.img_handle.byteswap:
                byteswap = True


        try:
            # Load the flat. Byteswap the flat if vid file is used
            flat = Image.loadFlat(*os.path.split(flat_file), dtype=self.current_image.dtype, \
                byteswap=byteswap)

        except:
            messagebox.showerror(title='Flat field file error', \
                message='Flat could not be loaded!')
            return False, None


        # Check if the size of the file matches
        if self.current_image.shape != flat.flat_img.shape:
            messagebox.showerror(title='Flat field file error', \
                message='The size of the flat field does not match the size of the image!')

            flat = None

        # Check if the flat field was successfuly loaded
        if flat is None:
            messagebox.showerror(title='Flat field file error', \
                message='The file you selected could not be loaded as a flat field!')



        return flat_file, flat



    def loadPlatepar(self):
        """ Open a file dialog and ask user to open the platepar file. """


        platepar = Platepar()

        # Check if platepar exists in the folder, and set it as the defualt file name if it does
        if self.config.platepar_name in os.listdir(self.dir_path):
            initialfile = self.config.platepar_name
        else:
            initialfile = ''

        # Load the platepar file
        platepar_file = openFileDialog(self.dir_path, initialfile, 'Select the platepar file', matplotlib)


        if not platepar_file:
            return False, None


        # Parse the platepar file
        try:
            self.platepar_fmt = platepar.read(platepar_file, use_flat=self.config.use_flat)
        except:
            platepar = None

        # Check if the platepar was successfuly loaded
        if platepar is None:
            messagebox.showerror(title='Platepar file error', \
                message='The file you selected could not be loaded as a platepar file!')

            self.loadPlatepar()


        # Always turn refraction on in platepar that is being used
        if not platepar.refraction:
            platepar.refraction = True
            print("Refraction compensation turned ON!")

        return platepar_file, platepar



    def loadImage(self):
        """ Load the current frame and apply calibration.

        Return:
            img, process_img:
                - img [2D ndarray] Image for viewing.
                - process_img [2D ndarray] Image on which processing will be done.
        """

        # If FF is given, reconstruct frames
        if self.img_handle is not None:

            # Take the current frame from FF file
            img = self.img_handle.loadFrame(avepixel=True)



        # Show the maxpixel if the key has been pressed
        if self.show_maxpixel and self.ff is not None:

            img = self.ff.maxpixel


        # Apply the deinterlace
        if self.deinterlace_mode > -1:

            # Set the deinterlace index to handle proper deinterlacing order
            if self.deinterlace_mode == 0:
                deinter_indx = 0

            else:
                deinter_indx = 1


            # Deinterlace the image using the appropriate method
            if (self.current_frame + deinter_indx*0.5)%1 == 0:
                img = Image.deinterlaceOdd(img)

            else:
                img = Image.deinterlaceEven(img)


        # Store the image prior to calibration
        process_img = np.copy(img)

        # Subtract the average and apply the flat field for image on which processing will be done
        if self.ff is not None:

            # Subtract the average without flat correction (only when more images are available)
            if not self.single_image_mode:
                process_img = Image.applyDark(process_img, self.ff.avepixel)


            # Apply flat
            if self.flat_struct is not None:
                process_img = Image.applyFlat(process_img, self.flat_struct)


        # Apply dark and flat (cannot be applied if there is no FF file) on the image for showing
        if self.ff is not None:

            # Subtract average to remove background stars (don't apply the dark then)
            if self.subtract_avepixel:

                img = Image.applyDark(img, self.ff.avepixel)

            else:

                # Apply dark
                if self.dark is not None:
                    img = Image.applyDark(img, self.dark)


            # Apply flat
            if (self.flat_struct is not None):
                img = Image.applyFlat(img, self.flat_struct)


        return img, process_img




    def updateImage(self, first_update=False):
        """ Updates the current plot. """

        # Reset circle patches
        self.circle_aperature = None
        self.circle_aperature_outer = None

        # Reset centroid patch
        self.centroid_handle = None

        # Reset photometry coloring
        self.photometry_coloring_handle = None

        # Save the previous zoom
        if self.current_image is not None:

            self.prev_xlim = plt.gca().get_xlim()
            self.prev_ylim = plt.gca().get_ylim()


        plt.clf()

        # Set the status update formatter
        plt.gca().format_coord = self.mouseOverStatus


        # Load the image
        #   img - image for viewing
        #   process_img - image for processing (centroiding, photometry) which has the average subtracted
        img, process_img = self.loadImage()

        # Image for processing
        self.current_image = process_img

        # Image for viewing before levels correction
        self.current_image_viewing = np.copy(img)


        if first_update:

            # Guess the bit depth from the array type
            self.bit_depth = 8*img.itemsize

            # Set the maximum image level after reading the bit depth
            self.img_level_max = 2**self.bit_depth - 1


        # Do auto levels
        if self.auto_levels:

            # Compute the edge percentiles (skip the first 2 rows)
            min_lvl = np.percentile(img[2:, :], 1)
            max_lvl = np.percentile(img[2:, :], 99.9)


            # Adjust levels (auto)
            img = Image.adjustLevels(img, min_lvl, self.img_gamma, max_lvl)

        else:

            # Adjust levels (manual)
            img = Image.adjustLevels(img, self.img_level_min, self.img_gamma, self.img_level_max)



        plt.imshow(img, cmap='gray', interpolation='nearest')

        if (self.prev_xlim is not None) and (self.prev_ylim is not None):

            # Restore previous zoom
            plt.xlim(self.prev_xlim)
            plt.ylim(self.prev_ylim)


        self.drawText()

        # Don't draw the picks in the photometry coloring more
        if not self.photometry_coloring_mode:

            # Plot image pick
            self.drawPicks(update_plot=False)


        # Plot the photometry coloring
        self.drawPhotometryColoring(update_plot=False)

        plt.gcf().canvas.draw()



    def drawText(self):
        """ Draw the text on the image. """

        # Setup a monospace font
        font = FontProperties()
        font.set_family('monospace')
        font.set_size(8)


        if self.show_key_help:


            # Draw info text

            # Generate image status text
            if self.show_maxpixel:
                text_str = 'maxpixel\n'

            else:

                text_str  = "Station name: {:s}\n".format(self.station_name)
                text_str += "Frame = {:.1f}\n".format(self.current_frame)

                # Print frame time
                if self.img_handle is not None:
                    text_str += "Time  = {:s}\n".format(self.img_handle.currentFrameTime(dt_obj=True).strftime("%Y%m%d %H:%M:%S.%f")[:-3])



            text_str += "Image gamma  = {:.2f}\n".format(self.img_gamma)
            text_str += "Camera gamma = {:.2f}\n".format(self.config.gamma)

            if self.platepar is not None:
                text_str += "Refraction   = {:s}\n".format(str(self.platepar.refraction))


            # Add info about applied image corrections
            if self.subtract_avepixel:

                text_str += 'Subtracted average'

            else:

                # Add info about dark and flats
                if self.dark is not None:

                    text_str += 'Dark'

                    if self.flat_struct is not None:
                        text_str += " + Flat\n"

                    else:
                        text_str += "\n"

                else:

                    if self.flat_struct is not None:
                        text_str += "Flat\n"




            # Get the current plot limit
            x_min, x_max = plt.gca().get_xlim()
            y_max, y_min = plt.gca().get_ylim()

            plt.gca().text(x_min + 10, y_min + 10, text_str, color='w', verticalalignment='top', \
                horizontalalignment='left', fontproperties=font)


            # Show text on image with instructions
            text_str  = 'Keys:\n'
            text_str += '-----------\n'
            text_str += 'Left/Right - Previous/next frame\n'
            text_str += 'Page Down/Up - +/- 25 frames\n'
            text_str += ',/. - Previous/next FR line\n'
            text_str += '+/- - Zoom in/out\n'
            text_str += 'R - Reset view\n'
            text_str += 'M - Show maxpixel\n'
            text_str += 'K - Subtract average\n'
            text_str += 'T - Toggle refraction correction\n'
            text_str += 'U/J - Img Gamma\n'
            text_str += 'P - Show lightcurve\n'
            text_str += 'CTRL + A - Auto levels\n'
            text_str += 'CTRL + D - Load dark\n'
            text_str += 'CTRL + F - Load flat\n'
            text_str += 'CTRL + P - Load platepar\n'
            text_str += 'CTRL + W - Save current frame\n'
            text_str += 'CTRL + S - Save FTPdetectinfo\n'
            text_str += '\n'
            text_str += 'Mouse:\n'
            text_str += '-----------\n'
            text_str += 'Left click - Centroid\n'
            text_str += 'CTRL + Left click - Manual pick\n'
            text_str += 'SHIFT + Left click - Photometry coloring add\n'
            text_str += 'SHIFT + Right click - Photometry coloring remove\n'
            text_str += 'Mouse scroll - Annulus size\n'
            text_str += '\n'
            text_str += 'F1 - Hide/show text'


            plt.gca().text(10, self.current_image.shape[0] - 5, text_str, color='w',
                verticalalignment='bottom', horizontalalignment='left', fontproperties=font)

        else:

            text_str = "Show text - F1"

            plt.gca().text(self.current_image.shape[1]/2, self.current_image.shape[0], text_str, color='w',
                verticalalignment='top', horizontalalignment='center', fontproperties=font, alpha=0.5)



    def getCurrentFrameTime(self, frame_no=None):
        """ Returns the time of the current frame.

        Keyword arguments:
            frame_no: [float] Frame for which to compute the time. None by default which returns the time
                of the current frame.
        """


        if self.img_handle is not None:

            # Get mean time
            time_data = self.img_handle.currentFrameTime(frame_no=frame_no)

        else:

            if frame_no is None:
                frame_no = self.current_frame

            # If there is no image handle, assume it's an FR file
            dt = filenameToDatetime(os.path.basename(self.fr_file)) \
                + datetime.timedelta(seconds=frame_no/float(self.fps))

            time_data = (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, \
                dt.microsecond/1000)


        return time_data


    def showLightcurve(self):
        """ Show the meteor lightcurve. """

        # Compute the intensity sum done on the previous frame
        self.computeIntensitySum()


        # Create the list of picks for saving
        centroids = []
        for pick in self.pick_list:

            # Make sure to centroid is picked and is not just the photometry
            if pick.x_centroid is None:
                continue

            centroids.append([pick.frame, pick.x_centroid, pick.y_centroid, pick.intensity_sum])


        # If there are less than 3 points, don't show the lightcurve
        if len(centroids) < 3:
            messagebox.showinfo('Lightcurve info', 'Less than 3 centroids!')
            return 1

        # Sort by frame number
        centroids = sorted(centroids, key=lambda x: x[0])

        # Extract frames and intensities
        fr_intens = [line for line in centroids if line[3] > 0]


        # If there are less than 3 points, don't show the lightcurve
        if len(fr_intens) < 3:
            messagebox.showinfo('Lightcurve info', 'Less than 3 points have intensities!')
            return 1


        # Extract frames and intensities
        frames, x_centroids, y_centroids, intensities = np.array(fr_intens).T


        # Init plot
        fig_p = plt.figure(facecolor=None)
        ax_p = fig_p.add_subplot(1, 1, 1)


        # If the platepar is available, compute the magnitudes, otherwise show the instrumental magnitude
        if self.platepar is not None:

            time_data = [self.getCurrentFrameTime()]*len(intensities)

            # Compute the magntiudes
            _, _, _, mag_data = xyToRaDecPP(time_data, x_centroids, y_centroids, intensities, self.platepar)


            # Plot the magnitudes
            ax_p.errorbar(frames, mag_data, yerr=self.platepar.mag_lev_stddev, capsize=5, color='k')


            if 'BSC' in self.config.star_catalog_file:
                mag_str = "V"

            elif 'gaia' in self.config.star_catalog_file.lower():
                mag_str = 'GAIA G band'

            else:
                mag_str = "{:.2f}B + {:.2f}V + {:.2f}R + {:.2f}I".format(*self.config.star_catalog_band_ratios)


            ax_p.set_ylabel("Apparent magnitude ({:s})".format(mag_str))

        else:


            # Compute the instrumental magnitude
            inst_mag = -2.5*np.log10(intensities)



            # Plot the magnitudes
            ax_p.plot(frames, inst_mag)

            ax_p.set_ylabel("Instrumental magnitude")



        ax_p.set_xlabel("Frame")

        ax_p.invert_yaxis()
        #ax_p.invert_xaxis()

        ax_p.grid()

        fig_p.show()



    def onKeyPress(self, event):
        """ Handles key presses. """


        # Cycle frames
        if event.key == 'left':
            self.prevFrame()

        elif event.key == 'right':
            self.nextFrame()


        # +/- 25 frames
        elif event.key == 'pagedown':

            self.prevFrame()

            for i in range(23):
                self.prevFrame(only_number_update=True)

            self.prevFrame()


        elif event.key == 'pageup':

            self.nextFrame()

            for i in range(23):
                self.nextFrame(only_number_update=True)

            self.nextFrame()


        # Zoom in/out
        elif event.key == '+':

            self.zoomImage(zoom_in=True)

            self.updateImage()

        elif event.key == '-':

            self.zoomImage(zoom_in=False)

            self.updateImage()


        # Previous line
        elif event.key == ',':

            if self.fr is not None:

                self.current_line -= 1

                self.current_line = self.current_line%self.fr.lines

                self.printStatus()


        # Next line
        elif event.key == '.':

            if self.fr is not None:

                self.current_line += 1

                self.current_line = self.current_line%self.fr.lines

                self.printStatus()



        # Show/hide keyboard shortcut help
        elif event.key == 'f1':
            self.show_key_help = not self.show_key_help
            self.updateImage()


        # Increase image gamma
        elif event.key == 'u':

            # Increase image gamma by a factor of 1.1x
            self.updateGamma(1.1)

        elif event.key == 'j':

            # Decrease image gamma by a factor of 0.9x
            self.updateGamma(0.9)


        # Toggle refraction
        elif event.key == 't':

            if self.platepar is not None:

                self.platepar.refraction = not self.platepar.refraction

                self.updateImage()


        # Show maxpixel instead of individual frames
        elif event.key == 'm':

            self.show_maxpixel = not self.show_maxpixel

            self.updateImage()


        # Subtract average pixel image to remove background stars
        elif event.key == 'k':

            # Only in multiple image mode
            if not self.single_image_mode:

                self.subtract_avepixel = not self.subtract_avepixel

                self.updateImage()

            else:
                print('The average cannot be subtracted in the single image mode!')



        elif event.key == 'r':

            # Reset the plot limits
            plt.xlim(0, self.current_image.shape[1])
            plt.ylim(self.current_image.shape[0], 0)

            self.updateImage()



        # Show the lightcurve
        elif event.key == 'p':

            self.showLightcurve()


        elif event.key == 'ctrl+w':

            # Save current frame to disk as image
            self.saveCurrentFrame()


        elif event.key == 'ctrl+s':

            # Save the FTPdetectinfo file
            self.saveFTPdetectinfo()

            # Save the state of the program
            self.saveState()

            # Save the JSON file with the picks
            self.saveJSON()


        # Load the dark frame
        elif event.key == 'ctrl+d':
            _, self.dark = self.loadDark()


            # Apply the dark to the flat
            if self.flat_struct is not None:
                self.flat_struct.applyDark(self.dark)

            self.updateImage()

            # Recompute the image intensities
            self.recomputeAllIntensitySums()


        # Load the flat field
        elif event.key == 'ctrl+f':
            _, self.flat_struct = self.loadFlat()

            self.updateImage()

            # Recompute the image intensities
            self.recomputeAllIntensitySums()


        # Load the platepar
        elif event.key == 'ctrl+p':
            _, self.platepar = self.loadPlatepar()

            self.updateImage()


        # Toggle auto levels
        elif event.key == 'ctrl+a':
            self.auto_levels = not self.auto_levels

            self.updateImage()


        elif event.key == 'shift':

            # Toggle the photometry coloring mode
            if not self.photometry_coloring_mode:

                self.photometry_coloring_mode = True

                self.updateImage()

                # Change the position of the star aperture circle
                self.drawCursorCircle()



    def onKeyRelease(self, event):
        """ Handles key releases. """

        if event.key == 'shift':

            # Toggle the photometry coloring mode
            if self.photometry_coloring_mode:

                self.photometry_coloring_mode = False

                # Redraw the centroids
                self.drawPicks()



    def onMouseMotion(self, event):
        """ Called with the mouse is moved. """

        # Check if the mouse is within bounds
        if (event.xdata is not None) and (event.ydata is not None):

            # Read mouse position
            self.mouse_x = event.xdata
            self.mouse_y = event.ydata


            # Change the position of the star aperture circle
            self.drawCursorCircle()

            if self.photometry_coloring_mode and self.photometry_coloring_color:

                # Color in the pixels for photometry
                self.changePhotometry(self.current_frame, self.photometryColoring(), \
                    add_photometry=self.photometry_add)

                self.drawPhotometryColoring(update_plot=True)



    def onScroll(self, event):
        """ Change aperture on scroll. """

        self.scroll_counter += event.step


        if self.scroll_counter > 1:

            if self.photometry_coloring_mode:
                self.photometry_aperture_radius += 1
            else:
                self.aperture_radius += 1

            self.scroll_counter = 0

        elif self.scroll_counter < -1:

            if self.photometry_coloring_mode:
                self.photometry_aperture_radius -= 1
            else:
                self.aperture_radius -= 1

            self.scroll_counter = 0


        # Check that the centroid aperture is in the proper limits
        if self.aperture_radius < 2:
            self.aperture_radius = 2

        if self.aperture_radius > 250:
            self.aperture_radius = 250

        # Check that the photometry aperture is in the proper limits
        if self.photometry_aperture_radius < 2:
            self.photometry_aperture_radius = 2

        if self.photometry_aperture_radius > 250:
            self.photometry_aperture_radius = 250


        self.drawCursorCircle()



    def onMouseRelease(self, event):
        """ Called when the mouse click is released. """

        # Photometry coloring - off
        if ((event.button == 1) or (event.button == 3)) and (event.key == 'shift'):
            self.photometry_coloring_color = False

        # Call the same function for mouse movements to update the variables in the background
        self.onMouseMotion(event)


        # Left mouse button, centroid or CTRL is not pressed
        if (event.button == 1) and (event.key != 'shift'):

            # Remove the old centroid scatter plot handle
            if self.centroid_handle is not None:
                self.centroid_handle.remove()

            # Centroid the star around the pressed coordinates
            self.x_centroid, self.y_centroid, _ = self.centroid()

            # If CTRL is pressed, place the pick manually - NOTE: the intensity might be off then!!!
            # 'control' is for Windows, 'ctrl+control' is for Linux
            if (event.key == 'control') or (event.key == 'ctrl+control'):
                self.x_centroid = self.mouse_x_press
                self.y_centroid = self.mouse_y_press


            # Add the centroid to the list
            self.addCentroid(self.current_frame, self.x_centroid, self.y_centroid)

            self.updateImage()


        # Remove centroid on right click
        if (event.button == 3) and (event.key != 'shift'):
            self.removeCentroid(self.current_frame)

            self.updateImage()



    def onMousePress(self, event):
        """ Called on mouse click press. """

        # Check if the mouse is within bounds
        if (event.xdata is not None) and (event.ydata is not None):

            # Store the mouse press location
            self.mouse_x_press = event.xdata
            self.mouse_y_press = event.ydata


        # Photometry coloring - on
        if ((event.button == 1) or (event.button == 3)) and (event.key == 'shift'):
            self.photometry_coloring_color = True

            # Color photometry pixels
            if (event.button == 1):
                self.photometry_add = True

            # Remove pixels
            else:
                self.photometry_add = False


            if self.photometry_coloring_mode and self.photometry_coloring_color:

                # Color in the pixels for photometry
                self.changePhotometry(self.current_frame, self.photometryColoring(), \
                    add_photometry=self.photometry_add)
                self.drawPhotometryColoring(update_plot=True)



    def mouseOverStatus(self, x, y):
        """ Format the status message which will be printed in the status bar below the plot.
        Arguments:
            x: [float] Plot X coordiante.
            y: [float] Plot Y coordinate.
        Return:
            [str]: formatted output string to be written in the status bar
        """


        # Write image X, Y coordinates and image intensity
        status_str = "x={:7.2f}  y={:7.2f}  Intens={:d}".format(x, y, self.current_image_viewing[int(y), int(x)])

        # Add coordinate info if platepar is present
        if self.platepar is not None:

            # Get the current frame time
            time_data = [self.getCurrentFrameTime()]

            # Compute RA, dec
            jd, ra, dec, _ = xyToRaDecPP(time_data, [x], [y], [1], self.platepar, extinction_correction=False)


            # Precess RA/Dec to epoch of date for alt/az computation
            ra_date, dec_date = equatorialCoordPrecession(J2000_JD.days, jd[0], np.radians(ra[0]), \
                np.radians(dec[0]))
            ra_date, dec_date = np.degrees(ra_date), np.degrees(dec_date)

            # Compute alt, az
            azim, alt = raDec2AltAz(ra_date, dec_date, jd[0], self.platepar.lat, self.platepar.lon)


            status_str += ",  Azim={:6.2f}  Alt={:6.2f} (date),  RA={:6.2f}  Dec={:+6.2f} (J2000)".format(\
                azim, alt, ra[0], dec[0])


        return status_str




    def zoomImage(self, zoom_in):
        """ Change the zoom if the image. """

        zoom_factor = 2.0/3

        # Get the current limits of the plot
        xmin, xmax = plt.gca().get_xlim()
        ymin, ymax = plt.gca().get_ylim()

        x_img = self.current_image.shape[1]
        y_img = self.current_image.shape[0]


        # Compute maximum length of side differences
        max_size = max(abs(xmax - xmin), abs(ymin - ymax))

        # Zoom in
        if zoom_in:

            # Compute the new size
            xsize = zoom_factor*max_size
            ysize = zoom_factor*max_size


        else:

            # Compute the new size
            xsize = (1.0/zoom_factor)*max_size
            ysize = (1.0/zoom_factor)*max_size


        # Compute the new limits
        xmin = self.mouse_x - xsize/2
        xmax = self.mouse_x + xsize/2
        ymin = self.mouse_y + ysize/2
        ymax = self.mouse_y - ysize/2


        # Check if the new bounds are within the limits
        if xmin < 0:
            xmin = 0
        elif xmin > x_img:
            xmin = x_img

        if xmax < 0:
            xmax = 0
        elif xmax > x_img:
            xmax = x_img


        if ymin < 0:
            ymin = 0
        elif ymin > y_img:
            ymin = y_img

        if ymax < 0:
            ymax = 0
        elif ymax > y_img:
            ymax = y_img

        # Set the new limits
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)


    def photometryColoring(self):
        """ Color pixels for photometry. """

        pixel_list = []

        mouse_x = int(self.mouse_x)
        mouse_y = int(self.mouse_y)

        ### Add all pixels within the aperture to the list for photometry ###

        x_list = range(mouse_x - self.photometry_aperture_radius, mouse_x \
            + self.photometry_aperture_radius + 1)
        y_list = range(mouse_y - self.photometry_aperture_radius, mouse_y \
            + self.photometry_aperture_radius + 1)

        for x in x_list:
            for y in y_list:

                # Skip pixels ourside the image
                if (x < 0) or (x > self.current_image.shape[1]):
                    continue

                if (y < 0) or (y > self.current_image.shape[0]):
                    continue

                # Check if the given pixels are within the aperture radius
                if ((x - mouse_x)**2 + (y - mouse_y)**2) <= self.photometry_aperture_radius**2:
                    pixel_list.append((x, y))


        ##########

        return pixel_list


    def computeIntensitySum(self):
        """ Compute the background subtracted sum of intensity of colored pixels. The background is estimated
            as the median of near pixels that are not colored.
        """

        # Find the pick done on the current frame
        pick_found = [pick for pick in self.pick_list if pick.frame == self.current_frame]

        if pick_found:

            pick = pick_found[0]

            # If there are no photometry pixels, set the intensity to 0
            if not pick.photometry_pixels:

                pick.intensity_sum = 1

                return None


            x_arr, y_arr = np.array(pick.photometry_pixels).T

            # Compute the centre of the colored pixels
            x_centre = np.mean(x_arr)
            y_centre = np.mean(y_arr)

            # Take a window twice the size of the colored pixels
            x_color_size = np.max(x_arr) - np.min(x_arr)
            y_color_size = np.max(y_arr) - np.min(y_arr)

            x_min = int(x_centre - x_color_size)
            x_max = int(x_centre + x_color_size)
            y_min = int(y_centre + y_color_size)
            y_max = int(y_centre - y_color_size)

            # If only the FR file is given, make sure the background is only estimated within the FR window
            if (self.fr is not None) and (self.ff is None):
                x_min = max(x_min, self.fr_xmin)
                x_max = min(x_max, self.fr_xmax)
                y_min = min(y_min, self.fr_ymin)
                y_max = max(y_max, self.fr_ymax)

            # Limit the size to be within the bounds
            if x_min < 0: x_min = 0
            if x_max > self.current_image.shape[1]: x_max = self.current_image.shape[1]
            if y_min > self.current_image.shape[0]: y_min = self.current_image.shape[0]
            if y_max < 0: y_max = 0


            # Take only the colored part
            mask_img = np.ones_like(self.current_image)
            mask_img[y_arr, x_arr] = 0
            masked_img = np.ma.masked_array(self.current_image, mask_img)
            crop_img = masked_img[y_max:y_min, x_min:x_max]

            # Perform gamma correction on the colored part
            crop_img = Image.gammaCorrection(crop_img, self.config.gamma)


            # Mask out the colored in pixels
            mask_img_bg = np.zeros_like(self.current_image)
            mask_img_bg[y_arr, x_arr] = 1

            # Take the image where the colored part is masked out and crop the surroundings
            masked_img_bg = np.ma.masked_array(self.current_image, mask_img_bg)
            crop_bg = masked_img_bg[y_max:y_min, x_min:x_max]

            # Perform gamma correction on the background
            crop_bg = Image.gammaCorrection(crop_bg, self.config.gamma)


            # Compute the median background
            background_lvl = np.ma.median(crop_bg)

            # Compute the background subtracted intensity sum
            pick.intensity_sum = np.ma.sum(crop_img - background_lvl)

            # Make sure the intensity sum is never 0
            if pick.intensity_sum <= 0:
                pick.intensity_sum = 1


    def recomputeAllIntensitySums(self):
        """ Recompute intensity sums for all frames. """


        # Store the current frame
        current_frame_bak = self.current_frame

        # Disable showing the maxpixel
        self.show_maxpixel = False

        # Go through all picks and recompute intensities
        for pick in self.pick_list:

            # Set the current frame
            self.setFrame(pick.frame, only_number_update=True)

            # Load the frame without showing it
            _, self.current_image = self.loadImage()

            # Compute the intensity sum
            self.computeIntensitySum()

        # Set the current frame back
        self.setFrame(current_frame_bak, only_number_update=True)

        self.updateImage()



    def drawCursorCircle(self):
        """ Adds a circle around the mouse cursor. """

        # Delete the old aperture circle
        if self.circle_aperture is not None:
            try:
                self.circle_aperture.remove()
            except:
                pass
            self.circle_aperture = None

        if self.circle_aperture_outer is not None:
            try:
                self.circle_aperture_outer.remove()
            except:
                pass
            self.circle_aperture_outer = None


        # If photometry coloring is on, show a purple annulus
        if self.photometry_coloring_mode:

            # Plot a circle of the given radius around the cursor
            self.circle_aperture = mpatches.Circle((self.mouse_x, self.mouse_y),
                self.photometry_aperture_radius, edgecolor='red', fc='red', alpha=0.5)

            plt.gca().add_patch(self.circle_aperture)

        else:

            # Plot a circle of the given radius around the cursor
            self.circle_aperture = mpatches.Circle((self.mouse_x, self.mouse_y),
                self.aperture_radius, edgecolor='yellow', fc='none')

            # Plot a circle of the given radius around the cursor, which is used to determine the region where the
            # background will be taken from
            self.circle_aperture_outer = mpatches.Circle((self.mouse_x, self.mouse_y),
                self.aperture_radius*2, edgecolor='yellow', fc='none', linestyle='dotted')

            plt.gca().add_patch(self.circle_aperture)
            plt.gca().add_patch(self.circle_aperture_outer)


        plt.gcf().canvas.draw()



    def drawPicks(self, update_plot=True):
        """ Plot the picks. """

        # Plot all picks
        for pick in self.pick_list:

            plt.scatter(pick.x_centroid, pick.y_centroid, marker='+', c='r', s=40, lw=1)


        # Find the pick done on the current frame
        pick_found = [pick for pick in self.pick_list if pick.frame == self.current_frame]

        # Plot the pick done on this image a bit larger
        if pick_found:

            pick = pick_found[0]

            # Draw the centroid on the image
            self.centroid_handle = plt.scatter(pick.x_centroid, pick.y_centroid, marker='+', c='r', s=150,
                lw=2)


            # Update canvas
            if update_plot:
                plt.gcf().canvas.draw()


    def drawPhotometryColoring(self, update_plot=True):
        """ Color pixels which will be used for photometry. """

        # Remove old photometry coloring
        if self.photometry_coloring_handle is not None:
            try:
                self.photometry_coloring_handle.remove()
            except:
                pass

        # Find the pick done on the current frame
        pick_found = [pick for pick in self.pick_list if pick.frame == self.current_frame]

        if pick_found:

            pick = pick_found[0]

            if pick.photometry_pixels is not None:

                # Create a coloring mask
                x_mask, y_mask = np.array(pick.photometry_pixels).T

                mask_img = np.zeros_like(self.current_image)
                mask_img[y_mask, x_mask] = 1

                # Create a RGB overlay
                mask_overlay = np.zeros((self.current_image.shape[0], self.current_image.shape[1], 4))
                mask_overlay[..., 1] = 1 # Green channel
                mask_overlay[..., 3] = 0.3*mask_img # Alpha channel

                self.photometry_coloring_handle = plt.imshow(mask_overlay, interpolation='nearest')

                # Update canvas
                if update_plot:
                    plt.gcf().canvas.draw()



    def addCentroid(self, frame, x_centroid, y_centroid):
        """ Add the centroid to the list of centroids. """

        print('Added centroid at ({:.2f}, {:.2f}) on frame {:.1f}'.format(x_centroid, y_centroid, frame))

        # Check if there are previous picks on this frame
        prev_pick = [i for i, pick in enumerate(self.pick_list) if pick.frame == frame]

        # Update centroids of previous pick if it exists
        if prev_pick:

            i = prev_pick[0]

            self.pick_list[i].x_centroid = x_centroid
            self.pick_list[i].y_centroid = y_centroid

        # Create a new pick
        else:

            pick = Pick()

            pick.frame = frame
            pick.x_centroid = x_centroid
            pick.y_centroid = y_centroid

            self.pick_list.append(pick)


    def removeCentroid(self, frame):
        """ Remove the centroid from the list of centroids. """

        # Check if there are previous picks on this frame
        prev_pick = [i for i, pick in enumerate(self.pick_list) if pick.frame == frame]

        if prev_pick:

            i = prev_pick[0]

            pick = self.pick_list[i]

            print('Removed centroid at ({:.2f}, {:.2f}) on frame {:.1f}'.format(pick.x_centroid, \
                pick.y_centroid, frame))

            # Remove the centroid
            self.pick_list.pop(i)


    def changePhotometry(self, frame, photometry_pixels, add_photometry):
        """ Add/remove photometry pixels of the pick. """

        # Check if there are previous picks on this frame
        prev_pick = [i for i, pick in enumerate(self.pick_list) if pick.frame == frame]

        # Update centroids of previous pick if it exists
        if prev_pick:

            i = prev_pick[0]

            # If there's no previous photometry, add an empty list
            if self.pick_list[i].photometry_pixels is None:
                self.pick_list[i].photometry_pixels = []

            if add_photometry:

                # Add the photometry pixels to the pick
                self.pick_list[i].photometry_pixels = list(set(self.pick_list[i].photometry_pixels \
                    + photometry_pixels))

            else:

                # Remove the photometry pixels to the pick
                self.pick_list[i].photometry_pixels = [px for px in self.pick_list[i].photometry_pixels if px not in photometry_pixels]


        # Add a new pick
        elif add_photometry:

            pick = Pick()

            pick.frame = frame
            pick.photometry_pixels = photometry_pixels

            self.pick_list.append(pick)



    def centroid(self):
        """ Find the centroid of the object clicked on the image. """

        ### Extract part of image around the mouse cursor ###
        ######################################################################################################

        # Outer circle radius
        outer_radius = self.aperture_radius*2

        x_min = int(round(self.mouse_x - outer_radius))
        if x_min < 0: x_min = 0

        x_max = int(round(self.mouse_x + outer_radius))
        if x_max > self.current_image.shape[1] - 1:
            x_max > self.current_image.shape[1] - 1

        y_min = int(round(self.mouse_y - outer_radius))
        if y_min < 0: y_min = 0

        y_max = int(round(self.mouse_y + outer_radius))
        if y_max > self.current_image.shape[0] - 1:
            y_max > self.current_image.shape[0] - 1


        # Crop the segment containing the centroid
        img_crop = self.current_image[y_min:y_max, x_min:x_max]

        # Apply camera gamma correction
        img_crop = Image.gammaCorrection(img_crop, self.config.gamma)

        ######################################################################################################


        ### Estimate the background ###
        ######################################################################################################
        bg_acc = 0
        bg_counter = 0
        for i in range(img_crop.shape[0]):
            for j in range(img_crop.shape[1]):

                # Calculate distance of pixel from centre of the cropped image
                i_rel = i - img_crop.shape[0]/2
                j_rel = j - img_crop.shape[1]/2
                pix_dist = math.sqrt(i_rel**2 + j_rel**2)

                # Take only those pixels between the inner and the outer circle
                if (pix_dist <= outer_radius) and (pix_dist > self.aperture_radius):
                    bg_acc += img_crop[i, j]
                    bg_counter += 1

        # Calculate mean background intensity
        bg_intensity = bg_acc/bg_counter

        ######################################################################################################


        ### Calculate the centroid ###
        ######################################################################################################
        x_acc = 0
        y_acc = 0
        intens_acc = 0

        for i in range(img_crop.shape[0]):
            for j in range(img_crop.shape[1]):

                # Calculate distance of pixel from centre of the cropped image
                i_rel = i - img_crop.shape[0]/2
                j_rel = j - img_crop.shape[1]/2
                pix_dist = math.sqrt(i_rel**2 + j_rel**2)

                # Take only those pixels between the inner and the outer circle
                if pix_dist <= self.aperture_radius:
                    x_acc += j*(img_crop[i, j] - bg_intensity)
                    y_acc += i*(img_crop[i, j] - bg_intensity)
                    intens_acc += img_crop[i, j] - bg_intensity


        x_centroid = x_acc/intens_acc + x_min
        y_centroid = y_acc/intens_acc + y_min

        ######################################################################################################

        return x_centroid, y_centroid, intens_acc


    def updateGamma(self, gamma_adj_factor):
        """ Change the image gamma by a given factor. """

        self.img_gamma *= gamma_adj_factor

        # Make sure gamma is in the proper range
        if self.img_gamma < 0.1: self.img_gamma = 0.1
        if self.img_gamma > 10: self.img_gamma = 10

        self.updateImage()



    def prevFrame(self, only_number_update=False):
        """ Cycle to the previous frame.

        Keyword arguments:
            only_number_update: [bool] Just cycle the frame number if True. False by default. This is used
                when skipping multiple frames.
        """

        if not only_number_update:

            # Compute the intensity sum done on the previous frame
            self.computeIntensitySum()


        # Decrement the frame numebr
        if self.img_handle is not None:
            self.img_handle.prevFrame()

            if not self.single_image_mode:
                self.current_frame = self.img_handle.current_frame

            # In the single image mode, continously cycle through frames
            else:
                self.current_frame -= 1

                if self.current_frame < 0:
                    self.current_frame = 0

        else:
            self.current_frame = (self.current_frame - self.frame_step)%self.nframes


        if not only_number_update:
            self.updateImage()

            self.printStatus()



    def nextFrame(self, only_number_update=False):
        """ Cycle to the next frame.
        Keyword arguments:
        only_number_update: [bool] Just cycle the frame number if True. False by default. This is used
                when skipping multiple frames.
        """

        if not only_number_update:

            # Compute the intensity sum done on the previous frame
            self.computeIntensitySum()


        # Increment the frame
        if self.img_handle is not None:
            self.img_handle.nextFrame()

            if not self.single_image_mode:
                self.current_frame = self.img_handle.current_frame

            # In the single image mode, continously cycle through frames
            else:
                self.current_frame += 1


        else:
            self.current_frame = (self.current_frame + self.frame_step)%self.nframes


        if not only_number_update:
            self.updateImage()
            self.printStatus()


    def setFrame(self, fr_num, only_number_update=False):
        """ Set the current frame number.

        Arguments:
            fr_num: [float] Frame number to set.
        Keyword arguments:
            only_number_update: [bool] Just cycle the frame number if True. False by default. This is used
                when skipping multiple frames.
        """

        if not only_number_update:

            # Compute the intensity sum done on the previous frame
            self.computeIntensitySum()


        # Increment the frame
        if self.img_handle is not None:
            self.img_handle.setFrame(fr_num)

            self.current_frame = self.img_handle.current_frame

        else:
            self.current_frame = fr_num%self.nframes


        if not only_number_update:
            self.updateImage()
            self.printStatus()


    def getRollingShutterCorrectedFrameNo(self, pick):
        """ Given a pick object, return rolling shutter corrected (or not, depending on the config) frame
            number.
        """

        # Correct the rolling shutter effect
        if self.config.deinterlace_order == -1:

            # Get image height
            if self.img_handle is None:
                img_h = self.config.height

            else:
                img_h = self.img_handle.ff.maxpixel.shape[0]

            # Compute the corrected frame time
            frame_no = RollingShutterCorrection.correctRollingShutterTemporal(pick.frame, \
                pick.y_centroid, img_h)

        # If global shutter, do no correction
        else:
            frame_no = pick.frame


        return frame_no


    def saveState(self):
        """ Save the current state of the program to a file, so it can be reloaded. """

        state_date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S.%f")[:-3]
        state_file = 'manualReduction_{:s}.state'.format(state_date_str)

        # Save the state to a pickle file
        savePickle(self, self.dir_path, state_file)

        # Write the latest pickle fine
        savePickle(self, self.dir_path, 'manualReduction_latest.state')

        print('Saved state to file:', state_file)



    def saveJSON(self):
        """ Save the picks in a JSON file. """

        # Compute the intensity sum done on the previous frame
        self.computeIntensitySum()


        json_dict = {}


        # If the platepar was loaded, save the station info
        station_dict = {}
        if self.platepar is not None:

            station_dict['station_id'] = self.platepar.station_code
            station_dict['lat'] = self.platepar.lat
            station_dict['lon'] = self.platepar.lon
            station_dict['elev'] = self.platepar.elev

            station_name = self.platepar.station_code

        else:

            station_dict['station_id'] = self.config.stationID
            station_dict['lat'] = self.config.latitude
            station_dict['lon'] = self.config.longitude
            station_dict['elev'] = self.config.elevation

            station_name = self.station_name

        # Add station data to JSON file
        json_dict['station'] = station_dict



        # Save reference time (Julian date)
        if self.img_handle is not None:

            # Get time from image handle
            jdt_ref = datetime2JD(self.img_handle.beginning_datetime)

        # FR file time
        else:

            # Get the JD of the beginning of the FR file
            jdt_ref = datetime2JD(filenameToDatetime(os.path.basename(self.fr_file)))

        # Set the reference JD
        json_dict['jdt_ref'] = jdt_ref


        # Set the frames per second
        json_dict['fps'] = self.fps


        ### Save picks to JSON file ###

        # Set measurement type to RA/Dec (meastype = 1)
        json_dict['meastype'] = 1


        centroids = []
        for pick in self.pick_list:

            # Make sure to centroid is picked and is not just the photometry
            if pick.x_centroid is None:
                continue


            # Compute RA/Dec of the pick if the platepar is available
            if self.platepar is not None:

                time_data = [self.getCurrentFrameTime(frame_no=pick.frame)]

                _, ra_data, dec_data, mag_data = xyToRaDecPP(time_data, [pick.x_centroid], \
                    [pick.y_centroid], [pick.intensity_sum], self.platepar)

                ra = ra_data[0]
                dec = dec_data[0]
                mag = mag_data[0]

            else:
                ra = dec = mag = None


            # Get the rolling shutter corrected (or not, depending on the config) frame number
            frame_no = self.getRollingShutterCorrectedFrameNo(pick)

            # Compute the time relative to the reference JD
            t_rel = frame_no/self.fps


            centroids.append([t_rel, pick.x_centroid, pick.y_centroid, ra, dec, pick.intensity_sum, mag])


        # Sort centroids by relative time
        centroids = sorted(centroids, key=lambda x: x[0])


        json_dict['centroids_labels'] = ['Time (s)', 'X (px)', 'Y (px)', 'RA (deg)', 'Dec (deg)', \
            'Summed intensity', 'Magnitude']
        json_dict['centroids'] = centroids

        ### ###

        # Create a name for the JSON file
        json_file_name = jd2Date(jdt_ref, dt_obj=True).strftime('%Y%m%d_%H%M%S.%f') + '_' \
            + station_name + '_picks.json'

        json_file_path = os.path.join(self.dir_path, json_file_name)

        with open(json_file_path, 'w') as f:
            json.dump(json_dict, f, indent=4, sort_keys=True)


        print('JSON with picks saved to:', json_file_path)



    def saveFTPdetectinfo(self):
        """ Saves the picks to a FTPdetectinfo file in the same folder where the first given file is. """

        # Compute the intensity sum done on the previous frame
        self.computeIntensitySum()


        # Generate a name for the FF file which will be written to FTPdetectinfo
        if self.img_handle is not None:

            dir_path = self.img_handle.dir_path

            # If the FF file is loaded, just copy its name
            if self.img_handle.input_type == 'ff':
                ff_name_ftp = self.img_handle.current_ff_file

            else:

                # Construct a fake FF file name
                ff_name_ftp = "FF_{:s}_".format(self.station_name) \
                 + self.img_handle.beginning_datetime.strftime("%Y%m%d_%H%M%S_") \
                    + "{:03d}".format(int(round(self.img_handle.beginning_datetime.microsecond/1000))) \
                    + "_0000000.fits"

        else:
            # Extract the time from the FR file otherwise
            dir_path, ff_name_ftp = os.path.split(self.fr_file)


        # Create the list of picks for saving
        centroids = []
        for pick in self.pick_list:

            # Make sure to centroid is picked and is not just the photometry
            if pick.x_centroid is None:
                continue

            # Get the rolling shutter corrected (or not, depending on the config) frame number
            frame_no = self.getRollingShutterCorrectedFrameNo(pick)

            centroids.append([frame_no, pick.x_centroid, pick.y_centroid, pick.intensity_sum])


        # If there are no centroids, don't save anything
        if len(centroids) == 0:
            messagebox.showinfo('FTPdetectinfo saving error', 'No centroids to save!')
            return 1

        # Sort by frame number
        centroids = sorted(centroids, key=lambda x: x[0])

        # Construct the meteor
        meteor_list = [[ff_name_ftp, 1, 0, 0, centroids]]


        # Remove the file extension of the image file
        ff_name_ftp = ff_name_ftp.replace('.bin', '').replace('.fits', '')

        # Create a name for the FTPdetectinfo
        ftpdetectinfo_name = "FTPdetectinfo_" + "_".join(ff_name_ftp.split('_')[1:]) + '_manual.txt'

        # Read the station code for the file name
        station_id = ff_name_ftp.split('_')[1]


        # Write the FTPdetect info
        writeFTPdetectinfo(meteor_list, dir_path, ftpdetectinfo_name, '', station_id, self.fps)

        print('FTPdetecinfo written to:', os.path.join(dir_path, ftpdetectinfo_name))


        # If the platepar is given, apply it to the reductions
        if self.platepar is not None:

            applyAstrometryFTPdetectinfo(self.dir_path, ftpdetectinfo_name, '', \
                UT_corr=self.platepar.UT_corr, platepar=self.platepar)

            print('Platepar applied to manual picks!')




    def saveCurrentFrame(self):
        """ Saves the current frame to disk. """

        # Generate a name for the FF file which will be written to FTPdetectinfo
        if self.img_handle is not None:

            dir_path = self.img_handle.dir_path

            # If the FF file is loaded, just copy its name
            if self.img_handle.input_type == 'ff':
                ff_name_ftp = self.img_handle.current_ff_file

            else:

                # Construct a fake FF file name
                ff_name_ftp = "FF_{:s}_".format(self.station_name) \
                    + self.img_handle.beginning_datetime.strftime("%Y%m%d_%H%M%S_") \
                    + "{:03d}".format(int(round(self.img_handle.beginning_datetime.microsecond/1000))) \
                    + "_0000000.fits"

        else:
            # Extract the time from the FR file otherwise
            dir_path, ff_name_ftp = os.path.split(self.fr_file)


        # Remove the file extension of the image file
        ff_name_ftp = ff_name_ftp.replace('.bin', '').replace('.fits', '')

        # Construct the file name
        frame_file_name = ff_name_ftp + "_frame_{:03d}".format(self.current_frame) + '.png'
        frame_file_path = os.path.join(dir_path, frame_file_name)

        # Save the frame to disk
        Image.saveImage(frame_file_path, self.current_image)

        print('Frame {:.1f} saved to: {:s}'.format(self.current_frame, frame_file_path))





if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="""Tool for manually picking positions of meteors on video frames and performing manual photometry.
        NOTE: The centroiding and photometry will always be done on the image with the subtracted average, except when only the FR file is given.
        """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('file1', metavar='FILE1', type=str, nargs=1, \
                    help='Path to one of the following: an FF file, an FR file (if an FF file is not available), a directory with PNG files, or to a saved .state file.')

    arg_parser.add_argument('input2', metavar='INPUT2', type=str, nargs='*', \
                    help='If an FF file was given, an FR file can be given in addition. If PNGs are used, this second argument must be the UTC time of frame 0 in the following format: YYYYMMDD_HHMMSS.uuu')

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
        help="Path to a config file which will be used instead of the default one.")

    arg_parser.add_argument('-b', '--begframe', metavar='FIRST_FRAME', type=int, \
        help="First frame to show.")

    arg_parser.add_argument('-t', '--timebeg', nargs=1, metavar='TIME', type=str, \
        help="The beginning time of the video file in the YYYYMMDD_hhmmss.uuuuuu format.")

    arg_parser.add_argument('-f', '--fps', metavar='FPS', type=float, \
        help="Frames per second of the video. If not given, it will be read from a) the FF file if available, b) from the config file.")

    arg_parser.add_argument('-d', '--deinterlace', nargs='?', type=int, default=-1, \
        help="Perform manual reduction on deinterlaced frames, even first by default. If odd first is desired, -d 1 should be used.")

    arg_parser.add_argument('-n', '--name', nargs=1, metavar='STATION_NAME', type=str, \
        help="Station name or code. If not given, it will just be 'manual', or it will be read from the FF file if used.")

    arg_parser.add_argument('-g', '--gamma', metavar='CAMERA_GAMMA', type=float, \
        help="Camera gamma value. Science grade cameras have 1.0, consumer grade cameras have 0.45. Adjusting this is essential for good photometry, and doing star photometry through SkyFit can reveal the real camera gamma.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################


    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, cml_args.file1)


    ff_name = None
    fr_name = None

    # Read the deinterlace
    #   -1 - no deinterlace
    #    0 - odd first
    #    1 - even first
    deinterlace_mode = cml_args.deinterlace
    if cml_args.deinterlace is None:
        deinterlace_mode = 0



    # If camera gamma was given, change the value in config
    if cml_args.gamma is not None:
        config.gamma = cml_args.gamma


    # Extract station name
    station_name = None
    if cml_args.name is not None:
        station_name = cml_args.name[0]


    # Extract inputs
    file1 = os.path.abspath(cml_args.file1[0])

    if cml_args.input2:
        input2 = cml_args.input2[0]
    else:
        input2 = None


    # If the second argument is None, try reading it as time
    if cml_args.timebeg is not None:
        input2 = cml_args.timebeg[0]


    ### Detect the input type ###

    # If only an FR file was given
    head1, tail1 = os.path.split(file1)

    # If the state file was given, load the state
    if tail1.endswith('.state'):

        # Load the manual redicution tool object from a state file
        manual_tool = loadPickle(head1, tail1)
        manual_tool.dir_path = head1
        manual_tool.updateImage()
        manual_tool.registerEventHandling()


    else:

        # If the second argument is an FR file, omit the beginning time
        if fr_name is not None:
            img_handle = detectInputType(file1, config, skip_ff_dir=True, fps=cml_args.fps)

        # Otherwise, do automatic detection of file type and feed it the beginning time
        else:

            beginning_time = None

            if input2 is not None:

                # Parse the time
                beginning_time = datetime.datetime.strptime(input2, "%Y%m%d_%H%M%S.%f")
                beginning_time = beginning_time.replace(tzinfo=pytz.UTC)


            img_handle = detectInputType(file1, config, beginning_time=beginning_time, skip_ff_dir=True,
                fps=cml_args.fps)


            # FR files can only be combined with FF files
            if img_handle is not None:

                print('Input type:', img_handle.input_type)

                if img_handle.input_type != 'ff':
                    fr_name = None

            else:
                sys.exit()


        # Init the tool
        manual_tool = ManualReductionTool(config, img_handle, fr_name, first_frame=cml_args.begframe, \
                fps=cml_args.fps, deinterlace_mode=deinterlace_mode, station_name=station_name)



    plt.tight_layout()
    plt.show()