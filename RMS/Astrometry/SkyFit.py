""" GUI tool for making initial plate estimations and manually fitting astrometric plates. """

# The MIT License

# Copyright (c) 2017 Denis Vida

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from __future__ import print_function, division, absolute_import

import os
import sys
import math
import copy
import time
import datetime
import argparse
import traceback
    
# tkinter import that works on both Python 2 and 3
try:
    import tkinter
    from tkinter import messagebox
except:
    import Tkinter as tkinter
    import tkMessageBox as messagebox

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
import PIL.Image, PIL.ImageDraw
import scipy.optimize
import scipy.ndimage

from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP, raDecToXYPP, \
    rotationWrtHorizon, rotationWrtHorizonToPosAngle, computeFOVSize, photomLine, photometryFit, \
    rotationWrtStandard, rotationWrtStandardToPosAngle, correctVignetting, extinctionCorrectionTrueToApparent
from RMS.Astrometry.AstrometryNetNova import novaAstrometryNetSolve
from RMS.Astrometry.Conversions import J2000_JD, date2JD, JD2HourAngle, raDec2AltAz, altAz2RADec
import RMS.ConfigReader as cr
import RMS.Formats.CALSTARS as CALSTARS
from RMS.Formats.Platepar import Platepar, getCatalogStarsImagePositions
from RMS.Formats.FrameInterface import detectInputType
from RMS.Formats import StarCatalog
from RMS.Pickling import loadPickle, savePickle
from RMS.Routines import Image
from RMS.Math import angularSeparation
from RMS.Misc import decimalDegreesToSexHours, openFileDialog

# Import Cython functions
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
from RMS.Astrometry.CyFunctions import subsetCatalog, equatorialCoordPrecession, eqRefractionTrueToApparent, \
    eqRefractionApparentToTrue, trueRaDec2ApparentAltAz, apparentAltAz2TrueRADec



# TkAgg has issues when opening an external file prompt, so other backends are forced if available
if matplotlib.get_backend() == 'TkAgg':

    backends = ['Qt5Agg', 'Qt4Agg']

    for bk in backends:
        
        # Try setting backend
        try:
            plt.switch_backend(bk)

        except:
            pass


print('Using backend: ', matplotlib.get_backend())



class FOVinputDialog(object):
    """ Dialog for inputting FOV centre in Alt/Az. """
    def __init__(self, parent):

        self.parent = parent

        # Set initial angle values
        self.azim = self.alt = self.rot = 0

        self.top = tkinter.Toplevel(parent)

        # Bind the Enter key to run the verify function
        self.top.bind('<Return>', self.verify)

        tkinter.Label(self.top, text="FOV centre (degrees) \nAzim +E of due N\nRotation from vertical").grid(row=0, columnspan=2)

        azim_label = tkinter.Label(self.top, text='Azim = ')
        azim_label.grid(row=1, column=0)
        self.azimuth = tkinter.Entry(self.top)
        self.azimuth.grid(row=1, column=1)
        self.azimuth.focus_set()

        elev_label = tkinter.Label(self.top, text='Alt  =')
        elev_label.grid(row=2, column=0)
        self.altitude = tkinter.Entry(self.top)
        self.altitude.grid(row=2, column=1)

        rot_label = tkinter.Label(self.top, text='Rotation  =')
        rot_label.grid(row=3, column=0)
        self.rotation = tkinter.Entry(self.top)
        self.rotation.grid(row=3, column=1)
        self.rotation.insert(0, '0')

        b = tkinter.Button(self.top, text="OK", command=self.verify)
        b.grid(row=4, columnspan=2)


    def verify(self, event=None):
        """ Check that the azimuth and altitude are withing the bounds. """

        try:
            # Read values
            self.azim = float(self.azimuth.get())%360
            self.alt  = float(self.altitude.get())
            self.rot = float(self.rotation.get())%360

            # Check that the values are within the bounds
            if (self.alt < 0) or (self.alt > 90):
                messagebox.showerror(title='Range error', message='The altitude is not within the limits!')
            else:
                self.top.destroy()

        except:
            messagebox.showerror(title='Range error', message='Please enter floating point numbers, not text!')


    def getAltAz(self):
        """ Returns inputed FOV centre. """

        return self.azim, self.alt, self.rot




class PlateTool(object):
    def __init__(self, dir_path, config, beginning_time=None, fps=None, gamma=None):
        """ SkyFit interactive window.

        Arguments:
            dir_path: [str] Absolute path to the directory containing image files.
            config: [Config struct]

        Keyword arguments:
            beginning_time: [datetime] Datetime of the video beginning. Optional, only can be given for
                video input formats.
            fps: [float] Frames per second, used only when images in a folder are used.
            gamma: [float] Camera gamma. None by default, then it will be used from the platepar file or
                config.
        """

        self.config = config
        self.dir_path = dir_path


        # If camera gamma was given, change the value in config
        if gamma is not None:
            config.gamma = gamma


        # Flag which regulates wheter the maxpixel or the avepixel image is shown (avepixel by default)
        self.img_type_flag = 'avepixel'

        # Star picking mode
        self.star_pick_mode = False
        self.star_selection_centroid = True
        self.circle_aperature = None
        self.circle_aperature_outer = None
        self.star_aperature_radius = 5
        self.x_centroid = self.y_centroid = None
        self.closest_cat_star_indx = None
        self.photom_deviatons_scat = []

        self.catalog_stars_visible = True
        self.draw_calstars = True

        self.draw_distortion = False

        self.show_key_help = 1

        # List of paired image and catalog stars
        self.paired_stars = []
        self.residuals = None

        # Positions of the mouse cursor
        self.mouse_x = 0
        self.mouse_y = 0

        # Position of mouse cursor when it was last pressed
        self.mouse_x_press = 0
        self.mouse_y_press = 0

        # Kwy increment
        self.key_increment = 1.0

        # Image gamma and levels
        self.bit_depth = self.config.bit_depth
        self.img_gamma = 1.0
        self.img_level_min = self.img_level_min_auto = 0
        self.img_level_max = self.img_level_max_auto = 2**self.bit_depth - 1

        # Invert image colors
        self.invert_levels = False

        self.img_data_raw = None
        self.img_data_processed = None

        self.adjust_levels_mode = False
        self.auto_levels = False

        # Platepar format (json or txt)
        self.platepar_fmt = None

        # Flat field
        self.flat_struct = None

        # Dark frame
        self.dark = None

        # Image coordinates of catalog stars
        self.catalog_x = self.catalog_y = None
        self.catalog_x_filtered = self.catalog_y_filtered = None
        self.mag_band_string = ''

        # Flag indicating that the first platepar fit has to be done
        self.first_platepar_fit = True



        # Toggle zoom window
        self.show_zoom_window = False

        # Position of the zoom window (NE, NW, SE, SW)
        self.zoom_window_pos = ''

        ######################################################################################################


        
        # Detect input file type and load appropriate input plugin
        self.img_handle = detectInputType(self.dir_path, self.config, beginning_time=beginning_time, fps=fps)


        # Extract the directory path if a file was given
        if os.path.isfile(self.dir_path):
            
            self.dir_path, _ = os.path.split(self.dir_path)



        # Load catalog stars
        self.catalog_stars = self.loadCatalogStars(self.config.catalog_mag_limit)
        self.cat_lim_mag = self.config.catalog_mag_limit

        # Check if the catalog exists
        if not self.catalog_stars.any():
            messagebox.showerror(title='Star catalog error', message='Star catalog from path ' \
                + os.path.join(self.config.star_catalog_path, self.config.star_catalog_file) \
                + 'could not be loaded!')
            sys.exit()
        else:
            print('Star catalog loaded!')


        # Find the CALSTARS file in the given folder
        calstars_file = None
        for cal_file in os.listdir(self.dir_path):
            if ('CALSTARS' in cal_file) and ('.txt' in cal_file):
                calstars_file = cal_file
                break

        if calstars_file is None:

            # Check if the calstars file is required
            if self.img_handle.require_calstars:
                
                messagebox.showinfo(title='CALSTARS error', \
                    message='CALSTARS file could not be found in the given directory!')

            self.calstars = {}

        else:

            # Load the calstars file
            calstars_list = CALSTARS.readCALSTARS(self.dir_path, calstars_file)

            # Convert the list to a dictionary
            self.calstars = {ff_file: star_data for ff_file, star_data in calstars_list}

            print('CALSTARS file: ' + calstars_file + ' loaded!')



        # Load the platepar file
        self.platepar_file, self.platepar = self.loadPlatepar()

        if self.platepar_file:

            print('Platepar loaded:', self.platepar_file)

            # Print the field of view size
            print("FOV: {:.2f} x {:.2f} deg".format(*computeFOVSize(self.platepar)))

        
        # If the platepar file was not loaded, set initial values from config
        else:
            self.makeNewPlatepar(update_image=False)

            # Create the name of the platepar file
            self.platepar_file = os.path.join(self.dir_path, self.config.platepar_name)


        # Set the given gamma value to platepar
        if gamma is not None:
            self.platepar.gamma = gamma



        # Load distorion type index
        self.dist_type_count = len(self.platepar.distortion_type_list)
        self.dist_type_index = self.platepar.distortion_type_list.index(self.platepar.distortion_type)


        ### INIT IMAGE ###

        self.fig, self.ax = plt.subplots(facecolor='black')

        # Init the first image
        self.updateImage(first_update=True)


        # Register keys with matplotlib
        self.registerEventHandling()



    def registerEventHandling(self):
        """ Register mouse button and key pressess with matplotlib. """


        self.fig.canvas.set_window_title('SkyFit')

        # Set the bacground color to black
        #matplotlib.rcParams['axes.facecolor'] = 'k'

        # Disable standard matplotlib keyboard shortcuts
        plt.rcParams['keymap.save'] = ''
        plt.rcParams['keymap.fullscreen'] = ''
        plt.rcParams['keymap.all_axes'] = ''
        plt.rcParams['keymap.quit'] = ''
        plt.rcParams['keymap.pan'] = ''
        plt.rcParams['keymap.forward'] = ''
        plt.rcParams['keymap.back'] = ''
        plt.rcParams['keymap.yscale'] = ''
        plt.rcParams['keymap.xscale'] = ''
        plt.rcParams['keymap.grid'] = ''
        plt.rcParams['keymap.grid_minor'] = ''


        
        self.ax.figure.canvas.mpl_connect('button_press_event', self.onMousePress)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.onMouseRelease)
        
        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.onMouseMotion)

        # Register which mouse/keyboard events will evoke which function
        self.ax.figure.canvas.mpl_connect('scroll_event', self.onScroll)
        self.scroll_counter = 0

        self.ax.figure.canvas.mpl_connect('key_press_event', self.onKeyPress)

        # Set the status updater
        self.ax.format_coord = self.mouseOverStatus



    def saveState(self):
        """ Save the current state of the program to a file, so it can be reloaded. """

        # state_date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S.%f")[:-3]
        # state_file = 'skyFit_{:s}.state'.format(state_date_str)

        # # Save the state to a pickle file
        # savePickle(self, self.dir_path, state_file)

        # Write the latest pickle fine
        savePickle(self, self.dir_path, 'skyFit_latest.state')

        print("Saved state to file")



    def onMousePress(self, event):
        """ Called when the mouse click is pressed. """

        # Record the mouse cursor positions
        self.mouse_x_press = event.xdata
        self.mouse_y_press = event.ydata


    def onMouseRelease(self, event):
        """ Called when the mouse click is released. """

        # Do nothing if some button on the toolbar is active (e.g. zoom, move)
        if plt.get_current_fig_manager().toolbar.mode:
            return None

        # Call the same function for mouse movements to update the variables in the background
        self.onMouseMotion(event)


        # If the histogram is on, adjust the levels
        if self.adjust_levels_mode:

            # Left mouse button sets the minimum level
            if event.button == 1:

                # Compute the new image level from clicked position
                img_level_min_new = event.xdata/self.img_data_raw.shape[1]*(2**self.bit_depth)

                # Make sure the minimum level is smaller than the maximum level
                if (img_level_min_new < self.img_level_max) and (img_level_min_new > 0):
                    self.img_level_min = img_level_min_new


            # Right mouse button sets the maximum level
            if event.button == 3:

                # Compute the new image level from clicked position
                img_level_max_new = event.xdata/self.img_data_raw.shape[1]*(2**self.bit_depth)

                # Make sure the minimum level is smaller than the maximum level
                if (img_level_max_new > self.img_level_min) and (img_level_max_new < (2**self.bit_depth - 1)):
                    self.img_level_max = img_level_max_new

            self.updateImage()


        # If the star picking mode is on
        elif self.star_pick_mode:

            # Left mouse button, select stars
            if event.button == 1:

                # If the centroid of the star has to be picked
                if self.star_selection_centroid:

                    # If CTRL is pressed, place the pick manually - NOTE: the intensity might be off then!!!
                    # 'control' is for Windows, 'ctrl+control' is for Linux
                    if (event.key == 'control') or (event.key == 'ctrl+control'):

                        self.x_centroid = self.mouse_x_press
                        self.y_centroid = self.mouse_y_press

                        # Compute the star intensity
                        _, _, self.star_intensity = self.centroidStar(prev_x_cent=self.x_centroid, \
                                    prev_y_cent=self.y_centroid)


                    # Centroid the star around the pick
                    else:

                        # Perform centroiding with 2 iterations
                        x_cent_tmp, y_cent_tmp, _ = self.centroidStar()

                        # Check that the centroiding was successful
                        if x_cent_tmp is not None:

                            # Centroid the star around the pressed coordinates
                            self.x_centroid, self.y_centroid, \
                                self.star_intensity = self.centroidStar(prev_x_cent=x_cent_tmp, \
                                    prev_y_cent=y_cent_tmp)

                        else:
                            return None

                    # Draw the centroid on the image
                    self.ax.scatter(self.x_centroid, self.y_centroid, marker='+', c='y', s=100, lw=3, \
                        alpha=0.5)

                    # Select the closest catalog star to the centroid as the first guess
                    self.closest_cat_star_indx = self.findClosestCatalogStarIndex(self.x_centroid, \
                        self.y_centroid)

                    # Plot the closest star as a purple cross
                    self.selected_cat_star_scatter = self.ax.scatter(self.catalog_x[self.closest_cat_star_indx], 
                        self.catalog_y[self.closest_cat_star_indx], marker='+', c='purple', s=100, lw=3)

                    # Update canvas
                    self.fig.canvas.draw()

                    # Switch to the mode where the catalog star is selected
                    self.star_selection_centroid = False

                    self.drawCursorCircle()


                # If the catalog star has to be picked
                else:

                    # Select the closest catalog star
                    self.closest_cat_star_indx = self.findClosestCatalogStarIndex(self.mouse_x_press, \
                        self.mouse_y_press)

                    if self.selected_cat_star_scatter is not None:
                        self.selected_cat_star_scatter.remove()

                    # Plot the closest star as a purple cross
                    self.selected_cat_star_scatter = self.ax.scatter(self.catalog_x[self.closest_cat_star_indx], \
                        self.catalog_y[self.closest_cat_star_indx], marker='+', c='purple', s=50, lw=2)

                    # Update canvas
                    self.fig.canvas.draw()


            # Right mouse button, deselect stars
            elif event.button == 3:

                if self.star_selection_centroid:

                    # Find the closest picked star
                    picked_indx = self.findClosestPickedStarIndex(self.mouse_x_press, self.mouse_y_press)

                    if self.paired_stars:
                        
                        # Remove the picked star from the list
                        self.paired_stars.pop(picked_indx)

                    self.updateImage()




    def onMouseMotion(self, event):
        """ Called with the mouse is moved. """


        # Read mouse position
        self.mouse_x = event.xdata
        self.mouse_y = event.ydata

        # Change the position of the star aperature circle
        if self.star_pick_mode:
            self.drawCursorCircle()



    def mouseOverStatus(self, x, y):
        """ Format the status message which will be printed in the status bar below the plot. 

        Arguments:
            x: [float] Plot X coordiante.
            y: [float] Plot Y coordinate.

        Return:
            [str]: formatted output string to be written in the status bar
        """


        # Write image X, Y coordinates and image intensity
        status_str = "x={:7.2f}  y={:7.2f}  Intens={:d}".format(x, y, self.img_data_raw[int(y), int(x)])

        # Add coordinate info if platepar is present
        if self.platepar is not None:

            # Get the current frame time
            time_data = [self.img_handle.currentTime()]

            # Compute RA, dec
            jd, ra, dec, _ = xyToRaDecPP(time_data, [x], [y], [1], self.platepar)


            # Precess RA/Dec to epoch of date for alt/az computation
            ra_date, dec_date = equatorialCoordPrecession(J2000_JD.days, jd[0], np.radians(ra[0]), 
                np.radians(dec[0]))
            ra_date, dec_date = np.degrees(ra_date), np.degrees(dec_date)

            # Compute alt, az
            azim, alt = raDec2AltAz(ra_date, dec_date, jd[0], self.platepar.lat, self.platepar.lon)


            status_str += ",  Azim={:6.2f}  Alt={:6.2f} (date),  RA={:6.2f}  Dec={:+6.2f} (J2000)".format(\
                azim, alt, ra[0], dec[0])


        return status_str

        

    def drawCursorCircle(self):
        """ Adds a circle around the mouse cursor. """

        # Delete the old aperature circle
        if self.circle_aperature is not None:
            self.circle_aperature.remove()
            self.circle_aperature = None

        if self.circle_aperature_outer is not None:
            self.circle_aperature_outer.remove()
            self.circle_aperature_outer = None

        # If centroid selection is on, show the annulus around the cursor
        if self.star_selection_centroid:

            # Plot a circle of the given radius around the cursor
            self.circle_aperature = matplotlib.patches.Circle((self.mouse_x, self.mouse_y),
                self.star_aperature_radius, edgecolor='yellow', fc='none')

            # Plot a circle of the given radius around the cursor, which is used to determine the region where the
            # background will be taken from
            self.circle_aperature_outer = matplotlib.patches.Circle((self.mouse_x, self.mouse_y),
                self.star_aperature_radius*2, edgecolor='yellow', fc='none', linestyle='dotted')

            self.ax.add_patch(self.circle_aperature)
            self.ax.add_patch(self.circle_aperature_outer)

        # If the catalog star selection mode is on, show a purple circle
        else:

            # Plot a purple circle
            self.circle_aperature = matplotlib.patches.Circle((self.mouse_x, self.mouse_y), 10, 
                edgecolor='purple', fc='none')

            self.ax.add_patch(self.circle_aperature)


        # Draw the zoom window
        if self.show_zoom_window:
            self.drawZoomWindow()


        self.fig.canvas.draw()



    def drawZoomWindow(self):
        """ Draw the zoom window in the corner. """

        if self.star_pick_mode:

            # If the mouse is outside the image, don't update anything
            if (self.mouse_x is None) or (self.mouse_y is None):
                return None

            img_w_half = self.current_ff.ncols//2
            img_h_half = self.current_ff.nrows//2

            # # Update the location of the zoom window
            # anchor_location = ''
            # if self.mouse_y > img_h_half:
            #     anchor_location += 'N'
            # else:
            #     anchor_location += 'S'

            # if self.mouse_x > img_w_half:
            #     anchor_location += 'W'
            # else:
            #     anchor_location += 'E'

            # self.zoom_axis.set_anchor(anchor_location)


            ### Extract a portion of image around the aperture ###

            window_radius = int(2*self.star_aperature_radius + np.sqrt(self.star_aperature_radius))

            x_min_orig = int(round(self.mouse_x - window_radius))
            if x_min_orig < 0: x_min = 0
            else: x_min = x_min_orig

            x_max_orig = int(round(self.mouse_x + window_radius))
            if x_max_orig > self.current_ff.ncols - 1:
                x_max = int(self.current_ff.ncols - 1)
            else: x_max = x_max_orig

            y_min_orig = int(round(self.mouse_y - window_radius))
            if y_min_orig < 0: y_min = 0
            else: y_min = y_min_orig

            y_max_orig = int(round(self.mouse_y + window_radius))
            if y_max_orig > self.current_ff.nrows - 1:
                y_max = int(self.current_ff.nrows - 1)
            else:
                y_max = y_max_orig


            # Crop the image
            img_crop = np.zeros((2*window_radius, 2*window_radius), dtype=self.img_data_processed.dtype)
            dx_min = x_min - x_min_orig
            dx_max = x_max - x_max_orig + 2*window_radius
            dy_min = y_min - y_min_orig
            dy_max = y_max - y_max_orig + 2*window_radius
            img_crop[dy_min:dy_max, dx_min:dx_max] = self.img_data_processed[y_min:y_max, x_min:x_max]


            ### ###


            # Zoom the image
            zoom_factor = 8

            # Make sure that the zoomed image is every larger than 1/2 of the whole image
            if 2*zoom_factor*window_radius > np.min([self.fig.bbox.ymax, self.fig.bbox.xmax])/2:

                # Compute a new zoom factor
                zoom_factor = np.floor((np.min([self.fig.bbox.ymax, \
                    self.fig.bbox.xmax])/2)/(2*window_radius))

                # Don't apply zoom if the image will be smaller
                if zoom_factor <= 1:
                    return None


            img_crop = scipy.ndimage.zoom(img_crop, zoom_factor, order=0)


            # Compute where the zoom will be shown on the image
            zoom_window_pos = ''
            if self.mouse_y < img_h_half:
                yo = 0
                zoom_window_pos += 'N'
            else:
                yo = self.fig.bbox.ymax - zoom_factor*2*window_radius
                zoom_window_pos += 'S'

            if self.mouse_x > img_w_half:
                xo = 0
                zoom_window_pos += 'W'
            else:
                xo = self.fig.bbox.xmax - zoom_factor*2*window_radius
                zoom_window_pos += 'E'

            # If the position of the zoom window has changed, reset the image
            if self.zoom_window_pos != zoom_window_pos:
                
                self.updateImage()
                self.zoom_window_pos = zoom_window_pos



            # Draw aperture circle to the image
            img = PIL.Image.fromarray(img_crop)
            img = img.convert('RGB')
            draw = PIL.ImageDraw.Draw(img)

            ul = zoom_factor*(window_radius - self.star_aperature_radius)
            br = zoom_factor*(window_radius + self.star_aperature_radius)
            draw.ellipse((ul, ul, br, br), outline='yellow')

            ul = zoom_factor*(window_radius - 2*self.star_aperature_radius)
            br = zoom_factor*(window_radius + 2*self.star_aperature_radius)
            draw.ellipse((ul, ul, br, br), outline='yellow')

            ### Draw centroids in the zoomed image ###

            def drawMarkersOnZoom(draw, x, y, color='blue', marker='x', width=1):

                # Check if the picked star is in the window, and plot it if it is
                if (x >= x_min_orig) and (x <= x_max_orig) and (y >= y_min_orig) and (y <= y_max_orig):

                    xp = zoom_factor*(x - x_min_orig) + 1
                    yp = zoom_factor*(y - y_min_orig) + 1

                    if marker == '+':
                        
                        # Draw an +
                        draw.line((xp + zoom_factor, yp, xp - zoom_factor, yp), fill=color, width=width)
                        draw.line((xp, yp - zoom_factor, xp, yp + zoom_factor), fill=color, width=width)

                    else:

                        # Draw an X
                        draw.line((xp + zoom_factor, yp + zoom_factor, xp - zoom_factor, yp - zoom_factor), \
                            fill=color, width=width)
                        draw.line((xp + zoom_factor, yp - zoom_factor, xp - zoom_factor, yp + zoom_factor), \
                            fill=color, width=width)


    
            # Plot centroid
            if self.x_centroid is not None:
                drawMarkersOnZoom(draw, self.x_centroid, self.y_centroid, color='green', marker='x', width=2)
                
            # Plot paired stars
            for paired_star in self.paired_stars:

                img_star, catalog_star = paired_star
                star_x, star_y, px_intens = img_star

                drawMarkersOnZoom(draw, star_x, star_y, color='blue', marker='x', width=2)


            # Draw catalog stars
            if self.catalog_stars_visible:

                for cat_x, cat_y in zip(self.catalog_x_filtered, self.catalog_y_filtered):

                    drawMarkersOnZoom(draw, cat_x, cat_y, color='red', marker='+')


            # Plot paired catalog star
            if self.closest_cat_star_indx is not None:
                
                drawMarkersOnZoom(draw, self.catalog_x[self.closest_cat_star_indx], \
                        self.catalog_y[self.closest_cat_star_indx], color='purple', marker='x', width=2)            
            

                
            ###


            # Convert the PIL image object back to array
            img_crop = np.array(img)

            # Plot the zoomed image
            self.fig.figimage(img_crop, xo=xo, yo=yo, zorder=5, cmap='gray', \
                vmin=np.min(self.img_data_processed), vmax=np.max(self.img_data_processed))



    def checkParamRange(self):
        """ Checks that the astrometry parameters are within the allowed range. """

        # Right ascension should be within 0-360
        self.platepar.RA_d = (self.platepar.RA_d + 360)%360

        # Keep the declination in the allowed range
        if self.platepar.dec_d >= 90:
            self.platepar.dec_d = 89.999

        if self.platepar.dec_d <= -90:
            self.platepar.dec_d = -89.999



    def photometry(self, show_plot=False):
        """ Perform the photometry on selectes stars. """

        if self.star_pick_mode:

            ### Make a photometry plot

            # Extract star intensities and star magnitudes
            star_coords = []
            radius_list = []
            px_intens_list = []
            catalog_ra = []
            catalog_dec = []
            catalog_mags = []
            for paired_star in self.paired_stars:

                img_star, catalog_star = paired_star

                star_x, star_y, px_intens = img_star
                star_ra, star_dec, star_mag = catalog_star


                # Skip intensities which were not properly calculated
                lsp = np.log10(px_intens)
                if np.isnan(lsp) or np.isinf(lsp):
                    continue

                star_coords.append([star_x, star_y])
                radius_list.append(np.hypot(star_x - self.platepar.X_res/2, star_y - self.platepar.Y_res/2))
                px_intens_list.append(px_intens)
                catalog_ra.append(star_ra)
                catalog_dec.append(star_dec)
                catalog_mags.append(star_mag)



            # Make sure there are more than 3 stars picked
            if len(px_intens_list) > 3:

                # Compute apparent magnitude corrected for extinction
                catalog_mags = extinctionCorrectionTrueToApparent(catalog_mags, catalog_ra, catalog_dec, \
                    date2JD(*self.img_handle.currentTime()), self.platepar)

                # Fit the photometric offset (disable vignetting fit if a flat is used)
                photom_params, fit_stddev, fit_resids = photometryFit(px_intens_list, radius_list, \
                    catalog_mags, fixed_vignetting=(0.0 if self.flat_struct is not None else None))

                photom_offset, vignetting_coeff = photom_params

                # Set photometry parameters
                self.platepar.mag_0 = -2.5
                self.platepar.mag_lev = photom_offset
                self.platepar.mag_lev_stddev = fit_stddev
                self.platepar.vignetting_coeff = vignetting_coeff


                # Remove previous photometry deviation labels 
                if len(self.photom_deviatons_scat) > 0:
                    for entry in self.photom_deviatons_scat:
                        resid_lbl, mag_lbl = entry
                        try:
                            resid_lbl.remove()
                            mag_lbl.remove()
                        except:
                            pass

                self.photom_deviatons_scat = []



                if self.catalog_stars_visible:

                    # Plot photometry deviations on the main plot as colour coded rings
                    star_coords = np.array(star_coords)
                    star_coords_x, star_coords_y = star_coords.T

                    for star_x, star_y, fit_diff, star_mag in zip (star_coords_x, star_coords_y, fit_resids, \
                        catalog_mags):

                        photom_resid_txt = "{:.2f}".format(fit_diff)

                        # Determine the size of the residual text, larger the residual, larger the text
                        photom_resid_size = 8 + np.abs(fit_diff)/(np.max(np.abs(fit_resids))/5.0)

                        # Plot the residual as text under the star
                        photom_resid_lbl = self.ax.text(star_x, star_y + 10, photom_resid_txt, \
                            verticalalignment='top', horizontalalignment='center', \
                            fontsize=photom_resid_size, color='w')

                        # Plot the star magnitude
                        star_mag_lbl = self.ax.text(star_x, star_y - 10, "{:+6.2f}".format(star_mag), \
                            verticalalignment='bottom', horizontalalignment='center', \
                            fontsize=10, color='r')


                        self.photom_deviatons_scat.append([photom_resid_lbl, star_mag_lbl])


                    self.fig.canvas.draw_idle()


                # Show the photometry fit plot
                if show_plot:

                    ### PLOT PHOTOMETRY FIT ###
                    # Note: An almost identical code exists in Utils.CalibrationReport

                    # Init plot for photometry
                    fig_p, (ax_p, ax_r) = plt.subplots(nrows=2, facecolor=None, figsize=(6.4, 7.2), \
                        gridspec_kw={'height_ratios':[2, 1]})

                    # Set photometry window title
                    fig_p.canvas.set_window_title('Photometry')


                    # Plot catalog magnitude vs. raw logsum of pixel intensities
                    lsp_arr = np.log10(np.array(px_intens_list))
                    ax_p.scatter(-2.5*lsp_arr, catalog_mags, s=5, c='r', zorder=3, alpha=0.5, \
                        label="Raw (extinction corrected)")

                    # Plot catalog magnitude vs. raw logsum of pixel intensities (only when no flat is used)
                    if self.flat_struct is None:

                        lsp_corr_arr = np.log10(correctVignetting(np.array(px_intens_list), \
                            np.array(radius_list), self.platepar.vignetting_coeff))

                        ax_p.scatter(-2.5*lsp_corr_arr, catalog_mags, s=5, c='b', zorder=3, alpha=0.5, \
                            label="Corrected for vignetting")


                    x_min, x_max = ax_p.get_xlim()
                    y_min, y_max = ax_p.get_ylim()

                    x_min_w = x_min - 3
                    x_max_w = x_max + 3
                    y_min_w = y_min - 3
                    y_max_w = y_max + 3

                    
                    # Plot fit info
                    fit_info = "Fit: {:+.1f}*LSP + {:.2f} +/- {:.2f} ".format(self.platepar.mag_0, \
                        self.platepar.mag_lev, fit_stddev) \
                        + "\nVignetting coeff = {:.5f}".format(self.platepar.vignetting_coeff) \
                        + "\nGamma = {:.2f}".format(self.platepar.gamma)

                    print(fit_info)

                    # Plot the line fit
                    logsum_arr = np.linspace(x_min_w, x_max_w, 10)
                    ax_p.plot(logsum_arr, photomLine((10**(logsum_arr/(-2.5)), np.zeros_like(logsum_arr)), \
                        photom_offset, self.platepar.vignetting_coeff), label=fit_info, \
                        linestyle='--', color='k', alpha=0.5, zorder=3)

                    ax_p.legend()
                        
                    ax_p.set_ylabel("Catalog magnitude ({:s})".format(self.mag_band_string))
                    ax_p.set_xlabel("Uncalibrated magnitude")

                    # Set wider axis limits
                    ax_p.set_xlim(x_min_w, x_max_w)
                    ax_p.set_ylim(y_min_w, y_max_w)

                    ax_p.invert_yaxis()
                    ax_p.invert_xaxis()

                    ax_p.grid()

                    ###


                    ### PLOT MAG DIFFERENCE BY RADIUS

                    img_diagonal = np.hypot(self.platepar.X_res/2, self.platepar.Y_res/2)                        
                    

                    # Plot radius from centre vs. fit residual (including vignetting)
                    ax_r.scatter(radius_list, fit_resids, s=5, c='b', alpha=0.5, zorder=3)


                    # Plot a zero line
                    ax_r.plot(np.linspace(0, img_diagonal, 10), np.zeros(10), linestyle='dashed', alpha=0.5, \
                        color='k')



                    # Plot the vignetting curve (only when no flat is used)
                    if self.flat_struct is None:

                        # Plot radius from centre vs. fit residual (excluding vignetting
                        fit_resids_novignetting = catalog_mags - photomLine((np.array(px_intens_list), \
                            np.array(radius_list)), photom_offset, 0.0)
                        ax_r.scatter(radius_list, fit_resids_novignetting, s=5, c='r', alpha=0.5, zorder=3)

                        px_sum_tmp = 1000
                        radius_arr_tmp = np.linspace(0, img_diagonal, 50)

                        # Plot the vignetting curve
                        vignetting_loss = 2.5*np.log10(px_sum_tmp) \
                            - 2.5*np.log10(correctVignetting(px_sum_tmp, radius_arr_tmp, \
                                self.platepar.vignetting_coeff))

                        ax_r.plot(radius_arr_tmp, vignetting_loss, linestyle='dotted', alpha=0.5, color='k')
                    

                    ax_r.grid()

                    ax_r.set_ylabel("Fit residuals (mag)")
                    ax_r.set_xlabel("Radius from centre (px)")

                    ax_r.set_xlim(0, img_diagonal)

                    
                    fig_p.tight_layout()
                    fig_p.show()


            else:
                print('Need more than 2 stars for photometry plot!')



    def updateRefRADec(self, skip_rot_update=False):
        """ Update the reference RA and Dec (true in J2000) from Alt/Az (apparent in epoch of date). """

        if not skip_rot_update:
            
            # Save the current rotation w.r.t horizon value
            self.platepar.rotation_from_horiz = rotationWrtHorizon(self.platepar)


        # Convert the reference apparent Alt/Az in the epoch of date to true RA/Dec in J2000
        ra, dec = apparentAltAz2TrueRADec(\
            np.radians(self.platepar.az_centre), np.radians(self.platepar.alt_centre), self.platepar.JD, \
            np.radians(self.platepar.lat), np.radians(self.platepar.lon))


        # Assign the computed RA/Dec to platepar
        self.platepar.RA_d = np.degrees(ra)
        self.platepar.dec_d = np.degrees(dec)


        if not skip_rot_update:

            # Update the position angle so that the rotation wrt horizon doesn't change
            self.platepar.pos_angle_ref = rotationWrtHorizonToPosAngle(self.platepar, \
                self.platepar.rotation_from_horiz)



    def onKeyPress(self, event):
        """ Traige what happes when an individual key is pressed. """


        # Switch images
        if event.key == 'left':
            self.prevImg()

        elif event.key == 'right':
            self.nextImg()


        elif event.key == 'm':
            self.toggleImageType()

        elif event.key == ',':

            # Decrement UT correction
            self.platepar.UT_corr -= 0.5

            # Update platepar JD
            self.platepar.JD += 0.5/24


            self.updateImage()


        elif event.key == '.':

            # Decrement UT correction
            self.platepar.UT_corr += 0.5

            # Update platepar JD
            self.platepar.JD -= 0.5/24


            self.updateImage()            


        # Move RA/Dec
        elif event.key == 'a':

            self.platepar.az_centre += self.key_increment
            self.updateRefRADec()

            self.updateImage()

        elif event.key == 'd':

            self.platepar.az_centre -= self.key_increment
            self.updateRefRADec()

            self.updateImage()

        elif event.key == 'w':

            self.platepar.alt_centre -= self.key_increment
            self.updateRefRADec()

            self.updateImage()

        elif event.key == 's':

            self.platepar.alt_centre += self.key_increment
            self.updateRefRADec()

            self.updateImage()

        # Move rotation parameter
        elif event.key == 'q':
            self.platepar.pos_angle_ref -= self.key_increment
            self.updateImage()

        elif event.key == 'e':
            self.platepar.pos_angle_ref += self.key_increment
            self.updateImage()


        # Change extinction scale
        elif event.key == '9':
            self.platepar.extinction_scale -= 0.1
            if self.platepar.extinction_scale < 0.1:
                self.platepar.extinction_scale = 0.1

            self.updateImage()

        # Change extinction scale
        elif event.key == '0':
            self.platepar.extinction_scale += 0.1
            if self.platepar.extinction_scale > 2.0:
                self.platepar.extinction_scale = 2.0

            self.updateImage()


        # Change catalog limiting magnitude
        elif event.key == 'r':
            self.cat_lim_mag += 0.1
            self.catalog_stars = self.loadCatalogStars(self.cat_lim_mag)
            self.updateImage()

        elif event.key == 'f':
            self.cat_lim_mag -= 0.1
            self.catalog_stars = self.loadCatalogStars(self.cat_lim_mag)
            self.updateImage()

        # Show/hide keyboard shortcut help
        elif event.key == 'f1':

            # Go through states of text visibility
            self.show_key_help += 1

            if self.show_key_help >= 3:
                self.show_key_help = 0


            self.updateImage()

        # Change image scale
        elif event.key == 'up':

            self.platepar.F_scale *= 1.0 + self.key_increment/100.0
            self.updateImage()

        elif event.key == 'down':
            self.platepar.F_scale *= 1.0 - self.key_increment/100.0
            self.updateImage()


        elif event.key == '1':

            # Increment X offset
            self.platepar.x_poly_rev[0] += 0.5
            self.platepar.x_poly_fwd[0] += 0.5
            self.updateImage()

        elif event.key == '2':

            # Decrement X offset
            self.platepar.x_poly_rev[0] -= 0.5
            self.platepar.x_poly_fwd[0] -= 0.5
            self.updateImage()


        elif event.key == '3':

            # Increment Y offset            
            self.platepar.y_poly_rev[0] += 0.5
            self.platepar.y_poly_fwd[0] += 0.5
            self.updateImage()

        elif event.key == '4':

            # Decrement Y offset
            self.platepar.y_poly_rev[0] -= 0.5
            self.platepar.y_poly_fwd[0] -= 0.5
            self.updateImage()

        elif event.key == '5':

            # Decrement X 1st order distortion
            self.platepar.x_poly_rev[1] -= 0.01
            self.platepar.x_poly_fwd[1] -= 0.01
            self.updateImage()

        elif event.key == '6':

            # Increment X 1st order distortion
            self.platepar.x_poly_rev[1] += 0.01
            self.platepar.x_poly_fwd[1] += 0.01
            self.updateImage()


        elif event.key == '7':

            # Decrement Y 1st order distortion
            self.platepar.y_poly_rev[2] -= 0.01
            self.platepar.y_poly_fwd[2] -= 0.01
            self.updateImage()

        elif event.key == '8':

            # Increment Y 1st order distortion
            self.platepar.y_poly_rev[2] += 0.01
            self.platepar.y_poly_fwd[2] += 0.01
            self.updateImage()


        # Set distortion types
        elif event.key == 'ctrl+1':

            self.dist_type_index = 0
            self.changeDistortionType()


        elif event.key == 'ctrl+2':

            self.dist_type_index = 1
            self.changeDistortionType()


        elif event.key == 'ctrl+3':

            self.dist_type_index = 2
            self.changeDistortionType()

        elif event.key == 'ctrl+4':

            self.dist_type_index = 3
            self.changeDistortionType()


        # Key increment
        elif event.key == '+':

            if self.key_increment <= 0.091:
                self.key_increment += 0.01
            elif self.key_increment <= 0.91:
                self.key_increment += 0.1
            else:
                self.key_increment += 1.0

            # Don't allow the increment to be larger than 20
            if self.key_increment > 20:
                self.key_increment = 20

            self.updateImage()


        elif event.key == '-':
            
            if self.key_increment <= 0.11:
                self.key_increment -= 0.01
            elif self.key_increment <= 1.11:
                self.key_increment -= 0.1
            else:
                self.key_increment -= 1.0

            # Don't allow the increment to be smaller than 0
            if self.key_increment <= 0:
                self.key_increment = 0.01
            
            self.updateImage()


        # Enter FOV centre
        elif event.key == 'v':

            self.platepar.RA_d, self.platepar.dec_d, self.platepar.rotation_from_horiz = self.getFOVcentre()
            
            # Compute reference Alt/Az to apparent coordinates, epoch of date
            az_centre, alt_centre = trueRaDec2ApparentAltAz( \
                np.radians(self.platepar.RA_d), np.radians(self.platepar.dec_d), self.platepar.JD, \
                np.radians(self.platepar.lat), np.radians(self.platepar.lon))

            self.platepar.az_centre, self.platepar.alt_centre = np.degrees(az_centre), np.degrees(alt_centre)

            # Compute the position angle
            self.platepar.pos_angle_ref = rotationWrtHorizonToPosAngle(self.platepar, \
                self.platepar.rotation_from_horiz)
            
            self.updateImage()


        # Write out the new platepar
        elif event.key == 'ctrl+s':

            # If the platepar is new, save it to the working directory
            if not self.platepar_file:
                self.platepar_file = os.path.join(self.dir_path, self.config.platepar_name)

            # Save the platepar file
            self.platepar.write(self.platepar_file, fmt=self.platepar_fmt, fov=computeFOVSize(self.platepar))
            print('Platepar written to:', self.platepar_file)

            # Save the state
            self.saveState()


        # Save the platepar as default (SHIFT+CTRL+S)
        elif event.key == 'ctrl+S':

            platepar_default_path = os.path.join(os.getcwd(), self.config.platepar_name)

            # Save the platepar file
            self.platepar.write(platepar_default_path, fmt=self.platepar_fmt)
            print('Default platepar written to:', platepar_default_path)


        # Create a new platepar
        elif event.key == 'ctrl+n':
            self.makeNewPlatepar()


        # Get initial parameters from astrometry.net
        elif (event.key == 'ctrl+x') or (event.key == 'ctrl+X'):

            # Overlay text on image indicating that astrometry.net is running
            self.ax.text(self.img_data_raw.shape[1]/2, self.img_data_raw.shape[0]/2, \
                "Solving with astrometry.net...", color='r', alpha=0.5, fontsize=16, ha='center', va='center')

            #self.ax.draw()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            # If shift was pressed, send only the list of x,y coords of extracted stars
            upload_image = True
            if event.key == 'ctrl+X':
                upload_image = False

            # Estimate initial parameters using astrometry.net
            self.getInitialParamsAstrometryNet(upload_image=upload_image)

            self.updateImage()


        # Toggle auto levels
        elif event.key == 'ctrl+a':
            self.auto_levels = not self.auto_levels

            self.updateImage()


        # Load the flat field
        elif event.key == 'ctrl+f':
            _, self.flat_struct = self.loadFlat()

            self.updateImage()


        # Load the dark frame
        elif event.key == 'ctrl+d':
            _, self.dark = self.loadDark()

            # Apply the dark to the flat
            if self.flat_struct is not None:
                self.flat_struct.applyDark(self.dark)

            self.updateImage()


        # Show/hide catalog stars
        elif event.key == 'h':

            self.catalog_stars_visible = not self.catalog_stars_visible

            self.updateImage()


        # Show/hide detected stars
        elif event.key == 'c':

            self.draw_calstars = not self.draw_calstars

            self.updateImage()


        # Show/hide distortion guides
        elif event.key == 'ctrl+i':

            self.draw_distortion = not self.draw_distortion

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


        # Toggle equal aspect
        elif event.key == 'g':

            if self.platepar is not None:

                self.platepar.equal_aspect = not self.platepar.equal_aspect

                self.updateImage()


        elif event.key == 'ctrl+h':

            # Toggle levels adustment mode
            self.adjust_levels_mode = not self.adjust_levels_mode

            self.updateImage()


        elif event.key == 'i':

            # Invert image levels
            self.invert_levels = not self.invert_levels

            self.updateImage()


        # Change modes from astrometry parameter changing to star picking
        elif event.key == 'ctrl+r':

            if self.star_pick_mode:
                self.disableStarPicking()
                self.circle_aperature = None
                self.circle_aperature_outer = None
                self.updateImage()

            else:
                self.enableStarPicking()



        # Toggle zoom window (SHIFT + Z)
        elif event.key == 'Z':
            
            if self.star_pick_mode:

                self.show_zoom_window = not self.show_zoom_window

                self.updateImage()

                self.drawCursorCircle()



        # Do a fit on the selected stars while in the star picking mode
        elif (event.key == 'ctrl+z') or (event.key == "ctrl+Z"):

            # If shift was pressed, reset distortion parameters to zero
            if event.key == "ctrl+Z":
                self.platepar.resetDistortionParameters()
                self.first_platepar_fit = True

            # If the first platepar is being made, do the fit twice
            if self.first_platepar_fit:
                self.fitPickedStars(first_platepar_fit=True)
                self.fitPickedStars(first_platepar_fit=True)
                self.first_platepar_fit = False

            else:
                # Otherwise, only fit the once
                self.fitPickedStars(first_platepar_fit=False)


            print('Plate fitted!')


        elif event.key == 'enter':

            if self.star_pick_mode:

                # If the right catalog star has been selected, save the pair to the list
                if not self.star_selection_centroid:

                    # Add the image/catalog pair to the list
                    self.paired_stars.append([[self.x_centroid, self.y_centroid, self.star_intensity], 
                        self.catalog_stars[self.closest_cat_star_indx]])

                    # Switch back to centroiding mode
                    self.star_selection_centroid = True
                    self.closest_cat_star_indx = None

                    self.updateImage()

                    self.drawCursorCircle()


        elif event.key == 'escape':

            if self.star_pick_mode:

                # If the ESC is pressed when the star has been centroided, reset the centroid
                if not self.star_selection_centroid:
                    self.star_selection_centroid = True
                    self.x_centroid = None
                    self.y_centroid = None
                    self.star_intensity = None

                    self.updateImage()

                    self.drawCursorCircle()


        elif event.key == 'p':

            # Show the photometry plot
            self.photometry(show_plot=True)


        elif event.key == 'l':

            if self.star_pick_mode:

                # Show astrometry residuals plot
                self.showAstrometryFitPlots()


        # Limit values of RA and Dec
        self.platepar.RA_d = self.platepar.RA_d%360

        if self.platepar.dec_d > 90:
            self.platepar.dec_d = 90

        elif self.platepar.dec_d < -90:
            self.platepar.dec_d = -90



    def onScroll(self, event):
        """ Change star selector aperature on scroll. """

        self.scroll_counter += event.step


        if self.scroll_counter > 1:
            self.star_aperature_radius += 1
            self.scroll_counter = 0

        elif self.scroll_counter < -1:
            self.star_aperature_radius -= 1
            self.scroll_counter = 0


        # Check that the star aperature is in the proper limits
        if self.star_aperature_radius < 2:
            self.star_aperature_radius = 2

        if self.star_aperature_radius > 100:
            self.star_aperature_radius = 100

        # Change the size of the star aperature circle
        if self.star_pick_mode:
            self.updateImage()
            self.drawCursorCircle()




    def toggleImageType(self):
        """ Toggle between the maxpixel and avepixel. """

        if self.img_type_flag == 'maxpixel':
            self.img_type_flag = 'avepixel'

        else:
            self.img_type_flag = 'maxpixel'

        self.updateImage()
            

    def loadCatalogStars(self, lim_mag):
        """ Loads stars from the BSC star catalog. 
    
        Arguments:
            lim_mag: [float] Limiting magnitude of catalog stars.

        """

        # Load the star catalog
        catalog_status = StarCatalog.readStarCatalog(self.config.star_catalog_path, \
            self.config.star_catalog_file, lim_mag=lim_mag, \
            mag_band_ratios=self.config.star_catalog_band_ratios)

        if catalog_status is False:
            raise FileNotFoundError("The star catalog file could not be loaded: {:s}".format(\
                os.path.join(self.config.star_catalog_path, self.config.star_catalog_file)))

        
        catalog_stars, self.mag_band_string, self.config.star_catalog_band_ratios = catalog_status

        return catalog_stars


    def drawPairedStars(self):
        """ Draws the stars that were picked for calibration. """

        if self.star_pick_mode:

            # Go through all paired stars
            for paired_star in self.paired_stars:

                img_star, catalog_star = paired_star

                x, y, _ = img_star

                # Plot all paired stars
                self.ax.scatter(x, y, marker='x', c='b', s=100, lw=3, alpha=0.5)


    def drawDistortion(self):
        """ Draw distortion guides. """

        # Only draw the distortion if we have a platepar
        if self.platepar:

            # Sample points on every image axis (start/end 5% from image corners)
            x_samples = 30
            y_samples = int(x_samples*(self.platepar.Y_res/self.platepar.X_res))
            corner_frac = 0.05
            x_samples = np.linspace(corner_frac*self.platepar.X_res, (1 - corner_frac)*self.platepar.X_res, \
                x_samples)
            y_samples = np.linspace(corner_frac*self.platepar.Y_res, (1 - corner_frac)*self.platepar.Y_res, \
                y_samples)

            # Create a platepar with no distortion
            platepar_nodist = copy.deepcopy(self.platepar)
            platepar_nodist.resetDistortionParameters(preserve_centre=True)

            # Make X, Y pairs
            xx, yy = np.meshgrid(x_samples, y_samples)
            x_arr, y_arr = np.stack([np.ravel(xx), np.ravel(yy)], axis=-1).T

            # Compute RA/Dec using the normal platepar for all pairs
            level_data = np.ones_like(x_arr)
            time_data = [self.img_handle.currentTime()]*len(x_arr)
            _, ra_data, dec_data, _ = xyToRaDecPP(time_data, x_arr, y_arr, level_data, self.platepar)

            # Compute X, Y back without the distortion
            jd = date2JD(*self.img_handle.currentTime())
            x_nodist, y_nodist = raDecToXYPP(ra_data, dec_data, jd, platepar_nodist)

            # Plot the differences in X, Y
            data = []
            color = 'r'
            for x0, y0, xnd, ynd in zip(x_arr, y_arr, x_nodist, y_nodist):
                data.append([x0, xnd])
                data.append([y0, ynd])
                data.append(color)

            plt.plot(*data, alpha=0.5)


    def changeDistortionType(self):
        """ Change the distortion type. """

        dist_type = self.platepar.distortion_type_list[self.dist_type_index]
        self.platepar.setDistortionType(dist_type)
        self.updateImage()

        # Indicate that the platepar has been reset
        self.first_platepar_fit = True

        print("Distortion model changed to: {:s}".format(dist_type))



    def drawLevelsAdjustmentHistogram(self, img):
        """ Draw a levels histogram over the image, so the levels can be adjusted. """

        # If auto levels are used, show those levels
        if self.auto_levels:
            level_min = self.img_level_min_auto
            level_max = self.img_level_max_auto

        else:
            level_min = self.img_level_min
            level_max = self.img_level_max

        nbins = int((2**self.config.bit_depth)/2)

        # Compute the intensity histogram
        hist, bin_edges = np.histogram(img.flatten(), density=True, range=(0, 2**self.bit_depth), \
            bins=nbins)

        # Scale the maximum histogram peak to half the image height
        hist *= img.shape[0]/np.max(hist)/2

        # Scale the edges to image width
        image_to_level_scale = img.shape[1]/np.max(bin_edges)
        bin_edges *= image_to_level_scale

        ax1 = self.ax
        ax2 = ax1.twinx().twiny()

        # Plot the histogram
        ax2.bar(bin_edges[:-1], hist, color='white', alpha=0.5, width=img.shape[1]/nbins, edgecolor='0.5')

        # Plot levels limits
        y_range = np.linspace(0, img.shape[0], 3)
        x_arr = np.zeros_like(y_range)

        ax2.plot(x_arr + level_min*image_to_level_scale, y_range, color='w')
        ax2.plot(x_arr + level_max*image_to_level_scale, y_range, color='w')

        ax2.invert_yaxis()

        ax2.set_ylim([0, img.shape[0]])
        ax2.set_xlim([0, img.shape[1]])

        # Set background color to black
        ax1.set_facecolor('black')
        ax2.set_facecolor('black')

        # Plot the image level range
        for i in range(8):
            rel_i = i/8
            ax2.text(rel_i*img.shape[1], 0, str(int(rel_i*2**self.bit_depth)), color='red')

        self.fig.sca(ax1)





    def updateImage(self, clear_plot=True, first_update=False):
        """ Update the matplotlib plot to show the current image. 

        Keyword arguments:
            clear_plot: [bool] If True, the plot will be cleared before plotting again (default).
            first_update: [bool] Special lines will be executed if this is True, e.g. the max level will be 
                computed. False by default.
        """

        # Reset circle patches
        self.circle_aperature = None
        self.circle_aperature_outer = None

        # Limit key increment so it can be lower than 0.01
        if self.key_increment < 0.01:
            self.key_increment = 0.01


        if clear_plot:
            self.fig.clf()
            self.ax = self.fig.add_subplot(111)

            # Set the status update formatter
            self.ax.format_coord = self.mouseOverStatus


        # Check that the calibration parameters are within the nominal range
        self.checkParamRange()


        # Load the current image
        self.current_ff = self.img_handle.loadChunk()


        if self.current_ff is None:
            
            # If the current FF couldn't be opened, go to the next
            messagebox.showerror(title='Read error', message='The current image is corrupted!')
            self.nextImg()

            return None


        # Choose appropriate image data
        if self.img_type_flag == 'maxpixel':
            img_data = self.current_ff.maxpixel

        else:
            img_data = self.current_ff.avepixel


        # Guess the bit depth from the array type
        self.bit_depth = 8*img_data.itemsize


        if first_update:
            
            # Set the maximum image level after reading the bit depth
            self.img_level_max = 2**self.bit_depth - 1


            # Set the image resolution to platepar after reading the first image
            self.config.width = img_data.shape[1]
            self.config.height = img_data.shape[0]
            self.platepar.X_res = img_data.shape[1]
            self.platepar.Y_res = img_data.shape[0]


            # # Scale the size of the annulus (normalize so the percieved size is the same as for 1280x720)
            # self.star_aperature_radius *= np.sqrt(img_data.shape[1]**2 + img_data.shape[0]**2)/1500



        # Apply dark
        if self.dark is not None:
            img_data = Image.applyDark(img_data, self.dark)


        # Apply flat
        if self.flat_struct is not None:
            img_data = Image.applyFlat(img_data, self.flat_struct)


        # Store image before levels modifications
        self.img_data_raw = np.copy(img_data)
            

        # Do auto levels
        if self.auto_levels:

            # Compute the edge percentiles
            self.img_level_min_auto = np.percentile(img_data, 0.1)
            self.img_level_max_auto = np.percentile(img_data, 99.95)


            # Adjust levels (auto)
            img_data = Image.adjustLevels(img_data, self.img_level_min_auto, self.img_gamma, \
                self.img_level_max_auto, scaleto8bits=True)

        else:
            
            # Adjust levels (manual)
            img_data = Image.adjustLevels(img_data, self.img_level_min, self.img_gamma, self.img_level_max,
                scaleto8bits=True)



        # Draw levels adjustment histogram
        if self.adjust_levels_mode:
            self.drawLevelsAdjustmentHistogram(self.img_data_raw)


        # Invert the image (assume 8 bit image)
        if self.invert_levels:
            img_data = 255 - img_data


        # Store image after levels modifications
        self.img_data_processed = np.copy(img_data)

        # Show the loaded image (defining the exent speeds up image drawimg)
        self.ax.imshow(img_data, cmap='gray', interpolation='nearest', \
            extent=(0, self.img_data_processed.shape[1], self.img_data_processed.shape[0], 0),
            vmin=0, vmax=255)

        # Draw stars that were paired in picking mode
        self.drawPairedStars()

        # Draw stars detected on this image
        if self.draw_calstars:
            self.drawCalstars()

        # Update centre of FOV in horizontal coordinates
        self.platepar.az_centre, self.platepar.alt_centre = raDec2AltAz(self.platepar.RA_d, \
            self.platepar.dec_d, self.platepar.JD, self.platepar.lat, self.platepar.lon)

        ### Draw catalog stars on the image using the current platepar ###
        ######################################################################################################

        # Get positions of catalog stars on the image
        ff_jd = date2JD(*self.img_handle.currentTime())
        self.catalog_x, self.catalog_y, catalog_mag = getCatalogStarsImagePositions(self.catalog_stars, \
            ff_jd, self.platepar)

        if self.catalog_stars_visible:
            cat_stars = np.c_[self.catalog_x, self.catalog_y, catalog_mag]

            # Take only those stars inside the FOV
            filtered_indices, _ = self.filterCatalogStarsInsideFOV(self.catalog_stars)
            cat_stars = cat_stars[filtered_indices]
            cat_stars = cat_stars[cat_stars[:, 0] > 0]
            cat_stars = cat_stars[cat_stars[:, 0] < self.current_ff.ncols]
            cat_stars = cat_stars[cat_stars[:, 1] > 0]
            cat_stars = cat_stars[cat_stars[:, 1] < self.current_ff.nrows]

            self.catalog_x_filtered, self.catalog_y_filtered, catalog_mag_filtered = cat_stars.T

            if len(catalog_mag_filtered):

                cat_mag_faintest = np.max(catalog_mag_filtered)

                # Plot catalog stars
                self.ax.scatter(self.catalog_x_filtered, self.catalog_y_filtered, c='r', marker='+', lw=1.0, \
                    alpha=0.5, s=((4.0 + (cat_mag_faintest - catalog_mag_filtered))/2.0)**(2*2.512))

            else:
                print('No catalog stars visible!')

        ######################################################################################################


        # Draw distortion guides
        if self.draw_distortion:
            self.drawDistortion()


        # Draw photometry
        if len(self.paired_stars) > 2:
            self.photometry()

        # Draw fit residuals
        if self.residuals is not None:
            self.drawFitResiduals()


        # Set plot limits
        self.ax.set_xlim([0, self.current_ff.ncols])
        self.ax.set_ylim([self.current_ff.nrows, 0])


        # Compute RA/Dec of the FOV centre
        ra_centre, dec_centre = self.computeCentreRADec()


        ### Setup a monospace font ###
        font = FontProperties()
        font.set_family('monospace')
        font.set_size(8)

        if self.invert_levels:
            font_color = 'k'
        else:
            font_color = 'w'

        ### ###

        
        if self.show_key_help == 0:
            text_str = 'Show fit parameters - F1'

            self.ax.text(10, self.current_ff.nrows, text_str, color=font_color, verticalalignment='bottom', \
                horizontalalignment='left', fontproperties=font)


        # Show plate info
        if self.show_key_help > 0:

            # Show text on image with platepar parameters
            text_str  = self.img_handle.name() + '\n' + self.img_type_flag + '\n\n'
            text_str += 'UT corr  = {:.1f}h\n'.format(self.platepar.UT_corr)
            text_str += 'Ref Az   = {:.3f}$\\degree$\n'.format(self.platepar.az_centre)
            text_str += 'Ref Alt  = {:.3f}$\\degree$\n'.format(self.platepar.alt_centre)
            text_str += 'Rot horiz = {:.3f}$\\degree$\n'.format(rotationWrtHorizon(self.platepar))
            text_str += 'Rot eq    = {:.3f}$\\degree$\n'.format(rotationWrtStandard(self.platepar))
            #text_str += 'Ref RA  = {:.3f}\n'.format(self.platepar.RA_d)
            #text_str += 'Ref Dec = {:.3f}\n'.format(self.platepar.dec_d)
            text_str += "Pix scale = {:.3f}'/px\n".format(60/self.platepar.F_scale)
            text_str += 'Lim mag   = {:.1f}\n'.format(self.cat_lim_mag)
            text_str += 'Increment = {:.2f}\n'.format(self.key_increment)
            text_str += 'Img Gamma = {:.2f}\n'.format(self.img_gamma)
            text_str += 'Camera Gamma = {:.2f}\n'.format(self.config.gamma)
            text_str += 'Extinct. scale  = {:.1f}x\n'.format(self.platepar.extinction_scale)
            text_str += "Refraction corr = {:s}\n".format(str(self.platepar.refraction))
            text_str += "Distortion type = {:s}\n".format(\
                self.platepar.distortion_type_list[self.dist_type_index])

            # Add aspect info if the radial distortion is used
            if not self.platepar.distortion_type.startswith("poly"):
                text_str += "Equal aspect    = {:s}\n".format(str(self.platepar.equal_aspect))

            text_str += '\n'
            sign, hh, mm, ss = decimalDegreesToSexHours(ra_centre)
            if sign < 0:
                sign_str = '-'
            else:
                sign_str = ' '
            text_str += 'RA centre  = {:s}{:02d}h {:02d}m {:05.2f}s\n'.format(sign_str, hh, mm, ss)
            text_str += 'Dec centre = {:.3f}$\\degree$\n'.format(dec_centre)
            self.ax.text(10, 10, text_str, color=font_color, verticalalignment='top', \
                horizontalalignment='left', fontproperties=font)


            if self.show_key_help == 1:
                text_str = 'Show keyboard shortcuts - F1'

                self.ax.text(10, self.current_ff.nrows, text_str, color=font_color, \
                    verticalalignment='bottom', horizontalalignment='left', fontproperties=font)

        # Show keyboard shortcuts
        if self.show_key_help > 1:

            # Show text on image with instructions
            text_str  = 'Keys:\n'
            text_str += '-----\n'
            text_str += 'A/D - Azimuth\n'
            text_str += 'S/W - Altitude\n'
            text_str += 'Q/E - Position angle\n'
            text_str += 'Up/Down - Scale\n'
            text_str += 'T - Toggle refraction correction\n'
            text_str += "9/0 - Extinction scale\n"

            # Add aspect info if the radial distortion is used
            if not self.platepar.distortion_type.startswith("poly"):
                text_str += 'G - Toggle equal aspect\n'

            text_str += '1/2 - X offset\n'
            text_str += '3/4 - Y offset\n'
            text_str += '5/6 - X 1st dist. coeff.\n'
            text_str += '7/8 - Y 1st dist. coeff.\n'
            text_str += 'CTRL + 1 - poly3+radial distortion\n'
            text_str += 'CTRL + 2 - radial3 distortion\n'
            text_str += 'CTRL + 3 - radial4 distortion\n'
            text_str += 'CTRL + 4 - radial5 distortion\n'
            text_str += '\n'
            text_str += ',/. - UT correction\n'
            text_str += 'R/F - Lim mag\n'
            text_str += '+/- - Increment\n'
            text_str += '\n'
            text_str += 'M - Toggle maxpixel/avepixel\n'
            text_str += 'H - Hide/show catalog stars\n'
            text_str += 'C - Hide/show detected stars\n'
            text_str += 'CTRL + I - Show/hide distortion\n'
            text_str += 'U/J - Img Gamma\n'
            text_str += 'I - Invert colors\n'
            text_str += 'CTRL + H - Adjust levels\n'
            text_str += 'V - FOV centre\n'
            text_str += '\n'
            text_str += 'CTRL + A - Auto levels\n'
            text_str += 'CTRL + D - Load dark\n'
            text_str += 'CTRL + F - Load flat\n'
            text_str += 'CTRL + X - astrometry.net img upload\n'
            text_str += 'CTRL + SHIFT + X - astrometry.net XY only\n'
            text_str += 'CTRL + R - Pick stars\n'
            text_str += 'SHIFT + Z - Show zoomed window\n'
            text_str += 'CTRL + N - New platepar\n'
            text_str += 'CTRL + S - Save platepar\n'
            text_str += 'SHIFT + CTRL + S - Save platepar as default\n'

            text_str += '\n'

            text_str += 'Hide on-screen text - F1\n'


            self.ax.text(8, self.current_ff.nrows - 5, text_str, color=font_color, \
                verticalalignment='bottom', horizontalalignment='left', fontproperties=font)


        # Show fitting instructions
        if self.star_pick_mode:
            text_str  = "STAR PICKING MODE"

            if self.show_key_help > 0:
                text_str += "\n'LEFT CLICK' - Centroid star\n"
                text_str += "'CTRL + LEFT CLICK' - Manual star position\n"
                text_str += "'CTRL + Z' - Fit stars\n"
                text_str += "'CTRL + SHIFT + Z' - Fit with initial distortion params set to 0\n"
                text_str += "'L' - Astrometry fit details\n"
                text_str += "'P' - Photometry fit"

            self.ax.text(self.current_ff.ncols/2, self.current_ff.nrows, text_str, color='r', \
                verticalalignment='bottom', horizontalalignment='center', fontproperties=font)


        self.fig.canvas.draw()



    def drawCalstars(self):
        """ Draw extracted stars on the current image. """

        # Check if the given FF files is in the calstars list
        if self.img_handle.name() in self.calstars:

            # Get the stars detected on this FF file
            star_data = self.calstars[self.img_handle.name()]

            # Get star coordinates
            y, x, _, _ = np.array(star_data).T

            self.ax.scatter(x, y, edgecolors='g', marker='o', facecolors='none', alpha=0.8, \
                linestyle='dotted')



    def updateGamma(self, gamma_adj_factor):
        """ Change the image gamma by a given factor. """

        self.img_gamma *= gamma_adj_factor

        # Make sure gamma is in the proper range
        if self.img_gamma < 0.1: self.img_gamma = 0.1
        if self.img_gamma > 10: self.img_gamma = 10

        self.updateImage()


    def computeCentreRADec(self):
        """ Compute RA and Dec of the FOV centre in degrees. """

        # The the time of the image
        img_time = self.img_handle.currentTime()

        # Convert the FOV centre to RA/Dec
        _, ra_centre, dec_centre, _ = xyToRaDecPP([img_time], [self.platepar.X_res/2], 
            [self.platepar.Y_res/2], [1], self.platepar)
        
        ra_centre = ra_centre[0]
        dec_centre = dec_centre[0]

        return ra_centre, dec_centre


    def filterCatalogStarsInsideFOV(self, catalog_stars):
        """ Take only catalogs stars which are inside the FOV. 
        
        Arguments:
            catalog_stars: [list] A list of (ra, dec, mag) tuples of catalog stars.
        """

        # Get RA/Dec of the FOV centre
        ra_centre, dec_centre = self.computeCentreRADec()

        # Calculate the FOV radius in degrees
        fov_y, fov_x = computeFOVSize(self.platepar)
        fov_radius = np.sqrt(fov_x**2 + fov_y**2)

        # Compute the current Julian date
        jd = date2JD(*self.img_handle.currentTime())
        

        # Take only those stars which are inside the FOV
        filtered_indices, filtered_catalog_stars = subsetCatalog(catalog_stars, ra_centre, dec_centre, \
            jd, self.platepar.lat, self.platepar.lon, fov_radius, self.cat_lim_mag)


        return filtered_indices, np.array(filtered_catalog_stars)


    def getInitialParamsAstrometryNet(self, upload_image=True):
        """ Get the estimate of the initial astrometric parameters using astromety.net. """

        fail = False
        solution = None

        # Construct FOV width estimate
        fov_w_range = [0.75*self.config.fov_w, 1.25*self.config.fov_w]


        # Check if the given FF files is in the calstars list
        if (self.img_handle.name() in self.calstars) and (not upload_image):

            # Get the stars detected on this FF file
            star_data = self.calstars[self.img_handle.name()]

            # Make sure that there are at least 10 stars
            if len(star_data) < 10:
                print('Less than 10 stars on the image!')
                fail = True

            else:

                # Get star coordinates
                y_data, x_data, _, _ = np.array(star_data).T

                # Get astrometry.net solution, pass the FOV width estimate
                solution = novaAstrometryNetSolve(x_data=x_data, y_data=y_data, fov_w_range=fov_w_range)

        else:
            fail = True


        # Try finding the soluting by uploading the whole image
        if fail or upload_image:

            print("Uploading the whole image to astrometry.net...")

            # If the image is 16bit or larger, rescale and convert it to 8 bit
            if self.img_data_raw.itemsize*8 > 8:

                # Rescale the image to 8bit
                img_data = np.copy(self.img_data_processed)
                img_data -= np.min(img_data)
                img_data = 255*(img_data/np.max(img_data))
                img_data = img_data.astype(np.uint8)

            else:
                img_data = self.img_data_raw

            solution = novaAstrometryNetSolve(img=img_data, fov_w_range=fov_w_range)



        if solution is None:
            messagebox.showerror(title='Astrometry.net error', \
                message='Astrometry.net failed to find a solution!')

            return None
            


        # Extract the parameters
        ra, dec, orientation, scale, fov_w, fov_h = solution

        jd = date2JD(*self.img_handle.currentTime())

        # Compute the position angle from the orientation
        pos_angle_ref = rotationWrtStandardToPosAngle(self.platepar, orientation)

        # Compute reference azimuth and altitude
        azim, alt = raDec2AltAz(ra, dec, jd, self.platepar.lat, self.platepar.lon)

        # Set parameters to platepar
        self.platepar.pos_angle_ref = pos_angle_ref
        self.platepar.F_scale = scale
        self.platepar.az_centre = azim
        self.platepar.alt_centre = alt

        self.updateRefRADec(skip_rot_update=True)
        
        # Save the current rotation w.r.t horizon value
        self.platepar.rotation_from_horiz = rotationWrtHorizon(self.platepar)

        # Print estimated parameters
        print()
        print('Astrometry.net solution:')
        print('------------------------')
        print(' RA    = {:.2f} deg'.format(ra))
        print(' Dec   = {:.2f} deg'.format(dec))
        print(' Azim  = {:.2f} deg'.format(self.platepar.az_centre))
        print(' Alt   = {:.2f} deg'.format(self.platepar.alt_centre))
        print(' Rot horiz   = {:.2f} deg'.format(self.platepar.rotation_from_horiz))
        print(' Orient eq   = {:.2f} deg'.format(orientation))
        print(' Pos angle   = {:.2f} deg'.format(pos_angle_ref))
        print(' Scale = {:.2f} arcmin/px'.format(60/self.platepar.F_scale))



    def getFOVcentre(self):
        """ Asks the user to input the centre of the FOV in altitude and azimuth. """

        # Get FOV centre
        root = tkinter.Tk()
        root.withdraw()
        d = FOVinputDialog(root)
        root.wait_window(d.top)
        self.azim_centre, self.alt_centre, rot_horizontal = d.getAltAz()

        root.destroy()

        # Get the middle time of the first FF
        img_time = self.img_handle.currentTime()

        # Set the reference platepar time to the time of the FF
        self.platepar.JD = date2JD(*img_time, UT_corr=float(self.platepar.UT_corr))

        # Set the reference hour angle
        self.platepar.Ho = JD2HourAngle(self.platepar.JD)%360

        # Convert FOV centre to RA, Dec
        ra, dec = altAz2RADec(self.azim_centre, self.alt_centre, date2JD(*img_time), \
            self.platepar.lat, self.platepar.lon)


        return ra, dec, rot_horizontal



    def loadPlatepar(self):
        """ Open a file dialog and ask user to open the platepar file. """

        platepar = Platepar()


        # Check if platepar exists in the folder, and set it as the default file name if it does
        if self.config.platepar_name in os.listdir(self.dir_path):
            initialfile = self.config.platepar_name
        else:
            initialfile = ''

        # Load the platepar file
        platepar_file = openFileDialog(self.dir_path, initialfile, 'Select the platepar file', matplotlib)

        if not platepar_file:
            return False, platepar

        # Parse the platepar file
        try:
            self.platepar_fmt = platepar.read(platepar_file, use_flat=self.config.use_flat)
            pp_status = True

        except Exception as e:
            print('Loading platepar failed with error:' + repr(e))
            print(*traceback.format_exception(*sys.exc_info()))

            pp_status = False

        # Check if the platepar was successfuly loaded
        if not pp_status:
            messagebox.showerror(title='Platepar file error', message='The file you selected could not be loaded as a platepar file!')
            
            platepar_file, platepar = self.loadPlatepar()

        
        # Set geo location and gamma from config, if they were updated
        if platepar is not None:
            
            # Update the location from the config file
            platepar.lat = self.config.latitude
            platepar.lon = self.config.longitude
            platepar.elev = self.config.elevation

            # Set the camera gamma from the config file
            platepar.gamma = self.config.gamma

            # Set station ID
            platepar.station_code = self.config.stationID

            # Compute the rotation w.r.t. horizon
            platepar.rotation_from_horiz = rotationWrtHorizon(platepar)


        self.first_platepar_fit = False

        return platepar_file, platepar



    def makeNewPlatepar(self, update_image=True):
        """ Make a new platepar from the loaded one, but set the parameters from the config file. """

        # Update the reference time
        img_time = self.img_handle.currentTime()
        self.platepar.JD = date2JD(*img_time)

        # Update the location from the config file
        self.platepar.lat = self.config.latitude
        self.platepar.lon = self.config.longitude
        self.platepar.elev = self.config.elevation

        # Update image resolution from config
        self.platepar.X_res = self.config.width
        self.platepar.Y_res = self.config.height

        # Set the camera gamma from the config file
        self.platepar.gamma = self.config.gamma

        # Estimate the scale
        scale_x = self.config.fov_w/self.config.width
        scale_y = self.config.fov_h/self.config.height
        self.platepar.F_scale = 1/((scale_x + scale_y)/2)

        # Set distortion polynomials to zero
        self.platepar.x_poly_fwd *= 0
        self.platepar.x_poly_rev *= 0
        self.platepar.y_poly_fwd *= 0
        self.platepar.y_poly_rev *= 0

        # Set the first coeffs to 0.5, as that is the real centre of the FOV
        self.platepar.x_poly_fwd[0] = 0.5
        self.platepar.x_poly_rev[0] = 0.5
        self.platepar.y_poly_fwd[0] = 0.5
        self.platepar.y_poly_rev[0] = 0.5

        # Set station ID
        self.platepar.station_code = self.config.stationID


        # Get reference RA, Dec of the image centre
        self.platepar.RA_d, self.platepar.dec_d, self.platepar.rotation_from_horiz = self.getFOVcentre()

        # Recalculate reference alt/az
        self.platepar.az_centre, self.platepar.alt_centre = raDec2AltAz(self.platepar.JD, \
            self.platepar.lon, self.platepar.lat, self.platepar.RA_d, self.platepar.dec_d)

        # Check that the calibration parameters are within the nominal range
        self.checkParamRange()

        # Compute the position angle
        self.platepar.pos_angle_ref = rotationWrtHorizonToPosAngle(self.platepar, \
            self.platepar.rotation_from_horiz)

        self.platepar.auto_check_fit_refined = False
        self.platepar.auto_recalibrated = False

        # Indicate that this is the first fit of the platepar
        self.first_platepar_fit = True

        # Reset paired stars
        self.paired_stars = []
        self.residuals = None

        # Indicate that a new platepar is being made
        self.new_platepar = True

        if update_image:
            self.updateImage()


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

        print(flat_file)

        try:
            # Load the flat, byteswap the flat if vid file is used or UWO png
            flat = Image.loadFlat(*os.path.split(flat_file), dtype=self.img_data_raw.dtype, \
                byteswap=self.img_handle.byteswap)
        except:
            messagebox.showerror(title='Flat field file error', \
                message='Flat could not be loaded!')
            return False, None


        # Check if the size of the file matches
        if self.img_data_raw.shape != flat.flat_img.shape:
            messagebox.showerror(title='Flat field file error', \
                message='The size of the flat field does not match the size of the image!')

            flat = None

        # Check if the flat field was successfuly loaded
        if flat is None:
            messagebox.showerror(title='Flat field file error', \
                message='The file you selected could not be loaded as a flat field!')

        

        return flat_file, flat



    def loadDark(self):
        """ Open a file dialog and ask user to load a dark frame. """

        dark_file = openFileDialog(self.dir_path, None, 'Select the dark frame file', matplotlib)

        if not dark_file:
            return False, None

        print(dark_file)


        try:

            # Load the dark
            dark = Image.loadDark(*os.path.split(dark_file), dtype=self.img_data_raw.dtype, 
                byteswap=self.img_handle.byteswap)

        except:
            messagebox.showerror(title='Dark frame error', \
                message='Dark frame could not be loaded!')

            return False, None

        dark = dark.astype(self.img_data_raw.dtype)



        # Check if the size of the file matches
        if self.img_data_raw.shape != dark.shape:
            messagebox.showerror(title='Dark field file error', \
                message='The size of the dark frame does not match the size of the image!')

            dark = None

        # Check if the dark frame was successfuly loaded
        if dark is None:
            messagebox.showerror(title='Dark field file error', \
                message='The file you selected could not be loaded as a dark field!')

        

        return dark_file, dark



    def nextImg(self):
        """ Shows the next FF file in the list. """

        # Don't allow image change while in star picking mode
        if self.star_pick_mode:
            messagebox.showwarning(title='Star picking mode', message='You cannot cycle through images while in star picking mode!')
            return

            
        self.img_handle.nextChunk()

        # Reset paired stars
        self.paired_stars = []
        self.residuals = None

        self.updateImage()



    def prevImg(self):
        """ Shows the previous FF file in the list. """

        # Don't allow image change while in star picking mode
        if self.star_pick_mode:
            messagebox.showwarning(title='Star picking mode', message='You cannot cycle through images while in star picking mode!')
            return


        self.img_handle.prevChunk()

        # Reset paired stars
        self.paired_stars = []
        self.residuals = None

        self.updateImage()



    def enableStarPicking(self):
        """ Enable the star picking mode where the star are manually selected for the fit. """

        self.star_pick_mode = True

        self.updateImage()




    def disableStarPicking(self):
        """ Disable the star picking mode where the star are manually selected for the fit. """

        self.star_pick_mode = False

        self.updateImage()




    def centroidStar(self, prev_x_cent=None, prev_y_cent=None):
        """ Find the centroid of the star clicked on the image. """


        # If the centroid from the previous iteration is given, use that as the centre
        if (prev_x_cent is not None) and (prev_y_cent is not None):
            mouse_x = prev_x_cent
            mouse_y = prev_y_cent

        else:
            mouse_x = self.mouse_x_press
            mouse_y = self.mouse_y_press


        # Check if the mouse was pressed outside the FOV
        if mouse_x is None:
            return None, None, None

        ### Extract part of image around the mouse cursor ###
        ######################################################################################################

        # Outer circle radius
        outer_radius = self.star_aperature_radius*2

        x_min = int(round(mouse_x - outer_radius))
        if x_min < 0: x_min = 0

        x_max = int(round(mouse_x + outer_radius))
        if x_max > self.current_ff.ncols - 1:
            x_max = self.current_ff.ncols - 1

        y_min = int(round(mouse_y - outer_radius))
        if y_min < 0: y_min = 0

        y_max = int(round(mouse_y + outer_radius))
        if y_max > self.current_ff.nrows - 1:
            y_max = self.current_ff.nrows - 1


        # Crop the image
        img_crop = self.img_data_raw[y_min:y_max, x_min:x_max]

        # Perform gamma correction
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
                if (pix_dist <= outer_radius) and (pix_dist > self.star_aperature_radius):
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
                if pix_dist <= self.star_aperature_radius:
                    x_acc += j*(img_crop[i, j] - bg_intensity)
                    y_acc += i*(img_crop[i, j] - bg_intensity)
                    intens_acc += img_crop[i, j] - bg_intensity


        x_centroid = x_acc/intens_acc + x_min
        y_centroid = y_acc/intens_acc + y_min

        ######################################################################################################

        return x_centroid, y_centroid, intens_acc



    def findClosestCatalogStarIndex(self, pos_x, pos_y):
        """ Finds the index of the closest catalog star on the image to the given image position. """

        min_index = 0
        min_dist = np.inf

        # Find the index of the closest catalog star to the given image coordinates
        for i, (x, y) in enumerate(zip(self.catalog_x, self.catalog_y)):
            
            dist = (pos_x - x)**2 + (pos_y - y)**2

            if dist < min_dist:
                min_dist = dist
                min_index = i

        return min_index


    def findClosestPickedStarIndex(self, pos_x, pos_y):
        """ Finds the index of the closest picked star on the image to the given image position. """

        min_index = 0
        min_dist = np.inf

        picked_x = [star[0][0] for star in self.paired_stars]
        picked_y = [star[0][1] for star in self.paired_stars]

        # Find the index of the closest catalog star to the given image coordinates
        for i, (x, y) in enumerate(zip(picked_x, picked_y)):
            
            dist = (pos_x - x)**2 + (pos_y - y)**2

            if dist < min_dist:
                min_dist = dist
                min_index = i

        return min_index



    def fitPickedStars(self, first_platepar_fit=False):
        """ Fit stars that are manually picked. The function first only estimates the astrometry parameters
            without the distortion, then just the distortion parameters, then all together.

        Keyword arguments:
            first_platepar_fit: [bool] First fit of the platepar with initial values.

        """

        # Fit the astrometry parameters, at least 5 stars are needed
        if len(self.paired_stars) < 4:
            messagebox.showwarning(title='Number of stars', message="At least 5 paired stars are needed to do the fit!")

            return self.platepar


        print()
        print("----------------------------------------")
        print("Fitting platepar...")


        # Extract paired catalog stars and image coordinates separately
        catalog_stars = np.array([cat_coords for img_coords, cat_coords in self.paired_stars])
        img_stars = np.array([img_coords for img_coords, cat_coords in self.paired_stars])


        # Get the Julian date of the image that's being fit
        jd = date2JD(*self.img_handle.currentTime())


        # Fit the platepar to paired stars
        self.platepar.fitAstrometry(jd, img_stars, catalog_stars, first_platepar_fit=first_platepar_fit)


        # Show platepar parameters
        print()
        print(self.platepar)


        ### Calculate the fit residuals for every fitted star ###
        
        # Get image coordinates of catalog stars
        catalog_x, catalog_y, catalog_mag = getCatalogStarsImagePositions(catalog_stars, jd, self.platepar)


        ## Compute standard coordinates ##
        
        # Platepar with no distortion
        pp_nodist = copy.deepcopy(self.platepar)
        pp_nodist.x_poly_rev *= 0
        pp_nodist.y_poly_rev *= 0

        standard_x, standard_y, _ = getCatalogStarsImagePositions(catalog_stars, jd, pp_nodist)

        ## ##




        residuals = []

        print()
        print('Residuals')
        print('----------')
        print(' No,   Img X,   Img Y, RA (deg), Dec (deg),    Mag, -2.5*LSP,    Cat X,   Cat Y,    Std X,   Std Y, Err amin,  Err px, Direction')

        # Calculate the distance and the angle between each pair of image positions and catalog predictions
        for star_no, (cat_x, cat_y, std_x, std_y, cat_coords, img_c) in enumerate(zip(catalog_x, catalog_y, \
            standard_x, standard_y, catalog_stars, img_stars)):
            
            img_x, img_y, sum_intens = img_c
            ra, dec, mag = cat_coords

            delta_x = cat_x - img_x
            delta_y = cat_y - img_y

            # Compute image residual and angle of the error
            angle = np.arctan2(delta_y, delta_x)
            distance = np.sqrt(delta_x**2 + delta_y**2)


            # Compute the residuals in ra/dec in angular coordiniates
            img_time = self.img_handle.currentTime()
            _, ra_img, dec_img, _ = xyToRaDecPP([img_time], [img_x], [img_y], [1], self.platepar)

            ra_img = ra_img[0]
            dec_img = dec_img[0]

            # Compute the angular distance in degrees
            angular_distance = np.degrees(angularSeparation(np.radians(ra), np.radians(dec), \
                np.radians(ra_img), np.radians(dec_img)))


            residuals.append([img_x, img_y, angle, distance, angular_distance])


            # Print out the residuals
            print('{:3d}, {:7.2f}, {:7.2f}, {:>8.3f}, {:>+9.3f}, {:+6.2f},  {:7.2f}, {:8.2f}, {:7.2f}, {:8.2f}, {:7.2f}, {:8.2f}, {:7.2f}, {:+9.1f}'.format(star_no + 1, img_x, img_y, \
                ra, dec, mag, -2.5*np.log10(sum_intens), cat_x, cat_y, std_x, std_y, 60*angular_distance, \
                distance, np.degrees(angle)))


        mean_angular_error = 60*np.mean([entry[4] for entry in residuals])

        # If the average angular error is larger than 60 arc minutes, report it in degrees
        if mean_angular_error > 60:
            mean_angular_error /= 60
            angular_error_label = 'deg'
        
        else:
            angular_error_label = 'arcmin'


        print('Average error: {:.2f} px, {:.2f} {:s}'.format(np.mean([entry[3] for entry in residuals]), \
            mean_angular_error, angular_error_label))

        # Print the field of view size
        print("FOV: {:.2f} x {:.2f} deg".format(*computeFOVSize(self.platepar)))


        ####################

        # Save the residuals
        self.residuals = residuals

        self.updateImage()

        self.drawFitResiduals()



    def drawFitResiduals(self):
        """ Draw fit residuals. """


        if self.residuals is not None:

            # Plot the residuals
            res_scale = 100
            for entry in self.residuals:

                img_x, img_y, angle, distance, angular_distance = entry

                # Calculate coordinates of the end of the residual line
                res_x = img_x + res_scale*np.cos(angle)*distance
                res_y = img_y + res_scale*np.sin(angle)*distance

                # Plot the image residuals
                self.ax.plot([img_x, res_x], [img_y, res_y], color='orange', alpha=0.25)

                
                # Convert the angular distance from degrees to equivalent image pixels
                ang_dist_img = angular_distance*self.platepar.F_scale
                res_x = img_x + res_scale*np.cos(angle)*ang_dist_img
                res_y = img_y + res_scale*np.sin(angle)*ang_dist_img
                
                # Plot the sky residuals
                self.ax.plot([img_x, res_x], [img_y, res_y], color='yellow', alpha=0.25, linestyle='dashed')


            self.fig.canvas.draw_idle()



    def showAstrometryFitPlots(self):
        """ Show window with astrometry fit details. """


        # Extract paired catalog stars and image coordinates separately
        catalog_stars = np.array([cat_coords for img_coords, cat_coords in self.paired_stars])
        img_stars = np.array([img_coords for img_coords, cat_coords in self.paired_stars])

        # Get the Julian date of the image that's being fit
        jd = date2JD(*self.img_handle.currentTime())


        ### Calculate the fit residuals for every fitted star ###
        
        # Get image coordinates of catalog stars
        catalog_x, catalog_y, catalog_mag = getCatalogStarsImagePositions(catalog_stars, jd, self.platepar)

        # Azimuth and elevation residuals
        x_list = []
        y_list = []
        radius_list = []
        skyradius_list = []
        azim_list = []
        elev_list = []
        azim_residuals = []
        elev_residuals = []
        x_residuals = []
        y_residuals = []
        radius_residuals = []
        skyradius_residuals = []


        # Get image time and Julian date
        img_time = self.img_handle.currentTime()
        jd = date2JD(*img_time)

        # Get RA/Dec of the FOV centre
        ra_centre, dec_centre = self.computeCentreRADec()

        # Calculate the distance and the angle between each pair of image positions and catalog predictions
        for star_no, (cat_x, cat_y, cat_coords, img_c) in enumerate(zip(catalog_x, catalog_y, catalog_stars, \
            img_stars)):
                
            # Compute image coordinates
            img_x, img_y, _ = img_c
            img_radius = np.hypot(img_x - self.platepar.X_res/2, img_y - self.platepar.Y_res/2)

            # Compute sky coordinates
            cat_ra, cat_dec, _ = cat_coords
            cat_ang_separation = np.degrees(angularSeparation(np.radians(cat_ra), np.radians(cat_dec), \
                np.radians(ra_centre), np.radians(dec_centre)))


            # Compute RA/Dec from image
            _, img_ra, img_dec, _ = xyToRaDecPP([img_time], [img_x], [img_y], [1], self.platepar)
            img_ra = img_ra[0]
            img_dec = img_dec[0]



            x_list.append(img_x)
            y_list.append(img_y)
            radius_list.append(img_radius)
            skyradius_list.append(cat_ang_separation)


            # Compute image residuals
            x_residuals.append(cat_x - img_x) 
            y_residuals.append(cat_y - img_y)
            radius_residuals.append(np.hypot(cat_x - self.platepar.X_res/2, cat_y - self.platepar.Y_res/2) \
                - img_radius)


            # Compute sky residuals
            img_ang_separation = np.degrees(angularSeparation(np.radians(img_ra), np.radians(img_dec), \
                np.radians(ra_centre), np.radians(dec_centre)))
            skyradius_residuals.append(cat_ang_separation - img_ang_separation)


            # # Correct the catalog RA/Dec for refraction
            # if self.platepar.refraction:
            #     cat_ra, cat_dec = eqRefractionTrueToApparent(np.radians(cat_ra), np.radians(cat_dec), jd, \
            #         np.radians(self.platepar.lat), np.radians(self.platepar.lon))
            #     cat_ra, cat_dec = np.degrees(cat_ra), np.degrees(cat_dec)


            # Compute azim/elev from the catalog
            azim_cat, elev_cat = raDec2AltAz(cat_ra, cat_dec, jd, self.platepar.lat, self.platepar.lon)

            azim_list.append(azim_cat)
            elev_list.append(elev_cat)


            # Compute azim/elev from image coordinates
            azim_img, elev_img = raDec2AltAz(img_ra, img_dec, jd, self.platepar.lat, self.platepar.lon)

            # Compute azim/elev residuals
            azim_residuals.append(((azim_cat - azim_img + 180)%360 - 180)*np.cos(np.radians(elev_cat)))
            elev_residuals.append(elev_cat - elev_img)


        
        # Init astrometry fit window
        fig_a, ( \
            (ax_azim, ax_elev, ax_skyradius), \
            (ax_x, ax_y, ax_radius) \
            ) = plt.subplots(ncols=3, nrows=2, facecolor=None, figsize=(12, 6))

        # Set figure title
        fig_a.canvas.set_window_title("Astrometry fit")


        # Plot azimuth vs azimuth error
        ax_azim.scatter(azim_list, 60*np.array(azim_residuals), s=2, c='k', zorder=3)

        ax_azim.grid()
        ax_azim.set_xlabel("Azimuth (deg, +E of due N)")
        ax_azim.set_ylabel("Azimuth error (arcmin)")


        # Plot elevation vs elevation error
        ax_elev.scatter(elev_list, 60*np.array(elev_residuals), s=2, c='k', zorder=3)

        ax_elev.grid()
        ax_elev.set_xlabel("Elevation (deg)")
        ax_elev.set_ylabel("Elevation error (arcmin)")

        # If the FOV is larger than 45 deg, set maximum limits on azimuth and elevation
        if np.hypot(*computeFOVSize(self.platepar)) > 45:
            ax_azim.set_xlim([0, 360])
            ax_elev.set_xlim([0, 90])


        # Plot sky radius vs radius error
        ax_skyradius.scatter(skyradius_list, 60*np.array(skyradius_residuals), s=2, c='k', zorder=3)

        ax_skyradius.grid()
        ax_skyradius.set_xlabel("Radius from centre (deg)")
        ax_skyradius.set_ylabel("Radius error (arcmin)")
        ax_skyradius.set_xlim([0, np.hypot(*computeFOVSize(self.platepar))/2])


        # Equalize Y limits, make them multiples of 5 arcmin, and set a minimum range of 5 arcmin
        azim_max_ylim = np.max(np.abs(ax_azim.get_ylim()))
        elev_max_ylim = np.max(np.abs(ax_elev.get_ylim()))
        skyradius_max_ylim = np.max(np.abs(ax_skyradius.get_ylim()))
        max_ylim = np.ceil(np.max([azim_max_ylim, elev_max_ylim, skyradius_max_ylim])/5)*5
        if max_ylim < 5.0:
            max_ylim = 5.0
        ax_azim.set_ylim([-max_ylim, max_ylim])
        ax_elev.set_ylim([-max_ylim, max_ylim])
        ax_skyradius.set_ylim([-max_ylim, max_ylim])



        # Plot X vs X error
        ax_x.scatter(x_list, x_residuals, s=2, c='k', zorder=3)

        ax_x.grid()
        ax_x.set_xlabel("X (px)")
        ax_x.set_ylabel("X error (px)")
        ax_x.set_xlim([0, self.img_data_raw.shape[1]])


        # Plot Y vs Y error
        ax_y.scatter(y_list, y_residuals, s=2, c='k', zorder=3)

        ax_y.grid()
        ax_y.set_xlabel("Y (px)")
        ax_y.set_ylabel("Y error (px)")
        ax_y.set_xlim([0, self.img_data_raw.shape[0]])


        # Plot radius vs radius error
        ax_radius.scatter(radius_list, radius_residuals, s=2, c='k', zorder=3)

        ax_radius.grid()
        ax_radius.set_xlabel("Radius (px)")
        ax_radius.set_ylabel("Radius error (px)")
        ax_radius.set_xlim([0, np.hypot(self.img_data_raw.shape[0]/2, self.img_data_raw.shape[1]/2)])



        # Equalize Y limits, make them integers, and set a minimum range of 1 px
        x_max_ylim = np.max(np.abs(ax_x.get_ylim()))
        y_max_ylim = np.max(np.abs(ax_y.get_ylim()))
        radius_max_ylim = np.max(np.abs(ax_radius.get_ylim()))
        max_ylim = np.ceil(np.max([x_max_ylim, y_max_ylim, radius_max_ylim]))
        if max_ylim < 1:
            max_ylim = 1.0
        ax_x.set_ylim([-max_ylim, max_ylim])
        ax_y.set_ylim([-max_ylim, max_ylim])
        ax_radius.set_ylim([-max_ylim, max_ylim])

        
        fig_a.tight_layout()
        fig_a.show()





if __name__ == '__main__':


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Tool for fitting astrometry plates and photometric calibration.")

    arg_parser.add_argument('dir_path', nargs=1, metavar='DIR_PATH', type=str, \
        help='Path to the folder with FF or image files, path to a video file, or to a state file. If images or videos are given, their names must be in the format: YYYYMMDD_hhmmss.uuuuuu')

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
        help="Path to a config file which will be used instead of the default one. To load the .config file in the given data directory, write '.' (dot).")

    arg_parser.add_argument('-t', '--timebeg', nargs=1, metavar='TIME', type=str, \
        help="The beginning time of the video file in the YYYYMMDD_hhmmss.uuuuuu format.")

    arg_parser.add_argument('-f', '--fps', metavar='FPS', type=float, \
        help="Frames per second when images are used. If not given, it will be read from the config file.")

    arg_parser.add_argument('-g', '--gamma', metavar='CAMERA_GAMMA', type=float, \
        help="Camera gamma value. Science grade cameras have 1.0, consumer grade cameras have 0.45. Adjusting this is essential for good photometry, and doing star photometry through SkyFit can reveal the real camera gamma.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################


    # If the state file was given, load the state
    if cml_args.dir_path[0].endswith('.state'):

        dir_path, state_name = os.path.split(cml_args.dir_path[0])

        # Load the SkyFit object from a state file
        plate_tool = loadPickle(dir_path, state_name)


        # Check if there are missing attributes
        if not hasattr(plate_tool, "invert_levels"):
            plate_tool.invert_levels = False

        if plate_tool.platepar is not None:
            if not hasattr(plate_tool.platepar, "equal_aspect"):
                plate_tool.platepar.equal_aspect = False

        if plate_tool.platepar is not None:
            if not hasattr(plate_tool.platepar, "extinction_scale"):
                plate_tool.platepar.extinction_scale = 1.0
            

        # Set the dir path in case it changed
        plate_tool.dir_path = dir_path

        # Init SkyFit
        plate_tool.updateImage(first_update=True)
        plate_tool.registerEventHandling()

        # Update image handle path
        if plate_tool.img_handle is not None:
            plate_tool.img_handle.dir_path = dir_path

        # Update platepar path
        if plate_tool.platepar_file is not None:
            plate_tool.platepar_file = os.path.join(dir_path, os.path.basename(plate_tool.platepar_file))



    else:

        # Extract the data directory path
        dir_path = cml_args.dir_path[0].replace('"', '')


        # Load the config file
        config = cr.loadConfigFromDirectory(cml_args.config, cml_args.dir_path)


        # Parse the beginning time into a datetime object
        if cml_args.timebeg is not None:

            beginning_time = datetime.datetime.strptime(cml_args.timebeg[0], "%Y%m%d_%H%M%S.%f")

        else:
            beginning_time = None

        # Init the plate tool instance
        plate_tool = PlateTool(dir_path, config, beginning_time=beginning_time, 
            fps=cml_args.fps, gamma=cml_args.gamma)



    plt.tight_layout()
    plt.show()