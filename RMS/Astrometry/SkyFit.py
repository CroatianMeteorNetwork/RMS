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

from RMS.Astrometry.ApplyAstrometry import altAzToRADec, xyToRaDecPP, raDec2AltAz, raDecToXY,\
    rotationWrtHorizon, rotationWrtHorizonToPosAngle, computeFOVSize, photomLine, photometryFit, \
    rotationWrtStandard, rotationWrtStandardToPosAngle
from RMS.Astrometry.AstrometryNetNova import novaAstrometryNetSolve
from RMS.Astrometry.Conversions import date2JD, jd2Date, JD2HourAngle
from RMS.Astrometry.FFTalign import alignPlatepar
import RMS.ConfigReader as cr
import RMS.Formats.CALSTARS as CALSTARS
from RMS.Formats.Platepar import Platepar
from RMS.Formats.FrameInterface import detectInputType
from RMS.Formats import StarCatalog
from RMS.Pickling import loadPickle, savePickle
from RMS.Routines import Image
from RMS.Math import angularSeparation
from RMS.Misc import decimalDegreesToSexHours, openFileDialog

# Import Cython functions
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
from RMS.Astrometry.CyFunctions import subsetCatalog



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
        self.img_level_min = 0
        self.img_level_max = 2**self.bit_depth - 1

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


        ### INIT IMAGE ###

        plt.figure(facecolor='black')

        # Init the first image
        self.updateImage(first_update=True)

        self.ax = plt.gca()


        # Register keys with matplotlib
        self.registerEventHandling()



    def registerEventHandling(self):
        """ Register mouse button and key pressess with matplotlib. """


        plt.gcf().canvas.set_window_title('SkyFit')

        # Set the bacground color to black
        #matplotlib.rcParams['axes.facecolor'] = 'k'

        # Disable standard matplotlib keyboard shortcuts
        plt.rcParams['keymap.save'] = ''
        plt.rcParams['keymap.fullscreen'] = ''
        plt.rcParams['keymap.all_axes'] = ''
        plt.rcParams['keymap.quit'] = ''
        plt.rcParams['keymap.pan'] = ''


        
        self.ax.figure.canvas.mpl_connect('button_press_event', self.onMousePress)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.onMouseRelease)
        
        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.onMouseMotion)

        # Register which mouse/keyboard events will evoke which function
        self.ax.figure.canvas.mpl_connect('scroll_event', self.onScroll)
        self.scroll_counter = 0

        self.ax.figure.canvas.mpl_connect('key_press_event', self.onKeyPress)



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

                    # Perform centroiding with 2 iterations

                    x_cent_tmp, y_cent_tmp, _ = self.centroidStar()

                    # Check that the centroiding was successful
                    if x_cent_tmp is not None:

                        # Centroid the star around the pressed coordinates
                        self.x_centroid, self.y_centroid, \
                            self.star_intensity = self.centroidStar(prev_x_cent=x_cent_tmp, \
                                prev_y_cent=y_cent_tmp)

                        # Draw the centroid on the image
                        plt.scatter(self.x_centroid, self.y_centroid, marker='+', c='y', s=100, lw=3, alpha=0.5)

                        # Select the closest catalog star to the centroid as the first guess
                        self.closest_cat_star_indx = self.findClosestCatalogStarIndex(self.x_centroid, self.y_centroid)

                        # Plot the closest star as a purple cross
                        self.selected_cat_star_scatter = plt.scatter(self.catalog_x[self.closest_cat_star_indx], 
                            self.catalog_y[self.closest_cat_star_indx], marker='+', c='purple', s=100, lw=3)

                        # Update canvas
                        plt.gcf().canvas.draw()

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
                    self.selected_cat_star_scatter = plt.scatter(self.catalog_x[self.closest_cat_star_indx], 
                        self.catalog_y[self.closest_cat_star_indx], marker='+', c='purple', s=50, lw=2)

                    # Update canvas
                    plt.gcf().canvas.draw()


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

            plt.gca().add_patch(self.circle_aperature)
            plt.gca().add_patch(self.circle_aperature_outer)

        # If the catalog star selection mode is on, show a purple circle
        else:

            # Plot a purple circle
            self.circle_aperature = matplotlib.patches.Circle((self.mouse_x, self.mouse_y), 10, 
                edgecolor='purple', fc='none')

            plt.gca().add_patch(self.circle_aperature)


        # Draw the zoom window
        if self.show_zoom_window:
            self.drawZoomWindow()


        plt.gcf().canvas.draw()



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
            if 2*zoom_factor*window_radius > np.min([plt.gcf().bbox.ymax, plt.gcf().bbox.xmax])/2:

                # Compute a new zoom factor
                zoom_factor = np.floor((np.min([plt.gcf().bbox.ymax, \
                    plt.gcf().bbox.xmax])/2)/(2*window_radius))

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
                yo = plt.gcf().bbox.ymax - zoom_factor*2*window_radius
                zoom_window_pos += 'S'

            if self.mouse_x > img_w_half:
                xo = 0
                zoom_window_pos += 'W'
            else:
                xo = plt.gcf().bbox.xmax - zoom_factor*2*window_radius
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
            plt.gcf().figimage(img_crop, xo=xo, yo=yo, zorder=5, cmap='gray', \
                vmin=np.min(self.img_data_processed), vmax=np.max(self.img_data_processed))



    def checkParamRange(self):
        """ Checks that the astrometry parameters are within the allowed range. """

        # Right ascension should be within 0-360
        self.platepar.RA_d = self.platepar.RA_d%360

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
            logsum_px = []
            catalog_mags = []
            for paired_star in self.paired_stars:

                img_star, catalog_star = paired_star

                star_x, star_y, px_intens = img_star
                _, _, star_mag = catalog_star

                lsp = np.log10(px_intens)

                # Skip intensities which were not properly calculated
                if np.isnan(lsp) or np.isinf(lsp):
                    continue

                star_coords.append([star_x, star_y])
                logsum_px.append(lsp)
                catalog_mags.append(star_mag)



            # Make sure there are more than 2 stars picked
            if len(logsum_px) > 2:

                # Fit the photometric offset
                photom_offset, fit_stddev, fit_resids = photometryFit(logsum_px, catalog_mags)


                # Set photometry parameters
                self.platepar.mag_0 = -2.5
                self.platepar.mag_lev = photom_offset
                self.platepar.mag_lev_stddev = fit_stddev


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
                        photom_resid_lbl = plt.text(star_x, star_y + 10, photom_resid_txt, \
                            verticalalignment='top', horizontalalignment='center', \
                            fontsize=photom_resid_size, color='w')

                        # Plot the star magnitude
                        star_mag_lbl = plt.text(star_x, star_y - 10, "{:+6.2f}".format(star_mag), \
                            verticalalignment='bottom', horizontalalignment='center', \
                            fontsize=10, color='r')


                        self.photom_deviatons_scat.append([photom_resid_lbl, star_mag_lbl])


                    plt.draw()


                # Show the photometry fit plot
                if show_plot:

                    ### PLOT PHOTOMETRY FIT ###

                    # Init plot for photometry
                    fig_p, (ax_p, ax_r) = plt.subplots(nrows=2, facecolor=None, figsize=(6.4, 7.2), \
                        gridspec_kw={'height_ratios':[2, 1]})

                    # Set photometry window title
                    fig_p.canvas.set_window_title('Photometry')


                    # Plot catalog magnitude vs. logsum of pixel intensities
                    self.photom_points = ax_p.scatter(-2.5*np.array(logsum_px), catalog_mags, s=5, c='r', \
                        zorder=3)

                    x_min, x_max = ax_p.get_xlim()
                    y_min, y_max = ax_p.get_ylim()

                    x_min_w = x_min - 3
                    x_max_w = x_max + 3
                    y_min_w = y_min - 3
                    y_max_w = y_max + 3

                    
                    # Plot fit info
                    fit_info = 'Fit: {:+.2f}LSP {:+.2f} +/- {:.2f} \nGamma = {:.2f}'.format(self.platepar.mag_0, \
                        self.platepar.mag_lev, fit_stddev, self.platepar.gamma)

                    print(fit_info)
                    #ax_p.text(x_min, y_min, fit_info, color='r', verticalalignment='top', horizontalalignment='left', fontsize=10)

                    # Plot the line fit
                    logsum_arr = np.linspace(x_min_w, x_max_w, 10)
                    ax_p.plot(logsum_arr, photomLine(logsum_arr/(-2.5), photom_offset), label=fit_info, \
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


                    img_diagonal = np.hypot(self.config.width/2, self.config.height/2)

                    # Compute radiuses from centre of stars used for the fit
                    rad_list = []
                    for star_x, star_y in zip (star_coords_x, star_coords_y):

                        # Compute radius ratio to corner
                        rad = np.hypot(star_x - self.config.width/2, star_y - self.config.height/2)
                        rad_list.append(rad)

                    
                    # Plot radius from centre vs. fit residual
                    ax_r.scatter(rad_list, fit_resids, s=5, c='r', zorder=3)
                    
                    ax_r.grid()

                    ax_r.set_ylabel("Fit residuals (mag)")
                    ax_r.set_xlabel("Radius from centre (px)")

                    ax_r.set_xlim(0, img_diagonal)

                    
                    fig_p.tight_layout()
                    fig_p.show()


            else:
                print('Need more than 2 stars for photometry plot!')



    def updateRefRADec(self, skip_rot_update=False):
        """ Update the reference RA and Dec from Alt/Az. """

        if not skip_rot_update:
            
            # Save the current rotation w.r.t horizon value
            self.platepar.rotation_from_horiz = rotationWrtHorizon(self.platepar)


        # Compute the datetime object of the reference Julian date
        time_data = [jd2Date(self.platepar.JD, dt_obj=True)]

        # Convert the reference alt/az to reference RA/Dec
        _, ra_data, dec_data = altAzToRADec(self.platepar.lat, self.platepar.lon, self.platepar.UT_corr, 
            time_data, [self.platepar.az_centre], [self.platepar.alt_centre], dt_time=True)

        # Assign the computed RA/Dec to platepar
        self.platepar.RA_d = ra_data[0]
        self.platepar.dec_d = dec_data[0]


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
            
            # Recalculate reference alt/az
            self.platepar.az_centre, self.platepar.alt_centre = raDec2AltAz(self.platepar.JD, \
                self.platepar.lon, self.platepar.lat, self.platepar.RA_d, self.platepar.dec_d)

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
            plt.text(self.img_data_raw.shape[1]/2, self.img_data_raw.shape[0]/2, \
                "Solving with astrometry.net...", color='r', alpha=0.5, fontsize=16, ha='center', va='center')

            plt.draw()
            plt.gcf().canvas.draw()
            plt.gcf().canvas.flush_events()

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


        # Increase image gamma
        elif event.key == 'u':

            # Increase image gamma by a factor of 1.1x
            self.updateGamma(1.1)

        elif event.key == 'j':

            # Decrease image gamma by a factor of 0.9x
            self.updateGamma(0.9)


        elif event.key == 'ctrl+h':

            # Toggle levels adustment mode
            self.adjust_levels_mode = not self.adjust_levels_mode

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

            # If shift was pressed, reset distorsion parameters to zero
            if event.key == "ctrl+Z":
                self.platepar.resetDistorsionParameters()

            # If the first platepar is being made, do the fit twice
            if self.first_platepar_fit:
                self.fitPickedStars()
                self.fitPickedStars()
                self.first_platepar_fit = False

            else:
                # Otherwise, only fit the once
                self.fitPickedStars()


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

        # Load catalog stars
        catalog_stars, self.mag_band_string, self.config.star_catalog_band_ratios = StarCatalog.readStarCatalog(\
            self.config.star_catalog_path, self.config.star_catalog_file, lim_mag=lim_mag, \
            mag_band_ratios=self.config.star_catalog_band_ratios)

        return catalog_stars


    def drawPairedStars(self):
        """ Draws the stars that were picked for calibration. """

        if self.star_pick_mode:

            # Go through all paired stars
            for paired_star in self.paired_stars:

                img_star, catalog_star = paired_star

                x, y, _ = img_star

                # Plot all paired stars
                plt.scatter(x, y, marker='x', c='b', s=100, lw=3, alpha=0.5)



    def drawLevelsAdjustmentHistogram(self, img):
        """ Draw a levels histogram over the image, so the levels can be adjusted. """

        nbins = int((2**self.config.bit_depth)/2)

        # Compute the intensity histogram
        hist, bin_edges = np.histogram(img.flatten(), density=True, range=(0, 2**self.bit_depth), \
            bins=nbins)

        # Scale the maximum histogram peak to half the image height
        hist *= img.shape[0]/np.max(hist)/2

        # Scale the edges to image width
        image_to_level_scale = img.shape[1]/np.max(bin_edges)
        bin_edges *= image_to_level_scale

        ax1 = plt.gca()
        ax2 = ax1.twinx().twiny()

        # Plot the histogram
        ax2.bar(bin_edges[:-1], hist, color='white', alpha=0.5, width=img.shape[1]/nbins, edgecolor='0.5')

        # Plot levels limits
        y_range = np.linspace(0, img.shape[0], 3)
        x_arr = np.zeros_like(y_range)

        ax2.plot(x_arr + self.img_level_min*image_to_level_scale, y_range, color='w')
        ax2.plot(x_arr + self.img_level_max*image_to_level_scale, y_range, color='w')

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

        plt.sca(ax1)





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
            plt.clf()


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
            min_lvl = np.percentile(img_data, 1)
            max_lvl = np.percentile(img_data, 99.9)


            # Adjust levels (auto)
            img_data = Image.adjustLevels(img_data, min_lvl, self.img_gamma, max_lvl, scaleto8bits=True)

        else:
            
            # Adjust levels (manual)
            img_data = Image.adjustLevels(img_data, self.img_level_min, self.img_gamma, self.img_level_max,
                scaleto8bits=True)



        # Draw levels adjustment histogram
        if self.adjust_levels_mode:
            self.drawLevelsAdjustmentHistogram(self.img_data_raw)


        # Store image after levels modifications
        self.img_data_processed = np.copy(img_data)

        # Show the loaded image (defining the exent speeds up image drawimg)
        plt.imshow(img_data, cmap='gray', interpolation='nearest', \
            extent=(0, self.img_data_processed.shape[1], self.img_data_processed.shape[0], 0),
            vmin=0, vmax=255)

        # Draw stars that were paired in picking mode
        self.drawPairedStars()

        # Draw stars detected on this image
        if self.draw_calstars:
            self.drawCalstars()

        # Update centre of FOV in horizontal coordinates
        self.platepar.az_centre, self.platepar.alt_centre = raDec2AltAz(self.platepar.JD, self.platepar.lon, 
            self.platepar.lat, self.platepar.RA_d, self.platepar.dec_d)

        ### Draw catalog stars on the image using the current platepar ###
        ######################################################################################################
        self.catalog_x, self.catalog_y, catalog_mag = self.getCatalogStarsImagePositions(self.catalog_stars, \
            self.platepar.lon, self.platepar.lat, self.platepar.RA_d, self.platepar.dec_d, \
            self.platepar.pos_angle_ref, self.platepar.F_scale, self.platepar.x_poly_rev, \
            self.platepar.y_poly_rev)

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
                plt.scatter(self.catalog_x_filtered, self.catalog_y_filtered, c='r', marker='+', lw=1.0, \
                    alpha=0.5, s=((4.0 + (cat_mag_faintest - catalog_mag_filtered))/2.0)**(2*2.512))

            else:
                print('No catalog stars visible!')

        ######################################################################################################


        # Draw photometry
        if len(self.paired_stars) > 2:
            self.photometry()

        # Draw fit residuals
        if self.residuals is not None:
            self.drawFitResiduals()


        # Set plot limits
        plt.xlim([0, self.current_ff.ncols])
        plt.ylim([self.current_ff.nrows, 0])


        # Compute RA/Dec of the FOV centre
        ra_centre, dec_centre = self.computeCentreRADec()


        # Setup a monospace font
        font = FontProperties()
        font.set_family('monospace')
        font.set_size(8)

        
        if self.show_key_help == 0:
            text_str = 'Show fit parameters - F1'

            plt.gca().text(10, self.current_ff.nrows, text_str, color='w', verticalalignment='bottom', 
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
            text_str += "Scale  = {:.3f}'/px\n".format(60/self.platepar.F_scale)
            text_str += 'Lim mag  = {:.1f}\n'.format(self.cat_lim_mag)
            text_str += 'Increment = {:.2f}\n'.format(self.key_increment)
            text_str += 'Img Gamma = {:.2f}\n'.format(self.img_gamma)
            text_str += 'Camera Gamma = {:.2f}\n'.format(self.config.gamma)
            text_str += '\n'
            sign, hh, mm, ss = decimalDegreesToSexHours(ra_centre)
            if sign < 0:
                sign_str = '-'
            else:
                sign_str = ' '
            text_str += 'RA centre  = {:s}{:02d}h {:02d}m {:05.2f}s\n'.format(sign_str, hh, mm, ss)
            text_str += 'Dec centre = {:.3f}$\\degree$\n'.format(dec_centre)
            plt.gca().text(10, 10, text_str, color='w', verticalalignment='top', horizontalalignment='left', \
                fontproperties=font)


            if self.show_key_help == 1:
                text_str = 'Show keyboard shortcuts - F1'

                plt.gca().text(10, self.current_ff.nrows, text_str, color='w', verticalalignment='bottom', 
                    horizontalalignment='left', fontproperties=font)

        # Show keyboard shortcuts
        if self.show_key_help > 1:

            # Show text on image with instructions
            text_str  = 'Keys:\n'
            text_str += '-----\n'
            text_str += 'A/D - Azimuth\n'
            text_str += 'S/W - Altitude\n'
            text_str += 'Q/E - Position angle\n'
            text_str += 'Up/Down - Scale\n'
            text_str += '1/2 - X offset\n'
            text_str += '3/4 - Y offset\n'
            text_str += '5/6 - X 1st dist. coeff.\n'
            text_str += '7/8 - Y 1st dist. coeff.\n'
            text_str += '\n'
            text_str += ',/. - UT correction\n'
            text_str += 'R/F - Lim mag\n'
            text_str += '+/- - Increment\n'
            text_str += '\n'
            text_str += 'M - Toggle maxpixel/avepixel\n'
            text_str += 'H - Hide/show catalog stars\n'
            text_str += 'C - Hide/show detected stars\n'
            text_str += 'U/J - Img Gamma\n'
            text_str += 'CTRL + H - Adjust levels\n'
            text_str += 'V - FOV centre\n'
            text_str += '\n'
            text_str += 'CTRL + A - Auto levels\n'
            text_str += 'CTRL + D - Load dark\n'
            text_str += 'CTRL + F - Load flat\n'
            text_str += 'CTRL + X - astrometry.net img upload\n'
            text_str += 'CTRL + SHIFT + X - astrometry.net XY\n'
            text_str += 'CTRL + R - Pick stars\n'
            text_str += 'SHIFT + Z - Show zoomed window\n'
            text_str += 'CTRL + N - New platepar\n'
            text_str += 'CTRL + S - Save platepar\n'
            text_str += 'SHIFT + CTRL + S - Save platepar as default\n'

            text_str += '\n'

            text_str += 'Hide on-screen text - F1\n'


            plt.gca().text(10, self.current_ff.nrows - 5, text_str, color='w', verticalalignment='bottom', 
                horizontalalignment='left', fontproperties=font)


        # Show fitting instructions
        if self.star_pick_mode:
            text_str  = "STAR PICKING MODE"

            if self.show_key_help > 0:
                text_str += "\nPRESS 'CTRL + Z' FOR STAR FITTING\n"
                text_str += "PRESS 'CTRL + SHIFT + Z' FOR STAR FITTING WITH INITIAL DISTORSION PARAMETES SET TO 0\n"
                text_str += "PRESS 'P' FOR PHOTOMETRY FIT"

            plt.gca().text(self.current_ff.ncols/2, self.current_ff.nrows, text_str, color='r', 
                verticalalignment='bottom', horizontalalignment='center', fontproperties=font)


        plt.gcf().canvas.draw()



    def drawCalstars(self):
        """ Draw extracted stars on the current image. """

        # Check if the given FF files is in the calstars list
        if self.img_handle.name() in self.calstars:

            # Get the stars detected on this FF file
            star_data = self.calstars[self.img_handle.name()]

            # Get star coordinates
            y, x, _, _ = np.array(star_data).T

            plt.scatter(x, y, edgecolors='g', marker='o', facecolors='none', alpha=0.8, linestyle='dotted')



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


        # Take only those stars which are inside the FOV
        filtered_indices, filtered_catalog_stars = subsetCatalog(catalog_stars, ra_centre, dec_centre, \
            fov_radius, self.cat_lim_mag)


        return filtered_indices, np.array(filtered_catalog_stars)



    def getCatalogStarsImagePositions(self, catalog_stars, lon, lat, ra_ref, dec_ref, pos_angle_ref, \
        F_scale, x_poly_rev, y_poly_rev):
        """ Get image positions of catalog stars using the current platepar values. 
    
        Arguments:
            catalog_stars: [2D list] A list of (ra, dec, mag) pairs of catalog stars.
            lon: [float] Longitude in degrees.
            lat: [float] Latitude in degrees.
            ra_ref: [float] Reference RA of the FOV centre (degrees).
            dec_ref: [float] Reference Dec of the FOV centre (degrees).
            pos_angle_ref: [float] Reference position angle in degrees.
            F_scale: [float] Image scale (px/deg).
            x_poly_rev: [ndarray float] Distorsion polynomial in X direction for reverse mapping.
            y_poly_rev: [ndarray float] Distorsion polynomail in Y direction for reverse mapping.

        Return:
            (x_array, y_array mag_catalog): [tuple of ndarrays] X, Y positons and magnitudes of stars on the 
                image.
        """

        ra_catalog, dec_catalog, mag_catalog = catalog_stars.T

        img_time = self.img_handle.currentTime()

        # Get the date of the middle of the FF exposure
        jd = date2JD(*img_time)

        # Convert star RA, Dec to image coordinates
        x_array, y_array = raDecToXY(ra_catalog, dec_catalog, jd, lat, lon, self.platepar.X_res, \
            self.platepar.Y_res, ra_ref, dec_ref, self.platepar.JD, pos_angle_ref, F_scale, x_poly_rev, \
            y_poly_rev, UT_corr=self.platepar.UT_corr)

        return x_array, y_array, mag_catalog


    def getPairedStarsSkyPositions(self, img_x, img_y, platepar):
        """ Compute RA, Dec of all paired stars on the image given the platepar. 
    
        Arguments:
            img_x: [ndarray] Array of column values of the stars.
            img_y: [ndarray] Array of row values of the stars.
            platepar: [Platepar instance] Platepar object.

        Return:
            (ra_array, dec_array): [tuple of ndarrays] Arrays of RA and Dec of stars on the image.
        """

        # Compute RA, Dec of image stars
        img_time = self.img_handle.currentTime()
        _, ra_array, dec_array, _ = xyToRaDecPP(len(img_x)*[img_time], img_x, img_y, len(img_x)*[1], \
            platepar)

        return ra_array, dec_array


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
        azim, alt = raDec2AltAz(jd, self.platepar.lon, self.platepar.lat, ra, dec)

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

        
        time_data = [img_time]

        # Convert FOV centre to RA, Dec
        _, ra_data, dec_data = altAzToRADec(self.platepar.lat, self.platepar.lon, self.platepar.UT_corr, 
            time_data, [self.azim_centre], [self.alt_centre])


        return ra_data[0], dec_data[0], rot_horizontal



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
            self.platepar_fmt = platepar.read(platepar_file)
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

            # Update image resolution from config
            platepar.X_res = self.config.width
            platepar.Y_res = self.config.height

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

        # Set distorsion polynomials to zero
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



    def fitPickedStars(self):
        """ Fit stars that are manually picked. The function first only estimates the astrometry parameters
            without the distortion, then just the distortion parameters, then all together.

        """

        # Fit the astrometry parameters, at least 5 stars are needed
        if len(self.paired_stars) < 4:
            messagebox.showwarning(title='Number of stars', message="At least 5 paired stars are needed to do the fit!")

            return False


        def _calcImageResidualsAstro(params, self, catalog_stars, img_stars):
            """ Calculates the differences between the stars on the image and catalog stars in image 
                coordinates with the given astrometrical solution. 

            """

            # Extract fitting parameters
            ra_ref, dec_ref, pos_angle_ref, F_scale = params

            img_x, img_y, _ = img_stars.T

            # Get image coordinates of catalog stars
            catalog_x, catalog_y, catalog_mag = self.getCatalogStarsImagePositions(catalog_stars, \
                self.platepar.lon, self.platepar.lat, ra_ref, dec_ref, pos_angle_ref, F_scale, \
                self.platepar.x_poly_rev, self.platepar.y_poly_rev)


            
            # Calculate the sum of squared distances between image stars and catalog stars
            dist_sum = np.sum((catalog_x - img_x)**2 + (catalog_y - img_y)**2)


            return dist_sum


        def _calcSkyResidualsAstro(params, self, catalog_stars, img_stars):
            """ Calculates the differences between the stars on the image and catalog stars in sky 
                coordinates with the given astrometrical solution. 

            """

            # Extract fitting parameters
            ra_ref, dec_ref, pos_angle_ref, F_scale = params

            pp_copy = copy.deepcopy(self.platepar)

            pp_copy.RA_d = ra_ref
            pp_copy.dec_d = dec_ref
            pp_copy.pos_angle_ref = pos_angle_ref
            pp_copy.F_scale = F_scale

            img_x, img_y, _ = img_stars.T

            # Get image coordinates of catalog stars
            ra_array, dec_array = self.getPairedStarsSkyPositions(img_x, img_y, pp_copy)

            ra_catalog, dec_catalog, _ = catalog_stars.T

            # Compute the sum of the angular separation
            separation_sum = np.sum(angularSeparation(np.radians(ra_array), np.radians(dec_array), \
                np.radians(ra_catalog), np.radians(dec_catalog))**2)


            return separation_sum



        def _calcImageResidualsDistorsion(params, self, catalog_stars, img_stars, dimension):
            """ Calculates the differences between the stars on the image and catalog stars in image 
                coordinates with the given astrometrical solution. 

            Arguments:
                ...
                dimension: [str] 'x' for X polynomial fit, 'y' for Y polynomial fit

            """

            if dimension == 'x':
                x_poly_rev = params
                y_poly_rev = np.zeros(12)

            else:
                x_poly_rev = np.zeros(12)
                y_poly_rev = params


            img_x, img_y, _ = img_stars.T

            # Get image coordinates of catalog stars
            catalog_x, catalog_y, catalog_mag = self.getCatalogStarsImagePositions(catalog_stars, \
                self.platepar.lon, self.platepar.lat, self.platepar.RA_d, self.platepar.dec_d, \
                self.platepar.pos_angle_ref, self.platepar.F_scale, x_poly_rev, y_poly_rev)


            # Calculate the sum of squared distances between image stars and catalog stars, per every
            #   dimension
            if dimension == 'x':
                dist_sum = np.sum((catalog_x - img_x)**2)

            else:
                dist_sum = np.sum((catalog_y - img_y)**2)


            return dist_sum


        def _calcSkyResidualsDistorsion(params, self, catalog_stars, img_stars, dimension):
            """ Calculates the differences between the stars on the image and catalog stars in sky 
                coordinates with the given astrometrical solution. 

            Arguments:
                ...
                dimension: [str] 'x' for X polynomial fit, 'y' for Y polynomial fit

            """

            pp_copy = copy.deepcopy(self.platepar)

            if dimension == 'x':
                pp_copy.x_poly_fwd = params

            else:
                pp_copy.y_poly_fwd = params


            img_x, img_y, _ = img_stars.T

            # Get image coordinates of catalog stars
            ra_array, dec_array = self.getPairedStarsSkyPositions(img_x, img_y, pp_copy)

            ra_catalog, dec_catalog, _ = catalog_stars.T

            # Compute the sum of the angular separation
            separation_sum = np.sum(angularSeparation(np.radians(ra_array), np.radians(dec_array), \
                np.radians(ra_catalog), np.radians(dec_catalog))**2)

            return separation_sum



        # Extract paired catalog stars and image coordinates separately
        catalog_stars = np.array([cat_coords for img_coords, cat_coords in self.paired_stars])
        img_stars = np.array([img_coords for img_coords, cat_coords in self.paired_stars])

        # print('ASTRO', _calcImageResidualsAstro([self.platepar.RA_d, self.platepar.dec_d, 
        #     self.platepar.pos_angle_ref, self.platepar.F_scale], self, catalog_stars, img_stars))

        # print('DIS_X', _calcImageResidualsDistorsion(self.platepar.x_poly_rev, self, catalog_stars, \
        #     img_stars, 'x'))

        # print('DIS_Y', _calcImageResidualsDistorsion(self.platepar.y_poly_rev, self, catalog_stars, \
        #     img_stars, 'y'))



        ### ASTROMETRIC PARAMETERS FIT ###

        # Initial parameters for the astrometric fit
        p0 = [self.platepar.RA_d, self.platepar.dec_d, self.platepar.pos_angle_ref, self.platepar.F_scale]

        # Fit the astrometric parameters using the reverse transform for reference        
        res = scipy.optimize.minimize(_calcImageResidualsAstro, p0, args=(self, catalog_stars, img_stars),
            method='Nelder-Mead')

        # # Fit the astrometric parameters using the forward transform for reference
        #   WARNING: USING THIS MAKES THE FIT UNSTABLE
        # res = scipy.optimize.minimize(_calcSkyResidualsAstro, p0, args=(self, catalog_stars, img_stars),
        #     method='Nelder-Mead')

        print(res.x)

        # Update fitted astrometric parameters
        self.platepar.RA_d, self.platepar.dec_d, self.platepar.pos_angle_ref, self.platepar.F_scale = res.x

        # Recalculate centre
        self.platepar.az_centre, self.platepar.alt_centre = raDec2AltAz(self.platepar.JD, self.platepar.lon, 
            self.platepar.lat, self.platepar.RA_d, self.platepar.dec_d)


        # Save the size of the image
        self.platepar.Y_res, self.platepar.X_res = self.current_ff.maxpixel.shape

        ### ###


        ### DISTORSION FIT ###

        # If there are more than 12 paired stars, fit the distortion parameters
        if len(self.paired_stars) > 12:

            ### REVERSE MAPPING FIT ###
            # Fit distorsion parameters in X direction, reverse mapping
            res = scipy.optimize.minimize(_calcImageResidualsDistorsion, self.platepar.x_poly_rev, args=(self, 
                catalog_stars, img_stars, 'x'), method='Nelder-Mead', options={'maxiter': 10000})

            # Exctact fitted X polynomial
            self.platepar.x_poly_rev = res.x

            print(res.x)

            # Fit distorsion parameters in Y direction, reverse mapping
            res = scipy.optimize.minimize(_calcImageResidualsDistorsion, self.platepar.y_poly_rev, args=(self, 
                catalog_stars, img_stars, 'y'), method='Nelder-Mead', options={'maxiter': 10000})

            # Extract fitted Y polynomial
            self.platepar.y_poly_rev = res.x

            print(res.x)

            ### ###

            

            # If this is the first fit of the distorsion, set the forward parametrs to be equal to the reverse
            if self.first_platepar_fit:

                self.platepar.x_poly_fwd = np.array(self.platepar.x_poly_rev)
                self.platepar.y_poly_fwd = np.array(self.platepar.y_poly_rev)

                self.first_platepar_fit = False



            ### FORWARD MAPPING FIT ###

            # Fit distorsion parameters in X direction, forward mapping
            res = scipy.optimize.minimize(_calcSkyResidualsDistorsion, self.platepar.x_poly_fwd, args=(self, 
                catalog_stars, img_stars, 'x'), method='Nelder-Mead', options={'maxiter': 10000})

            # Exctact fitted X polynomial
            self.platepar.x_poly_fwd = res.x

            print(res.x)

            # Fit distorsion parameters in Y direction, forward mapping
            res = scipy.optimize.minimize(_calcSkyResidualsDistorsion, self.platepar.y_poly_fwd, args=(self, 
                catalog_stars, img_stars, 'y'), method='Nelder-Mead', options={'maxiter': 10000})

            # Extract fitted Y polynomial
            self.platepar.y_poly_fwd = res.x

            print(res.x)

            ### ###

        else:
            print('Too few stars to fit the distorsion, only the astrometric parameters where fitted!')


        # Set the list of stars used for the fit to the platepar
        fit_star_list = []
        for img_coords, cat_coords in self.paired_stars:

            # Compute the Julian date of the image
            img_time = self.img_handle.currentTime()
            jd = date2JD(*img_time)

            # Store time, image coordinate x, y, intensity, catalog ra, dec, mag
            fit_star_list.append([jd] + img_coords + cat_coords.tolist())

        self.platepar.star_list = fit_star_list


        # Set the flag to indicate that the platepar was manually fitted
        self.auto_check_fit_refined = False

        ### ###


        ### Calculate the fit residuals for every fitted star ###
        
        # Get image coordinates of catalog stars
        catalog_x, catalog_y, catalog_mag = self.getCatalogStarsImagePositions(catalog_stars, \
            self.platepar.lon, self.platepar.lat, self.platepar.RA_d, self.platepar.dec_d, \
            self.platepar.pos_angle_ref, self.platepar.F_scale, self.platepar.x_poly_rev, \
            self.platepar.y_poly_rev)


        residuals = []

        print()
        print('Residuals')
        print('----------')
        print(' No,   Img X,   Img Y, RA (deg), Dec (deg),    Mag, -2.5*LSP,    Cat X,   Cat Y, Err amin,  Err px, Direction')

        # Calculate the distance and the angle between each pair of image positions and catalog predictions
        for star_no, (cat_x, cat_y, cat_coords, img_c) in enumerate(zip(catalog_x, catalog_y, catalog_stars, \
            img_stars)):
            
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
            print('{:3d}, {:7.2f}, {:7.2f}, {:>8.3f}, {:>+9.3f}, {:+6.2f},  {:7.2f}, {:8.2f}, {:7.2f}, {:8.2f}, {:7.2f}, {:+9.1f}'.format(star_no + 1, img_x, img_y, \
                ra, dec, mag, -2.5*np.log10(sum_intens), cat_x, cat_y, 60*angular_distance, distance, np.degrees(angle)))


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
                plt.plot([img_x, res_x], [img_y, res_y], color='orange', alpha=0.25)

                
                # Convert the angular distance from degrees to equivalent image pixels
                ang_dist_img = angular_distance*self.platepar.F_scale
                res_x = img_x + res_scale*np.cos(angle)*ang_dist_img
                res_y = img_y + res_scale*np.sin(angle)*ang_dist_img
                
                # Plot the sky residuals
                plt.plot([img_x, res_x], [img_y, res_y], color='yellow', alpha=0.25, linestyle='dashed')


            plt.draw()





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

        # Load the manual redicution tool object from a state file
        plate_tool = loadPickle(dir_path, state_name)

        # Set the dir path in case it changed
        plate_tool.dir_path = dir_path

        # Init SkyFit
        plate_tool.updateImage(first_update=True)
        plate_tool.registerEventHandling()


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