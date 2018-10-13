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
import argparse
    
# tkinter import that works on both Python 2 and 3
try:
    import tkinter
    from tkinter import filedialog, messagebox
except:
    import Tkinter as tkinter
    import tkFileDialog as filedialog
    import tkMessageBox as messagebox

import numpy as np

import scipy.optimize

import matplotlib
import matplotlib.pyplot as plt

import RMS.ConfigReader as cr
import RMS.Formats.CALSTARS as CALSTARS
from RMS.Formats.Platepar import Platepar
from RMS.Formats.FFfile import read as readFF
from RMS.Formats.FFfile import validFFName
from RMS.Formats.FFfile import getMiddleTimeFF
from RMS.Formats import StarCatalog
from RMS.Astrometry.ApplyAstrometry import altAz2RADec, XY2CorrectedRADecPP, raDec2AltAz, raDecToCorrectedXY
from RMS.Astrometry.Conversions import date2JD, jd2Date
from RMS.Routines import Image
from RMS.Math import angularSeparation

# Import Cython functions
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
from RMS.Astrometry.CyFunctions import subsetCatalog


class FOVinputDialog(object):
    """ Dialog for inputting FOV centre in Alt/Az. """
    def __init__(self, parent):

        self.parent = parent

        # Set initial angle values
        self.azim = self.alt = 0

        self.top = tkinter.Toplevel(parent)

        # Bind the Enter key to run the verify function
        self.top.bind('<Return>', self.verify)

        tkinter.Label(self.top, text="FOV centre (degrees)").grid(row=0, columnspan=2)

        azim_label = tkinter.Label(self.top, text='Azim = ')
        azim_label.grid(row=1, column=0)
        self.azimuth = tkinter.Entry(self.top)
        self.azimuth.grid(row=1, column=1)
        self.azimuth.focus_set()

        elev_label = tkinter.Label(self.top, text='Alt  =')
        elev_label.grid(row=2, column=0)
        self.altitude = tkinter.Entry(self.top)
        self.altitude.grid(row=2, column=1)

        b = tkinter.Button(self.top, text="OK", command=self.verify)
        b.grid(row=3, columnspan=2)


    def verify(self, event=None):
        """ Check that the azimuth and altitude are withing the bounds. """

        try:
            # Read values
            self.azim = float(self.azimuth.get())
            self.alt  = float(self.altitude.get())

            # Check that the values are within the bounds
            if ((self.azim < 0) and (self.azim > 360)) or ((self.alt < 0) or (self.alt > 90)):
                messagebox.showerror(title='Range error', message='The azimuth or altitude are not within the limits!')
            else:
                self.top.destroy()

        except:
            messagebox.showerror(title='Range error', message='The azimuth or altitude are not within the limits!')


    def getAltAz(self):
        """ Returns inputed FOV centre. """

        return self.azim, self.alt




class PlateTool(object):
    def __init__(self, dir_path, config):
        """ SkyFit interactive window.

        Arguments:
            dir_path: [str] Absolute path to the directory containing image files.
            config: [COnfig struct]

        """

        self.config = config
        self.dir_path = dir_path

        print('Using FF files from:', self.dir_path)

        # Flag which regulates wheter the maxpixel or the avepixel image is shown (avepixel by default)
        self.img_type_flag = 'avepixel'

        # Star picking mode
        self.star_pick_mode = False
        self.star_selection_centroid = True
        self.circle_aperature = None
        self.circle_aperature_outer = None
        self.star_aperature_radius = 5
        self.x_centroid = self.y_centroid = None
        self.photom_deviatons_scat = []

        self.catalog_stars_visible = True

        self.show_key_help = False

        # List of paired image and catalog stars
        self.paired_stars = []

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

        self.adjust_levels_mode = False

        # Platepar format (json or txt)
        self.platepar_fmt = None

        # Flat field
        self.flat_struct = None

        # Image coordinates of catalog stars
        self.catalog_x = self.catalog_y = None



        # Load catalog stars
        self.catalog_stars = self.loadCatalogStars(self.config.catalog_mag_limit)
        self.cat_lim_mag = self.config.catalog_mag_limit

        # Check if the BSC catalog exists
        if not self.catalog_stars.any():
            messagebox.showerror(title='Star catalog error', message='Star catalog from path ' \
                + os.path.join(self.config.star_catalog_path, self.config.star_catalog_file) \
                + 'could not be loaded!')
            sys.exit()
        else:
            print('Star catalog loaded!')


        # Find the CALSTARS file in the given folder
        calstars_file = None
        for cal_file in os.listdir(dir_path):
            if ('CALSTARS' in cal_file) and ('.txt' in cal_file):
                calstars_file = cal_file
                break

        if calstars_file is None:
            messagebox.showerror(title='CALSTARS error', message='CALSTARS file could not be found in the given directory!')
            sys.exit()

        # Load the calstars file
        calstars_list = CALSTARS.readCALSTARS(dir_path, calstars_file)

        # Convert the list to a dictionary
        self.calstars = {ff_file: star_data for ff_file, star_data in calstars_list}

        print('CALSTARS file: ' + calstars_file + ' loaded!')

        # A list of FF files which have any stars on them
        calstars_ff_files = [line[0] for line in calstars_list]


        self.ff_list = []

        # Get a list of FF files in the folder
        for file_name in os.listdir(dir_path):
            if validFFName(file_name) and (file_name in calstars_ff_files):
                self.ff_list.append(file_name)


        # Check that there are any FF files in the folder
        if not self.ff_list:
            messagebox.showinfo(title='File list warning', message='No FF files in the selected folder, or in the CALSTARS file!')

            sys.exit()


        # Sort the FF list
        self.ff_list = sorted(self.ff_list)

        # Init the first file
        self.current_ff_index = 0
        self.current_ff_file = self.ff_list[self.current_ff_index]




        # Load the platepar file
        self.platepar_file, self.platepar = self.loadPlatepar()

        print('Platepar loaded:', self.platepar_file)

        # Print the field of view size
        print("FOV: {:.2f} x {:.2f} deg".format(*self.computeFOVSize()))
        
        # If the platepar file was not loaded, set initial values from config
        if not self.platepar_file:
            self.makeNewPlatepar(update_image=False)

            # Create the name of the platepar file
            self.platepar_file = os.path.join(self.dir_path, self.config.platepar_name)


        ### INIT IMAGE ###

        plt.figure(facecolor='black')

        # Init the first image
        self.updateImage()

        self.ax = plt.gca()

        # Set the bacground color to black
        #matplotlib.rcParams['axes.facecolor'] = 'k'

        # Disable standard matplotlib keyboard shortcuts
        plt.rcParams['keymap.save'] = ''
        plt.rcParams['keymap.fullscreen'] = ''
        plt.rcParams['keymap.all_axes'] = ''
        plt.rcParams['keymap.quit'] = ''


        
        self.ax.figure.canvas.mpl_connect('button_press_event', self.onMousePress)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.onMouseRelease)
        
        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.onMouseMotion)

        # Register which mouse/keyboard events will evoke which function
        self.ax.figure.canvas.mpl_connect('scroll_event', self.onScroll)
        self.scroll_counter = 0

        self.ax.figure.canvas.mpl_connect('key_press_event', self.onKeyPress)



    def onMousePress(self, event):
        """ Called when the mouse click is pressed. """

        # Record the mouse cursor positions
        self.mouse_x_press = event.xdata
        self.mouse_y_press = event.ydata


    def onMouseRelease(self, event):
        """ Called when the mouse click is released. """

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


        plt.gcf().canvas.draw()



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

                # Skip intensities which were not properly calculated
                if np.isnan(px_intens) or np.isinf(px_intens):
                    continue

                star_coords.append([star_x, star_y])
                logsum_px.append(np.log10(px_intens))
                catalog_mags.append(star_mag)



            # Make sure there are more than 2 stars picked
            if len(logsum_px) > 2:

                def _photomLine(x, k):
                    # The slope is fixed to -2.5, coming from the definition of magnitude
                    return -2.5*x + k


                # Fit a line to the star data, where only the intercept has to be estimated
                photom_params, _ = scipy.optimize.curve_fit(_photomLine, logsum_px, catalog_mags, \
                    method='trf', loss='soft_l1')


                # Calculate the standard deviation
                fit_resids = np.array(catalog_mags) - _photomLine(np.array(logsum_px), *photom_params)
                fit_stddev = np.std(fit_resids)


                # Set photometry parameters
                self.platepar.mag_0 = -2.5
                self.platepar.mag_lev = photom_params[0]
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
                        star_mag_lbl = plt.text(star_x, star_y - 10, "{:+.2}".format(star_mag), \
                            verticalalignment='bottom', horizontalalignment='center', \
                            fontsize=10, color='r')


                        self.photom_deviatons_scat.append([photom_resid_lbl, star_mag_lbl])


                    plt.draw()


                # Show the photometry fit plot
                if show_plot:

                    # Init plot
                    fig_p = plt.figure(facecolor=None)
                    ax_p = fig_p.add_subplot(1, 1, 1)

                    # Plot catalog magnitude vs. logsum of pixel intensities
                    self.photom_points = ax_p.scatter(logsum_px, catalog_mags, s=5, c='r')

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
                    ax_p.plot(logsum_arr, _photomLine(logsum_arr, *photom_params), label=fit_info, linestyle='--', color='k', alpha=0.5)

                    ax_p.legend()

                    mag_str = "{:.2f}B + {:.2f}V + {:.2f}R + {:.2f}I".format(*self.config.star_catalog_band_ratios)
                    ax_p.set_ylabel("Catalog magnitude ({:s})".format(mag_str))
                    ax_p.set_xlabel("Logsum pixel")

                    # Set wider axis limits
                    ax_p.set_xlim(x_min_w, x_max_w)
                    ax_p.set_ylim(y_min_w, y_max_w)

                    ax_p.invert_yaxis()

                    ax_p.grid()

                    
                    fig_p.show()
                    #plt.show()
                    #fig_p.clf()


            else:
                print('Need more than 2 stars for photometry plot!')





    def onKeyPress(self, event):
        """ Traige what happes when an individual key is pressed. """


        # Switch images
        if event.key == 'left':
            self.prevFF()

        elif event.key == 'right':
            self.nextFF()


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
            self.platepar.RA_d -= self.key_increment
            self.updateImage()

        elif event.key == 'd':
            self.platepar.RA_d += self.key_increment
            self.updateImage()

        elif event.key == 'w':
            self.platepar.dec_d += self.key_increment
            self.updateImage()

        elif event.key == 's':
            self.platepar.dec_d -= self.key_increment
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
            self.show_key_help = not self.show_key_help
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
            self.platepar.x_poly[0] += 0.5
            self.updateImage()

        elif event.key == '2':

            # Decrement X offset
            self.platepar.x_poly[0] -= 0.5
            self.updateImage()


        elif event.key == '3':

            # Increment Y offset            
            self.platepar.y_poly[0] += 0.5
            self.updateImage()

        elif event.key == '4':

            # Decrement Y offset
            self.platepar.y_poly[0] -= 0.5
            self.updateImage()

        elif event.key == '5':

            # Decrement X 1st order distortion
            self.platepar.x_poly[1] -= 0.01
            self.updateImage()

        elif event.key == '6':

            # Increment X 1st order distortion
            self.platepar.x_poly[1] += 0.01
            self.updateImage()


        elif event.key == '7':

            # Decrement Y 1st order distortion
            self.platepar.y_poly[2] -= 0.01
            self.updateImage()

        elif event.key == '8':

            # Increment Y 1st order distortion
            self.platepar.y_poly[2] += 0.01
            self.updateImage()


        # Key increment
        elif event.key == '+':
            self.key_increment += 0.1
            self.updateImage()

        elif event.key == '-':
            self.key_increment -= 0.1
            self.updateImage()


        # Enter FOV centre
        elif event.key == 'v':

            self.platepar.RA_d, self.platepar.dec_d = self.getFOVcentre()
            
            # Recalculate reference alt/az
            self.platepar.az_centre, self.platepar.alt_centre = raDec2AltAz(self.platepar.JD, \
                self.platepar.lon, self.platepar.lat, self.platepar.RA_d, self.platepar.dec_d)
            
            self.updateImage()


        # Write out the new platepar
        elif event.key == 'ctrl+s':

            # If the platepar is new, save it to the working directory
            if not self.platepar_file:
                self.platepar_file = os.path.join(self.dir_path, self.config.platepar_name)

            # Save the platepar file
            self.platepar.write(self.platepar_file, fmt=self.platepar_fmt, fov=self.computeFOVSize())
            print('Platepar written to:', self.platepar_file)


        # Save the platepar as default (SHIFT+CTRL+S)
        elif event.key == 'ctrl+S':

            platepar_default_path = os.path.join(os.getcwd(), self.config.platepar_name)

            # Save the platepar file
            self.platepar.write(platepar_default_path, fmt=self.platepar_fmt)
            print('Default platepar written to:', platepar_default_path)


        # Create a new platepar
        elif event.key == 'ctrl+n':
            self.makeNewPlatepar()


        # Load the flat
        elif event.key == 'ctrl+f':
            _, self.flat_struct = self.loadFlat()

            self.updateImage()


        # Show/hide catalog stars
        elif event.key == 'h':

            if self.catalog_stars_visible:
                self.catalog_stars_visible = False

            else:
                self.catalog_stars_visible = True

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


        # Do a fit on the selected stars while in the star picking mode
        elif event.key == 'ctrl+z':

            # if self.star_pick_mode:

            #     # Do 3 fit iterations
            #     for i in range(3):

            #         print('Fitting iteration {:d}/3'.format(i + 1))

            #         self.fitPickedStars()

            #     print('Plate fitted!')

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

                    self.updateImage()

                    self.drawCursorCircle()

        elif event.key == 'escape':

            if self.star_pick_mode:

                # If the ESC is pressed when the star has been centroided, reset the centroid
                if not self.star_selection_centroid:
                    self.star_selection_centroid = True

                    self.updateImage()


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

        if self.star_aperature_radius > 50:
            self.star_aperature_radius = 50

        # Change the size of the star aperature circle
        if self.star_pick_mode:
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
        catalog_stars = StarCatalog.readStarCatalog(self.config.star_catalog_path, \
            self.config.star_catalog_file, lim_mag=lim_mag, \
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
                plt.scatter(x, y, marker='x', c='b', s=100, lw=3)



    def drawLevelsAdjustmentHistogram(self, img):

        nbins = int((2**self.config.bit_depth)/2)

        # Compute the intensity histogram
        hist, bin_edges = np.histogram(img.flatten(), normed=True, range=(0, 2**self.bit_depth), \
            bins=nbins)

        # Scale the maximum histogram peak to half the image height
        hist *= img.shape[0]/np.max(hist)/2

        # Scale the edges to image width
        image_to_level_scale = img.shape[1]/np.max(bin_edges)
        bin_edges *= image_to_level_scale

        ax1 = plt.gca()
        ax2 = ax1.twinx().twiny()

        # Plot the histogram
        ax2.bar(bin_edges[:-1], hist, color='white', alpha=0.5, width=img.shape[1]/nbins, edgecolor='k')

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

        plt.sca(ax1)





    def updateImage(self, clear_plot=True):
        """ Update the matplotlib plot to show the current image. 

        Keyword arguments:
            clear_plot: [bool] If True, the plot will be cleared before plotting again (default).
        """

        # Reset circle patches
        self.circle_aperature = None
        self.circle_aperature_outer = None

        # Limit key increment so it can be lower than 0.1
        if self.key_increment < 0.1:
            self.key_increment = 0.1


        if clear_plot:
            plt.clf()


        # Check that the calibration parameters are within the nominal range
        self.checkParamRange()


        # Load the FF from the current file
        self.current_ff = readFF(self.dir_path, self.current_ff_file)


        if self.current_ff is None:
            
            # If the current FF couldn't be opened, go to the next
            messagebox.showerror(title='Read error', message='The current FF file is corrupted!')
            self.nextFF()

            return None


        # Choose appropriate image data
        if self.img_type_flag == 'maxpixel':
            img_data = self.current_ff.maxpixel

        else:
            img_data = self.current_ff.avepixel


        # Guess the bit depth from the array type
        self.bit_depth = 8*img_data.itemsize


        # Apply flat
        if self.flat_struct is not None:
            img_data = Image.applyFlat(img_data, self.flat_struct)


        # Store image before modifications
        self.img_data_raw = np.copy(img_data)


        ### Adjust image levels

        img_data = Image.adjustLevels(img_data, self.img_level_min, self.img_gamma, self.img_level_max, \
            self.bit_depth)

        ###

        # Draw levels adjustment histogram
        if self.adjust_levels_mode:
            self.drawLevelsAdjustmentHistogram(self.img_data_raw)

        # Show the loaded image
        plt.imshow(img_data, cmap='gray')

        # Draw stars that were paired in picking mode
        self.drawPairedStars()

        # Draw stars detected on this image
        self.drawCalstars()

        # Update centre of FOV in horizontal coordinates
        self.platepar.az_centre, self.platepar.alt_centre = raDec2AltAz(self.platepar.JD, self.platepar.lon, 
            self.platepar.lat, self.platepar.RA_d, self.platepar.dec_d)

        ### Draw catalog stars on the image using the current platepar ###
        ######################################################################################################
        self.catalog_x, self.catalog_y, catalog_mag = self.getCatalogStarPositions(self.catalog_stars, \
            self.platepar.lon, self.platepar.lat, self.platepar.RA_d, self.platepar.dec_d, 
            self.platepar.pos_angle_ref, self.platepar.F_scale, self.platepar.x_poly, self.platepar.y_poly)

        if self.catalog_stars_visible:
            cat_stars = np.c_[self.catalog_x, self.catalog_y, catalog_mag]

            # Take only those stars inside the FOV
            filtered_indices, _ = self.filterCatalogStarsInsideFOV(self.catalog_stars)
            cat_stars = cat_stars[filtered_indices]
            cat_stars = cat_stars[cat_stars[:, 0] > 0]
            cat_stars = cat_stars[cat_stars[:, 0] < self.current_ff.ncols]
            cat_stars = cat_stars[cat_stars[:, 1] > 0]
            cat_stars = cat_stars[cat_stars[:, 1] < self.current_ff.nrows]

            catalog_x_filtered, catalog_y_filtered, catalog_mag_filtered = cat_stars.T

            if len(catalog_mag_filtered):

                cat_mag_faintest = np.max(catalog_mag_filtered)

                # Plot catalog stars
                plt.scatter(catalog_x_filtered, catalog_y_filtered, c='r', marker='+', lw=1.0, alpha=0.5, \
                    s=((4.0 + (cat_mag_faintest - catalog_mag_filtered))/2.0)**(2*2.512))

            else:
                print('No catalog stars visible!')

        ######################################################################################################


        # Draw photometry
        if len(self.paired_stars) > 2:
            self.photometry()


        # Set plot limits
        plt.xlim([0, self.current_ff.ncols])
        plt.ylim([self.current_ff.nrows, 0])


        # Show text on the top
        if self.star_pick_mode:
            text_str  = "STAR PICKING MODE\n"
            text_str += "PRESS 'CTRL + Z' FOR STAR FITTING\n"
            text_str += "PRESS 'P' FOR PHOTOMETRY FIT"

            plt.gca().text(self.current_ff.ncols/2, self.current_ff.nrows - 10, text_str, color='r', 
                verticalalignment='top', horizontalalignment='center', fontsize=8)

        # Show text on image with platepar parameters
        text_str  = self.current_ff_file + '\n' + self.img_type_flag + '\n\n'
        text_str += 'UT corr = {:.1f}\n'.format(self.platepar.UT_corr)
        text_str += 'Ref RA  = {:.3f}\n'.format(self.platepar.RA_d)
        text_str += 'Ref Dec = {:.3f}\n'.format(self.platepar.dec_d)
        text_str += 'PA      = {:.3f}\n'.format(self.platepar.pos_angle_ref)
        text_str += 'F_scale = {:.3f}\n'.format(self.platepar.F_scale)
        text_str += 'Lim mag = {:.1f}\n'.format(self.cat_lim_mag)
        text_str += 'Increment = {:.3f}\n'.format(self.key_increment)
        text_str += 'Img Gamma = {:.2f}\n'.format(self.img_gamma)
        plt.gca().text(10, 10, text_str, color='w', verticalalignment='top', horizontalalignment='left', 
            fontsize=8)

        if self.show_key_help:

            # Show text on image with instructions
            text_str  = 'Keys:\n'
            text_str += '-----\n'
            text_str += 'A/D - RA\n'
            text_str += 'S/W - Dec\n'
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
            text_str += 'U/J - Img Gamma\n'
            text_str += 'CTRL + H - Adjust levels\n'
            text_str += 'V - FOV centre\n'
            text_str += '\n'
            text_str += 'CTRL + F - Load flat\n'
            text_str += 'CTRL + R - Pick stars\n'
            text_str += 'CTRL + N - New platepar\n'
            text_str += 'CTRL + S - Save platepar\n'
            text_str += 'SHIFT + CTRL + S - Save platepar as default\n'

            text_str += '\n'

            text_str += 'Hide keyboard shortcuts - F1\n'


            plt.gca().text(10, self.current_ff.nrows - 5, text_str, color='w', verticalalignment='bottom', 
                horizontalalignment='left', fontsize=8)

        else:
            text_str = 'Show keyboard shortcuts - F1'

            plt.gca().text(10, self.current_ff.nrows, text_str, color='w', verticalalignment='bottom', 
                horizontalalignment='left', fontsize=8)


        plt.gcf().canvas.draw()



    def drawCalstars(self):
        """ Draw extracted stars on the current image. """

        # Get the stars detected on this FF file
        star_data = self.calstars[self.current_ff_file]

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



    def filterCatalogStarsInsideFOV(self, catalog_stars):
        """ Take only catalogs stars which are inside the FOV. 
        
        Arguments:
            catalog_stars: [list] A list of (ra, dec, mag) tuples of catalog stars.
        """

        # The the time of the midle of the FF file
        ff_middle_time = getMiddleTimeFF(self.current_ff_file, self.config.fps, ret_milliseconds=True)

        # Convert the FOV centre to RA/Dec
        _, ra_centre, dec_centre, _ = XY2CorrectedRADecPP([ff_middle_time], [self.platepar.X_res/2], 
            [self.platepar.Y_res/2], [0], self.platepar)

        print('RA/Dec centre:', ra_centre, dec_centre)
        
        ra_centre = ra_centre[0]
        dec_centre = dec_centre[0]

        # Calculate the FOV radius in degrees
        fov_x = (self.platepar.X_res/2)/self.platepar.F_scale
        fov_y = (self.platepar.Y_res/2)/self.platepar.F_scale

        fov_radius = np.sqrt(fov_x**2 + fov_y**2)


        # filtered_catalog_stars = []
        # filtered_indices = []

        # # Calculate minimum and maximum declination
        # dec_min = dec_centre - fov_radius
        # if dec_min < -90:
        #     dec_min = -90

        # dec_max = dec_centre + fov_radius
        # if dec_max > 90:
        #     dec_max = 90

        # # Take only those catalog stars which should be inside the FOV
        # for i, (ra, dec, _) in enumerate(catalog_stars):

        #     # Skip if the declination is too large
        #     if dec > dec_max:
        #         continue

        #     # End the loop if the declination is too small
        #     if dec < dec_min:
        #         break

        #     # Calculate angular separation between the FOV centre and the catalog star
        #     ang_sep = math.degrees(math.acos(math.sin(math.radians(dec))*math.sin(math.radians(dec_centre)) \
        #         + math.cos(math.radians(dec))*math.cos(math.radians(dec_centre))*math.cos(math.radians(ra) \
        #         - math.radians(ra_centre))))

        #     if ang_sep <= fov_radius:

        #         # Add stars which are roughly inside the FOV to the OK list
        #         filtered_catalog_stars.append(catalog_stars[i])
        #         filtered_indices.append(i)

        filtered_indices, filtered_catalog_stars = subsetCatalog(catalog_stars, ra_centre, dec_centre, \
            fov_radius, self.cat_lim_mag)


        return filtered_indices, np.array(filtered_catalog_stars)



    def getCatalogStarPositions(self, catalog_stars, lon, lat, ra_ref, dec_ref, pos_angle_ref, F_scale, \
        x_poly, y_poly):
        """ Get image positions of catalog stars using the current platepar values. 
    
        Arguments:
            catalog_stars: [2D list] A list of (ra, dec, mag) pairs of catalog stars.
            lon: [float] Longitude in degrees.
            lat: [float] Latitude in degrees.
            ra_ref: [float] Referent RA of the FOV centre (degrees).
            dec_ref: [float] Referent Dec of the FOV centre (degrees).
            pos_angle_ref: [float] Referent position angle in degrees.
            F_scale: [float] Image scale in pix/arcsec for CIF resolution.
            x_poly: [ndarray float] Distorsion polynomial in X direction.
            y_poly: [ndarray float] Distorsion polynomail in Y direction.

        Return:
            (x_array, y_array): [tuple of floats] X and Y positons of stars on the image.
        """

        ra_catalog, dec_catalog, mag_catalog = catalog_stars.T

        ff_middle_time = getMiddleTimeFF(self.current_ff_file, self.config.fps, ret_milliseconds=True)

        # Get the date of the middle of the FF exposure
        jd = date2JD(*ff_middle_time)

        # Convert star RA, Dec to image coordinates
        x_array, y_array = raDecToCorrectedXY(ra_catalog, dec_catalog, jd, lat, lon, self.platepar.X_res, \
            self.platepar.Y_res, ra_ref, dec_ref, self.platepar.JD, pos_angle_ref, F_scale, x_poly, y_poly, 
            UT_corr=self.platepar.UT_corr)

        return x_array, y_array, mag_catalog



    def getFOVcentre(self):
        """ Asks the user to input the centre of the FOV in altitude and azimuth. """

        # Get FOV centre
        root = tkinter.Tk()
        root.withdraw()
        d = FOVinputDialog(root)
        root.wait_window(d.top)
        self.azim_centre, self.alt_centre = d.getAltAz()

        root.destroy()

        # Get the middle time of the first FF
        ff_middle_time = getMiddleTimeFF(self.current_ff_file, self.config.fps, ret_milliseconds=True)

        # Set the referent platepar time to the time of the FF
        self.platepar.JD = date2JD(*ff_middle_time, UT_corr=float(self.platepar.UT_corr))

        # Set the referent hour angle
        T = (self.platepar.JD - 2451545)/36525.0
        Ho = 280.46061837 + 360.98564736629*(self.platepar.JD - 2451545.0) + 0.000387933*T**2 \
            - (T**3)/38710000.0

        self.platepar.Ho = Ho

        
        time_data = [ff_middle_time]

        # Convert FOV centre to RA, Dec
        _, ra_data, dec_data = altAz2RADec(self.platepar.lat, self.platepar.lon, self.platepar.UT_corr, 
            time_data, [self.azim_centre], [self.alt_centre])


        return ra_data[0], dec_data[0]



    def loadPlatepar(self):
        """ Open a file dialog and ask user to open the platepar file. """

        root = tkinter.Tk()
        root.withdraw()
        root.update()

        platepar = Platepar()


        # Check if platepar exists in the folder, and set it as the defualt file name if it does
        if self.config.platepar_name in os.listdir(self.dir_path):
            initialfile = self.config.platepar_name
        else:
            initialfile = ''

        # Load the platepar file
        platepar_file = filedialog.askopenfilename(initialdir=self.dir_path, \
            initialfile=initialfile, title='Select the platepar file')

        root.update()
        root.quit()
        # root.destroy()

        if not platepar_file:
            return False, platepar

        print(platepar_file)

        # Parse the platepar file
        try:
            self.platepar_fmt = platepar.read(platepar_file)
        except:
            platepar = False

        # Check if the platepar was successfuly loaded
        if not platepar:
            messagebox.showerror(title='Platepar file error', message='The file you selected could not be loaded as a platepar file!')
            
            self.loadPlatepar()

        

        return platepar_file, platepar



    def makeNewPlatepar(self, update_image=True):
        """ Make a new platepar from the loaded one, but set the parameters from the config file. """

        # Update the reference time
        ff_middle_time = getMiddleTimeFF(self.current_ff_file, self.config.fps, ret_milliseconds=True)
        self.platepar.JD = date2JD(*ff_middle_time)

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
        self.platepar.x_poly *= 0
        self.platepar.y_poly *= 0

        # Set station ID
        self.platepar.station_code = self.config.stationID


        # Get referent RA, Dec of the image centre
        self.platepar.RA_d, self.platepar.dec_d = self.getFOVcentre()

        # Recalculate reference alt/az
        self.platepar.az_centre, self.platepar.alt_centre = raDec2AltAz(self.platepar.JD, \
            self.platepar.lon, self.platepar.lat, self.platepar.RA_d, self.platepar.dec_d)

        # Check that the calibration parameters are within the nominal range
        self.checkParamRange()


        if update_image:
            self.updateImage()



    def computeFOVSize(self):
        """ Computes the size of the FOV in deg from the given platepar. 
        
        Return:
            fov_h: [float] Horizontal FOV in degrees.
            fov_v: [float] Vertical FOV in degrees.
        """

        # Construct poinits on the middle of every side of the image
        time_data = np.array(4*[jd2Date(self.platepar.JD)])
        x_data = np.array([0, self.platepar.X_res, self.platepar.X_res/2, self.platepar.X_res/2])
        y_data = np.array([self.platepar.Y_res/2, self.platepar.Y_res/2, 0, self.platepar.Y_res])
        level_data = np.ones(4)

        # Compute RA/Dec of the points
        _, ra_data, dec_data, _ = XY2CorrectedRADecPP(time_data, x_data, y_data, level_data, self.platepar)

        ra1, ra2, ra3, ra4 = ra_data
        dec1, dec2, dec3, dec4 = dec_data

        # Compute horizontal FOV
        fov_h = np.degrees(angularSeparation(np.radians(ra1), np.radians(dec1), np.radians(ra2), \
            np.radians(dec2)))

        # Compute vertical FOV
        fov_v = np.degrees(angularSeparation(np.radians(ra3), np.radians(dec3), np.radians(ra4), \
            np.radians(dec4)))


        return fov_h, fov_v


    def loadFlat(self):
        """ Open a file dialog and ask user to load a flat field. """

        root = tkinter.Tk()
        root.withdraw()
        root.update()


        # Check if flat exists in the folder, and set it as the defualt file name if it does
        if self.config.flat_file in os.listdir(self.dir_path):
            initialfile = self.config.flat_file
        else:
            initialfile = ''

        # Load the platepar file
        flat_file = filedialog.askopenfilename(initialdir=self.dir_path, \
            initialfile=initialfile, title='Select the flat field file')

        root.update()
        root.quit()

        if not flat_file:
            return False, None

        print(flat_file)

        # Parse the platepar file
        try:
            flat = Image.loadFlat(*os.path.split(flat_file))
        except:
            flat = None


        # Check if the size of the file matches
        if self.current_ff.maxpixel.shape != flat.flat_img.shape:
            messagebox.showerror(title='Flat field file error', message='The size of the flat field does not match the size of the image!')
            flat = None

        # Check if the platepar was successfuly loaded
        if flat is None:
            messagebox.showerror(title='Flat field file error', message='The file you selected could not be loaded as a flat field!')

        

        return flat_file, flat


    def nextFF(self):
        """ Shows the next FF file in the list. """

        # Don't allow image change while in star picking mode
        if self.star_pick_mode:
            messagebox.showwarning(title='Star picking mode', message='You cannot cycle through images while in star picking mode!')
            return

        self.current_ff_index = (self.current_ff_index + 1)%len(self.ff_list)
        self.current_ff_file = self.ff_list[self.current_ff_index]

        self.paired_stars = []

        self.updateImage()



    def prevFF(self):
        """ Shows the previous FF file in the list. """

        # Don't allow image change while in star picking mode
        if self.star_pick_mode:
            messagebox.showwarning(title='Star picking mode', message='You cannot cycle through images while in star picking mode!')
            return

        self.current_ff_index = (self.current_ff_index - 1)%len(self.ff_list)
        self.current_ff_file = self.ff_list[self.current_ff_index]

        self.paired_stars = []

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

        ### Extract part of image around the mouse cursor ###
        ######################################################################################################

        # Outer circle radius
        outer_radius = self.star_aperature_radius*2

        x_min = int(round(mouse_x - outer_radius))
        if x_min < 0: x_min = 0

        x_max = int(round(mouse_x + outer_radius))
        if x_max > self.current_ff.ncols - 1:
            x_max > self.current_ff.ncols - 1

        y_min = int(round(mouse_y - outer_radius))
        if y_min < 0: y_min = 0

        y_max = int(round(mouse_y + outer_radius))
        if y_max > self.current_ff.nrows - 1:
            y_max > self.current_ff.nrows - 1


        if self.img_type_flag == 'maxpixel':
            img_data = self.current_ff.maxpixel

        else:
            img_data = self.current_ff.avepixel


        # Crop the image
        img_crop = img_data[y_min:y_max, x_min:x_max]

        # perform gamma correction
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

        # Fit the astrometry parameters, at least 8 stars are needed
        if len(self.paired_stars) < 6:
            messagebox.showwarning(title='Number of stars', message="At least 6 paired stars are needed to do the fit!")

            return False


        def _calcImageResidualsAstro(params, self, catalog_stars, img_stars):
            """ Calculates the differences between the stars on the image and catalog stars in image 
                coordinates with the given astrometrical solution. 

            """

            # Extract fitting parameters
            ra_ref, dec_ref, pos_angle_ref, F_scale = params

            img_x, img_y, _ = img_stars.T

            # Get image coordinates of catalog stars
            catalog_x, catalog_y, catalog_mag = self.getCatalogStarPositions(catalog_stars, self.platepar.lon, 
                self.platepar.lat, ra_ref, dec_ref, pos_angle_ref, F_scale, self.platepar.x_poly, 
                self.platepar.y_poly)


            
            # Calculate the sum of squared distances between image stars and catalog stars
            dist_sum = np.sum((catalog_x - img_x)**2 + (catalog_y - img_y)**2)


            return dist_sum



        def _calcImageResidualsDistorsion(params, self, catalog_stars, img_stars, dimension):
            """ Calculates the differences between the stars on the image and catalog stars in image 
                coordinates with the given astrometrical solution. 

            Arguments:
                ...
                dimension: [str] 'x' for X polynomial fit, 'y' for Y polynomial fit

            """

            if dimension == 'x':
                x_poly = params
                y_poly = np.zeros(12)
                #y_poly = self.platepar.y_poly

            else:
                x_poly = np.zeros(12)
                #x_poly = self.platepar.x_poly
                y_poly = params


            img_x, img_y, _ = img_stars.T

            # Get image coordinates of catalog stars
            catalog_x, catalog_y, catalog_mag = self.getCatalogStarPositions(catalog_stars, self.platepar.lon, 
                self.platepar.lat, self.platepar.RA_d, self.platepar.dec_d, self.platepar.pos_angle_ref, 
                self.platepar.F_scale, x_poly, y_poly)


            # Calculate the sum of squared distances between image stars and catalog stars, per every
            #   dimension
            if dimension == 'x':
                dist_sum = np.sum((catalog_x - img_x)**2)

            else:
                dist_sum = np.sum((catalog_y - img_y)**2)


            return dist_sum



        # Extract paired catalog stars and image coordinates separately
        catalog_stars = np.array([cat_coords for img_coords, cat_coords in self.paired_stars])
        img_stars = np.array([img_coords for img_coords, cat_coords in self.paired_stars])

        print('ASTRO', _calcImageResidualsAstro([self.platepar.RA_d, self.platepar.dec_d, 
            self.platepar.pos_angle_ref, self.platepar.F_scale], self, catalog_stars, img_stars))

        print('DIS_X', _calcImageResidualsDistorsion(self.platepar.x_poly, self, catalog_stars, img_stars, 'x'))

        print('DIS_Y', _calcImageResidualsDistorsion(self.platepar.x_poly, self, catalog_stars, img_stars, 'y'))


        # Initial parameters for the astrometric fit
        p0 = [self.platepar.RA_d, self.platepar.dec_d, self.platepar.pos_angle_ref, self.platepar.F_scale]

        # Fit the astrometric parameters
        res = scipy.optimize.minimize(_calcImageResidualsAstro, p0, args=(self, catalog_stars, img_stars),
            method='Nelder-Mead')

        print(res)

        # Update fitted astrometric parameters
        self.platepar.RA_d, self.platepar.dec_d, self.platepar.pos_angle_ref, self.platepar.F_scale = res.x

        # Recalculate centre
        self.platepar.az_centre, self.platepar.alt_centre = raDec2AltAz(self.platepar.JD, self.platepar.lon, 
            self.platepar.lat, self.platepar.RA_d, self.platepar.dec_d)


        # Save the size of the image
        self.platepar.Y_res, self.platepar.X_res = self.current_ff.maxpixel.shape


        # Fit distorsion parameters in X direction
        res = scipy.optimize.minimize(_calcImageResidualsDistorsion, self.platepar.x_poly, args=(self, 
            catalog_stars, img_stars, 'x'), method='Nelder-Mead', options={'maxiter': 10000})

        # Exctact fitted X polynomial
        self.platepar.x_poly = res.x

        print(res)

        # Fit distorsion parameters in Y direction
        res = scipy.optimize.minimize(_calcImageResidualsDistorsion, self.platepar.y_poly, args=(self, 
            catalog_stars, img_stars, 'y'), method='Nelder-Mead', options={'maxiter': 10000})

        # Extract fitted Y polynomial
        self.platepar.y_poly = res.x

        print(res)



        ### Calculate the fit residuals for every fitted star ###
        
        # Get image coordinates of catalog stars
        catalog_x, catalog_y, catalog_mag = self.getCatalogStarPositions(catalog_stars, self.platepar.lon, 
            self.platepar.lat, self.platepar.RA_d, self.platepar.dec_d, self.platepar.pos_angle_ref, 
            self.platepar.F_scale, self.platepar.x_poly, self.platepar.y_poly)


        residuals = []

        print()
        print('Residuals')
        print('----------')
        print(' No,   Img X,   Img Y, RA (deg), Dec (deg),    Mag,   Cat X,   Cat Y,    Dist,  Angle')

        # Calculate the distance and the angle between each pair of image positions and catalog predictions
        for star_no, (cat_x, cat_y, cat_coords, img_c) in enumerate(zip(catalog_x, catalog_y, catalog_stars, \
            img_stars)):
            
            img_x, img_y, _ = img_c
            ra, dec, mag = cat_coords

            delta_x = cat_x - img_x
            delta_y = cat_y - img_y

            angle = np.arctan2(delta_y, delta_x)
            distance = np.sqrt(delta_x**2 + delta_y**2)


            residuals.append([img_x, img_y, angle, distance])

            # Print out the residuals
            print('{:3d}, {:7.2f}, {:7.2f}, {:>8.3f}, {:>+9.3f}, {:+6.2}, {:7.2f}, {:7.2f}, {:7.2f}, {:+5.1f}'.format(star_no + 1, img_x, img_y, \
                ra, dec, mag, cat_x, cat_y, distance, np.degrees(angle)))

        print('Average distance: {:.2f} px'.format(np.mean([entry[3] for entry in residuals])))

        # Print the field of view size
        print("FOV: {:.2f} x {:.2f} deg".format(*self.computeFOVSize()))


        ####################


        self.updateImage()


        # Plot the residuals
        res_scale = 100
        for entry in residuals:

            img_x, img_y, angle, distance = entry

            # Calculate coordinates of the end of the residual line
            res_x = img_x + res_scale*np.cos(angle)*distance
            res_y = img_y + res_scale*np.sin(angle)*distance

            plt.plot([img_x, res_x], [img_y, res_y], color='orange')


        plt.draw()



if __name__ == '__main__':


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Tool for fitting astrometry plates and photometric calibration.")

    arg_parser.add_argument('dir_path', nargs=1, metavar='DIR_PATH', type=str, \
        help='Path to the folder with FF files.')

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
        help="Path to a config file which will be used instead of the default one.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    if cml_args.config is not None:

        config_file = os.path.abspath(cml_args.config[0].replace('"', ''))

        print('Loading config file:', config_file)

        # Load the given config file
        config = cr.parse(config_file)

    else:
        # Load the default configuration file
        config = cr.parse(".config")


    # Init the plate tool instance
    plate_tool = PlateTool(cml_args.dir_path[0].replace('"', ''), config)

    plt.tight_layout()
    plt.show()