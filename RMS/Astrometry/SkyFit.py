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

import cv2

import RMS.Formats.BSC as BSC
import RMS.Formats.CALSTARS as CALSTARS
from RMS.Formats.Platepar import Platepar
from RMS.Formats.FFfile import read as readFF
from RMS.Formats.FFfile import validFFName
from RMS.Formats.FFfile import getMiddleTimeFF
import RMS.ConfigReader as cr
from RMS.Astrometry.ApplyAstrometry import altAz2RADec, XY2CorrectedRADec
from RMS.Astrometry.Conversions import date2JD
from RMS.Routines.Image import adjustLevels


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

        self.config = config
        self.dir_path = dir_path

        # Star picking mode
        self.star_pick_mode = False
        self.star_selection_centroid = True
        self.circle_aperature = None
        self.circle_aperature_outer = None
        self.star_aperature_radius = 5
        self.x_centroid = self.y_centroid = None

        self.catalog_stars_visible = True

        self.show_key_help = True

        # List of paired image and catalog stars
        self.paired_stars = []

        # Positions of the mouse cursor
        self.mouse_x = 0
        self.mouse_y = 0

        # Kwy increment
        self.key_increment = 1.0

        # Image gamma
        self.img_gamma = 1.0

        # Time difference from UT
        self.UT_corr = 0

        # Platepar format (json or txt)
        self.platepar_fmt = None

        # Load catalog stars
        self.catalog_stars = self.loadCatalogStars(self.config.catalog_mag_limit)
        self.cat_lim_mag = self.config.catalog_mag_limit

        # Check if the BSC exists
        if not self.catalog_stars.any():
            messagebox.showerror(title='Star catalog error', message='Star catalog from path ' \
                + os.path.join(self.config.star_catalog_path, self.config.star_catalog_file) \
                + 'could not be loaded!')
            sys.exit()
        else:
            print('Star catalog loaded!')


        # Image coordinates of catalog stars
        self.catalog_x = self.catalog_y = None



        # Find the CALSTARS file in the given folder
        calstars_file = None
        for calstars_file in os.listdir(dir_path):
            if ('CALSTARS' in calstars_file) and ('.txt' in calstars_file):
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
            messagebox.showinfo(title='File list warning', message='No FF files in the selected folder!')

            sys.exit()

        self.current_ff_index = 0
        self.current_ff_file = self.ff_list[self.current_ff_index]


        # Load the platepar file
        self.platepar_file, self.platepar = self.loadPlatepar()
        
        # If the platepar file was not loaded, set initial values from config
        if not self.platepar_file:
            self.makeNewPlatepar(update_image=False)
            self.platepar.RA_d, self.platepar.dec_d = self.getFOVcentre()

        plt.figure(facecolor='black')

        # Init the first image
        self.updateImage()

        self.ax = plt.gca()

        # Set the bacground color to black
        matplotlib.rcParams['axes.facecolor'] = 'k'

        # Disable standard matplotlib keyboard shortcuts
        plt.rcParams['keymap.save'] = ''
        plt.rcParams['keymap.fullscreen'] = ''
        plt.rcParams['keymap.all_axes'] = ''

        
        # self.ax.figure.canvas.mpl_connect('button_press_event', self.onMousePress)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.onMouseRelease)
        
        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.onMouseMotion)

        # Register which mouse/keyboard events will evoke which function
        self.ax.figure.canvas.mpl_connect('scroll_event', self.onScroll)
        self.scroll_counter = 0

        self.ax.figure.canvas.mpl_connect('key_press_event', self.onKeyPress)


    def onMouseRelease(self, event):
        """ Called when the mouse click is released. """

        # Call the same function for mouse movements to update the variables in the background
        self.onMouseMotion(event)

        # If the star picking mode is on
        if self.star_pick_mode:

            # Left mouse button, select stars
            if event.button == 1:

                # If the centroid of the star has to be picked
                if self.star_selection_centroid:

                    # Centroid the star around the pressed coordinates
                    self.x_centroid, self.y_centroid, self.star_intensity = self.centroidStar()

                    # Draw the centroid on the image
                    plt.scatter(self.x_centroid, self.y_centroid, marker='+', c='y', s=50, lw=2)

                    # Select the closest catalog star to the centroid as the first guess
                    self.closest_cat_star_indx = self.findClosestCatalogStarIndex(self.x_centroid, self.y_centroid)

                    # Plot the closest star as a purple cross
                    self.selected_cat_star_scatter = plt.scatter(self.catalog_x[self.closest_cat_star_indx], 
                        self.catalog_y[self.closest_cat_star_indx], marker='+', c='purple', s=50, lw=2)

                    # Update canvas
                    plt.gcf().canvas.draw()

                    # Switch to the mode where the catalog star is selected
                    self.star_selection_centroid = False

                    self.drawCursorCircle()


                # If the catalog star has to be picked
                else:

                    # Select the closest catalog star
                    self.closest_cat_star_indx = self.findClosestCatalogStarIndex(self.mouse_x, self.mouse_y)

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
                    picked_indx = self.findClosestPickedStarIndex(self.mouse_x, self.mouse_y)

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



    def onKeyPress(self, event):
        """ Traige what happes when an individual key is pressed. """


        # Switch images
        if event.key == 'left':
            self.prevFF()

        elif event.key == 'right':
            self.nextFF()


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
            self.cat_lim_mag -= 0.1
            self.catalog_stars = self.loadCatalogStars(self.cat_lim_mag)
            self.updateImage()

        elif event.key == 'f':
            self.cat_lim_mag += 0.1
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
            self.updateImage()


        # Write out the new platepar
        elif event.key == 'ctrl+s':

            # If the platepar is new, save it to the working directory
            if not self.platepar_file:
                self.platepar_file = os.path.join(self.dir_path, self.config.platepar_name)

            # Save the platepar file
            self.platepar.write(self.platepar_file, fmt=self.platepar_fmt)
            print('Platepar written to:', self.platepar_file)


        # Create a new platepar
        elif event.key == 'ctrl+n':
            self.makeNewPlatepar()


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

            if self.star_pick_mode:

                self.fitPickedStars()


        elif event.key == 'enter':

            if self.star_pick_mode:

                # If the right catalog star has been selected, save the pair to the list
                if not self.star_selection_centroid:

                    # Add the image/catalog pair to the list
                    self.paired_stars.append([[self.x_centroid, self.y_centroid], 
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




    def loadCatalogStars(self, lim_mag):
        """ Loads stars from the BSC star catalog. 
    
        Arguments:
            lim_mag: [float] Limiting magnitude of catalog stars.

        """

        # Load catalog stars
        catalog_stars = BSC.readBSC(self.config.star_catalog_path, self.config.star_catalog_file, 
            lim_mag=lim_mag)

        return catalog_stars


    def drawPairedStars(self):
        """ Draws the stars that were picked for calibration. """

        if self.star_pick_mode:

            # Go through all paired stars
            for paired_star in self.paired_stars:

                img_star, catalog_star = paired_star

                x, y = img_star

                # Plot all paired stars
                plt.scatter(x, y, marker='x', color='b')



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

        # Load the FF from the current file
        self.current_ff = readFF(self.dir_path, self.current_ff_file)

        img_data = self.current_ff.maxpixel

        # Adjust image levels
        img_data = adjustLevels(img_data, 0, self.img_gamma, (2**self.config.bit_depth -1), self.config.bit_depth)

        # Show the loaded maxpixel
        plt.imshow(img_data, cmap='gray')

        # Draw stars that were paired in picking mode
        self.drawPairedStars()

        # Draw stars detected on this image
        self.drawCalstars()

        # Update centre of FOV in horizontal coordinates
        self.platepar.az_centre, self.platepar.alt_centre = self.calcRefCentre(self.platepar.JD, self.platepar.lon, 
            self.platepar.lat, self.platepar.RA_d, self.platepar.dec_d)

        ### Draw catalog stars on the image using the current platepar ###
        ######################################################################################################
        self.catalog_x, self.catalog_y = self.getCatalogStarPositions(self.catalog_stars, self.platepar.lon, 
            self.platepar.lat, self.platepar.az_centre, self.platepar.alt_centre, self.platepar.pos_angle_ref, 
            self.platepar.F_scale, self.platepar.x_poly, self.platepar.y_poly)

        if self.catalog_stars_visible:
            cat_stars = np.c_[self.catalog_x, self.catalog_y]

            # Take only those stars inside the FOV
            filtered_indices, _ = self.filterCatalogStarsInsideFOV(self.catalog_stars)
            cat_stars = cat_stars[filtered_indices]
            cat_stars = cat_stars[cat_stars[:, 0] > 0]
            cat_stars = cat_stars[cat_stars[:, 0] < self.current_ff.ncols]
            cat_stars = cat_stars[cat_stars[:, 1] > 0]
            cat_stars = cat_stars[cat_stars[:, 1] < self.current_ff.nrows]

            catalog_x_filtered, catalog_y_filtered = cat_stars.T

            # Plot catalog stars (mew - marker edge width)
            plt.scatter(catalog_x_filtered, catalog_y_filtered, c='r', marker='+', lw=1.0, alpha=0.5)

        ######################################################################################################


        # Set plot limits
        plt.xlim([0, self.current_ff.ncols])
        plt.ylim([self.current_ff.nrows, 0])

        # Show text on the top
        if self.star_pick_mode:
            text_str = 'STAR PICKING MODE, PRESS CTRL + Z FOR FITTING'

            plt.gca().text(self.current_ff.ncols/2, self.current_ff.nrows - 10, text_str, color='r', 
                verticalalignment='top', horizontalalignment='center', fontsize=8)

        if self.show_key_help:
            # Show text on image with platepar parameters
            text_str  = self.current_ff_file + '\n\n'
            text_str += 'RA  = {:.3f}\n'.format(self.platepar.RA_d)
            text_str += 'Dec = {:.3f}\n'.format(self.platepar.dec_d)
            text_str += 'PA  = {:.3f}\n'.format(self.platepar.pos_angle_ref)
            text_str += 'F_scale = {:.3f}\n'.format(self.platepar.F_scale)
            text_str += 'Lim mag = {:.1f}\n'.format(self.cat_lim_mag)
            text_str += 'Increment = {:.3}\n'.format(self.key_increment)
            text_str += 'Img Gamma = {:.2}\n'.format(self.img_gamma)
            plt.gca().text(10, 10, text_str, color='w', verticalalignment='top', horizontalalignment='left', 
                fontsize=8)

            # Show text on image with instructions
            text_str  = 'Keys:\n'
            text_str += 'RA  - A/D\n'
            text_str += 'Dec - S/W\n'
            text_str += 'PA  - Q/E\n'
            text_str += 'F_scale - Up/Down\n'
            text_str += 'Lim mag - R/F\n'
            text_str += 'Increment - +/-\n'
            text_str += 'Img Gamma - U/J\n'
            text_str += 'Hide/show catalog stars - H\n'
            text_str += 'FOV centre - V\n'
            text_str += 'Pick stars - CTRL + R\n'
            text_str += 'New platepar - CTRL + N\n'
            text_str += 'Save platepar - CTRL + S\n'
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

        plt.scatter(x, y, edgecolors='g', marker='o', facecolors='none')


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
        _, ra_centre, dec_centre, _ = XY2CorrectedRADec([ff_middle_time], [self.platepar.X_res/2], 
            [self.platepar.Y_res/2], [0], self.UT_corr, self.platepar.lat, self.platepar.lon, 
            self.platepar.Ho, self.platepar.X_res, self.platepar.Y_res, self.platepar.RA_d, 
            self.platepar.dec_d, self.platepar.pos_angle_ref, self.platepar.F_scale, self.platepar.w_pix, 
            self.platepar.mag_0, self.platepar.mag_lev, self.platepar.x_poly, self.platepar.y_poly)

        # Calculate the FOV radius in degrees
        fov_x = (self.platepar.X_res/2)*(3600/self.platepar.F_scale)*(384/self.platepar.X_res)/3600
        fov_y = (self.platepar.Y_res/2)*(3600/self.platepar.F_scale)*(288/self.platepar.Y_res)/3600

        fov_radius = np.sqrt(fov_x**2 + fov_y**2)

        filtered_catalog_stars = []
        filtered_indices = []

        # Take only those catalog stars which should be inside the FOV
        for i, (ra, dec, _) in enumerate(catalog_stars):

            # Calculate angular separation between the FOV centre and the catalog star
            ang_sep = math.degrees(math.acos(math.sin(math.radians(dec))*math.sin(math.radians(dec_centre)) \
                + math.cos(math.radians(dec))*math.cos(math.radians(dec_centre))*math.cos(math.radians(ra) - math.radians(ra_centre))))

            if ang_sep <= fov_radius:

                # Add stars which are roughly inside the FOV to the OK list
                filtered_catalog_stars.append(catalog_stars[i])
                filtered_indices.append(i)


        return filtered_indices, np.array(filtered_catalog_stars)



    def calcRefCentre(self, JD, lon, lat, ra_ref, dec_ref):
        """ Calculate the referent azimuth and altitude of the centre of the FOV from the given RA/Dec. 

        Arguments:
            JD: [float] Referent Julian date.
            lon: [float] Longitude +E in degrees.
            lat: [float] Latitude +N in degrees.
            ra_ref: [float] Referent RA at referent time in degrees.
            dec_ref: [float] Referent declination at referent time in degrees.
        """

        T = (JD - 2451545)/36525.0
        Ho = (280.46061837 + 360.98564736629*(JD - 2451545) + 0.000387933*T**2 - (T**3)/38710000)%360

        h = Ho + lon - ra_ref
        sh = math.sin(math.radians(h))
        sd = math.sin(math.radians(dec_ref))
        sl = math.sin(math.radians(lat))
        ch = math.cos(math.radians(h))
        cd = math.cos(math.radians(dec_ref))
        cl = math.cos(math.radians(lat))
        x = -ch*cd*sl + sd*cl
        y = -sh*cd
        z = ch*cd*cl + sd*sl
        r = math.sqrt(x**2 + y**2)

        az_centre = (math.degrees(math.atan2(y, x)))%360
        alt_centre = math.degrees(math.atan2(z, r))

        return az_centre, alt_centre



    def getCatalogStarPositions(self, catalog_stars, lon, lat, az_centre, alt_centre, pos_angle_ref, F_scale, 
        x_poly, y_poly):
        """ Draw catalog stars using the current platepar values. 
    
        Arguments:
            catalog_stars: [2D list] A list of (ra, dec, mag) pairs of catalog stars.
            lon: [float] Longitude in degrees.
            lat: [float] Latitude in degrees.
            az_centre: [float] Azimuth of the FOV centre at referent time in degrees.
            alt_centre: [float] Altitude of the FOV centre at referent time in degrees.
            pos_angle_ref: [float] Referent position angle in degrees.
            F_scale: [float] Image scale in pix/arcsec for CIF resolution.
            x_poly: [ndarray float] Distorsion polynomial in X direction.
            y_poly: [ndarray float] Distorsion polynomail in Y direction.
        """

        ra_catalog, dec_catalog, _ = catalog_stars.T

        ff_middle_time = getMiddleTimeFF(self.current_ff_file, self.config.fps, ret_milliseconds=True)

        # Get the date of the middle of the FF exposure
        jd = date2JD(*ff_middle_time, UT_corr=self.UT_corr)

        T = (jd - 2451545)/36525.0
        Ho = 280.46061837 + 360.98564736629*(jd - 2451545) + 0.000387933*T**2 - (T**3)/38710000.0

        sl = math.sin(math.radians(lat))
        cl = math.cos(math.radians(lat))

        salt = math.sin(math.radians(alt_centre))
        saz = math.sin(math.radians(az_centre))
        calt = math.cos(math.radians(alt_centre))
        caz = math.cos(math.radians(az_centre))
        x = -saz*calt
        y = -caz*sl*calt + salt*cl
        HA = math.degrees(math.atan2(x, y))

        # Centre of FOV
        RA_centre = (Ho + lon - HA)%360
        dec_centre = math.degrees(math.asin(sl*salt + cl*calt*caz))

        x_array = np.zeros_like(ra_catalog)
        y_array = np.zeros_like(ra_catalog)

        for i, (ra_star, dec_star) in enumerate(zip(ra_catalog, dec_catalog)):

            # Gnomonization of star coordinates to image coordinates
            ra1 = math.radians(RA_centre)
            dec1 = math.radians(dec_centre)
            ra2 = math.radians(ra_star)
            dec2 = math.radians(dec_star)
            ad = math.acos(math.sin(dec1)*math.sin(dec2) + math.cos(dec1)*math.cos(dec2)*math.cos(ra2 - ra1))
            radius = math.degrees(ad)
            sinA = math.cos(dec2)*math.sin(ra2 - ra1)/math.sin(ad)
            cosA = (math.sin(dec2) - math.sin(dec1)*math.cos(ad))/(math.cos(dec1) * math.sin(ad))
            theta = -math.degrees(math.atan2(sinA, cosA))
            theta = theta + pos_angle_ref - 90.0

            # Calculate the image coordinates (scale the F_scale from CIF resolution)
            X1 = radius*math.cos(math.radians(theta))*F_scale
            Y1 = radius*math.sin(math.radians(theta))*F_scale

            # Calculate distortion in X direction
            dX = (x_poly[0]
                + x_poly[1]*X1
                + x_poly[2]*Y1
                + x_poly[3]*X1**2
                + x_poly[4]*X1*Y1
                + x_poly[5]*Y1**2
                + x_poly[6]*X1**3
                + x_poly[7]*X1**2*Y1
                + x_poly[8]*X1*Y1**2
                + x_poly[9]*Y1**3
                + x_poly[10]*X1*np.sqrt(X1**2 + Y1**2)
                + x_poly[11]*Y1*np.sqrt(X1**2 + Y1**2))

            # Add the distortion correction and calculate X image coordinates
            Xpix = (X1 - dX)*self.platepar.X_res/384.0 + self.platepar.X_res/2

            # Calculate distortion in Y direction
            dY = (y_poly[0]
                + y_poly[1]*X1
                + y_poly[2]*Y1
                + y_poly[3]*X1**2
                + y_poly[4]*X1*Y1
                + y_poly[5]*Y1**2
                + y_poly[6]*X1**3
                + y_poly[7]*X1**2*Y1
                + y_poly[8]*X1*Y1**2
                + y_poly[9]*Y1**3
                + y_poly[10]*Y1*np.sqrt(X1**2 + Y1**2)
                + y_poly[11]*X1*np.sqrt(X1**2 + Y1**2))

            # Add the distortion correction and calculate Y image coordinates
            Ypix = (Y1 - dY)*self.platepar.Y_res/288.0 + self.platepar.Y_res/2

            x_array[i] = Xpix
            y_array[i] = Ypix


        return x_array, y_array



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
        self.platepar.JD = date2JD(*ff_middle_time, UT_corr=self.UT_corr)

        
        time_data = [ff_middle_time]

        print(self.azim_centre, self.alt_centre)
        print(time_data)

        # Convert FOV centre to RA, Dec
        _, ra_data, dec_data = altAz2RADec(self.platepar.lat, self.platepar.lon, self.UT_corr, time_data, 
            [self.azim_centre], [self.alt_centre])

        return ra_data[0], dec_data[0]



    def loadPlatepar(self):
        """ Open a file dialog and ask user to open the platepar file. """

        root = tkinter.Tk()
        root.withdraw()

        platepar = Platepar()

        # Load the platepar file
        platepar_file = filedialog.askopenfilename(initialdir=self.dir_path, \
            title='Select the platepar file')

        root.destroy()

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

        # Update the location from the config file
        self.platepar.lat = self.config.latitude
        self.platepar.lon = self.config.longitude
        self.platepar.elev = self.config.elevation

        # Update image resolution from config
        self.platepar.X_res = self.config.width
        self.platepar.Y_res = self.config.height

        # Estimate the scale
        scale_x = self.config.fov_w/self.config.width*(self.config.width/384)
        scale_y = self.config.fov_h/self.config.height*(self.config.height/288)
        self.platepar.F_scale = 1/((scale_x + scale_y)/2)

        # Set distorsion polynomials to zero
        self.platepar.x_poly *= 0
        self.platepar.y_poly *= 0

        # Set station ID
        self.platepar.station_code = self.config.stationID

        if update_image:
            self.updateImage()



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




    def centroidStar(self):
        """ Find the centroid of the star clicked on the image. """

        ### Extract part of image around the mouse cursor ###
        ######################################################################################################

        # Outer circle radius
        outer_radius = self.star_aperature_radius*2

        x_min = int(round(self.mouse_x - outer_radius))
        if x_min < 0: x_min = 0

        x_max = int(round(self.mouse_x + outer_radius))
        if x_max > self.current_ff.ncols - 1:
            x_max > self.current_ff.ncols - 1

        y_min = int(round(self.mouse_y - outer_radius))
        if y_min < 0: y_min = 0

        y_max = int(round(self.mouse_y + outer_radius))
        if y_max > self.current_ff.nrows - 1:
            y_max > self.current_ff.nrows - 1

        img_crop = self.current_ff.maxpixel[y_min:y_max, x_min:x_max]

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

            # Calculate the centre of FOV
            az_centre, alt_centre = self.calcRefCentre(self.platepar.JD, self.platepar.lon, self.platepar.lat, 
                ra_ref, dec_ref)

            img_x, img_y = img_stars.T

            # Get image coordinates of catalog stars
            catalog_x, catalog_y = self.getCatalogStarPositions(catalog_stars, self.platepar.lon, 
                self.platepar.lat, az_centre, alt_centre, pos_angle_ref, F_scale, self.platepar.x_poly, 
                self.platepar.y_poly)


            dist_sum = 0

            # Calculate the sum of squared distances between image stars and catalog stars
            for i in range(len(catalog_x)):
                dist_sum += (catalog_x[i] - img_x[i])**2 + (catalog_y[i] - img_y[i])**2


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

            else:
                x_poly = np.zeros(12)
                y_poly = params

            # Calculate the centre of FOV
            az_centre, alt_centre = self.calcRefCentre(self.platepar.JD, self.platepar.lon, self.platepar.lat, 
                self.platepar.RA_d, self.platepar.dec_d)

            img_x, img_y = img_stars.T

            # Get image coordinates of catalog stars
            catalog_x, catalog_y = self.getCatalogStarPositions(catalog_stars, self.platepar.lon, 
                self.platepar.lat, az_centre, alt_centre, self.platepar.pos_angle_ref, self.platepar.F_scale, 
                x_poly, y_poly)


            dist_sum = 0

            # Calculate the sum of squared distances between image stars and catalog stars
            for i in range(len(catalog_x)):
                dist_sum += (catalog_x[i] - img_x[i])**2 + (catalog_y[i] - img_y[i])**2


            return dist_sum



        # Extract paired catalog stars and image coordinates separately
        catalog_stars = np.array([cat_coords for img_coords, cat_coords in self.paired_stars])
        img_stars = np.array([img_coords for img_coords, cat_coords in self.paired_stars])

        print(catalog_stars)
        print(img_stars)

        print('ASTRO', _calcImageResidualsAstro([self.platepar.RA_d, self.platepar.dec_d, 
            self.platepar.pos_angle_ref, self.platepar.F_scale], self, catalog_stars, img_stars))

        print('DIS_X', _calcImageResidualsDistorsion(self.platepar.x_poly, self, catalog_stars, img_stars, 'x'))



        # Initial parameters for the astrometric fit
        p0 = [self.platepar.RA_d, self.platepar.dec_d, self.platepar.pos_angle_ref, self.platepar.F_scale]

        # Fit the astrometric parameters
        res = scipy.optimize.minimize(_calcImageResidualsAstro, p0, args=(self, catalog_stars, img_stars),
            method='Nelder-Mead')

        print(res)

        # Update fitted astrometric parameters
        self.platepar.RA_d, self.platepar.dec_d, self.platepar.pos_angle_ref, self.platepar.F_scale = res.x

        # Recalculate centre
        self.calcRefCentre(self.platepar.JD, self.platepar.lon, self.platepar.lat, self.platepar.RA_d, 
            self.platepar.dec_d)


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

        self.updateImage()



if __name__ == '__main__':


    if len(sys.argv) < 2:
        print('Usage: python -m RMS.Astrometry.PlateTool /path/to/FRbin/dir/')
        sys.exit()

    dir_path = sys.argv[1].replace('"', '')


    # Load the configuration file
    config = cr.parse(".config")

    plate_tool = PlateTool(dir_path, config)

    plt.tight_layout()
    plt.show()