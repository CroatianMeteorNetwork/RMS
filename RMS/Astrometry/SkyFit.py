""" GUI tool for making initial plate estimations and manually fitting astrometric plates. """

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

import matplotlib
import matplotlib.pyplot as plt

import RMS.Formats.BSC as BSC
import RMS.Formats.CALSTARS as CALSTARS
from RMS.Formats.Platepar import PlateparCMN
from RMS.Formats.FFbin import read as readFF
from RMS.Formats.FFbin import validName as validFFName
import RMS.ConfigReader as cr
from RMS.Astrometry.ApplyAstrometry import raDecToXY, altAz2RADec, applyFieldCorrection
from RMS.Astrometry.AstrometryCheckFit import getMiddleTimeFF
from RMS.Astrometry.Conversions import date2JD


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

        # List of paired image and catalog stars
        self.paired_stars = []

        # Positions of the mouse cursor
        self.mouse_x = 0
        self.mouse_y = 0

        # Kwy increment
        self.key_increment = 1.0

        # Time difference from UT
        self.UT_corr = 0

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


        if self.star_selection_centroid:

            # Centroid the star around the pressed coordinates
            self.x_centroid, self.y_centroid = self.centroidStar()

            # Draw the centroid on the image
            plt.scatter(self.x_centroid, self.y_centroid, marker='+', c='y', s=50)

            
            # Select the closest catalog star to the centroid as the first guess
            self.closest_cat_star_indx = self.findClosestCatalogStarIndex(self.x_centroid, self.y_centroid)

            # Plot the closest star as a purple cross
            self.selected_cat_star_scatter = plt.scatter(self.catalog_x[self.closest_cat_star_indx], 
                self.catalog_y[self.closest_cat_star_indx], marker='+', c='purple', s=50)

            # Update canvas
            plt.gcf().canvas.draw()

            print(self.closest_cat_star_indx)
            print(self.catalog_x[self.closest_cat_star_indx], 
                self.catalog_y[self.closest_cat_star_indx], self.catalog_stars[self.closest_cat_star_indx])


            # Switch to the mode where the catalog star is selected
            self.star_selection_centroid = False


        else:

            # Select the closest catalog star
            self.closest_cat_star_indx = self.findClosestCatalogStarIndex(self.mouse_x, self.mouse_y)

            if self.selected_cat_star_scatter is not None:
                self.selected_cat_star_scatter.remove()

            # Plot the closest star as a purple cross
            self.selected_cat_star_scatter = plt.scatter(self.catalog_x[self.closest_cat_star_indx], 
                self.catalog_y[self.closest_cat_star_indx], marker='+', c='purple', s=50)

            # Update canvas
            plt.gcf().canvas.draw()

            print(self.closest_cat_star_indx)
            print(self.catalog_x[self.closest_cat_star_indx], 
                self.catalog_y[self.closest_cat_star_indx], self.catalog_stars[self.closest_cat_star_indx])



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
            self.platepar.write(self.platepar_file)
            print('Platepar written to:', self.platepar_file)


        # Create a new platepar
        elif event.key == 'ctrl+n':
            self.makeNewPlatepar()


        # Change modes from astrometry parameter changing to star picking
        elif event.key == 'ctrl+r':

            if self.star_pick_mode:
                self.disableStarPicking()
                self.circle_aperature = None
                self.circle_aperature_outer = None
                self.updateImage()

            else:
                self.enableStarPicking()
                self.paired_stars = []



        elif event.key == 'enter':

            if self.star_pick_mode:

                # If the right catalog star has been selected, save the pair to the list
                if not self.star_selection_centroid:

                    # Add the image/catalog pair to the list
                    self.paired_stars.append([[self.x_centroid, self.y_centroid], 
                        self.catalog_stars[self.closest_cat_star_indx]])

                    # Switch back to centroiding mode
                    self.star_selection_centroid = True

                    print(self.paired_stars)

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

        # Show the loaded maxpixel
        plt.imshow(self.current_ff.maxpixel, cmap='gray')


        # Draw stars detected on this image
        self.drawCalstars()

        # Update centre of FOV in horizontal coordinates
        self.calcRefCentre()

        ### Draw catalog stars on the image using the current platepar ###
        ######################################################################################################
        self.catalog_x, self.catalog_y = self.getCatalogStars()

        cat_stars = np.c_[self.catalog_x, self.catalog_y]

        # Take only those stars inside the FOV
        cat_stars = cat_stars[cat_stars[:, 0] > 0]
        cat_stars = cat_stars[cat_stars[:, 0] < self.current_ff.ncols]
        cat_stars = cat_stars[cat_stars[:, 1] > 0]
        cat_stars = cat_stars[cat_stars[:, 1] < self.current_ff.nrows]

        catalog_x_filtered, catalog_y_filtered = cat_stars.T

        plt.scatter(catalog_x_filtered, catalog_y_filtered, c='r', marker='+')

        ######################################################################################################


        # Set plot limits
        plt.xlim([0, self.current_ff.ncols])
        plt.ylim([self.current_ff.nrows, 0])

        # Show text on image with platepar parameters
        text_str  = self.current_ff_file + '\n\n'
        text_str += 'RA  = {:.3f}\n'.format(self.platepar.RA_d)
        text_str += 'Dec = {:.3f}\n'.format(self.platepar.dec_d)
        text_str += 'PA  = {:.3f}\n'.format(self.platepar.pos_angle_ref)
        text_str += 'F_scale = {:.3f}\n'.format(self.platepar.F_scale)
        text_str += 'Lim mag = {:.1f}\n'.format(self.cat_lim_mag)
        text_str += 'Increment = {:.3}\n'.format(self.key_increment)
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
        text_str += 'FOV centre - V\n'
        text_str += 'New platepar - CTRL + N\n'
        text_str += 'Save platepar - CTRL + S\n'
        plt.gca().text(10, self.current_ff.nrows - 10, text_str, color='w', verticalalignment='bottom', 
            horizontalalignment='left', fontsize=8)

        plt.gcf().canvas.draw()



    def drawCalstars(self):
        """ Draw extracted stars on the current image. """

        # Get the stars detected on this FF file
        star_data = self.calstars[self.current_ff_file]

        # Get star coordinates
        y, x, _, _ = np.array(star_data).T

        plt.scatter(x, y, edgecolors='g', marker='o', facecolors='none')



    def calcRefCentre(self):
        """ Calculate the referent azimuth and altitude of the centre of the FOV from the given RA/Dec. """

        # Julian date is the one of the referent date and time
        JD = self.platepar.JD

        T = (JD - 2451545)/36525
        Ho = (280.46061837 + 360.98564736629*(JD - 2451545) + 0.000387933*T**2 - (T**3)/38710000)%360

        h = Ho + self.platepar.lon - self.platepar.RA_d
        sh = math.sin(math.radians(h))
        sd = math.sin(math.radians(self.platepar.dec_d))
        sl = math.sin(math.radians(self.platepar.lat))
        ch = math.cos(math.radians(h))
        cd = math.cos(math.radians(self.platepar.dec_d))
        cl = math.cos(math.radians(self.platepar.lat))
        x = -ch*cd*sl + sd*cl
        y = -sh*cd
        z = ch*cd*cl + sd*sl
        r = math.sqrt(x**2 + y**2)

        self.platepar.az_centre = (math.degrees(math.atan2(y, x)))%360
        self.platepar.alt_centre = math.degrees(math.atan2(z, r))



    def getCatalogStars(self):
        """ Draw catalog stars using the current platepar values. """

        ra_catalog, dec_catalog, _ = self.catalog_stars.T

        ff_middle_time = getMiddleTimeFF(self.current_ff_file, self.config.fps, ret_milliseconds=True)

        # Get the date of the middle of the FF exposure
        jd = date2JD(*ff_middle_time, UT_corr=self.UT_corr)

        T = (jd - 2451545)/36525
        Ho = 280.46061837 + 360.98564736629*(jd - 2451545) + 0.000387933*T**2 - (T**3)/38710000

        sl = math.sin(math.radians(self.platepar.lat))
        cl = math.cos(math.radians(self.platepar.lat))

        salt = math.sin(math.radians(self.platepar.alt_centre))
        saz = math.sin(math.radians(self.platepar.az_centre))
        calt = math.cos(math.radians(self.platepar.alt_centre))
        caz = math.cos(math.radians(self.platepar.az_centre))
        x = -saz*calt
        y = -caz*sl*calt + salt*cl
        HA = math.degrees(math.atan2(x, y))

        # Centre of FOV
        RA_centre = (Ho + self.platepar.lon - HA)%360
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
            theta = theta + self.platepar.pos_angle_ref - 90

            # Calculate the image coordinates (scale the F_scale from CIF resolution)
            x = radius*math.cos(math.radians(theta))*self.platepar.F_scale*(self.platepar.X_res/384)
            y = radius*math.sin(math.radians(theta))*self.platepar.F_scale*(self.platepar.Y_res/288)

            X1 = x
            Y1 = y
            delta_XY = 1

            # Extract distortion poynomials
            x_poly = self.platepar.x_poly
            y_poly = self.platepar.y_poly

            # while (delta_XY > 0.1):
            #     dX = (x_poly[0] + x_poly[1]*X1 + x_poly[2]*Y1 + x_poly[3]*X1**2 + x_poly[4]*X1*Y1 + x_poly[5]*Y1**2 + x_poly[6]*X1**3 + x_poly[7]*X1*X1*Y1 + x_poly[8]*X1*Y1**2 + x_poly[9]*Y1**3 + x_poly[10]*X1*math.sqrt(X1*X1 + Y1*Y1) + x_poly[11]*Y1*math.sqrt(X1*X1 + Y1*Y1))
            #     #dY = (P.Y1 + P.Y2 * X1 + P.Y3 * Y1 + P.Y4 * X1 * X1 + P.Y5 * X1 * Y1 + P.Y6 * Y1 * Y1 + P.Y7 * X1 * X1 * X1 + P.Y8 * X1 * X1 * Y1 + P.Y9 * X1 * Y1 * Y1 + P.Y10 * Y1 * Y1 * Y1 + P.Y11 * Y1 * Sqrt(X1 * X1 + Y1 * Y1) + P.Y12 * X1 * Sqrt(X1 * X1 + Y1 * Y1))
            #     dY = (y_poly[0] + y_poly[1]*X1 + y_poly[2]*Y1 + y_poly[3]*X1**2 + y_poly[4]*X1*Y1 + y_poly[5]*Y1**2 + y_poly[6]*X1**3 + y_poly[7]*X1*X1*Y1 + y_poly[8]*X1*Y1**2 + y_poly[9]*Y1**3 + y_poly[10]*X1*math.sqrt(X1*X1 + Y1*Y1) + y_poly[11]*Y1*math.sqrt(X1*X1 + Y1*Y1))
            #     delta_xX = X1 - x + dX
            #     delta_yY = Y1 - y + dY
            #     delta_XY = math.sqrt(delta_xX*delta_xX + delta_yY*delta_yY)
            #     X1 = x - dX
            #     Y1 = y - dY

            # X1 = X1 + 192
            # Y1 = Y1 + 144

            X1 = X1 + self.platepar.X_res/2
            Y1 = Y1 + self.platepar.Y_res/2

            x_array[i] = X1
            y_array[i] = Y1


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

        # Load the platepar file
        platepar_file = filedialog.askopenfilename(initialdir=self.dir_path, \
            title='Select the platepar file')

        print(platepar_file)

        # Parse the platepar file
        try:
            platepar = PlateparCMN()
            platepar.read(platepar_file)
        except:
            platepar = False

        # Check if the platepar was successfuly loaded
        if not platepar:
            messagebox.showerror(title='Platepar file error', message='The file you selected could not be loaded as a platepar file!')
            
            self.loadPlatepar()

        root.destroy()

        return platepar_file, platepar



    def makeNewPlatepar(self):
        """ Make a new platepar from the loaded one, but set the parameters from the config file. """

        # Update the location from the config file
        self.platepar.lat = self.config.latitude
        self.platepar.lon = self.config.longitude
        self.platepar.elev = self.config.elevation

        # Update image resolution from config
        self.platepar.X_res = self.config.width
        self.platepar.Y_res = self.config.height

        # Set distorsion polynomials to zero
        self.platepar.x_poly *= 0
        self.platepar.y_poly *= 0

        # Set station ID
        self.platepar.station_code = self.config.stationID

        self.updateImage()



    def nextFF(self):
        """ Shows the next FF file in the list. """

        self.current_ff_index = (self.current_ff_index + 1)%len(self.ff_list)
        self.current_ff_file = self.ff_list[self.current_ff_index]

        self.updateImage()



    def prevFF(self):
        """ Shows the previous FF file in the list. """

        self.current_ff_index = (self.current_ff_index - 1)%len(self.ff_list)
        self.current_ff_file = self.ff_list[self.current_ff_index]

        self.updateImage()



    def enableStarPicking(self):
        """ Enable the star picking mode where the star are manually selected for the fit. """

        self.star_pick_mode = True




    def disableStarPicking(self):
        """ Disable the star picking mode where the star are manually selected for the fit. """

        self.star_pick_mode = False




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

        return x_centroid, y_centroid



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