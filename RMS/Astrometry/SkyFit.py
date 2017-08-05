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

        plt.figure(facecolor='black')

        # Init the first image
        self.updateImage()

        self.ax = plt.gca()

        # Set the bacground color to black
        #plt.figure(facecolor='k')
        matplotlib.rcParams['axes.facecolor'] = 'k'

        # Disable standard matplotlib keyboard shortcuts
        plt.rcParams['keymap.save'] = ''
        plt.rcParams['keymap.fullscreen'] = ''
        plt.rcParams['keymap.all_axes'] = ''

        
        # self.ax.figure.canvas.mpl_connect('button_press_event', self.onMousePress)
        # self.ax.figure.canvas.mpl_connect('button_release_event', self.onMouseRelease)
        # self.ax.figure.canvas.mpl_connect('motion_notify_event', self.onMouseMotion)

        # Register which mouse/keyboard events will evoke which function
        self.ax.figure.canvas.mpl_connect('scroll_event', self.onScroll)
        self.scroll_counter = 0

        self.ax.figure.canvas.mpl_connect('key_press_event', self.onKeyPress)
        


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
            self.platepar.rot_param -= self.key_increment
            self.updateImage()

        elif event.key == 'e':
            self.platepar.rot_param += self.key_increment
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


        # Limit values of RA and Dec
        self.platepar.RA_d = self.platepar.RA_d%360

        if self.platepar.dec_d > 90:
            self.platepar.dec_d = 90

        elif self.platepar.dec_d < -90:
            self.platepar.dec_d = -90






    def onScroll(self, event):
        """ Switch images on scroll. """

        self.scroll_counter += event.step

        print('blah', event.step)

        if self.scroll_counter > 1:
            self.nextFF()
            self.scroll_counter = 0

        elif self.scroll_counter < -1:
            self.prevFF()
            self.scroll_counter = 0


    def loadCatalogStars(self, lim_mag):
        """ Loads stars from the BSC star catalog. 
    
        Arguments:
            lim_mag: [float] Limiting magnitude of catalog stars.

        """

        # Load catalog stars
        catalog_stars = BSC.readBSC(self.config.star_catalog_path, self.config.star_catalog_file, 
            lim_mag=lim_mag)

        return catalog_stars



    def updateImage(self):
        """ Update the matplotlib plot to show the current image. """

        # Limit key increment so it can be lower than 0.1
        if self.key_increment < 0.1:
            self.key_increment = 0.1


        plt.clf()

        # Load the FF from the current file
        self.current_ff = readFF(self.dir_path, self.current_ff_file)

        # Show the loaded maxpixel
        plt.imshow(self.current_ff.maxpixel, cmap='gray')

        # Draw stars detected on this image
        self.drawCalstars()

        # Draw catalog stars on the image using the current platepar
        self.drawCatalogStars()

        # Set plot limits
        plt.xlim([0, self.current_ff.ncols])
        plt.ylim([self.current_ff.nrows, 0])

        # Show text on image with platepar parameters
        text_str  = 'RA  = {:.3f}\n'.format(self.platepar.RA_d)
        text_str += 'Dec = {:.3f}\n'.format(self.platepar.dec_d)
        text_str += 'PA  = {:.3f}\n'.format(self.platepar.rot_param)
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



    def drawCatalogStars(self):
        """ Draw catalog stars using the current platepar values. """

        ra_catalog, dec_catalog, _ = self.catalog_stars.T

        ff_middle_time = getMiddleTimeFF(self.current_ff_file, self.config.fps, ret_milliseconds=True)

        # Get the date of the middle of the FF exposure
        jd = date2JD(*ff_middle_time, UT_corr=self.UT_corr)

        # Convert values of catalog stars from RA/Dec to image coordinates using the platepar
        x_array, y_array = raDecToXY(ra_catalog, dec_catalog, self.platepar.RA_d, self.platepar.dec_d, jd, self.platepar.JD,
            self.platepar.rot_param, self.platepar.F_scale)

        # Apply the field correction
        x_array, y_array, _ = applyFieldCorrection(self.platepar.x_poly, self.platepar.y_poly, 
            self.platepar.X_res, self.platepar.Y_res, self.platepar.F_scale, x_array, y_array, 
            np.zeros_like(x_array))

        cat_stars = np.c_[x_array, y_array]

        # Take only those stars inside the FOV
        cat_stars = cat_stars[cat_stars[:, 0] > 0]
        cat_stars = cat_stars[cat_stars[:, 0] < self.current_ff.ncols]
        cat_stars = cat_stars[cat_stars[:, 1] > 0]
        cat_stars = cat_stars[cat_stars[:, 1] < self.current_ff.nrows]

        x_array, y_array = cat_stars.T

        plt.scatter(x_array, y_array, c='r', marker='+')



    def getFOVcentre(self):
        """ Asks the user to input the centre of the FOV in altitude and azimuth. """

        # Get FOV centre
        root = tkinter.Tk()
        root.withdraw()
        d = FOVinputDialog(root)
        root.wait_window(d.top)
        self.azim_centre, self.alt_centre = d.getAltAz()

        root.destroy()

        # Get the time of the current FF file
        time_data = [getMiddleTimeFF(self.current_ff_file, self.config.fps, ret_milliseconds=True)]

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