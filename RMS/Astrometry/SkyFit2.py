from __future__ import print_function, division, absolute_import, unicode_literals

import os
import sys
import math
import copy
import time
import datetime
import argparse
import traceback
import copy
import cProfile
import pstats

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.ndimage

try:
    import tkinter
    from tkinter import messagebox
except:
    import Tkinter as tkinter
    import tkMessageBox as messagebox

from pyqtgraph import Qt
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui

from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP, raDecToXYPP, \
    rotationWrtHorizon, rotationWrtHorizonToPosAngle, computeFOVSize, photomLine, photometryFit, \
    rotationWrtStandard, rotationWrtStandardToPosAngle, correctVignetting, extinctionCorrectionTrueToApparent
from RMS.Astrometry.Conversions import J2000_JD, date2JD, JD2HourAngle, altAz2RADec, trueRaDec2ApparentAltAz, \
    apparentAltAz2TrueRADec
from RMS.Astrometry.AstrometryNetNova import novaAstrometryNetSolve
import RMS.ConfigReader as cr
import RMS.Formats.CALSTARS as CALSTARS
from RMS.Formats.Platepar import Platepar, getCatalogStarsImagePositions
from RMS.Formats.FrameInterface import detectInputType
from RMS.Formats import StarCatalog
from RMS.Pickling import loadPickle, savePickle
from RMS.Routines import Image
from RMS.Math import angularSeparation
from RMS.Misc import decimalDegreesToSexHours, openFileDialog
from RMS.Routines.AddCelestialGrid import updateRaDecGrid, updateAzAltGrid

import pyximport

pyximport.install(setup_args={'include_dirs': [np.get_include()]})
from RMS.Astrometry.CyFunctions import subsetCatalog  # , eqRefractionTrueToApparent, eqRefractionApparentToTrue

from RMS.Astrometry.CustomPyqtgraphClasses import *


class FOVinputDialog(object):
    """ Dialog for inputting FOV centre in Alt/Az. """

    def __init__(self, parent):

        self.parent = parent

        # Set initial angle values
        self.azim = self.alt = self.rot = 0

        self.top = tkinter.Toplevel(parent)

        # Bind the Enter key to run the verify function
        self.top.bind('<Return>', self.verify)
        self.top.protocol("WM_DELETE_WINDOW", self.cancel)

        tkinter.Label(self.top, text="FOV centre (degrees) \nAzim +E of due N\nRotation from vertical").grid(row=0,
                                                                                                             columnspan=2)

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
        b.grid(row=4, column=0, columnspan=1)

        b2 = tkinter.Button(self.top, text="cancel", command=self.cancel)
        b2.grid(row=4, column=1, columnspan=1)

    def cancel(self, event=None):
        self.azim = None
        self.alt = None
        self.rot = None
        self.top.destroy()

    def verify(self, event=None):
        """ Check that the azimuth and altitude are withing the bounds. """

        try:
            # Read values
            self.azim = float(self.azimuth.get())%360
            self.alt = float(self.altitude.get())
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


class PlateTool(QtWidgets.QMainWindow):
    def __init__(self, dir_path, config, beginning_time=None, fps=None, gamma=None, startUI=True):
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
        super(PlateTool, self).__init__()

        self.config = config
        self.dir_path = dir_path

        # If camera gamma was given, change the value in config
        if gamma is not None:
            config.gamma = gamma

        # Flag which regulates wheter the maxpixel or the avepixel image is shown (avepixel by default)
        self.img_type_flag = 'avepixel'

        # Star picking mode=
        self.star_selection_centroid = True
        self.star_aperature_radius = 5
        self.x_centroid = self.y_centroid = None
        self.closest_cat_star_indx = None

        # List of paired image and catalog stars
        self.paired_stars = []
        self.residuals = None

        # Positions of the mouse cursor
        self.mouse_x = 0
        self.mouse_y = 0

        # Position of mouse cursor when it was last pressed
        self.mouse_press_x = 0
        self.mouse_press_y = 0

        # Kwy increment
        self.key_increment = 1.0

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
                                                                     + os.path.join(self.config.star_catalog_path,
                                                                                    self.config.star_catalog_file) \
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
                messagebox.showinfo(title='CALSTARS error',
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
            self.makeNewPlatepar()

            # Create the name of the platepar file
            self.platepar_file = os.path.join(self.dir_path, self.config.platepar_name)

        # Set the given gamma value to platepar
        if gamma is not None:
            self.platepar.gamma = gamma

        # Load distorion type index
        self.dist_type_count = len(self.platepar.distortion_type_list)
        self.dist_type_index = self.platepar.distortion_type_list.index(self.platepar.distortion_type)

        # SETUP WINDOW
        if startUI:
            self.setupUI()

    def setupUI(self, loaded_file=False):
        """ Setup pyqt UI with widgets """
        self.central = QtWidgets.QWidget()
        self.setWindowTitle('SkyFit')
        self.setCentralWidget(self.central)

        layout = QtWidgets.QGridLayout()
        self.central.setLayout(layout)

        self.show_key_help = 1

        # timing
        self.i = 0
        self.n = 100
        self.frames = np.zeros(self.n)

        # Image on left
        self.view_widget = pg.GraphicsView()
        self.img_frame = ViewBox()
        self.img_frame.setAspectLocked()
        self.img_frame.setMenuEnabled(False)
        self.view_widget.setCentralWidget(self.img_frame)
        self.img_frame.invertY()
        layout.addWidget(self.view_widget, 0, 1)

        # zoom window
        self.show_zoom_window = False
        self.show_zoom_window_size = 200
        self.v_zoom = pg.GraphicsView(self.view_widget)
        self.zoom_window = ViewBox()
        self.zoom_window.setAspectLocked()
        self.zoom_window.setMouseEnabled(False, False)
        self.zoom_window.setMenuEnabled(False)
        self.v_zoom.setFixedWidth(self.show_zoom_window_size)
        self.v_zoom.setFixedHeight(self.show_zoom_window_size)
        self.zoom()
        self.v_zoom.hide()
        self.v_zoom.setCentralItem(self.zoom_window)
        self.v_zoom.move(QtCore.QPoint(0, 0))
        self.v_zoom_left = True  # whether to draw zoom window on left or right
        self.zoom_window.invertY()

        # top left label
        self.label1 = TextItem(color=(0, 0, 0), fill=(255, 255, 255, 100))
        self.label1.setTextWidth(200)
        self.label1.setZValue(1000)
        self.label1.setParentItem(self.img_frame)

        # bottom left label
        self.label2 = TextItem(color=(0, 0, 0), fill=(255, 255, 255, 100))
        self.label2.setTextWidth(200)
        self.label2.setZValue(1000)
        self.label2.setParentItem(self.img_frame)

        # bottom information
        self.label3 = QtWidgets.QLabel()
        layout.addWidget(self.label3, 1, 1)

        self.catalog_stars_visible = True

        # catalog star markers (main window)
        self.cat_star_markers = pg.ScatterPlotItem()
        self.img_frame.addItem(self.cat_star_markers)
        self.cat_star_markers.setPen('r')
        self.cat_star_markers.setBrush((0, 0, 0, 0))
        self.cat_star_markers.setSymbol(Crosshair())
        self.cat_star_markers.setZValue(4)

        # catalog star markers (zoom window)
        self.cat_star_markers2 = pg.ScatterPlotItem()
        self.zoom_window.addItem(self.cat_star_markers2)
        self.cat_star_markers2.setPen('r')
        self.cat_star_markers2.setBrush((0, 0, 0, 0))
        self.cat_star_markers2.setSize(10)
        self.cat_star_markers2.setSymbol(Cross())
        self.cat_star_markers2.setZValue(4)

        # selected catalog star markers (main window)
        self.sel_cat_star_markers = pg.ScatterPlotItem()
        self.img_frame.addItem(self.sel_cat_star_markers)
        self.sel_cat_star_markers.setPen('b')
        self.sel_cat_star_markers.setSize(10)
        self.sel_cat_star_markers.setSymbol(Cross())
        self.sel_cat_star_markers.setZValue(4)

        # selected catalog star markers (zoom window)
        self.sel_cat_star_markers2 = pg.ScatterPlotItem()
        self.zoom_window.addItem(self.sel_cat_star_markers2)
        self.sel_cat_star_markers2.setPen('b')
        self.sel_cat_star_markers2.setSize(10)
        self.sel_cat_star_markers2.setSymbol(Cross())
        self.sel_cat_star_markers2.setZValue(4)

        # centroid star markers (main window)
        self.centroid_star_markers = pg.ScatterPlotItem()
        self.img_frame.addItem(self.centroid_star_markers)
        self.centroid_star_markers.setPen((255, 165, 0))
        self.centroid_star_markers.setSize(15)
        self.centroid_star_markers.setSymbol(Plus())
        self.centroid_star_markers.setZValue(4)

        # centroid star markers (zoom window)
        self.centroid_star_markers2 = pg.ScatterPlotItem()
        self.zoom_window.addItem(self.centroid_star_markers2)
        self.centroid_star_markers2.setPen((255, 165, 0))
        self.centroid_star_markers2.setSize(15)
        self.centroid_star_markers2.setSymbol(Plus())
        self.centroid_star_markers2.setZValue(4)

        self.draw_calstars = True

        # calstar markers (main window)
        self.calstar_markers = pg.ScatterPlotItem()
        self.img_frame.addItem(self.calstar_markers)
        self.calstar_markers.setPen((0, 255, 0, 100))
        self.calstar_markers.setBrush((0, 0, 0, 0))
        self.calstar_markers.setSize(10)
        self.calstar_markers.setSymbol('o')
        self.calstar_markers.setZValue(2)

        # calstar markers (zoom window)
        self.calstar_markers2 = pg.ScatterPlotItem()
        self.zoom_window.addItem(self.calstar_markers2)
        self.calstar_markers2.setPen((0, 255, 0, 100))
        self.calstar_markers2.setBrush((0, 0, 0, 0))
        self.calstar_markers2.setSize(20)
        self.calstar_markers2.setSymbol('o')
        self.calstar_markers2.setZValue(5)
        text_str = "STAR PICKING MODE\n"
        text_str += "'LEFT CLICK' - Centroid star\n"
        text_str += "'CTRL + LEFT CLICK' - Manual star position\n"
        text_str += "'CTRL + Z' - Fit stars\n"
        text_str += "'CTRL + SHIFT + Z' - Fit with initial distortion params set to 0\n"
        text_str += "'L' - Astrometry fit details\n"
        text_str += "'P' - Photometry fit"
        self.star_pick_info = TextItem(text_str, anchor=(0.5, 0.5), color=(255, 255, 255))
        self.star_pick_info.setAlign(QtCore.Qt.AlignCenter)
        self.star_pick_info.hide()
        self.star_pick_info.setZValue(10000)
        self.star_pick_info.setParentItem(self.img_frame)

        # default variables even when constructor isnt called
        self.star_pick_mode = False

        # cursor
        self.cursor = CursorItem(self.star_aperature_radius, pxmode=True)
        self.img_frame.addItem(self.cursor, ignoreBounds=True)
        self.cursor.hide()
        self.cursor.setZValue(20)

        # cursor (window)
        self.cursor2 = CursorItem(self.star_aperature_radius, pxmode=True, thickness=2)
        self.zoom_window.addItem(self.cursor2, ignoreBounds=True)
        self.cursor2.hide()
        self.cursor2.setZValue(20)

        # distortion lines (window)
        self.draw_distortion = False
        self.distortion_lines = pg.PlotCurveItem(connect='pairs', pen=(255, 255, 0, 200))
        self.distortion_lines.hide()
        self.img_frame.addItem(self.distortion_lines)
        self.distortion_lines.setZValue(2)

        self.grid_visible = 1
        self.celestial_grid = pg.PlotCurveItem(pen=pg.mkPen((255, 255, 255, 150), style=QtCore.Qt.DotLine))
        # pr=cProfile.Profile()
        # pr.enable()
        # self.onGridChanged() # called in updatestars
        # pr.disable()
        # ps = pstats.Stats(pr, stream=sys.stdout).sort_stats('time')
        # ps.print_stats()

        self.celestial_grid.setZValue(1)
        self.img_frame.addItem(self.celestial_grid)

        # fit residuals
        self.residual_lines = pg.PlotCurveItem(connect='pairs', pen=pg.mkPen((255, 255, 0),
                                                                             style=QtCore.Qt.DashLine))
        self.img_frame.addItem(self.residual_lines)
        self.residual_lines.setZValue(2)

        # text
        self.stdev_text_filter = 0
        self.residual_text = TextItemList()
        self.img_frame.addItem(self.residual_text)
        self.residual_text.setZValue(10)

        # img_level_min and img_level_max are not updated constantly, it is used for storing levels
        # when transferring between images. Instead call self.hist.getLevels()
        self.img_level_min = self.img_level_min_auto = 0
        self.img_level_max = self.img_level_max_auto = 0  # these will be changed when the image is loaded
        self.bit_depth = self.config.bit_depth  # Image gamma and levels
        self.auto_levels = False
        self.tab = RightOptionsTab(self)

        # adding img
        self.img = None
        self.img_zoom = None

        self.updateImage(first_update=True)
        self.tab.settings.updateInvertColours()
        self.tab.settings.updateImageGamma()
        if loaded_file:
            self.updatePairedStars()

        # make connections to sidebar gui
        self.tab.param_manager.sigElevChanged.connect(self.onExtinctionChanged)
        self.tab.param_manager.sigLocationChanged.connect(self.onAzAltChanged)
        self.tab.param_manager.sigAzAltChanged.connect(self.onAzAltChanged)
        self.tab.param_manager.sigRotChanged.connect(self.onRotChanged)
        self.tab.param_manager.sigScaleChanged.connect(self.onScaleChanged)
        self.tab.param_manager.sigFitParametersChanged.connect(self.onFitParametersChanged)
        self.tab.param_manager.sigExtinctionChanged.connect(self.onExtinctionChanged)

        self.tab.param_manager.sigRefractionToggled.connect(self.onRefractionChanged)
        self.tab.param_manager.sigEqAspectToggled.connect(self.onRefractionChanged)

        self.tab.param_manager.sigFitPressed.connect(lambda: self.fitPickedStars(first_platepar_fit=False))
        self.tab.param_manager.sigPhotometryPressed.connect(lambda: self.photometry(show_plot=True))
        self.tab.param_manager.sigAstrometryPressed.connect(self.showAstrometryFitPlots)

        self.tab.settings.sigMaxAveToggled.connect(self.toggleImageType)
        self.tab.settings.sigCatStarsToggled.connect(self.toggleShowCatStars)
        self.tab.settings.sigCalStarsToggled.connect(self.toggleShowCalStars)
        self.tab.settings.sigDistortionToggled.connect(self.toggleDistortion)
        self.tab.settings.sigGridToggled.connect(self.onGridChanged)
        self.tab.settings.sigInvertToggled.connect(self.toggleInvertColours)

        layout.addWidget(self.tab, 0, 2)

        # mouse binding
        self.img_frame.scene().sigMouseMoved.connect(self.onMouseMoved)
        # NOTE: clicking event doesnt trigger if moving
        self.img_frame.scene().sigMouseClicked.connect(self.onMouseClicked)
        self.img_frame.sigResized.connect(self.onFrameResize)

        self.scrolls_back = 0

        self.setMinimumSize(1200, 800)
        self.show()
        self.updateLeftLabels()
        self.star_pick_info.setPos(self.img_frame.width()/2, self.img_frame.height() - 50)

    def onRefractionChanged(self):
        self.updateStars()
        self.updateLeftLabels()

    def onGridChanged(self):
        if self.grid_visible == 0:
            self.celestial_grid.hide()
        elif self.grid_visible == 1:
            self.celestial_grid.show()
            updateRaDecGrid(self.celestial_grid, self.platepar)
        else:
            self.celestial_grid.show()
            updateAzAltGrid(self.celestial_grid, self.platepar)

    def onScaleChanged(self):
        self.updateFitResiduals()
        self.updateStars()
        self.updateLeftLabels()

    def onFitParametersChanged(self):
        self.updateDistortion()
        self.updateStars()
        self.updateLeftLabels()

    def onExtinctionChanged(self):
        self.photometry()
        self.updateLeftLabels()

    def onAzAltChanged(self):
        self.platepar.updateRefRADec()
        self.updateStars()
        self.updateLeftLabels()

    def onRotChanged(self):
        self.platepar.pos_angle_ref = rotationWrtHorizonToPosAngle(self.platepar,
                                                                   self.platepar.rotation_from_horiz)
        self.updateStars()
        self.updateLeftLabels()

    def mouseOverStatus(self, x, y):
        """ Format the status message which will be printed in the status bar below the plot.

        Arguments:
            x: [float] Plot X coordiante.
            y: [float] Plot Y coordinate.

        Return:
            [str]: formatted output string to be written in the status bar
        """

        # Write image X, Y coordinates and image intensity
        if 0 <= x <= self.img.data.shape[0] - 1 and 0 <= y <= self.img.data.shape[1] - 1:
            status_str = "x={:7.2f}  y={:7.2f}  Intens={:d}".format(x, y, self.img.data[int(x), int(y)])
        else:
            status_str = "x={:7.2f}  y={:7.2f}  Intens=--".format(x, y)

        # Add coordinate info if platepar is present
        if self.platepar is not None:
            # Get the current frame time
            time_data = [self.img_handle.currentTime()]

            # Compute RA, dec
            jd, ra, dec, _ = xyToRaDecPP(time_data, [x], [y], [1], self.platepar, extinction_correction=False)

            # Compute alt, az
            azim, alt = trueRaDec2ApparentAltAz(ra[0], dec[0], jd[0], self.platepar.lat, self.platepar.lon,
                                                self.platepar.refraction)

            status_str += ",  Azim={:6.2f}  Alt={:6.2f} (date), RA={:6.2f}  Dec={:+6.2f} (J2000)".format(
                azim, alt, ra[0], dec[0])

        return status_str

    def updateBottomLabel(self):
        """ Update bottom label with current mouse position """
        self.label3.setText(self.mouseOverStatus(self.mouse_x, self.mouse_y))

    def zoom(self):
        """ Update the zoom window to zoom on the correct position """
        self.zoom_window.autoRange()
        # zoom_scale = 0.1
        # self.zoom_window.scaleBy(zoom_scale, QPoint(*self.mouse))
        self.zoom_window.setXRange(self.mouse_x - 20, self.mouse_x + 20)
        self.zoom_window.setYRange(self.mouse_y - 20, self.mouse_y + 20)

    def updateLeftLabels(self):
        """ Update the two labels on the left with their information """

        ra_centre, dec_centre = self.computeCentreRADec()

        # Show text on image with platepar parameters
        text_str = self.img_handle.name() + '\n' + self.img_type_flag + '\n\n'
        text_str += 'Ref Az   = {:.3f}°\n'.format(self.platepar.az_centre)
        text_str += 'Ref Alt  = {:.3f}°\n'.format(self.platepar.alt_centre)
        text_str += 'Rot horiz = {:.3f}°\n'.format(rotationWrtHorizon(self.platepar))
        text_str += 'Rot eq    = {:.3f}°\n'.format(rotationWrtStandard(self.platepar))
        # text_str += 'Ref RA  = {:.3f}\n'.format(self.platepar.RA_d)
        # text_str += 'Ref Dec = {:.3f}\n'.format(self.platepar.dec_d)
        text_str += "Pix scale = {:.3f}'/px\n".format(60/self.platepar.F_scale)
        text_str += 'Lim mag   = {:.1f}\n'.format(self.cat_lim_mag)
        text_str += 'Increment = {:.2f}\n'.format(self.key_increment)
        text_str += 'Img Gamma = {:.2f}\n'.format(self.img.gamma)
        text_str += 'Camera Gamma = {:.2f}\n'.format(self.config.gamma)
        text_str += "Refraction corr = {:s}\n".format(str(self.platepar.refraction))
        text_str += "Distortion type = {:s}\n".format(
            self.platepar.distortion_type)

        # Add aspect info if the radial distortion is used
        if not self.platepar.distortion_type.startswith("poly"):
            text_str += "Equal aspect    = {:s}\n".format(str(self.platepar.equal_aspect))

        text_str += "Extinction Scale = {:.2f}\n".format(self.platepar.extinction_scale)
        text_str += '\n'
        sign, hh, mm, ss = decimalDegreesToSexHours(ra_centre)
        if sign < 0:
            sign_str = '-'
        else:
            sign_str = ' '
        text_str += 'RA centre  = {:s}{:02d}h {:02d}m {:05.2f}s\n'.format(sign_str, hh, mm, ss)
        text_str += 'Dec centre = {:.3f}°\n'.format(dec_centre)

        self.label1.setText(text_str)

        text_str = 'Keys:\n'
        text_str += '-----\n'
        text_str += 'A/D - Azimuth\n'
        text_str += 'S/W - Altitude\n'
        text_str += 'Q/E - Position angle\n'
        text_str += 'Up/Down - Scale\n'
        text_str += 'T - Toggle refraction correction\n'

        # Add aspect info if the radial distortion is used
        if not self.platepar.distortion_type.startswith("poly"):
            text_str += 'G - Toggle equal aspect\n'
            text_str += 'B - Dist = img centre toggle\n'

        text_str += '1/2 - X offset\n'
        text_str += '3/4 - Y offset\n'
        text_str += '5/6 - X 1st dist. coeff.\n'
        text_str += '7/8 - Y 1st dist. coeff.\n'
        text_str += '9/0 - extinction scale\n'
        text_str += 'CTRL + 1 - poly3+radial distortion\n'
        text_str += 'CTRL + 2 - radial3 distortion\n'
        text_str += 'CTRL + 3 - radial4 distortion\n'
        text_str += 'CTRL + 4 - radial5 distortion\n'
        text_str += '\n'
        text_str += 'R/F - Lim mag\n'
        text_str += '+/- - Increment\n'
        text_str += '\n'
        text_str += 'M - Toggle maxpixel/avepixel\n'
        text_str += 'H - Hide/show catalog stars\n'
        text_str += 'C - Hide/show detected stars\n'
        text_str += 'CTRL + I - Show/hide distortion\n'
        text_str += 'U/J - Img Gamma\n'
        text_str += 'I - Invert colors\n'
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

        self.label2.setText(text_str)
        self.label2.setPos(0, self.img_frame.height() - self.label2.boundingRect().height())

    def updateStars(self):
        """ Updates only the stars """
        # Draw stars that were paired in picking mode
        self.updatePairedStars()
        self.onGridChanged()

        # Draw stars detected on this image
        if self.draw_calstars:
            self.drawCalstars()

        ### Draw catalog stars on the image using the current platepar ###
        ######################################################################################################

        # Get positions of catalog stars on the image
        ff_jd = date2JD(*self.img_handle.currentTime())
        self.catalog_x, self.catalog_y, catalog_mag = getCatalogStarsImagePositions(self.catalog_stars,
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
                size = ((4.0 + (cat_mag_faintest - catalog_mag_filtered))/2.0)**(2*2.512*0.5)

                self.cat_star_markers.setPoints(x=self.catalog_x_filtered, y=self.catalog_y_filtered, size=size)
                self.cat_star_markers2.setPoints(x=self.catalog_x_filtered, y=self.catalog_y_filtered, size=size)
            else:
                print('No catalog stars visible!')

    def updateImage(self, first_update=False):
        """ Update self.img with the img_handle at the current chunk """

        # remove old image
        if not first_update:
            self.img_frame.removeItem(self.img)
            self.zoom_window.removeItem(self.img_zoom)
            gamma = self.img.gamma
            invert = self.img.invert_img
        else:
            gamma = 1
            invert = False

        self.current_ff = self.img_handle.loadChunk()

        if self.current_ff is None:
            # If the current FF couldn't be opened, go to the next
            messagebox.showerror(title='Read error', message='The current image is corrupted!')
            self.nextImg()
            return

        # Choose appropriate image data
        max_data = self.current_ff.maxpixel
        ave_data = self.current_ff.avepixel

        # Apply dark
        if self.dark is not None:
            max_data = Image.applyDark(max_data, self.dark)
            ave_data = Image.applyDark(ave_data, self.dark)

        # Apply flat
        if self.flat_struct is not None:
            max_data = Image.applyFlat(max_data, self.flat_struct)
            ave_data = Image.applyFlat(ave_data, self.flat_struct)

        # pyqtgraph rotates the images for some reason
        max_data = np.rot90(max_data, k=3)
        max_data = np.flip(max_data, axis=1)
        ave_data = np.rot90(ave_data, k=3)
        ave_data = np.flip(ave_data, axis=1)

        data_dict = {'maxpixel': max_data, 'avepixel': ave_data}

        # update display with image
        self.img = ImageItem(image=data_dict, default_key=self.img_type_flag, gamma=gamma, invert=invert)
        self.img_zoom = ImageItem(image=data_dict, default_key=self.img_type_flag, gamma=gamma, invert=invert)

        # add new images to viewboxes
        self.img_frame.addItem(self.img)
        self.img_frame.autoRange(padding=0)
        self.zoom_window.addItem(self.img_zoom)

        self.img_level_min_auto = np.percentile(data_dict[self.img_type_flag], 0.1)
        self.img_level_max_auto = np.percentile(data_dict[self.img_type_flag], 99.95)

        self.bit_depth = 8*ave_data.itemsize
        # first update
        if first_update:
            self.img_level_max = 2**self.bit_depth - 1

            # Set the image resolution to platepar after reading the first image
            self.config.width = data_dict[self.img_type_flag].shape[0]
            self.config.height = data_dict[self.img_type_flag].shape[1]
            self.platepar.X_res = data_dict[self.img_type_flag].shape[0]
            self.platepar.Y_res = data_dict[self.img_type_flag].shape[1]

        elif not self.auto_levels:
            self.img_level_min, self.img_level_max = self.tab.hist.getLevels()

        # update histogram and its levels
        self.tab.hist.setImageItem(self.img)
        self.tab.hist.setImages(self.img_zoom)

        # set levels for image
        if self.auto_levels:
            self.tab.hist.setLevels(self.img_level_min_auto, self.img_level_max_auto)
        else:
            self.tab.hist.setLevels(self.img_level_min, self.img_level_max)

        # update everything for setup
        self.updateStars()
        self.updateLeftLabels()
        self.updateFitResiduals()
        self.updateDistortion()

    def updatePairedStars(self):
        """
            Draws the stars that were picked for calibration as well as draw the
            residuals and star magnitude
        """
        if len(self.paired_stars) > 0:
            self.sel_cat_star_markers.setData(pos=[pair[0][:2] for pair in self.paired_stars])
            self.sel_cat_star_markers2.setData(pos=[pair[0][:2] for pair in self.paired_stars])
        else:
            self.sel_cat_star_markers.clear()
            self.sel_cat_star_markers2.clear()

        self.centroid_star_markers.clear()
        self.centroid_star_markers2.clear()

        # Draw photometry
        if len(self.paired_stars) > 2:
            self.photometry()

        self.tab.param_manager.updatePairedStars()

    def drawCalstars(self):
        """ Draw extracted stars on the current image. """

        # Check if the given FF files is in the calstars list
        if self.img_handle.name() in self.calstars:
            # Get the stars detected on this FF file
            star_data = self.calstars[self.img_handle.name()]

            # Get star coordinates
            y, x, _, _ = np.array(star_data).T

            self.calstar_markers.setPoints(x=x, y=y)
            self.calstar_markers2.setPoints(x=x, y=y)

    def changeDistortionType(self):
        """ Change the distortion type. """

        dist_type = self.platepar.distortion_type_list[self.dist_type_index]
        self.platepar.setDistortionType(dist_type)
        self.updateDistortion()

        # Indicate that the platepar has been reset
        self.first_platepar_fit = True

        print("Distortion model changed to: {:s}".format(dist_type))

    def nextImg(self):
        """ Shows the next FF file in the list. """

        # Don't allow image change while in star picking mode
        if self.star_pick_mode:
            messagebox.showwarning(title='Star picking mode',
                                   message='You cannot cycle through images while in star picking mode!')
            return

        self.img_handle.nextChunk()

        # remove markers
        self.calstar_markers.clear()
        self.calstar_markers2.clear()
        self.cat_star_markers.clear()
        self.cat_star_markers2.clear()

        # Reset paired stars
        self.paired_stars = []
        self.residuals = None

        self.updateImage()

    def prevImg(self):
        """ Shows the previous FF file in the list. """

        # Don't allow image change while in star picking mode
        if self.star_pick_mode:
            messagebox.showwarning(title='Star picking mode',
                                   message='You cannot cycle through images while in star picking mode!')
            return

        self.img_handle.prevChunk()

        # remove markers
        self.calstar_markers.clear()
        self.calstar_markers2.clear()
        self.cat_star_markers.clear()
        self.cat_star_markers2.clear()

        # Reset paired stars
        self.paired_stars = []
        self.residuals = None

        self.updateImage()

    def onFrameResize(self):
        self.label2.setPos(0, self.img_frame.height() - self.label2.boundingRect().height())
        self.star_pick_info.setPos(self.img_frame.width()/2, self.img_frame.height() - 50)

    def saveState(self):
        """
        Saves the state of the object to a file so that when loading, it will appear the
        same as before

        Can be loaded by calling:
        python -m RMS.Astrometry.SkyFit2 PATH/skyFit2_latest.state --config .
        """
        to_remove = []

        dic = copy.copy(self.__dict__)
        for k, v in dic.items():
            if v.__class__.__bases__[0] is not object and not isinstance(v, bool) and not isinstance(v, float):
                to_remove.append(k)  # remove class that inherits from something

        for remove in to_remove:
            del dic[remove]

        savePickle(dic, self.dir_path, 'skyFit2_latest.state')
        print("Saved state to file")

    def loadState(self, dir_path, state_name):
        # function should theoretically work if called in the middle of the program
        variables = loadPickle(dir_path, state_name)
        for k, v in variables.items():
            setattr(self, k, v)

        # updating old state files with new platepar variables
        if self.platepar is not None:
            if not hasattr(self, "equal_aspect"):
                self.platepar.equal_aspect = False

            if not hasattr(self, "force_distortion_centre"):
                self.platepar.force_distortion_centre = False

            if not hasattr(self, "extinction_scale"):
                self.platepar.extinction_scale = 1.0

        #  if setupUI hasnt already been called, call it
        if not hasattr(self, 'central'):
            self.setupUI(loaded_file=True)
        else:
            self.updateImage()
            self.updateLeftLabels()

    def onMouseClicked(self, event):
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if self.star_pick_mode:
            self.mouse_press_x = self.mouse_x  # pos.x()# why is this wrong when clicking on a marker?
            self.mouse_press_y = self.mouse_y  # pos.y()

            if event.button() == QtCore.Qt.LeftButton:
                if self.star_selection_centroid:
                    # If CTRL is pressed, place the pick manually - NOTE: the intensity might be off then!!!
                    if modifiers == QtCore.Qt.ControlModifier:
                        self.x_centroid = self.mouse_press_x
                        self.y_centroid = self.mouse_press_y

                        # Compute the star intensity
                        _, _, self.star_intensity = self.centroidStar(prev_x_cent=self.x_centroid,
                                                                      prev_y_cent=self.y_centroid)
                    else:
                        # Perform centroiding with 2 iterations
                        x_cent_tmp, y_cent_tmp, _ = self.centroidStar()

                        # Check that the centroiding was successful
                        if x_cent_tmp is not None:

                            # Centroid the star around the pressed coordinates
                            self.x_centroid, self.y_centroid, \
                            self.star_intensity = self.centroidStar(prev_x_cent=x_cent_tmp,
                                                                    prev_y_cent=y_cent_tmp)

                        else:
                            return None

                    self.centroid_star_markers.addPoints(x=[self.x_centroid], y=[self.y_centroid])
                    self.centroid_star_markers2.addPoints(x=[self.x_centroid], y=[self.y_centroid])

                    # Select the closest catalog star to the centroid as the first guess
                    self.closest_cat_star_indx = self.findClosestCatalogStarIndex(self.x_centroid,
                                                                                  self.y_centroid)
                    self.sel_cat_star_markers.addPoints(x=[self.catalog_x[self.closest_cat_star_indx]],
                                                        y=[self.catalog_y[self.closest_cat_star_indx]])
                    self.sel_cat_star_markers2.addPoints(x=[self.catalog_x[self.closest_cat_star_indx]],
                                                         y=[self.catalog_y[self.closest_cat_star_indx]])

                    # Switch to the mode where the catalog star is selected
                    self.star_selection_centroid = False
                    self.cursor.setMode(self.star_selection_centroid)

                else:
                    # Select the closest catalog star
                    self.closest_cat_star_indx = self.findClosestCatalogStarIndex(self.mouse_press_x,
                                                                                  self.mouse_press_y)

                    # REMOVE marker for previously selected
                    self.sel_cat_star_markers.setData(pos=[pair[0][:2] for pair in self.paired_stars])
                    self.sel_cat_star_markers2.setData(pos=[pair[0][:2] for pair in self.paired_stars])

                    self.sel_cat_star_markers.addPoints(x=[self.catalog_x[self.closest_cat_star_indx]],
                                                        y=[self.catalog_y[self.closest_cat_star_indx]])
                    self.sel_cat_star_markers2.addPoints(x=[self.catalog_x[self.closest_cat_star_indx]],
                                                         y=[self.catalog_y[self.closest_cat_star_indx]])

            elif event.button() == QtCore.Qt.RightButton:
                if self.star_selection_centroid:
                    # Find the closest picked star
                    picked_indx = self.findClosestPickedStarIndex(self.mouse_press_x, self.mouse_press_y)

                    if self.paired_stars:
                        # Remove the picked star from the list
                        self.paired_stars.pop(picked_indx)

                    self.updatePairedStars()
                    self.updateFitResiduals()
                    self.photometry()

    def onMouseMoved(self, event):
        pos = event
        if self.img_frame.sceneBoundingRect().contains(pos):
            self.img_frame.setFocus()
            mp = self.img_frame.mapSceneToView(pos)

            self.cursor.setCenter(mp)
            self.cursor2.setCenter(mp)
            self.mouse_x, self.mouse_y = mp.x(), mp.y()

            self.zoom()

            # move zoom window to correct location
            range_ = self.img_frame.getState()['viewRange'][0]
            if mp.x() > (range_[1] - range_[0])/2 + range_[0]:
                self.v_zoom_left = True
                if self.show_key_help != 2:
                    self.v_zoom.move(QtCore.QPoint(self.label1.boundingRect().width(), 0))
                else:
                    self.v_zoom.move(QtCore.QPoint(0, 0))
            else:
                self.v_zoom_left = False
                self.v_zoom.move(QtCore.QPoint(self.img_frame.size().width() - self.show_zoom_window_size, 0))

            self.updateBottomLabel()

        # self.printFrameRate()

    def printFrameRate(self):
        try:
            print('FPS: {}'.format(np.average(self.frames)))
            self.frames[self.i] = 1/(time.time() - self.time)
            self.i = (self.i + 1)%self.n
        except ZeroDivisionError:
            pass
        self.time = time.time()

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
        self.residual_text.clear()
        if len(px_intens_list) > 3:

            # Compute apparent magnitude corrected for extinction
            catalog_mags = extinctionCorrectionTrueToApparent(catalog_mags, catalog_ra, catalog_dec,
                                                              date2JD(*self.img_handle.currentTime()),
                                                              self.platepar)

            # Fit the photometric offset (disable vignetting fit if a flat is used)
            photom_params, fit_stddev, fit_resids = photometryFit(px_intens_list, radius_list,
                                                                  catalog_mags, fixed_vignetting=(
                    0.0 if self.flat_struct is not None else None))

            photom_offset, vignetting_coeff = photom_params

            # Set photometry parameters
            self.platepar.mag_0 = -2.5
            self.platepar.mag_lev = photom_offset
            self.platepar.mag_lev_stddev = fit_stddev
            self.platepar.vignetting_coeff = vignetting_coeff

            if self.catalog_stars_visible:

                # Plot photometry deviations on the main plot as colour coded rings
                star_coords = np.array(star_coords)
                star_coords_x, star_coords_y = star_coords.T

                std = np.std(fit_resids)
                for star_x, star_y, fit_diff, star_mag in zip(star_coords_x, star_coords_y, fit_resids,
                                                              catalog_mags):
                    photom_resid_txt = "{:.2f}".format(fit_diff)

                    # Determine the size of the residual text, larger the residual, larger the text
                    photom_resid_size = 8 + np.abs(fit_diff)/(np.max(np.abs(fit_resids))/5.0)

                    if self.stdev_text_filter*std <= abs(fit_diff):
                        text1 = TextItem(photom_resid_txt, anchor=(0.5, -0.5))
                        text1.setPos(star_x, star_y)
                        text1.setFont(QtGui.QFont('times', photom_resid_size))
                        text1.setColor(QtGui.QColor(255, 255, 255))
                        text1.setAlign(QtCore.Qt.AlignCenter)
                        self.residual_text.addTextItem(text1)

                        text2 = TextItem("{:+6.2f}".format(star_mag), anchor=(0.5, 1.5))
                        text2.setPos(star_x, star_y)
                        text2.setFont(QtGui.QFont('times', 10))
                        text2.setColor(QtGui.QColor(255, 0, 0))
                        text2.setAlign(QtCore.Qt.AlignCenter)
                        self.residual_text.addTextItem(text2)
                self.residual_text.update()

            # Show the photometry fit plot
            if show_plot:

                ### PLOT PHOTOMETRY FIT ###
                # Note: An almost identical code exists in Utils.CalibrationReport

                # Init plot for photometry
                fig_p, (ax_p, ax_r) = plt.subplots(nrows=2, facecolor=None, figsize=(6.4, 7.2),
                                                   gridspec_kw={'height_ratios': [2, 1]})

                # Set photometry window title
                fig_p.canvas.set_window_title('Photometry')

                # Plot catalog magnitude vs. raw logsum of pixel intensities
                lsp_arr = np.log10(np.array(px_intens_list))
                ax_p.scatter(-2.5*lsp_arr, catalog_mags, s=5, c='r', zorder=3, alpha=0.5,
                             label="Raw (extinction corrected)")

                # Plot catalog magnitude vs. raw logsum of pixel intensities (only when no flat is used)
                if self.flat_struct is None:
                    lsp_corr_arr = np.log10(correctVignetting(np.array(px_intens_list),
                                                              np.array(radius_list),
                                                              self.platepar.vignetting_coeff))

                    ax_p.scatter(-2.5*lsp_corr_arr, catalog_mags, s=5, c='b', zorder=3, alpha=0.5,
                                 label="Corrected for vignetting")

                x_min, x_max = ax_p.get_xlim()
                y_min, y_max = ax_p.get_ylim()

                x_min_w = x_min - 3
                x_max_w = x_max + 3
                y_min_w = y_min - 3
                y_max_w = y_max + 3

                # Plot fit info
                fit_info = "Fit: {:+.1f}*LSP + {:.2f} +/- {:.2f} ".format(self.platepar.mag_0,
                                                                          self.platepar.mag_lev, fit_stddev) \
                           + "\nVignetting coeff = {:.5f}".format(self.platepar.vignetting_coeff) \
                           + "\nGamma = {:.2f}".format(self.platepar.gamma)

                print('Fit Info:')
                print(fit_info)
                print()

                # Plot the line fit
                logsum_arr = np.linspace(x_min_w, x_max_w, 10)
                ax_p.plot(logsum_arr, photomLine((10**(logsum_arr/(-2.5)), np.zeros_like(logsum_arr)),
                                                 photom_offset, self.platepar.vignetting_coeff), label=fit_info,
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
                ax_r.plot(np.linspace(0, img_diagonal, 10), np.zeros(10), linestyle='dashed', alpha=0.5,
                          color='k')

                # Plot the vignetting curve (only when no flat is used)
                if self.flat_struct is None:
                    # Plot radius from centre vs. fit residual (excluding vignetting
                    fit_resids_novignetting = catalog_mags - photomLine((np.array(px_intens_list),
                                                                         np.array(radius_list)), photom_offset, 0.0)
                    ax_r.scatter(radius_list, fit_resids_novignetting, s=5, c='r', alpha=0.5, zorder=3)

                    px_sum_tmp = 1000
                    radius_arr_tmp = np.linspace(0, img_diagonal, 50)

                    # Plot the vignetting curve
                    vignetting_loss = 2.5*np.log10(px_sum_tmp) \
                                      - 2.5*np.log10(correctVignetting(px_sum_tmp, radius_arr_tmp,
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

    def keyPressEvent(self, event):
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if event.key() == QtCore.Qt.Key_A and modifiers == QtCore.Qt.ControlModifier:
            self.auto_levels = not self.auto_levels
            if self.auto_levels:
                self.img_level_min, self.img_level_max = self.tab.hist.getLevels()
                self.tab.hist.setLevels(self.img_level_min_auto, self.img_level_max_auto)
                self.tab.hist.setMovable(False)
            else:
                self.tab.hist.setLevels(self.img_level_min, self.img_level_max)
                self.tab.hist.setMovable(True)
            # this updates image automatically

        elif event.key() == QtCore.Qt.Key_D and modifiers == QtCore.Qt.ControlModifier:
            _, self.dark = self.loadDark()

            # Apply the dark to the flat
            if self.flat_struct is not None:
                self.flat_struct.applyDark(self.dark)

            self.updateImage()

        elif event.key() == QtCore.Qt.Key_S and (modifiers & QtCore.Qt.ControlModifier) and \
                (modifiers & QtCore.Qt.ShiftModifier):
            platepar_default_path = os.path.join(os.getcwd(), self.config.platepar_name)

            # Save the platepar file
            self.platepar.write(platepar_default_path, fmt=self.platepar_fmt)
            print('Default platepar written to:', platepar_default_path)

        elif event.key() == QtCore.Qt.Key_S and modifiers == QtCore.Qt.ControlModifier:
            # If the platepar is new, save it to the working directory
            if not self.platepar_file:
                self.platepar_file = os.path.join(self.dir_path, self.config.platepar_name)

            # Save the platepar file
            self.platepar.write(self.platepar_file, fmt=self.platepar_fmt, fov=computeFOVSize(self.platepar))
            print('Platepar written to:', self.platepar_file)

            # Save the state
            self.saveState()

        elif event.key() == QtCore.Qt.Key_R and modifiers == QtCore.Qt.ControlModifier:
            self.star_pick_mode = not self.star_pick_mode
            if self.star_pick_mode:
                self.img_frame.setMouseEnabled(False, False)
                self.star_pick_info.show()
                self.cursor.show()
                self.cursor2.show()
                if self.show_zoom_window:
                    self.v_zoom.show()
            else:
                self.img_frame.setMouseEnabled(True, True)
                self.star_pick_info.hide()
                self.v_zoom.hide()
                self.cursor2.hide()
                self.cursor.hide()
            # updates automatically

            # Set distortion types
        elif event.key() == QtCore.Qt.Key_1 and modifiers == QtCore.Qt.ControlModifier:

            self.dist_type_index = 0
            self.changeDistortionType()
            self.tab.param_manager.updatePlatepar()
            self.updateLeftLabels()
            self.updateDistortion()
            self.updateStars()

        elif event.key() == QtCore.Qt.Key_2 and modifiers == QtCore.Qt.ControlModifier:

            self.dist_type_index = 1
            self.changeDistortionType()
            self.tab.param_manager.updatePlatepar()
            self.updateLeftLabels()
            self.updateDistortion()
            self.updateStars()

        elif event.key() == QtCore.Qt.Key_3 and modifiers == QtCore.Qt.ControlModifier:

            self.dist_type_index = 2
            self.changeDistortionType()
            self.tab.param_manager.updatePlatepar()
            self.updateLeftLabels()
            self.updateDistortion()
            self.updateStars()

        elif event.key() == QtCore.Qt.Key_4 and modifiers == QtCore.Qt.ControlModifier:

            self.dist_type_index = 3
            self.changeDistortionType()
            self.tab.param_manager.updatePlatepar()
            self.updateLeftLabels()
            self.updateDistortion()
            self.updateStars()

        elif event.key() == QtCore.Qt.Key_N and modifiers == QtCore.Qt.ControlModifier:
            self.makeNewPlatepar()
            self.tab.param_manager.updatePlatepar()
            self.updateLeftLabels()
            self.updateStars()
            self.updateDistortion()

        elif event.key() == QtCore.Qt.Key_F and modifiers == QtCore.Qt.ControlModifier:
            _, self.flat_struct = self.loadFlat()

            self.updateImage()

        elif event.key() == QtCore.Qt.Key_I and modifiers == QtCore.Qt.ControlModifier:
            self.toggleDistortion()
            self.tab.settings.updateShowDistortion()

        elif event.key() == QtCore.Qt.Key_Z and modifiers == QtCore.Qt.ShiftModifier:
            if self.star_pick_mode:
                self.show_zoom_window = not self.show_zoom_window
                if self.show_zoom_window:
                    self.v_zoom.show()
                else:
                    self.v_zoom.hide()

        # Do a fit on the selected stars while in the star picking mode
        elif event.key() == QtCore.Qt.Key_Z and modifiers == QtCore.Qt.ControlModifier:

            # If shift was pressed, reset distortion parameters to zero
            if modifiers == QtCore.Qt.ShiftModifier:
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

        elif event.key() == QtCore.Qt.Key_G and modifiers == QtCore.Qt.ControlModifier:
            self.grid_visible = (self.grid_visible + 1)%3
            self.onGridChanged()
            self.tab.settings.updateShowGrid()

        elif event.key() == QtCore.Qt.Key_Left:
            self.prevImg()

        elif event.key() == QtCore.Qt.Key_Right:
            self.nextImg()

        elif event.key() == QtCore.Qt.Key_M:
            self.toggleImageType()
            self.tab.settings.updateMaxAvePixel()

        elif event.key() == QtCore.Qt.Key_A:
            self.platepar.az_centre += self.key_increment
            self.platepar.updateRefRADec()

            self.tab.param_manager.updatePlatepar()
            self.updateLeftLabels()
            self.updateStars()

        elif event.key() == QtCore.Qt.Key_D:
            self.platepar.az_centre -= self.key_increment
            self.platepar.updateRefRADec()

            self.tab.param_manager.updatePlatepar()
            self.updateLeftLabels()
            self.updateStars()

        elif event.key() == QtCore.Qt.Key_W:
            self.platepar.alt_centre -= self.key_increment
            self.platepar.updateRefRADec()

            self.tab.param_manager.updatePlatepar()
            self.updateLeftLabels()
            self.updateStars()

        elif event.key() == QtCore.Qt.Key_S:
            self.platepar.alt_centre += self.key_increment
            self.platepar.updateRefRADec()

            self.tab.param_manager.updatePlatepar()
            self.updateLeftLabels()
            self.updateStars()

        # Move rotation parameter
        elif event.key() == QtCore.Qt.Key_Q:
            self.platepar.pos_angle_ref -= self.key_increment
            self.platepar.rotation_from_horiz = rotationWrtHorizon(self.platepar)

            self.updateLeftLabels()
            self.updateStars()
            self.tab.param_manager.updatePlatepar()

        elif event.key() == QtCore.Qt.Key_E:
            self.platepar.pos_angle_ref += self.key_increment
            self.platepar.rotation_from_horiz = rotationWrtHorizon(self.platepar)

            self.updateLeftLabels()
            self.updateStars()
            self.tab.param_manager.updatePlatepar()

        # Change image scale
        elif event.key() == QtCore.Qt.Key_Up:
            self.platepar.F_scale *= 1.0 + self.key_increment/100.0

            self.updateLeftLabels()
            self.updateStars()
            self.tab.param_manager.updatePlatepar()

        elif event.key() == QtCore.Qt.Key_Down:
            self.platepar.F_scale *= 1.0 - self.key_increment/100.0

            self.updateLeftLabels()
            self.updateStars()
            self.tab.param_manager.updatePlatepar()

        elif event.key() == QtCore.Qt.Key_1:

            # Increment X offset
            self.platepar.x_poly_rev[0] += 0.5
            self.platepar.x_poly_fwd[0] += 0.5

            self.tab.param_manager.updatePlatepar()
            if self.tab.param_manager.fit_parameters.currentIndex() != 2:
                self.tab.param_manager.fit_parameters.setCurrentIndex(0)
            self.updateStars()
            self.updateDistortion()

        elif event.key() == QtCore.Qt.Key_2:

            # Decrement X offset
            self.platepar.x_poly_rev[0] -= 0.5
            self.platepar.x_poly_fwd[0] -= 0.5

            self.tab.param_manager.updatePlatepar()
            if self.tab.param_manager.fit_parameters.currentIndex() != 2:
                self.tab.param_manager.fit_parameters.setCurrentIndex(0)
            self.updateStars()
            self.updateDistortion()

        elif event.key() == QtCore.Qt.Key_3:

            # Increment Y offset
            self.platepar.y_poly_rev[0] += 0.5
            self.platepar.y_poly_fwd[0] += 0.5

            self.tab.param_manager.updatePlatepar()
            if self.tab.param_manager.fit_parameters.currentIndex() != 3:
                self.tab.param_manager.fit_parameters.setCurrentIndex(1)
            self.updateStars()
            self.updateDistortion()

        elif event.key() == QtCore.Qt.Key_4:

            # Decrement Y offset
            self.platepar.y_poly_rev[0] -= 0.5
            self.platepar.y_poly_fwd[0] -= 0.5

            self.tab.param_manager.updatePlatepar()
            if self.tab.param_manager.fit_parameters.currentIndex() != 3:
                self.tab.param_manager.fit_parameters.setCurrentIndex(1)
            self.updateStars()
            self.updateDistortion()

        elif event.key() == QtCore.Qt.Key_5:

            # Decrement X 1st order distortion
            self.platepar.x_poly_rev[1] -= 0.01
            self.platepar.x_poly_fwd[1] -= 0.01

            self.tab.param_manager.updatePlatepar()
            if self.tab.param_manager.fit_parameters.currentIndex() != 2:
                self.tab.param_manager.fit_parameters.setCurrentIndex(0)
            self.updateStars()
            self.updateDistortion()

        elif event.key() == QtCore.Qt.Key_6:

            # Increment X 1st order distortion
            self.platepar.x_poly_rev[1] += 0.01
            self.platepar.x_poly_fwd[1] += 0.01

            self.tab.param_manager.updatePlatepar()
            if self.tab.param_manager.fit_parameters.currentIndex() != 2:
                self.tab.param_manager.fit_parameters.setCurrentIndex(0)
            self.updateStars()
            self.updateDistortion()

        elif event.key() == QtCore.Qt.Key_7:

            # Decrement Y 1st order distortion
            self.platepar.y_poly_rev[2] -= 0.01
            self.platepar.y_poly_fwd[2] -= 0.01

            self.tab.param_manager.updatePlatepar()
            if self.tab.param_manager.fit_parameters.currentIndex() != 3:
                self.tab.param_manager.fit_parameters.setCurrentIndex(1)
            self.updateStars()
            self.updateDistortion()

        elif event.key() == QtCore.Qt.Key_8:

            # Increment Y 1st order distortion
            self.platepar.y_poly_rev[2] += 0.01
            self.platepar.y_poly_fwd[2] += 0.01

            self.tab.param_manager.updatePlatepar()
            if self.tab.param_manager.fit_parameters.currentIndex() != 3:
                self.tab.param_manager.fit_parameters.setCurrentIndex(1)
            self.updateStars()
            self.updateDistortion()

        elif event.key() == QtCore.Qt.Key_9:
            self.platepar.extinction_scale += 0.1

            self.tab.param_manager.updatePlatepar()
            self.onExtinctionChanged()

        elif event.key() == QtCore.Qt.Key_0:
            self.platepar.extinction_scale -= 0.1

            self.tab.param_manager.updatePlatepar()
            self.onExtinctionChanged()

        # Change catalog limiting magnitude
        elif event.key() == QtCore.Qt.Key_R:
            self.cat_lim_mag += 0.1
            self.catalog_stars = self.loadCatalogStars(self.cat_lim_mag)

            self.updateLeftLabels()
            self.updateStars()
            self.tab.settings.updateLimMag()

        elif event.key() == QtCore.Qt.Key_F:
            self.cat_lim_mag -= 0.1
            self.catalog_stars = self.loadCatalogStars(self.cat_lim_mag)

            self.updateLeftLabels()
            self.updateStars()
            self.tab.settings.updateLimMag()

        elif event.key() == QtCore.Qt.Key_F1:
            self.show_key_help += 1
            if self.show_key_help >= 3:
                self.show_key_help = 0

            if self.show_key_help == 0:
                self.label1.show()
                self.label2.hide()
            elif self.show_key_help == 1:
                self.label1.show()
                self.label2.show()
            else:
                self.label1.hide()
                self.label2.hide()

            if self.v_zoom_left:
                if self.show_key_help != 2:
                    self.v_zoom.move(QtCore.QPoint(self.label1.boundingRect().width(), 0))
                else:
                    self.v_zoom.move(QtCore.QPoint(0, 0))

            if self.show_key_help == 1:
                text_str = "STAR PICKING MODE\n"
                text_str += "'LEFT CLICK' - Centroid star\n"
                text_str += "'CTRL + LEFT CLICK' - Manual star position\n"
                text_str += "'CTRL + Z' - Fit stars\n"
                text_str += "'CTRL + SHIFT + Z' - Fit with initial distortion params set to 0\n"
                text_str += "'L' - Astrometry fit details\n"
                text_str += "'P' - Photometry fit"
            else:
                text_str = ''
            self.star_pick_info.setText(text_str)


        # Key increment
        elif event.key() == QtCore.Qt.Key_Plus:

            if self.key_increment <= 0.091:
                self.key_increment += 0.01
            elif self.key_increment <= 0.91:
                self.key_increment += 0.1
            else:
                self.key_increment += 1.0

            # Don't allow the increment to be larger than 20
            if self.key_increment > 20:
                self.key_increment = 20

            self.updateLeftLabels()

        elif event.key() == QtCore.Qt.Key_Minus:

            if self.key_increment <= 0.11:
                self.key_increment -= 0.01
            elif self.key_increment <= 1.11:
                self.key_increment -= 0.1
            else:
                self.key_increment -= 1.0

            # Don't allow the increment to be smaller than 0
            if self.key_increment <= 0:
                self.key_increment = 0.01

            self.updateLeftLabels()


        # Enter FOV centre
        elif event.key() == QtCore.Qt.Key_V:
            data = self.getFOVcentre()
            if data:
                self.platepar.RA_d, self.platepar.dec_d, self.platepar.rotation_from_horiz = data

                # Compute reference Alt/Az to apparent coordinates, epoch of date
                az_centre, alt_centre = trueRaDec2ApparentAltAz(
                    self.platepar.RA_d, self.platepar.dec_d, self.platepar.JD,
                    self.platepar.lat, self.platepar.lon, self.platepar.refraction)

                self.platepar.az_centre, self.platepar.alt_centre = az_centre, alt_centre

                # Compute the position angle
                self.platepar.pos_angle_ref = rotationWrtHorizonToPosAngle(self.platepar,
                                                                           self.platepar.rotation_from_horiz)

                self.tab.param_manager.updatePlatepar()
                self.updateLeftLabels()
                self.updateStars()

        elif event.key() == QtCore.Qt.Key_G:
            if self.platepar is not None:
                self.platepar.equal_aspect = not self.platepar.equal_aspect

                self.tab.param_manager.updatePlatepar()
                self.updateLeftLabels()
                self.updateStars()

        # Get initial parameters from astrometry.net
        elif event.key() == QtCore.Qt.Key_X:
            print("Solving with astrometry.net")

            upload_image = True
            if modifiers == QtCore.Qt.ShiftModifier:
                upload_image = False

            # Estimate initial parameters using astrometry.net
            self.getInitialParamsAstrometryNet(upload_image=upload_image)

            self.updateDistortion()
            self.updateLeftLabels()
            self.updateStars()
            self.tab.param_manager.updatePlatepar()


        elif event.key() == QtCore.Qt.Key_C:
            self.toggleShowCalStars()
            self.tab.settings.updateShowCalStars()
            # updates image automatically

        # Increase image gamma
        elif event.key() == QtCore.Qt.Key_U:
            # Increase image gamma by a factor of 1.1x
            self.img.updateGamma(1/0.9)
            if self.img_zoom:
                self.img_zoom.updateGamma(1/0.9)
            self.updateLeftLabels()
            self.tab.settings.updateImageGamma()

        elif event.key() == QtCore.Qt.Key_J:
            # Decrease image gamma by a factor of 0.9x
            self.img.updateGamma(0.9)
            if self.img_zoom:
                self.img_zoom.updateGamma(0.9)

            self.updateLeftLabels()
            self.tab.settings.updateImageGamma()

        elif event.key() == QtCore.Qt.Key_T:
            if self.platepar is not None:
                self.platepar.refraction = not self.platepar.refraction

                self.onRefractionChanged()
                self.tab.param_manager.updatePlatepar()

        elif event.key() == QtCore.Qt.Key_B:
            if self.platepar is not None:
                self.platepar.force_distortion_centre = not self.platepar.force_distortion_centre
                self.updateStars()
                self.updateLeftLabels()

        elif event.key() == QtCore.Qt.Key_H:
            self.toggleShowCatStars()
            self.tab.settings.updateShowCatStars()

        elif event.key() == QtCore.Qt.Key_I:
            self.toggleInvertColours()
            self.tab.settings.updateInvertColours()

        elif event.key() == QtCore.Qt.Key_Return:
            if self.star_pick_mode:
                # If the right catalog star has been selected, save the pair to the list
                if not self.star_selection_centroid:
                    # Add the image/catalog pair to the list
                    self.paired_stars.append([[self.x_centroid, self.y_centroid, self.star_intensity],
                                              self.catalog_stars[self.closest_cat_star_indx]])

                    # Switch back to centroiding mode
                    self.star_selection_centroid = True
                    self.closest_cat_star_indx = None
                    self.cursor.setMode(self.star_selection_centroid)
                    self.updatePairedStars()

        elif event.key() == QtCore.Qt.Key_Escape:
            if self.star_pick_mode:
                # If the ESC is pressed when the star has been centroided, reset the centroid
                if not self.star_selection_centroid:
                    self.star_selection_centroid = True
                    self.x_centroid = None
                    self.y_centroid = None
                    self.star_intensity = None
                    self.cursor.setMode(self.star_selection_centroid)
                    self.updatePairedStars()

        elif event.key() == QtCore.Qt.Key_P:
            # Show the photometry plot
            self.photometry(show_plot=True)

        elif event.key() == QtCore.Qt.Key_L:
            if self.star_pick_mode:
                # Show astrometry residuals plot
                self.showAstrometryFitPlots()

        # Limit values of RA and Dec
        self.platepar.RA_d = self.platepar.RA_d%360

        if self.platepar.dec_d > 90:
            self.platepar.dec_d = 90

        elif self.platepar.dec_d < -90:
            self.platepar.dec_d = -90

    def wheelEvent(self, event):
        """ Change star selector aperature on scroll. """
        delta = event.angleDelta().y()
        modifier = QtWidgets.QApplication.keyboardModifiers()

        if self.img_frame.sceneBoundingRect().contains(event.pos()) and self.star_pick_mode:
            if modifier == QtCore.Qt.ControlModifier:
                if delta < 0:
                    self.scrolls_back = 0
                    self.star_aperature_radius += 1
                    self.cursor.setRadius(self.star_aperature_radius)
                    self.cursor2.setRadius(self.star_aperature_radius)
                elif delta > 0 and self.star_aperature_radius > 1:
                    self.scrolls_back = 0
                    self.star_aperature_radius -= 1
                    self.cursor.setRadius(self.star_aperature_radius)
                    self.cursor2.setRadius(self.star_aperature_radius)
            else:
                if delta < 0:
                    self.scrolls_back += 1
                    if self.scrolls_back > 2:
                        self.img_frame.autoRange(padding=0)
                elif delta > 0:
                    self.scrolls_back = 0
                    self.img_frame.scaleBy([0.80, 0.80], QtCore.QPoint(self.mouse_x, self.mouse_y))

    def toggleImageType(self):
        """ Toggle between the maxpixel and avepixel. """
        if self.img_type_flag == 'maxpixel':
            self.img_type_flag = 'avepixel'

        else:
            self.img_type_flag = 'maxpixel'

        ### UPDATE IMAGE ###
        self.img.selectImage(self.img_type_flag)
        self.img_zoom.selectImage(self.img_type_flag)

    def toggleShowCatStars(self):
        """ Toggle between showing catalog stars and not """
        self.catalog_stars_visible = not self.catalog_stars_visible
        if self.catalog_stars_visible:
            self.cat_star_markers.show()
            self.cat_star_markers2.show()
        else:
            self.cat_star_markers.hide()
            self.cat_star_markers2.hide()

        self.photometry()

    def toggleShowCalStars(self):
        self.draw_calstars = not self.draw_calstars
        if self.draw_calstars:
            self.calstar_markers.show()
            self.calstar_markers2.show()
        else:
            self.calstar_markers.hide()
            self.calstar_markers2.hide()

    def toggleDistortion(self):
        self.draw_distortion = not self.draw_distortion

        if self.draw_distortion:
            self.distortion_lines.show()
            self.updateDistortion()
        else:
            self.distortion_lines.hide()

    def toggleInvertColours(self):
        self.img.invert()
        self.img_zoom.invert()

    def loadCatalogStars(self, lim_mag):
        """ Loads stars from the BSC star catalog.

        Arguments:
            lim_mag: [float] Limiting magnitude of catalog stars.

        """

        # Load catalog stars
        catalog_stars, self.mag_band_string, self.config.star_catalog_band_ratios = StarCatalog.readStarCatalog(
            self.config.star_catalog_path, self.config.star_catalog_file, lim_mag=lim_mag,
            mag_band_ratios=self.config.star_catalog_band_ratios)

        return catalog_stars

    def updateDistortion(self):
        """ Draw distortion guides. """
        # Only draw the distortion if we have a platepar
        if self.platepar:
            # Sample points on every image axis (start/end 5% from image corners)
            x_samples = 30
            y_samples = int(x_samples*(self.platepar.Y_res/self.platepar.X_res))
            corner_frac = 0.05
            x_samples = np.linspace(corner_frac*self.platepar.X_res, (1 - corner_frac)*self.platepar.X_res,
                                    x_samples)
            y_samples = np.linspace(corner_frac*self.platepar.Y_res, (1 - corner_frac)*self.platepar.Y_res,
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

            x = [None]*2*len(x_arr)
            x[::2] = x_arr
            x[1::2] = x_nodist

            y = [None]*2*len(y_arr)
            y[::2] = y_arr
            y[1::2] = y_nodist

            self.distortion_lines.setData(x=x, y=y)

    def computeCentreRADec(self):
        """ Compute RA and Dec of the FOV centre in degrees. """

        # The the time of the image
        img_time = self.img_handle.currentTime()

        # Convert the FOV centre to RA/Dec
        _, ra_centre, dec_centre, _ = xyToRaDecPP([img_time], [self.platepar.X_res/2],
                                                  [self.platepar.Y_res/2], [1], self.platepar,
                                                  extinction_correction=False)

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
        filtered_indices, filtered_catalog_stars = subsetCatalog(catalog_stars, ra_centre, dec_centre,
                                                                 jd, self.platepar.lat, self.platepar.lon, fov_radius,
                                                                 self.cat_lim_mag)

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
            if self.img.data.itemsize*8 > 8:

                # Rescale the image to 8bit
                minv, maxv = self.tab.hist.getLevels()
                img_data = Image.adjustLevels(self.img.data, minv, self.img.gamma, maxv)
                img_data -= np.min(img_data)
                img_data = 255*(img_data/np.max(img_data))
                img_data = img_data.astype(np.uint8)

            else:
                img_data = self.img.data

            solution = novaAstrometryNetSolve(img=img_data, fov_w_range=fov_w_range)

        if solution is None:
            messagebox.showerror(title='Astrometry.net error',
                                 message='Astrometry.net failed to find a solution!')

            return None

        # Extract the parameters
        ra, dec, orientation, scale, fov_w, fov_h = solution

        jd = date2JD(*self.img_handle.currentTime())

        # Compute the position angle from the orientation
        pos_angle_ref = rotationWrtStandardToPosAngle(self.platepar, orientation)

        # Compute reference azimuth and altitude
        azim, alt = trueRaDec2ApparentAltAz(ra, dec, jd, self.platepar.lat, self.platepar.lon)

        # Set parameters to platepar
        self.platepar.pos_angle_ref = pos_angle_ref
        self.platepar.F_scale = scale
        self.platepar.az_centre = azim
        self.platepar.alt_centre = alt

        self.platepar.updateRefRADec(skip_rot_update=True)

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

        data = d.getAltAz()
        if all([x is None for x in data]):
            return None

        self.azim_centre, self.alt_centre, rot_horizontal = data

        root.destroy()

        # Get the middle time of the first FF
        img_time = self.img_handle.currentTime()

        # Set the reference platepar time to the time of the FF
        self.platepar.JD = date2JD(*img_time)

        # Set the reference hour angle
        self.platepar.Ho = JD2HourAngle(self.platepar.JD)%360

        # Convert FOV centre to RA, Dec
        ra, dec = altAz2RADec(self.azim_centre, self.alt_centre, date2JD(*img_time),
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
            messagebox.showerror(title='Platepar file error',
                                 message='The file you selected could not be loaded as a platepar file!')

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

    def makeNewPlatepar(self):
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
        self.platepar.az_centre, self.platepar.alt_centre = trueRaDec2ApparentAltAz(self.platepar.JD,
                                                                                    self.platepar.lon,
                                                                                    self.platepar.lat,
                                                                                    self.platepar.RA_d,
                                                                                    self.platepar.dec_d)

        # Check that the calibration parameters are within the nominal range
        self.checkParamRange()

        # Compute the position angle
        self.platepar.pos_angle_ref = rotationWrtHorizonToPosAngle(self.platepar,
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
            flat = Image.loadFlat(*os.path.split(flat_file), dtype=self.img.data.dtype,
                                  byteswap=self.img_handle.byteswap)
        except:
            messagebox.showerror(title='Flat field file error',
                                 message='Flat could not be loaded!')
            return False, None

        # Check if the size of the file matches
        if self.img.data.shape != flat.flat_img.shape:
            messagebox.showerror(title='Flat field file error',
                                 message='The size of the flat field does not match the size of the image!')

            flat = None

        # Check if the flat field was successfuly loaded
        if flat is None:
            messagebox.showerror(title='Flat field file error',
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
            dark = Image.loadDark(*os.path.split(dark_file), dtype=self.img.data.dtype,
                                  byteswap=self.img_handle.byteswap)

        except:
            messagebox.showerror(title='Dark frame error',
                                 message='Dark frame could not be loaded!')

            return False, None

        dark = dark.astype(self.img.data.dtype)

        # Check if the size of the file matches
        if self.img.data.shape != dark.shape:
            messagebox.showerror(title='Dark field file error',
                                 message='The size of the dark frame does not match the size of the image!')

            dark = None

        # Check if the dark frame was successfuly loaded
        if dark is None:
            messagebox.showerror(title='Dark field file error',
                                 message='The file you selected could not be loaded as a dark field!')

        return dark_file, dark

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

    def centroidStar(self, prev_x_cent=None, prev_y_cent=None):
        """ Find the centroid of the star clicked on the image. """

        # If the centroid from the previous iteration is given, use that as the centre
        if (prev_x_cent is not None) and (prev_y_cent is not None):
            mouse_x = prev_x_cent
            mouse_y = prev_y_cent

        else:
            mouse_x = self.mouse_press_x
            mouse_y = self.mouse_press_y

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
        img_crop = self.img.data[x_min:x_max, y_min:y_max]

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
        if bg_counter == 0:
            print('Zero division error')
            raise NotImplementedError
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
                    x_acc += i*(img_crop[i, j] - bg_intensity)
                    y_acc += j*(img_crop[i, j] - bg_intensity)
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
        img_stars = np.array(self.paired_stars)[:, 0]
        catalog_stars = np.array(self.paired_stars)[:, 1]

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
        print(
            ' No,   Img X,   Img Y, RA (deg), Dec (deg),    Mag, -2.5*LSP,    Cat X,   Cat Y,    Std X,   Std Y, Err amin,  Err px, Direction')

        # Calculate the distance and the angle between each pair of image positions and catalog predictions
        for star_no, (cat_x, cat_y, std_x, std_y, cat_coords, img_c) in enumerate(zip(catalog_x, catalog_y,
                                                                                      standard_x, standard_y,
                                                                                      catalog_stars, img_stars)):
            img_x, img_y, sum_intens = img_c
            ra, dec, mag = cat_coords

            delta_x = cat_x - img_x
            delta_y = cat_y - img_y

            # Compute image residual and angle of the error
            angle = np.arctan2(delta_y, delta_x)
            distance = np.sqrt(delta_x**2 + delta_y**2)

            # Compute the residuals in ra/dec in angular coordinates
            img_time = self.img_handle.currentTime()
            _, ra_img, dec_img, _ = xyToRaDecPP([img_time], [img_x], [img_y], [1], self.platepar,
                                                extinction_correction=False)

            ra_img = ra_img[0]
            dec_img = dec_img[0]

            # Compute the angular distance in degrees
            angular_distance = angularSeparation(ra, dec, ra_img, dec_img)

            residuals.append([img_x, img_y, angle, distance, angular_distance])

            # Print out the residuals
            print(
                '{:3d}, {:7.2f}, {:7.2f}, {:>8.3f}, {:>+9.3f}, {:+6.2f},  {:7.2f}, {:8.2f}, {:7.2f}, {:8.2f}, {:7.2f}, {:8.2f}, {:7.2f}, {:+9.1f}'.format(
                    star_no + 1, img_x, img_y,
                    ra, dec, mag, -2.5*np.log10(sum_intens), cat_x, cat_y, std_x, std_y, 60*angular_distance,
                    distance, np.degrees(angle)))

        mean_angular_error = 60*np.mean([entry[4] for entry in residuals])

        # If the average angular error is larger than 60 arc minutes, report it in degrees
        if mean_angular_error > 60:
            mean_angular_error /= 60
            angular_error_label = 'deg'

        else:
            angular_error_label = 'arcmin'

        print('Average error: {:.2f} px, {:.2f} {:s}'.format(np.mean([entry[3] for entry in residuals]),
                                                             mean_angular_error, angular_error_label))

        # Print the field of view size
        print("FOV: {:.2f} x {:.2f} deg".format(*computeFOVSize(self.platepar)))

        ####################

        # Save the residuals
        self.residuals = residuals

        self.updateDistortion()
        self.updateLeftLabels()
        self.updateStars()
        self.updateFitResiduals()
        self.tab.param_manager.updatePlatepar()

    def updateFitResiduals(self):
        """ Draw fit residuals. """

        if self.residuals is not None:

            x1 = []
            y1 = []

            x2 = []
            y2 = []
            # pen1 = QtGui.QPen(QtGui.QColor(255, 165, 0, 255))
            # pen2 = QtGui.QPen(QtGui.QColor(255, 255, 0, 255))
            # pen2.setStyle(QtCore.Qt.DashLine)
            # Plot the residuals
            res_scale = 100
            for entry in self.residuals:
                img_x, img_y, angle, distance, angular_distance = entry

                # Calculate coordinates of the end of the residual line
                res_x = img_x + res_scale*np.cos(angle)*distance
                res_y = img_y + res_scale*np.sin(angle)*distance

                # Plot the image residuals
                x1.extend([img_x, res_x])
                y1.extend([img_y, res_y])

                # Convert the angular distance from degrees to equivalent image pixels
                ang_dist_img = angular_distance*self.platepar.F_scale
                res_x = img_x + res_scale*np.cos(angle)*ang_dist_img
                res_y = img_y + res_scale*np.sin(angle)*ang_dist_img

                # Plot the sky residuals
                x2.extend([img_x, res_x])
                y2.extend([img_y, res_y])

            self.residual_lines.setData(x=x2, y=y2)
        else:
            self.residual_lines.clear()

    def showAstrometryFitPlots(self):
        """ Show window with astrometry fit details. """

        # Extract paired catalog stars and image coordinates separately
        img_stars = np.array(self.paired_stars)[:, 0]
        catalog_stars = np.array(self.paired_stars)[:, 1]

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
        for star_no, (cat_x, cat_y, cat_coords, img_c) in enumerate(zip(catalog_x, catalog_y, catalog_stars,
                                                                        img_stars)):
            # Compute image coordinates
            img_x, img_y, _ = img_c
            img_radius = np.hypot(img_x - self.platepar.X_res/2, img_y - self.platepar.Y_res/2)

            # Compute sky coordinates
            cat_ra, cat_dec, _ = cat_coords
            cat_ang_separation = angularSeparation(cat_ra, cat_dec, ra_centre, dec_centre)

            # Compute RA/Dec from image
            _, img_ra, img_dec, _ = xyToRaDecPP([img_time], [img_x], [img_y], [1], self.platepar,
                                                extinction_correction=False)
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
            img_ang_separation = angularSeparation(img_ra, img_dec, ra_centre, dec_centre)
            skyradius_residuals.append(cat_ang_separation - img_ang_separation)

            # Compute azim/elev from the catalog
            azim_cat, elev_cat = trueRaDec2ApparentAltAz(cat_ra, cat_dec, jd, self.platepar.lat, self.platepar.lon)

            azim_list.append(azim_cat)
            elev_list.append(elev_cat)

            # Compute azim/elev from image coordinates
            azim_img, elev_img = trueRaDec2ApparentAltAz(img_ra, img_dec, jd, self.platepar.lat, self.platepar.lon)

            # Compute azim/elev residuals
            azim_residuals.append(((azim_cat - azim_img + 180)%360 - 180)*np.cos(np.radians(elev_cat)))
            elev_residuals.append(elev_cat - elev_img)

        # Init astrometry fit window
        fig_a, (
            (ax_azim, ax_elev, ax_skyradius),
            (ax_x, ax_y, ax_radius)
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
        ax_x.set_xlim([0, self.img.data.shape[1]])

        # Plot Y vs Y error
        ax_y.scatter(y_list, y_residuals, s=2, c='k', zorder=3)

        ax_y.grid()
        ax_y.set_xlabel("Y (px)")
        ax_y.set_ylabel("Y error (px)")
        ax_y.set_xlim([0, self.img.data.shape[0]])

        # Plot radius vs radius error
        ax_radius.scatter(radius_list, radius_residuals, s=2, c='k', zorder=3)

        ax_radius.grid()
        ax_radius.set_xlabel("Radius (px)")
        ax_radius.set_ylabel("Radius error (px)")
        ax_radius.set_xlim([0, np.hypot(self.img.data.shape[0]/2, self.img.data.shape[1]/2)])

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

    arg_parser.add_argument('dir_path', nargs=1, metavar='DIR_PATH', type=str,
                            help='Path to the folder with FF or image files, path to a video file, or to a state file.'
                                 ' If images or videos are given, their names must be in the format: YYYYMMDD_hhmmss.uuuuuu')

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str,
                            help="Path to a config file which will be used instead of the default one."
                                 " To load the .config file in the given data directory, write '.' (dot).")

    arg_parser.add_argument('-t', '--timebeg', nargs=1, metavar='TIME', type=str,
                            help="The beginning time of the video file in the YYYYMMDD_hhmmss.uuuuuu format.")

    arg_parser.add_argument('-f', '--fps', metavar='FPS', type=float,
                            help="Frames per second when images are used. If not given, it will be read from the config file.")

    arg_parser.add_argument('-g', '--gamma', metavar='CAMERA_GAMMA', type=float,
                            help="Camera gamma value. Science grade cameras have 1.0, consumer grade cameras have 0.45. "
                                 "Adjusting this is essential for good photometry, and doing star photometry through SkyFit"
                                 " can reveal the real camera gamma.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    app = QtWidgets.QApplication(sys.argv)

    # If the state file was given, load the state
    if cml_args.dir_path[0].endswith('.state'):

        dir_path, state_name = os.path.split(cml_args.dir_path[0])
        config = cr.loadConfigFromDirectory(cml_args.config, cml_args.dir_path)

        # create plate_tool without calling its constructor then calling loadstate
        plate_tool = PlateTool.__new__(PlateTool)
        super(PlateTool, plate_tool).__init__()
        plate_tool.loadState(dir_path, state_name)

        # Set the dir path in case it changed
        plate_tool.dir_path = dir_path

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

        plate_tool = PlateTool(dir_path, config, beginning_time=beginning_time,
                               fps=cml_args.fps, gamma=cml_args.gamma)
    sys.exit(app.exec_())
