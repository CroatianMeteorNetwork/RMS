from __future__ import print_function, division, absolute_import, unicode_literals

import os
import math
import argparse
import traceback
import copy
import cProfile
import json
import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP, raDecToXYPP, \
    rotationWrtHorizon, rotationWrtHorizonToPosAngle, computeFOVSize, photomLine, photometryFit, \
    rotationWrtStandard, rotationWrtStandardToPosAngle, correctVignetting, \
    extinctionCorrectionTrueToApparent, applyAstrometryFTPdetectinfo
from RMS.Astrometry.Conversions import date2JD, JD2HourAngle, trueRaDec2ApparentAltAz, \
    apparentAltAz2TrueRADec, jd2Date, datetime2JD
from RMS.Astrometry.AstrometryNetNova import novaAstrometryNetSolve
import RMS.ConfigReader as cr
import RMS.Formats.CALSTARS as CALSTARS
from RMS.Formats.Platepar import Platepar, getCatalogStarsImagePositions
from RMS.Formats.FrameInterface import detectInputTypeFolder, detectInputTypeFile
from RMS.Formats.FTPdetectinfo import writeFTPdetectinfo
from RMS.Formats import StarCatalog
from RMS.Pickling import loadPickle, savePickle
from RMS.Math import angularSeparation, RMSD
from RMS.Misc import decimalDegreesToSexHours, openFileDialog, openFolderDialog
from RMS.Routines.AddCelestialGrid import updateRaDecGrid, updateAzAltGrid
from RMS.Routines.CustomPyqtgraphClasses import *
from RMS.Routines import RollingShutterCorrection

import pyximport
pyximport.install(setup_args={'include_dirs': [np.get_include()]})
from RMS.Astrometry.CyFunctions import subsetCatalog


def qmessagebox(message="", title="Error", message_type="warning"):
    msg = QtGui.QMessageBox()
    if message_type == "warning":
        msg.setIcon(QtGui.QMessageBox.Warning)
    elif message_type == "error":
        msg.setIcon(QtGui.QMessageBox.Critical)
    else:
        msg.setIcon(QtGui.QMessageBox.Information)
    msg.setText(message)
    msg.setWindowTitle(title)
    msg.setStandardButtons(QtGui.QMessageBox.Ok)
    msg.exec_()


class QFOVinputDialog(QtWidgets.QDialog):

    def __init__(self, *args, **kwargs):
        super(QFOVinputDialog, self).__init__(*args, **kwargs)

        self.setWindowTitle("Pointing information")

        btn = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel

        buttonBox = QtWidgets.QDialogButtonBox(btn)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

        self.azim_edit = QtWidgets.QLineEdit(self)
        self.alt_edit = QtWidgets.QLineEdit(self)
        self.rot_edit = QtWidgets.QLineEdit(self)

        azim_validator = QtGui.QDoubleValidator(-180, 360, 9)
        azim_validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
        self.azim_edit.setValidator(azim_validator)
        alt_validator = QtGui.QDoubleValidator(0, 90, 9)
        alt_validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
        self.alt_edit.setValidator(alt_validator)
        rot_validator = QtGui.QDoubleValidator(-180, 360, 9)
        rot_validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
        self.rot_edit.setValidator(rot_validator)
        layout = QtWidgets.QVBoxLayout(self)

        layout.addWidget(QtWidgets.QLabel("Please enter FOV centre (degrees),\nAzimuth +E of due N\nRotation from vertical"))

        formlayout = QtWidgets.QFormLayout()
        formlayout.setLabelAlignment(QtCore.Qt.AlignLeft)

        formlayout.addRow("Azimuth", self.azim_edit)
        formlayout.addRow("Altitude", self.alt_edit)
        formlayout.addRow("Rotation", self.rot_edit)

        layout.addLayout(formlayout)
        layout.addWidget(buttonBox)
        self.setLayout(layout)

    def getInputs(self):
        try:
            azim = float(self.azim_edit.text()) % 360
            alt = float(self.alt_edit.text())
            rot = float(self.rot_edit.text()) % 360
        except ValueError:
            return 0, 0, 0

        return azim, alt, rot


class PlateTool(QtWidgets.QMainWindow):
    def __init__(self, input_path, config, beginning_time=None, fps=None, gamma=None, use_fr_files=False, \
        startUI=True):
        """ SkyFit interactive window.

        Arguments:
            input_path: [str] Absolute path to the directory containing FF or image files, or a path to a 
                video file.
            config: [Config struct]

        Keyword arguments:
            beginning_time: [datetime] Datetime of the video beginning. Optional, only can be given for
                video input formats.
            fps: [float] Frames per second, used only when images in a folder are used.
            gamma: [float] Camera gamma. None by default, then it will be used from the platepar file or
                config.
            use_fr_files: [bool] Include FR files together with FF files. False by default.
            startUI: [bool] Start the GUI. True by default.
        """

        super(PlateTool, self).__init__()

        # Mode of operation - skyfit for fitting astrometric plates, manualreduction for manual picking
        #   of position on frames and photometry
        self.mode = 'skyfit'
        self.mode_list = ['skyfit', 'manualreduction']


        self.input_path = input_path
        if os.path.isfile(self.input_path):
            self.dir_path = os.path.dirname(self.input_path)
        else:
            self.dir_path = self.input_path


        self.config = config

        # Store forced time of first frame
        self.beginning_time = beginning_time

        # Extract the directory path if a file was given
        if os.path.isfile(self.dir_path):
            self.dir_path, _ = os.path.split(self.dir_path)

        # If camera gamma was given, change the value in config
        if gamma is not None:
            config.gamma = gamma

        # If the FPS was given, change the FPS in the config file
        if fps is not None:
            config.fps = fps

        self.use_fr_files = use_fr_files

        # Star picking mode variables
        self.star_aperature_radius = 5
        self.x_centroid = self.y_centroid = None
        self.closest_cat_star_indx = None

        # List of paired image and catalog stars
        self.pick_list = {}
        self.paired_stars = []
        self.residuals = None

        # Positions of the mouse cursor
        self.mouse_x = 0
        self.mouse_y = 0

        # Key increment
        self.key_increment = 1.0
        
        # Init a blank platepar
        self.platepar = Platepar()

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

        ###################################################################################################
        # LOADING STARS

        # Load catalog stars
        self.catalog_stars = self.loadCatalogStars(self.config.catalog_mag_limit)
        self.cat_lim_mag = self.config.catalog_mag_limit

        # Check if the catalog exists
        if not self.catalog_stars.any():

            qmessagebox(title='Star catalog error', \
                message='Star catalog from path ' \
                    + os.path.join(self.config.star_catalog_path, self.config.star_catalog_file) \
                    + 'could not be loaded!',
                message_type="error")

            sys.exit()

        else:
            print('Star catalog loaded!')


        self.calstars = {}
        self.loadCalstars()



        # Detect data input type and init the image handle
        self.detectInputType(load=True, beginning_time=beginning_time, use_fr_files=self.use_fr_files)


        ###################################################################################################
        # PLATEPAR

        # Load the platepar file
        self.loadPlatepar()


        # Set the given gamma value to platepar
        if gamma is not None:
            self.platepar.gamma = gamma


        # Load distorion type index
        self.dist_type_index = self.platepar.distortion_type_list.index(self.platepar.distortion_type)


        ###################################################################################################

        print()

        # INIT WINDOW
        if startUI:
            self.setupUI()



    def setupUI(self, loaded_file=False):
        """ Setup pyqt UI with widgets. No variables worth saving should be defined here.

        Keyword arguments:
            loaded_file: [bool] Loaded a state from a file. False by default.
        """

        self.central = QtWidgets.QWidget()
        self.setCentralWidget(self.central)

        layout = QtWidgets.QGridLayout()
        self.central.setLayout(layout)

        ###################################################################################################
        # TOP MENU

        menu = self.menuBar()

        self.new_platepar_action = QtWidgets.QAction("New platepar")
        self.new_platepar_action.setShortcut("Ctrl+N")  # key bindings here do not get passed to keypress
        self.new_platepar_action.triggered.connect(self.makeNewPlatepar)

        self.load_platepar_action = QtWidgets.QAction("Load platepar")
        self.load_platepar_action.setShortcut('Ctrl+P')
        self.load_platepar_action.triggered.connect(lambda: self.loadPlatepar(update=True))

        self.save_platepar_action = QtWidgets.QAction("Save platepar")
        self.save_platepar_action.triggered.connect(self.savePlatepar)

        self.save_reduction_action = QtWidgets.QAction('Save state and reduction')
        self.save_reduction_action.setShortcut('Ctrl+S')
        self.save_reduction_action.triggered.connect(lambda: [self.saveState(),
                                                              self.saveFTPdetectinfo(),
                                                              self.saveJSON()])

        self.save_current_frame_action = QtWidgets.QAction('Save current frame')
        self.save_current_frame_action.setShortcut('Ctrl+W')
        self.save_current_frame_action.triggered.connect(self.saveCurrentFrame)

        self.save_default_platepar_action = QtWidgets.QAction("Save default platepar")
        self.save_default_platepar_action.setShortcut('Ctrl+Shift+S')
        self.save_default_platepar_action.triggered.connect(self.saveDefaultPlatepar)

        self.save_state_platepar_action = QtWidgets.QAction("Save state and platepar")
        self.save_state_platepar_action.triggered.connect(lambda: [self.saveState(), self.savePlatepar()])
        self.save_state_platepar_action.setShortcut('Ctrl+S')

        self.load_state_action = QtWidgets.QAction("Load state")
        self.load_state_action.triggered.connect(self.findLoadState)

        self.station_action = QtWidgets.QAction("Change station")
        self.station_action.triggered.connect(self.changeStation)

        self.toggle_info_action = QtWidgets.QAction("Toggle Info")
        self.toggle_info_action.triggered.connect(self.toggleInfo)
        self.toggle_info_action.setShortcut('F1')

        self.toggle_zoom_window = QtWidgets.QAction("Toggle zoom window")
        self.toggle_zoom_window = QtWidgets.QAction("Toggle zoom window")
        self.toggle_zoom_window.triggered.connect(self.toggleZoomWindow)
        self.toggle_zoom_window.setShortcut('shift+Z')

        self.file_menu = menu.addMenu('File')
        self.view_menu = menu.addMenu('View')

        # TESTING
        self.i = 0
        self.n = 100
        self.frames = np.zeros(self.n)
        self.profile = cProfile.Profile()
        self.keys_pressed = []  # keeps track of all the keys pressed

        ###################################################################################################
        # STATUS BAR ON BOTTOM

        # bottom information
        self.status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self.status_bar)

        self.skyfit_button = QtWidgets.QPushButton('SkyFit')
        self.skyfit_button.pressed.connect(lambda: self.changeMode('skyfit'))
        self.manualreduction_button = QtWidgets.QPushButton('ManualReduction')
        self.manualreduction_button.pressed.connect(lambda: self.changeMode('manualreduction'))
        self.status_bar.addPermanentWidget(self.skyfit_button)
        self.status_bar.addPermanentWidget(self.manualreduction_button)

        ###################################################################################################
        # CENTRAL WIDGET (DISPLAY)

        # Main Image
        self.scrolls_back = 0
        self.clicked = 0

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
        self.show_key_help = 1

        self.label1 = TextItem(color=(0, 0, 0), fill=(255, 255, 255, 100))
        self.label1.setTextWidth(200)
        self.label1.setZValue(1000)
        self.label1.setParentItem(self.img_frame)

        # bottom left label
        self.label2 = TextItem(color=(0, 0, 0), fill=(255, 255, 255, 100))
        self.label2.setTextWidth(200)
        self.label2.setZValue(1000)
        self.label2.setParentItem(self.img_frame)

        # F1 info label
        self.label_f1 = TextItem(color=(0, 0, 0), fill=(255, 255, 255, 100))
        self.label_f1.setTextWidth(100)
        self.label_f1.setZValue(1000)
        self.label_f1.setParentItem(self.img_frame)
        self.label_f1.hide()

        self.catalog_stars_visible = True

        # catalog star markers (main window)
        self.cat_star_markers = pg.ScatterPlotItem()
        self.cat_star_markers.setPen('r')
        self.cat_star_markers.setBrush((0, 0, 0, 0))
        self.cat_star_markers.setSymbol(Crosshair())
        self.cat_star_markers.setZValue(4)
        self.img_frame.addItem(self.cat_star_markers)

        # catalog star markers (zoom window)
        self.cat_star_markers2 = pg.ScatterPlotItem()
        self.cat_star_markers2.setPen('r')
        self.cat_star_markers2.setBrush((0, 0, 0, 0))
        self.cat_star_markers2.setSize(20)
        self.cat_star_markers2.setSymbol(Crosshair())
        self.cat_star_markers2.setZValue(4)
        self.zoom_window.addItem(self.cat_star_markers2)

        self.selected_stars_visible = True

        # selected catalog star markers (main window)
        self.sel_cat_star_markers = pg.ScatterPlotItem()
        self.sel_cat_star_markers.setPen('b', width=3)
        self.sel_cat_star_markers.setSize(10)
        self.sel_cat_star_markers.setSymbol(Cross())
        self.sel_cat_star_markers.setZValue(4)
        self.img_frame.addItem(self.sel_cat_star_markers)

        # selected catalog star markers (zoom window)
        self.sel_cat_star_markers2 = pg.ScatterPlotItem()
        self.sel_cat_star_markers2.setPen('b', width=3)
        self.sel_cat_star_markers2.setSize(10)
        self.sel_cat_star_markers2.setSymbol(Cross())
        self.sel_cat_star_markers2.setZValue(4)
        self.zoom_window.addItem(self.sel_cat_star_markers2)

        # centroid star markers (main window)
        self.centroid_star_markers = pg.ScatterPlotItem()
        self.centroid_star_markers.setPen((255, 165, 0), width=2)
        self.centroid_star_markers.setSize(20)
        self.centroid_star_markers.setSymbol(Plus())
        self.centroid_star_markers.setZValue(4)
        self.img_frame.addItem(self.centroid_star_markers)

        # centroid star markers (zoom window)
        self.centroid_star_markers2 = pg.ScatterPlotItem()
        self.centroid_star_markers2.setPen((255, 165, 0), width=2)
        self.centroid_star_markers2.setSize(20)
        self.centroid_star_markers2.setSymbol(Plus())
        self.centroid_star_markers2.setZValue(4)
        self.zoom_window.addItem(self.centroid_star_markers2)

        self.draw_calstars = True

        # calstar markers (main window)
        self.calstar_markers = pg.ScatterPlotItem()
        self.calstar_markers.setPen((0, 255, 0, 100))
        self.calstar_markers.setBrush((0, 0, 0, 0))
        self.calstar_markers.setSize(10)
        self.calstar_markers.setSymbol('o')
        self.calstar_markers.setZValue(2)
        self.img_frame.addItem(self.calstar_markers)

        # calstar markers (zoom window)
        self.calstar_markers2 = pg.ScatterPlotItem()
        self.calstar_markers2.setPen((0, 255, 0, 100))
        self.calstar_markers2.setBrush((0, 0, 0, 0))
        self.calstar_markers2.setSize(20)
        self.calstar_markers2.setSymbol('o')
        self.calstar_markers2.setZValue(5)
        self.zoom_window.addItem(self.calstar_markers2)

        # pick markers (manual reduction)
        self.pick_marker = pg.ScatterPlotItem()
        self.pick_marker.setSymbol(Plus())
        self.pick_marker.setZValue(5)
        self.img_frame.addItem(self.pick_marker)

        # pick marker (manual reduction - zoom window)
        self.pick_marker2 = pg.ScatterPlotItem()
        self.pick_marker2.setSymbol(Plus())
        self.pick_marker2.setZValue(5)
        self.zoom_window.addItem(self.pick_marker2)

        # Star pick info
        text_str = "STAR PICKING MODE\n"
        text_str += "LEFT CLICK - Centroid star\n"
        text_str += "CTRL + LEFT CLICK - Manual star position\n"
        text_str += "CTRL + SCROLL - Aperture radius adjust\n"
        text_str += "CTRL + Z - Fit stars\n"
        text_str += "CTRL + SHIFT + Z - Fit with initial distortion params set to 0\n"
        text_str += "L - Astrometry fit details\n"
        text_str += "P - Photometry fit"
        self.star_pick_info = TextItem(text_str, anchor=(0.5, 0.5), color=(255, 255, 255))
        self.star_pick_info.setAlign(QtCore.Qt.AlignCenter)
        self.star_pick_info.hide()
        self.star_pick_info.setZValue(10)
        self.star_pick_info.setParentItem(self.img_frame)
        self.star_pick_info.setPos(self.platepar.X_res/2, self.platepar.Y_res - 55)

        # Default variables even when constructor isnt called
        self.star_pick_mode = False

        # Cursor
        self.cursor = CursorItem(self.star_aperature_radius, pxmode=True)
        self.img_frame.addItem(self.cursor, ignoreBounds=True)
        self.cursor.hide()
        self.cursor.setZValue(20)

        # Cursor (window)
        self.cursor2 = CursorItem(self.star_aperature_radius, pxmode=True, thickness=2)
        self.zoom_window.addItem(self.cursor2, ignoreBounds=True)
        self.cursor2.hide()
        self.cursor2.setZValue(20)

        # Distortion lines (window)
        self.draw_distortion = False
        self.distortion_lines = pg.PlotCurveItem(connect='pairs', pen=(255, 255, 0, 200))
        self.distortion_lines.hide()
        self.img_frame.addItem(self.distortion_lines)
        self.distortion_lines.setZValue(2)

        # Celestial grid
        self.grid_visible = 1
        self.celestial_grid = pg.PlotCurveItem(pen=pg.mkPen((255, 255, 255, 150), style=QtCore.Qt.DotLine))
        self.celestial_grid.setZValue(1)
        self.img_frame.addItem(self.celestial_grid)


        # Fit residuals (image, orange)
        self.residual_lines_img = pg.PlotCurveItem(connect='pairs', pen=pg.mkPen((255, 128, 0),
                                                                             style=QtCore.Qt.DashLine))
        self.img_frame.addItem(self.residual_lines_img)
        self.residual_lines_img.setZValue(2)
        
        # Fit residuals (astrometric, yellow)
        self.residual_lines_astro = pg.PlotCurveItem(connect='pairs', pen=pg.mkPen((255, 255, 0),
                                                                             style=QtCore.Qt.DashLine))
        self.img_frame.addItem(self.residual_lines_astro)
        self.residual_lines_astro.setZValue(2)


        # Text
        self.stdev_text_filter = 0
        self.residual_text = TextItemList()
        self.img_frame.addItem(self.residual_text)
        self.residual_text.setZValue(10)

        ###################################################################################################
        # RIGHT WIDGET


        # If the file is being loaded, detect the input type
        if loaded_file and (not hasattr(self, "img_handle")):

            if hasattr(self, "beginning_time"):
                beginning_time = self.beginning_time
            else:
                beginning_time = None

            # Detect data input type and init the image handle
            self.detectInputType(load=True, beginning_time=beginning_time)


        # adding img
        gamma = 1
        invert = False
        self.img_type_flag = 'avepixel'
        self.img = ImageItem(img_handle=self.img_handle, gamma=gamma, invert=invert)
        self.img_frame.addItem(self.img)
        self.img_frame.autoRange(padding=0)

        self.fr_box = QtWidgets.QGraphicsRectItem()
        self.fr_box.setPen(QtGui.QColor(255, 0, 0, 255))
        self.fr_box.setBrush(QtGui.QColor(0, 0, 0, 0))
        self.fr_box.hide()
        self.img_frame.addItem(self.fr_box)

        self.img_zoom = ImageItem(img_handle=self.img_handle, gamma=gamma, invert=invert)
        self.zoom_window.addItem(self.img_zoom)

        lut = np.array([[0, 0, 0, 0], [0, 255, 0, 76]], dtype=np.ubyte)
        self.region = pg.ImageItem(lut=lut)
        self.region.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)
        self.region.setZValue(10)
        self.img_frame.addItem(self.region)

        self.tab = RightOptionsTab(self)
        self.tab.hist.setImageItem(self.img)
        self.tab.hist.setImages(self.img_zoom)
        self.tab.hist.setLevels(0, 2**(8*self.img.data.itemsize) - 1)

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
        self.tab.param_manager.sigEqAspectToggled.connect(self.onFitParametersChanged)
        self.tab.param_manager.sigForceDistortionToggled.connect(self.onFitParametersChanged)

        # Connect astronmetry & photometry buttons to functions
        self.tab.param_manager.sigFitPressed.connect(lambda: self.fitPickedStars(first_platepar_fit=False))
        self.tab.param_manager.sigPhotometryPressed.connect(lambda: self.photometry(show_plot=True))
        self.tab.param_manager.sigAstrometryPressed.connect(self.showAstrometryFitPlots)
        self.tab.param_manager.sigResetDistortionPressed.connect(self.resetDistortion)

        self.tab.settings.sigMaxAveToggled.connect(self.toggleImageType)
        self.tab.settings.sigCatStarsToggled.connect(self.toggleShowCatStars)
        self.tab.settings.sigCalStarsToggled.connect(self.toggleShowCalStars)
        self.tab.settings.sigSelStarsToggled.connect(self.toggleShowSelectedStars)
        self.tab.settings.sigPicksToggled.connect(self.toggleShowPicks)
        self.tab.settings.sigRegionToggled.connect(self.toggleShowRegion)
        self.tab.settings.sigDistortionToggled.connect(self.toggleDistortion)
        self.tab.settings.sigGridToggled.connect(self.onGridChanged)
        self.tab.settings.sigInvertToggled.connect(self.toggleInvertColours)

        layout.addWidget(self.tab, 0, 2)

        ###################################################################################################
        # SETUP

        # mouse binding
        self.img_frame.scene().sigMouseMoved.connect(self.onMouseMoved)
        self.img_frame.sigMouseReleased.connect(self.onMouseReleased)
        self.img_frame.sigMousePressed.connect(self.onMousePressed)
        self.img_frame.sigResized.connect(self.onFrameResize)

        self.setMinimumSize(1200, 800)
        self.show()

        self.updateLeftLabels()
        self.updateStars()
        self.updateDistortion()
        self.tab.param_manager.updatePlatepar()
        self.changeMode(self.mode)


    def changeMode(self, new_mode):
        """
        Changes the mode to either 'skyfit' or 'manualreduction', updating the gui accordingly. Will not 
        update image if the mode stays the same.

        Arguments:
            new_mode [str]: either 'skyfit' or 'manualreduction'

        """
        # Won't update image if not necessary
        if self.mode == new_mode:
            first_time = True
        else:
            first_time = False


        if new_mode == 'skyfit':

            self.mode = 'skyfit'
            self.skyfit_button.setDisabled(True)
            self.manualreduction_button.setDisabled(False)
            self.setWindowTitle('SkyFit')

            self.updateLeftLabels()
            self.tab.onSkyFit()
            self.pick_marker.hide()
            self.pick_marker2.hide()
            self.region.hide()
            self.img_frame.setMouseEnabled(True, True)
            self.star_pick_mode = False
            self.cursor.hide()
            self.cursor2.hide()
            self.fr_box.hide()

            if not first_time and self.img.img_handle.input_type != 'dfn':
                self.img_zoom.loadImage(self.mode, self.img_type_flag)
                self.img.loadImage(self.mode, self.img_type_flag)

            for action in self.file_menu.actions():
                self.file_menu.removeAction(action)

            for action in self.view_menu.actions():
                self.view_menu.removeAction(action)

            self.file_menu.addActions([self.new_platepar_action,
                                       self.load_platepar_action,
                                       self.save_platepar_action,
                                       self.save_default_platepar_action,
                                       self.save_state_platepar_action,
                                       self.load_state_action,
                                       self.station_action])

            self.view_menu.addActions([self.toggle_info_action,
                                       self.toggle_zoom_window])

            text_str = "STAR PICKING MODE\n"
            text_str += "LEFT CLICK - Centroid star\n"
            text_str += "CTRL + LEFT CLICK - Manual star position\n"
            text_str += "CTRL + SCROLL - Aperture radius adjust\n"
            text_str += "CTRL + Z - Fit stars\n"
            text_str += "CTRL + SHIFT + Z - Fit with initial distortion params set to 0\n"
            text_str += "L - Astrometry fit details\n"
            text_str += "P - Photometry fit"
            self.star_pick_info.setText(text_str)

        else:
            self.mode = 'manualreduction'
            self.skyfit_button.setDisabled(False)
            self.manualreduction_button.setDisabled(True)
            self.setWindowTitle('ManualReduction')

            self.img_type_flag = 'avepixel'
            self.tab.settings.updateMaxAvePixel()
            self.img_zoom.loadImage(self.mode, self.img_type_flag)
            self.img.loadImage(self.mode, self.img_type_flag)
            self.showFRBox()

            for action in self.file_menu.actions():
                self.file_menu.removeAction(action)

            for action in self.view_menu.actions():
                self.view_menu.removeAction(action)

            self.file_menu.addActions([self.save_reduction_action,
                                       self.save_current_frame_action,
                                       self.load_platepar_action,
                                       self.load_state_action])

            self.view_menu.addActions([self.toggle_info_action,
                                       self.toggle_zoom_window])
            self.star_pick_info.setText('')

            self.updateLeftLabels()
            # self.show_zoom_window = False
            # self.zoom_window.hide()

            self.resetStarPick()
            self.img_frame.setMouseEnabled(True, True)
            self.star_pick_info.hide()
            self.star_pick_mode = False
            self.cursor.hide()
            self.cursor2.hide()
            self.tab.onManualReduction()
            self.pick_marker.show()
            self.pick_marker2.show()
            self.region.show()

    def changeStation(self):
        """
        Opens folder explorer window for user to select new station folder, then loads a platepar from that
        folder, and reads the config file, updating the gui to show what it should
        """
        dir_path = openFolderDialog(os.path.dirname(self.dir_path), 'Select new station folder', matplotlib)
        if not dir_path:
            return

        self.dir_path = dir_path
        self.config = cr.loadConfigFromDirectory('.', self.dir_path)
        self.detectInputType()
        self.catalog_stars = self.loadCatalogStars(self.config.catalog_mag_limit)
        self.cat_lim_mag = self.config.catalog_mag_limit
        self.loadCalstars()
        self.loadPlatepar(update=True)
        print()

        self.img.changeHandle(self.img_handle)
        self.img_zoom.changeHandle(self.img_handle)
        self.tab.hist.setLevels(0, 2**(8*self.img.data.itemsize) - 1)
        self.img_frame.autoRange(padding=0)

        self.paired_stars = []
        self.updatePairedStars()
        self.pick_list = {}
        self.residuals = None
        self.updateFitResiduals()
        self.updatePicks()
        self.drawPhotometryColoring()
        self.photometry()

        self.updateLeftLabels()
        self.tab.debruijn.updateTable()

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

    def onFrameResize(self):
        """ What happens when the window is resized. """

        self.label2.setPos(self.img_frame.width() - self.label2.boundingRect().width(), \
            self.img_frame.height() - self.label2.boundingRect().height())
        self.label_f1.setPos(self.img_frame.width() - self.label_f1.boundingRect().width(), \
            self.img_frame.height() - self.label_f1.boundingRect().height())

        self.star_pick_info.setPos(self.img_frame.width()/2, self.img_frame.height() - 50)

        if self.config.height/self.config.width < self.img_frame.height()/self.img_frame.width():
            self.img_frame.setLimits(xMin=0,
                                     xMax=self.config.width,
                                     yMin=None,
                                     yMax=None)
        else:
            self.img_frame.setLimits(xMin=None,
                                     xMax=None,
                                     yMin=0,
                                     yMax=self.config.height)

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
            time_data = [self.img.img_handle.currentTime()]

            # Compute RA, dec
            jd, ra, dec, _ = xyToRaDecPP(time_data, [x], [y], [1], self.platepar, extinction_correction=False)

            # Compute alt, az
            azim, alt = trueRaDec2ApparentAltAz(ra[0], dec[0], jd[0], self.platepar.lat, self.platepar.lon, \
                                                self.platepar.refraction)

            status_str += ",  Azim={:6.2f}  Alt={:6.2f} (date), RA={:6.2f}  Dec={:+6.2f} (J2000)".format(
                azim, alt, ra[0], dec[0])

        return status_str

    def zoom(self):
        """ Update the zoom window to zoom on the correct position """
        self.zoom_window.autoRange()
        # zoom_scale = 0.1
        # self.zoom_window.scaleBy(zoom_scale, QPoint(*self.mouse))
        self.zoom_window.setXRange(self.mouse_x - 20, self.mouse_x + 20)
        self.zoom_window.setYRange(self.mouse_y - 20, self.mouse_y + 20)

    def updateBottomLabel(self):
        """ Update bottom label with current mouse position """
        self.status_bar.showMessage(self.mouseOverStatus(self.mouse_x, self.mouse_y))

    def updateLeftLabels(self):
        """ Update the two labels on the left with their information """

        if self.mode == 'skyfit':
            ra_centre, dec_centre = self.computeCentreRADec()

            # Show text on image with platepar parameters
            text_str = "Station: {:s} \n".format(self.platepar.station_code)
            text_str += self.img_handle.name() + '\n\n'
            text_str += self.img_type_flag + '\n'
            text_str += u'Ref Az   = {:.3f}\N{DEGREE SIGN}\n'.format(self.platepar.az_centre)
            text_str += u'Ref Alt  = {:.3f}\N{DEGREE SIGN}\n'.format(self.platepar.alt_centre)
            text_str += u'Rot horiz = {:.3f}\N{DEGREE SIGN}\n'.format(rotationWrtHorizon(self.platepar))
            text_str += u'Rot eq    = {:.3f}\N{DEGREE SIGN}\n'.format(rotationWrtStandard(self.platepar))
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
            text_str += u'Dec centre = {:.3f}\N{DEGREE SIGN}'.format(dec_centre)

        else:
            text_str = "Station: {:s} \n".format(self.platepar.station_code)
            text_str += self.img_handle.name() + '\n\n'
            text_str += self.img_type_flag + '\n'
            text_str += "Time  = {:s}\n".format(
                self.img_handle.currentFrameTime(dt_obj=True).strftime("%Y/%m/%d %H:%M:%S.%f")[:-3])
            text_str += 'Frame = {:d}\n'.format(self.img.getFrame())
            text_str += 'Image gamma = {:.2f}\n'.format(self.img.gamma)
            text_str += 'Camera gamma = {:.2f}\n'.format(self.config.gamma)
            text_str += 'Refraction = {:s}'.format(str(self.platepar.refraction))

        self.label1.setText(text_str)

        if self.mode == 'skyfit':
            text_str = 'Keys:\n'
            text_str += '-----\n'
            text_str += 'F1 - Hide/show this text\n'
            text_str += 'Left/Right - Previous/next image\n'
            text_str += 'CTRL + Left/Right - +/- 10 images\n'
            text_str += 'A/D - Azimuth\n'
            text_str += 'S/W - Altitude\n'
            text_str += 'Q/E - Position angle\n'
            text_str += 'Up/Down - Scale\n'
            text_str += 'T - Toggle refraction correction\n'

            # Add aspect info if the radial distortion is used
            if not self.platepar.distortion_type.startswith("poly"):
                text_str += 'G - Toggle equal aspect\n'
                text_str += 'Y - Toggle asymmetry correction\n'
                text_str += 'B - Dist = img centre toggle\n'

            text_str += '1/2 - X offset\n'
            text_str += '3/4 - Y offset\n'
            text_str += '5/6 - X 1st dist. coeff.\n'
            text_str += '7/8 - Y 1st dist. coeff.\n'
            text_str += '9/0 - extinction scale\n'
            text_str += 'CTRL + 1 - poly3+radial distortion\n'
            text_str += 'CTRL + 2 - poly3+radial3 distortion\n'
            text_str += 'CTRL + 3 - radial3 distortion\n'
            text_str += 'CTRL + 4 - radial5 distortion\n'
            text_str += 'CTRL + 5 - radial7 distortion\n'
            text_str += 'CTRL + 6 - radial9 distortion\n'
            text_str += '\n'
            text_str += 'CTRL + R - Pick stars\n'
            text_str += '\n'
            text_str += 'Scroll - zoom in/out\n'
            text_str += 'R/F - Lim mag\n'
            text_str += '+/- - Increment adjust\n'
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
            text_str += 'SHIFT + Z - Show zoomed window\n'
            text_str += 'CTRL + N - New platepar\n'
            text_str += 'CTRL + S - Save platepar & state'
        else:
            text_str = 'Keys:\n'
            text_str += '-----------\n'
            text_str += 'F1 - Hide/show this text\n'
            text_str += 'Left/Right - Previous/next frame\n'
            text_str += 'CTRL + Left/Right - +/- 10 frames\n'
            text_str += 'Down/Up - +/- 25 frames\n'
            text_str += ',/. - Previous/next FR line\n'
            text_str += '\n'
            text_str += 'CTRL + R - Pick points\n'
            text_str += 'Left click - Centroid\n'
            text_str += 'CTRL + Left click - Force pick\n'
            text_str += 'ALT + Left click - Mark gap (DFN)\n'
            text_str += '\n'
            text_str += 'Scroll - zoom in/out\n'
            text_str += 'M - Show maxpixel\n'
            text_str += 'K - Subtract average\n'
            text_str += 'T - Toggle refraction correction\n'
            text_str += 'U/J - Img Gamma\n'
            text_str += '\n'
            text_str += 'P - Show lightcurve\n'
            text_str += 'CTRL + A - Auto levels\n'
            text_str += 'CTRL + D - Load dark\n'
            text_str += 'CTRL + F - Load flat\n'
            text_str += 'CTRL + P - Load platepar\n'
            text_str += 'CTRL + W - Save current frame\n'
            text_str += 'CTRL + S - Save FTPdetectinfo'

        self.label2.setText(text_str)
        self.label2.setPos(self.img_frame.width() - self.label2.boundingRect().width(), \
            self.img_frame.height() - self.label2.boundingRect().height())


        # F1 info label which will be shown when labels 1 and 2 are hidden
        self.label_f1.setText("F1 - Show hotkeys")
        self.label_f1.setPos(self.img_frame.width() - self.label_f1.boundingRect().width(), \
            self.img_frame.height() - self.label_f1.boundingRect().height())

    def updateStars(self):
        """ Updates only the stars, including catalog stars, calstars and paired stars """


        # Draw stars that were paired in picking mode
        self.updatePairedStars()
        self.onGridChanged()  # for ease of use

        # Draw stars detected on this image
        if self.draw_calstars:
            self.updateCalstars()

        ### Draw catalog stars on the image using the current platepar ###
        ######################################################################################################

        # Get positions of catalog stars on the image
        ff_jd = date2JD(*self.img_handle.currentTime())
        self.catalog_x, self.catalog_y, catalog_mag = getCatalogStarsImagePositions(self.catalog_stars,
                                                                                    ff_jd, self.platepar)

        cat_stars_xy = np.c_[self.catalog_x, self.catalog_y, catalog_mag]

        ### Take only those stars inside the FOV  and iamge ###

        # Get indices of stars inside the fov
        filtered_indices, _ = self.filterCatalogStarsInsideFOV(self.catalog_stars)

        # Create a mask to filter out all stars outside the image and the FOV
        filter_indices_mask = np.zeros(len(cat_stars_xy), dtype=np.bool)
        filter_indices_mask[filtered_indices] = True
        filtered_indices_all = filter_indices_mask & (cat_stars_xy[:, 0] > 0) \
                                                & (cat_stars_xy[:, 0] < self.platepar.X_res) \
                                                & (cat_stars_xy[:, 1] > 0) \
                                                & (cat_stars_xy[:, 1] < self.platepar.Y_res)

        # Filter out catalog image stars
        cat_stars_xy = cat_stars_xy[filtered_indices_all]

        # Create a filtered catalog
        self.catalog_stars_filtered = self.catalog_stars[filtered_indices_all]

        # Create a list of filtered catalog image coordinates
        self.catalog_x_filtered, self.catalog_y_filtered, catalog_mag_filtered = cat_stars_xy.T

        # Show stars on the image
        if self.catalog_stars_visible:

            # Only show if there are any stars to show
            if len(catalog_mag_filtered):

                cat_mag_faintest = np.max(catalog_mag_filtered)

                # Plot catalog stars
                size = ((4.0 + (cat_mag_faintest - catalog_mag_filtered))/2.0)**(2*2.512*0.5)

                self.cat_star_markers.setPoints(x=self.catalog_x_filtered, y=self.catalog_y_filtered, \
                    size=size)
                self.cat_star_markers2.setPoints(x=self.catalog_x_filtered, y=self.catalog_y_filtered, \
                    size=size)
            else:
                print('No catalog stars visible!')


    def updatePairedStars(self):
        """
            Draws the stars that were picked for calibration as well as draw the
            residuals and star magnitude
        """
        if len(self.paired_stars) > 0:
            self.sel_cat_star_markers.setData(pos=[pair[0][:2] for pair in self.paired_stars])
            self.sel_cat_star_markers2.setData(pos=[pair[0][:2] for pair in self.paired_stars])
        else:
            self.sel_cat_star_markers.setData(pos=[])
            self.sel_cat_star_markers2.setData(pos=[])

        self.centroid_star_markers.setData(pos=[])
        self.centroid_star_markers2.setData(pos=[])

        # Draw photometry
        if len(self.paired_stars) > 2:
            self.photometry()

        self.tab.param_manager.updatePairedStars()


    def updateCalstars(self):
        """ Draw extracted stars on the current image. """

        # Check if the given FF files is in the calstars list
        if self.img_handle.name() in self.calstars:
            # Get the stars detected on this FF file
            star_data = self.calstars[self.img_handle.name()]

            # Get star coordinates
            y, x, _, _ = np.array(star_data).T

            self.calstar_markers.setPoints(x=x, y=y)
            self.calstar_markers2.setPoints(x=x, y=y)
        else:
            self.calstar_markers.setData(pos=[])
            self.calstar_markers2.setData(pos=[])


    def updatePicks(self):
        """ Draw pick markers for manualreduction """

        red = pg.mkPen((255, 0, 0))
        yellow = pg.mkPen((255, 255, 0))

        current = []
        current_pen = red

        data1 = []
        data2 = []
        for frame, pick in self.pick_list.items():
            if pick['x_centroid']:
                if self.img.getFrame() == frame:
                    current = [(pick['x_centroid'], pick['y_centroid'])]
                    if pick['mode'] == 1:
                        current_pen = red
                    else:
                        current_pen = yellow

                elif pick['mode'] == 1:
                    data1.append([pick['x_centroid'], pick['y_centroid']])

                else:
                    data2.append([pick['x_centroid'], pick['y_centroid']])

        plus = Plus()

        self.pick_marker.clear()
        self.pick_marker.addPoints(pos=current, size=30, pen=current_pen)
        self.pick_marker.addPoints(pos=data1, size=10, pen=red)
        self.pick_marker.addPoints(pos=data2, size=10, pen=yellow)

        self.pick_marker2.clear()
        self.pick_marker2.addPoints(pos=current, size=30, pen=current_pen)
        self.pick_marker2.addPoints(pos=data1, size=10, pen=red)
        self.pick_marker2.addPoints(pos=data2, size=10, pen=yellow)


    def updateFitResiduals(self):
        """ Draw fit residual lines. """

        if self.residuals is not None:

            x1 = []
            y1 = []

            x2 = []
            y2 = []

            # Plot the residuals (enlarge 100x)
            res_scale = 100
            for entry in self.residuals:
                img_x, img_y, angle, distance, angular_distance = entry


                ### Limit the distance to the edge of the image ###
                # All angles are reference to a line pointing right, angles increase clockwise

                # Residual angle
                ang_test = angle%(2*np.pi)

                # Compute the angles of every corner relative to the point
                ul_ang = np.arctan2(                  0 - img_y,                   0 - img_x)%(2*np.pi)
                ur_ang = np.arctan2(                  0 - img_y, self.platepar.X_res - img_x)%(2*np.pi)
                ll_ang = np.arctan2(self.platepar.Y_res - img_y,                   0 - img_x)%(2*np.pi)
                lr_ang = np.arctan2(self.platepar.Y_res - img_y, self.platepar.X_res - img_x)%(2*np.pi)


                # Locate the point in the correct quadrant and compute the distance to the edge of the image
                dist_side = distance
                if   (ang_test > ul_ang) and (ang_test < ur_ang):
                    # Upper side
                    # Compute the distance to the side of the image
                    dist_side = abs(img_y/np.cos(ang_test - 3/2*np.pi))

                elif (ang_test > ur_ang) or (ang_test < lr_ang):
                    # Right side
                    dist_side = abs((self.platepar.X_res - img_x)/np.cos(ang_test))

                elif (ang_test > lr_ang) and (ang_test < ll_ang):
                    # Bottom side
                    dist_side = abs((self.platepar.Y_res - img_y)/np.cos(ang_test - np.pi/2))

                else:
                    # Left side
                    dist_side = abs(img_x/np.cos(ang_test - np.pi))


                # Limit the distance for plotting to the side of the image
                distance_plot = min([res_scale*distance, dist_side])

                ### ###

                # Calculate coordinates of the end of the residual line
                res_x = img_x + np.cos(angle)*distance_plot
                res_y = img_y + np.sin(angle)*distance_plot

                # Save image residuals
                x1.extend([img_x, res_x])
                y1.extend([img_y, res_y])

                # Convert the angular distance from degrees to equivalent image pixels
                ang_dist_img = angular_distance*self.platepar.F_scale
                ang_dist_img_plot = min([res_scale*ang_dist_img, dist_side])
                res_x = img_x + np.cos(angle)*ang_dist_img_plot
                res_y = img_y + np.sin(angle)*ang_dist_img_plot

                # Save sky residuals
                x2.extend([img_x, res_x])
                y2.extend([img_y, res_y])

            self.residual_lines_img.setData(x=x1, y=y1)
            self.residual_lines_astro.setData(x=x2, y=y2)
        else:
            self.residual_lines_img.clear()
            self.residual_lines_astro.clear()


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


    def photometry(self, show_plot=False):
        """
        Perform the photometry on selected stars. Updates residual text above and below picked stars

        Arguments:
            show_plot: if true, will show a plot of the photometry

        """

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

            if self.selected_stars_visible:

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
                        text2.setColor(QtGui.QColor(0, 255, 0))
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
                    fit_resids_novignetting = catalog_mags - photomLine((np.array(px_intens_list), \
                                                                         np.array(radius_list)), \
                                                                         photom_offset, 0.0)
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


    def changeDistortionType(self):
        """ Change the distortion type. """

        dist_type = self.platepar.distortion_type_list[self.dist_type_index]
        self.platepar.setDistortionType(dist_type)
        self.updateDistortion()

        # Indicate that the platepar has been reset
        self.first_platepar_fit = True

        print("Distortion model changed to: {:s}".format(dist_type))


    def resetDistortion(self):
        """ Reset distortion parameters to default values. """

        self.platepar.resetDistortionParameters()
        self.onFitParametersChanged()
        self.updateFitResiduals()
        self.tab.param_manager.updatePlatepar()



    def nextImg(self, n=1):
        """
        Increments the image index by value n. n=1 will go to next image and n=-1
        will go to the previous. In manualreduction, nextImg will not change chunks
        but will change frames, and n can be any integer and the frame will increment
        by that much.

        Arguments:
            n [int]: The number of images to go forward or backward

        """
        
        if self.mode == 'skyfit':

            # Don't allow image change while in star picking mode
            if self.star_pick_mode:
                qmessagebox(title='Star picking mode', \
                            message='You cannot cycle through images while in star picking mode!',
                            message_type="warning")

                return None

            # don't change images if there's no image to change to
            if (self.img.img_handle.input_type == 'dfn') and (self.img.img_handle.total_images == 1):
                return None

            if n > 0:
                for _ in range(n):
                    self.img.nextChunk()

            elif n < 0:
                for _ in range(abs(n)):
                    self.img.prevChunk()

            self.img_zoom.loadImage(self.mode, self.img_type_flag)
            self.img.loadImage(self.mode, self.img_type_flag)

            # remove markers
            self.calstar_markers.setData(pos=[])
            self.calstar_markers2.setData(pos=[])
            self.cat_star_markers.setData(pos=[])
            self.cat_star_markers2.setData(pos=[])
            self.pick_marker.setData(pos=[])

            # Reset paired stars
            self.pick_list = {}
            self.paired_stars = []
            self.residuals = None
            self.drawPhotometryColoring()

            self.updateStars()

        # Manual reduction mode
        else:


            # For DFN or single images, only allow changing frames to +1 from the highest, and -1 from the 
            #   lowest frame number
            if (self.img.img_handle.input_type == 'dfn') or \
                ((self.img.img_handle.input_type == 'images') and self.img.img_handle.single_image_mode):

                change_allowed = True

                # If the pick list is empty, don't allow changing the frame
                if not len(self.pick_list):
                    change_allowed = False

                else:

                    # Get a range of frames
                    max_frame = max(self.pick_list.keys())
                    min_frame = min(self.pick_list.keys())
                    next_frame = self.img.getFrame() + n

                    # Only allow changing frames to adjecent ones to the max/min
                    if (next_frame < (min_frame - 1)) or (next_frame > (max_frame + 1)):
                        change_allowed = False


                if not change_allowed:

                    qmessagebox(title='Frame counter error',
                                message="The frame number on DFN images cannot be advanced if a pick was not made!",
                                message_type="info")

                    return None


            self.computeIntensitySum()

                
            # Change shown frame either by one or more
            if n == 1:
                self.img.nextFrame()
            else:
                self.img.setFrame(self.img.getFrame() + n)

            self.img_zoom.loadImage(self.mode, self.img_type_flag)
            self.img.loadImage(self.mode, self.img_type_flag)

            self.updatePicks()
            self.drawPhotometryColoring()
            self.showFRBox()



        self.updateLeftLabels()


    def showFRBox(self):
        """ On manual reduction, call this and a red rectangle will be drawn around the FR cutout """

        if self.img.img_handle.name().startswith('FR'):

            fr = self.img_handle.loadChunk()

            # Show the red box around the FR cutout if the current frame is within the frame range of the
            #   current line in the FR file
            if fr.t[self.img_handle.current_line][0] <= self.img.getFrame() \
                <= fr.t[self.img_handle.current_line][-1]:

                fr_no = self.img.getFrame() - fr.t[self.img_handle.current_line][0]
                
                x = int(fr.xc[self.img_handle.current_line][fr_no] - \
                        fr.size[self.img_handle.current_line][fr_no]/2)

                y = int(fr.yc[self.img_handle.current_line][fr_no] - \
                        fr.size[self.img_handle.current_line][fr_no]/2)

                h = fr.size[self.img_handle.current_line][fr_no]

                self.fr_box.setRect(x, y, h, h)
                self.fr_box.show()

            else:
                self.fr_box.hide()


    def saveState(self):
        """
        Saves the state of the object to a file so that when loading, it will appear the same as before.

        Can be loaded by calling:
        python -m RMS.Astrometry.SkyFit2 PATH/skyFit2_latest.state --config .
        """

        # This is pretty thrown together. It's to get around an error where pyqt widgets can't be saved to a
        # pickle. If there's a better way to do this, go ahead.

        # Currently, any important variables should be initialized in the constructor (and cannot be classes
        # that inherit). Anything that can be generated for that information should be done in setupUI.

        to_remove = []

        dic = copy.copy(self.__dict__)
        for k, v in dic.items():

            if (v.__class__.__bases__[0] is not object) and (not isinstance(v, bool)) and \
                (not isinstance(v, float)):

                # Remove class that inherits from something
                to_remove.append(k)  

        for remove in to_remove:
            del dic[remove]

        savePickle(dic, self.dir_path, 'skyFitMR_latest.state')
        print("Saved state to file")


    def findLoadState(self):
        """ Opens file dialog to find .state file to load then calls loadState """

        file = openFileDialog(self.dir_path, None, 'Load .state file', matplotlib, \
                              [('State File', '*.state'), \
                               ('All Files', '*')])

        if file:
            self.loadState(os.path.dirname(file), os.path.basename(file))


    def loadState(self, dir_path, state_name, beginning_time=None):
        """ Loads state with path to file dir_path and file name state_name. Works mid-program and at the start of
        the program (if done properly).

        Loaded state will not be identical to the previous, since saveState doesn't save all information.
        Variables initialized the constructor will be loaded (including self.platepar, self.pick_list and 
        others).

        Arguments:
            dir_path: [str] Path to directory (ex. C:/path).
            state_name: [str] File name (ex. file.state).

        Keyword arguments:
            beginning_time: [datetime] Datetime of the video beginning. Optional, only can be given for
                video input formats.

        """

        variables = loadPickle(dir_path, state_name)
        for k, v in variables.items():
            setattr(self, k, v)

        # updating old state files with new platepar variables
        if self.platepar is not None:
            if not hasattr(self.platepar, "equal_aspect"):
                self.platepar.equal_aspect = False

            if not hasattr(self.platepar, "force_distortion_centre"):
                self.platepar.force_distortion_centre = False

            if not hasattr(self.platepar, "asymmetry_corr"):
                self.platepar.asymmetry_corr = False

            if not hasattr(self.platepar, "extinction_scale"):
                self.platepar.extinction_scale = 1.0


            if not hasattr(self.platepar, "extinction_scale"):
                self.platepar.extinction_scale = 1.0


            # Update platepar distortion indices
            self.platepar.setDistortionType(self.platepar.distortion_type, reset_params=False)
            self.dist_type_index = self.platepar.distortion_type_list.index(self.platepar.distortion_type)

            # Update the array length if an old platepar version was loaded which was shorter
            self.platepar.padDictParams()



        # Update the platepar path
        if hasattr(self, "platepar_file"):
            if self.platepar_file is not None:

                # Extract the platepar name
                platepar_dir, platepar_name = os.path.split(self.platepar_file)

                # If the platepar dir is the same as the old dir path, replace it with the new dir path
                if os.path.realpath(self.dir_path) == os.path.realpath(platepar_dir):

                    # Update the path to the platepar
                    self.platepar_file = os.path.join(dir_path, platepar_name)


        # Set the dir path in case it changed
        self.dir_path = dir_path

        # Update the dir path in the img_handle
        if hasattr(self, "img_handle"):
            self.img_handle.dir_path = dir_path

        # Update possibly missing input_path variable
        if not hasattr(self, "input_path"):
            self.input_path = dir_path


        # Update the possibly missing begin time
        if not hasattr(self, "beginning_time"):
            self.beginning_time = beginning_time


        # If setupUI hasn't already been called, call it
        if not hasattr(self, 'central'):

            self.setupUI(loaded_file=True)

        else:
            
            # Get and set the new img_handle
            self.detectInputType(load=True)  
            self.img.changeHandle(self.img_handle)
            self.img_zoom.changeHandle(self.img_handle)

            self.tab.hist.setLevels(0, 2**(8*self.img.data.itemsize) - 1)
            self.img_frame.autoRange(padding=0)

            self.updateCalstars()
            self.updateStars()
            self.updateDistortion()
            self.updateFitResiduals()

            self.tab.param_manager.updatePlatepar()
            self.tab.debruijn.updateTable()
            self.changeMode(self.mode)

            self.updateLeftLabels()


    def onMouseReleased(self, event):
        self.clicked = 0


    def onMouseMoved(self, event):

        pos = event

        if self.img_frame.sceneBoundingRect().contains(pos):
            
            self.img_frame.setFocus()
            mp = self.img_frame.mapSceneToView(pos)

            self.cursor.setCenter(mp)
            self.cursor2.setCenter(mp)
            self.mouse_x, self.mouse_y = mp.x(), mp.y()

            self.zoom()

            # Move zoom window to correct location
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

            if self.clicked and self.cursor.mode == 2:
                self.changePhotometry(self.img.getFrame(), self.photometryColoring(),
                                      add_photometry=self.clicked == 1)
                self.drawPhotometryColoring()

        # self.printFrameRate()

    def onMousePressed(self, event):

        if event.button() == QtCore.Qt.LeftButton:
            self.clicked = 1
        elif event.button() == QtCore.Qt.MiddleButton:
            self.clicked = 2
        elif event.button() == QtCore.Qt.RightButton:
            self.clicked = 3

        modifiers = QtWidgets.QApplication.keyboardModifiers()
        
        if self.star_pick_mode:  # redundant

            # Add star pair in SkyFit
            if self.mode == 'skyfit':

                # Add star
                if event.button() == QtCore.Qt.LeftButton:
                    if self.cursor.mode == 0:

                        # If CTRL is pressed, place the pick manually - NOTE: the intensity might be off then!!!
                        if modifiers & QtCore.Qt.ControlModifier:
                            self.x_centroid = self.mouse_x
                            self.y_centroid = self.mouse_y

                            # Compute the star intensity
                            _, _, self.star_intensity = self.centroid(prev_x_cent=self.x_centroid,
                                                                      prev_y_cent=self.y_centroid)
                        else:
                            # Perform centroiding with 2 iterations
                            x_cent_tmp, y_cent_tmp, _ = self.centroid()

                            # Check that the centroiding was successful
                            if x_cent_tmp is not None:

                                # Centroid the star around the pressed coordinates
                                self.x_centroid, self.y_centroid, self.star_intensity = self.centroid(
                                    prev_x_cent=x_cent_tmp,
                                    prev_y_cent=y_cent_tmp)

                            else:
                                return None

                        self.centroid_star_markers.addPoints(x=[self.x_centroid], y=[self.y_centroid])
                        self.centroid_star_markers2.addPoints(x=[self.x_centroid], y=[self.y_centroid])

                        # Select the closest catalog star to the centroid as the first guess
                        self.closest_cat_star_indx = self.findClosestCatalogStarIndex(self.x_centroid,
                                                                                      self.y_centroid)
                        self.sel_cat_star_markers.addPoints(x=[self.catalog_x_filtered[self.closest_cat_star_indx]],
                                                            y=[self.catalog_y_filtered[self.closest_cat_star_indx]])
                        self.sel_cat_star_markers2.addPoints(x=[self.catalog_x_filtered[self.closest_cat_star_indx]],
                                                             y=[self.catalog_y_filtered[self.closest_cat_star_indx]])

                        # Switch to the mode where the catalog star is selected
                        self.cursor.setMode(1)

                    elif self.cursor.mode == 1:

                        # Select the closest catalog star
                        self.closest_cat_star_indx = self.findClosestCatalogStarIndex(self.mouse_x, \
                                                                                      self.mouse_y)

                        # REMOVE marker for previously selected
                        self.sel_cat_star_markers.setData(pos=[pair[0][:2] for pair in self.paired_stars])
                        self.sel_cat_star_markers2.setData(pos=[pair[0][:2] for pair in self.paired_stars])

                        self.sel_cat_star_markers.addPoints(x=[self.catalog_x_filtered[self.closest_cat_star_indx]],
                                                            y=[self.catalog_y_filtered[self.closest_cat_star_indx]])
                        self.sel_cat_star_markers2.addPoints(x=[self.catalog_x_filtered[self.closest_cat_star_indx]],
                                                             y=[self.catalog_y_filtered[self.closest_cat_star_indx]])

                # Remove star pair
                elif event.button() == QtCore.Qt.RightButton:
                    if self.cursor.mode == 0:

                        # Find the closest picked star
                        picked_indx = self.findClosestPickedStarIndex(self.mouse_x, self.mouse_y)

                        if self.paired_stars:
                            # Remove the picked star from the list
                            self.paired_stars.pop(picked_indx)

                        self.updatePairedStars()
                        self.updateFitResiduals()
                        self.photometry()

            # Add centroid in manual reduction
            else:
                if event.button() == QtCore.Qt.LeftButton:
                    if self.cursor.mode == 0:
                        mode = 1
                        if modifiers & QtCore.Qt.ControlModifier or \
                                ((modifiers & QtCore.Qt.AltModifier or QtCore.Qt.Key_0 in self.keys_pressed) and
                                 self.img.img_handle.input_type == 'dfn'):
                            self.x_centroid, self.y_centroid = self.mouse_x, self.mouse_y
                        else:
                            self.x_centroid, self.y_centroid, _ = self.centroid()

                        if (modifiers & QtCore.Qt.AltModifier or QtCore.Qt.Key_0 in self.keys_pressed) and \
                                self.img.img_handle.input_type == 'dfn':
                            mode = 0

                        self.addCentroid(self.img.getFrame(), self.x_centroid, self.y_centroid, mode=mode)

                        self.updatePicks()
                    elif self.cursor.mode == 2:
                        self.changePhotometry(self.img.getFrame(), self.photometryColoring(),
                                              add_photometry=True)
                        self.drawPhotometryColoring()

                elif event.button() == QtCore.Qt.RightButton:
                    if self.cursor.mode == 0:
                        self.removeCentroid(self.img.getFrame())
                        self.updatePicks()
                    elif self.cursor.mode == 2:
                        self.changePhotometry(self.img.getFrame(), self.photometryColoring(),
                                              add_photometry=False)
                        self.drawPhotometryColoring()

    def keyPressEvent(self, event):

        # Read modifiers (e.g. CTRL, SHIFT)
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        qmodifiers = QtWidgets.QApplication.queryKeyboardModifiers()

        self.keys_pressed.append(event.key())

        # Toggle auto levels
        if event.key() == QtCore.Qt.Key_A and (modifiers == QtCore.Qt.ControlModifier):
            
            self.tab.hist.toggleAutoLevels()
            # This updates image automatically

        # Load the dark
        elif event.key() == QtCore.Qt.Key_D and (modifiers == QtCore.Qt.ControlModifier):
            
            _, self.dark = self.loadDark()

            # Set focus back on the SkyFit window
            self.activateWindow()

            # Apply the dark to the flat
            if self.flat_struct is not None:
                self.flat_struct.applyDark(self.dark)

            self.img.dark = self.dark
            self.img_zoom.flat_struct = self.flat_struct
            self.img.flat_struct = self.flat_struct
            self.img_zoom.reloadImage()
            self.img.reloadImage()

        # Load the flat
        elif event.key() == QtCore.Qt.Key_F and (modifiers == QtCore.Qt.ControlModifier):
            
            _, self.flat_struct = self.loadFlat()

            # Set focus back on the SkyFit window
            self.activateWindow()

            # self.img.dark = self.dark
            self.img_zoom.flat_struct = self.flat_struct
            self.img.flat_struct = self.flat_struct
            self.img_zoom.reloadImage()
            self.img.reloadImage()

        # Toggle the star picking mode
        elif event.key() == QtCore.Qt.Key_R and (modifiers == QtCore.Qt.ControlModifier):
            self.star_pick_mode = not self.star_pick_mode
            if self.star_pick_mode:
                self.img_frame.setMouseEnabled(False, False)
                self.cursor.show()
                self.cursor2.show()

                self.star_pick_info.show()
            else:
                self.img_frame.setMouseEnabled(True, True)
                self.cursor2.hide()
                self.cursor.hide()

                self.star_pick_info.hide()
            # updates automatically

        # Toggle grid
        elif (event.key() == QtCore.Qt.Key_G) and (modifiers == QtCore.Qt.ControlModifier):
            self.grid_visible = (self.grid_visible + 1)%3
            self.onGridChanged()
            self.tab.settings.updateShowGrid()


        # Previous image/frame
        elif event.key() == QtCore.Qt.Key_Left:

            n = -1

            # Skip 10 images back if CTRL is pressed
            if modifiers == QtCore.Qt.ControlModifier:
                n = -10

            self.nextImg(n=n)


        # Next image/frame
        elif event.key() == QtCore.Qt.Key_Right:

            n = 1

            # Skip 10 images forward if CTRL is pressed
            if modifiers == QtCore.Qt.ControlModifier:
                n = 10

            self.nextImg(n=n)


        # Switch between maxpixel and avepixel
        elif event.key() == QtCore.Qt.Key_M:
            self.toggleImageType()
            self.tab.settings.updateMaxAvePixel()


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

        # Toggle refraction
        elif event.key() == QtCore.Qt.Key_T:
            if self.platepar is not None:
                self.platepar.refraction = not self.platepar.refraction

                self.onRefractionChanged()
                self.tab.param_manager.updatePlatepar()

        # Toggle showing the catalog stars
        elif event.key() == QtCore.Qt.Key_H:
            self.toggleShowCatStars()
            self.tab.settings.updateShowCatStars()

        # Toggle inverting colors
        elif (event.key() == QtCore.Qt.Key_I) and not (modifiers == QtCore.Qt.ControlModifier):
            self.toggleInvertColours()
            self.tab.settings.updateInvertColours()


        # Handle keys in the SkyFit mode
        elif self.mode == 'skyfit':

            # Change distortion type to poly3+radial
            if (event.key() == QtCore.Qt.Key_1) and (modifiers == QtCore.Qt.ControlModifier):

                self.dist_type_index = 0
                self.changeDistortionType()
                self.tab.param_manager.updatePlatepar()
                self.updateLeftLabels()
                self.updateDistortion()
                self.updateStars()

            # Change distortion type to poly3+radial3
            if (event.key() == QtCore.Qt.Key_2) and (modifiers == QtCore.Qt.ControlModifier):

                self.dist_type_index = 1
                self.changeDistortionType()
                self.tab.param_manager.updatePlatepar()
                self.updateLeftLabels()
                self.updateDistortion()
                self.updateStars()

            # Change distortion type to radial3
            elif (event.key() == QtCore.Qt.Key_3) and (modifiers == QtCore.Qt.ControlModifier):

                self.dist_type_index = 2
                self.changeDistortionType()
                self.tab.param_manager.updatePlatepar()
                self.updateLeftLabels()
                self.updateDistortion()
                self.updateStars()

            # Change distortion type to radial5
            elif (event.key() == QtCore.Qt.Key_4) and (modifiers == QtCore.Qt.ControlModifier):

                self.dist_type_index = 3
                self.changeDistortionType()
                self.tab.param_manager.updatePlatepar()
                self.updateLeftLabels()
                self.updateDistortion()
                self.updateStars()

            # Change distortion type to radial7
            elif (event.key() == QtCore.Qt.Key_5) and (modifiers == QtCore.Qt.ControlModifier):

                self.dist_type_index = 4
                self.changeDistortionType()
                self.tab.param_manager.updatePlatepar()
                self.updateLeftLabels()
                self.updateDistortion()
                self.updateStars()

            # Change distortion type to radial7
            elif (event.key() == QtCore.Qt.Key_6) and (modifiers == QtCore.Qt.ControlModifier):

                self.dist_type_index = 5
                self.changeDistortionType()
                self.tab.param_manager.updatePlatepar()
                self.updateLeftLabels()
                self.updateDistortion()
                self.updateStars()

            # Make new platepar
            elif (event.key() == QtCore.Qt.Key_N) and (modifiers == QtCore.Qt.ControlModifier):
                self.makeNewPlatepar()

            # Show distortion
            elif (event.key() == QtCore.Qt.Key_I) and (modifiers == QtCore.Qt.ControlModifier):
                self.toggleDistortion()
                self.tab.settings.updateShowDistortion()


            # Do a fit on the selected stars while in the star picking mode
            elif (event.key() == QtCore.Qt.Key_Z) and ((modifiers == QtCore.Qt.ControlModifier) \
                or (modifiers == (QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier))):

                # If shift was pressed, reset distortion parameters to zero
                if modifiers == (QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier):
                    print("Resetting the distortion coeffs and refitting...")
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

            # Increase reference azimuth
            elif event.key() == QtCore.Qt.Key_A:
                self.platepar.az_centre += self.key_increment
                
                self.checkParamRange()
                self.platepar.updateRefRADec(preserve_rotation=True)
                self.checkParamRange()

                self.tab.param_manager.updatePlatepar()
                self.updateLeftLabels()
                self.updateStars()

            # Decrease reference azimuth
            elif event.key() == QtCore.Qt.Key_D:
                self.platepar.az_centre -= self.key_increment

                self.checkParamRange()
                self.platepar.updateRefRADec(preserve_rotation=True)
                self.checkParamRange()

                self.tab.param_manager.updatePlatepar()
                self.updateLeftLabels()
                self.updateStars()

            # Increase reference altitude
            elif event.key() == QtCore.Qt.Key_W:
                self.platepar.alt_centre -= self.key_increment

                self.checkParamRange()
                self.platepar.updateRefRADec(preserve_rotation=True)
                self.checkParamRange()

                self.tab.param_manager.updatePlatepar()
                self.updateLeftLabels()
                self.updateStars()

            # Decrease reference altitude
            elif event.key() == QtCore.Qt.Key_S:
                self.platepar.alt_centre += self.key_increment

                self.checkParamRange()
                self.platepar.updateRefRADec(preserve_rotation=True)
                self.checkParamRange()

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


            # Change extintion scale
            elif event.key() == QtCore.Qt.Key_9:
                self.platepar.extinction_scale += 0.1

                self.tab.param_manager.updatePlatepar()
                self.onExtinctionChanged()

            elif event.key() == QtCore.Qt.Key_0:
                self.platepar.extinction_scale -= 0.1

                self.tab.param_manager.updatePlatepar()
                self.onExtinctionChanged()


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

                # Load the new FOV centre
                data = self.getFOVcentre()

                if data is not None:

                    self.platepar.RA_d, self.platepar.dec_d, self.platepar.rotation_from_horiz = data

                    # Compute reference Alt/Az to apparent coordinates, epoch of date
                    self.platepar.az_centre, self.platepar.alt_centre = trueRaDec2ApparentAltAz( \
                        self.platepar.RA_d, self.platepar.dec_d, self.platepar.JD, \
                        self.platepar.lat, self.platepar.lon, self.platepar.refraction)

                    # Compute the position angle
                    self.platepar.pos_angle_ref = rotationWrtHorizonToPosAngle(self.platepar, \
                        self.platepar.rotation_from_horiz)

                    # Check that the calibration parameters are within the nominal range
                    self.checkParamRange()

                    self.tab.param_manager.updatePlatepar()
                    self.updateLeftLabels()
                    self.updateStars()


            # Toggle equal aspect ratio for radial distortions
            elif event.key() == QtCore.Qt.Key_G:

                if self.platepar is not None:

                    self.platepar.equal_aspect = not self.platepar.equal_aspect

                    self.tab.param_manager.updatePlatepar()
                    self.updateLeftLabels()
                    self.updateStars()


            # Toggle asymmetry correction for radial distortions
            elif event.key() == QtCore.Qt.Key_Y:

                if self.platepar is not None:

                    self.platepar.asymmetry_corr = not self.platepar.asymmetry_corr

                    self.tab.param_manager.updatePlatepar()
                    self.updateLeftLabels()
                    self.updateStars()


            # Get initial parameters from astrometry.net
            elif (event.key() == QtCore.Qt.Key_X) and ((modifiers == QtCore.Qt.ControlModifier) \
                or (modifiers == (QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier))):

                print("Solving with astrometry.net")

                upload_image = True

                # If shift was pressed, only upload the detected stars
                if modifiers == (QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier):
                    upload_image = False

                # Estimate initial parameters using astrometry.net
                self.getInitialParamsAstrometryNet(upload_image=upload_image)

                self.updateDistortion()
                self.updateLeftLabels()
                self.updateStars()
                self.tab.param_manager.updatePlatepar()


            # Toggle showing detected stars
            elif event.key() == QtCore.Qt.Key_C:
                self.toggleShowCalStars()
                self.tab.settings.updateShowCalStars()
                # updates image automatically


            # Force distortion centre to image centre
            elif event.key() == QtCore.Qt.Key_B:
                if self.platepar is not None:
                    self.platepar.force_distortion_centre = not self.platepar.force_distortion_centre

                    self.tab.param_manager.updatePlatepar()
                    self.updateStars()
                    self.updateLeftLabels()


            elif (event.key() == QtCore.Qt.Key_Return) or (event.key() == QtCore.Qt.Key_Enter):
                
                if self.star_pick_mode:
                    
                    # If the right catalog star has been selected, save the pair to the list
                    if self.cursor.mode == 1:

                        # Add the image/catalog pair to the list
                        self.paired_stars.append([[self.x_centroid, self.y_centroid, self.star_intensity], \
                                                  self.catalog_stars_filtered[self.closest_cat_star_indx]])

                        # Switch back to centroiding mode
                        self.closest_cat_star_indx = None
                        self.cursor.setMode(0)
                        self.updatePairedStars()

            elif event.key() == QtCore.Qt.Key_Escape:
                if self.star_pick_mode:
                    
                    # If the ESC is pressed when the star has been centroided, reset the centroid
                    self.resetStarPick()
                    self.updatePairedStars()

            # Show the photometry plot
            elif event.key() == QtCore.Qt.Key_P:
                self.photometry(show_plot=True)

            # Show astrometry residuals plot
            elif event.key() == QtCore.Qt.Key_L:
                if self.star_pick_mode:
                    self.showAstrometryFitPlots()


        # Handle key presses in the manual reduction mode
        elif self.mode == 'manualreduction':

            if (qmodifiers & QtCore.Qt.ShiftModifier) and (self.img.img_handle.input_type != 'dfn'):
                self.cursor.setMode(2)

            if event.key() == QtCore.Qt.Key_P:
                self.showLightcurve()

            elif event.key() == QtCore.Qt.Key_D or event.key() == QtCore.Qt.Key_Space:
                self.nextImg(n=1)

            elif event.key() == QtCore.Qt.Key_A:
                self.nextImg(n=-1)

            elif event.key() == QtCore.Qt.Key_Up:
                self.nextImg(n=25)

            elif event.key() == QtCore.Qt.Key_Down:
                self.nextImg(n=-25)

            elif event.key() == QtCore.Qt.Key_Comma:
                if hasattr(self.img.img_handle, 'current_line'):
                    print('Current line: {}'.format(self.img.img_handle.current_line))
                    self.img.prevLine()

            elif event.key() == QtCore.Qt.Key_Period:
                if hasattr(self.img.img_handle, 'current_line'):
                    print('Current line: {}'.format(self.img.img_handle.current_line))
                    self.img.nextLine()

    def keyReleaseEvent(self, event):
        
        try:
            self.keys_pressed.remove(event.key())
        except ValueError:
            pass  # this will happen for key presses that are not passed to keypressevent (taken by menu hotkey)
        
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        qmodifiers = QtWidgets.QApplication.queryKeyboardModifiers()

        if self.mode == 'skyfit':
            pass
        else:
            if qmodifiers != QtCore.Qt.ShiftModifier:
                self.cursor.setMode(0)


    def wheelEvent(self, event):
        """ Change star selector aperature on scroll. """
        
        delta = event.angleDelta().y()
        modifier = QtWidgets.QApplication.keyboardModifiers()

        # Handle scroll events when in the star pricking mode
        if self.img_frame.sceneBoundingRect().contains(event.pos()) and self.star_pick_mode:

            # If control is pressed, change the size of the aperture
            if modifier & QtCore.Qt.ControlModifier:

                # Increase aperture size
                if delta < 0:
                    self.scrolls_back = 0
                    self.star_aperature_radius += 0.5
                    self.cursor.setRadius(self.star_aperature_radius)
                    self.cursor2.setRadius(self.star_aperature_radius)

                # Decrease aperture size
                elif delta > 0 and self.star_aperature_radius > 1:
                    self.scrolls_back = 0
                    self.star_aperature_radius -= 0.5
                    self.cursor.setRadius(self.star_aperature_radius)
                    self.cursor2.setRadius(self.star_aperature_radius)

            else:

                # Unzoom the image
                if delta < 0:

                    self.scrolls_back += 1

                    # Reset the zoom if scrolled back multiple times
                    if self.scrolls_back > 1:
                        self.img_frame.autoRange(padding=0)

                    else:
                        self.img_frame.scaleBy([1.2, 1.2], QtCore.QPoint(self.mouse_x, self.mouse_y))

                # Zoom in the image
                elif delta > 0:
                    self.scrolls_back = 0
                    self.img_frame.scaleBy([0.8, 0.8], QtCore.QPoint(self.mouse_x, self.mouse_y))



    def checkParamRange(self):
        """ Checks that the astrometry parameters are within the allowed range. """

        # Right ascension should be within 0-360
        self.platepar.RA_d = (self.platepar.RA_d + 360)%360

        # Keep the declination in the allowed range
        if self.platepar.dec_d >= 90:
            self.platepar.dec_d = 89.999

        if self.platepar.dec_d <= -90:
            self.platepar.dec_d = -89.999


        # Right ascension should be within 0-360
        self.platepar.az_centre = (self.platepar.az_centre + 360)%360

        # Keep the declination in the allowed range
        if self.platepar.alt_centre >= 90:
            self.platepar.alt_centre = 89.999

        if self.platepar.alt_centre <= -90:
            self.platepar.alt_centre = -89.999


    def resetStarPick(self):
        """ Call when finished starpicking """
        if self.cursor.mode:
            self.x_centroid = None
            self.y_centroid = None
            self.star_intensity = None
            self.cursor.setMode(0)
            self.updatePairedStars()

    def toggleZoomWindow(self):
        """ Toggle whether to show the zoom window """
        self.show_zoom_window = not self.show_zoom_window
        if self.show_zoom_window:
            self.v_zoom.show()
        else:
            self.v_zoom.hide()

    def toggleInfo(self):
        """ Toggle left label info """

        self.show_key_help += 1

        if self.show_key_help >= 3:
            self.show_key_help = 0

        if self.show_key_help == 0:
            self.label1.show()
            self.label2.hide()
            self.label_f1.show()

        elif self.show_key_help == 1:
            self.label1.show()
            self.label2.show()
            self.label_f1.hide()

        else:
            self.label1.hide()
            self.label2.hide()
            self.label_f1.show()

        if self.v_zoom_left:
            if self.show_key_help != 2:
                self.v_zoom.move(QtCore.QPoint(self.label1.boundingRect().width(), 0))
            else:
                self.v_zoom.move(QtCore.QPoint(0, 0))

        if self.show_key_help == 1 and self.star_pick_mode:
            self.star_pick_info.show()
        else:
            self.star_pick_info.hide()


    def toggleImageType(self):
        """ Toggle between the maxpixel and avepixel. """
        if self.img.img_handle.input_type == 'dfn':
            return

        if self.img_type_flag == 'maxpixel':
            self.img_type_flag = 'avepixel'
        else:
            self.img_type_flag = 'maxpixel'

        self.img_zoom.loadImage(self.mode, self.img_type_flag)
        self.img.loadImage(self.mode, self.img_type_flag)

        self.img.setLevels(self.tab.hist.getLevels())
        self.img_zoom.setLevels(self.tab.hist.getLevels())
        self.updateLeftLabels()

    def toggleShowCatStars(self):
        """ Toggle between showing catalog stars and not """
        self.catalog_stars_visible = not self.catalog_stars_visible
        if self.catalog_stars_visible:
            self.cat_star_markers.show()
            self.cat_star_markers2.show()
        else:
            self.cat_star_markers.hide()
            self.cat_star_markers2.hide()

    def toggleShowSelectedStars(self):
        """ Toggle whether to show the selected stars """
        self.selected_stars_visible = not self.selected_stars_visible
        if self.selected_stars_visible:
            self.sel_cat_star_markers.show()
            self.sel_cat_star_markers2.show()
            self.residual_lines_astro.show()
            self.residual_lines_img.show()
        else:
            self.sel_cat_star_markers.hide()
            self.sel_cat_star_markers2.hide()
            self.residual_lines_astro.hide()
            self.residual_lines_img.hide()

        self.photometry()

    def toggleShowCalStars(self):
        """ Toggle whether to show the calstars (green circles) """
        self.draw_calstars = not self.draw_calstars
        if self.draw_calstars:
            self.calstar_markers.show()
            self.calstar_markers2.show()
        else:
            self.calstar_markers.hide()
            self.calstar_markers2.hide()

    def toggleShowPicks(self):
        """ Toggle whether to show the picks for manualreduction """
        if self.pick_marker.isVisible():
            self.pick_marker.hide()
            self.pick_marker2.hide()
        else:
            self.pick_marker.show()
            self.pick_marker2.hide()

    def toggleShowRegion(self):
        """ Togle whether to show the photometry region for manualreduction """
        if self.region.isVisible():
            self.region.hide()
        else:
            self.region.show()

    def toggleDistortion(self):
        """ Toggle whether to show the distortion lines"""
        self.draw_distortion = not self.draw_distortion

        if self.draw_distortion:
            self.distortion_lines.show()
            self.updateDistortion()
        else:
            self.distortion_lines.hide()

    def toggleInvertColours(self):
        self.img.invert()
        self.img_zoom.invert()

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
            qmessagebox(title='Astrometry.net error',
                        message='Astrometry.net failed to find a solution!',
                        message_type="error")

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
        d = QFOVinputDialog(self)
        if d.exec_():
             data = d.getInputs()
        else:
            return 0, 0, 0

        self.azim_centre, self.alt_centre, rot_horizontal = data

        # Get the middle time of the first FF
        img_time = self.img_handle.currentTime()

        # Set the reference platepar time to the time of the FF
        self.platepar.JD = date2JD(*img_time)

        # Set the reference hour angle
        self.platepar.Ho = JD2HourAngle(self.platepar.JD)%360

        # Convert FOV centre to RA, Dec
        ra, dec = apparentAltAz2TrueRADec(self.azim_centre, self.alt_centre, date2JD(*img_time),
                                          self.platepar.lat, self.platepar.lon)

        return ra, dec, rot_horizontal


    def detectInputType(self,  beginning_time=None, use_fr_files=False, load=False):
        """
        Tries to find image files to load by looking at the self.dir_path folder. If the files
        in the folder must be loaded individually rather than a group, opens a file explorer
        for you to select one to load.

        Keyword arguments:
            beginning_time: [datetime] Datetime of the video beginning. Optional, only can be given for
                video input formats.
            use_fr_files: [bool] Include FR files together with FF files. False by default.
            load: [bool] If state was most recently loaded. Allows you to skip the file dialog if
                        you know which file is to opened.


        """

        img_handle = None

        # Load a state file
        if load and os.path.isfile(self.input_path):
            img_handle = detectInputTypeFile(self.input_path, self.config, beginning_time=beginning_time)
        
        # Load given data from a folder
        elif os.path.isdir(self.input_path):

            # Detect input file type and load appropriate input plugin
            img_handle = detectInputTypeFolder(self.dir_path, self.config, beginning_time=beginning_time, \
                use_fr_files=self.use_fr_files)

            # If the data was not being able to load from the folder, choose a file to load
            if img_handle is None:
                self.input_path = openFileDialog(self.dir_path, None, 'Select image/video file to open', \
                    matplotlib, \
                                        [('All Readable Files',
                                          '*.fits;*.bin;*.mp4;*.avi;*.mkv;*.vid;*.png;*.jpg;*.bmp;*.nef'),
                                         ('All Files', '*'),
                                         ('FF and FR Files', '*.fits;*.bin'),
                                         ('Video Files', '*.mp4;*.avi;*.mkv'),
                                         ('VID Files', '*.vid'),
                                         ('FITS Files', '*.fits'), ('BIN Files', '*.bin'),
                                         ('Image Files', '*.png;*.jpg;*.bmp;*.nef')])


        # If no previous ways of opening data was sucessful, open a file
        if img_handle is None:
            img_handle = detectInputTypeFile(self.input_path, self.config, beginning_time=beginning_time)


        self.img_handle = img_handle


    def loadCalstars(self):
        """ Loads data from calstars file and updates self.calstars """
        # Find the CALSTARS file in the given folder
        calstars_file = None
        for cal_file in os.listdir(self.dir_path):
            if ('CALSTARS' in cal_file) and ('.txt' in cal_file):
                calstars_file = cal_file
                break

        if calstars_file is None:

            # Check if the calstars file is required
            if hasattr(self, 'img_handle') and self.img_handle.require_calstars:
                qmessagebox(title='CALSTARS error',
                            message='CALSTARS file could not be found in the given directory!',
                            message_type="info")

            self.calstars = {}

        else:

            # Load the calstars file
            calstars_list = CALSTARS.readCALSTARS(self.dir_path, calstars_file)

            # Convert the list to a dictionary
            self.calstars = {ff_file: star_data for ff_file, star_data in calstars_list}

            print('CALSTARS file: ' + calstars_file + ' loaded!')

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

    def loadPlatepar(self, update=False):
        """
        Open a file dialog and ask user to open the platepar file, changing self.platepar and self.platepar_file

        Arguments:
            update: [bool] Whether to update the gui after loading new platepar (leave as False if gui objects
                            may not exist)

        """

        platepar = Platepar()

        # Check if platepar exists in the folder, and set it as the default file name if it does
        if self.config.platepar_name in os.listdir(self.dir_path):
            initialfile = self.config.platepar_name
        else:
            initialfile = ''

        # Load the platepar file
        platepar_file = None #openFileDialog(self.dir_path, initialfile, 'Select the platepar file', matplotlib,
                             #               [('Platepar Files', '*.cal'), ('All File', '*')])

        if platepar_file is None:
            self.platepar = platepar
            self.makeNewPlatepar()
            self.platepar_file = os.path.join(self.dir_path, self.config.platepar_name)
            self.first_platepar_fit = True

        else:
            # Parse the platepar file
            try:
                self.platepar_fmt = platepar.read(platepar_file, use_flat=self.config.use_flat)

            except Exception as e:
                print('Loading platepar failed with error:' + repr(e))
                print(*traceback.format_exception(*sys.exc_info()))

                qmessagebox(title='Platepar file error',
                            message='The file you selected could not be loaded as a platepar file!',
                            message_type="error")

                self.loadPlatepar(update)
                return

            print('Platepar loaded:', platepar_file)
            print("FOV: {:.2f} x {:.2f} deg".format(*computeFOVSize(platepar)))

            # Set geo location and gamma from config, if they were updated

            # Update the location from the config file
            platepar.lat = self.config.latitude
            platepar.lon = self.config.longitude
            platepar.elev = self.config.elevation

            platepar.X_res = self.config.width
            platepar.Y_res = self.config.height

            # Set the camera gamma from the config file
            platepar.gamma = self.config.gamma

            # Set station ID
            platepar.station_code = self.config.stationID

            # Compute the rotation w.r.t. horizon
            platepar.rotation_from_horiz = rotationWrtHorizon(platepar)

            self.first_platepar_fit = False

            self.platepar_file, self.platepar = platepar_file, platepar

        if update:
            self.updateStars()
            self.tab.param_manager.updatePlatepar()
            self.updateLeftLabels()

    def savePlatepar(self):
        """  Save platepar to a file """
        # If the platepar is new, save it to the working directory
        if not self.platepar_file:
            self.platepar_file = os.path.join(self.dir_path, self.config.platepar_name)

        # Save the platepar file
        self.platepar.write(self.platepar_file, fmt=self.platepar_fmt, fov=computeFOVSize(self.platepar))
        print('Platepar written to:', self.platepar_file)

    def saveDefaultPlatepar(self):
        platepar_default_path = os.path.join(os.getcwd(), self.config.platepar_name)

        # Save the platepar file
        self.platepar.write(platepar_default_path, fmt=self.platepar_fmt)
        print('Default platepar written to:', platepar_default_path)

    def saveCurrentFrame(self):
        """ Saves the current frame to disk. """

        # Generate a name for the FF file which will be written to FTPdetectinfo
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

        # Remove the file extension of the image file
        ff_name_ftp = ff_name_ftp.replace('.bin', '').replace('.fits', '')

        # Construct the file name
        frame_file_name = ff_name_ftp + "_frame_{:03d}".format(self.img.getFrame()) + '.png'
        frame_file_path = os.path.join(dir_path, frame_file_name)

        # Save the frame to disk
        Image.saveImage(frame_file_path, self.img.getFrame())

        print('Frame {:.1f} saved to: {:s}'.format(self.img.getFrame(), frame_file_path))

    def makeNewPlatepar(self):
        """ Make a new platepar from the loaded one, but set the parameters from the config file. """

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
        self.platepar.F_scale = 1.0/((scale_x + scale_y)/2)

        # Reset the distortion coeffs
        self.platepar.resetDistortionParameters()


        # Set station ID
        self.platepar.station_code = self.config.stationID

        # Get the FOV centre if the image handle is available so the time can be extracted
        if hasattr(self, 'img_handle'):

            # Get reference RA, Dec of the image centre
            self.platepar.RA_d, self.platepar.dec_d, self.platepar.rotation_from_horiz = self.getFOVcentre()

            # Recalculate reference alt/az
            self.platepar.az_centre, self.platepar.alt_centre = trueRaDec2ApparentAltAz(self.platepar.RA_d, \
                self.platepar.dec_d, self.platepar.JD, self.platepar.lat, self.platepar.lon)


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

        if hasattr(self, 'tab'):
            self.tab.param_manager.updatePlatepar()
            self.updateLeftLabels()
            self.updateStars()
            self.updateDistortion()


    def loadFlat(self):
        """ Open a file dialog and ask user to load a flat field. """

        # Check if flat exists in the folder, and set it as the defualt file name if it does
        if self.config.flat_file in os.listdir(self.dir_path):
            initialfile = self.config.flat_file
        else:
            initialfile = ''

        flat_file = openFileDialog(self.dir_path, initialfile, 'Select the flat field file', matplotlib,
                                   [('Image Files', '*.png;*.jpg;*.bmp'),
                                    ('All Files', '*')])

        if not flat_file:
            return False, None

        print(flat_file)

        try:
            # Load the flat, byteswap the flat if vid file is used or UWO png
            flat = Image.loadFlat(*os.path.split(flat_file), dtype=self.img.data.dtype,
                                  byteswap=self.img_handle.byteswap)
            flat.flat_img = np.swapaxes(flat.flat_img, 0, 1)
        except:
            qmessagebox(title='Flat field file error',
                        message='Flat could not be loaded!',
                        message_type="error")

            return False, None

        # Check if the size of the file matches
        if self.img.data.shape != flat.flat_img.shape:
            qmessagebox(title='Flat field file error',
                        message='The size of the flat field does not match the size of the image!',
                        message_type="error")

            flat = None

        # Check if the flat field was successfuly loaded
        if flat is None:
            qmessagebox(title='Flat field file error',
                        message='The file you selected could not be loaded as a flat field!',
                        message_type="error")

        return flat_file, flat

    def loadDark(self):
        """ Open a file dialog and ask user to load a dark frame. """

        dark_file = openFileDialog(self.dir_path, None, 'Select the dark frame file', matplotlib,
                                   [('Image Files', '*.png;*.jpg;*.bmp'),
                                    ('All Files', '*')])

        if not dark_file:
            return False, None

        print(dark_file)

        try:

            # Load the dark
            dark = Image.loadDark(*os.path.split(dark_file), dtype=self.img.data.dtype,
                                  byteswap=self.img_handle.byteswap)

        except:
            qmessagebox(title='Dark frame error',
                        message='Dark frame could not be loaded!',
                        message_type="error")

            return False, None

        dark = dark.astype(self.img.data.dtype)

        # Check if the size of the file matches
        if self.img.data.shape != dark.shape:
            qmessagebox(title='Dark field file error',
                        message='The size of the dark frame does not match the size of the image!',
                        message_type="error")

            dark = None

        # Check if the dark frame was successfuly loaded
        if dark is None:
            qmessagebox(title='Dark field file error',
                        message='The file you selected could not be loaded as a dark field!',
                        message_type="error")

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

    def addCentroid(self, frame, x_centroid, y_centroid, mode=1):
        """
        Adds or modifies a pick marker at given frame to self.pick_list with given information

        Arguments:
            frame: [int] Frame to add/modify pick to
            x_centroid: [float] x coordinate of pick
            y_centroid: [float] y coordinate of pick
            mode: [0 or 1] The mode of the pick, 0 is yellow, 1 is red

        """
        print('Added centroid at ({:.2f}, {:.2f}) on frame {:d}'.format(x_centroid, y_centroid, frame))

        pick = self.getCurrentPick()
        if pick:
            pick['x_centroid'] = x_centroid
            pick['y_centroid'] = y_centroid
            pick['mode'] = mode

        else:
            pick = {'x_centroid': x_centroid,
                    'y_centroid': y_centroid,
                    'mode': mode,
                    'intensity_sum': 1,
                    'photometry_pixels': None}
            self.pick_list[frame] = pick

        self.tab.debruijn.modifyRow(frame, mode)

    def removeCentroid(self, frame):
        """
        Removes the pick from given frame if it is there

        Arguments:
            frame: [int] frame to remove pick from

        """
        pick = self.getCurrentPick()
        if pick and pick['x_centroid']:
            print('Removed centroid at ({:.2f}, {:.2f}) on frame {:d}'.format(pick['x_centroid'],
                                                                              pick['y_centroid'],
                                                                              self.img.getFrame()))

            self.pick_list[self.img.getFrame()]['x_centroid'] = None
            self.pick_list[self.img.getFrame()]['y_centroid'] = None
            self.pick_list[self.img.getFrame()]['mode'] = None
            self.tab.debruijn.removeRow(frame)

    def centroid(self, prev_x_cent=None, prev_y_cent=None):
        """ Find the centroid of the star clicked on the image. """

        # If the centroid from the previous iteration is given, use that as the centre
        if (prev_x_cent is not None) and (prev_y_cent is not None):
            mouse_x = prev_x_cent
            mouse_y = prev_y_cent

        else:
            mouse_x = self.mouse_x
            mouse_y = self.mouse_y

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
        if x_max > self.platepar.X_res - 1:
            x_max = self.platepar.X_res - 1

        y_min = int(round(mouse_y - outer_radius))
        if y_min < 0: y_min = 0

        y_max = int(round(mouse_y + outer_radius))
        if y_max > self.platepar.Y_res - 1:
            y_max = self.platepar.Y_res - 1

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
        for i, (x, y) in enumerate(zip(self.catalog_x_filtered, self.catalog_y_filtered)):

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
            qmessagebox(title='Number of stars', message="At least 5 paired stars are needed to do the fit!", message_type="warning")

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

        # ## Compute standard coordinates ##

        # # Platepar with no distortion
        # pp_nodist = copy.deepcopy(self.platepar)
        # pp_nodist.x_poly_rev *= 0
        # pp_nodist.y_poly_rev *= 0

        # standard_x, standard_y, _ = getCatalogStarsImagePositions(catalog_stars, jd, pp_nodist)

        # ## ##

        residuals = []

        print()
        print('Residuals')
        print('----------')
        print(
            ' No,       Img X,       Img Y, RA cat (deg), Dec cat (deg),    Mag, -2.5*LSP,    Cat X,   Cat Y, RA img (deg), Dec img (deg), Err amin,  Err px, Direction')

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

            # Compute the residuals in ra/dec in angular coordinates
            img_time = self.img_handle.currentTime()
            _, ra_img, dec_img, _ = xyToRaDecPP([img_time], [img_x], [img_y], [1], self.platepar, \
                                                extinction_correction=False)

            ra_img = ra_img[0]
            dec_img = dec_img[0]

            # Compute the angular distance in degrees
            angular_distance = np.degrees(angularSeparation(np.radians(ra), np.radians(dec), \
                np.radians(ra_img), np.radians(dec_img)))

            residuals.append([img_x, img_y, angle, distance, angular_distance])

            # Print out the residuals
            print(
                '{:3d}, {:11.6f}, {:11.6f}, {:>12.6f}, {:>+13.6f}, {:+6.2f},  {:7.2f}, {:8.2f}, {:7.2f}, {:>12.6f}, {:>+13.6f}, {:8.2f}, {:7.2f}, {:+9.1f}'.format(
                    star_no + 1, img_x, img_y, ra, dec, mag, -2.5*np.log10(sum_intens), cat_x, cat_y, \
                    ra_img, dec_img, 60*angular_distance, distance, np.degrees(angle)))


        # Compute RMSD errors
        rmsd_angular = 60*RMSD([entry[4] for entry in residuals])
        rmsd_img = RMSD([entry[3] for entry in residuals])

        # If the average angular error is larger than 60 arc minutes, report it in degrees
        if rmsd_angular > 60:
            rmsd_angular /= 60
            angular_error_label = 'deg'

        else:
            angular_error_label = 'arcmin'

        print('RMSD: {:.2f} px, {:.2f} {:s}'.format(rmsd_img, rmsd_angular, angular_error_label))

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
            img_ang_separation = np.degrees(angularSeparation(np.radians(img_ra), np.radians(img_dec), \
                np.radians(ra_centre), np.radians(dec_centre)))
            skyradius_residuals.append(cat_ang_separation - img_ang_separation)

            # Compute azim/elev from the catalog
            azim_cat, elev_cat = trueRaDec2ApparentAltAz(cat_ra, cat_dec, jd, self.platepar.lat, \
                self.platepar.lon)

            azim_list.append(azim_cat)
            elev_list.append(elev_cat)

            # Compute azim/elev from image coordinates
            azim_img, elev_img = trueRaDec2ApparentAltAz(img_ra, img_dec, jd, self.platepar.lat, \
                self.platepar.lon)

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
        ax_x.set_xlim([0, self.platepar.X_res])

        # Plot Y vs Y error
        ax_y.scatter(y_list, y_residuals, s=2, c='k', zorder=3)

        ax_y.grid()
        ax_y.set_xlabel("Y (px)")
        ax_y.set_ylabel("Y error (px)")
        ax_y.set_xlim([0, self.platepar.Y_res])

        # Plot radius vs radius error
        ax_radius.scatter(radius_list, radius_residuals, s=2, c='k', zorder=3)

        ax_radius.grid()
        ax_radius.set_xlabel("Radius (px)")
        ax_radius.set_ylabel("Radius error (px)")
        ax_radius.set_xlim([0, np.hypot(self.platepar.X_res/2, self.platepar.Y_res/2)])

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

    def computeIntensitySum(self):
        """ Compute the background subtracted sum of intensity of colored pixels. The background is estimated
            as the median of near pixels that are not colored.
        """

        # Find the pick done on the current frame
        pick = self.getCurrentPick()

        if pick:
            # If there are no photometry pixels, set the intensity to 0
            if not pick['photometry_pixels']:
                pick['intensity_sum'] = 1
                return None

            x_arr, y_arr = np.array(pick['photometry_pixels']).T

            # Compute the centre of the colored pixels
            x_centre = np.mean(x_arr)
            y_centre = np.mean(y_arr)

            # Take a window twice the size of the colored pixels
            x_color_size = np.max(x_arr) - np.min(x_arr)
            y_color_size = np.max(y_arr) - np.min(y_arr)

            x_min = int(x_centre - x_color_size)
            x_max = int(x_centre + x_color_size)
            y_min = int(y_centre - y_color_size)
            y_max = int(y_centre + y_color_size)

            # Limit the size to be within the bounds
            if x_min < 0: x_min = 0
            if x_max > self.img.data.shape[0]: x_max = self.img.data.shape[0]
            if y_min < 0: y_min = 0
            if y_max > self.img.data.shape[1]: y_max = self.img.data.shape[1]

            # Take only the colored part
            mask_img = np.ones_like(self.img.data)
            mask_img[x_arr, y_arr] = 0
            masked_img = np.ma.masked_array(self.img.data, mask_img)
            crop_img = masked_img[x_min:x_max, y_min:y_max]

            # Perform gamma correction on the colored part
            # crop_img = Image.gammaCorrection(crop_img, self.config.gamma)

            # Mask out the colored in pixels
            mask_img_bg = np.zeros_like(self.img.data)
            mask_img_bg[x_arr, y_arr] = 1

            # Take the image where the colored part is masked out and crop the surroundings
            masked_img_bg = np.ma.masked_array(self.img.data, mask_img_bg)
            crop_bg = masked_img_bg[x_min:x_max, y_min:y_max]

            # Perform gamma correction on the background
            # crop_bg = Image.gammaCorrection(crop_bg, self.config.gamma)

            # Compute the median background
            background_lvl = np.ma.median(crop_bg)

            # Compute the background subtracted intensity sum
            pick['intensity_sum'] = np.ma.sum(crop_img - background_lvl)

            # Make sure the intensity sum is never 0
            if pick['intensity_sum'] <= 0:
                pick['intensity_sum'] = 1

    def showLightcurve(self):
        """ Show the meteor lightcurve. """

        # Compute the intensity sum done on the previous frame
        self.computeIntensitySum()

        # Create the list of picks for saving
        centroids = []
        for frame, pick in self.pick_list.items():
            centroids.append([frame, pick['x_centroid'], pick['y_centroid'], pick['intensity_sum']])

        # If there are less than 3 points, don't show the lightcurve
        if len(centroids) < 3:
            
            qmessagebox(title='Lightcurve info',
                        message='Less than 3 centroids!',
                        message_type="info")

            return 1

        # Sort by frame number
        centroids = sorted(centroids, key=lambda x: x[0])

        # Extract frames and intensities
        fr_intens = [line for line in centroids if line[3] > 0]

        # If there are less than 3 points, don't show the lightcurve
        if len(fr_intens) < 3:

            qmessagebox(title='Lightcurve info',
                        message='Less than 3 points have intensities!',
                        message_type="info")

            return 1

        # Extract frames and intensities
        frames, x_centroids, y_centroids, intensities = np.array(fr_intens).T

        # Init plot
        fig_p = plt.figure(facecolor=None)
        ax_p = fig_p.add_subplot(1, 1, 1)

        # If the platepar is available, compute the magnitudes, otherwise show the instrumental magnitude
        if self.platepar is not None:

            time_data = [self.img.img_handle.currentFrameTime()]*len(intensities)

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
        # ax_p.invert_xaxis()

        ax_p.grid()

        fig_p.show()

    def changePhotometry(self, frame, photometry_pixels, add_photometry):
        """ Add/remove photometry pixels of the pick. """
        pick = self.getCurrentPick()

        if pick:
            if pick['photometry_pixels'] is None:
                pick['photometry_pixels'] = []

            if add_photometry:
                # Add the photometry pixels to the pick
                pick['photometry_pixels'] = list(set(pick['photometry_pixels'] + photometry_pixels))

            else:
                # Remove the photometry pixels to the pick
                pick['photometry_pixels'] = [px for px in pick['photometry_pixels'] if px not in photometry_pixels]

        # Add a new pick
        elif add_photometry:
            pick = {'x_centroid': None,
                    'y_centroid': None,
                    'mode': None,
                    'intensity_sum': None,
                    'photometry_pixels': photometry_pixels}

            self.pick_list[frame] = pick

    def getCurrentPick(self):
        try:
            return self.pick_list[self.img.getFrame()]
        except KeyError:
            return None

    def resetPickFrames(self, new_initial_frame, reverse=False):
        """
        Updates self.pick_list so that the frames so that the first frame is new_initial_frame
        and all other frames are mapped accordingly.

        Used to map all guessed frame values to where they should be on the de bruijn sequence.
        The order may need to be reversed if the pattern was found on the reversed sequence.

        Arguments:
            new_initial_frame: [int] New frame to map the frame to
            reverse: [bool] Whether the order of the frames needs to be reversed

        Returns:
            The function used to map the frames to the current

        """
        first_frame = 1024
        for frame, pick in self.pick_list.items():
            if frame < first_frame and pick['x_centroid'] is not None:
                first_frame = frame
        temp = {}

        if reverse is False:
            f = lambda frame: frame - first_frame + new_initial_frame
        else:
            f = lambda frame: 2**(9 + 1) - (frame - first_frame + new_initial_frame)

        for frame, pick in self.pick_list.items():
            temp[f(frame)] = self.pick_list[frame]

        self.pick_list = temp
        self.updatePicks()

        return f

    def photometryColoring(self):
        """
        Color pixels for photometry.

        Returns: [list of tuples with 2 elements] List of the x and y coordinates of all
                    pixels to be coloured
        """

        pixel_list = []

        mouse_x = int(self.mouse_x)
        mouse_y = int(self.mouse_y)

        ### Add all pixels within the aperture to the list for photometry ###

        x_list = range(mouse_x - int(self.star_aperature_radius), mouse_x \
                       + int(self.star_aperature_radius) + 1)
        y_list = range(mouse_y - int(self.star_aperature_radius), mouse_y \
                       + int(self.star_aperature_radius) + 1)

        for x in x_list:
            for y in y_list:

                # Skip pixels ourside the image
                if (x < 0) or (x > self.img.data.shape[0]) or (y < 0) or (y > self.img.data.shape[1]):
                    continue

                # Check if the given pixels are within the aperture radius
                if ((x - mouse_x)**2 + (y - mouse_y)**2) <= self.star_aperature_radius**2:
                    pixel_list.append((x, y))

        ##########
        return pixel_list

    def drawPhotometryColoring(self):
        """ Updates image to have the colouring in the current frame """
        pick = self.getCurrentPick()

        if pick and pick['photometry_pixels']:
            # Create a coloring mask
            x_mask, y_mask = np.array(pick['photometry_pixels']).T

            mask_img = np.zeros(self.img.data.shape)
            mask_img[x_mask, y_mask] = 255

            self.region.setImage(mask_img)
        else:
            self.region.setImage(np.array([[0]]))

    def saveFTPdetectinfo(self):
        """ Saves the picks to a FTPdetectinfo file in the same folder where the first given file is. """

        # Compute the intensity sum done on the previous frame
        self.computeIntensitySum()

        # Generate a name for the FF file which will be written to FTPdetectinfo
        dir_path = self.img_handle.dir_path

        # If the FF file is loaded, just copy its name
        if self.img_handle.input_type == 'ff':
            ff_name_ftp = self.img_handle.current_ff_file

        else:
            # Construct a fake FF file name
            ff_name_ftp = "FF_{:s}_".format(self.platepar.station_code) \
                          + self.img_handle.beginning_datetime.strftime("%Y%m%d_%H%M%S_") \
                          + "{:03d}".format(int(self.img_handle.beginning_datetime.microsecond//1000)) \
                          + "_0000000.fits"

        # Create the list of picks for saving
        centroids = []
        for frame, pick in self.pick_list.items():

            # Make sure to centroid is picked and is not just the photometry
            if pick['x_centroid'] is None:
                continue

            # Get the rolling shutter corrected (or not, depending on the config) frame number
            frame_no = self.getRollingShutterCorrectedFrameNo(frame, pick)

            if pick['mode'] == 1:
                centroids.append([frame_no, pick['x_centroid'], pick['y_centroid'], pick['intensity_sum']])

        # If there are no centroids, don't save anything
        if len(centroids) == 0:
            
            qmessagebox(title='FTPdetectinfo saving error',
                        message='No centroids to save!',
                        message_type="info")

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
        writeFTPdetectinfo(meteor_list, dir_path, ftpdetectinfo_name, '', station_id, self.img_handle.fps)

        print('FTPdetecinfo written to:', os.path.join(dir_path, ftpdetectinfo_name))

        # If the platepar is given, apply it to the reductions
        if self.platepar is not None:
            applyAstrometryFTPdetectinfo(self.dir_path, ftpdetectinfo_name, '',
                                         UT_corr=self.platepar.UT_corr, platepar=self.platepar)

            print('Platepar applied to manual picks!')

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

        jdt_ref = datetime2JD(self.img_handle.beginning_datetime)

        # Set the reference JD
        json_dict['jdt_ref'] = jdt_ref

        # Set the frames per second
        json_dict['fps'] = self.img_handle.fps

        ### Save picks to JSON file ###

        # Set measurement type to RA/Dec (meastype = 1)
        json_dict['meastype'] = 1

        centroids = []
        for frame, pick in self.pick_list.items():

            # Make sure to centroid is picked and is not just the photometry
            if pick['x_centroid'] is None:
                continue

            # Compute RA/Dec of the pick if the platepar is available
            if self.platepar is not None:

                time_data = [self.img_handle.currentFrameTime()]

                _, ra_data, dec_data, mag_data = xyToRaDecPP(time_data, [pick['x_centroid']],
                                                             [pick['y_centroid']], [pick['intensity_sum']],
                                                             self.platepar)

                ra = ra_data[0]
                dec = dec_data[0]
                mag = mag_data[0]

            else:
                ra = dec = mag = None

            # Get the rolling shutter corrected (or not, depending on the config) frame number
            frame_no = self.getRollingShutterCorrectedFrameNo(frame, pick)

            # Compute the time relative to the reference JD
            t_rel = frame_no/self.img_handle.fps

            centroids.append([t_rel, pick['x_centroid'], pick['y_centroid'], ra, dec, pick['intensity_sum'], mag])

        # Sort centroids by relative time
        centroids = sorted(centroids, key=lambda x: x[0])

        json_dict['centroids_labels'] = ['Time (s)', 'X (px)', 'Y (px)', 'RA (deg)', 'Dec (deg)',
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

    def getRollingShutterCorrectedFrameNo(self, frame, pick):
        """ Given a pick object, return rolling shutter corrected (or not, depending on the config) frame
            number.
        """

        # Correct the rolling shutter effect
        if self.config.deinterlace_order == -1:

            # Get image height
            if self.img_handle is None:
                img_h = self.config.height

            else:
                img_h = self.img.data.shape[0]

            # Compute the corrected frame time
            frame_no = RollingShutterCorrection.correctRollingShutterTemporal(frame, pick['y_centroid'], img_h)

        # If global shutter, do no correction
        else:
            frame_no = frame

        return frame_no

    def printFrameRate(self):
        try:
            print('FPS: {}'.format(np.average(self.frames)))
            self.frames[self.i] = 1/(time.time() - self.time)
            self.i = (self.i + 1)%self.n
        except ZeroDivisionError:
            pass
        self.time = time.time()


if __name__ == '__main__':
    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Tool for fitting astrometry plates and photometric calibration.")

    arg_parser.add_argument('input_path', metavar='INPUT_PATH', type=str,
                            help='Path to the folder with FF or image files, path to a video file, or to a state file.'
                                 ' If images or videos are given, their names must be in the format: YYYYMMDD_hhmmss.uuuuuu')

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str,
                            help="Path to a config file which will be used instead of the default one."
                                 " To load the .config file in the given data directory, write '.' (dot).")

    arg_parser.add_argument('-r', '--fr', action="store_true", \
        help="""Use FR files. """)

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


    # Parse the beginning time into a datetime object
    if cml_args.timebeg is not None:

        beginning_time = datetime.datetime.strptime(cml_args.timebeg[0], "%Y%m%d_%H%M%S.%f")

    else:
        beginning_time = None

    app = QtWidgets.QApplication(sys.argv)

    # If the state file was given, load the state
    if cml_args.input_path.endswith('.state'):

        dir_path, state_name = os.path.split(cml_args.input_path)
        config = cr.loadConfigFromDirectory(cml_args.config, cml_args.input_path)

        # Create plate_tool without calling its constructor then calling loadstate
        plate_tool = PlateTool.__new__(PlateTool)
        super(PlateTool, plate_tool).__init__()
        plate_tool.loadState(dir_path, state_name, beginning_time=beginning_time)

    else:

        # Extract the data directory path
        input_path = cml_args.input_path.replace('"', '')
        if os.path.isfile(input_path):
            dir_path = os.path.dirname(input_path)
        else:
            dir_path = input_path

        # Load the config file
        config = cr.loadConfigFromDirectory(cml_args.config, dir_path)

        # Init SkyFit
        plate_tool = PlateTool(input_path, config, beginning_time=beginning_time, fps=cml_args.fps, \
            gamma=cml_args.gamma, use_fr_files=cml_args.fr)


    # Run the GUI app
    sys.exit(app.exec_())
