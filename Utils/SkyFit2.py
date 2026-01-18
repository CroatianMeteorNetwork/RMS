from __future__ import print_function, division, absolute_import, unicode_literals

import os
import math
import argparse
import traceback
import copy
import cProfile
import json
import datetime
import collections
import glob
import sys
import random
import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets, QtGui
except Exception as exc:
    message = [
        "SkyFit requires PyQtGraph/PyQt5 for its GUI components, but the import failed.",
        "The most common causes are missing GUI dependencies or Windows being unable to allocate enough",
        "virtual memory (e.g. the paging file is too small).",
        f"Original import error: {exc}",
        "Fix the underlying issue and re-run SkyFit."
    ]

    # Provide an actionable tip if Windows reports an undersized paging file.
    if isinstance(exc, ImportError) and "paging file" in str(exc).lower():
        message.append(
            "Windows reported that the paging file is too small. Increase the paging file size or free RAM and try again."
        )

    print("\n".join(message))
    sys.exit(1)


from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP, raDecToXYPP, \
    rotationWrtHorizon, rotationWrtHorizonToPosAngle, computeFOVSize, photomLine, photometryFit, \
    rotationWrtStandard, rotationWrtStandardToPosAngle, correctVignetting, \
    extinctionCorrectionTrueToApparent, applyAstrometryFTPdetectinfo, getFOVSelectionRadius
from RMS.Astrometry.AtmosphericExtinction import atmosphericExtinctionCorrection
from RMS.Astrometry.StarClasses import CatalogStar, GeoPoint, PairedStars
from RMS.Astrometry.StarFilters import filterPhotometricOutliers, filterBlendedStars, filterHighFWHMStars
from RMS.Astrometry.Conversions import date2JD, JD2HourAngle, trueRaDec2ApparentAltAz, \
    apparentAltAz2TrueRADec, J2000_JD, jd2Date, datetime2JD, JD2LST, geo2Cartesian, vector2RaDec, raDec2Vector
from RMS.Astrometry.AstrometryNet import astrometryNetSolve
from RMS.Astrometry.NNalign import alignPlatepar
import RMS.ConfigReader as cr
from RMS.ExtractStars import extractStarsAndSave, extractStarsFF
import RMS.Formats.CALSTARS as CALSTARS
from RMS.Formats.Platepar import Platepar, getCatalogStarsImagePositions
from RMS.Formats.FFfile import convertFRNameToFF, constructFFName
from RMS.Formats.FrameInterface import detectInputTypeFolder, detectInputTypeFile
from RMS.Formats.FTPdetectinfo import writeFTPdetectinfo
from RMS.Formats import StarCatalog
from RMS.Pickling import loadPickle, savePickle
from RMS.Math import angularSeparation, RMSD, vectNorm
from RMS.Misc import decimalDegreesToSexHours
from RMS.Routines.AddCelestialGrid import updateRaDecGrid, updateAzAltGrid
from RMS.Routines.CustomPyqtgraphClasses import ViewBox, TextItem, TextItemList, Crosshair, Plus, Cross, CursorItem, ImageItem, RightOptionsTab
from RMS.Routines.GreatCircle import fitGreatCircle, greatCircle
from RMS.Routines.SphericalPolygonCheck import sphericalPolygonCheck
from RMS.Routines.Image import loadFlat, loadDark, applyFlat, applyDark, signalToNoise, gammaCorrectionImage, adjustLevels, saveImage
from RMS.Routines.MaskImage import getMaskFile, MaskStructure
from RMS.Routines import RollingShutterCorrection
from RMS.Misc import maxDistBetweenPoints, getRmsRootDir
from Utils.KalmanFilter import KalmanFilter

import pyximport
pyximport.install(setup_args={'include_dirs': [np.get_include()]})
from RMS.Astrometry.CyFunctions import subsetCatalog, equatorialCoordPrecession
from RMS.Routines.SatellitePositions import SatellitePredictor, loadTLEs, loadRobustTLEs, findClosestTLEFile, SKYFIELD_AVAILABLE
from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP
from RMS.Astrometry.Conversions import datetime2JD

# Load the ASTRA module
try:
    import html, re

    from PyQt5.QtWidgets import (
        QDialog, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QGroupBox, QComboBox, QFileDialog,
        QProgressBar, QGridLayout
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
    from PyQt5 import QtCore
    from PyQt5.QtGui import QFont

    from Utils.Astra import ASTRA, PYSWARMS_AVAILABLE

    if not PYSWARMS_AVAILABLE:
        raise ImportError("pyswarms not installed in Utils.Astra")

    ASTRA_IMPORTED = True

except Exception as e:
    ASTRA_IMPORTED = False
    print("ASTRA is an automated tool for picking positions of meteors.")
    print("If you don't plan to use it, you can ignore this error message.")
    print(f'ASTRA import error: {e}')



##############################################################################################################
# ASTRA GUI Code

# Only initialize if ASTRA has been loaded
if ASTRA_IMPORTED:
    class AstraConfigDialog(QDialog):
        """
        A dialog window for configuring and running ASTRA (Astrometric Streak Tracking and Refinement Algorithm).
        
        This dialog provides a GUI for:
        - Loading pick data from ECSV/TXT files
        - Configuring PSO (Particle Swarm Optimization) parameters
        - Setting ASTRA algorithm parameters
        - Configuring Kalman filter settings
        - Running ASTRA and Kalman filter processing
        - Monitoring progress and status
        
        Args:
            run_load_callback: Callback function for loading pick data
            run_astra_callback: Callback function for running ASTRA
            run_kalman_callback: Callback function for running Kalman filter
            skyfit_instance: Instance of the main SkyFit application
            parent: Parent widget
        """

        def __init__(self, run_load_callback=None, run_astra_callback=None, run_kalman_callback=None, 
                    skyfit_instance=None, parent=None, on_close_callback=None):

            super().__init__(parent)

            # Allow ASTRA to be minimized
            self.setWindowFlag(QtCore.Qt.Window, True)
            self.setWindowFlag(QtCore.Qt.WindowSystemMenuHint, True)
            self.setWindowFlag(QtCore.Qt.WindowMinimizeButtonHint, True)
            self.setWindowFlag(QtCore.Qt.WindowMaximizeButtonHint, True)
            self.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, True)
            self.setWindowFlags(
                QtCore.Qt.Window
                | QtCore.Qt.WindowSystemMenuHint
                | QtCore.Qt.WindowMinimizeButtonHint
                | QtCore.Qt.WindowMaximizeButtonHint
                | QtCore.Qt.WindowCloseButtonHint
            )
            self.setWindowModality(QtCore.Qt.NonModal)

            # Make ASTRAGUI close gracefully
            self.on_close_callback = on_close_callback
            self.thread = None
            self.worker = None
            self.kalman_worker = None

            self.setWindowTitle("ASTRA Configuration")
            self.setMinimumWidth(900)
            self.config = {}
            self.run_astra_callback = run_astra_callback
            self.run_kalman_callback = run_kalman_callback
            self.load_picks_callback = run_load_callback
            self.skyfit_instance = skyfit_instance

            main_layout = QVBoxLayout()

            # === Kick-start method selection ===
            pick_method_group = QGroupBox("INFO AND PICK LOADING")
            pick_layout = QVBoxLayout()

            intro_label = QLabel(
                "<b>ASTRA: Astrometric Streak Tracking and Refinement Algorithm</b> <br> ASTRA is an algoritm for automating manual EMCCD picking/photometry, and can also be used to refine manual picks/photometry."
            )
            intro_label.setWordWrap(True)
            intro_label.setAlignment(Qt.AlignCenter)
            pick_layout.addWidget(intro_label)

            info_label = QLabel(
                "<b>ASTRA requires (at least) 3 frame-adjacent leading-edge picks at a good-SNR section AND 2 leading edge picks at the frames marking the start/end of the event. These can be loaded through ECSV/txt files, or done manually.</b> <br> Hover over parameters and READY/NOT READY icons for info."
            )
            info_label.setWordWrap(True)
            info_label.setAlignment(Qt.AlignCenter)
            pick_layout.addWidget(info_label)

            # Create two rows: first for file picker, second for revert button
            file_picker_layout = QHBoxLayout()
            self.file_picker_button = QPushButton("SELECT ECSV/TXT FILE")
            self.file_picker_button.clicked.connect(self.selectFile)
            file_picker_layout.addWidget(self.file_picker_button)
            pick_layout.addLayout(file_picker_layout)

            self.selected_file_label = QLabel("No file selected")
            pick_layout.addWidget(self.selected_file_label)

            pick_method_group.setLayout(pick_layout)
            main_layout.addWidget(pick_method_group)

            # === LATEX CONVERSIONS ===
            _GREEK = {
            "alpha":"α","beta":"β","gamma":"γ","delta":"δ","epsilon":"ε","zeta":"ζ","eta":"η",
            "theta":"θ","iota":"ι","kappa":"κ","lambda":"λ","mu":"μ","nu":"ν","xi":"ξ",
            "pi":"π","rho":"ρ","sigma":"σ","tau":"τ","upsilon":"υ","phi":"φ","chi":"χ",
            "psi":"ψ","omega":"ω",
            }
            
            def toHTMLMath(s: str) -> str:
                t = html.escape(s)

                # 1) underscores → HTML subscript tags
                def _subber(m):
                    base = m.group(1)
                    subs = m.group(2)
                    for part in subs.split('_'):
                        base += f'<sub>{part}</sub>'
                    return base
                t = re.sub(r'([A-Za-zΑ-Ωα-ω]+)_([A-Za-z0-9_]+)', _subber, t)

                # 2) greek conversion inside and outside subscripts
                for name, sym in _GREEK.items():
                    # Replace in normal text
                    t = re.sub(rf'\b{re.escape(name)}\b', sym, t)
                    # Replace inside <sub>...</sub>
                    t = re.sub(rf'(<sub>){re.escape(name)}(</sub>)', rf'\1{sym}\2', t)

                # 3) superscripts
                t = re.sub(r'\^([0-9]+)', r'<sup>\1</sup>', t)

                # 4) cosmetic replace for (0-1)
                t = t.replace('(0-1)', '[0, 1]')

                return t

            def addGridFields(field_dict, defaults, title, tooltips=None, on_change=False):
                group = QGroupBox(title)
                layout = QGridLayout()
                tts = tooltips or {}

                def _is_bool_like(val: str) -> bool:
                    return isinstance(val, str) and val.strip().lower() in ("true", "false")

                # helper to connect widget change -> on_change() if provided
                def _wire_change(widget, kind="auto"):
                    if not callable(on_change):
                        return
                    try:
                        if isinstance(widget, QLineEdit):
                            widget.textChanged.connect(lambda *_: on_change())
                        elif isinstance(widget, QComboBox):
                            # use currentTextChanged so we react to programmatic + user changes
                            widget.currentTextChanged.connect(lambda *_: on_change())
                        else:
                            # fallback: try generic signals if needed
                            if hasattr(widget, "editingFinished"):
                                widget.editingFinished.connect(lambda *_: on_change())
                    except Exception:
                        pass

                row = 0
                col = 0

                for key, default in defaults.items():
                    key_html = toHTMLMath(key)
                    label = QLabel(key_html.replace('</span></body></html>', ':</span></body></html>'))

                    # tooltip
                    tt_raw = tts.get(key, "")
                    tt_html = toHTMLMath(tt_raw) if tt_raw else ""
                    if tt_html:
                        label.setToolTip(tt_html)

                    # --- Special handling for pick_offset ---
                    if key == "pick_offset":
                        # Dropdown: center, leading-edge, custom
                        combo = QComboBox()
                        combo.addItems(["center", "leading-edge", "custom"])

                        def _is_floatlike(s):
                            try:
                                float(str(s))
                                return True
                            except Exception:
                                return False

                        custom_edit = QLineEdit()
                        custom_edit.setPlaceholderText("e.g. 0.25")

                        if str(default) in ("center", "leading-edge"):
                            combo.setCurrentText(str(default))
                            custom_edit.setEnabled(False)
                            custom_edit.setText("")
                        elif _is_floatlike(default):
                            combo.setCurrentText("custom")
                            custom_edit.setEnabled(True)
                            custom_edit.setText(str(default))
                        else:
                            combo.setCurrentText("leading-edge")
                            custom_edit.setEnabled(False)
                            custom_edit.setText("")

                        if tt_html:
                            combo.setToolTip(tt_html)
                            custom_edit.setToolTip(tt_html)

                        # Toggle custom field
                        def _on_pick_offset_changed(text):
                            custom_edit.setEnabled(text == "custom")
                            # also notify about change
                            if callable(on_change):
                                on_change()

                        combo.currentTextChanged.connect(_on_pick_offset_changed)

                        # Place widgets
                        layout.addWidget(label, row, col)
                        h = QHBoxLayout()
                        h.addWidget(combo, 1)
                        h.addWidget(custom_edit, 1)
                        container = QtWidgets.QWidget()
                        container.setLayout(h)      
                        layout.addWidget(container, row, col + 1)

                        # Store references (two entries for later resolution)
                        field_dict["pick_offset_mode"] = combo
                        field_dict["pick_offset_custom"] = custom_edit

                        # Wire change notifications
                        _wire_change(combo)
                        _wire_change(custom_edit)

                    else:
                        # --- default behavior for everything else ---
                        if _is_bool_like(default):
                            field = QComboBox()
                            field.addItems(["True", "False"])
                            field.setCurrentText("True" if str(default).strip().lower() == "true" else "False")
                        else:
                            field = QLineEdit(default)

                        if tt_html:
                            field.setToolTip(tt_html)

                        layout.addWidget(label, row, col)
                        layout.addWidget(field, row, col + 1)
                        field_dict[key] = field

                        # Wire change notifications
                        _wire_change(field)

                    # advance grid position (2 columns per row)
                    if col == 0:
                        col = 2
                    else:
                        col = 0
                        row += 1

                group.setLayout(layout)
                return group  # Return the group widget directly

            # === PSO Settings ===
            self.pso_fields = {}
            pso_defaults = {
                "w (0-1)": "0.9", "c_1 (0-1)": "0.4", "c_2 (0-1)": "0.3",
                "max itter": "100", "n_particles": "100", "V_c (0-1)": "0.3",
                "ftol": "1e-4", "ftol_itter": "25", "expl_c": "3", "P_sigma": "3"
            }

            # === ASTRA General Settings ===
            self.astra_fields = {}
            astra_defaults = {
                "star_thresh": "3", "min SNR": "5",
                "P_crop": "1.5", "sigma_init (px)": "2", "sigma_max": "1.2",
                "L_max": "1.5", "Verbose": "False", "photom_thresh" : "0.01", 
                "Save Animation": "False", "pick_offset" : "leading-edge"
            }

            # === Kalman Filter Settings ===
            self.kalman_fields = {}
            kalman_defaults = {
                "Monotonicity": "True", "sigma_xy (px)": "0.5", "sigma_vxy (%)": "100", "save results" : "False"
            }

            # === PARAMETER GUIDE ===
            PSO_TT = {
                "w (0-1)": "PSO particle inertia. Higher = more exploration.",
                "c_1 (0-1)": "Cognitive weight (pull to particle's best).",
                "c_2 (0-1)": "Social weight (pull to global best).",
                "max itter": "Maximum PSO iterations.",
                "n_particles": "Number of PSO particles.",
                "V_c (0-1)": "Max velocity as fraction of parameter range.",
                "ftol": "Stop when objective change % < ftol.",
                "ftol_itter": "Minimum consecutive iters below ftol to stop.",
                "expl_c": "Initial seeding spread coefficient. Higher = more exploration.",
                "P_sigma": "Second-pass bound looseness for local fitting. Higher = looser bounds"
            }

            ASTRA_TT = {
                "star_thresh": "Background mask threshold (σ above mean).",
                "min SNR": "Minimum SNR to keep a pick.",
                "P_crop": "Crop padding coefficient.",
                "sigma_init (px)": "Initial Gaussian σ guess (px).",
                "sigma_max": "Max σ multiplier (upper bound).",
                "L_max": "Max length multiplier (upper bound).",
                "photom_thresh": "Luminosity threshold for photometry pixels (fraction of peak).",
                "Verbose": "Verbose console logging (True/False).",
                "Save Animation" : "Save animation showing fit, crop, and residuals for each frame.",
                "pick_offset" : "Pick position relative to the Gaussian center along the streak axis. Options: 'center', 'leading-edge', or a custom float (multiples of length STD)."
            }   

            KALMAN_TT = {
                "Monotonicity": "Enforce monotonic motion along dominant axis (True/False).",
                "sigma_xy (px)": "STD of position estimate errors (px).",
                "sigma_vxy (%)": "STD of velocity estimate errors (in percent).",
                "save results" : "Save the uncertainties from the kalman filter into a .csv file"
            }

            main_layout.addWidget(
                addGridFields(self.pso_fields, pso_defaults, "PSO PARAMETER SETTINGS", PSO_TT, on_change=self.storeConfig)
            )
            main_layout.addWidget(
                addGridFields(self.astra_fields, astra_defaults, "ASTRA PARAMETER SETTINGS", ASTRA_TT, on_change=self.storeConfig)
            )
            main_layout.addWidget(
                addGridFields(self.kalman_fields, kalman_defaults, "KALMAN FILTER SETTINGS", KALMAN_TT, on_change=self.storeConfig)
            )

            # === Progress Bar ===
            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            # Progress label and ASTRA status indicator
            progress_status_layout = QHBoxLayout()
            progress_label = QLabel("Progress:")
            progress_status_layout.addWidget(progress_label)

            # ASTRA status label and dot
            self.astra_status_label = QLabel("ASTRA:")
            self.astra_status_dot = QLabel()

            # Kalman status label and dot
            self.kalman_status_label = QLabel("KALMAN:")
            self.kalman_status_dot = QLabel()

            # Place ASTRA and KALMAN status beside each other, right-aligned
            progress_status_layout.addStretch()
            progress_status_layout.addWidget(self.astra_status_label)
            progress_status_layout.addWidget(self.astra_status_dot)
            progress_status_layout.addSpacing(16)  # Optional: add space between ASTRA and KALMAN
            progress_status_layout.addWidget(self.kalman_status_label)
            progress_status_layout.addWidget(self.kalman_status_dot)

            main_layout.addLayout(progress_status_layout)
            main_layout.addWidget(self.progress_bar)

            # === Control Buttons ===
            btn_layout = QHBoxLayout()
            self.run_astra_btn = QPushButton("RUN ASTRA")
            self.run_kalman_btn = QPushButton("RUN KALMAN")
            self.run_astra_btn.clicked.connect(self.startASTRAThread)
            self.run_kalman_btn.clicked.connect(self.startKalmanThread)
            btn_layout.addWidget(self.run_astra_btn)
            btn_layout.addWidget(self.run_kalman_btn)
            main_layout.addLayout(btn_layout)

            # Move REVERT button below RUN ASTRA/KALMAN buttons as one row
            revert_layout = QHBoxLayout()
            self.set_prev_picks_button = QPushButton("REVERT ASTRA/KALMAN PICKS")
            self.set_prev_picks_button.clicked.connect(self.setPreviousPicks)
            revert_layout.addWidget(self.set_prev_picks_button)
            main_layout.addLayout(revert_layout)

            # Now that buttons exist, set initial status
            self.setASTRAStatus(False)  # Default to not ready (red)
            self.setKalmanStatus(False)  # Default to not ready (red)
            self.setRevertStatus(False)  # Default to not ready (disabled)

            self.setLayout(main_layout)

        def setConfig(self, config):
            """
            Updates the configuration and sets all UI elements to match the new config values.
            
            Args:
                config (dict): Configuration dictionary with pso, astra, and kalman settings
            """
            self.config = config
            
            # Update file path if provided
            if "file_path" in config and config["file_path"]:
                self.selected_file_label.setText(config["file_path"])
            
            # Update PSO fields
            if "pso" in config:
                for key, value in config["pso"].items():
                    if key in self.pso_fields:
                        if hasattr(self.pso_fields[key], "setCurrentText"):
                            self.pso_fields[key].setCurrentText(str(value))
                        else:
                            self.pso_fields[key].setText(str(value))
            
            # Update ASTRA fields, handling pick_offset specially
            if "astra" in config:
                for key, value in config["astra"].items():
                    # Skip special pick_offset handling for now
                    if key == "pick_offset":
                        continue
                    
                    if key in self.astra_fields:
                        if hasattr(self.astra_fields[key], "setCurrentText"):
                            self.astra_fields[key].setCurrentText(str(value))
                        else:
                            self.astra_fields[key].setText(str(value))
                
                # Special handling for pick_offset
                if "pick_offset" in config["astra"]:
                    pick_offset = config["astra"]["pick_offset"]
                    
                    # Check if we have the mode + custom fields
                    if "pick_offset_mode" in self.astra_fields and "pick_offset_custom" in self.astra_fields:
                        # If pick_offset is "center" or "leading-edge", set mode and clear custom
                        if pick_offset in ["center", "leading-edge"]:
                            self.astra_fields["pick_offset_mode"].setCurrentText(pick_offset)
                            self.astra_fields["pick_offset_custom"].setText("")
                            self.astra_fields["pick_offset_custom"].setEnabled(False)
                        # Otherwise it's a custom value
                        else:
                            self.astra_fields["pick_offset_mode"].setCurrentText("custom")
                            self.astra_fields["pick_offset_custom"].setText(str(pick_offset))
                            self.astra_fields["pick_offset_custom"].setEnabled(True)
                    # Fallback for backward compatibility
                    elif "pick_offset" in self.astra_fields:
                        if hasattr(self.astra_fields["pick_offset"], "setCurrentText"):
                            self.astra_fields["pick_offset"].setCurrentText(str(pick_offset))
                        else:
                            self.astra_fields["pick_offset"].setText(str(pick_offset))
            
            # Update Kalman fields
            if "kalman" in config:
                for key, value in config["kalman"].items():
                    if key in self.kalman_fields:
                        if hasattr(self.kalman_fields[key], "setCurrentText"):
                            self.kalman_fields[key].setCurrentText(str(value))
                        else:
                            self.kalman_fields[key].setText(str(value))
            
            # Refresh readiness indicators
            if hasattr(self.skyfit_instance, "checkASTRACanRun"):
                self.skyfit_instance.checkASTRACanRun()
            if hasattr(self.skyfit_instance, "checkKalmanCanRun"):
                self.skyfit_instance.checkKalmanCanRun()

        def selectFile(self):
            """Opens dialog to select ECSV/txt file"""
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select the ECSV or TXT file",
                "",
                "ECSV or TXT files (*.ecsv *.txt);;All files (*)"
            )
            if file_path:
                self.selected_file_label.setText(file_path)
                self.storeConfig()
                if self.load_picks_callback:
                    self.load_picks_callback(self.config)

        def _stopAstraWorker(self, hard_kill_timeout_ms=3000):
            """Try to stop Astra worker gracefully; escalate if needed."""
            if hasattr(self, "thread") and self.thread and self.thread.isRunning():
                try:
                    if hasattr(self, "worker") and self.worker:
                        # cooperative stop request
                        if hasattr(self.worker, "stop"):
                            self.worker.stop()
                    # ask the thread to finish its event loop
                    self.thread.requestInterruption()
                    self.thread.quit()
                    if not self.thread.wait(hard_kill_timeout_ms):
                        # last resort (unsafe): terminate
                        self.thread.terminate()
                        self.thread.wait(1000)
                except Exception:
                    pass  # don't block close on errors

        def _stopKalmanWorker(self, hard_kill_timeout_ms=3000):
            if hasattr(self, "kalman_worker") and self.kalman_worker and self.kalman_worker.isRunning():
                try:
                    # cooperative stop
                    if hasattr(self.kalman_worker, "stop"):
                        self.kalman_worker.stop()
                    self.kalman_worker.requestInterruption()
                    self.kalman_worker.quit()
                    self.kalman_worker.wait(hard_kill_timeout_ms)
                except Exception:
                    pass

        def _handleClose(self):
            # Stop background work
            self._stopKalmanWorker()
            self._stopAstraWorker()

            # Fire optional close callback
            if callable(self.on_close_callback):
                try:
                    self.on_close_callback()
                except Exception:
                    pass

        def closeEvent(self, event):
            self._is_closing = True
            # if you clear the handle on the parent, do it before finished signals run:
            if self.skyfit_instance is not None and getattr(self.skyfit_instance, "astra_dialog", None) is self:
                self.skyfit_instance.astra_dialog = None

            print("Closed ASTRA GUI - Aborting all threaded processes")
            # stop workers (your existing stop logic here)
            self._stopKalmanWorker()
            self._stopAstraWorker()
            super().closeEvent(event)

        def reject(self):
            # Triggered by ESC/window manager close as well
            self._handleClose()
            super().reject()

        def setRevertStatus(self, ready):
            """
            Sets the revert status button state.
            """
            self.set_prev_picks_button.setEnabled(ready)

        def setASTRAStatus(self, ready, hover_text=""):
            """
            Sets the ASTRA status dot color: green if ready, red if not.
            If ready == "WARN", sets color to yellow and text to READY.
            Optionally sets a tooltip (hover text) to inform the user.
            Disables the ASTRA button if not ready.
            """
            if ready == "WARN":
                status_text = "READY"
                color = "#FFC107"  # Yellow
                enable_btn = True
            elif ready == "PROCESSING":
                status_text = "RUNNING"
                color = "#2196F3"  # Blue
                enable_btn = False
            else:
                status_text = "READY" if ready else "NOT READY"
                color = "#4CAF50" if ready else "#F44336"  # Green or Red
                enable_btn = bool(ready)
            self.astra_status_dot.setText(status_text)
            self.astra_status_dot.setAlignment(Qt.AlignCenter)
            self.astra_status_dot.setStyleSheet(
                f"background-color: {color}; color: white; border-radius: 6px; min-width: 80px; min-height: 20px;"
                "max-height: 40px; font-weight: bold;"
            )
            self.astra_status_dot.setToolTip(hover_text or "")
            self.run_astra_btn.setEnabled(enable_btn)

        def setKalmanStatus(self, ready, hover_text=""):
            """
            Sets the Kalman status dot color: green if ready, red if not.
            Optionally sets a tooltip (hover text) to inform the user.
            Disables the KALMAN button if not ready.
            """

            if ready == "WARN":
                status_text = "READY"
                color = "#FFC107"  # Yellow
                enable_btn = True
            elif ready == "PROCESSING":
                status_text = "RUNNING"
                color = "#2196F3"  # Blue
                enable_btn = False
            else:
                status_text = "READY" if ready else "NOT READY"
                color = "#4CAF50" if ready else "#F44336"  # Green or Red
                enable_btn = bool(ready)
            self.kalman_status_dot.setText(status_text)
            self.kalman_status_dot.setAlignment(Qt.AlignCenter)
            self.kalman_status_dot.setStyleSheet(
                f"background-color: {color}; color: white; border-radius: 6px; min-width: 80px; min-height: 20px;" 
                "max-height: 40px; font-weight: bold;"
            )
            self.kalman_status_dot.setToolTip(hover_text or "")
            self.run_kalman_btn.setEnabled(enable_btn)

        def updateProgress(self, value):
            """Sets the progress bar to a given value
            args:
                - value (int): The value to set the progress bar to (0-100).
            """
            self.progress_bar.setValue(value)
        
        def storeConfig(self):
            """
            Stores the current configuration from the UI elements.
            """
            def _value_of(w):
                # QComboBox has currentText(); QLineEdit has text()
                return w.currentText() if hasattr(w, "currentText") else w.text()

            # PSO and Kalman straight-through
            pso = {k: _value_of(v) for k, v in self.pso_fields.items()}
            kalman = {k: _value_of(v) for k, v in self.kalman_fields.items()}

            # ASTRA: handle pick_offset specially
            astra = {}
            for k, v in self.astra_fields.items():
                if k not in ("pick_offset_mode", "pick_offset_custom"):
                    astra[k] = _value_of(v)

            # Resolve pick_offset final value
            if "pick_offset_mode" in self.astra_fields and "pick_offset_custom" in self.astra_fields:
                mode = self.astra_fields["pick_offset_mode"].currentText()
                if mode == "custom":
                    custom_val = self.astra_fields["pick_offset_custom"].text().strip()
                    # Store the FLOAT STRING itself, NOT the word 'custom'
                    astra["pick_offset"] = custom_val
                else:
                    astra["pick_offset"] = mode
            else:
                # Backward-compat fallback
                astra["pick_offset"] = _value_of(self.astra_fields.get("pick_offset", QLineEdit("leading-edge")))

            self.config = {
                "file_path": self.selected_file_label.text(),
                "pso":   pso,
                "astra": astra,
                "kalman": kalman,
            }

            # Store config in parent in case of close
            self.skyfit_instance.astra_config_params = self.config


        def getConfig(self):
            """Returns the ASTRA config object."""
            self.storeConfig()
            return self.config
        
        def startASTRAThread(self):
            """Creates and runs an ASTRA process on a seperate worker thread"""

            self.storeConfig()
            errors = self.checkConfig()
            if errors != {}:
                self.skyfit_instance.setMessageBox(title="ASTRA Configuration Error",
                                                message="\n".join(errors.values()),
                                                type='error')
                return

            config = self.getConfig()

            self.thread = QThread()
            self.worker = AstraWorker(config, self.skyfit_instance)
            print('ASTRA Object Created! Processing beginning (30-80 seconds)...')
            self.worker.moveToThread(self.thread)

            self.run_astra_btn.setEnabled(False)
            self.run_kalman_btn.setEnabled(False)
            self.file_picker_button.setEnabled(False)
            self.set_prev_picks_button.setEnabled(False)
            self.setASTRAStatus("PROCESSING", "ASTRA is running...")

            self.worker.progress.connect(self.updateProgress)

            self.thread.started.connect(self.worker.run)

            self.worker.results_ready.connect(
                self.skyfit_instance.integrateASTRAResults, 
                QtCore.Qt.QueuedConnection
            )

            # Clean up and re-enable UI
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            

            # Restore interactivity
            self.worker.finished.connect(lambda: (not getattr(self, "_is_closing", False)) and self.file_picker_button.setEnabled(True))

            # Only call parent checks if THIS dialog is still the active one
            self.worker.finished.connect(lambda:
                (self.skyfit_instance is not None)
                and (getattr(self.skyfit_instance, "astra_dialog", None) is self)
                and self.skyfit_instance.checkASTRACanRun()
            )
            self.worker.finished.connect(lambda:
                (self.skyfit_instance is not None)
                and (getattr(self.skyfit_instance, "astra_dialog", None) is self)
                and self.skyfit_instance.checkKalmanCanRun()
            )
            self.worker.finished.connect(lambda:
                (self.skyfit_instance is not None)
                and (getattr(self.skyfit_instance, "astra_dialog", None) is self)
                and self.skyfit_instance.checkPickRevertCanRun()
            )

            self.thread.start()

        def setPreviousPicks(self):
            """Hits parent instance to revert picks to previous state"""
            self.skyfit_instance.reverseASTRAPicks()
        
        def checkFloat(self, value):
            """Checks if a value is castable to float"""
            try:
                float(value)
                return True
            except (ValueError, TypeError):
                return False

        def checkConfig(self):
            """Checks the config only has valid values"""
            config = self.getConfig()

            pso_ranges_and_types = {
                "w (0-1)": (0.0, 1.0, float),
                "c_1 (0-1)": (0.0, 1.0, float),
                "c_2 (0-1)": (0.0, 1.0, float),
                "max itter": (1, None, int),
                "n_particles": (1, None, int),
                "V_c (0-1)": (0.0, 1.0, float),
                "ftol": (0.0, 100.0, float),
                "ftol_itter": (1, None, int),
                "expl_c": (1.0, None, float),
                "P_sigma": (0.0, None, float)
            }

            astra_ranges_and_types = {
                "star_thresh": (0, None, float), "min SNR": (0, None, float),
                "P_crop": (0, None, float), "sigma_init (px)": (0.1, None, float), "sigma_max": (1, None, float),
                "L_max": (1, None, float), "Verbose": (True, False, bool), "photom_thresh": (0, 1, float),
                "Save Animation": (True, False, bool),
                # NOTE: only 'center' and 'leading-edge' are valid literals; 'custom' is resolved to a float string at store time
                "pick_offset": (["center", "leading-edge"], None, str)
            }
            kalman_ranges_and_types = {
                "Monotonicity": (True, False, bool), "sigma_xy (px)": (0, None, float), 
                "sigma_vxy (%)": (0, None, float), "save results": (True, False, bool)
            }

            errors = {}

            # Check PSO parameters
            for param, (min_val, max_val, param_type) in pso_ranges_and_types.items():
                value_str = config["pso"].get(param, "")
                try:
                    if param_type == bool:
                        value = value_str.strip().lower() == "true"
                        if value not in [True, False]:
                            errors[f"pso.{param}"] = f"{param} must be True or False, got {value_str}"
                    else:
                        value = param_type(value_str)
                        if min_val is not None and value < min_val:
                            errors[f"pso.{param}"] = f"{param} must be >= {min_val}, got {value}"
                        elif max_val is not None and value > max_val:
                            errors[f"pso.{param}"] = f"{param} must be <= {max_val}, got {value}"
                except (ValueError, TypeError):
                    errors[f"pso.{param}"] = f"{param} must be {param_type.__name__}, got {value_str}"

            # Check ASTRA parameters
            for param, (min_val, max_val, param_type) in astra_ranges_and_types.items():
                value_str = config["astra"].get(param, "")
                try:
                    if param == "pick_offset":
                        value = value_str.strip()
                        # Allow literals or a float
                        if value not in ["center", "leading-edge"] and self.checkFloat(value) is False:
                            errors[f"astra.{param}"] = (
                                f"{param} must be 'center', 'leading-edge', or a float; got {value_str}"
                            )
                    elif param_type == bool:
                        value = value_str.strip().lower() == "true"
                        if value not in [True, False]:
                            errors[f"astra.{param}"] = f"{param} must be True or False, got {value_str}"
                    else:
                        value = param_type(value_str)
                        if min_val is not None and value < min_val:
                            errors[f"astra.{param}"] = f"{param} must be >= {min_val}, got {value}"
                        elif max_val is not None and value > max_val:
                            errors[f"astra.{param}"] = f"{param} must be <= {max_val}, got {value}"
                except (ValueError, TypeError):
                    errors[f"astra.{param}"] = f"{param} must be {param_type.__name__}, got {value_str}"

            # Check Kalman parameters
            for param, (min_val, max_val, param_type) in kalman_ranges_and_types.items():
                value_str = config["kalman"].get(param, "")
                try:
                    if param_type == bool:
                        value = value_str.strip().lower() == "true"
                        if value not in [True, False]:
                            errors[f"kalman.{param}"] = f"{param} must be True or False, got {value_str}"
                    else:
                        value = param_type(value_str)
                        if min_val is not None and value < min_val:
                            errors[f"kalman.{param}"] = f"{param} must be >= {min_val}, got {value}"
                        elif max_val is not None and value > max_val:
                            errors[f"kalman.{param}"] = f"{param} must be <= {max_val}, got {value}"
                except (ValueError, TypeError):
                    errors[f"kalman.{param}"] = f"{param} must be {param_type.__name__}, got {value_str}"

            return errors


        def startKalmanThread(self):
            """Creates and runs a Kalman process on a separate worker thread"""

            self.storeConfig()
            errors = self.checkConfig()
            if errors != {}:
                self.skyfit_instance.setMessageBox(title="ASTRA Configuration Error",
                                                message="\n".join(errors.values()),
                                                type='error')
                return
            self.config = self.getConfig()

            self.run_kalman_btn.setEnabled(False)
            self.run_astra_btn.setEnabled(False)
            self.file_picker_button.setEnabled(False)
            self.set_prev_picks_button.setEnabled(False)
            self.setASTRAStatus("PROCESSING", "ASTRA is running...")

            self.kalman_worker = KalmanWorker(self.skyfit_instance, self.config)
            self.kalman_worker.progress.connect(self.updateProgress)
            self.kalman_worker.results_ready.connect(
                self.skyfit_instance.applyKalmanResults,
                QtCore.Qt.QueuedConnection
            )

            # Restore interactivity
            self.kalman_worker.finished.connect(lambda: (not getattr(self, "_is_closing", False)) and self.file_picker_button.setEnabled(True))

            self.kalman_worker.finished.connect(lambda:
                (self.skyfit_instance is not None)
                and (getattr(self.skyfit_instance, "astra_dialog", None) is self)
                and self.skyfit_instance.checkASTRACanRun()
            )
            self.kalman_worker.finished.connect(lambda:
                (self.skyfit_instance is not None)
                and (getattr(self.skyfit_instance, "astra_dialog", None) is self)
                and self.skyfit_instance.checkKalmanCanRun()
            )
            self.kalman_worker.finished.connect(lambda:
                (self.skyfit_instance is not None)
                and (getattr(self.skyfit_instance, "astra_dialog", None) is self)
                and self.skyfit_instance.checkPickRevertCanRun()
            )

            self.kalman_worker.finished.connect(lambda: (not getattr(self, "_is_closing", False)) and self.updateProgress(100))

            self.kalman_worker.start()  

    class KalmanWorker(QThread):
        progress = pyqtSignal(int)
        finished = pyqtSignal()
        results_ready = pyqtSignal(object)

        def __init__(self, skyfit_instance, config):
            super().__init__()
            self.skyfit_instance = skyfit_instance
            self.config = config
            self._stop = False

        def stop(self):
            self._stop = True

        def _progress_guard(self, value):
            if self._stop or self.isInterruptionRequested():
                raise RuntimeError("Kalman aborted by user")
            self.progress.emit(value)

        def run(self):
            try:

                if self._stop or self.isInterruptionRequested():
                    self.finished.emit()
                    return
                
                result = self.skyfit_instance.runKalmanFromConfig(
                    self.config, progress_callback=self._progress_guard
                )

                if result is not None:
                    self.results_ready.emit(result)

            except Exception as e:
                print(f'Error running Kalman: {e}')

            finally:
                self.finished.emit()

    class AstraWorker(QObject):
        finished = pyqtSignal()
        progress = pyqtSignal(int)
        results_ready = pyqtSignal(object)

        def __init__(self, config, skyfit_instance):
            super().__init__()
            self.config = config
            self.skyfit_instance = skyfit_instance
            self._stop = False

        def stop(self):
            self._stop = True

        def _progress_guard(self, value):
            # Called by ASTRA during processing
            if self._stop:
                raise RuntimeError("ASTRA aborted by user")
            self.progress.emit(value)

        def run(self):
            # try:
            # Optional early exit
            if self._stop:
                print('early exit')
                self.finished.emit()
                return

            # Prepare data
            data_dict = self.skyfit_instance.prepareASTRAData(self.config)
            if data_dict is False or self._stop:
                print('no data dict')
                self.finished.emit()
                return

            # Run ASTRA; pass our guard so we can interrupt mid-flight
            astra = ASTRA(**data_dict, progress_callback=self._progress_guard)
            if self._stop:
                print('interupted')
                self.finished.emit()
                return

            astra.process()

            if not self._stop:
                self.results_ready.emit(astra)

    def launchASTRAGUI(run_astra_callback=None,
                        run_kalman_callback=None,
                        run_load_callback=None,
                        parent=None,
                        skyfit_instance=None,
                        on_close_callback=None):
        dialog = AstraConfigDialog(
            run_astra_callback=run_astra_callback,
            run_kalman_callback=run_kalman_callback,
            run_load_callback=run_load_callback,
            parent=parent,
            skyfit_instance=skyfit_instance,
            on_close_callback=on_close_callback
        )
        dialog.show()
        return dialog



class QFOVinputDialog(QtWidgets.QDialog):

    lenses = "none"
    lenses_vbox = None

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

        # Reference lenses options
        groupbox = QtWidgets.QGroupBox("Template lenses:")
        groupbox.setCheckable(False)
        layout.addWidget(groupbox)

        self.lenses_vbox = QtWidgets.QVBoxLayout()
        groupbox.setLayout(self.lenses_vbox)

        fov = QtWidgets.QRadioButton("No distortion (default)")
        fov.lenses = "none"
        fov.setChecked(True)
        fov.toggled.connect(self.lensesSelected)
        self.lenses_vbox.addWidget(fov)

        layout.addWidget(buttonBox)
        self.setLayout(layout)

    def loadLensTemplates(self, config, data_dir, width, height):
        """ Load the lens templates and add them to the dialog box. The provided resolution will be used to
            enable only the templates with the same resolution.

        Arguments:
            width: [int] Image width.
            height: [int] Image height.
        """

        print()
        print('Loading platepar templates from:')
        print(" ", config.platepar_template_dir)
        print(" ", data_dir)

        # Find all template_*.cal files in both the data and config directories
        platepar_template_files = []
        for root_dir in [config.platepar_template_dir, data_dir]:
            for template_file in glob.glob(os.path.join(root_dir, 'template_*.cal')):
                platepar_template_files.append(template_file)


        # Load the lens templates
        templates = []
        for template_path in platepar_template_files:

            with open(template_path) as f:
                data = json.load(f)

            if ('template_metadata' not in data) or \
                     ('description' not in data['template_metadata']):
                
                print('WARNING: Missing or invalid "template_metadata" section in file ' + template_file)

                continue

            templates.append({
                       'description' : data['template_metadata']['description'],
                       'X_res' : data['X_res'],
                       'Y_res' : data['Y_res'],
                       'file_name' : template_path
                      })
        templates.sort(key=lambda x: x['description'])

        # add lenses options to the dialog box
        for template in templates:
            fov = self.createTemplateLensOption(template['file_name'], template['description'])

            # enable option if resolution is compatible
            if template['X_res'] == width and template['Y_res'] == height:
                fov.setEnabled(True)

    def createTemplateLensOption(self, lenses_id, lenses_description):

        fov = QtWidgets.QRadioButton(lenses_description)
        fov.lenses = lenses_id
        fov.setEnabled(False)
        fov.toggled.connect(self.lensesSelected)
        self.lenses_vbox.addWidget(fov)
        return fov

    def lensesSelected(self):
        radioButton = self.sender()
        if radioButton.isChecked():
            self.lenses = radioButton.lenses


    def getInputs(self):

        try:
            azim = float(self.azim_edit.text())%360

        except ValueError:
            print("Azimuth could not be read as a number! Assuming 90 deg.")
            azim = 90


        try:
            alt = float(self.alt_edit.text())
        
        except ValueError:
            print("Altitude could not be read as a number! Assuming 45 deg.")
            alt = 45

        # Read the rotation
        rot_text = self.rot_edit.text()
        if rot_text:
            rot = float(rot_text)%360
        else:
            # If the rotation is not given, set it to 0
            rot = 0

        lenses = self.lenses

        return azim, alt, rot, lenses


class GeoPoints(object):
    def __init__(self, geo_points_input):

        self.geo_points_input = geo_points_input

        # Geo coordinates (degrees, meters)
        self.names = []
        self.lat_data = []
        self.lon_data = []
        self.ele_data = []

        # Equatorial coordinates (degrees)
        self.ra_data = []
        self.dec_data = []

        # Load the points from a file
        self.load()


    def load(self):
        """ Load the geo catalog file. """
        
        if os.path.isfile(self.geo_points_input):
            with open(self.geo_points_input) as f:
                for line in f:

                    # Skip comments
                    if line.startswith("#"):
                        continue

                    line = line.replace('\n', '').replace('\r', '')
                    line = line.split(',')

                    name, lat, lon, ele = line

                    self.names.append(name)
                    self.lat_data.append(float(lat))
                    self.lon_data.append(float(lon))
                    self.ele_data.append(float(ele))




    def update(self, platepar, jd):
        """ Project points to the observer's point of view. """

        # Reset RA/Dec array
        self.ra_data = []
        self.dec_data = []

        # Compute ECI coordinates of the observer's location
        ref_eci = geo2Cartesian(platepar.lat, platepar.lon, platepar.elev, jd)

        for name, lat, lon, elev in zip(self.names, self.lat_data, self.lon_data, self.ele_data):

            # Compute ECI coordinates of the current point
            eci = geo2Cartesian(lat, lon, elev, jd)

            # Compute the vector pointing from the reference position to the current position
            eci_point = np.array(eci) - np.array(ref_eci)

            # Compute ra/dec in radians
            ra, dec = vector2RaDec(eci_point)

            # # Compute alt/az
            # azim, alt = raDec2AltAz(np.radians(ra), np.radians(dec), jd, np.radians(platepar.lat), \
            #     np.radians(platepar.lon))

            # print("{:>25s}, {:8.3f}, {:7.3f}".format(name, np.degrees(azim), np.degrees(alt)))


            # Precess RA/Dec to J2000
            ra, dec = equatorialCoordPrecession(jd, J2000_JD.days, np.radians(ra), np.radians(dec))

            self.ra_data.append(np.degrees(ra))
            self.dec_data.append(np.degrees(dec))


        self.ra_data = np.array(self.ra_data)
        self.dec_data = np.array(self.dec_data)


# CatalogStar, GeoPoint, and PairedStars are now imported from RMS.Astrometry.StarClasses


class PlateTool(QtWidgets.QMainWindow):
    def __init__(self, input_path, config, beginning_time=None, fps=None, gamma=None, use_fr_files=False,
        geo_points_input=None, startUI=True, mask=None, nobg=False, peribg=False, flipud=False,
        flatbiassub=False, exposure_ratio=1.0, show_sattracks=False, tle_file=None):
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
            geo_points_input: [str] Path to a file with a list of geo coordinates which will be projected on
                the image as seen from the perspective of the observer.
            startUI: [bool] Start the GUI. True by default.
            mask: [str] Path to a mask file.
            nobg: [bool] Do not subtract the background for photometry. False by default.
            peribg: [bool] Perform background subtraction using the average of the pixels adjacent to the 
                coloured mask instead of the avepixel. False by default.
            flipud: [bool] Flip the image upside down. False by default.
            flatbiassub: [bool] Subtract flat and bias frames. False by default.
            exposure_ratio: [float] Exposure ratio between stars and meteors. Used for magnitude scaling of 
                meteors observed on long exposure images with shutters. The correct exp. ratio is already 
                automatically applied for DFN images. 1.0 by default.
            show_sattracks: [bool] Show satellite tracks. False by default.
            tle_file: [str] Path to a TLE file for satellite tracks.
        """

        super(PlateTool, self).__init__()

        # Mode of operation - skyfit for fitting astrometric plates, manualreduction for manual picking
        #   of position on frames and photometry
        self.mode = 'skyfit'
        self.mode_list = ['skyfit', 'manualreduction']
        self.max_pixels_between_matched_stars = np.inf
        self.autopan_mode = False
        self.input_path = input_path
        if os.path.isfile(self.input_path):
            self.dir_path = os.path.dirname(self.input_path)
        else:
            self.dir_path = self.input_path


        self.config = config

        # Force the CV2 backend when SkyFit is being used
        self.config.media_backend = 'cv2'

        # Store forced time of first frame
        self.beginning_time = beginning_time

        # Store the background subtraction flag
        self.no_background_subtraction = nobg

        # Store the peripheric background subtraction flag
        self.peripheral_background_subtraction = peribg

        # Store the flip upside down flag
        self.flipud = flipud

        # Store the flat and bias subtraction flag
        self.flatbiassub = flatbiassub

        # Store the exposure ratio
        self.exposure_ratio = exposure_ratio

        # Extract the directory path if a file was given
        if os.path.isfile(self.dir_path):
            self.dir_path, _ = os.path.split(self.dir_path)

        # If camera gamma was given, change the value in config
        if gamma is not None:
            config.gamma = gamma

        # If the FPS was given, change the FPS in the config file
        self.fps = fps
        if self.fps is not None:
            config.fps = fps

        self.use_fr_files = use_fr_files

        
        # Load the file with geo points, if given
        self.geo_points_input = geo_points_input
        
        self.geo_points_obj = None

        if self.geo_points_input is not None:
            
            if os.path.isfile(self.geo_points_input):
                self.geo_points_obj = GeoPoints(self.geo_points_input)

            else:
                print("The file with geo points does not exist:", self.geo_points_input)


        # Measure points on the ground, not on the sky
        self.meas_ground_points = False

        # Do the astrometric pick and the photometry in a single click
        self.single_click_photometry = False

        # Star picking mode variables
        self.star_aperture_radius = 5
        self.x_centroid = self.y_centroid = self.star_fwhm = self.snr_centroid = None
        self.saturated_centroid = False
        self.closest_type = None
        self.closest_cat_star_indx = None

        # List of paired image and catalog stars
        self.pick_list = {}
        self.paired_stars = PairedStars()
        self.residuals = None

        # Autopan coordinates
        self.old_autopan_x, self.old_autopan_y = None, None
        self.current_autopan_x, self.current_autopan_y = None, None

        # List of unsuitable stars
        self.unsuitable_stars = PairedStars()

        # Positions of the mouse cursor
        self.mouse_x = 0
        self.mouse_y = 0

        # Key increment
        self.key_increment = 1.0
        
        # Init a blank platepar
        self.platepar = Platepar()

        # Platepar format (json or txt)
        self.platepar_fmt = None

        # Store the mask
        self.mask = mask

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

        # Flag indicating that only the pointing will be fit, not the distortion
        self.fit_only_pointing = False

        # Flag indicating that the scale will not be fit
        self.fixed_scale = False

        # Flag indicating that the astrometry will be automatically re-fit when the station is moved (only 
        #   when geopoints are available)
        self.station_moved_auto_refit = False

        # Photometry parameters
        self.photom_fit_stddev = None
        self.photom_fit_resids = None

        # Compute the saturation threshold
        self.saturation_threshold = int(round(0.98*(2**self.config.bit_depth - 1)))

        # ASTRA class variables
        if ASTRA_IMPORTED:

            # List of large pick changes (import, ASTRA, kalman) to support reversion
            self.previous_picks = []

            # Initialize the ASTRA dialog reference
            self.astra_dialog = None
            self.astra_config_params = None

        
        # Satellite tracks config
        self.show_sattracks = show_sattracks
        self.tle_file = tle_file
        self.satellite_tracks = []
        
        # Cache for FOV polygon to avoid recomputation
        self.fov_poly_cache = None
        self.fov_poly_jd = None

        self.sat_track_curves = []
        self.sat_track_labels = []
        self.sat_track_arrows = []
        self.sat_markers = []

        if self.show_sattracks:
             if SKYFIELD_AVAILABLE:
                print("Satellite tracks enabled.")
             else:
                print("WARNING: --sattracks requested but skyfield not installed.")
                self.show_sattracks = False


    
        ###################################################################################################


        # Detect data input type and init the image handle
        self.detectInputType(load=True, beginning_time=beginning_time, use_fr_files=self.use_fr_files)

        # Update the FPS if it's forced
        self.setFPS()


        ###################################################################################################
        
        # Display options
        self.show_spectral_type = False
        self.show_star_names = False
        self.show_constellations = False
        self.selected_stars_visible = True

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
            print('Star catalog loaded: ', self.config.star_catalog_file)


        self.calstars = {}
        self.loadCalstars()

        # Star detection override parameters
        self.star_detection_override_enabled = False  # Use override parameters instead of CALSTARS
        self.star_detection_override_data = {}  # Store re-detected stars per FF file
        self.override_intensity_threshold = 18
        self.override_neighborhood_size = 10
        self.override_max_stars = 200
        self.override_gamma = 1.0
        self.override_segment_radius = 4
        self.override_max_feature_ratio = 0.8
        self.override_roundness_threshold = 0.5
        self._original_config_gamma = None  # Store original config gamma for restoration

        # Mask drawing state
        self.mask_draw_mode = False
        self.mask_current_polygon = []  # Points being drawn
        self.mask_polygons = []  # List of completed polygons
        self.mask_dragging_vertex = None  # (polygon_idx, vertex_idx) or ('current', vertex_idx)

        # Flat image for mask editing background
        self.flat_image_data = None  # Loaded flat.bmp data
        self.mask_use_flat_background = False  # Whether to show flat as background


        ###################################################################################################
        # PLATEPAR

        # Load the platepar file
        self.loadPlatepar()


        # Set the given gamma value to platepar
        if gamma is not None:
            self.platepar.gamma = gamma


        # Load distortion type index
        self.dist_type_index = self.platepar.distortion_type_list.index(self.platepar.distortion_type)

        ###################################################################################################

        print()

        # INIT WINDOW
        if startUI:
            self.setupUI()


        ###################################################################################################

        # Automatically load the flat and bias in UWO data mode
        if self.usingUWOData():
            _, self.flat_struct = self.loadFlat()
            _, self.dark = self.loadDark()

                        
            # Set focus back on the SkyFit window
            self.activateWindow()

            # Apply the dark to the flat if the flatbiassub flag is set
            if self.flatbiassub and (self.flat_struct is not None):
                
                self.flat_struct.applyDark(self.dark)

                self.img.flat_struct = self.flat_struct
                self.img_zoom.flat_struct = self.flat_struct


            self.img.dark = self.dark
            self.img_zoom.dark = self.dark

            self.img_zoom.flat_struct = self.flat_struct
            self.img.flat_struct = self.flat_struct
            self.img_zoom.reloadImage()
            self.img.reloadImage()

        # If the satellite tracks should be shown, load the TLE data and show the tracks
        if show_sattracks:
            self.loadSatelliteTracks()


    def closeEvent(self, event):
        """ Handle window close event to properly exit application. """
        event.accept()
        QtWidgets.QApplication.quit()


    def setFPS(self):
        """ Update the FPS if it's forced. """

        # Force FPS for videos if needed
        if (self.fps is not None) and (self.img_handle is not None):
            if self.img_handle.input_type == "video":
                self.img_handle.fps = self.fps
                print("Forcing video FPS to:", self.fps)


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
                                                              self.saveFTPdetectinfo(ECSV_saved=self.saveECSV())])

        self.save_current_frame_action = QtWidgets.QAction('Save current frame')
        self.save_current_frame_action.setShortcut('Ctrl+W')
        self.save_current_frame_action.triggered.connect(self.saveCurrentFrame)

        self.save_default_platepar_action = QtWidgets.QAction("Save default platepar...")
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
        self.status_bar.setFont(QtGui.QFont('monospace'))
        self.setStatusBar(self.status_bar)

        self.skyfit_button = QtWidgets.QPushButton('SkyFit')
        self.skyfit_button.pressed.connect(lambda: self.changeMode('skyfit'))
        self.manualreduction_button = QtWidgets.QPushButton('ManualReduction')
        self.manualreduction_button.pressed.connect(lambda: self.changeMode('manualreduction'))
        self.status_bar.addPermanentWidget(self.skyfit_button)
        self.status_bar.addPermanentWidget(self.manualreduction_button)

        # Image navigation slider (like a video timeline)
        self.image_navigation_label = QtWidgets.QLabel('Image: 1 / 1')
        self.image_navigation_label.setMinimumWidth(80)
        self.status_bar.addPermanentWidget(self.image_navigation_label)

        self.image_navigation_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.image_navigation_slider.setMinimum(1)
        self.image_navigation_slider.setMaximum(1)
        self.image_navigation_slider.setValue(1)
        self.image_navigation_slider.setMinimumWidth(200)
        self.image_navigation_slider.setMaximumWidth(400)
        self.image_navigation_slider.setToolTip("Drag or click to navigate through images")
        self.image_navigation_slider.valueChanged.connect(self.jumpToImage)
        self.status_bar.addPermanentWidget(self.image_navigation_slider)

        self.nextstar_button = QtWidgets.QPushButton('SkyFit')
        self.nextstar_button.pressed.connect(self.jumpNextStar)

        ###################################################################################################
        # CENTRAL WIDGET (DISPLAY)

        # Main Image
        self.scrolls_back = 0
        self.clicked = 0
        # Track press position for click vs drag detection (scene coords for panning compatibility)
        self.press_scene_x = None
        self.press_scene_y = None
        self.press_button = None
        self.press_modifiers = None

        # Init the central image window
        self.view_widget = pg.GraphicsView()
        self.img_frame = ViewBox()
        self.img_frame.setAspectLocked()
        self.img_frame.setMenuEnabled(False)

        # Override the scroll function
        self.img_frame.wheelEvent = self.wheelEvent

        # Install event filter to catch mouse release (ViewBox doesn't receive it during panning)
        self.view_widget.viewport().installEventFilter(self)

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

        # Create font and metrics for overlay labels
        label_font = QtGui.QFont('monospace', 8)
        label_fm = QtGui.QFontMetrics(label_font)

        self.label1 = TextItem(color=(0, 0, 0), fill=(255, 255, 255, 100))
        self.label1.setFont(label_font)
        self.label1.setTextWidth(label_fm.averageCharWidth() * 35)  # ~35 chars wide
        self.label1.setZValue(1000)
        self.label1.setParentItem(self.img_frame)

        # bottom left label
        self.label2 = TextItem(color=(0, 0, 0), fill=(255, 255, 255, 100))
        self.label2.setFont(label_font)
        self.label2.setTextWidth(label_fm.averageCharWidth() * 38)  # ~38 chars wide
        self.label2.setZValue(1000)
        self.label2.setParentItem(self.img_frame)

        # F1 info label
        self.label_f1 = TextItem(color=(0, 0, 0), fill=(255, 255, 255, 100))
        self.label_f1.setFont(label_font)
        self.label_f1.setTextWidth(label_fm.averageCharWidth() * 20)  # ~20 chars wide
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

        # geo points markers (main window)
        self.geo_markers = pg.ScatterPlotItem()
        self.geo_markers.setPen('g')
        self.geo_markers.setBrush((0, 0, 0, 0))
        self.geo_markers.setSymbol(Plus())
        self.geo_markers.setZValue(4)
        self.img_frame.addItem(self.geo_markers)

        # geo points markers (zoom window)
        self.geo_markers2 = pg.ScatterPlotItem()
        self.geo_markers2.setPen('g')
        self.geo_markers2.setBrush((0, 0, 0, 0))
        self.geo_markers2.setSize(20)
        self.geo_markers2.setSymbol(Plus())
        self.geo_markers2.setZValue(4)
        self.zoom_window.addItem(self.geo_markers2)

        # astrometry.net matched star markers (main window) - cyan color
        # These show catalog star positions that matched to input stars
        self.astrometry_matched_markers = pg.ScatterPlotItem()
        self.astrometry_matched_markers.setPen('c', width=2)  # cyan
        self.astrometry_matched_markers.setBrush((0, 0, 0, 0))
        self.astrometry_matched_markers.setSize(15)
        self.astrometry_matched_markers.setSymbol('o')  # circle
        self.astrometry_matched_markers.setZValue(5)
        self.img_frame.addItem(self.astrometry_matched_markers)

        # astrometry.net matched star markers (zoom window)
        self.astrometry_matched_markers2 = pg.ScatterPlotItem()
        self.astrometry_matched_markers2.setPen('c', width=2)
        self.astrometry_matched_markers2.setBrush((0, 0, 0, 0))
        self.astrometry_matched_markers2.setSize(25)
        self.astrometry_matched_markers2.setSymbol('o')
        self.astrometry_matched_markers2.setZValue(5)
        self.zoom_window.addItem(self.astrometry_matched_markers2)

        # astrometry.net quad star markers (main window) - magenta color
        # These are the specific catalog stars used for the initial quad pattern matching
        self.astrometry_quad_markers = pg.ScatterPlotItem()
        self.astrometry_quad_markers.setPen('m', width=2)  # magenta
        self.astrometry_quad_markers.setBrush((0, 0, 0, 0))
        self.astrometry_quad_markers.setSize(20)
        self.astrometry_quad_markers.setSymbol('s')  # square
        self.astrometry_quad_markers.setZValue(5)
        self.img_frame.addItem(self.astrometry_quad_markers)

        # astrometry.net quad star markers (zoom window)
        self.astrometry_quad_markers2 = pg.ScatterPlotItem()
        self.astrometry_quad_markers2.setPen('m', width=2)
        self.astrometry_quad_markers2.setBrush((0, 0, 0, 0))
        self.astrometry_quad_markers2.setSize(30)
        self.astrometry_quad_markers2.setSymbol('s')
        self.astrometry_quad_markers2.setZValue(5)
        self.zoom_window.addItem(self.astrometry_quad_markers2)

        # Store astrometry.net solution info (populated when astrometry.net is run)
        self.astrometry_solution_info = None
        self.astrometry_stars_visible = False

        # Initially hide astrometry.net markers
        self.astrometry_matched_markers.hide()
        self.astrometry_matched_markers2.hide()
        self.astrometry_quad_markers.hide()
        self.astrometry_quad_markers2.hide()

        self.catalog_stars_visible = True
        self.show_spectral_type = False
        self.show_star_names = False
        self.show_constellations = False
        self.selected_stars_visible = True
        self.unsuitable_stars_visible = True

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

        # selected unsuitable star markers (main window)
        self.unsuitable_star_markers = pg.ScatterPlotItem()
        self.unsuitable_star_markers.setPen('r', width=3)
        self.unsuitable_star_markers.setSize(10)
        self.unsuitable_star_markers.setSymbol('s')
        self.unsuitable_star_markers.setZValue(4)
        self.img_frame.addItem(self.unsuitable_star_markers)

        # selected catalog star markers (zoom window)
        self.unsuitable_star_markers2 = pg.ScatterPlotItem()
        self.unsuitable_star_markers2.setPen('r', width=3)
        self.unsuitable_star_markers2.setSize(10)
        self.unsuitable_star_markers2.setSymbol('s')
        self.unsuitable_star_markers2.setZValue(4)
        self.zoom_window.addItem(self.unsuitable_star_markers2)

        self.unsuitable_stars_visble = True

        # Distortion center marker (red cross) - main window
        self.distortion_center_marker = pg.ScatterPlotItem()
        self.distortion_center_marker.setPen((255, 0, 0), width=2)
        self.distortion_center_marker.setSize(30)
        self.distortion_center_marker.setSymbol(Plus())
        self.distortion_center_marker.setZValue(5)
        self.img_frame.addItem(self.distortion_center_marker)

        # Distortion center marker (red cross) - zoom window
        self.distortion_center_marker2 = pg.ScatterPlotItem()
        self.distortion_center_marker2.setPen((255, 0, 0), width=2)
        self.distortion_center_marker2.setSize(30)
        self.distortion_center_marker2.setSymbol(Plus())
        self.distortion_center_marker2.setZValue(5)
        self.zoom_window.addItem(self.distortion_center_marker2)

        # calstar markers outer rings (dark green) - main window
        self.calstar_markers_outer = pg.ScatterPlotItem()
        self.calstar_markers_outer.setPen(pg.mkPen((0, 100, 0, 255), width=1.5))
        self.calstar_markers_outer.setBrush((0, 0, 0, 0))
        self.calstar_markers_outer.setSize(14)
        self.calstar_markers_outer.setSymbol('o')
        self.calstar_markers_outer.setZValue(2)
        self.img_frame.addItem(self.calstar_markers_outer)

        # calstar markers inner rings (bright green) - main window
        self.calstar_markers = pg.ScatterPlotItem() 
        self.calstar_markers.setPen(pg.mkPen((50, 255, 50, 255), width=1.1))
        self.calstar_markers.setBrush((0, 0, 0, 0))
        self.calstar_markers.setSize(10)
        self.calstar_markers.setSymbol('o')
        self.calstar_markers.setZValue(2)
        self.img_frame.addItem(self.calstar_markers)

        # calstar markers outer rings (dark green) - zoom window
        self.calstar_markers_outer2 = pg.ScatterPlotItem()
        self.calstar_markers_outer2.setPen(pg.mkPen((0, 100, 0, 255), width=1.5))
        self.calstar_markers_outer2.setBrush((0, 0, 0, 0))
        self.calstar_markers_outer2.setSize(28)
        self.calstar_markers_outer2.setSymbol('o')
        self.calstar_markers_outer2.setZValue(5)
        self.zoom_window.addItem(self.calstar_markers_outer2)

        # calstar markers inner rings (bright green) - zoom window
        self.calstar_markers2 = pg.ScatterPlotItem()
        self.calstar_markers2.setPen(pg.mkPen((50, 255, 50, 255), width=1.1))

        # Satellite tracks items
        self.sat_track_curves = []
        self.sat_track_labels = []

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
        self.star_pick_info_text_str = "STAR PICKING MODE keys:\n"
        self.star_pick_info_text_str += "LEFT CLICK - Centroid star\n"
        self.star_pick_info_text_str += "CTRL + LEFT CLICK - Manual star position\n"
        self.star_pick_info_text_str += "ENTER or SPACE - Accept pair\n"
        self.star_pick_info_text_str += "CTRL + SPACE - Mark pair bad\n"
        self.star_pick_info_text_str += "SHIFT + SPACE - Jump random\n"
        self.star_pick_info_text_str += "RIGHT CLICK - Remove pair\n"
        self.star_pick_info_text_str += "CTRL + SCROLL - Aperture radius adjust\n"
        self.star_pick_info_text_str += "CTRL + Z - Fit stars\n"
        self.star_pick_info_text_str += "CTRL + SHIFT + Z - Fit with initial distortion params set to 0\n"
        self.star_pick_info_text_str += "L - Astrometry fit plot\n"
        self.star_pick_info_text_str += "P - Photometry fit plot"
        self.star_pick_info = TextItem(self.star_pick_info_text_str, anchor=(0.0, 0.75), color=(0, 0, 0), fill=(255, 255, 255, 100))
        self.star_pick_info.setFont(QtGui.QFont('monospace', 8))
        self.star_pick_info.setAlign(QtCore.Qt.AlignLeft)
        self.star_pick_info.hide()
        self.star_pick_info.setZValue(10)
        self.star_pick_info.setParentItem(self.img_frame)
        self.star_pick_info.setPos(0, self.platepar.Y_res)

        # Default variables even when constructor isn't called
        self.star_pick_mode = False

        # Cursor
        self.cursor = CursorItem(self.star_aperture_radius, pxmode=True)
        self.img_frame.addItem(self.cursor, ignoreBounds=True)
        self.cursor.hide()
        self.cursor.setZValue(20)

        # Cursor (window)
        self.cursor2 = CursorItem(self.star_aperture_radius, pxmode=True, thickness=2)
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


        # Great circle fit
        self.great_circle_line = pg.PlotCurveItem(pen=pg.mkPen((138, 43, 226, 255), style=QtCore.Qt.DotLine))
        self.great_circle_line.setZValue(1)
        self.img_frame.addItem(self.great_circle_line)


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

        # Add spectral types text items
        self.spectral_type_text_list = TextItemList()
        self.img_frame.addItem(self.spectral_type_text_list)

        # Plotting item for constellation lines
        # Background (black border)
        self.constellation_lines_bg = pg.PlotCurveItem(
            pen=pg.mkPen((0, 0, 0, 128), width=5), connect='pairs')
        self.constellation_lines_bg.setZValue(4)
        self.constellation_lines_bg.hide()
        self.img_frame.addItem(self.constellation_lines_bg)

        # Foreground (white line)
        self.constellation_lines_fg = pg.PlotCurveItem(
            pen=pg.mkPen((255, 255, 255, 128), width=2), connect='pairs')
        self.constellation_lines_fg.setZValue(5)
        self.constellation_lines_fg.hide()
        self.img_frame.addItem(self.constellation_lines_fg)

        # Load constellation lines
        constellations_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
            "../share/constellation_lines.csv")
        try:
            self.constellation_data = np.loadtxt(constellations_path, delimiter=",")
        except Exception as e:
            print("Could not load constellation lines from:", constellations_path)
            self.constellation_data = None

        ###################################################################################################
        # RIGHT WIDGET

        # If the file is being loaded, detect the input type
        if loaded_file:

            if not hasattr(self, "img_handle"):
                detect_input_type = True

            else:
                if self.img_handle is None:
                    detect_input_type = True


            if detect_input_type:

                if hasattr(self, "beginning_time"):
                    beginning_time = self.beginning_time
                else:
                    beginning_time = None

                # Detect data input type and init the image handle
                self.detectInputType(load=True, beginning_time=beginning_time)

                # If picks were made, change the frame to the first pick
                if len(self.pick_list):
                    self.img_handle.setFrame(min(self.pick_list.keys()))


        # adding img
        gamma = 1
        invert = False

        # Add saturation mask (R, G, B, alpha) - alpha can only be 0 or 1
        saturation_mask_img = np.zeros_like(self.img_handle.loadChunk().maxpixel).T
        self.saturation_mask_img = np.zeros(saturation_mask_img.shape + (4, ), dtype='uint8')
        self.saturation_mask = pg.ImageItem()
        self.saturation_mask.setImage(self.saturation_mask_img)
        self.saturation_mask.setZValue(1)
        self.img_frame.addItem(self.saturation_mask)

        # Add main image
        self.img_type_flag = 'avepixel'
        self.img = ImageItem(img_handle=self.img_handle, gamma=gamma, invert=invert,
                             saturation_mask=self.saturation_mask)
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

        self.region_zoom = pg.ImageItem(lut=lut)
        self.region_zoom.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)
        self.region_zoom.setZValue(10)
        self.zoom_window.addItem(self.region_zoom)

        # Mask overlay (red tint over masked areas)
        mask_lut = np.array([[0, 0, 0, 0], [255, 0, 0, 80]], dtype=np.ubyte)
        self.mask_overlay = pg.ImageItem(lut=mask_lut)
        self.mask_overlay.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)
        self.mask_overlay.setZValue(8)
        self.img_frame.addItem(self.mask_overlay)
        self.mask_overlay.hide()

        # Current polygon being drawn (yellow dashed line)
        self.mask_current_line = pg.PlotCurveItem(
            pen=pg.mkPen((255, 255, 0), width=2, style=QtCore.Qt.DashLine))
        self.mask_current_line.setZValue(15)
        self.img_frame.addItem(self.mask_current_line)

        # Vertex markers for current polygon (yellow)
        self.mask_vertex_markers = pg.ScatterPlotItem(
            pen=pg.mkPen((255, 255, 255), width=1),
            brush=pg.mkBrush(255, 255, 0, 200), size=10)
        self.mask_vertex_markers.setZValue(16)
        self.img_frame.addItem(self.mask_vertex_markers)

        # Vertex markers for completed polygons (red/white)
        self.mask_completed_vertex_markers = pg.ScatterPlotItem(
            pen=pg.mkPen((255, 255, 255), width=1),
            brush=pg.mkBrush(255, 100, 100, 200), size=8)
        self.mask_completed_vertex_markers.setZValue(14)
        self.img_frame.addItem(self.mask_completed_vertex_markers)

        # Storage for completed polygon graphics
        self.mask_polygon_items = []

        self.tab = RightOptionsTab(self)
        self.tab.hist.setImageItem(self.img)
        self.tab.hist.setImages(self.img_zoom)
        self.tab.hist.setLevels(0, 2**(8*self.img.data.itemsize) - 1)

        self.tab.settings.updateInvertColours()
        self.tab.settings.updateImageGamma()
        if loaded_file:
            self.updatePairedStars()

        # Make connections to sidebar gui
        #self.tab.param_manager.sigElevChanged.connect(self.onExtinctionChanged)
        self.tab.param_manager.sigLocationChanged.connect(self.onAzAltChanged)
        self.tab.param_manager.sigAzAltChanged.connect(self.onAzAltChanged)
        self.tab.param_manager.sigRotChanged.connect(self.onRotChanged)
        self.tab.param_manager.sigScaleChanged.connect(self.onScaleChanged)
        self.tab.param_manager.sigFitParametersChanged.connect(self.onFitParametersChanged)
        self.tab.param_manager.sigExtinctionChanged.connect(self.onExtinctionChanged)
        self.tab.param_manager.sigVignettingChanged.connect(self.onVignettingChanged)

        self.tab.param_manager.sigFitOnlyPointingToggled.connect(self.onFitParametersChanged)
        self.tab.param_manager.sigRefractionToggled.connect(self.onRefractionChanged)
        self.tab.param_manager.sigEqAspectToggled.connect(self.onFitParametersChanged)
        self.tab.param_manager.sigForceDistortionToggled.connect(self.onFitParametersChanged)
        self.tab.param_manager.sigOnVignettingFixedToggled.connect(self.onVignettingChanged)

        # Connect astrometry & photometry buttons to functions
        self.tab.param_manager.sigFitPressed.connect(self.fitPickedStars)
        self.tab.param_manager.sigAutoFitPressed.connect(self.autoFitAstrometryNet)
        self.tab.param_manager.sigFindBestFramePressed.connect(self.findBestFrame)
        self.tab.param_manager.sigNextStarPressed.connect(self.jumpNextStar)
        self.tab.param_manager.sigPhotometryPressed.connect(lambda: self.photometry(show_plot=True))
        self.tab.param_manager.sigAstrometryPressed.connect(self.showAstrometryFitPlots)
        self.tab.param_manager.sigResetDistortionPressed.connect(self.resetDistortion)

        self.tab.geolocation.sigLocationChanged.connect(self.onAzAltChanged)
        self.tab.geolocation.sigReloadGeoPoints.connect(self.reloadGeoPoints)
        self.tab.geolocation.sigFitPressed.connect(self.fitPickedStars)

        # Star detection override signals
        self.tab.star_detection.sigRedetectStars.connect(self.redetectStars)
        self.tab.star_detection.sigRedetectAllImages.connect(self.redetectAllImages)
        self.tab.star_detection.sigUseOverrideToggled.connect(self.toggleStarDetectionOverride)
        self.tab.star_detection.sigIntensityThresholdChanged.connect(self.updateIntensityThreshold)
        self.tab.star_detection.sigNeighborhoodSizeChanged.connect(self.updateNeighborhoodSize)
        self.tab.star_detection.sigMaxStarsChanged.connect(self.updateMaxStars)
        self.tab.star_detection.sigGammaChanged.connect(self.updateGamma)
        self.tab.star_detection.sigSegmentRadiusChanged.connect(self.updateSegmentRadius)
        self.tab.star_detection.sigMaxFeatureRatioChanged.connect(self.updateMaxFeatureRatio)
        self.tab.star_detection.sigRoundnessThresholdChanged.connect(self.updateRoundnessThreshold)

        # Mask widget signals
        self.tab.mask.sigDrawModeToggled.connect(self.toggleMaskDrawMode)
        self.tab.mask.sigClearPolygons.connect(self.clearMaskPolygons)
        self.tab.mask.sigSaveMask.connect(self.saveMask)
        self.tab.mask.sigLoadMask.connect(self.loadMaskDialog)
        self.tab.mask.sigShowOverlayToggled.connect(self.toggleMaskOverlay)
        self.tab.mask.sigUseFlatToggled.connect(self.toggleMaskFlatBackground)

        # Check for flat.bmp and setup mask tab
        self.checkAndSetupFlatForMask()

        # Handle tab changes (for restoring image when leaving mask tab)
        self.tab.sigTabChanged.connect(self.onTabChanged)

        self.tab.settings.sigMaxAveToggled.connect(self.toggleImageType)
        self.tab.settings.sigCatStarsToggled.connect(self.toggleShowCatStars)
        self.tab.settings.sigSpectralTypeToggled.connect(self.toggleShowSpectralType)
        self.tab.settings.sigStarNamesToggled.connect(self.toggleShowStarNames)
        self.tab.settings.sigConstellationToggled.connect(self.toggleShowConstellations)
        self.tab.settings.sigCalStarsToggled.connect(self.toggleShowCalStars)
        self.tab.settings.sigSelStarsToggled.connect(self.toggleShowSelectedStars)
        self.tab.settings.sigPicksToggled.connect(self.toggleShowPicks)
        self.tab.settings.sigGreatCircleToggled.connect(self.toggleShowGreatCircle)
        self.tab.settings.sigRegionToggled.connect(self.toggleShowRegion)
        self.tab.settings.sigDistortionToggled.connect(self.toggleDistortion)
        self.tab.settings.sigMeasGroundPointsToggled.connect(self.toggleMeasGroundPoints)
        self.tab.settings.sigGridToggled.connect(self.onGridChanged)
        self.tab.settings.sigInvertToggled.connect(self.toggleInvertColours)
        self.tab.settings.sigAutoPanToggled.connect(self.toggleAutoPan)
        self.tab.settings.sigSingleClickPhotometryToggled.connect(self.toggleSingleClickPhotometry)
        self.tab.settings.sigSatTracksToggled.connect(self.toggleShowSatTracks)
        self.tab.settings.sigLoadTLEPressed.connect(self.loadTLEFileDialog)
        self.tab.settings.sigClearTLEPressed.connect(self.clearTLESelection)
        self.tab.settings.sigRedrawSatTracksPressed.connect(self.redrawSatelliteTracks)

        layout.addWidget(self.tab, 0, 2)

        ###################################################################################################
        # SETUP

        # mouse binding
        self.img_frame.scene().sigMouseMoved.connect(self.onMouseMoved)
        self.img_frame.sigMouseReleased.connect(self.onMouseReleased)
        self.img_frame.sigMousePressed.connect(self.onMousePressed)
        self.img_frame.sigResized.connect(self.onFrameResize)

        self.setMinimumSize(1200, 800)

        # Size window to 80% of screen dimensions to avoid resize-to-fullscreen issue
        # while still working on any screen size
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.8), int(screen.height() * 0.8))

        self.show()

        self.updateLeftLabels()
        self.updateImageNavigationDisplay()
        self.updateStars()
        self.updateDistortion()
        self.tab.param_manager.updatePlatepar()
        self.initStarDetectionOverrides()
        self.initMaskFromFile()
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
            self.great_circle_line.hide()
            self.region.hide()
            if hasattr(self, "region_zoom"):
                self.region_zoom.hide()
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

            # Update TLE label
            tle_text = "latest downloaded"
            if self.tle_file:
                tle_text = os.path.basename(self.tle_file)
            self.tab.settings.updateTLELabel(tle_text)

            self.star_pick_info.setText(self.star_pick_info_text_str)

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
            #self.star_pick_info.setText(self.star_pick_info_text_str)

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
            self.great_circle_line.show()
            self.region.show()
            if hasattr(self, "region_zoom"):
                self.region_zoom.show()

            # Update the great circle
            self.updateGreatCircle()


    def changeStation(self):
        """
        Opens folder explorer window for user to select new station folder, then loads a platepar from that
        folder, and reads the config file, updating the gui to show what it should
        """
        dir_path = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select new station folder",
                                                              self.dir_path, QtWidgets.QFileDialog.ShowDirsOnly))

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

        self.paired_stars = PairedStars()
        self.updatePairedStars()
        self.pick_list = {}
        self.residuals = None
        self.updateFitResiduals()
        self.updatePicks()
        self.drawPhotometryColoring()
        self.photometry()

        self.updateLeftLabels()
        self.updateImageNavigationDisplay()
        self.tab.debruijn.updateTable()

    def onRefractionChanged(self):
        
        # Update the reference apparent alt/az, as the refraction influences the pointing
        self.platepar.updateRefAltAz()

        self.updateMeasurementRefractionCorrection()

        self.updateStars()
        self.updateLeftLabels()
        self.tab.param_manager.updatePlatepar()


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

    def onVignettingChanged(self):
        self.photometry()
        self.updateLeftLabels()

    def onAzAltChanged(self):
        self.platepar.updateRefRADec(preserve_rotation=True)
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

        self.star_pick_info.setPos(0, self.img_frame.height() - 50)

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
            x: [float] Plot X coordinate.
            y: [float] Plot Y coordinate.

        Return:
            [str]: formatted output string to be written in the status bar
        """

        # Write image X, Y coordinates and image intensity
        if 0 <= x <= self.img.data.shape[0] - 1 and 0 <= y <= self.img.data.shape[1] - 1:
            status_str = "x={:7.2f}  y={:7.2f}  Intens={:5d}".format(x, y, self.img.data[int(x), int(y)])
        else:
            status_str = "x={:7.2f}  y={:7.2f}  Intens=--".format(x, y)

        # Add coordinate info if platepar is present
        if self.platepar is not None:

            # Get the current frame time
            time_data = [self.img.img_handle.currentTime()]

            # Use a modified platepar if ground points are being picked
            pp_tmp = copy.deepcopy(self.platepar)
            if self.meas_ground_points:
                pp_tmp.switchToGroundPicks()

            # Compute RA, dec
            jd, ra, dec, _ = xyToRaDecPP(time_data, [x], [y], [1], pp_tmp, extinction_correction=False)

            # Compute alt, az
            azim, alt = trueRaDec2ApparentAltAz(ra[0], dec[0], jd[0], pp_tmp.lat, pp_tmp.lon, \
                                                pp_tmp.refraction)


            # If ground points are measured, change the text for alt/az
            if self.meas_ground_points:
                status_str += ",  Azim={:6.2f} Alt={:6.2f} (GROUND)".format(azim, alt)
            else:
                status_str += ",  Azim={:6.2f} Alt={:6.2f} (date)".format(azim, alt)


            # Add RA/Dec info
            status_str += ", RA={:6.2f} Dec={:+6.2f} (J2000)".format(ra[0], dec[0])

            # Show mode for debugging purposes

            if self.star_pick_mode and not self.autopan_mode:
                pass
            elif self.star_pick_mode and self.autopan_mode:
                status_str += ", Auto pan"

            if self.max_pixels_between_matched_stars != np.inf:
                percentage_complete = min([100,100*(len(self.paired_stars)+len(self.unsuitable_stars))/
                                                                len(self.catalog_x_filtered)])

                if self.max_pixels_between_matched_stars != 0:
                    status_str += ", max gap {:.0f}px".format(self.max_pixels_between_matched_stars)

                status_str += " good:{} bad:{} progress {:.0f}%".format(
                    len(self.paired_stars), len(self.unsuitable_stars), percentage_complete)

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

        # Sky fit
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
            text_str += u'Dec centre = {:.3f}\N{DEGREE SIGN}\n'.format(dec_centre)
            text_str += 'FOV = {:.2f}x{:.2f}\N{DEGREE SIGN}'.format(*computeFOVSize(self.platepar))

        # Manual reduction
        else:
            text_str = "Station: {:s} \n".format(self.platepar.station_code)
            text_str += self.img_handle.name() + '\n\n'
            text_str += self.img_type_flag + '\n'
            text_str += "Time  = {:s}\n".format(
                self.img_handle.currentFrameTime(dt_obj=True).strftime("%Y/%m/%d %H:%M:%S.%f")[:-3])
            text_str += 'Frame = {:d}\n'.format(self.img.getFrame())
            if self.img_handle.input_type == "ff":
                if self.use_fr_files:
                    text_str += 'Line = {:d}\n'.format(self.img_handle.current_line)
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
            if self.show_sattracks:
                text_str += 'CTRL + T - Toggle satellite tracks\n'
            text_str += 'CTRL + I - Show/hide distortion\n'
            text_str += 'U/J - Img Gamma\n'
            text_str += 'I - Invert colors\n'
            text_str += 'V - FOV centre\n'
            text_str += '\n'
            text_str += 'CTRL + A - Auto levels\n'
            text_str += 'CTRL + D - Load dark\n'
            text_str += 'CTRL + F - Load flat\n'
            text_str += 'CTRL + G - Cycle grids\n'
            text_str += 'CTRL + U - Pan to next\n'
            text_str += 'CTRL + O - Toggle auto pan\n'
            text_str += 'CTRL + X - astrometry.net img upload\n'
            text_str += 'CTRL + SHIFT + X - astrometry.net XY only\n'
            text_str += 'SHIFT + Q - Quick align test (debug)\n'
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
            text_str += 'ALT/Num0 + Left click - Mark gap (DFN)\n'
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
            text_str += 'CTRL + S - Save FTPdetectinfo\n'
            text_str += '\n'
            text_str += 'CTRL + K - Open ASTRA GUI'

        self.label2.setText(text_str)
        self.label2.setPos(self.img_frame.width() - self.label2.boundingRect().width(), \
            self.img_frame.height() - self.label2.boundingRect().height())


        self.label_f1.setText("F1 - Show hotkeys")
        self.label_f1.setPos(self.img_frame.width() - self.label_f1.boundingRect().width(), \
            self.img_frame.height() - self.label_f1.boundingRect().height())

        # Update satellite marker position
        self.updateSatelliteMarker()



    def updateStars(self, only_update_catalog=False):
        """ Updates only the stars, including catalog stars, calstars and paired stars.
         
        Keyword arguments:
            only_update_catalog: [bool] If True, only the catalog stars will be updated. (default: False)

        """

        if not only_update_catalog:

            # Draw stars that were paired in picking mode
            self.updatePairedStars()
            self.onGridChanged()  # for ease of use

            # Draw stars detected on this image
            if self.draw_calstars:
                self.updateCalstars()

            # Update constellation lines
            if self.show_constellations:
                self.updateConstellations()
            elif self.constellation_lines_bg.isVisible():
                self.constellation_lines_bg.hide()
                self.constellation_lines_fg.hide()


        # If in skyfit mode, take the time of the chunk
        # If in manual reduction mode, take the time of the current frame
        if self.mode == 'skyfit':
            ff_jd = date2JD(*self.img_handle.currentTime())
        else:
            ff_jd = date2JD(*self.img_handle.currentFrameTime())

        # Update the geo points
        if self.geo_points_obj is not None:

            # Compute RA/Dec of geo points
            self.geo_points_obj.update(self.platepar, ff_jd)

            # RA, dec, and fake magnitude of geo points
            self.geo_points = np.c_[self.geo_points_obj.ra_data, self.geo_points_obj.dec_data, \
                np.ones_like(self.geo_points_obj.ra_data)]

            # Compute image coordinates of geo points (always without refraction)
            pp_noref = copy.deepcopy(self.platepar)
            pp_noref.refraction = False
            pp_noref.updateRefRADec(preserve_rotation=True)
            self.geo_x, self.geo_y, _ = getCatalogStarsImagePositions(self.geo_points, ff_jd, pp_noref)

            geo_xy = np.c_[self.geo_x, self.geo_y]

            # Get indices of points inside the fov
            filtered_indices, _ = self.filterCatalogStarsInsideFOV(self.geo_points, \
                remove_under_horizon=False, sort_declination=True)

            # Create a mask to filter out all points outside the image and the FOV
            filter_indices_mask = np.zeros(len(geo_xy), dtype=bool)
            filter_indices_mask[filtered_indices] = True
            filtered_indices_all = filter_indices_mask & (geo_xy[:, 0] > 0) \
                                                    & (geo_xy[:, 0] < self.platepar.X_res) \
                                                    & (geo_xy[:, 1] > 0) \
                                                    & (geo_xy[:, 1] < self.platepar.Y_res)

            self.geo_filtered_indices = filtered_indices_all


            # Hold a list of geo points (equatorial coordinates) which are visible inside the FOV (with a
            #   fake magnitude)
            geo_xy = geo_xy[self.geo_filtered_indices]
            self.geo_x_filtered, self.geo_y_filtered = geo_xy.T

            # Plot geo points
            if self.catalog_stars_visible:
                geo_size = 5
                self.geo_markers.setData(x=self.geo_x_filtered + 0.5, y=self.geo_y_filtered + 0.5, \
                    size=geo_size)
                self.geo_markers2.setData(x=self.geo_x_filtered + 0.5, y=self.geo_y_filtered + 0.5, \
                    size=geo_size)


        ### Draw catalog stars on the image using the current platepar ###
        ######################################################################################################

        # Get positions of catalog stars on the image
        self.catalog_x, self.catalog_y, catalog_mag = getCatalogStarsImagePositions(self.catalog_stars, \
                                                                                    ff_jd, self.platepar)

        cat_stars_xy = np.c_[self.catalog_x, self.catalog_y, catalog_mag]

        ### Take only those stars inside the FOV  and image ###

        # Get indices of stars inside the fov
        filtered_indices, _ = self.filterCatalogStarsInsideFOV(self.catalog_stars)

        # Create a mask to filter out all stars outside the image and the FOV
        filter_indices_mask = np.zeros(len(cat_stars_xy), dtype=bool)
        filter_indices_mask[filtered_indices] = True
        filtered_indices_all = filter_indices_mask & (cat_stars_xy[:, 0] > 0) \
                                                & (cat_stars_xy[:, 0] < self.platepar.X_res) \
                                                & (cat_stars_xy[:, 1] > 0) \
                                                & (cat_stars_xy[:, 1] < self.platepar.Y_res)


        # Filter out catalog image stars
        cat_stars_xy_unmasked = cat_stars_xy[filtered_indices_all]

        # Create a filtered catalog
        self.catalog_stars_filtered_unmasked = self.catalog_stars[filtered_indices_all]

        # Filter spectral type
        if hasattr(self, 'catalog_stars_spectral_type') and (self.catalog_stars_spectral_type is not None):
            spectral_type_unmasked = self.catalog_stars_spectral_type[filtered_indices_all]
        else:
            spectral_type_unmasked = None

        # Filter star names
        if hasattr(self, 'catalog_stars_preferred_names') and (self.catalog_stars_preferred_names is not None):
            star_names_unmasked = self.catalog_stars_preferred_names[filtered_indices_all]
        else:
            star_names_unmasked = None

        # Filter common names
        if hasattr(self, 'catalog_stars_common_names') and (self.catalog_stars_common_names is not None):
            common_names_unmasked = self.catalog_stars_common_names[filtered_indices_all]
        else:
            common_names_unmasked = None


        if (self.mask is None) or (not hasattr(self.mask, 'img')):
            cat_stars_xy, self.catalog_stars_filtered = [], []
            self.catalog_stars_spectral_type_filtered = []
            self.catalog_stars_preferred_names_filtered = []
            self.catalog_stars_common_names_filtered = []

            # Prepare iterators
            if spectral_type_unmasked is None:
                spectral_type_unmasked = [None]*len(cat_stars_xy_unmasked)

            if star_names_unmasked is None:
                star_names_unmasked = [None]*len(cat_stars_xy_unmasked)

            if common_names_unmasked is None:
                common_names_unmasked = [None]*len(cat_stars_xy_unmasked)

            iterator = zip(cat_stars_xy_unmasked, self.catalog_stars_filtered_unmasked, spectral_type_unmasked, star_names_unmasked, common_names_unmasked)

            for star_xy, star_radec, star_spec, star_name, common_name in iterator:
                cat_stars_xy.append(star_xy)
                self.catalog_stars_filtered.append(star_radec)
                self.catalog_stars_spectral_type_filtered.append(star_spec)
                self.catalog_stars_preferred_names_filtered.append(star_name)
                self.catalog_stars_common_names_filtered.append(common_name)

        else:

            cat_stars_xy, self.catalog_stars_filtered = [], []
            self.catalog_stars_spectral_type_filtered = []
            self.catalog_stars_preferred_names_filtered = []
            self.catalog_stars_common_names_filtered = []

            # Prepare iterators
            if spectral_type_unmasked is None:
                spectral_type_unmasked = [None]*len(cat_stars_xy_unmasked)

            if star_names_unmasked is None:
                star_names_unmasked = [None]*len(cat_stars_xy_unmasked)

            if common_names_unmasked is None:
                common_names_unmasked = [None]*len(cat_stars_xy_unmasked)

            iterator = zip(cat_stars_xy_unmasked, self.catalog_stars_filtered_unmasked, spectral_type_unmasked, star_names_unmasked, common_names_unmasked)


            for star_xy, star_radec, star_spec, star_name, common_name in iterator:

                # Make sure that the dimensions of the mask match the image dimensions
                if (self.mask.img.shape[0] == self.img.data.shape[0]) or \
                   (self.mask.img.shape[1] == self.img.data.shape[1]):

                    # Check if the star is inside the mask
                    if self.mask.img[int(star_xy[1]), int(star_xy[0])] != 0:
                        cat_stars_xy.append(star_xy)
                        self.catalog_stars_filtered.append(star_radec)
                        self.catalog_stars_spectral_type_filtered.append(star_spec)
                        self.catalog_stars_preferred_names_filtered.append(star_name)
                        self.catalog_stars_common_names_filtered.append(common_name)

                # If the mask dimensions don't match the image dimensions, ignore the mask
                else:
                    cat_stars_xy.append(star_xy)
                    self.catalog_stars_filtered.append(star_radec)
                    self.catalog_stars_spectral_type_filtered.append(star_spec)
                    self.catalog_stars_preferred_names_filtered.append(star_name)
                    self.catalog_stars_common_names_filtered.append(common_name)

        # Convert to an array in any case
        cat_stars_xy = np.array(cat_stars_xy)

        # Create a list of filtered catalog image coordinates
        if len(cat_stars_xy):
            self.catalog_x_filtered, self.catalog_y_filtered, catalog_mag_filtered = cat_stars_xy.T
        else:
            self.catalog_x_filtered, self.catalog_y_filtered, catalog_mag_filtered = [], [], []

        # Show stars on the image
        if self.catalog_stars_visible:

            # Only show if there are any stars to show
            if len(catalog_mag_filtered):

                cat_mag_faintest = np.max(catalog_mag_filtered)

                # Plot catalog stars
                self.catalog_marker_size = ((4.0 + (cat_mag_faintest - catalog_mag_filtered))/2.0)**(2*2.512*0.5)

                self.cat_star_markers.setData(x=self.catalog_x_filtered + 0.5, \
                    y=self.catalog_y_filtered + 0.5, size=self.catalog_marker_size)
                self.cat_star_markers2.setData(x=self.catalog_x_filtered + 0.5, \
                    y=self.catalog_y_filtered + 0.5, size=self.catalog_marker_size)
                
                # Plot spectral type text
                self.spectral_type_text_list.clear()
                
                # Check if we should render anything
                if (self.show_spectral_type or self.show_star_names) and \
                   hasattr(self, 'catalog_stars_spectral_type_filtered') and \
                   (self.catalog_stars_spectral_type_filtered is not None):
                    
                    
                    # Prepare iterators
                    # We need to handle cases where names might not exist (optional field)
                    if hasattr(self, 'catalog_stars_preferred_names_filtered') and \
                       (self.catalog_stars_preferred_names_filtered is not None):
                        iter_names = self.catalog_stars_preferred_names_filtered
                    else:
                        iter_names = [None]*len(self.catalog_stars_spectral_type_filtered)

                    if hasattr(self, 'catalog_stars_common_names_filtered') and \
                       (self.catalog_stars_common_names_filtered is not None):
                        iter_common_names = self.catalog_stars_common_names_filtered
                    else:
                        iter_common_names = [None]*len(self.catalog_stars_spectral_type_filtered)


                    iterator = zip(self.catalog_stars_spectral_type_filtered, iter_names, iter_common_names)

                    for i, (spec_type, star_name, common_name) in enumerate(iterator):
                        
                        html_text = ""
                        has_text = False

                        # Add spectral type
                        if self.show_spectral_type and (spec_type is not None) and (len(spec_type) > 0):
                            
                            # Determine color based on spectral type
                            # Default light green
                            hex_color = "#90ee90" 
                            
                            first_char = spec_type[0].upper()
                            if 'infrared' in spec_type.lower():
                                hex_color = "#990000" # Red
                            elif first_char == 'O':
                                hex_color = "#9bb0ff" # Blue
                            elif first_char == 'B':
                                hex_color = "#aabfff" # Blue-white
                            elif first_char == 'A':
                                hex_color = "#cad7ff" # White-Blue
                            elif first_char == 'F':
                                hex_color = "#f8f7ff" # White
                            elif first_char == 'G':
                                hex_color = "#fff4ea" # Yellow-white
                            elif first_char == 'K':
                                hex_color = "#ffd2a1" # Orange
                            elif first_char == 'M':
                                hex_color = "#ffcc6f" # Red-orange


                            # Replace "infrared" with "IR"
                            if 'infrared' in spec_type.lower():
                                spec_type = "IR"
                            

                            html_text += f'<span style="color: {hex_color};">{spec_type}</span>'
                            has_text = True


                        # Add star name
                        if self.show_star_names and (star_name is not None) and (len(star_name) > 0):

                            if has_text:
                                html_text += "<br>"

                            # Use HD/catalog name for SIMBAD URL
                            url_name = star_name.replace(' ', '+')
                            # Use common name for display (e.g., "Sirius" instead of "HD 48915")
                            display_name = common_name if common_name else star_name
                            # Greyish color for name, formatted as link
                            html_text += f'<a href="https://simbad.cds.unistra.fr/simbad/sim-id?Ident={url_name}" style="color: #dddddd; text-decoration: none;">{display_name}</a>'
                            has_text = True


                        if not has_text:
                            continue

                        # Add text item with HTML
                        # Anchor (1.0, 0.5) places text right edge at anchor point
                        # Small offset clears the marker without drifting too much on zoom
                        text_item = TextItem(html=html_text, anchor=(1.0, 0.5))
                        text_item.setAlign(QtCore.Qt.AlignRight)
                        text_item.setPos(self.catalog_x_filtered[i] - 3,
                                         self.catalog_y_filtered[i] + 0.5)
                        self.spectral_type_text_list.addTextItem(text_item)
            else:
                print('No catalog stars visible!')


    def updatePairedStars(self):
        """ Draws the stars that were picked for calibration as well as draw the residuals and star magnitude.
        """
        if len(self.paired_stars) > 0:
            self.sel_cat_star_markers.setData(pos=self.paired_stars.imageCoords(draw=True))
            self.sel_cat_star_markers2.setData(pos=self.paired_stars.imageCoords(draw=True))

        else:
            self.sel_cat_star_markers.setData(pos=[])
            self.sel_cat_star_markers2.setData(pos=[])

        if len(self.unsuitable_stars) > 0:
            self.unsuitable_star_markers.setData(pos=self.unsuitable_stars.imageCoords(draw=True))
            self.unsuitable_star_markers2.setData(pos=self.unsuitable_stars.imageCoords(draw=True))
        else:
            self.unsuitable_star_markers.setData(pos=[])
            self.unsuitable_star_markers2.setData(pos=[])

        self.centroid_star_markers.setData(pos=[])
        self.centroid_star_markers2.setData(pos=[])

        # Draw photometry
        if len(self.paired_stars) >= 2:
            self.photometry()

        self.tab.param_manager.updatePairedStars(min_fit_stars=self.getMinFitStars())


    def updateCalstars(self):
        """ Draw extracted stars on the current image. """

        # Handle using FR files
        ff_name_c = convertFRNameToFF(self.img_handle.name())

        # Choose data source: override detections or original CALSTARS
        star_data = None
        if self.star_detection_override_enabled and ff_name_c in self.star_detection_override_data:
            star_data = np.array(self.star_detection_override_data[ff_name_c])
        elif ff_name_c in self.calstars:
            star_data = np.array(self.calstars[ff_name_c])

        if star_data is not None and len(star_data) > 0:
            # Get star coordinates
            y = star_data[:, 0]
            x = star_data[:, 1]

            # Set both the inner and outer rings
            self.calstar_markers.setData(x=x + 0.5, y=y + 0.5)
            self.calstar_markers2.setData(x=x + 0.5, y=y + 0.5)
            self.calstar_markers_outer.setData(x=x + 0.5, y=y + 0.5)
            self.calstar_markers_outer2.setData(x=x + 0.5, y=y + 0.5)

        else:
            # Clear all markers if no stars
            self.calstar_markers.setData(pos=[])
            self.calstar_markers2.setData(pos=[])
            self.calstar_markers_outer.setData(pos=[])
            self.calstar_markers_outer2.setData(pos=[])


    def initStarDetectionOverrides(self):
        """ Initialize star detection override parameters from config. """
        if hasattr(self.config, 'intensity_threshold'):
            self.override_intensity_threshold = self.config.intensity_threshold
        if hasattr(self.config, 'neighborhood_size'):
            self.override_neighborhood_size = self.config.neighborhood_size
        if hasattr(self.config, 'max_stars'):
            self.override_max_stars = self.config.max_stars
        if hasattr(self.config, 'gamma'):
            self.override_gamma = self.config.gamma
        if hasattr(self.config, 'segment_radius'):
            self.override_segment_radius = self.config.segment_radius
        if hasattr(self.config, 'max_feature_ratio'):
            self.override_max_feature_ratio = self.config.max_feature_ratio
        if hasattr(self.config, 'roundness_threshold'):
            self.override_roundness_threshold = self.config.roundness_threshold

        # Update UI sliders
        self.tab.star_detection.loadFromConfig(self.config)


    def updateIntensityThreshold(self, value):
        """ Update intensity threshold override parameter. """
        self.override_intensity_threshold = value

    def updateNeighborhoodSize(self, value):
        """ Update neighborhood size override parameter. """
        self.override_neighborhood_size = value

    def updateMaxStars(self, value):
        """ Update max stars override parameter. """
        self.override_max_stars = value

    def updateGamma(self, value):
        """ Update gamma override parameter. """
        self.override_gamma = value

        # If override is enabled, also sync config.gamma and platepar.gamma
        if self.star_detection_override_enabled:
            self.config.gamma = value
            if self.platepar is not None:
                self.platepar.gamma = value

        # Sync display gamma (Settings tab) with camera gamma
        self.img.setGamma(value)
        self.img_zoom.setGamma(value)
        # Block signals to prevent infinite loop
        self.tab.settings.img_gamma.blockSignals(True)
        self.tab.settings.img_gamma.setValue(value)
        self.tab.settings.img_gamma.blockSignals(False)
        self.updateLeftLabels()

    def updateSegmentRadius(self, value):
        """ Update segment radius override parameter. """
        self.override_segment_radius = value

    def updateMaxFeatureRatio(self, value):
        """ Update max feature ratio override parameter. """
        self.override_max_feature_ratio = value

    def updateRoundnessThreshold(self, value):
        """ Update roundness threshold override parameter. """
        self.override_roundness_threshold = value


    def toggleStarDetectionOverride(self):
        """ Toggle between using override detections and original CALSTARS. """
        self.star_detection_override_enabled = not self.star_detection_override_enabled

        # Update both platepar.gamma and config.gamma to match the detection source
        if self.star_detection_override_enabled:
            # Store original config gamma before overriding
            if self._original_config_gamma is None:
                self._original_config_gamma = getattr(self.config, 'gamma', 1.0)

            # Using override detections - use override gamma everywhere
            if self.platepar is not None:
                self.platepar.gamma = self.override_gamma
            self.config.gamma = self.override_gamma
            print(f"Switched to override detections (gamma={self.override_gamma:.3f})")
        else:
            # Using original CALSTARS - restore original gamma
            if self._original_config_gamma is not None:
                if self.platepar is not None:
                    self.platepar.gamma = self._original_config_gamma
                self.config.gamma = self._original_config_gamma
                print(f"Switched to original CALSTARS (gamma={self._original_config_gamma:.3f})")

        # Update the display
        self.updateCalstars()

        # Update status in UI
        ff_name = self.img_handle.name()
        if self.star_detection_override_enabled and ff_name in self.star_detection_override_data:
            star_count = len(self.star_detection_override_data[ff_name])
            self.tab.star_detection.updateStatus(True, star_count)
        else:
            self.tab.star_detection.updateStatus(False)


    def redetectStars(self):
        """ Re-detect stars on current image using override parameters. """
        print(f"Re-detecting stars with: threshold={self.override_intensity_threshold}, "
              f"neighborhood={self.override_neighborhood_size}, max_stars={self.override_max_stars}, "
              f"gamma={self.override_gamma:.3f}")

        # Get current FF file name
        ff_name = self.img_handle.name()

        # Call extractStarsFF with override parameters
        try:
            # Temporarily modify config to use override parameters
            original_intensity_threshold = getattr(self.config, 'intensity_threshold', 18)
            original_neighborhood_size = getattr(self.config, 'neighborhood_size', 10)
            original_max_stars = getattr(self.config, 'max_stars', 200)
            original_gamma = getattr(self.config, 'gamma', 1.0)
            original_segment_radius = getattr(self.config, 'segment_radius', 4)
            original_max_feature_ratio = getattr(self.config, 'max_feature_ratio', 0.8)
            original_roundness_threshold = getattr(self.config, 'roundness_threshold', 0.5)

            # Set override values
            self.config.intensity_threshold = self.override_intensity_threshold
            self.config.neighborhood_size = self.override_neighborhood_size
            self.config.max_stars = self.override_max_stars
            self.config.gamma = self.override_gamma
            self.config.segment_radius = self.override_segment_radius
            self.config.max_feature_ratio = self.override_max_feature_ratio
            self.config.roundness_threshold = self.override_roundness_threshold

            try:
                star_list = extractStarsFF(
                    self.dir_path,
                    ff_name,
                    config=self.config,
                    flat_struct=self.flat_struct if hasattr(self, 'flat_struct') else None,
                    dark=self.dark if hasattr(self, 'dark') else None,
                    mask=self.mask if hasattr(self, 'mask') else None
                )
            finally:
                # Restore original config values
                self.config.intensity_threshold = original_intensity_threshold
                self.config.neighborhood_size = original_neighborhood_size
                self.config.max_stars = original_max_stars
                self.config.gamma = original_gamma
                self.config.segment_radius = original_segment_radius
                self.config.max_feature_ratio = original_max_feature_ratio
                self.config.roundness_threshold = original_roundness_threshold

            if star_list:
                # extractStarsFF returns: ff_name, x_arr, y_arr, amplitude, intensity, fwhm, background, snr, saturated_count
                ff_name_ret, x_arr, y_arr, amplitude, intensity, fwhm, background, snr, saturated_count = star_list

                # Construct star data in CALSTARS format: Y(0) X(1) IntensSum(2) Ampltd(3) FWHM(4) BgLvl(5) SNR(6) NSatPx(7)
                # Note: intensity=IntensSum (integrated), amplitude=Ampltd (peak)
                star_data = list(zip(y_arr, x_arr, intensity, amplitude, fwhm, background, snr, saturated_count))

                # Store the override detected stars
                self.star_detection_override_data[ff_name] = star_data
                original_count = len(self.calstars.get(ff_name, []))
                print(f"  Detected {len(star_data)} stars (original: {original_count})")

                # Enable override mode and update display
                self.star_detection_override_enabled = True
                self.tab.star_detection.use_override_checkbox.setChecked(True)

                # Sync gamma: keep override gamma in config and platepar
                if self._original_config_gamma is None:
                    self._original_config_gamma = original_gamma
                self.config.gamma = self.override_gamma
                if self.platepar is not None:
                    self.platepar.gamma = self.override_gamma
                print(f"  Using override gamma={self.override_gamma:.3f} for photometry")

                self.updateCalstars()
                self.tab.star_detection.updateStatus(True, len(star_data))

            else:
                print("  No stars detected")
                self.tab.star_detection.updateStatus(False)

        except Exception as e:
            print(f"  Error during star detection: {e}")
            traceback.print_exc()


    def redetectAllImages(self):
        """ Re-detect stars on all images using override parameters. """
        print("Re-detecting stars on all images...")

        # Get list of all FF files
        ff_files = list(self.calstars.keys())
        if not ff_files:
            print("  No FF files found in CALSTARS")
            return

        total = len(ff_files)
        success_count = 0

        # Save original config values
        original_intensity_threshold = getattr(self.config, 'intensity_threshold', 18)
        original_neighborhood_size = getattr(self.config, 'neighborhood_size', 10)
        original_max_stars = getattr(self.config, 'max_stars', 200)
        original_gamma = getattr(self.config, 'gamma', 1.0)
        original_segment_radius = getattr(self.config, 'segment_radius', 4)
        original_max_feature_ratio = getattr(self.config, 'max_feature_ratio', 0.8)
        original_roundness_threshold = getattr(self.config, 'roundness_threshold', 0.5)

        # Set override values
        self.config.intensity_threshold = self.override_intensity_threshold
        self.config.neighborhood_size = self.override_neighborhood_size
        self.config.max_stars = self.override_max_stars
        self.config.gamma = self.override_gamma
        self.config.segment_radius = self.override_segment_radius
        self.config.max_feature_ratio = self.override_max_feature_ratio
        self.config.roundness_threshold = self.override_roundness_threshold

        try:
            for i, ff_name in enumerate(ff_files):
                print(f"  Processing {i+1}/{total}: {ff_name}")

                try:
                    star_list = extractStarsFF(
                        self.dir_path,
                        ff_name,
                        config=self.config,
                        flat_struct=self.flat_struct if hasattr(self, 'flat_struct') else None,
                        dark=self.dark if hasattr(self, 'dark') else None,
                        mask=self.mask if hasattr(self, 'mask') else None
                    )

                    if star_list:
                        ff_name_ret, x_arr, y_arr, amplitude, intensity, fwhm, background, snr, saturated_count = star_list
                        # CALSTARS format: Y(0) X(1) IntensSum(2) Ampltd(3) FWHM(4) BgLvl(5) SNR(6) NSatPx(7)
                        star_data = list(zip(y_arr, x_arr, intensity, amplitude, fwhm, background, snr, saturated_count))
                        self.star_detection_override_data[ff_name] = star_data
                        success_count += 1

                except Exception as e:
                    print(f"    Error: {e}")

        finally:
            # Restore original config values
            self.config.intensity_threshold = original_intensity_threshold
            self.config.neighborhood_size = original_neighborhood_size
            self.config.max_stars = original_max_stars
            self.config.gamma = original_gamma
            self.config.segment_radius = original_segment_radius
            self.config.max_feature_ratio = original_max_feature_ratio
            self.config.roundness_threshold = original_roundness_threshold

        print(f"  Completed: {success_count}/{total} images processed")

        # Enable override mode and update display
        self.star_detection_override_enabled = True
        self.tab.star_detection.use_override_checkbox.setChecked(True)

        # Sync gamma: keep override gamma in config and platepar
        if self._original_config_gamma is None:
            self._original_config_gamma = original_gamma
        self.config.gamma = self.override_gamma
        if self.platepar is not None:
            self.platepar.gamma = self.override_gamma
        print(f"  Using override gamma={self.override_gamma:.3f} for photometry")

        self.updateCalstars()

        ff_name = self.img_handle.name()
        if ff_name in self.star_detection_override_data:
            self.tab.star_detection.updateStatus(True, len(self.star_detection_override_data[ff_name]))
        else:
            self.tab.star_detection.updateStatus(True)


    ###################################################################################################
    # MASK DRAWING METHODS
    ###################################################################################################

    def initMaskFromFile(self):
        """Auto-load mask.bmp if it exists in the working directory."""
        mask_path = os.path.join(self.dir_path, "mask.bmp")
        if os.path.exists(mask_path):
            self.loadMaskFromFile(mask_path)

    def toggleMaskDrawMode(self):
        """Toggle mask polygon drawing mode."""
        self.mask_draw_mode = self.tab.mask.draw_button.isChecked()
        if self.mask_draw_mode:
            self.mask_current_polygon = []
        else:
            # If there are points, close the polygon
            if len(self.mask_current_polygon) >= 3:
                self.mask_polygons.append(self.mask_current_polygon.copy())
                self.tab.mask.setUnsaved(True)
            self.mask_current_polygon = []
            self.tab.mask.setDrawMode(False)
        self.updateMaskDisplay()
        self.tab.mask.updateStatus(len(self.mask_polygons))

    def addMaskPoint(self, x, y):
        """Add a point to the current polygon being drawn."""
        # Get image dimensions - shape[0] is X, shape[1] is Y in this codebase
        max_x = self.img.data.shape[0] - 1
        max_y = self.img.data.shape[1] - 1

        # Snap to edges if within threshold or outside bounds
        snap_threshold = 15
        if x < snap_threshold:
            x = 0
        elif x > max_x - snap_threshold:
            x = max_x
        if y < snap_threshold:
            y = 0
        elif y > max_y - snap_threshold:
            y = max_y

        # Clamp to valid bounds
        x = np.clip(x, 0, max_x)
        y = np.clip(y, 0, max_y)

        self.mask_current_polygon.append((x, y))
        self.updateMaskDisplay()
        self.tab.mask.updateStatus(len(self.mask_polygons), len(self.mask_current_polygon))

    def closeMaskPolygon(self):
        """Close the current polygon and add it to the list."""
        if len(self.mask_current_polygon) >= 3:
            self.mask_polygons.append(self.mask_current_polygon.copy())
            self.tab.mask.setUnsaved(True)
        self.mask_current_polygon = []
        self.mask_draw_mode = False
        self.tab.mask.setDrawMode(False)
        self.updateMaskDisplay()
        self.tab.mask.updateStatus(len(self.mask_polygons))

    def clearMaskPolygons(self):
        """Clear all mask polygons."""
        self.mask_polygons = []
        self.mask_current_polygon = []
        self.mask_draw_mode = False
        self.mask_dragging_vertex = None
        self.tab.mask.setDrawMode(False)
        self.tab.mask.setUnsaved(True)
        self.updateMaskDisplay()
        self.tab.mask.updateStatus(0)

    def findNearestMaskVertex(self, x, y, threshold=15):
        """Find the nearest vertex to (x, y) within threshold.
        Returns ('current', idx) for current polygon or (poly_idx, vert_idx) for completed polygons.
        """
        min_dist = threshold
        result = None

        # Check current polygon vertices
        for i, (vx, vy) in enumerate(self.mask_current_polygon):
            dist = np.hypot(x - vx, y - vy)
            if dist < min_dist:
                min_dist = dist
                result = ('current', i)

        # Check completed polygon vertices
        for poly_idx, polygon in enumerate(self.mask_polygons):
            for vert_idx, (vx, vy) in enumerate(polygon):
                dist = np.hypot(x - vx, y - vy)
                if dist < min_dist:
                    min_dist = dist
                    result = (poly_idx, vert_idx)

        return result

    def findNearestMaskEdge(self, x, y, threshold=15):
        """Find the nearest edge to (x, y) within threshold.
        Returns (poly_idx, insert_idx) where insert_idx is the index to insert a new vertex,
        or ('current', insert_idx) for the current polygon being drawn.
        """
        min_dist = threshold
        result = None

        def point_to_segment_dist(px, py, x1, y1, x2, y2):
            """Calculate distance from point (px, py) to line segment (x1,y1)-(x2,y2)."""
            # Vector from segment start to point
            dx, dy = px - x1, py - y1
            # Segment vector
            sx, sy = x2 - x1, y2 - y1
            # Segment length squared
            seg_len_sq = sx*sx + sy*sy
            if seg_len_sq == 0:
                return np.hypot(dx, dy)  # Degenerate segment
            # Parameter t for projection onto segment (clamped to [0,1])
            t = max(0, min(1, (dx*sx + dy*sy) / seg_len_sq))
            # Closest point on segment
            closest_x = x1 + t * sx
            closest_y = y1 + t * sy
            return np.hypot(px - closest_x, py - closest_y)

        # Check current polygon edges (if has at least 2 points)
        if len(self.mask_current_polygon) >= 2:
            for i in range(len(self.mask_current_polygon) - 1):
                x1, y1 = self.mask_current_polygon[i]
                x2, y2 = self.mask_current_polygon[i + 1]
                dist = point_to_segment_dist(x, y, x1, y1, x2, y2)
                if dist < min_dist:
                    min_dist = dist
                    result = ('current', i + 1)  # Insert after vertex i

        # Check completed polygon edges
        for poly_idx, polygon in enumerate(self.mask_polygons):
            if len(polygon) >= 2:
                for i in range(len(polygon)):
                    x1, y1 = polygon[i]
                    x2, y2 = polygon[(i + 1) % len(polygon)]  # Wrap around for closed polygon
                    dist = point_to_segment_dist(x, y, x1, y1, x2, y2)
                    if dist < min_dist:
                        min_dist = dist
                        result = (poly_idx, i + 1)  # Insert after vertex i

        return result

    def insertMaskVertex(self, edge_ref, x, y):
        """Insert a new vertex at position (x, y) at the specified edge location."""
        if edge_ref[0] == 'current':
            insert_idx = edge_ref[1]
            self.mask_current_polygon.insert(insert_idx, (x, y))
            self.updateMaskDisplay()
            self.tab.mask.updateStatus(len(self.mask_polygons), len(self.mask_current_polygon))
        else:
            poly_idx, insert_idx = edge_ref
            if poly_idx < len(self.mask_polygons):
                self.mask_polygons[poly_idx].insert(insert_idx, (x, y))
                self.tab.mask.setUnsaved(True)
                self.updateMaskDisplay()
                self.tab.mask.updateStatus(len(self.mask_polygons))

    def deleteMaskVertex(self, vertex_ref):
        """Delete a vertex from a polygon."""
        if vertex_ref[0] == 'current':
            idx = vertex_ref[1]
            if len(self.mask_current_polygon) > 0:
                del self.mask_current_polygon[idx]
                self.updateMaskDisplay()
                self.tab.mask.updateStatus(len(self.mask_polygons), len(self.mask_current_polygon))
        else:
            poly_idx, vert_idx = vertex_ref
            if poly_idx < len(self.mask_polygons):
                polygon = self.mask_polygons[poly_idx]
                if len(polygon) > 3:
                    # Keep polygon if it still has at least 3 vertices
                    del polygon[vert_idx]
                    self.tab.mask.setUnsaved(True)
                    self.updateMaskDisplay()
                    self.tab.mask.updateStatus(len(self.mask_polygons))
                else:
                    # Delete entire polygon if less than 3 vertices would remain
                    del self.mask_polygons[poly_idx]
                    self.tab.mask.setUnsaved(True)
                    self.updateMaskDisplay()
                    self.tab.mask.updateStatus(len(self.mask_polygons))

    def moveMaskVertex(self, vertex_ref, new_x, new_y):
        """Move a vertex to new position with edge snapping."""
        # Apply edge snapping
        max_x = self.img.data.shape[0] - 1
        max_y = self.img.data.shape[1] - 1
        snap_threshold = 15

        if new_x < snap_threshold:
            new_x = 0
        elif new_x > max_x - snap_threshold:
            new_x = max_x
        if new_y < snap_threshold:
            new_y = 0
        elif new_y > max_y - snap_threshold:
            new_y = max_y

        new_x = np.clip(new_x, 0, max_x)
        new_y = np.clip(new_y, 0, max_y)

        if vertex_ref[0] == 'current':
            idx = vertex_ref[1]
            if idx < len(self.mask_current_polygon):
                self.mask_current_polygon[idx] = (new_x, new_y)
        else:
            poly_idx, vert_idx = vertex_ref
            if poly_idx < len(self.mask_polygons) and vert_idx < len(self.mask_polygons[poly_idx]):
                self.mask_polygons[poly_idx][vert_idx] = (new_x, new_y)
                self.tab.mask.setUnsaved(True)

        self.updateMaskDisplay()
        self.tab.mask.updateStatus(len(self.mask_polygons))

    def updateMaskDisplay(self):
        """Update all mask graphics items."""
        # Update current polygon line
        if len(self.mask_current_polygon) > 0:
            pts = np.array(self.mask_current_polygon)
            # Close the polygon visually
            if len(pts) >= 3:
                pts_closed = np.vstack([pts, pts[0]])
                self.mask_current_line.setData(x=pts_closed[:, 0] + 0.5, y=pts_closed[:, 1] + 0.5)
            else:
                self.mask_current_line.setData(x=pts[:, 0] + 0.5, y=pts[:, 1] + 0.5)
            self.mask_vertex_markers.setData(pos=pts + 0.5)
        else:
            self.mask_current_line.setData(x=[], y=[])
            self.mask_vertex_markers.setData(pos=[])

        # Remove old polygon items
        for item in self.mask_polygon_items:
            self.img_frame.removeItem(item)
        self.mask_polygon_items = []

        # Draw completed polygons and collect vertices (outlines only, overlay handles fill)
        all_completed_vertices = []
        for polygon in self.mask_polygons:
            pts = np.array(polygon)
            pts_closed = np.vstack([pts, pts[0]])
            line = pg.PlotCurveItem(
                x=pts_closed[:, 0] + 0.5,
                y=pts_closed[:, 1] + 0.5,
                pen=pg.mkPen((255, 0, 0, 200), width=2))
            line.setZValue(12)
            self.img_frame.addItem(line)
            self.mask_polygon_items.append(line)
            # Collect vertices for markers
            all_completed_vertices.extend(polygon)

        # Update completed polygon vertex markers
        if len(all_completed_vertices) > 0:
            verts = np.array(all_completed_vertices)
            self.mask_completed_vertex_markers.setData(pos=verts + 0.5)
        else:
            self.mask_completed_vertex_markers.setData(pos=[])

        # Update mask overlay
        self.updateMaskOverlayImage()

    def updateMaskOverlayImage(self):
        """Update the mask overlay image based on polygons."""
        import cv2

        if not self.tab.mask.show_overlay.isChecked():
            self.mask_overlay.hide()
            return

        if len(self.mask_polygons) == 0:
            self.mask_overlay.hide()
            return

        # Get image dimensions - shape[0] is X, shape[1] is Y in this codebase
        img_width = self.img.data.shape[0]
        img_height = self.img.data.shape[1]

        # Create mask image (0 = clear, 1 = masked)
        mask_img = np.zeros((img_height, img_width), dtype=np.uint8)

        for polygon in self.mask_polygons:
            pts = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(mask_img, [pts], 1)

        # Transpose for pyqtgraph display
        self.mask_overlay.setImage(mask_img.T)
        self.mask_overlay.show()

    def toggleMaskOverlay(self, visible):
        """Toggle mask overlay visibility (includes outlines and vertices)."""
        if visible:
            self.updateMaskOverlayImage()
            # Show polygon outlines
            for item in self.mask_polygon_items:
                item.show()
            # Show completed vertex markers
            self.mask_completed_vertex_markers.show()
        else:
            self.mask_overlay.hide()
            # Hide polygon outlines
            for item in self.mask_polygon_items:
                item.hide()
            # Hide completed vertex markers
            self.mask_completed_vertex_markers.hide()

    def loadFlatImage(self):
        """Load flat.bmp from data directory if it exists."""
        flat_path = os.path.join(self.dir_path, "flat.bmp")

        if os.path.isfile(flat_path):
            try:
                from RMS.Routines.Image import loadImage
                flat_img = loadImage(flat_path, flatten=0)

                # Convert to single channel if needed
                if len(flat_img.shape) > 2:
                    flat_img = flat_img[:, :, 0]

                self.flat_image_data = flat_img
                print(f"Loaded flat image: {flat_path}")
                return True
            except Exception as e:
                print(f"Failed to load flat image: {e}")
                self.flat_image_data = None
                return False
        else:
            self.flat_image_data = None
            return False

    def checkAndSetupFlatForMask(self):
        """Check for flat.bmp and setup mask tab accordingly."""
        flat_exists = self.loadFlatImage()

        # Block signal to prevent showing flat during initialization
        self.tab.mask.use_flat.blockSignals(True)
        self.tab.mask.setFlatAvailable(flat_exists, use_by_default=flat_exists)
        self.tab.mask.use_flat.blockSignals(False)

        if flat_exists:
            self.mask_use_flat_background = True

    def toggleMaskFlatBackground(self, use_flat):
        """Toggle between flat.bmp and current image as mask editing background."""
        self.mask_use_flat_background = use_flat

        # Only change the image if we're currently on the mask tab
        mask_tab_index = self.tab.indexOf(self.tab.mask)
        if self.tab.currentIndex() != mask_tab_index:
            return

        if use_flat and self.flat_image_data is not None:
            # Switch to flat image
            self.img.setImage(self.flat_image_data.T)
            print("Mask background: using flat.bmp")
            # Hide all stars when showing flat
            self.cat_star_markers.hide()
            self.cat_star_markers2.hide()
            self.geo_markers.hide()
            self.calstar_markers.hide()
            self.calstar_markers2.hide()
            self.calstar_markers_outer.hide()
            self.calstar_markers_outer2.hide()
        else:
            # Switch back to current image
            self.img.loadImage(self.mode, self.img_type_flag)
            print("Mask background: using current image")
            # Show stars when showing current image (if enabled)
            if self.catalog_stars_visible:
                self.cat_star_markers.show()
                self.cat_star_markers2.show()
                self.geo_markers.show()
            if self.draw_calstars:
                self.calstar_markers.show()
                self.calstar_markers2.show()
                self.calstar_markers_outer.show()
                self.calstar_markers_outer2.show()

    def onTabChanged(self, old_index, new_index):
        """Handle tab changes - restore image when leaving mask tab."""
        # Mask tab is at index 4 (Levels=0, Fit Parameters=1, Station=2, Star Detection=3, Mask=4)
        mask_tab_index = self.tab.indexOf(self.tab.mask)

        if old_index == mask_tab_index and self.mask_use_flat_background:
            # Leaving mask tab while flat was shown - restore current image
            self.img.loadImage(self.mode, self.img_type_flag)
            print("Restored current image (left mask tab)")

        elif new_index == mask_tab_index and self.mask_use_flat_background and self.flat_image_data is not None:
            # Entering mask tab with flat enabled - show flat
            self.img.setImage(self.flat_image_data.T)
            print("Showing flat.bmp for mask editing")

        # Handle mask tab visibility
        if new_index == mask_tab_index:
            self.img_frame.panning_enabled = False

            # Always hide picks, selected stars, residual lines, and astrometry.net markers on mask tab
            self.pick_marker.hide()
            self.pick_marker2.hide()
            self.sel_cat_star_markers.hide()
            self.sel_cat_star_markers2.hide()
            self.residual_lines_img.hide()
            self.residual_lines_astro.hide()
            self.astrometry_quad_markers.hide()
            self.astrometry_quad_markers2.hide()
            self.astrometry_matched_markers.hide()
            self.astrometry_matched_markers2.hide()
            # Clear error text (TextItemList children are parented to frame, not list)
            self.residual_text.clear()

            # Hide all stars when flat is displayed
            if self.mask_use_flat_background and self.flat_image_data is not None:
                self.cat_star_markers.hide()
                self.cat_star_markers2.hide()
                self.geo_markers.hide()
                self.calstar_markers.hide()
                self.calstar_markers2.hide()
                self.calstar_markers_outer.hide()
                self.calstar_markers_outer2.hide()

        elif old_index == mask_tab_index:
            # Leaving mask tab - restore visibility based on user settings
            self.img_frame.panning_enabled = True

            if self.catalog_stars_visible:
                self.cat_star_markers.show()
                self.cat_star_markers2.show()
                self.geo_markers.show()
            if self.selected_stars_visible:
                self.sel_cat_star_markers.show()
                self.sel_cat_star_markers2.show()
                self.residual_lines_img.show()
                self.residual_lines_astro.show()
            if self.draw_calstars:
                self.calstar_markers.show()
                self.calstar_markers2.show()
                self.calstar_markers_outer.show()
                self.calstar_markers_outer2.show()
            if self.astrometry_stars_visible:
                self.astrometry_quad_markers.show()
                self.astrometry_quad_markers2.show()
                self.astrometry_matched_markers.show()
                self.astrometry_matched_markers2.show()
            # Restore error text by re-running photometry
            self.photometry()
            # Picks only visible in manual reduction mode
            if self.mode == 'manualreduction':
                self.pick_marker.show()
                self.pick_marker2.show()
        else:
            self.img_frame.panning_enabled = True

    def generateMaskImage(self):
        """Generate mask.bmp image from polygons."""
        import cv2

        img_width = self.img.data.shape[0]
        img_height = self.img.data.shape[1]

        mask = np.full((img_height, img_width), 255, dtype=np.uint8)

        for polygon in self.mask_polygons:
            pts = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 0)

        return mask

    def saveMask(self):
        """Save mask to file and update self.mask for star detection."""
        import cv2
        from PyQt5.QtWidgets import QFileDialog

        default_path = os.path.join(self.config.config_file_path, "mask.bmp")
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Mask", default_path, "BMP Files (*.bmp);;All Files (*)")

        if file_path:
            mask_img = self.generateMaskImage()
            cv2.imwrite(file_path, mask_img)
            print(f"Mask saved to: {file_path}")

            # Update self.mask so star detection uses the new mask
            self.mask = MaskStructure(mask_img)
            print("Mask updated for star detection")

            # Mark as saved
            self.tab.mask.setUnsaved(False)
            self.tab.mask.updateStatus(len(self.mask_polygons))

    def loadMaskDialog(self):
        """Open dialog to load a mask file."""
        from PyQt5.QtWidgets import QFileDialog

        default_path = self.dir_path
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Mask", default_path, "BMP Files (*.bmp);;All Files (*)")

        if file_path:
            self.loadMaskFromFile(file_path)

    def loadMaskFromFile(self, mask_path):
        """Load mask.bmp and convert masked regions to editable polygons."""
        import cv2

        if not os.path.exists(mask_path):
            print(f"Mask file not found: {mask_path}")
            return

        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            print(f"Failed to load mask: {mask_path}")
            return

        # Clear existing
        self.mask_polygons = []
        self.mask_current_polygon = []

        # Find contours of masked (black) regions
        inverted = cv2.bitwise_not(mask_img)
        contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create polygon for each contour
        for contour in contours:
            # Simplify to reduce points
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # Convert to list of (x, y) tuples
            points = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
            if len(points) >= 3:
                self.mask_polygons.append(points)

        print(f"Loaded {len(self.mask_polygons)} polygon(s) from mask")
        self.updateMaskDisplay()

        # Update self.mask so star detection uses the loaded mask
        self.mask = MaskStructure(mask_img)
        print("Mask updated for star detection")

        # Mark as saved (loaded mask is in sync)
        self.tab.mask.setUnsaved(False)
        self.tab.mask.updateStatus(len(self.mask_polygons))

    ###################################################################################################


    def updatePicks(self):
        """ Draw pick markers for manualreduction """

        pick_color = pg.mkPen((255, 0, 0)) # red
        gap_color = pg.mkPen((255, 215, 0)) # gold

        current = []
        current_pen = pick_color

        data1 = []
        data2 = []

        # Sort picks by frame
        sorted_picks = collections.OrderedDict(sorted(self.pick_list.items(), key=lambda t: t[0]))

        # Get the color for every picks to draw
        for frame, pick in sorted_picks.items():
            if pick['x_centroid'] is not None:

                # Get the color of the current pick
                if self.img.getFrame() == frame:
                    current = [(pick['x_centroid'] + 0.5, pick['y_centroid'] + 0.5)]

                    # Position picks
                    if pick['mode'] == 1:
                        current_pen = pick_color

                    # Gap picks
                    else:
                        current_pen = gap_color

                # Get colors for other picks
                elif pick['mode'] == 1:
                    data1.append([pick['x_centroid'] + 0.5, pick['y_centroid'] + 0.5])

                elif pick['mode'] == 0:
                    data2.append([pick['x_centroid'] + 0.5, pick['y_centroid'] + 0.5])


        ### Add markers to the screen ###
        self.pick_marker.clear()

        # Add the current frame pick
        self.pick_marker.addPoints(pos=current, size=30, pen=current_pen)

        # Plot pick colors
        self.pick_marker.addPoints(pos=data1, size=10, pen=pick_color)

        # Plot gap colors
        self.pick_marker.addPoints(pos=data2, size=10, pen=gap_color)


        # Draw zoom window picks
        self.pick_marker2.clear()
        self.pick_marker2.addPoints(pos=current, size=30, pen=current_pen)
        self.pick_marker2.addPoints(pos=data1, size=10, pen=pick_color)
        self.pick_marker2.addPoints(pos=data2, size=10, pen=gap_color)

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

            self.residual_lines_img.setData(x=np.array(x1) + 0.5, y=np.array(y1) + 0.5)
            self.residual_lines_astro.setData(x=np.array(x2) + 0.5, y=np.array(y2) + 0.5)
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
            _, ra_data, dec_data, _ = xyToRaDecPP(time_data, x_arr, y_arr, level_data, self.platepar, 
                                                  precompute_pointing_corr=True)

            # Compute X, Y back without the distortion
            jd = date2JD(*self.img_handle.currentTime())
            x_nodist, y_nodist = raDecToXYPP(ra_data, dec_data, jd, platepar_nodist)

            x = [None]*2*len(x_arr)
            x[::2] = x_arr
            x[1::2] = x_nodist

            y = [None]*2*len(y_arr)
            y[::2] = y_arr
            y[1::2] = y_nodist

            self.distortion_lines.setData(x=np.array(x) + 0.5, y=np.array(y) + 0.5)

            # Update distortion center marker
            self.updateDistortionCenterMarker()


    def updateDistortionCenterMarker(self):
        """Update the distortion center (optical axis) marker position."""
        if self.platepar:
            # Get the distortion center from the platepar
            x_centre, y_centre = self.platepar.getDistortionCentre()

            # Update marker position (add 0.5 for pixel center alignment)
            self.distortion_center_marker.setData(x=[x_centre + 0.5], y=[y_centre + 0.5])
            self.distortion_center_marker2.setData(x=[x_centre + 0.5], y=[y_centre + 0.5])


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
        elevation_list = []
        snr_list = []
        saturation_list = []

        for paired_star in self.paired_stars.allCoords():

            img_star, catalog_star = paired_star

            star_x, star_y, fwhm, px_intens, snr, saturated = img_star
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
            snr_list.append(snr)
            saturation_list.append(saturated)

            # Compute the azimuth and elevation of the star
            _, alt = trueRaDec2ApparentAltAz(star_ra, star_dec, date2JD(*self.img_handle.currentTime()),
                                                self.platepar.lat, self.platepar.lon, 
                                                self.platepar.refraction)
            
            elevation_list.append(alt)


        # Skip the photometry if there are no stars
        if len(star_coords) == 0:
            
            self.photom_fit_stddev = None
            self.photom_fit_resids = []

            return


        self.residual_text.clear()

        # Compute apparent magnitude corrected for extinction
        catalog_mags = extinctionCorrectionTrueToApparent(catalog_mags, catalog_ra, catalog_dec,
                                                            date2JD(*self.img_handle.currentTime()),
                                                            self.platepar)


        # Determine if the vignetting should be kept fixed. Only if:
        # a) Explicitly kept fixed
        # b) The flat is used, then the vignetting coeff is zero
        fixed_vignetting = None
        if self.flat_struct is not None:
            fixed_vignetting = 0.0

        elif self.platepar.vignetting_fixed:
            fixed_vignetting = self.platepar.vignetting_coeff

        
        # Set the fit weights so that everyting with SNR > 10 is weighted the maximum value
        weights = np.clip(snr_list, 0, 10)/10.0

        # Fit the photometric offset (disable vignetting fit if a flat is used)
        # The fit is going to be weighted by the signal to noise ratio to reduce the influence of
        #  faint stars with large errors
        # Saturated stars are excluded from the fit
        photom_params, self.photom_fit_stddev, self.photom_fit_resids = photometryFit(
            px_intens_list, radius_list, catalog_mags, fixed_vignetting=fixed_vignetting,
            weights=weights, exclude_list=saturation_list)

        photom_offset, vignetting_coeff = photom_params

        # Set photometry parameters
        self.platepar.mag_0 = -2.5
        self.platepar.mag_lev = photom_offset
        self.platepar.mag_lev_stddev = self.photom_fit_stddev
        self.platepar.vignetting_coeff = vignetting_coeff

        # Update the values in the platepar tab in the GUI
        self.tab.param_manager.updatePlatepar()

        if self.selected_stars_visible and (len(star_coords) > 0):

            # Plot photometry deviations on the main plot as colour coded rings
            star_coords = np.array(star_coords)
            star_coords_x, star_coords_y = star_coords.T

            std = np.std(self.photom_fit_resids)
            for star_x, star_y, fit_diff, star_mag, snr in zip(star_coords_x, star_coords_y,
                                                            self.photom_fit_resids, catalog_mags,
                                                            self.paired_stars.snr()
                                                            ):

                photom_resid_txt = "{:.2f}".format(fit_diff)

                snr_txt = "S/N\n{:.1f}".format(snr)

                # Determine the size of the residual text, larger the residual, larger the text
                photom_resid_size = int(8 + np.abs(fit_diff)/(np.max(np.abs(self.photom_fit_resids))/5.0))

                # Determine the RGB color of the SNR text.
                # SNR > 10 is green, SNR < 10 is yellow, SNR < 5 is orange, SNR < 3 is red
                if snr > 10:
                    # Green
                    snr_color = QtGui.QColor(0, 255, 0)
                elif (snr < 10) and (snr >= 5):
                    # Yellow
                    snr_color = QtGui.QColor(255, 255, 0)
                elif (snr < 5) and (snr >= 3):
                    # Orange
                    snr_color = QtGui.QColor(255, 165, 0)
                else:
                    # Red
                    snr_color = QtGui.QColor(255, 0, 0)

                if self.stdev_text_filter*std <= abs(fit_diff):

                    # Add the photometric residual text below the star
                    text_resid = TextItem(photom_resid_txt, anchor=(0.5, -0.5), interaction=False)
                    text_resid.setPos(star_x, star_y)
                    text_resid.setFont(QtGui.QFont('Arial', photom_resid_size))
                    text_resid.setColor(QtGui.QColor(255, 255, 255))
                    text_resid.setAlign(QtCore.Qt.AlignCenter)
                    self.residual_text.addTextItem(text_resid)

                    # Add the star magnitude above the star
                    text_mag = TextItem("{:+6.2f}".format(star_mag), anchor=(0.5, 1.5), interaction=False)
                    text_mag.setPos(star_x, star_y)
                    text_mag.setFont(QtGui.QFont('Arial', 10))
                    text_mag.setColor(QtGui.QColor(0, 255, 0))
                    text_mag.setAlign(QtCore.Qt.AlignCenter)
                    self.residual_text.addTextItem(text_mag)

                    # Add SNR to the right of the star
                    text_snr = TextItem(snr_txt, anchor=(-0.25, 0.5), interaction=False)
                    text_snr.setPos(star_x, star_y)
                    text_snr.setFont(QtGui.QFont('Arial', 8))
                    text_snr.setColor(snr_color)
                    text_snr.setAlign(QtCore.Qt.AlignCenter)
                    self.residual_text.addTextItem(text_snr)



            self.residual_text.update()

        # Show the photometry fit plot
        if show_plot:

            ### PLOT PHOTOMETRY FIT ###
            # Note: An almost identical code exists in Utils.CalibrationReport

            # # Init plot for photometry
            # fig_p, (ax_p, ax_r, ax_e) = plt.subplots(nrows=3, facecolor=None, figsize=(6.4, 8),
            #                                    gridspec_kw={'height_ratios': [3, 1, 1]})

            # Init plot for photometry
            fig_p = plt.figure(figsize=(10, 5))  # Adjust the figure size as needed

            # Create a grid with 2 columns and 2 rows
            gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

            # Large plot on the left
            ax_p = fig_p.add_subplot(gs[:, 0])

            # Two smaller plots on the right, one on top of the other
            ax_r = fig_p.add_subplot(gs[0, 1])
            ax_e = fig_p.add_subplot(gs[1, 1])

            # Set photometry window title
            try:
                fig_p.canvas.set_window_title('Photometry')

            except AttributeError:
                fig_p.canvas.manager.window.setWindowTitle('Photometry')
            
            except:
                print("Warning: Could not set window title for photometry plot.")

            # Plot catalog magnitude vs. raw logsum of pixel intensities
            lsp_arr = np.log10(np.array(px_intens_list))
            ax_p.scatter(-2.5*lsp_arr, catalog_mags, s=5, c='r', zorder=3, alpha=0.5,
                            label="Raw (extinction corrected)")

            # Circle saturated stars in red empty circles
            saturation_label_set = False
            for lsp, cat_mag, sat in zip(lsp_arr, catalog_mags, saturation_list):

                if sat:

                    # Set the label only once
                    if not saturation_label_set:
                        saturation_label = "Saturated"
                        saturation_label_set = True
                    else:
                        saturation_label = None

                    ax_p.scatter(-2.5*lsp, cat_mag, s=30, zorder=3, edgecolor='r',
                                    facecolor='none', label=saturation_label)

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
            fit_info = "{:+.1f}*LSP + {:.2f} $\\pm$ {:.2f} mag".format(self.platepar.mag_0,
                                                                        self.platepar.mag_lev,
                                                                        self.photom_fit_stddev) \
                        + "\nVignetting coeff = {:.5f} rad/px".format(self.platepar.vignetting_coeff) \
                        + "\nGamma = {:.2f}".format(self.platepar.gamma)

            print()
            print('Photometric fit:')
            print(fit_info)
            print()

            # Plot the line fit
            logsum_arr = np.linspace(x_min_w, x_max_w, 10)
            ax_p.plot(
                logsum_arr,
                photomLine((10**(logsum_arr/(-2.5)), np.zeros_like(logsum_arr)), photom_offset,
                            self.platepar.vignetting_coeff),
                label=fit_info, linestyle='--', color='k', alpha=0.5, zorder=3)

            ax_p.legend()

            ax_p.set_ylabel("Catalog magnitude ({:s})".format(self.mag_band_string))
            ax_p.set_xlabel("Uncalibrated magnitude")

            # Set wider axis limits
            ax_p.set_xlim(x_min_w, x_max_w)
            ax_p.set_ylim(y_min_w, y_max_w)

            ax_p.invert_yaxis()
            ax_p.invert_xaxis()

            # Force equal aspect ratio
            ax_p.set_aspect('equal', adjustable='box')

            ax_p.grid()

            ###

            ### PLOT MAG DIFFERENCE BY RADIUS

            img_diagonal = np.hypot(self.platepar.X_res/2, self.platepar.Y_res/2)

            # Plot radius from centre vs. fit residual (including vignetting)
            ax_r.scatter(radius_list, self.photom_fit_resids, s=10, c='b', alpha=0.5, zorder=3)

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

                ax_r.plot(radius_arr_tmp, vignetting_loss, linestyle='dotted', alpha=0.5, color='k',
                            label='Vignetting model')
                
                ax_r.legend()

            ax_r.grid()

            ax_r.set_ylabel("Fit res. (mag)")
            ax_r.set_xlabel("Radius from centre (px)")

            ax_r.set_xlim(0, img_diagonal)

            ### PLOT MAG DIFFERENCE BY ELEVATION

            # Plot elevation vs. fit residual
            ax_e.scatter(elevation_list, self.photom_fit_resids, s=10, c='b', alpha=0.5, zorder=3)

            # Compute the fit residuals without extinction
            fit_resids_noext = \
                self.photom_fit_resids + self.platepar.extinction_scale*atmosphericExtinctionCorrection(
                    np.array(elevation_list), self.platepar.elev)
            
            # Plot elevation vs. fit residual (excluding extinction)
            ax_e.scatter(elevation_list, fit_resids_noext, s=5, c='k', alpha=0.5, zorder=3,
                            label="No extinction, vig. included")


            # Compute the extinction model
            elev_arr = np.linspace(np.min(elevation_list), np.max(elevation_list), 100)
            extinction_model = self.platepar.extinction_scale*atmosphericExtinctionCorrection(
                elev_arr, self.platepar.elev)
            
            # Plot the extinction model
            ax_e.plot(elev_arr, extinction_model, linestyle='dotted', alpha=0.5, color='k', 
                        label='Extinction model')
            

            # Plot a zero line
            ax_e.plot(elev_arr, np.zeros_like(elev_arr), linestyle='dashed', alpha=0.5, color='k')
            
            
            ax_e.grid()
            ax_e.legend()

            ax_e.set_ylabel("Fit res. (mag)")
            ax_e.set_xlabel("Elevation (deg)")

            ###

            fig_p.tight_layout()
            fig_p.show()


    def filterPhotometricOutliers(self, sigma_threshold=2.5):
        """
        Filter paired_stars by removing photometric outliers.

        Performs preliminary photometry and removes stars whose magnitude
        residuals exceed sigma_threshold standard deviations.

        Arguments:
            sigma_threshold: [float] Number of standard deviations for outlier detection.

        Returns:
            int: Number of stars removed.
        """
        if len(self.paired_stars) < 10:
            return 0

        # Collect magnitude residuals for non-saturated stars
        residuals = []
        valid_indices = []
        ra_list = []
        dec_list = []

        for i, (x, y, fwhm, intens_acc, obj, snr, saturated) in enumerate(
                self.paired_stars.paired_stars):

            # Skip saturated stars
            if saturated:
                continue

            # Skip geo points (only process catalog stars)
            if hasattr(obj, 'pick_type') and obj.pick_type == "geopoint":
                continue

            ra, dec, cat_mag = obj.coords()

            # Skip invalid intensities
            if intens_acc <= 0 or np.isnan(intens_acc) or np.isinf(intens_acc):
                continue

            # Compute instrumental magnitude
            inst_mag = -2.5 * np.log10(intens_acc)

            # Store for extinction correction
            residuals.append((cat_mag, inst_mag))
            valid_indices.append(i)
            ra_list.append(ra)
            dec_list.append(dec)

        if len(residuals) < 5:
            return 0

        # Apply extinction correction to catalog magnitudes
        cat_mags = np.array([r[0] for r in residuals])
        inst_mags = np.array([r[1] for r in residuals])

        cat_mags_corrected = extinctionCorrectionTrueToApparent(
            cat_mags, ra_list, dec_list,
            date2JD(*self.img_handle.currentTime()),
            self.platepar)

        # Compute residuals (relative - offset cancels in sigma calc)
        mag_residuals = cat_mags_corrected - inst_mags

        # Sigma clipping
        median = np.median(mag_residuals)
        std = np.std(mag_residuals)

        if std < 0.01:  # Avoid division issues
            return 0

        # Find outliers
        outlier_mask = np.abs(mag_residuals - median) > sigma_threshold * std
        outlier_indices = set(valid_indices[i] for i, is_outlier
                              in enumerate(outlier_mask) if is_outlier)

        # Remove outliers from paired_stars
        if len(outlier_indices) > 0:
            new_paired_stars = PairedStars()
            for i, (x, y, fwhm, intens_acc, obj, snr, saturated) in enumerate(
                    self.paired_stars.paired_stars):
                if i not in outlier_indices:
                    new_paired_stars.addPair(x, y, fwhm, intens_acc, obj, snr, saturated)
            self.paired_stars = new_paired_stars

        removed_count = len(outlier_indices)
        if removed_count > 0:
            print(f"Removed {removed_count} photometric outliers (>{sigma_threshold} sigma)")

        return removed_count


    def filterBlendedStars(self, blend_radius_arcsec=30.0, mag_diff_limit=2.0):
        """
        Filter paired_stars by removing likely blended stars.

        A star is considered blended if there are other bright catalog stars
        within blend_radius_arcsec of the matched catalog star position.

        Arguments:
            blend_radius_arcsec: [float] Radius in arcseconds to check for neighbors.
            mag_diff_limit: [float] Only consider neighbors within this mag of matched star.

        Returns:
            int: Number of stars removed.
        """
        # Get all catalog stars for neighbor lookup
        if not hasattr(self, 'catalog_stars') or self.catalog_stars is None:
            return 0

        self.paired_stars, removed_count = filterBlendedStars(
            self.paired_stars, self.catalog_stars,
            blend_radius_arcsec=blend_radius_arcsec,
            mag_diff_limit=mag_diff_limit,
            verbose=True
        )

        return removed_count


    def filterHighFWHMStars(self, fraction=0.10):
        """
        Filter paired_stars by removing the worst fraction of stars by FWHM.

        Stars with high FWHM tend to have worse centroiding precision due to:
        - Blended sources
        - Extended objects (galaxies)
        - Poor atmospheric seeing
        - Saturation/defocus

        Arguments:
            fraction: [float] Fraction of stars to remove (0.10 = top 10% highest FWHM).

        Returns:
            int: Number of stars removed.
        """
        self.paired_stars, removed_count = filterHighFWHMStars(
            self.paired_stars,
            fraction=fraction,
            verbose=True
        )

        return removed_count


    def balanceCatalogMagnitude(self):
        """
        Balance catalog magnitude limit to have ~2x more catalog stars than detected stars.

        This improves NN matching by ensuring a good ratio between detected and catalog stars.
        Updates self.cat_lim_mag and reloads catalog_stars if needed.

        Returns:
            bool: True if balancing was performed, False otherwise.
        """
        # Get current FF file name
        ff_name_c = convertFRNameToFF(self.img_handle.name())

        # Get detected stars count
        n_detected = 0
        if self.star_detection_override_enabled and ff_name_c in self.star_detection_override_data:
            n_detected = len(self.star_detection_override_data[ff_name_c])
        elif ff_name_c in self.calstars:
            n_detected = len(self.calstars[ff_name_c])

        if n_detected < 10:
            return False

        # Compute JD for projection
        jd = date2JD(*self.img_handle.currentTime())

        # Get current catalog stars in FOV
        _, catalog_stars_extended = self.filterCatalogStarsInsideFOV(self.catalog_stars)

        # Project to image and count stars strictly inside FOV
        catalog_x, catalog_y, _ = getCatalogStarsImagePositions(
            catalog_stars_extended, jd, self.platepar)
        in_fov = (catalog_x >= 0) & (catalog_x < self.platepar.X_res) & \
                 (catalog_y >= 0) & (catalog_y < self.platepar.Y_res)
        n_catalog = np.sum(in_fov)

        # Target: 1.5x to 2.5x detected stars
        target_min = int(n_detected * 1.5)
        target_max = int(n_detected * 2.5)

        if target_min <= n_catalog <= target_max:
            # Already in range
            return False

        print()
        print("Balancing catalog stars ({:d}) to match detected stars ({:d})...".format(
            n_catalog, n_detected))
        print("  Target range: {:d} - {:d} catalog stars".format(target_min, target_max))

        # Binary search for optimal magnitude limit
        # Use actual current limit, not config default (user may have adjusted with +/-)
        original_mag_limit = self.cat_lim_mag
        current_mag_limit = original_mag_limit
        mag_low, mag_high = 3.0, 12.0
        best_mag_limit = current_mag_limit
        best_n_catalog = n_catalog

        for iteration in range(10):  # Max 10 iterations
            if n_catalog < target_min:
                # Need more catalog stars - increase mag limit
                mag_low = current_mag_limit
                current_mag_limit = (current_mag_limit + mag_high) / 2.0
            elif n_catalog > target_max:
                # Too many catalog stars - decrease mag limit
                mag_high = current_mag_limit
                current_mag_limit = (mag_low + current_mag_limit) / 2.0
            else:
                # In range, done
                break

            # Reload catalog with new limit
            old_cat_lim_mag = self.cat_lim_mag
            self.cat_lim_mag = current_mag_limit
            temp_catalog = self.loadCatalogStars(current_mag_limit)
            _, temp_catalog_fov = self.filterCatalogStarsInsideFOV(temp_catalog)
            self.cat_lim_mag = old_cat_lim_mag  # Restore temporarily

            # Project and filter to strict FOV
            temp_x, temp_y, _ = getCatalogStarsImagePositions(temp_catalog_fov, jd, self.platepar)
            in_fov = (temp_x >= 0) & (temp_x < self.platepar.X_res) & \
                     (temp_y >= 0) & (temp_y < self.platepar.Y_res)
            n_catalog = np.sum(in_fov)

            print("    Iter {:d}: mag_limit={:.2f}, catalog_in_fov={:d}".format(
                iteration + 1, current_mag_limit, n_catalog))

            if target_min <= n_catalog <= target_max:
                best_mag_limit = current_mag_limit
                best_n_catalog = n_catalog
                print("    -> In target range, done")
                break

            # Track best result so far
            if abs(n_catalog - (target_min + target_max)/2) < abs(best_n_catalog - (target_min + target_max)/2):
                best_mag_limit = current_mag_limit
                best_n_catalog = n_catalog

        # Apply best magnitude limit found
        if best_mag_limit != original_mag_limit:
            print("  Adjusted catalog mag limit: {:.1f} -> {:.1f}".format(
                original_mag_limit, best_mag_limit))

            # Permanently update cat_lim_mag and reload catalog
            self.cat_lim_mag = best_mag_limit
            self.catalog_stars = self.loadCatalogStars(best_mag_limit)

            # Update GUI to show the balanced magnitude limit
            self.updateLeftLabels()
            self.tab.settings.updateLimMag()

            return True
        else:
            print("  Could not reach target range (best: {:d} stars at mag_limit={:.1f})".format(
                best_n_catalog, best_mag_limit))
            return False


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

        self.first_platepar_fit = True



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
            self.paired_stars = PairedStars()
            self.unsuitable_stars = PairedStars()
            self.residuals = None

            # Clear residual overlay from previous image
            self.updateFitResiduals()
            self.residual_text.clear()

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

                    # Only allow changing frames to adjacent ones to the max/min
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
            self.updateStars(only_update_catalog=True)
            self.drawPhotometryColoring()
            self.showFRBox()



        self.updateLeftLabels()
        self.updateImageNavigationDisplay()


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


    def jumpToImage(self, image_num):
        """ Jump directly to a specific image number (1-indexed).

        Arguments:
            image_num: [int] Target image number (1-indexed, 1 to total_images)
        """
        # Only works in skyfit mode with multiple images
        if self.mode != 'skyfit':
            return

        # Check if we have a valid image handle with ff_list
        if not hasattr(self.img_handle, 'ff_list'):
            return

        # Convert 1-indexed UI to 0-indexed internal
        target_index = image_num - 1
        current_index = self.img_handle.current_ff_index

        # Calculate delta and navigate
        delta = target_index - current_index
        if delta != 0:
            # Block signals to prevent recursive calls
            self.image_navigation_slider.blockSignals(True)
            self.nextImg(n=delta)
            self.image_navigation_slider.blockSignals(False)


    def updateImageNavigationDisplay(self):
        """ Update the image navigation slider and label to show current position. """
        # Only update if we have a multi-image handle with ff_list
        if not hasattr(self.img_handle, 'ff_list'):
            self.image_navigation_slider.hide()
            self.image_navigation_label.hide()
            return

        # Show slider for multi-image inputs
        self.image_navigation_slider.show()
        self.image_navigation_label.show()

        # Update bounds and current value
        total_images = len(self.img_handle.ff_list)
        current_index = self.img_handle.current_ff_index

        self.image_navigation_slider.blockSignals(True)
        self.image_navigation_slider.setMaximum(total_images)
        self.image_navigation_slider.setValue(current_index + 1)  # 1-indexed display
        self.image_navigation_slider.blockSignals(False)

        # Update label text
        self.image_navigation_label.setText(f'Image: {current_index + 1} / {total_images}')


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
        # print('input path', dic['input_path'])
        for k, v in dic.items():

            if (v.__class__.__bases__[0] is not object) and (not isinstance(v, bool)) and \
                (not isinstance(v, float)):

                # Remove class that inherits from something
                to_remove.append(k)
                # print(k,v)

        for remove in to_remove:
            del dic[remove]

        # Explicitly remove pyqtgraph items that might not have been caught
        if 'sat_track_curves' in dic:
            del dic['sat_track_curves']
        if 'sat_track_labels' in dic:
            del dic['sat_track_labels']
        if 'sat_track_arrows' in dic:
            del dic['sat_track_arrows']
        if 'sat_markers' in dic:
            del dic['sat_markers']

        # Save the FF file name if the input type is FF
        if self.img_handle.input_type == 'ff':
            dic['ff_file'] = self.img_handle.name()
            
        savePickle(dic, self.dir_path, 'skyFitMR_latest.state')
        print("Saved state to file")


    def findLoadState(self):
        """ Opens file dialog to find .state file to load then calls loadState """

        file_ = QtWidgets.QFileDialog.getOpenFileName(self, "Load .state file", self.dir_path,
                                                      "State file (*.state);;All files (*)")[0]

        if file_:
            self.loadState(os.path.dirname(file_), os.path.basename(file_))


    def loadState(self, dir_path, state_name, beginning_time=None, mask=None):
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
        
        # Init catalog_stars_spectral_type if it doesn't exist (loading old state files)
        if not hasattr(self, 'catalog_stars_spectral_type'):
            self.catalog_stars_spectral_type = None
        
        # Init catalog_stars_preferred_names if it doesn't exist
        if not hasattr(self, 'catalog_stars_preferred_names'):
            self.catalog_stars_preferred_names = None

        # Init show_constellations if it doesn't exist
        if not hasattr(self, 'show_constellations'):
            self.show_constellations = False

        # Init catalog_stars_common_names if it doesn't exist
        if not hasattr(self, 'catalog_stars_common_names'):
            self.catalog_stars_common_names = None

        # Updating old state files with new platepar variables
        if self.platepar is not None:
            if not hasattr(self.platepar, "equal_aspect"):
                self.platepar.equal_aspect = True

            if not hasattr(self.platepar, "force_distortion_centre"):
                self.platepar.force_distortion_centre = False

            if not hasattr(self.platepar, "asymmetry_corr"):
                self.platepar.asymmetry_corr = False

            if not hasattr(self.platepar, "extinction_scale"):
                self.platepar.extinction_scale = 1.0

            if not hasattr(self.platepar, "vignetting_fixed"):
                self.platepar.vignetting_fixed = False

            if not hasattr(self.platepar, "measurement_apparent_to_true_refraction"):
                self.platepar.measurement_apparent_to_true_refraction = False


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

        # Update the path to the RMS root directory
        self.config.rms_root_dir = getRmsRootDir()
        
        # Swap the fixed variable name
        if hasattr(self, "star_aperature_radius"):
            self.star_aperture_radius = self.star_aperature_radius


        # Update img_handle parameters
        if hasattr(self, "img_handle"):

            # Update the dir path
            self.img_handle.dir_path = dir_path

            # Add the missing flipud parameter
            if not hasattr(self.img_handle, "flipud"):
                self.img_handle.flipud = False

            # If the input type is FF and the path to the actual FF file got saved, update it
            if self.img_handle.input_type == 'ff':
                if "ff_file" in variables:
                    
                    # If the file is available in the input path, update the FF file path
                    ff_file = variables["ff_file"]

                    if os.path.isfile(os.path.join(dir_path, ff_file)):
                        self.img_handle.setCurrentFF(ff_file)

            # Make sure an option is not missing
            if self.img_handle.input_type == 'images':

                # Add the fripon mode flag if it's missing
                if not hasattr(self.img_handle, "fripon_mode"):
                    self.img_handle.fripon_mode = False
                    self.img_handle.fripon_header = None

            # Make sure an option is not missing
            if self.img_handle.input_type == 'images':
                if not hasattr(self.img_handle, "cabernet_status"):
                    self.img_handle.cabernet_status = False

            # Make sure an option is not missing from the UWO png mode
            if self.img_handle.input_type == 'images':
                if self.img_handle.uwo_png_mode:

                    if not hasattr(self.img_handle, "uwo_magick_type"):

                        # Disable uwo mode just to read the first frame without issues
                        self.img_handle.uwo_png_mode = False
                        
                        # Load the first image
                        img = self.img_handle.loadFrame(fr_no=0)

                        # Re-enable uwo mode
                        self.img_handle.uwo_png_mode = True

                        # Get the magick type
                        self.img_handle.uwo_magick_type = self.img_handle.getUWOMagickType(img)


        # Update possibly missing input_path variable
        if not hasattr(self, "input_path"):
            self.input_path = dir_path

        # Update the input path if it exists
        else:
            if self.input_path:

                # Normalize separator so it's system independent
                self.input_path = self.input_path.replace('\\', os.sep).replace('/', os.sep)

                # Check if the last part of the input path is a file or a directory (check for the dot)
                tmp_name = os.path.basename(self.input_path)
                if "." in tmp_name:
                    self.input_path = os.path.join(dir_path, tmp_name)
                else:
                    self.input_path = dir_path

                print(self.input_path)

        # Update the possibly missing params
        if not hasattr(self, "dark"):
            self.dark = None

        # Update the possibly missing params
        if not hasattr(self, "fit_only_pointing"):
            self.fit_only_pointing = False

        # Update the possibly missing params
        if not hasattr(self, "fixed_scale"):
            self.fixed_scale = False

        # Update the possibly missing params
        if not hasattr(self, "station_moved_auto_refit"):
            self.station_moved_auto_refit = False

        # Update the possibly missing params
        if not hasattr(self, "geo_points_obj"):
            self.geo_points_obj = None


        # Update the possibly missing begin time
        if not hasattr(self, "beginning_time"):
            self.beginning_time = beginning_time


        # Add the mask
        if not hasattr(self, "mask"):
            self.mask = mask


        # If the previous beginning time is None and the new one is not, update the beginning time
        if (self.beginning_time is None) and (beginning_time is not None):
            self.beginning_time = beginning_time


        if not hasattr(self, "pick_list"):
            self.pick_list = {}

        # If SNR and saturation flags are missing in the pick list, add them
        for _, pick in self.pick_list.items():
            if 'background_intensity' not in pick:
                pick['background_intensity'] = 0.0
            if 'snr' not in pick:
                pick['snr'] = 1.0
            if 'saturated' not in pick:
                pick['saturated'] = False

        # Update possibly missing flag for measuring ground points
        if not hasattr(self, "meas_ground_points"):
            self.meas_ground_points = False

        if not hasattr(self, "autopan_mode"):
            self.autopan_mode = False

        if not hasattr(self, "unsuitable_stars"):
            self.unsuitable_stars = PairedStars()

        if not hasattr(self, "max_pixels_between_matched_stars"):
            self.max_pixels_between_matched_stars = np.inf

        # Update possibly missing flag for measuring ground points
        if not hasattr(self, "single_click_photometry"):
            self.single_click_photometry = False

        # Update possibly missing flag for not subtracting the background
        if not hasattr(self, "no_background_subtraction"):
            self.no_background_subtraction = False

        # Update possibly missing flag for peripheral background estiamtion
        if not hasattr(self, "peripheral_background_subtraction"):
            self.peripheral_background_subtraction = False

        # Update the possibly missing flag for flipping the image upside down
        if not hasattr(self, "flipud"):
            self.flipud = False

        # Update the possibly missing flag for subtracting the bias from the flat
        if not hasattr(self, "flatbiassub"):
            self.flatbiassub = False

        # Update the possibly missing exposure ratio variable
        if not hasattr(self, "exposure_ratio"):
            self.exposure_ratio = 1.0


        # Add the possibily missing variables for ASTRA
        if not hasattr(self, "astra_dialog"):
            self.astra_dialog = None

        if not hasattr(self, "astra_config_params"):
            self.astra_config_params = None

        if not hasattr(self, "saturation_threshold"):
            self.saturation_threshold = int(round(0.98*(2**self.config.bit_depth - 1)))

        if not hasattr(self, "snr_centroid"):
            self.snr_centroid = 1.0

        if not hasattr(self, "saturated_centroid"):
            self.saturated_centroid = False

        # Add the missing satellite overlay variables
        if not hasattr(self, "show_sattracks"):
            self.show_sattracks = False
        if not hasattr(self, "tle_file"):
            self.tle_file = None
        if not hasattr(self, "satellite_tracks"):
            self.satellite_tracks = []
        if not hasattr(self, "fov_poly_cache"):
            self.fov_poly_cache = None
        if not hasattr(self, "fov_poly_jd"):
            self.fov_poly_jd = None
        if not hasattr(self, "sat_track_curves"):
            self.sat_track_curves = []
        if not hasattr(self, "sat_track_labels"):
            self.sat_track_labels = []
        if not hasattr(self, "sat_track_arrows"):
            self.sat_track_arrows = []
        if not hasattr(self, "sat_markers"):
            self.sat_markers = []

        # If the paired stars are a list (old version), reset it to a new version where it's an object
        if isinstance(self.paired_stars, list):

            paired_stars_new = PairedStars()

            for entry in self.paired_stars:
                
                img_coords, sky_coords = entry
                x, y, fwhm, intens_acc, snr = img_coords
                sky_obj = CatalogStar(*sky_coords)

                paired_stars_new.addPair(x, y, fwhm, intens_acc, sky_obj, snr=snr)

            self.paired_stars = paired_stars_new

        # Add missing paired_stars parameters
        else:
            for paired_star in self.paired_stars.paired_stars:

                # Add SNR if it's missing
                if len(paired_star) == 4:
                    paired_star.append(1.0)

                # Add the saturation flag is it's missing
                if len(paired_star) == 5:
                    paired_star.append(False)

                # If the FWHM is missing, add it to the 3rd index
                if len(paired_star) == 6:
                    paired_star.insert(2, 0.0)


        if self.platepar is not None:

            # Compute if the measurement should be post-corrected for refraction, because it was not
            #   taken into account during the astrometry calibration procedure
            self.updateMeasurementRefractionCorrection()


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


        # Update the FPS if forced
        if hasattr(self, 'fps'):
            self.setFPS()


    def eventFilter(self, obj, event):
        """Event filter to catch mouse release on view_widget viewport (ViewBox doesn't receive it during panning)."""
        if event.type() == QtCore.QEvent.MouseButtonRelease:
            # Convert widget coords to scene coords
            scene_pos = self.view_widget.mapToScene(event.pos())
            self.handleMouseRelease(event.button(), scene_pos.x(), scene_pos.y())
        return False  # Don't consume the event

    def handleMouseRelease(self, button, scene_x, scene_y):
        """Handle mouse release for star picking (called from eventFilter)."""
        # Stop mask vertex dragging
        self.mask_dragging_vertex = None

        # Check if this was a click (not a drag) for star picking
        if self.press_scene_x is not None and self.star_pick_mode:
            # Check if mouse moved more than threshold in screen pixels
            drag_distance = np.hypot(scene_x - self.press_scene_x, scene_y - self.press_scene_y)
            click_threshold = 5.0  # screen pixels

            if drag_distance < click_threshold:
                self.handleStarPick(self.press_button, self.press_modifiers)

        # Clear press tracking and clicked state
        self.press_scene_x = None
        self.press_scene_y = None
        self.press_button = None
        self.press_modifiers = None
        self.clicked = 0

    def onMouseReleased(self, event):
        # Note: This may not be called during panning - handleMouseRelease via eventFilter is the main handler
        # Keep this for non-panning scenarios and mask vertex dragging
        self.mask_dragging_vertex = None


    def handleStarPick(self, button, modifiers):
        """Handle star picking on click (not drag). Called from onMouseReleased."""

        # Add star pair in SkyFit
        if self.mode == 'skyfit':

            # Add star
            if button == QtCore.Qt.LeftButton:

                if self.cursor.mode == 0:

                    # If CTRL is pressed, place the pick manually - NOTE: the intensity might be off then!!!
                    if modifiers & QtCore.Qt.ControlModifier:
                        self.x_centroid = self.mouse_x - 0.5
                        self.y_centroid = self.mouse_y - 0.5

                        # Compute the star intensity
                        (
                            _, _, self.star_fwhm, self.star_intensity, self.star_snr, self.star_saturated
                        ) = self.centroid(
                            prev_x_cent=self.x_centroid, prev_y_cent=self.y_centroid
                            )
                    else:

                        # Check if a star centroid is available from CALSTARS, and use it first because
                        #   the PSF fit should result in a better centroid estimate

                        # Check if the closest CALSTARS star is within the radius

                        calstars_centroid = False
                        if self.img_handle is not None:

                            # Handle using FR files too
                            ff_name_c = convertFRNameToFF(self.img_handle.name())

                            # Use override data if enabled, otherwise CALSTARS
                            star_data = None
                            if self.star_detection_override_enabled and ff_name_c in self.star_detection_override_data:
                                star_data = np.array(self.star_detection_override_data[ff_name_c])
                            elif ff_name_c in self.calstars:
                                star_data = np.array(self.calstars[ff_name_c])

                            if star_data is not None:

                                if len(star_data):

                                    # Get star coordinates
                                    stars_x = star_data[:, 1]
                                    stars_y = star_data[:, 0]

                                    # Compute the distance from the mouse press
                                    mouse_x = self.mouse_x - 0.5
                                    mouse_y = self.mouse_y - 0.5
                                    dist_arr = np.hypot(stars_x - mouse_x, stars_y - mouse_y)

                                    # Find the closest distance
                                    closest_dist_indx = np.argmin(dist_arr)

                                    # If the CALSTARS entry is within the aperture radius, take that star
                                    if dist_arr[closest_dist_indx] <= self.star_aperture_radius:

                                        self.x_centroid = stars_x[closest_dist_indx]
                                        self.y_centroid = stars_y[closest_dist_indx]

                                        # Compute the star intensity
                                        _, _, self.star_fwhm, \
                                            self.star_intensity, self.star_snr, self.star_saturated = \
                                            self.centroid( \
                                            prev_x_cent=self.x_centroid, prev_y_cent=self.y_centroid)

                                        calstars_centroid = True


                        # If a CALSTARS star was not found, run a normal centroid
                        if not calstars_centroid:

                            # Perform centroiding with 2 iterations
                            x_cent_tmp, y_cent_tmp, _, _, _, _ = self.centroid()

                            # Check that the centroiding was successful
                            if x_cent_tmp is not None:

                                # Centroid the star around the pressed coordinates
                                (
                                    self.x_centroid, self.y_centroid, self.star_fwhm,
                                    self.star_intensity, self.star_snr, self.star_saturated
                                ) = self.centroid(prev_x_cent=x_cent_tmp, prev_y_cent=y_cent_tmp)

                            else:
                                return None

                    # Add the centroid to the plot
                    self.centroid_star_markers.addPoints(x=[self.x_centroid + 0.5], \
                        y=[self.y_centroid + 0.5])
                    self.centroid_star_markers2.addPoints(x=[self.x_centroid + 0.5], \
                        y=[self.y_centroid + 0.5])


                    # Find coordinates of the star or geo points closest to the clicked point
                    x_data, y_data = self.findClickedStarOrGeoPoint(self.x_centroid, self.y_centroid)


                    # Add a star marker to the main and zoom windows
                    self.sel_cat_star_markers.addPoints(x=np.array(x_data) + 0.5, \
                        y=np.array(y_data) + 0.5)
                    self.sel_cat_star_markers2.addPoints(x=np.array(x_data) + 0.5, \
                        y=np.array(y_data) + 0.5)

                    # Switch to the mode where the catalog star is selected
                    self.cursor.setMode(1)


                elif self.cursor.mode == 1:

                    # REMOVE marker for previously selected
                    self.sel_cat_star_markers.setData(pos=self.paired_stars.imageCoords(draw=True))
                    self.sel_cat_star_markers2.setData(pos=self.paired_stars.imageCoords(draw=True))


                    # Find coordinates of the star or geo points closest to the clicked point
                    x_data, y_data = self.findClickedStarOrGeoPoint(self.mouse_x, self.mouse_y)

                    # Add the new point
                    self.sel_cat_star_markers.addPoints(x=np.array(x_data) + 0.5, \
                        y=np.array(y_data) + 0.5)
                    self.sel_cat_star_markers2.addPoints(x=np.array(x_data) + 0.5, \
                        y=np.array(y_data) + 0.5)


            # Remove star pair on right click
            elif button == QtCore.Qt.RightButton:
                if self.cursor.mode == 0:

                    # Remove the closest picked star from the list
                    self.paired_stars.removeClosestPair(self.mouse_x, self.mouse_y)

                    self.updatePairedStars()
                    self.updateFitResiduals()
                    self.photometry()

        # Add centroid in manual reduction
        else:
            if button == QtCore.Qt.LeftButton:

                if self.cursor.mode == 0:
                    mode = 1

                    if modifiers & QtCore.Qt.ControlModifier or \
                            ((modifiers & QtCore.Qt.AltModifier or QtCore.Qt.Key_0 in self.keys_pressed) and
                             self.img.img_handle.input_type == 'dfn'):

                        self.x_centroid, self.y_centroid = self.mouse_x - 0.5, self.mouse_y - 0.5

                    else:
                        (
                            self.x_centroid, self.y_centroid, self.star_fwhm,
                            _, self.snr_centroid, self.saturated_centroid
                        ) = self.centroid()

                    if (modifiers & QtCore.Qt.AltModifier or QtCore.Qt.Key_0 in self.keys_pressed) and \
                            self.img.img_handle.input_type == 'dfn':
                        mode = 0

                    self.addCentroid(self.img.getFrame(), self.x_centroid, self.y_centroid, mode=mode,
                                     snr=self.snr_centroid, saturated=self.saturated_centroid)

                    self.updatePicks()

                    # Add photometry coloring if single-click photometry is turned on
                    if self.single_click_photometry:

                        self.changePhotometry(self.img.getFrame(), self.photometryColoring(),
                                          add_photometry=True)
                        self.drawPhotometryColoring()


                elif self.cursor.mode == 2:
                    self.changePhotometry(self.img.getFrame(), self.photometryColoring(),
                                          add_photometry=True)
                    self.drawPhotometryColoring()

            elif button == QtCore.Qt.RightButton:
                if self.cursor.mode == 0:
                    self.removeCentroid(self.img.getFrame())
                    self.updatePicks()
                elif self.cursor.mode == 2:
                    self.changePhotometry(self.img.getFrame(), self.photometryColoring(),
                                          add_photometry=False)
                    self.drawPhotometryColoring()


    def onMouseMoved(self, event):

        pos = event

        if self.img_frame.sceneBoundingRect().contains(pos):

            self.img_frame.setFocus()
            mp = self.img_frame.mapSceneToView(pos)

            self.cursor.setCenter(mp)
            self.cursor2.setCenter(mp)
            self.mouse_x, self.mouse_y = mp.x(), mp.y()

            # Handle mask vertex dragging
            if self.mask_dragging_vertex is not None:
                self.moveMaskVertex(self.mask_dragging_vertex, mp.x() - 0.5, mp.y() - 0.5)

            self.zoom()

            # Move zoom window to correct location
            range_ = self.img_frame.getState()['viewRange'][0]
            if mp.x() > (range_[1] - range_[0])/2 + range_[0]:
                self.v_zoom_left = True
                if self.show_key_help != 2:
                    self.v_zoom.move(QtCore.QPoint(int(self.label1.boundingRect().width()), 0))
                else:
                    self.v_zoom.move(QtCore.QPoint(0, 0))
            else:
                self.v_zoom_left = False
                self.v_zoom.move(QtCore.QPoint(int(self.img_frame.size().width() - self.show_zoom_window_size), 0))

            self.updateBottomLabel()

            if self.clicked and self.cursor.mode == 2:
                self.changePhotometry(self.img.getFrame(), self.photometryColoring(),
                                      add_photometry=self.clicked == 1)
                self.drawPhotometryColoring()

        # self.printFrameRate()


    def findClickedStarOrGeoPoint(self, x, y):
        """ Find the coordinate of the star or geo point closest to the clicked point.  """

        # Select the closest catalog star to the centroid as the first guess
        self.closest_type, closest_indx = self.findClosestCatalogStarIndex(x, y)

        if self.closest_type == 'catalog':

            # Fetch the coordinates of the catalog star
            self.closest_cat_star_indx = closest_indx
            x_data = [self.catalog_x_filtered[self.closest_cat_star_indx]]
            y_data = [self.catalog_y_filtered[self.closest_cat_star_indx]]

        else:

            # Find the index among all geo points visible in the FOV of the one closest to the clicked 
            #   position
            self.closest_geo_point_indx = closest_indx

            # Fetch the coordinates of the geo point
            x_data = [self.geo_x[self.closest_geo_point_indx]]
            y_data = [self.geo_y[self.closest_geo_point_indx]]

        return x_data, y_data


    def onMousePressed(self, event):

        if event.button() == QtCore.Qt.LeftButton:
            self.clicked = 1
        elif event.button() == QtCore.Qt.MiddleButton:
            self.clicked = 2
        elif event.button() == QtCore.Qt.RightButton:
            self.clicked = 3

        modifiers = QtWidgets.QApplication.keyboardModifiers()

        # Store press position for click vs drag detection (used for star picking)
        # Use scene coordinates (screen position) not view coordinates, because panning changes view coords
        pos = event.scenePos()
        self.press_scene_x = pos.x()
        self.press_scene_y = pos.y()
        self.press_button = event.button()
        self.press_modifiers = modifiers

        # Handle mask drawing/editing
        if self.mask_draw_mode or len(self.mask_polygons) > 0 or len(self.mask_current_polygon) > 0:
            pos = event.scenePos()
            mp = self.img_frame.mapSceneToView(pos)
            click_x, click_y = mp.x() - 0.5, mp.y() - 0.5

            # Check if clicking near an existing vertex
            vertex_hit = self.findNearestMaskVertex(click_x, click_y, threshold=15)

            if event.button() == QtCore.Qt.LeftButton:
                if vertex_hit is not None:
                    # Start dragging this vertex
                    self.mask_dragging_vertex = vertex_hit
                    return
                elif modifiers & QtCore.Qt.ControlModifier:
                    # CTRL+click: insert vertex on nearest edge
                    edge_hit = self.findNearestMaskEdge(click_x, click_y, threshold=15)
                    if edge_hit is not None:
                        self.insertMaskVertex(edge_hit, click_x, click_y)
                        return
                elif self.mask_draw_mode:
                    # Add new point
                    self.addMaskPoint(click_x, click_y)
                    return
            elif event.button() == QtCore.Qt.RightButton:
                if vertex_hit is not None:
                    # Delete this vertex
                    self.deleteMaskVertex(vertex_hit)
                    return

        # Star picking is handled in onMouseReleased to distinguish clicks from drags (panning)

    def keyPressEvent(self, event):

        # Read modifiers (e.g. CTRL, SHIFT)
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        qmodifiers = QtWidgets.QApplication.queryKeyboardModifiers()

        self.keys_pressed.append(event.key())

        # Handle mask drawing - Space or Enter to close polygon
        if self.mask_draw_mode and len(self.mask_current_polygon) >= 3:
            if event.key() in (QtCore.Qt.Key_Space, QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                if modifiers == QtCore.Qt.NoModifier:
                    self.closeMaskPolygon()
                    return

        # Toggle auto levels
        if event.key() == QtCore.Qt.Key_A and (modifiers == QtCore.Qt.ControlModifier):

            self.tab.hist.toggleAutoLevels()
            # This updates image automatically

        # Load the dark
        elif event.key() == QtCore.Qt.Key_D and (modifiers == QtCore.Qt.ControlModifier):
            
            _, self.dark = self.loadDark(force_dialog=True)

            # Set focus back on the SkyFit window
            self.activateWindow()

            # Apply the dark to the flat if the flatbiassub flag is set
            if self.flatbiassub and (self.flat_struct is not None):
                
                self.flat_struct.applyDark(self.dark)

                self.img.flat_struct = self.flat_struct
                self.img_zoom.flat_struct = self.flat_struct

            self.img.dark = self.dark
            self.img_zoom.dark = self.dark

            self.img_zoom.reloadImage()
            self.img.reloadImage()

        # Jump to the next star
        elif event.key() == QtCore.Qt.Key_Space and (modifiers == QtCore.Qt.ShiftModifier):

            self.jumpNextStar(miss_this_one=True)
            self.updateBottomLabel()


        # Toggle satellite tracks
        elif event.key() == QtCore.Qt.Key_T and (modifiers == QtCore.Qt.ControlModifier):
             self.toggleSatelliteTracks()



        # Load the flat
        elif event.key() == QtCore.Qt.Key_F and (modifiers == QtCore.Qt.ControlModifier):
            
            _, self.flat_struct = self.loadFlat(force_dialog=True)

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
            self.updateBottomLabel()

            if self.star_pick_mode:
                #self.img_frame.setMouseEnabled(False, False)
                self.cursor.show()
                self.cursor2.show()

                self.star_pick_info.show()

                # Enable the Next button for star panning
                self.tab.param_manager.next_star_button.setEnabled(True)

            else:
                self.img_frame.setMouseEnabled(True, True)
                self.cursor2.hide()
                self.cursor.hide()

                self.star_pick_info.hide()

                # Disable the Next button for star panning
                self.tab.param_manager.next_star_button.setEnabled(False)


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
        elif event.key() == QtCore.Qt.Key_U and not modifiers == QtCore.Qt.ControlModifier:

            # Increase image gamma by a factor of 1.1x
            self.img.updateGamma(1/0.9)
            if self.img_zoom:
                self.img_zoom.updateGamma(1/0.9)
            self.updateLeftLabels()
            self.tab.settings.updateImageGamma()

        elif event.key() == QtCore.Qt.Key_J and not modifiers == QtCore.Qt.ControlModifier:

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
        elif event.key() == QtCore.Qt.Key_H and not (modifiers == QtCore.Qt.ShiftModifier):
            self.toggleShowCatStars()
            self.tab.settings.updateShowCatStars()

        # Toggle showing astrometry.net matched stars (Shift+H)
        elif event.key() == QtCore.Qt.Key_H and (modifiers == QtCore.Qt.ShiftModifier):
            self.toggleShowAstrometryNetStars()
            if self.astrometry_solution_info is not None:
                matched_count = len(self.astrometry_solution_info.get('matched_pairs', []))
                quad_count = len(self.astrometry_solution_info.get('quad_stars', []))
                status = "shown" if self.astrometry_stars_visible else "hidden"
                print("Astrometry.net stars {:s}: {:d} matched (cyan), {:d} quad (magenta)".format(
                    status, matched_count, quad_count))
            else:
                print("No astrometry.net solution available. Run 'CTRL+X' or 'CTRL+SHIFT+X' first.")

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

            # Change distortion type to poly3+radial5
            if (event.key() == QtCore.Qt.Key_3) and (modifiers == QtCore.Qt.ControlModifier):

                self.dist_type_index = 6
                self.changeDistortionType()
                self.tab.param_manager.updatePlatepar()
                self.updateLeftLabels()
                self.updateDistortion()
                self.updateStars()

            # Change distortion type to radial3
            elif (event.key() == QtCore.Qt.Key_4) and (modifiers == QtCore.Qt.ControlModifier):

                self.dist_type_index = 7
                self.changeDistortionType()
                self.tab.param_manager.updatePlatepar()
                self.updateLeftLabels()
                self.updateDistortion()
                self.updateStars()

            # Change distortion type to radial5
            elif (event.key() == QtCore.Qt.Key_5) and (modifiers == QtCore.Qt.ControlModifier):

                self.dist_type_index = 8
                self.changeDistortionType()
                self.tab.param_manager.updatePlatepar()
                self.updateLeftLabels()
                self.updateDistortion()
                self.updateStars()

            # Change distortion type to radial7
            elif (event.key() == QtCore.Qt.Key_6) and (modifiers == QtCore.Qt.ControlModifier):

                self.dist_type_index = 9
                self.changeDistortionType()
                self.tab.param_manager.updatePlatepar()
                self.updateLeftLabels()
                self.updateDistortion()
                self.updateStars()

            # Change distortion type to radial9
            elif (event.key() == QtCore.Qt.Key_7) and (modifiers == QtCore.Qt.ControlModifier):

                self.dist_type_index = 10
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

                self.fitPickedStars()

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

            # Pan to unmatched star most distant from all other matched stars

            elif event.key() == QtCore.Qt.Key_U and modifiers == QtCore.Qt.ControlModifier:

                self.jumpNextStar(miss_this_one=False)

            elif event.key() == QtCore.Qt.Key_O and modifiers == QtCore.Qt.ControlModifier:

                self.toggleAutoPan()
                self.tab.settings.updateAutoPan()
                self.updateBottomLabel()



            # Move rotation parameter (plain Q only, not Shift+Q)
            elif event.key() == QtCore.Qt.Key_Q and not (modifiers == QtCore.Qt.ShiftModifier):
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


            # Change extinction scale
            elif event.key() == QtCore.Qt.Key_9:
                
                self.platepar.extinction_scale += 0.1

                self.tab.param_manager.updatePlatepar()
                self.onExtinctionChanged()

            elif event.key() == QtCore.Qt.Key_0:

                self.platepar.extinction_scale -= 0.1

                if self.platepar.extinction_scale < 0:
                    self.platepar.extinction_scale = 0.0

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

                    (
                        self.platepar.RA_d, 
                        self.platepar.dec_d, 
                        self.platepar.rotation_from_horiz, 
                        self.lenses
                    ) = data

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

            

            # Force distortion centre to image centre
            elif event.key() == QtCore.Qt.Key_B:
                if self.platepar is not None:
                    # Use remapCoeffsForFlagChange to preserve distortion coefficients
                    new_value = not self.platepar.force_distortion_centre
                    self.platepar.remapCoeffsForFlagChange('force_distortion_centre', new_value)

                    self.tab.param_manager.updatePlatepar()
                    self.updateLeftLabels()
                    self.updateStars()
                    self.updateDistortion()
                    self.tab.param_manager.onIndexChanged()


            # Toggle equal aspect ratio for radial distortions
            elif event.key() == QtCore.Qt.Key_G:

                if self.platepar is not None:
                    # Use remapCoeffsForFlagChange to preserve distortion coefficients
                    new_value = not self.platepar.equal_aspect
                    self.platepar.remapCoeffsForFlagChange('equal_aspect', new_value)

                    self.tab.param_manager.updatePlatepar()
                    self.updateLeftLabels()
                    self.updateStars()
                    self.updateDistortion()
                    self.tab.param_manager.onIndexChanged()


            # Toggle asymmetry correction for radial distortions
            elif event.key() == QtCore.Qt.Key_Y:

                if self.platepar is not None:
                    # Use remapCoeffsForFlagChange to preserve distortion coefficients
                    new_value = not self.platepar.asymmetry_corr
                    self.platepar.remapCoeffsForFlagChange('asymmetry_corr', new_value)

                    self.tab.param_manager.updatePlatepar()
                    self.updateLeftLabels()
                    self.updateStars()
                    self.updateDistortion()
                    self.tab.param_manager.onIndexChanged()


            # Get initial parameters from astrometry.net (same as Auto Fit button)
            elif (event.key() == QtCore.Qt.Key_X) and ((modifiers == QtCore.Qt.ControlModifier) \
                or (modifiers == (QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier))):

                # Use the same auto-fit path as the button (includes catalog balancing, quick alignment)
                self.autoFitAstrometryNet()

            # Test quick align with config mag limit (Shift+Q) - simulates CheckFit/ApplyRecalibrate
            elif (event.key() == QtCore.Qt.Key_Q) and (modifiers == QtCore.Qt.ShiftModifier):
                self.testQuickAlignWithConfigMagLimit()


            # Toggle showing detected stars
            elif event.key() == QtCore.Qt.Key_C:
                self.toggleShowCalStars()
                self.tab.settings.updateShowCalStars()
                # updates image automatically


            # Save the point to the matched stars list by pressing Enter or Space or to the
            # unsuitable stars

            elif (event.key() == QtCore.Qt.Key_Return) or (event.key() == QtCore.Qt.Key_Enter) \
                or (event.key() == QtCore.Qt.Key_Space):

                if self.star_pick_mode:
                    
                    # Check if the star has been skipped
                    unsuitable = False
                    if modifiers == QtCore.Qt.ControlModifier:
                        
                        # If a star has been skipped, mark it as unsuitable
                        if (self.old_autopan_x is not None) and (self.old_autopan_y is not None):
                            
                            # Check that a new star has been selected
                            if (self.old_autopan_x != self.current_autopan_x) or \
                                (self.old_autopan_y != self.current_autopan_y):
                                
                                unsuitable = True
                        
                        elif (self.current_autopan_x is None) and (self.current_autopan_y is None):
                            unsuitable = False

                        else:
                            unsuitable = True
                    
                    if unsuitable:

                        print("Unsuitable star at coordinates: ({}, {})".format(self.current_autopan_x, self.current_autopan_y))

                        self.unsuitable_stars.addPair(self.current_autopan_x, self.current_autopan_y,
                                                        0, 0, None)
                        self.updateBottomLabel()
                        self.unsuitable_star_markers.addPoints(x=[self.current_autopan_x],
                                                                y=[self.current_autopan_y])
                        self.unsuitable_star_markers2.addPoints(x=[self.current_autopan_x],
                                                                y=[self.current_autopan_y])

                    # If the catalog star or geo points has been selected, save the pair to the list
                    if self.cursor.mode == 1:

                        # Star catalog points
                        if self.closest_type == 'catalog':
                            selected_coords = self.catalog_stars_filtered[self.closest_cat_star_indx]
                            self.closest_cat_star_indx = None

                            # Init a catalog star pair object
                            pair_obj = CatalogStar(*selected_coords)

                        # Geo coordinates of the selected points
                        else:
                            selected_coords = self.geo_points[self.closest_geo_point_indx]

                            # Set a fixed value for star intensity and SNR
                            self.star_fwhm = 1.0
                            self.star_intensity = 10.0
                            self.star_snr = 1.0
                            self.star_saturated = False

                            # Init a geo point pair object
                            pair_obj = GeoPoint(self.geo_points_obj, self.closest_geo_point_indx)

                            self.closest_geo_point_indx = None


                        # Add the image/catalog pair to the list
                        if not unsuitable:
                            self.paired_stars.addPair(self.x_centroid, self.y_centroid, self.star_fwhm,
                                    self.star_intensity, pair_obj, 
                                    snr=self.star_snr, saturated=self.star_saturated)

                        # Switch back to centroiding mode
                        self.cursor.setMode(0)
                        self.updatePairedStars()

                        if self.autopan_mode:

                            self.updateBottomLabel()

                            self.jumpNextStar(miss_this_one=False)


                    else:

                        # Jump to next star if CTRL + SPACE is pressed
                        if modifiers == QtCore.Qt.ControlModifier:
                            print("Jumping to the next star")
                            self.jumpNextStar(miss_this_one=False)


            elif event.key() == QtCore.Qt.Key_Escape:
                if self.star_pick_mode:
                    
                    # If the ESC is pressed when the star has been centroided, reset the centroid
                    self.resetStarPick()
                    self.updatePairedStars()

            # Show the photometry plot
            elif event.key() == QtCore.Qt.Key_P and not modifiers == QtCore.Qt.ControlModifier:
                self.photometry(show_plot=True)

            # Show astrometry residuals plot
            elif event.key() == QtCore.Qt.Key_L:
                if self.star_pick_mode:
                    self.showAstrometryFitPlots()


        # Handle key presses in the manual reduction mode
        elif self.mode == 'manualreduction':

            # Set photometry mode
            if (qmodifiers & QtCore.Qt.ShiftModifier):
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

            # Launch ASTRA GUI
            elif (event.key() == QtCore.Qt.Key_K) and (modifiers == QtCore.Qt.ControlModifier):
                
                if ASTRA_IMPORTED:
                    
                    # If the ASTRA dialog does not exist, create it
                    if not hasattr(self, "astra_dialog") or self.astra_dialog is None:
                        
                        self.openAstraGUI()
                        self.astra_dialog.finished.connect(self.clearAstraDialogReference)
                        
                        if self.astra_config_params is not None:
                            self.astra_dialog.setConfig(self.astra_config_params)
                    
                    # If the dialog already exists, close it
                    else:
                        self.astra_config_params = self.astra_dialog.getConfig()
                        self.astra_dialog.close()
                        self.clearAstraDialogReference()

                else:
                    qmessagebox(title="ASTRA is not available",
                    message='ASTRA was not correctly imported, check pyswarms is installed.' '' \
                    'Aborted open ASTRA GUI process.',
                    message_type='error')

    def clearAstraDialogReference(self):
        """Clears the reference to the ASTRA GUI"""

        self.astra_dialog = None

    def clearAllPicks(self):
        """Clears the pick_list, and updates the GUI."""

        # Clear all picks
        self.pick_list = {}

        # Update GUI to remove drawn picks
        self.updatePicks()

    def openAstraGUI(self):
        """Opens the ASTRA dialog box on another thread to ensure SkyFit2 responsiveness."""

        # Launch the ASTRA Dialog on a seperate thread, with callbacks for buttons
        self.astra_dialog = launchASTRAGUI(
            run_astra_callback=None,
            run_kalman_callback=self.runKalmanFromConfig,
            run_load_callback=self.loadPicksFromFile,
            skyfit_instance=self
        )

        if self.astra_config_params is not None:
            self.astra_dialog.setConfig(self.astra_config_params)

        self.checkASTRACanRun()
        self.checkPickRevertCanRun()
        self.checkKalmanCanRun()
    
    def loadPicksFromFile(self, config):
        """Loads picks and associated values from either ECSV or DetApp txt file.
        args: 
            config (dict) : Config returned from the ASTRA GUI
        """

        # Unpack file path
        file_path = config["file_path"]

        # Check if the file path exists
        if not os.path.exists(file_path):
            print(f'ERROR: No Valid ECSV or .txt file selected')
            return
        
        # DetApp picks in a ev*.txt files
        if os.path.basename(file_path).startswith('ev') and os.path.basename(file_path).endswith('.txt'):
            
            # Create a temp pick_list
            pick_list = {}

            # Load all picks and indices from loadEvTxt
            pick_list = self.loadEvTxt(file_path)

        # Open ECSV file (even if it doesn't have an .ecsv extension)
        else:

            # Create a temp pick_list
            pick_list = {}

            # Load the picks and indices from loadECSV
            try:
                pick_list = self.loadECSV(file_path)

            except Exception as e:
                
                print(f'ERROR: Failed to load ECSV file {file_path}: {e}')
                qmessagebox(
                    title='ECSV Load Error',
                    message=f'Failed to load ECSV file {file_path}: {e}',
                    message_type="error"
                )
                return


        # Check if the returned values from load are None
        if pick_list is None:
            return # Warning was raised in loadEvTxt/loadECSV

        # Check if the pick list is empty
        if pick_list == {}:
            qmessagebox(
                title='Pick Load Error',
                message=f'No picks loaded from {file_path}',
                message_type="error"
            )
            return

        # Add previous picks to list, if the pick is non-empty
        if np.array(
            sorted(
            k for k in self.pick_list.keys()
            if (
                self.pick_list[k].get('x_centroid') is not None
                or self.pick_list[k].get('y_centroid') is not None
            )
            ),
            dtype=int
        ).size > 0:
            # Deep copy to avoid storing references to mutable objects
            self.previous_picks.append(copy.deepcopy(self.pick_list))

        # Update the main pick list & GUI
        self.clearAllPicks()
        self.pick_list = pick_list
        self.updateGreatCircle()

        # Print out added centroids
        for k in pick_list.keys():
            print(f'Added centroid at ({pick_list[k]["x_centroid"]},'
                  f' {pick_list[k]["y_centroid"]}) on frame {k}')

        # Finally update the GUI picks
        self.updatePicks()

        # Update Kalman/ASTRA Ready status if instance exists
        if hasattr(self, 'astra_dialog') and self.astra_dialog is not None:
            self.checkASTRACanRun()
            self.checkKalmanCanRun()
            self.checkPickRevertCanRun()

        # Send console and GUI updates
        print(f'Loaded {len(pick_list.keys())} picks from {file_path}!')

    def checkKalmanCanRun(self):
        """Checks if kalman filter can be run, updates astra GUI"""
        
        # Unpack picks with non-None values
        keys = np.array(
            sorted(
            k for k in self.pick_list.keys()
            if (
                self.pick_list[k].get('x_centroid') is not None
                or self.pick_list[k].get('y_centroid') is not None
            )
            ),
            dtype=int
        )

        # Enough points for a good run
        if keys.size >= 5:  
            tt = 'Ready to run.'
            self.astra_dialog.setKalmanStatus(True, tt)

        # Minimum amount of points to run
        elif keys.size >= 3:
            tt = 'Ready to run. WARNING: Kalman should be generally run with at least 5 points.'
            self.astra_dialog.setKalmanStatus("WARN", tt)

        # Not enough points to run
        else:
            tt = 'Not ready. Kalman requires at least 3 points (generally >= 5 points).'
            self.astra_dialog.setKalmanStatus(False, tt)

    def reverseASTRAPicks(self):
        """Reverts the ASTRA picks to the previous state of picks."""
        
        # Only revert if picks are non-empty
        if not self.previous_picks == []:

            # Pop newest pick off stack
            old_picks = self.previous_picks.pop()

            # Clear old picks
            self.clearAllPicks()

            # Restore old picks
            for frame_num in old_picks.keys():
                self.pick_list[frame_num] = old_picks[frame_num]

            # Update picks in GUI
            self.updateGreatCircle()
            self.updatePicks()

            # Print out reverted picks confirmation
            print(f'Reverted picks to previous state with {len(old_picks.keys())} picks.')
        
        # Update if picks can be reverted again
        self.checkASTRACanRun()
        self.checkKalmanCanRun()
        self.checkPickRevertCanRun()
            
    def checkPickRevertCanRun(self):
        """Checks if reverseASTRAPicks() can be run, connected to ASTRA GUI"""

        if self.previous_picks == []:
            # If previous picks are empty, set revert status to False
            self.astra_dialog.setRevertStatus(False)
        else:
            # If previous picks are not empty, set revert status to True
            self.astra_dialog.setRevertStatus(True)

    def checkASTRACanRun(self):
        """Checks if ASTRA can be run, updates astra GUI"""

        # Instantiate boolean vars
        middle_points_bool = False
        ending_points_bool = False
        middle_includes_ends_bool = False
        middle_includes_both_ends_bool = False

        # Only include keys where x_centroid or y_centroid is not None
        keys = np.array(
            sorted(
            k for k in self.pick_list.keys()
            if (
                self.pick_list[k].get('x_centroid') is not None
                or self.pick_list[k].get('y_centroid') is not None
            )
            ),
            dtype=int
        )

        if keys.size >= 3:

            # True where [k[i], k[i+1], k[i+2]] are consecutive integers
            triples = (keys[:-2] + 1 == keys[1:-1]) & (keys[:-2] + 2 == keys[2:])

            # indices i that start a consecutive triple
            i_triples = np.where(triples)[0]

            # exclude triples that touch the first or last key in the whole list
            # i == 0 uses the very first key; i+2 == len(keys)-1 uses the very last
            inner = i_triples[(i_triples > 0) & (i_triples + 2 < len(keys) - 1)]

            middle_points_bool = inner.size > 0

            middle_includes_ends_bool = i_triples.size > 0

            if keys.size == 3 and middle_includes_ends_bool:
                middle_includes_both_ends_bool = True

        else:
            middle_points_bool = False
            middle_includes_ends_bool = False

        # True if there are ending points which are not part of any sequence of picks (distinct)
        ending_points_bool = True if (len(keys) >= 2 and keys[0] + 1 != keys[-1]) else False

        # If there are only three points
        if middle_includes_both_ends_bool:
            tt = "Ready to run. WARNING: Three middle picks includes both endpoints, " \
            "ASTRA will only process the three points"
            self.astra_dialog.setASTRAStatus('WARN', tt)

        # If there are not enough consecutive points
        elif middle_points_bool:
            # Picks include three in middle, and two endpoints
            tt = "Ready to run, there are at least three consecutive points and two endpoints"
            self.astra_dialog.setASTRAStatus(True, tt)
        
        # If the three middle points includes one, but not both endpoints
        elif middle_includes_ends_bool:
            # Picks include three in the middle, which are part of the endpoints
            tt = "Ready to run. WARNING: Three consecutive middle points includes end-points - " \
            "ASTRA will stop at either beginning/end of middle points."
            self.astra_dialog.setASTRAStatus('WARN', tt)
        
        else:
            # If there are only distinct ending points selected, and no consequitive middle points
            if ending_points_bool:
                tt = "Not ready. End points selected, please pick three consecutive frames at high-SNR."
                self.astra_dialog.setASTRAStatus(False, tt)
            # If there are no picks, or no messages triggered (not ready)
            else:
                tt = "Not ready. Select three consecutive frames at high SNR, " \
                "and the start/end frames of the streak"
                self.astra_dialog.setASTRAStatus(False, tt)

    def prepareASTRAData(self, astra_config):
        """Prepares data from SkyFit2 class vars for ASTRA"""

        print("Running ASTRA with:", astra_config)

        # Sort pick list according to keys
        self.pick_list = dict(sorted(self.pick_list.items()))

        # Load the keys for picks which are not empty (not just photometry picks)
        pick_frame_indices = np.array(
            [key for key in self.pick_list.keys()
             if self.pick_list[key]['x_centroid'] is not None 
             and self.pick_list[key]['y_centroid'] is not None],
            dtype=int
        )

        # Prepare pick_dict (keys -> frame indexes, values include x,y centroid)

        pick_dict = {i : {
            "x_centroid" : self.pick_list[i]['x_centroid'],
            "y_centroid" : self.pick_list[i]['y_centroid']
                } for i in pick_frame_indices}


        # Prepare the flat to be passed to ASTRA
        flat = None
        if self.flat_struct is not None:
            flat = copy.deepcopy(self.flat_struct)
            flat.flat_img = flat.flat_img.T

        # Prepare the dark to be passed to ASTRA
        dark = None
        if self.dark is not None:
            dark = copy.deepcopy(self.dark)
            dark = dark.T

        # Package data for ASTRA - import later using dict comprehension
        data_dict = {
            "img_obj" : self.img_handle,
            "pick_dict" : pick_dict,
            "astra_config" : astra_config,
            "data_path" : self.dir_path,
            "config" : self.config,
            "dark" : dark,
            "flat" : flat
        }

        return data_dict


    def integrateASTRAResults(self, astra):
        """Integrates ASTRA results into the SkyFit2 instance."""

        # Add previous picks to list
        if np.array(
            sorted(
            k for k in self.pick_list.keys()
            if (
                self.pick_list[k].get('x_centroid') is not None
                or self.pick_list[k].get('y_centroid') is not None
            )
            ),
            dtype=int
        ).size > 0:
            # Deep copy to avoid storing references to mutable objects
            self.previous_picks.append(copy.deepcopy(self.pick_list))

        # Add ASTRA picks to the pick list
        self.clearAllPicks()  # Clear previous picks

        # Get and set pick_list from ASTRA
        self.pick_list = astra.getResults(skyfit_format=True)

        # Print added message
        for pick_frame, pick in self.pick_list.items():
            print(f'Added centroid at ({pick["x_centroid"]}, {pick["y_centroid"]}) '
                  f'on frame {pick_frame}')

        # Update picks on GUI and update greatCircle
        self.updateGreatCircle()
        self.updatePicks()

        # Update Kalman/ASTRA Ready status if instance exists
        if hasattr(self, 'astra_dialog') and self.astra_dialog is not None:
            self.checkASTRACanRun()
            self.checkKalmanCanRun()
            self.checkPickRevertCanRun()

        # Print message telling ASTRA has been run
        print(f'Loaded {astra.getTotalPicks()} Picks from ASTRA! '
              f'Minimum SNR of {astra.getMinSnr()}')

    def setMessageBox(self, title, message, type):
        """Target function for ASTRA_GUI to set message boxes."""
        qmessagebox(title=title, message=message, message_type=type)

    def runKalmanFromConfig(self, astra_config, progress_callback=None):
        """Runs the Kalman filter with the given ASTRA configuration."""

        print("Running Kalman with:", astra_config)

        if progress_callback is not None:
            progress_callback(0)

        # Take only keys of non-empty picks (not just photometry), maintaining order
        ordered_keys = [key for key, _ in sorted(self.pick_list.items())]
        pick_frame_indices = [
            key for key in ordered_keys
            if self.pick_list[key]['x_centroid'] is not None
            and self.pick_list[key]['y_centroid'] is not None
        ]

        # Prepare measurements and times for Kalman filter
        measurements = [
            (self.pick_list[key]['x_centroid'], self.pick_list[key]['y_centroid'])
            for key in pick_frame_indices
        ]
        times = [self.img_handle.currentFrameTime(key, dt_obj=True) for key in pick_frame_indices]

        # Extract kalman settings from astra_config
        sigma_xy = astra_config['kalman']['sigma_xy (px)']
        perc_sigma_vxy = astra_config['kalman']['sigma_vxy (%)']
        monotonicity = astra_config['kalman']['Monotonicity']
        save_stats_results = astra_config['kalman']['save results']

        # Run Kalman on the ASTRA instance, extract new picks
        kalman = KalmanFilter(
            sigma_xy=sigma_xy,
            perc_sigma_vxy=perc_sigma_vxy,
            measurements=measurements,
            times=times,
            monotonicity=monotonicity,
            save_stats_results=save_stats_results,
            save_path=self.dir_path
        )

        xypicks = kalman.getPicks()

        if progress_callback is not None:
            progress_callback(100)

        return {
            'frame_indices': pick_frame_indices,
            'xypicks': xypicks.tolist(),
            'save_results': save_stats_results,
        }

    def applyKalmanResults(self, kalman_result):
        """Integrates Kalman smoothing results into the SkyFit2 instance."""

        if not kalman_result:
            return

        frame_indices = kalman_result.get('frame_indices', [])
        xypicks = kalman_result.get('xypicks', [])
        save_results = kalman_result.get('save_results', 'false')

        # Add previous picks to list
        if np.array(
            sorted(
                k for k in self.pick_list.keys()
                if (
                    self.pick_list[k].get('x_centroid') is not None
                    or self.pick_list[k].get('y_centroid') is not None
                )
            ),
            dtype=int
        ).size > 0:
            # Deep copy to avoid storing references to mutable objects
            self.previous_picks.append(copy.deepcopy(self.pick_list))

        # Print message to show Kalman has been applied
        print(f'Kalman filter applied to {len(xypicks)} picks!')

        # Adjust all picks positions
        for frame_number, pick in zip(frame_indices, xypicks):
            frame_number = int(frame_number)
            if frame_number in self.pick_list:
                self.pick_list[frame_number]["x_centroid"] = pick[0]
                self.pick_list[frame_number]["y_centroid"] = pick[1]

                # Print adjustment message
                print(f'Adjusted centroid at ({pick[0]}, {pick[1]}) on frame {frame_number}')

        # Update picks on GUI and greatCircle
        self.updateGreatCircle()
        self.updatePicks()

        # Update kalman/astra readiness if instance exists
        if hasattr(self, 'astra_dialog') and self.astra_dialog is not None:
            self.checkASTRACanRun()
            self.checkKalmanCanRun()
            self.checkPickRevertCanRun()

        # Print message showing kalman finished
        if str(save_results).lower() == 'true':
            print(f'Saved to CSV & Loaded {len(frame_indices)} Picks from Kalman Smoothing.')
        else:
            print(f'Loaded {len(frame_indices)} Picks from Kalman Smoothing!')

    def loadEvTxt(self, txt_file_path):
        """
        Loads the Ev*.txt file and adds the relevant info to pick_list
        Args:
            txt_file_path (str): Path to the Ev*.txt file to load
        Returns:
            picks (dict): (N : [8]) dict following same format as self.pick_list 
        """

        picks = [] # (N, x, y) array of picks
        pick_frame_indices = [] # (N,) array of frame indices

        # Temp bool to extract header line
        first_bool = False

        # Wrap in try except for uncaught errors
        try:
            # Opens and parses pick file
            with open(txt_file_path, 'r') as file:
                for i, line in enumerate(file):
                    if line.strip() and not line.startswith('#'):

                        # Clean the line
                        line = [part.strip() for part in line.split() if part]

                        # Unpack the header
                        if first_bool == False:
                            first_bool = True

                            # Unpack header info
                            column_names = [part.strip() for part in temp_line.split()[1:] if part]

                            # Extract indices
                            frame_number_idx = column_names.index('fr') if 'fr' in column_names else None
                            x_centroid_idx = column_names.index('cx') if 'cx' in column_names else None
                            y_centroid_idx = column_names.index('cy') if 'cy' in column_names else None

                            # Raise error if not found in txt file
                            if frame_number_idx is None or x_centroid_idx is None or y_centroid_idx is None:
                                qmessagebox(title="TXT File Format Error", 
                                            message="TXT file must contain 'fr', 'cx', and 'cy' columns.",
                                            message_type="error")
                                return None
                        
                        else:
                            # Unpack the pick
                            frame_number = int(line[frame_number_idx])
                            cx, cy = float(line[x_centroid_idx]), float(line[y_centroid_idx])

                            picks.append({
                                'x_centroid': cx,
                                'y_centroid': cy,
                                'mode' : 1,
                                'intensity_sum' : 1,
                                'photometry_pixels' : None,
                                'background_intensity' : 0,
                                'snr' : 1,
                                'saturated' : False,
                            })
                            pick_frame_indices.append(frame_number)

                    # Store a temp line to hit previous for col names
                    temp_line = line

            # Pack picks in self.pick_list format
            pick_list = {frame: pick for frame, pick in zip(pick_frame_indices, picks)}

            return pick_list

        # Raise error message
        except Exception as e:
            qmessagebox(title="File Read Error", 
                        message=f"Unknown Error reading TXT file, check correct file loaded.: {str(e)}",
                        message_type="error")
            return None

    def loadECSV(self, ECSV_file_path):
        """
        Loads the ECSV file and adds the relevant info to pick_list
        Args:
            ECSV_file_path (str): Path to the ECSV file to load
        Returns:
            picks (dict): (N : [8]) dict following same format as self.pick_list 
        """

        # Instantiate arrays to be populated
        picks = []  # N x args_dict array for addCentroid
        pick_frame_indices = []  # (N,) array of frame indices
        pick_frame_times = []  # (N,) array of frame times

        # wrap in try except to catch unknown errors
        try:
            # Opens and parses the ECSV file
            with open(ECSV_file_path, 'r') as file:

                # Read the file contents
                contents = file.readlines()
        
                # Temp bool to get the column names
                first_bool = False

                # Process the contents
                for line in contents:

                    # Clean the line
                    line = [part.strip() for part in line.split(',') if part]

                    # Skip header lines
                    if line[0].startswith('#'):
                        continue

                    if first_bool == False:
                        # Set first bool to True so header is not unpacked twice
                        first_bool = True

                        # Unpack column names
                        column_names = line

                        # Map column names to their indices
                        pick_frame_times_idx = column_names.index('datetime') if 'datetime' in column_names \
                                                                                        else None
                        x_ind = column_names.index('x_image') if 'x_image' in column_names \
                                                                                        else None
                        y_ind = column_names.index('y_image') if 'y_image' in column_names \
                                                                                        else None
                        background_ind = column_names.index('background_pixel_value') \
                                                        if 'background_pixel_value' in column_names else None
                        sat_ind = column_names.index('saturated_pixels') if 'saturated_pixels' in column_names \
                                                                                        else None
                        snr_ind = column_names.index('snr') if 'snr' in column_names else None

                        # Ensure essential values are in the ECSV
                        if x_ind is None or y_ind is None or pick_frame_times_idx is None:
                            qmessagebox(title="ECSV File Format Error", 
                                        message="ECSV file must contain 'x_image', 'y_image', "
                                        "and 'datetime' columns.",
                                        message_type="error")
                            return None

                        continue

                    else:
                        # Unpack line

                        # Populate arrays
                        cx, cy = float(line[x_ind]), float(line[y_ind])
                        background = float(line[background_ind]) if background_ind is not None else 0
                        saturated = bool(line[sat_ind]) if sat_ind is not None else False
                        snr = float(line[snr_ind]) if snr_ind is not None else 1
                        pick_frame_times.append(datetime.datetime.strptime(line[pick_frame_times_idx], 
                                                                           '%Y-%m-%dT%H:%M:%S.%f'))

                        # Load in pick parameters, use default for other values 
                        picks.append({
                            'x_centroid': cx,
                            'y_centroid': cy,
                            'mode': 1,
                            'intensity_sum': 1,
                            'photometry_pixels': None,
                            'background_intensity': background,
                            'snr': snr,
                            'saturated': saturated,
                        })

            # Converts times into frame indices, accounting for floating-point errors
            pick_frame_indices = []
            frame_count = self.img_handle.total_frames
            time_idx = 0
            for i in range(frame_count):
                frame_time = self.img_handle.currentFrameTime(frame_no=i, dt_obj=True)
                time = pick_frame_times[time_idx]
                if frame_time == time or \
                    frame_time == time + datetime.timedelta(microseconds=1) or \
                    frame_time == time - datetime.timedelta(microseconds=1):
                    pick_frame_indices.append(i)
                    time_idx += 1
                if time_idx >= len(pick_frame_times):
                    break

            # if arrays are different dimensions raise error
            if len(picks) != len(pick_frame_indices):
                qmessagebox(title="Pick/Frame Index Mismatch",
                            message="Mismatch between number of picks and frame indices. " \
                            "Please check the ECSV file for frame-time mismatch errors.",
                            message_type="error")
                return None
            pick_list = {frame: pick for frame, pick in zip(pick_frame_indices, picks)}

            return pick_list
        
        # Raise error box
        except Exception as e:
            qmessagebox(title='File Read Error',
                        message=f"Unknown Error reading ECSV file, check correct file loaded.: {str(e)}",
                        message_type="error")
            return None


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


    def wheelEvent(self, event, axis=None):
        """ Change star selector aperture on scroll. """
        
        # Read the wheel direction
        try:
            delta = event.delta()
        except AttributeError:
            delta = event.angleDelta().y()   

        modifier = QtWidgets.QApplication.keyboardModifiers()

        # Handle scroll events
        if self.img_frame.sceneBoundingRect().contains(event.pos()):

            # If control is pressed in star picking mode, change the size of the aperture
            if (modifier & QtCore.Qt.ControlModifier) and self.star_pick_mode:

                # Increase aperture size
                if delta < 0:
                    self.scrolls_back = 0
                    self.star_aperture_radius += 0.5
                    self.cursor.setRadius(self.star_aperture_radius)
                    self.cursor2.setRadius(self.star_aperture_radius)

                # Decrease aperture size
                elif delta > 0 and self.star_aperture_radius > 1:
                    self.scrolls_back = 0
                    self.star_aperture_radius -= 0.5
                    self.cursor.setRadius(self.star_aperture_radius)
                    self.cursor2.setRadius(self.star_aperture_radius)

            else:

                # Unzoom the image
                if delta < 0:

                    # Track number of scroll backs in the star picking mode so it resets the image after
                    #   multiple zoomouts
                    if self.star_pick_mode:
                        self.scrolls_back += 1
                    else:
                        self.scrolls_back = 0

                    # Reset the zoom if scrolled back multiple times
                    if self.scrolls_back > 1:
                        self.img_frame.autoRange(padding=0)

                    else:
                        self.img_frame.wheelEventModified(event, axis)

                # Zoom in the image
                elif delta > 0:
                    self.scrolls_back = 0
                    self.img_frame.wheelEventModified(event, axis)



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
            self.star_fwhm = None
            self.snr_centroid = None
            self.saturated_centroid = False
            self.star_intensity = None
            self.star_snr = None
            self.star_saturated = False
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
                self.v_zoom.move(QtCore.QPoint(int(self.label1.boundingRect().width()), 0))
            else:
                self.v_zoom.move(QtCore.QPoint(0, 0))

        if (self.show_key_help == 1) and self.star_pick_mode:
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

    def updateConstellations(self):
        """ Projects and draws constellation lines. """

        if self.constellation_data is None:
            return

        # Get current JD time
        jd = date2JD(*self.img_handle.currentTime())

        # Unpack constellation data
        from_ra, from_dec = self.constellation_data[:, 0], self.constellation_data[:, 1]
        to_ra, to_dec = self.constellation_data[:, 2], self.constellation_data[:, 3]
        
        # 0. Generate FOV polygon
        w = self.platepar.X_res
        h = self.platepar.Y_res
        fov_poly = []

        if self.fov_poly_cache is not None and self.fov_poly_jd == jd:
            fov_poly = self.fov_poly_cache
        else:
            # Define edges: (x1, y1) -> (x2, y2)
            edges = [
                ((0, 0), (w, 0)),   # Top
                ((w, 0), (w, h)),   # Right
                ((w, h), (0, h)),   # Bottom
                ((0, h), (0, 0))    # Left
            ]
            
            samples_per_side = 10
            
            try:
                for (x_start, y_start), (x_end, y_end) in edges:
                    xs = np.linspace(x_start, x_end, samples_per_side, endpoint=False)
                    ys = np.linspace(y_start, y_end, samples_per_side, endpoint=False)
                    
                    # Prepare inputs
                    n = len(xs)
                    jd_arr = [jd]*n
                    level_arr = [1]*n
                    
                    _, r_arr, d_arr, _ = xyToRaDecPP(jd_arr, xs, ys, level_arr, self.platepar, jd_time=True, extinction_correction=False)
                    
                    for r, d in zip(r_arr, d_arr):
                        fov_poly.append((r, d))
                        
                # Update cache
                self.fov_poly_cache = fov_poly
                self.fov_poly_jd = jd
                
            except Exception as e:
                print(f"Error computing FOV polygon for constellations: {e}")
                return

        # 1. Filter using sphericalPolygonCheck
        # Check start and end points
        # If either is inside, keep the line
        test_points_from = np.c_[from_ra, from_dec]
        test_points_to = np.c_[to_ra, to_dec]
        
        mask_from = np.array(sphericalPolygonCheck(fov_poly, test_points_from))
        mask_to = np.array(sphericalPolygonCheck(fov_poly, test_points_to))
        
        # Keep line if EITHER endpoint is inside
        mask = mask_from | mask_to

        if np.any(mask):
            from_x, from_y = raDecToXYPP(from_ra[mask], from_dec[mask], jd, self.platepar)
            to_x, to_y = raDecToXYPP(to_ra[mask], to_dec[mask], jd, self.platepar)

            # 2. Filter based on image bounds (keep as redundant safety or remove?)
            # Spherical check is accurate, but projection might still yield points slightly outside
            # which is fine. The previous bounds check was mostly for the angular filter artifacts.
            # We can relax it or remove it, but let's keep a loose one just in case.
            
            margin = 200 # Larger margin to allow lines entering from outside
            in_bounds_from = (from_x > -margin) & (from_x < w + margin) & (from_y > -margin) & (from_y < h + margin)
            in_bounds_to = (to_x > -margin) & (to_x < w + margin) & (to_y > -margin) & (to_y < h + margin)
            mask_bounds = in_bounds_from | in_bounds_to
            
            from_x = from_x[mask_bounds]
            from_y = from_y[mask_bounds]
            to_x = to_x[mask_bounds]
            to_y = to_y[mask_bounds]

            if len(from_x) > 0:
                # Interleave start and end points
                pts_x = np.empty((from_x.size + to_x.size,), dtype=from_x.dtype)
                pts_x[0::2] = from_x
                pts_x[1::2] = to_x

                pts_y = np.empty((from_y.size + to_y.size,), dtype=from_y.dtype)
                pts_y[0::2] = from_y
                pts_y[1::2] = to_y
                
                self.constellation_lines_bg.setData(pts_x, pts_y)
                self.constellation_lines_bg.show()

                self.constellation_lines_fg.setData(pts_x, pts_y)
                self.constellation_lines_fg.show()
            else:
                self.constellation_lines_bg.hide()
                self.constellation_lines_fg.hide()
            
        else:
            self.constellation_lines_bg.hide()
            self.constellation_lines_fg.hide()


    def toggleShowCatStars(self):
        """ Toggle showing/hiding catalog stars. """
        self.catalog_stars_visible = not self.catalog_stars_visible

        # Hide markers if disabling
        if not self.catalog_stars_visible:
            self.cat_star_markers.hide()
            self.cat_star_markers2.hide()
            self.geo_markers.hide()
            self.geo_markers.hide()
            self.spectral_type_text_list.hide()

        else:
            self.cat_star_markers.show()
            self.cat_star_markers2.show()
            self.geo_markers.show()
            self.geo_markers.show()
            if self.show_spectral_type or self.show_star_names:
                self.spectral_type_text_list.show()
        
        # Redraw
        self.updateStars()

        # Update the checkbox
        self.tab.settings.updateShowCatStars()

    def toggleShowSpectralType(self):
        """ Toggle showing/hiding spectral types. """
        self.show_spectral_type = not self.show_spectral_type

        # Hide/show text items
        if self.show_spectral_type or self.show_star_names:
            if self.catalog_stars_visible:
                self.spectral_type_text_list.show()
        else:
            self.spectral_type_text_list.hide()

        # Redraw
        self.updateStars()

        # Update the checkbox
        self.tab.settings.updateShowSpectralType()

    def toggleShowStarNames(self):
        """ Toggle showing/hiding star names. """
        self.show_star_names = not self.show_star_names

        # Hide/show text items
        if self.show_spectral_type or self.show_star_names:
            if self.catalog_stars_visible:
                self.spectral_type_text_list.show()
        else:
            self.spectral_type_text_list.hide()

        # Redraw
        self.updateStars()

        # Update the checkbox
        self.tab.settings.updateShowStarNames()

    def toggleShowConstellations(self):
        """ Toggle showing/hiding constellation lines. """
        self.show_constellations = not self.show_constellations

        # Force update
        if self.show_constellations:
            self.updateConstellations()
        else:
            self.constellation_lines_bg.hide()
            self.constellation_lines_fg.hide()

        # Update the checkbox
        self.tab.settings.updateShowConstellations()

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

    def toggleShowAstrometryNetStars(self):
        """ Toggle whether to show astrometry.net matched stars """
        self.astrometry_stars_visible = not self.astrometry_stars_visible
        if self.astrometry_stars_visible:
            self.astrometry_matched_markers.show()
            self.astrometry_matched_markers2.show()
            self.astrometry_quad_markers.show()
            self.astrometry_quad_markers2.show()
            # Update the markers with current solution info
            self.updateAstrometryNetStarMarkers()
        else:
            self.astrometry_matched_markers.hide()
            self.astrometry_matched_markers2.hide()
            self.astrometry_quad_markers.hide()
            self.astrometry_quad_markers2.hide()

    def updateAstrometryNetStarMarkers(self):
        """ Update the astrometry.net star markers from stored solution info """
        if self.astrometry_solution_info is None:
            self.astrometry_matched_markers.setData(pos=[])
            self.astrometry_matched_markers2.setData(pos=[])
            self.astrometry_quad_markers.setData(pos=[])
            self.astrometry_quad_markers2.setData(pos=[])
            return

        matched_pairs = self.astrometry_solution_info.get('matched_pairs', [])
        quad_stars = self.astrometry_solution_info.get('quad_stars', [])

        # Update matched star markers (catalog positions of matched stars)
        if matched_pairs:
            x_matched = [p['catalog_x'] for p in matched_pairs]
            y_matched = [p['catalog_y'] for p in matched_pairs]
            self.astrometry_matched_markers.setData(x=x_matched, y=y_matched)
            self.astrometry_matched_markers2.setData(x=x_matched, y=y_matched)
        else:
            self.astrometry_matched_markers.setData(pos=[])
            self.astrometry_matched_markers2.setData(pos=[])

        # Update quad star markers
        if quad_stars:
            x_quad = [s['x_pix'] for s in quad_stars]
            y_quad = [s['y_pix'] for s in quad_stars]
            self.astrometry_quad_markers.setData(x=x_quad, y=y_quad)
            self.astrometry_quad_markers2.setData(x=x_quad, y=y_quad)
        else:
            self.astrometry_quad_markers.setData(pos=[])
            self.astrometry_quad_markers2.setData(pos=[])

    def toggleSatelliteTracks(self):
        """ Toggle whether to show satellite tracks """

        if not SKYFIELD_AVAILABLE:
            print("Skyfield not available - cannot show satellite tracks.")
            return

        self.show_sattracks = not self.show_sattracks
        print(f"Satellite tracks: {self.show_sattracks}")

        if self.show_sattracks:
            self.loadSatelliteTracks()
        else:
            # Will clear if self.show_sattracks is False
            self.drawSatelliteTracks()

    def toggleShowPicks(self):
        """ Toggle whether to show the picks for manualreduction """

        if self.pick_marker.isVisible():
            self.pick_marker.hide()
            self.pick_marker2.hide()
        else:
            self.pick_marker.show()
            self.pick_marker2.show()

    def toggleShowGreatCircle(self):
        """ Toggle the visibility of the great circle line. """
        
        if self.great_circle_line.isVisible():
            self.great_circle_line.hide()
        else:
            self.great_circle_line.show()


    def toggleShowRegion(self):
        """ Toggle whether to show the photometry region for manualreduction """
        if self.region.isVisible():
            self.region.hide()
            if hasattr(self, "region_zoom"):
                self.region_zoom.hide()
        else:
            self.region.show()
            if hasattr(self, "region_zoom"):
                self.region_zoom.show()

    def toggleDistortion(self):
        """ Toggle whether to show the distortion lines"""
        self.draw_distortion = not self.draw_distortion

        if self.draw_distortion:
            self.distortion_lines.show()
            self.updateDistortion()
        else:
            self.distortion_lines.hide()


    def toggleMeasGroundPoints(self):
        """ Toggle measuring points on the ground, vs on the sky. """
        self.meas_ground_points = not self.meas_ground_points

        self.updateMeasurementRefractionCorrection()


    def toggleInvertColours(self):
        self.img.invert()
        self.img_zoom.invert()

    def toggleAutoPan(self):

        self.img.autopan()
        self.autopan_mode = not self.autopan_mode


    def toggleSingleClickPhotometry(self):
        self.single_click_photometry = not self.single_click_photometry


    def updateMeasurementRefractionCorrection(self):

        # If the refraction is disabled and the group points are not measured, then correct the measured
        #   points on the sky for refraction (as it is not taken into account)
        # WARNING: This should not be used if the distortion model itself compensates for the refraction 
        #   (e.g. the polynomial model)
        self.platepar.measurement_apparent_to_true_refraction = (not self.platepar.refraction) \
            and (not self.meas_ground_points)


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

    def filterCatalogStarsInsideFOV(self, catalog_stars, remove_under_horizon=True, sort_declination=False):
        """ Take only catalogs stars which are inside the FOV.

        Arguments:
            catalog_stars: [ndarray] 2D array where entries are (ra, dec, mag). Note that the array needs
            to be sorted by descending declination!

        Keyword arguments:
            remove_under_horizon: [bool] Remove stars below the horizon (-5 deg below).
            sort_declination: [bool] Sort the stars by descending declination. Only needs to be done for geo
                points.
        """

        if sort_declination:
            
            # Sort by descending declination (needed for fast filtering)
            dec_sorted_ind = np.argsort(catalog_stars[:, 1])[::-1]
            catalog_stars = catalog_stars[dec_sorted_ind]


        # Get RA/Dec of the FOV centre
        ra_centre, dec_centre = self.computeCentreRADec()

        # Calculate the FOV radius in degrees
        fov_radius = getFOVSelectionRadius(self.platepar)

        # Compute the current Julian date
        jd = date2JD(*self.img_handle.currentTime())

        # Take only those stars which are inside the FOV
        filtered_indices, filtered_catalog_stars = subsetCatalog(catalog_stars, ra_centre, dec_centre, \
            jd, self.platepar.lat, self.platepar.lon, fov_radius, self.cat_lim_mag, \
            remove_under_horizon=remove_under_horizon)

        filtered_catalog_stars = np.array(filtered_catalog_stars)


        # Return original indexing if it was sorted by declination
        if sort_declination and len(filtered_indices):

            # Restore original indexing
            filtered_indices = dec_sorted_ind[filtered_indices]


        return filtered_indices, filtered_catalog_stars


    def tryQuickAlignment(self):
        """ Try to align platepar using existing pointing as starting point.

        Uses alignPlatepar() to refine pointing, then checks if the fit is good enough
        by counting matched stars. If successful, performs full NN-based astrometry fit.

        Returns:
            success: [bool] True if quick alignment succeeded and we can skip astrometry.net
        """
        print()
        print("Trying quick alignment with existing platepar...")
        self.status_bar.showMessage("Trying quick alignment...")
        QtWidgets.QApplication.processEvents()

        # Get current FF file name
        ff_name_c = convertFRNameToFF(self.img_handle.name())

        # Use override data if enabled and available, otherwise use original CALSTARS
        if self.star_detection_override_enabled and ff_name_c in self.star_detection_override_data:
            detected_stars = np.array(self.star_detection_override_data[ff_name_c])
        elif ff_name_c in self.calstars:
            detected_stars = np.array(self.calstars[ff_name_c])
        else:
            print("  No detected stars available - falling back to astrometry.net")
            return False

        detected_stars = np.array(detected_stars)
        if len(detected_stars) < 10:
            print("  Less than 10 detected stars - falling back to astrometry.net")
            return False

        # Get detected star coordinates (note: calstars format is [y, x, ...])
        # CALSTARS format: Y(0) X(1) IntensSum(2) Ampltd(3) FWHM(4) BgLvl(5) SNR(6) NSatPx(7)
        det_y = detected_stars[:, 0]
        det_x = detected_stars[:, 1]
        # Use index 2 (IntensSum/integrated intensity) not index 3 (Ampltd/peak amplitude)
        det_intens = detected_stars[:, 2] if detected_stars.shape[1] > 2 else np.ones(len(det_x))
        det_fwhm = detected_stars[:, 4] if detected_stars.shape[1] > 4 else np.zeros(len(det_x))
        det_snr = detected_stars[:, 6] if detected_stars.shape[1] > 6 else np.ones(len(det_x))
        det_saturated = detected_stars[:, 7] if detected_stars.shape[1] > 7 else np.zeros(len(det_x))

        # Prepare calstars_coords for alignPlatepar (x, y format)
        calstars_coords = np.column_stack([det_x, det_y])

        # Get time for the current frame
        calstars_time = list(self.img_handle.currentTime())

        # Compute JD
        jd = date2JD(*calstars_time)

        # Try alignPlatepar to refine pointing
        try:
            pp_aligned = alignPlatepar(self.config, self.platepar, calstars_time, calstars_coords)
        except Exception as e:
            print("  alignPlatepar failed: {} - falling back to astrometry.net".format(str(e)))
            return False

        # Check if alignPlatepar improved the fit by counting matched stars
        # Get catalog stars within FOV using the aligned platepar
        _, catalog_stars_fov = self.filterCatalogStarsInsideFOV(self.catalog_stars)

        if len(catalog_stars_fov) < 10:
            print("  Not enough catalog stars in FOV - falling back to astrometry.net")
            return False

        # Project catalog stars to image coordinates
        catalog_x, catalog_y, _ = getCatalogStarsImagePositions(catalog_stars_fov, jd, pp_aligned)

        # Count matches: detected stars with a catalog star within match_radius pixels
        match_radius = 10.0  # pixels
        n_matched = 0
        for i in range(len(det_x)):
            dist = np.sqrt((catalog_x - det_x[i])**2 + (catalog_y - det_y[i])**2)
            if np.min(dist) < match_radius:
                n_matched += 1

        match_fraction = n_matched / len(det_x) if len(det_x) > 0 else 0

        print("  Quick alignment: {}/{} stars matched ({:.1f}%) within {}px".format(
            n_matched, len(det_x), 100*match_fraction, match_radius))

        # Require at least 10 matched stars and >50% match rate
        min_matched_stars = 10
        min_match_fraction = 0.5

        if n_matched < min_matched_stars or match_fraction < min_match_fraction:
            print("  Insufficient matches - falling back to astrometry.net")
            return False

        # Quick alignment succeeded - apply the aligned platepar and do full NN fit
        print("  Quick alignment successful! Skipping astrometry.net")
        self.platepar = pp_aligned

        # Update JD and hour angle
        self.platepar.JD = jd
        self.platepar.Ho = JD2HourAngle(jd) % 360

        # Save user's distortion settings for final fit
        user_distortion_type = self.platepar.distortion_type
        user_equal_aspect = self.platepar.equal_aspect
        user_asymmetry_corr = self.platepar.asymmetry_corr
        user_force_distortion_centre = self.platepar.force_distortion_centre
        user_refraction = self.platepar.refraction
        user_fit_only_pointing = self.fit_only_pointing
        user_fixed_scale = self.fixed_scale

        # Use standard fitting settings for NN fit, but KEEP existing distortion params
        # The aligned platepar from alignPlatepar() has good pointing AND distortion coefficients
        # Setting reset_params=True would zero out distortion and cause RANSAC to diverge
        self.platepar.refraction = True
        self.platepar.equal_aspect = True
        self.platepar.asymmetry_corr = False
        self.platepar.force_distortion_centre = False
        self.platepar.setDistortionType("radial5-odd", reset_params=False)

        # Prepare detected stars array for NN fit
        img_stars_arr = np.column_stack([det_x, det_y, det_intens])

        # Perform full NN-based fit
        print()
        print("NN-based fitting...")
        print("  Starting from: RA={:.2f} Dec={:.2f} Scale={:.3f} arcmin/px".format(
            self.platepar.RA_d, self.platepar.dec_d, 60/self.platepar.F_scale))
        self.status_bar.showMessage("Fitting astrometry...")
        QtWidgets.QApplication.processEvents()

        try:
            ransac_result = self.platepar.fitAstrometry(
                jd, img_stars_arr, self.catalog_stars,
                first_platepar_fit=True,
                use_nn_cost=True
            )
            print("  NN fit complete: RA={:.2f} Dec={:.2f} Scale={:.3f} arcmin/px".format(
                self.platepar.RA_d, self.platepar.dec_d, 60/self.platepar.F_scale))
        except Exception as e:
            print("  NN fit failed: {} - falling back to astrometry.net".format(str(e)))
            return False

        # Populate paired_stars directly from RANSAC matched pairs
        self.paired_stars = PairedStars()

        if ransac_result is not None:
            img_stars_matched, catalog_matched = ransac_result
            for i in range(len(img_stars_matched)):
                img_x, img_y, img_intens = img_stars_matched[i]
                cat_star = catalog_matched[i]
                sky_obj = CatalogStar(cat_star[0], cat_star[1], cat_star[2])

                # Look up FWHM, SNR, and saturation from detected stars
                fwhm, snr, saturated = 2.5, 1.0, False
                if len(det_x) > 0:
                    distances = np.sqrt((det_x - img_x)**2 + (det_y - img_y)**2)
                    closest_idx = np.argmin(distances)
                    if distances[closest_idx] < 3.0:  # Within 3 pixels
                        fwhm = det_fwhm[closest_idx]
                        snr = det_snr[closest_idx]
                        saturated = det_saturated[closest_idx] > 0

                self.paired_stars.addPair(
                    img_x, img_y,
                    fwhm,
                    img_intens,
                    sky_obj,
                    snr=snr,
                    saturated=saturated
                )

        print("  Matched {} star pairs".format(len(self.paired_stars)))

        # Restore user's distortion settings for final refinement fit
        # Must restore flags BEFORE calling setDistortionType so poly_length is computed correctly
        self.platepar.equal_aspect = user_equal_aspect
        self.platepar.asymmetry_corr = user_asymmetry_corr
        self.platepar.force_distortion_centre = user_force_distortion_centre
        self.platepar.refraction = user_refraction
        self.fit_only_pointing = user_fit_only_pointing
        self.fixed_scale = user_fixed_scale
        # Now set distortion type which will adjust poly_length and pad coefficients
        # Use reset_params=False to preserve the fitted coefficients, just add zeros for new terms
        self.platepar.setDistortionType(user_distortion_type, reset_params=False)

        # Filter photometric outliers before final fit
        if len(self.paired_stars) >= 15:
            removed = self.filterPhotometricOutliers(sigma_threshold=2.5)
            if removed > 0:
                print("Pairs after photometric filtering: {}".format(len(self.paired_stars)))

        # Filter blended stars before final fit
        if len(self.paired_stars) >= 15:
            removed = self.filterBlendedStars(blend_radius_arcsec=30.0)
            if removed > 0:
                print("Pairs after blend filtering: {}".format(len(self.paired_stars)))

        # Filter high FWHM stars before final fit (remove top 10%)
        if len(self.paired_stars) >= 15:
            removed = self.filterHighFWHMStars(fraction=0.10)
            if removed > 0:
                print("Pairs after FWHM filtering: {}".format(len(self.paired_stars)))

        # Do a final fit with user's distortion settings
        if len(self.paired_stars) >= 10:
            print("Final refinement with user settings (distortion={})...".format(user_distortion_type))
            self.fitPickedStars()

        # Reset photometry fit residuals
        self.photom_fit_resids = None

        self.status_bar.showMessage("Quick alignment complete: {} stars".format(len(self.paired_stars)))

        return True


    def testQuickAlignWithConfigMagLimit(self):
        """ Test quick align using config.catalog_mag_limit (simulates CheckFit/ApplyRecalibrate).

        This is a diagnostic tool to test how NNalign performs with the same catalog
        that CheckFit and ApplyRecalibrate would use.
        """
        import sys

        print(flush=True)
        print("=" * 70, flush=True)
        print("TEST: Quick align with config.catalog_mag_limit = {:.1f}".format(
            self.config.catalog_mag_limit), flush=True)
        print("      (Current SkyFit2 cat_lim_mag = {:.1f})".format(self.cat_lim_mag), flush=True)
        print("=" * 70, flush=True)

        # Store original platepar values to verify it's not modified
        orig_ra = self.platepar.RA_d
        orig_dec = self.platepar.dec_d
        orig_rot = self.platepar.pos_angle_ref
        print("  BEFORE: RA={:.4f} Dec={:.4f} Rot={:.4f}".format(orig_ra, orig_dec, orig_rot), flush=True)

        # Get detected stars from current image (same as tryQuickAlignment)
        ff_name_c = self.img_handle.current_ff_file if hasattr(self.img_handle, 'current_ff_file') else None

        # Use override data if enabled and available, otherwise use original CALSTARS
        detected_stars = None
        if self.star_detection_override_enabled and ff_name_c in self.star_detection_override_data:
            detected_stars = np.array(self.star_detection_override_data[ff_name_c])
        elif ff_name_c is not None and ff_name_c in self.calstars:
            detected_stars = np.array(self.calstars[ff_name_c])

        if detected_stars is None or len(detected_stars) < 5:
            print("  Not enough detected stars ({})".format(
                0 if detected_stars is None else len(detected_stars)), flush=True)
            return

        # Get star coordinates (CALSTARS format: Y(0) X(1) ...)
        det_y = detected_stars[:, 0]
        det_x = detected_stars[:, 1]
        calstars_coords = np.column_stack([det_x, det_y])

        # Get time for current image
        if hasattr(self.img_handle, 'currentFrameTime'):
            calstars_time = list(self.img_handle.currentFrameTime(dt_obj=False))
            if len(calstars_time) == 6:
                calstars_time.append(0)
        else:
            calstars_time = list(jd2Date(self.img_handle.currentTime(), dt_obj=False))
            if len(calstars_time) == 6:
                calstars_time.append(0)

        print("  Detected stars: {}".format(len(calstars_coords)), flush=True)
        print("  Image time: {}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}".format(*calstars_time[:6]), flush=True)
        sys.stdout.flush()

        # Call alignPlatepar (same as CheckFit/ApplyRecalibrate would)
        print("  Calling alignPlatepar()...", flush=True)
        sys.stdout.flush()
        try:
            pp_aligned = alignPlatepar(self.config, self.platepar, calstars_time, calstars_coords)
        except Exception as e:
            print("  alignPlatepar FAILED: {}".format(str(e)), flush=True)
            traceback.print_exc()
            return

        # Check if original platepar was modified (it shouldn't be!)
        if self.platepar.RA_d != orig_ra or self.platepar.dec_d != orig_dec:
            print("  WARNING: Original platepar was MODIFIED! This is a bug!", flush=True)
            print("    RA: {:.4f} -> {:.4f}".format(orig_ra, self.platepar.RA_d), flush=True)
            print("    Dec: {:.4f} -> {:.4f}".format(orig_dec, self.platepar.dec_d), flush=True)

        # Check if platepar changed
        if pp_aligned is self.platepar:
            print("  alignPlatepar returned ORIGINAL platepar (fit failed or drift exceeded)", flush=True)
            return

        # Report results
        print("  alignPlatepar result:", flush=True)
        print("    RA:  {:.4f} -> {:.4f} deg (delta: {:.4f})".format(
            orig_ra, pp_aligned.RA_d, pp_aligned.RA_d - orig_ra), flush=True)
        print("    Dec: {:.4f} -> {:.4f} deg (delta: {:.4f})".format(
            orig_dec, pp_aligned.dec_d, pp_aligned.dec_d - orig_dec), flush=True)
        print("    Rot: {:.4f} -> {:.4f} deg (delta: {:.4f})".format(
            orig_rot, pp_aligned.pos_angle_ref,
            pp_aligned.pos_angle_ref - orig_rot), flush=True)

        # Apply the aligned platepar to see the result visually
        self.platepar.RA_d = pp_aligned.RA_d
        self.platepar.dec_d = pp_aligned.dec_d
        self.platepar.pos_angle_ref = pp_aligned.pos_angle_ref
        self.platepar.F_scale = pp_aligned.F_scale
        self.platepar.JD = pp_aligned.JD
        self.platepar.Ho = pp_aligned.Ho
        self.platepar.rotation_from_horiz = rotationWrtHorizon(self.platepar)
        self.platepar.updateRefAltAz()

        # Update display
        self.updateStars()
        self.updateLeftLabels()
        self.updateDistortion()
        self.tab.param_manager.updatePlatepar()

        print("  APPLIED aligned platepar to display", flush=True)
        print("=" * 70, flush=True)
        print(flush=True)
        sys.stdout.flush()


    def findBestFrame(self):
        """ Find the frame with the best star distribution for calibration.

        Uses the CALSTARS data to score each frame based on spatial distribution
        of detected stars. Navigates to the best frame.
        """
        from RMS.Astrometry.AutoPlatepar import selectBestFrame

        # Check if we have CALSTARS data
        if not hasattr(self, 'calstars') or self.calstars is None or len(self.calstars) == 0:
            QtWidgets.QMessageBox.warning(
                self, "No CALSTARS Data",
                "No CALSTARS data available. Run star detection first."
            )
            return

        # Check if we have a multi-image handle with ff_list
        if not hasattr(self.img_handle, 'ff_list'):
            QtWidgets.QMessageBox.warning(
                self, "Single Image",
                "This feature requires multiple images (FF files)."
            )
            return

        self.status_bar.showMessage("Finding best frame...")
        QtWidgets.QApplication.processEvents()

        # Get image dimensions from config
        img_width = self.config.width
        img_height = self.config.height

        # Build set of available image filenames (basenames only), excluding placeholders
        available_images = set()
        for ff_path in self.img_handle.ff_list:
            basename = os.path.basename(ff_path)
            # Skip placeholder images
            if "_placeholder" not in basename:
                available_images.add(basename)

        # Find the best frame from all CALSTARS
        best_ff, best_score, all_scores = selectBestFrame(
            self.calstars, img_width, img_height, verbose=False
        )

        if best_ff is None:
            QtWidgets.QMessageBox.warning(
                self, "No Best Frame Found",
                "Could not determine the best frame from CALSTARS data."
            )
            self.status_bar.showMessage("Best frame search failed")
            return

        # Check if best frame is available in the image list
        best_ff_available = best_ff in available_images

        # Find best frame among only available images
        calstars_available = {k: v for k, v in self.calstars.items() if k in available_images}

        if len(calstars_available) == 0:
            QtWidgets.QMessageBox.warning(
                self, "No Matching Frames",
                "None of the CALSTARS frames match the available images."
            )
            self.status_bar.showMessage("No matching frames")
            return

        best_available_ff, best_available_score, _ = selectBestFrame(
            calstars_available, img_width, img_height, verbose=False
        )

        # Determine which frame to use
        selected_ff = None
        selected_score = None

        if best_ff_available:
            # Best overall is available - use it directly
            selected_ff = best_ff
            selected_score = best_score
        elif best_available_ff is not None:
            # Best overall not available, but we have a best among available
            # Ask user what to do
            msg_box = QtWidgets.QMessageBox(self)
            msg_box.setWindowTitle("Best Frame for Calibration")

            # Get star counts for display
            n_stars_best = len(self.calstars.get(best_ff, []))
            n_stars_available = len(self.calstars.get(best_available_ff, []))

            msg_box.setText(
                "The frame with the best star distribution in CALSTARS\n"
                "doesn't have a saved image file in the given directory.\n\n"
                "You can either:\n"
                "  • Calibrate automatically using the star data\n"
                "    (a placeholder will be shown instead of the image)\n"
                "  • Go to the best available image and calibrate from there"
            )
            msg_box.setInformativeText(
                f"From CALSTARS:\n"
                f"  Best frame:       {best_ff}  ({n_stars_best} stars)\n"
                f"  Best with image:  {best_available_ff}  ({n_stars_available} stars)"
            )

            # Add custom buttons
            auto_fit_btn = msg_box.addButton("Auto Fit (placeholder)", QtWidgets.QMessageBox.ActionRole)
            navigate_btn = msg_box.addButton("Go to Best Image", QtWidgets.QMessageBox.ActionRole)
            cancel_btn = msg_box.addButton(QtWidgets.QMessageBox.Cancel)

            msg_box.setDefaultButton(auto_fit_btn)
            msg_box.exec_()

            clicked = msg_box.clickedButton()

            if clicked == auto_fit_btn:
                # Run AutoPlatepar on the best CALSTARS frame
                self.status_bar.showMessage(f"Running auto fit on {best_ff}...")
                QtWidgets.QApplication.processEvents()

                from RMS.Astrometry.AutoPlatepar import autoFitPlatepar

                try:
                    # autoFitPlatepar returns (platepar, matched_stars, ff_name)
                    result = autoFitPlatepar(
                        self.dir_path,
                        self.config,
                        self.catalog_stars,
                        ff_name=best_ff
                    )

                    if result is None:
                        new_platepar = None
                    else:
                        new_platepar, matched_stars, used_ff = result

                    if new_platepar is not None:
                        # Create a placeholder PNG image with diagonal stripes
                        height, width = self.config.height, self.config.width
                        placeholder = np.full((height, width), 24, dtype=np.uint8)

                        # Add diagonal stripes (lighter gray)
                        stripe_width = 40
                        stripe_spacing = 80
                        for i in range(-(height + width), height + width, stripe_spacing):
                            for offset in range(stripe_width):
                                y_coords = np.arange(height)
                                x_coords = i + y_coords + offset
                                valid = (x_coords >= 0) & (x_coords < width)
                                placeholder[y_coords[valid], x_coords[valid]] = 40

                        # Create placeholder filename based on original FF name
                        # e.g. FF_USV003_20250416_091633_928_0628224.fits
                        #   -> FF_USV003_20250416_091633_928_0628224_placeholder.png
                        base_name = os.path.splitext(best_ff)[0]
                        placeholder_name = f"{base_name}_placeholder.png"
                        placeholder_path = os.path.join(self.dir_path, placeholder_name)

                        # Save the placeholder PNG
                        from PIL import Image
                        img_pil = Image.fromarray(placeholder)
                        img_pil.save(placeholder_path)
                        print(f"Created placeholder image: {placeholder_path}")

                        # Copy CALSTARS data to placeholder filename so detected stars are shown
                        if best_ff in self.calstars:
                            self.calstars[placeholder_name] = self.calstars[best_ff]

                        # Refresh the file list and navigate to the placeholder
                        # Re-detect input type to include new file
                        self.img_handle = detectInputTypeFolder(
                            self.dir_path, self.config,
                            beginning_time=None, fps=self.fps
                        )
                        self.img.changeHandle(self.img_handle)

                        # Find and navigate to the placeholder
                        target_index = None
                        for i, ff_path in enumerate(self.img_handle.ff_list):
                            if os.path.basename(ff_path) == placeholder_name:
                                target_index = i
                                break

                        if target_index is not None:
                            current_index = self.img_handle.current_ff_index
                            delta = target_index - current_index
                            if delta != 0:
                                self.nextImg(n=delta)
                        else:
                            # Fallback - just set the image directly
                            self.img.setImage(placeholder.T)

                        # Update the current platepar
                        self.platepar = new_platepar
                        self.tab.param_manager.updatePlatepar()
                        self.updateDistortion()

                        # Set up paired_stars from matched results
                        if matched_stars is not None:
                            self.paired_stars = matched_stars
                            self.tab.param_manager.updatePairedStars(len(self.paired_stars))

                        self.updateStars()
                        self.updateLeftLabels()

                        # Compute and display residuals
                        if len(self.paired_stars) >= self.getMinFitStars():
                            self.fitPickedStars()

                        # Save the platepar
                        platepar_path = os.path.join(self.dir_path, self.config.platepar_name)
                        self.platepar.write(platepar_path, fmt=self.platepar_fmt)

                        self.status_bar.showMessage(
                            f"Auto fit complete: {placeholder_name} - {len(self.paired_stars)} stars"
                        )
                        print(f"\nAuto fit complete on {best_ff}")
                        print(f"Platepar saved to: {platepar_path}")
                    else:
                        QtWidgets.QMessageBox.warning(
                            self, "Auto Fit Failed",
                            f"Failed to create platepar from {best_ff}."
                        )
                        self.status_bar.showMessage("Auto fit failed")

                except Exception as e:
                    QtWidgets.QMessageBox.warning(
                        self, "Auto Fit Error",
                        f"Error during auto fit: {str(e)}"
                    )
                    self.status_bar.showMessage("Auto fit error")
                    import traceback
                    traceback.print_exc()

                return

            elif clicked == navigate_btn:
                selected_ff = best_available_ff
                selected_score = best_available_score
            else:
                self.status_bar.showMessage("Best frame search cancelled")
                return
        else:
            QtWidgets.QMessageBox.warning(
                self, "No Valid Frame",
                "Could not find a valid frame to navigate to."
            )
            self.status_bar.showMessage("No valid frame found")
            return

        # Find the index of selected frame in ff_list
        target_index = None
        for i, ff_name in enumerate(self.img_handle.ff_list):
            if os.path.basename(ff_name) == selected_ff or ff_name == selected_ff:
                target_index = i
                break

        if target_index is None:
            # Should not happen at this point, but handle gracefully
            self.status_bar.showMessage("Error finding frame index")
            return

        # Navigate to the selected frame
        current_index = self.img_handle.current_ff_index
        delta = target_index - current_index

        if delta != 0:
            self.nextImg(n=delta)

        # Get score details for status message
        score_info = all_scores.get(selected_ff, {})
        n_stars = score_info.get('quality_details', {}).get('n_stars', 0)

        self.status_bar.showMessage(
            f"Best frame: {selected_ff} (score={selected_score:.3f}, {n_stars} stars)"
        )

        print(f"\nBest frame: {selected_ff}")
        print(f"  Score: {selected_score:.3f}")
        print(f"  Stars: {n_stars}")


    def autoFitAstrometryNet(self):
        """ Auto fit using astrometry.net. Called from Auto Fit button. """

        # If there are existing matched star pairs, warn the user they will be replaced
        if len(self.paired_stars) > 0:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Replace Matched Stars?",
                "You have {} matched star pair(s) that will be replaced by auto-fit.\n\n"
                "Do you want to continue?".format(len(self.paired_stars)),
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return

        # Balance catalog magnitude before any fitting (affects both quick and full paths)
        self.balanceCatalogMagnitude()

        # First, try quick alignment with existing platepar (much faster than astrometry.net)
        quick_fit_success = self.tryQuickAlignment()

        if not quick_fit_success:
            # Fall back to astrometry.net if quick alignment failed
            self.getInitialParamsAstrometryNet(upload_image=False)

        # Update the GUI
        self.updateDistortion()
        self.updateLeftLabels()
        self.updateStars()
        self.tab.param_manager.updatePlatepar()


    def getInitialParamsAstrometryNet(self, upload_image=True):
        """ Get the estimate of the initial astrometric parameters using astrometry.net. """

        # Show status and process events so GUI updates
        self.status_bar.showMessage("Solving with astrometry.net...")
        QtWidgets.QApplication.processEvents()

        fail = False
        solution = None

        # Construct FOV width estimate (0.75x to 1.5x range)
        fov_w_range = [0.75*self.config.fov_w, 1.5*self.config.fov_w]

        # Handle using FR files too
        ff_name_c = convertFRNameToFF(self.img_handle.name())

        # Find and load a mask file is there is one
        mask = getMaskFile(self.dir_path, self.config)

        # Compute JD for astrometry.net matching
        jd = date2JD(*self.img_handle.currentTime())

        # Use override data if enabled and available, otherwise use original CALSTARS
        has_star_data = False
        star_data = None
        if self.star_detection_override_enabled and ff_name_c in self.star_detection_override_data:
            star_data = np.array(self.star_detection_override_data[ff_name_c])
            has_star_data = True
        elif ff_name_c in self.calstars:
            star_data = np.array(self.calstars[ff_name_c])
            has_star_data = True

        if has_star_data and (not upload_image):

            # Make sure that there are at least 10 stars
            if len(star_data) < 10:
                print('Less than 10 stars on the image!')
                fail = True

            else:

                # Get star coordinates
                y_data = star_data[:, 0]
                x_data = star_data[:, 1]

                # Get star intensities for brightness-based matching (column 2 is IntensSum)
                input_intensities = None
                if star_data.shape[1] > 2:
                    input_intensities = star_data[:, 2]

                # Get astrometry.net solution, pass the FOV width estimate
                solution = astrometryNetSolve(x_data=x_data, y_data=y_data, fov_w_range=fov_w_range,
                                              fov_w_hint=self.config.fov_w, mask=mask,
                                              x_center=self.platepar.X_res/2, y_center=self.platepar.Y_res/2,
                                              lat=self.platepar.lat, lon=self.platepar.lon, jd=jd,
                                              input_intensities=input_intensities,
                                              verbose=True)

        else:
            fail = True

        # Try finding the solution by uploading the whole image
        if fail or upload_image:

            print("Using the whole image in astrometry.net...")
            self.status_bar.showMessage("Uploading image to astrometry.net...")
            QtWidgets.QApplication.processEvents()

            # If the image is 16bit or larger, rescale and convert it to 8 bit
            if self.img.data.itemsize*8 > 8:

                # Rescale the image to 8bit
                minv, maxv = self.tab.hist.getLevels()
                img_data = adjustLevels(self.img.data, minv, self.img.gamma, maxv)
                img_data -= np.min(img_data)
                img_data = 255*(img_data/np.max(img_data))
                img_data = img_data.astype(np.uint8)

            else:
                img_data = self.img.data

            solution = astrometryNetSolve(img=img_data.T, fov_w_range=fov_w_range,
                                          fov_w_hint=self.config.fov_w, mask=mask,
                                          lat=self.platepar.lat, lon=self.platepar.lon, jd=jd)

        if solution is None:
            self.status_bar.showMessage("Astrometry.net failed to find a solution")
            qmessagebox(title='Astrometry.net error',
                        message='Astrometry.net failed to find a solution!',
                        message_type="error")

            return None
        

        # Update status: solution found
        self.status_bar.showMessage("Astrometry.net solution found, processing...")
        QtWidgets.QApplication.processEvents()

        # Save user's settings for the final fit
        user_distortion_type = self.platepar.distortion_type
        user_equal_aspect = self.platepar.equal_aspect
        user_asymmetry_corr = self.platepar.asymmetry_corr
        user_force_distortion_centre = self.platepar.force_distortion_centre
        user_refraction = self.platepar.refraction
        user_fit_only_pointing = self.fit_only_pointing
        user_fixed_scale = self.fixed_scale

        # Set intermediate fitting parameters (simple, robust settings)
        # Use simple settings for stability with few stars during initial passes
        self.platepar.refraction = True
        self.platepar.equal_aspect = True
        self.platepar.asymmetry_corr = False
        self.platepar.force_distortion_centre = False

        # Start with radial5-odd for initial fitting, reset distortion params
        self.platepar.setDistortionType("radial5-odd", reset_params=True)


        # Extract the parameters
        ra, dec, rot_standard, scale, fov_w, fov_h, star_data, solution_info = solution

        # Store solution info for potential visualization
        self.astrometry_solution_info = solution_info

        # Set the platepar reference JD and compute the reference hour angle
        # (jd was computed earlier for the astrometry.net call)
        self.platepar.JD = jd
        self.platepar.Ho = JD2HourAngle(jd) % 360

        # Compute reference azimuth and altitude
        azim, alt = trueRaDec2ApparentAltAz(ra, dec, jd, self.platepar.lat, self.platepar.lon)

        # Set parameters to platepar
        self.platepar.F_scale = scale
        self.platepar.az_centre = azim
        self.platepar.alt_centre = alt

        self.platepar.updateRefRADec(skip_rot_update=True)

        self.platepar.pos_angle_ref = rotationWrtStandardToPosAngle(self.platepar, rot_standard)

        # Print estimated parameters
        print()
        print('Astrometry.net solution:')
        print('------------------------')
        print(' RA    = {:.2f} deg'.format(self.platepar.RA_d))
        print(' Dec   = {:.2f} deg'.format(self.platepar.dec_d))
        print(' Azim  = {:.2f} deg'.format(self.platepar.az_centre))
        print(' Alt   = {:.2f} deg'.format(self.platepar.alt_centre))
        print(' Rot horiz   = {:.2f} deg'.format(self.platepar.rotation_from_horiz))
        print(' Rot eq      = {:.2f} deg'.format(rot_standard))
        print(' Pos angle   = {:.2f} deg'.format(self.platepar.pos_angle_ref))
        print(' Scale = {:.3f} arcmin/px'.format(60/self.platepar.F_scale))
        print(' FOV = {:.2f} x {:.2f} deg'.format(fov_w, fov_h))

        # Print solution info if available
        if solution_info is not None:
            quad_stars = solution_info.get('quad_stars', [])
            logodds = solution_info.get('logodds')
            input_count = solution_info.get('input_star_count', 0)

            if logodds is not None:
                print(' Log odds = {:.2f}'.format(logodds))
            print(' Input stars = {:d}'.format(input_count))
            print(' Quad stars = {:d}'.format(len(quad_stars)))

        # Update the GUI to show the astrometry.net solution before NN refinement
        self.updateLeftLabels()
        self.updateStars()
        self.drawPhotometryColoring()
        QtWidgets.QApplication.processEvents()

        # Ask user if they want to continue with NN refinement
        reply = QtWidgets.QMessageBox.question(self, 'Astrometry.net Solution',
            'Astrometry.net found a solution.\n\n'
            'RA = {:.2f} deg, Dec = {:.2f} deg\n'
            'Scale = {:.3f} arcmin/px\n\n'
            'Continue with NN-based refinement?'.format(
                self.platepar.RA_d, self.platepar.dec_d, 60/self.platepar.F_scale),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.Yes)

        if reply == QtWidgets.QMessageBox.No:
            self.status_bar.showMessage("Astrometry.net solution applied (no refinement)")
            return self.platepar

        # Match detected stars to RMS catalog and fit distortion iteratively
        # Use RMS's own catalog (better than astrometry.net's index stars)
        # Iterative approach: start with bright stars + wide radius, fit, then tighten
        print()
        print("Iterative star matching with RMS catalog...")
        self.status_bar.showMessage("Matching stars with catalog...")
        QtWidgets.QApplication.processEvents()

        # Use the same catalog filtering as the GUI display (proven to work correctly)
        # self.catalog_stars is the full catalog, filterCatalogStarsInsideFOV does RA/Dec filtering
        _, catalog_stars_extended = self.filterCatalogStarsInsideFOV(self.catalog_stars)
        print("  Catalog stars within FOV (filterCatalogStarsInsideFOV): {:d}".format(len(catalog_stars_extended)))

        # Strict XY filter: project to image and keep only those inside image bounds
        catalog_x_ext, catalog_y_ext, catalog_mag_ext = getCatalogStarsImagePositions(
            catalog_stars_extended, jd, self.platepar)
        in_fov_xy = (catalog_x_ext >= 0) & (catalog_x_ext < self.platepar.X_res) & \
                    (catalog_y_ext >= 0) & (catalog_y_ext < self.platepar.Y_res)

        catalog_stars = catalog_stars_extended[in_fov_xy]

        print("  Catalog stars in strict FOV: {:d}".format(len(catalog_stars)))

        # Get detected stars - use override data if enabled, otherwise CALSTARS
        # CALSTARS format: Y(0) X(1) IntensSum(2) Ampltd(3) FWHM(4) BgLvl(5) SNR(6) NSatPx(7)
        detected_stars = None
        if self.star_detection_override_enabled and ff_name_c in self.star_detection_override_data:
            detected_stars = np.array(self.star_detection_override_data[ff_name_c])
        elif ff_name_c in self.calstars:
            detected_stars = np.array(self.calstars[ff_name_c])

        if detected_stars is not None and len(detected_stars) > 0:
            det_y = detected_stars[:, 0]
            det_x = detected_stars[:, 1]
            # Use index 2 (IntensSum/integrated intensity) not index 3 (Ampltd/peak amplitude)
            det_intens = detected_stars[:, 2] if detected_stars.shape[1] > 2 else np.ones(len(det_x))
            det_fwhm = detected_stars[:, 4] if detected_stars.shape[1] > 4 else np.zeros(len(det_x))
            det_snr = detected_stars[:, 6] if detected_stars.shape[1] > 6 else np.ones(len(det_x))
            det_saturated = detected_stars[:, 7] if detected_stars.shape[1] > 7 else np.zeros(len(det_x))
        else:
            print("No detected stars available for matching")
            det_x, det_y, det_intens = np.array([]), np.array([]), np.array([])
            det_fwhm, det_snr, det_saturated = np.array([]), np.array([]), np.array([])

        print("  Detected stars: {:d}".format(len(det_x)))

        # First pass: Use NN cost function to refine pointing + distortion
        # This doesn't require explicit star matching - more robust for initial fit
        if len(det_x) >= 10 and len(catalog_stars) >= 10:
            print()
            print("NN-based fitting (no explicit matching)...")

            # Prepare detected stars array [x, y, intensity]
            img_stars_arr = np.column_stack([det_x, det_y, det_intens])

            # Use NN cost function to fit pointing + distortion
            # Pass extended catalog (before strict XY filter) - NN iterations re-filter
            # dynamically as distortion improves, allowing edge stars to "appear"
            self.platepar.setDistortionType("radial5-odd", reset_params=True)
            try:
                self.platepar.fitAstrometry(
                    jd, img_stars_arr, catalog_stars_extended,  # Extended catalog for edge stars
                    first_platepar_fit=True,
                    use_nn_cost=True
                )
                print("  NN fit complete: RA={:.2f} Dec={:.2f} Scale={:.3f} arcmin/px".format(
                    self.platepar.RA_d, self.platepar.dec_d, 60/self.platepar.F_scale))
            except Exception as e:
                print("  NN fit failed: {}".format(str(e)))

            # Populate paired_stars from NN matches for visualization
            self.paired_stars = PairedStars()
            if self.platepar.star_list:
                for entry in self.platepar.star_list:
                    # star_list format: [jd, x, y, intensity, ra, dec, mag]
                    _, x, y, intensity, ra, dec, mag = entry
                    sky_obj = CatalogStar(ra, dec, mag)

                    # Look up SNR, FWHM, and saturation from calstars by finding nearest detected star
                    fwhm, snr, saturated = 0.0, 1.0, False
                    if len(det_x) > 0:
                        # Find the closest detected star to this matched star
                        distances = np.sqrt((det_x - x)**2 + (det_y - y)**2)
                        closest_idx = np.argmin(distances)
                        if distances[closest_idx] < 3.0:  # Within 3 pixels
                            fwhm = det_fwhm[closest_idx]
                            snr = det_snr[closest_idx]
                            saturated = det_saturated[closest_idx] > 0

                    self.paired_stars.addPair(x, y, fwhm, intensity, sky_obj, snr=snr, saturated=saturated)
                print("  Loaded {} matched pairs".format(len(self.platepar.star_list)))

        # Finalize the fit with user's settings
        if len(self.paired_stars) >= 10:
            # Restore user's settings for the final fit
            print()
            print("Restoring user settings: {:s}".format(user_distortion_type))
            self.platepar.equal_aspect = user_equal_aspect
            self.platepar.asymmetry_corr = user_asymmetry_corr
            self.platepar.force_distortion_centre = user_force_distortion_centre
            self.platepar.setDistortionType(user_distortion_type, reset_params=False)
            self.platepar.refraction = user_refraction
            self.fit_only_pointing = user_fit_only_pointing
            self.fixed_scale = user_fixed_scale

            # Filter photometric outliers before final fit
            if len(self.paired_stars) >= 15:
                removed = self.filterPhotometricOutliers(sigma_threshold=2.5)
                if removed > 0:
                    print("Pairs after photometric filtering: {}".format(len(self.paired_stars)))

            # Filter blended stars before final fit
            if len(self.paired_stars) >= 15:
                removed = self.filterBlendedStars(blend_radius_arcsec=30.0)
                if removed > 0:
                    print("Pairs after blend filtering: {}".format(len(self.paired_stars)))

            # Filter high FWHM stars before final fit (remove top 10%)
            if len(self.paired_stars) >= 15:
                removed = self.filterHighFWHMStars(fraction=0.10)
                if removed > 0:
                    print("Pairs after FWHM filtering: {}".format(len(self.paired_stars)))

            # Do the final fit with user's settings
            print()
            print("Final fit with user settings...")
            self.status_bar.showMessage("Fitting astrometry with {:d} stars...".format(len(self.paired_stars)))
            QtWidgets.QApplication.processEvents()
            self.first_platepar_fit = True
            self.fitPickedStars()

            # Update the display
            self.updateStars()
            self.status_bar.showMessage("Auto-fit complete: {:d} stars matched".format(len(self.paired_stars)))
        else:
            # Restore user's settings even if fit failed
            self.platepar.equal_aspect = user_equal_aspect
            self.platepar.asymmetry_corr = user_asymmetry_corr
            self.platepar.force_distortion_centre = user_force_distortion_centre
            self.platepar.setDistortionType(user_distortion_type, reset_params=True)
            self.platepar.refraction = user_refraction
            self.fit_only_pointing = user_fit_only_pointing
            self.fixed_scale = user_fixed_scale

            print("  Not enough matched stars for fitting (need >= 10)")
            self.updateStars()
            self.status_bar.showMessage("Auto-fit: not enough star matches")

        # Show astrometry.net quad stars if available
        if self.astrometry_solution_info is not None:
            quad_count = len(self.astrometry_solution_info.get('quad_stars', []))
            if quad_count > 0:
                self.astrometry_stars_visible = True
                self.astrometry_quad_markers.show()
                self.astrometry_quad_markers2.show()
                self.updateAstrometryNetStarMarkers()
                print()
                print("Showing astrometry.net quad stars: {:d} (magenta)".format(quad_count))
                print("  Magenta = catalog stars used for initial geometric match")
                print("Press Shift+H to toggle visibility")


    def getFOVcentre(self):
        """ Asks the user to input the centre of the FOV in altitude and azimuth. """

        # Get FOV centre
        d = QFOVinputDialog(self)
        d.loadLensTemplates(self.config, self.dir_path, self.config.width, self.config.height)
        if d.exec_():
             data = d.getInputs()
        else:
            return 0, 0, 0, "none"

        self.azim_centre, self.alt_centre, rot_horizontal, lenses_template_file = data

        # read platepar data from a reference file
        if lenses_template_file != "none":
            self.loadPlatepar(update=False, platepar_file=lenses_template_file)

        # Wrap azimuth to 0-360 range
        self.azim_centre %= 360

        # Limit the altitude to 89 deg
        if self.alt_centre >= 90:
            self.alt_centre = 89.0

        # Wrap the rotation in the 0-360 range
        rot_horizontal %= 360

        # Get the middle time of the first FF
        img_time = self.img_handle.currentTime()

        # Set the reference platepar time to the time of the FF
        self.platepar.JD = date2JD(*img_time)

        # Set the reference hour angle
        self.platepar.Ho = JD2HourAngle(self.platepar.JD)%360

        # Convert FOV centre to RA, Dec
        ra, dec = apparentAltAz2TrueRADec(self.azim_centre, self.alt_centre, date2JD(*img_time),
                                          self.platepar.lat, self.platepar.lon)

        return ra, dec, rot_horizontal, lenses_template_file


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
        if os.path.isfile(self.input_path):
            img_handle = detectInputTypeFile(self.input_path, self.config, beginning_time=beginning_time, 
                                             flipud=self.flipud)
        
        # Load given data from a folder
        elif os.path.isdir(self.input_path):

            # Detect input file type and load appropriate input plugin
            img_handle = detectInputTypeFolder(self.dir_path, self.config, beginning_time=beginning_time, \
                use_fr_files=self.use_fr_files, flipud=self.flipud)

            # If the data was not being able to load from the folder, choose a file to load
            if img_handle is None:
                self.input_path = QtWidgets.QFileDialog.getOpenFileName(self, "Select image/video file to open",
                    self.dir_path, "All readable files (*.fits *.bin *.mp4 *.avi *.mkv *.vid *.png *.jpg *.bmp *.nef *.tif);;" + \
                                   "All files (*);;" + \
                                   "FF and FR Files (*.fits;*.bin);;" + \
                                   "Video Files (*.mp4 *.avi *.mkv);;" + \
                                   "VID Files (*.vid);;" + \
                                   "FITS Files (*.fits);;" + \
                                   "BIN Files (*.bin);;" + \
                                   "Image Files (*.png *.jpg *.bmp *.nef *.tif)")[0]


        # If no previous ways of opening data was successful, open a file
        if img_handle is None:
            img_handle = detectInputTypeFile(self.input_path, self.config, beginning_time=beginning_time, 
                                             flipud=self.flipud)

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

                return None


            # Try generating CALSTARS automatically
            else:

                print("The CALSTARS file is missing, trying to generate it automatically...")
                calstars_list = extractStarsAndSave(self.config, self.dir_path)


        else:

            # Load the calstars file
            calstars_list, _ = CALSTARS.readCALSTARS(self.dir_path, calstars_file)

            print('CALSTARS file: ' + calstars_file + ' loaded!')


        # Convert the list to a dictionary
        self.calstars = {ff_file: star_data for ff_file, star_data in calstars_list}


    def reloadGeoPoints(self):
        """ Reload the file with geo points. """

        if self.geo_points_obj is not None:
            if os.path.isfile(self.geo_points_input):
                self.geo_points_obj = GeoPoints(self.geo_points_input)

                # Remove all geo points from the picked list
                self.paired_stars.removeGeoPoints()

                self.updateStars()


    def loadCatalogStars(self, lim_mag):
        """ Loads stars from the BSC star catalog.

        Arguments:
            lim_mag: [float] Limiting magnitude of catalog stars.

        """

        # If the star catalog path doesn't exist, use the catalog available in the repository
        if not os.path.isdir(self.config.star_catalog_path):
            self.config.star_catalog_path = os.path.join(self.config.rms_root_dir, 'Catalogs')
            print("Updated catalog path to: ", self.config.star_catalog_path)

        # Compute the number of years from J2000
        years_from_J2000 = (
            self.img_handle.beginning_datetime - datetime.datetime(2000, 1, 1, 12, 0, 0)
                ).days/365.25
    
        # Load catalog stars
        catalog_results = StarCatalog.readStarCatalog(
            self.config.star_catalog_path, self.config.star_catalog_file,
            years_from_J2000=years_from_J2000,
            lim_mag=lim_mag,
            mag_band_ratios=self.config.star_catalog_band_ratios,
            additional_fields=['spectraltype_esphs', 'preferred_name', 'common_name', 'bayer_name'])

        if len(catalog_results) == 4:
            self.catalog_stars, self.mag_band_string, self.config.star_catalog_band_ratios, extras = catalog_results
        else:
            self.catalog_stars, self.mag_band_string, self.config.star_catalog_band_ratios = catalog_results
            extras = {}

        # Extract spectral type
        if 'spectraltype_esphs' in extras:
            # Decode bytes to strings if necessary
            self.catalog_stars_spectral_type = np.array([x.decode('utf-8') for x in extras['spectraltype_esphs']])
        else:
            self.catalog_stars_spectral_type = None

        # Extract preferred name
        if 'preferred_name' in extras:
            # Decode bytes to strings if necessary
            self.catalog_stars_preferred_names = np.array([x.decode('utf-8') for x in extras['preferred_name']])

            # Extract bayer_name if available
            if 'bayer_name' in extras:
                self.catalog_stars_bayer_names = np.array([x.decode('utf-8') for x in extras['bayer_name']])
            else:
                self.catalog_stars_bayer_names = None

            # Extract common_name if available
            if 'common_name' in extras:
                raw_common_names = np.array([x.decode('utf-8') for x in extras['common_name']])
            else:
                raw_common_names = None

            # Build display names with priority: common_name > bayer_name > preferred_name
            display_names = []
            for i in range(len(self.catalog_stars_preferred_names)):
                pref = self.catalog_stars_preferred_names[i].strip()
                common = raw_common_names[i].strip() if raw_common_names is not None else ''
                bayer = self.catalog_stars_bayer_names[i].strip() if self.catalog_stars_bayer_names is not None else ''

                if common:
                    display_names.append(common)
                elif bayer:
                    display_names.append(bayer)
                else:
                    display_names.append(pref)
            self.catalog_stars_common_names = np.array(display_names)
        else:
            self.catalog_stars_preferred_names = None
            self.catalog_stars_common_names = None
            self.catalog_stars_bayer_names = None

        return self.catalog_stars


    def loadPlatepar(self, update=False, platepar_file=None):
        """
        Open a file dialog and ask user to open the platepar file, changing self.platepar and self.platepar_file

        Arguments:
            update: [bool] Whether to update the gui after loading new platepar (leave as False if gui objects
                            may not exist)
            platepar_file: [string] path to a platepar file to be loaded. If not specified a dialog box in GUI
                            will be opened so user can specify one

        """

        platepar = Platepar()

        if self.config.platepar_name in os.listdir(self.dir_path):
            initial_file = os.path.join(self.dir_path, self.config.platepar_name)
        else:
            initial_file = self.dir_path

        # Open the file dialog no 'platepar_file' parameter was specified
        if platepar_file is None:
            platepar_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select the platepar file", 
                                                                  initial_file,
                                                                  "Platepar files (*.cal);;All files (*)")[0]

        if platepar_file == '':
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

            # Update reference alt/az
            platepar.updateRefAltAz()

            self.first_platepar_fit = False

            self.platepar_file, self.platepar = platepar_file, platepar


        if update:
            self.updateStars()
            self.drawPhotometryColoring()

            self.tab.param_manager.updatePlatepar()
            self.updateLeftLabels()

    def savePlatepar(self):
        """  Save platepar to a file """

        # If the platepar is new, save it to the working directory
        if (not self.platepar_file) or (not os.path.isfile(self.platepar_file)):
            self.platepar_file = os.path.join(self.dir_path, self.config.platepar_name)

        # Save the platepar file
        self.platepar.write(self.platepar_file, fmt=self.platepar_fmt, fov=computeFOVSize(self.platepar))
        print('Platepar written to:', self.platepar_file)

    def saveDefaultPlatepar(self):
        default_path = os.path.join(self.config.config_file_path, self.config.platepar_name)

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Default Platepar", default_path,
            "Platepar files (*.cal);;All Files (*)")

        if file_path:
            self.platepar.write(file_path, fmt=self.platepar_fmt)
            print('Default platepar written to:', file_path)

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
        saveImage(frame_file_path, self.img.getFrame())

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

        # Set the default vignetting coefficient scaled for this resolution
        self.platepar.addVignettingCoeff(use_flat=self.config.use_flat)

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
            (
                self.platepar.RA_d, 
                self.platepar.dec_d, 
                self.platepar.rotation_from_horiz, 
                self.lenses
            ) = self.getFOVcentre()

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
        self.paired_stars = PairedStars()
        self.residuals = None

        # Indicate that a new platepar is being made
        self.new_platepar = True

        if hasattr(self, 'tab'):
            self.tab.param_manager.updatePlatepar()
            self.updateLeftLabels()
            self.updateStars()
            self.updateDistortion()


    def usingUWOData(self):
        """ Return True if using any UWO instrument data. """

        if self.img_handle is None:
            return False

        if self.img_handle.input_type == 'images':
            if self.img_handle.uwo_png_mode:
                return True

        elif self.img_handle.input_type == 'vid':
                return True

        return False


    def loadFlat(self, force_dialog=False):
        """ Open a file dialog and ask user to load a flat field. """

        file_names_to_check = [
            self.config.flat_file
            ]
        
        # If we're running in the UWO mode or using a .vid file, add the UWO flat file name
        if self.usingUWOData():
            file_names_to_check.append('flat.png')

        # Check if any of the flat files exist in the folder
        initial_file = None
        for file_name in file_names_to_check:
            if file_name in os.listdir(self.dir_path):
                initial_file = os.path.join(self.dir_path, file_name)
                break


        # If using UWO files, automatically load the flat file and skip the dialog
        if not force_dialog and self.usingUWOData() and initial_file is not None:
            flat_file = initial_file

        else:

            if initial_file is None:
                initial_file = self.dir_path

            # Open the file dialog to select the flat field file
            flat_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select the flat field file", initial_file,
                                                    "Image files (*.png *.jpg *.bmp);;All files (*)")[0]

        if not flat_file:
            return False, None

        print("Loading the flat file:", flat_file)

        try:
            # Load the flat, byteswap the flat if vid file is used or UWO png
            flat = loadFlat(*os.path.split(flat_file), dtype=self.img.data.dtype,
                      byteswap=self.img_handle.byteswap)
            flat.flat_img = np.swapaxes(flat.flat_img, 0, 1)

            print("Flat loaded successfully!")
            
        except Exception as e:
            
            print("Loading the flat failed with error: " + repr(e))
            print()
            print(*traceback.format_exception(*sys.exc_info()))

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

        # Check if the flat field was successfully loaded
        if flat is None:
            qmessagebox(title='Flat field file error',
                        message='The file you selected could not be loaded as a flat field!',
                        message_type="error")

        return flat_file, flat


    def loadDark(self, force_dialog=False):
        """ Open a file dialog and ask user to load a dark frame. """

        file_names_to_check = [
            self.config.dark_file
            ]

        # If we're running in the UWO mode or using a .vid file, add the UWO dark file name
        if self.usingUWOData():
            file_names_to_check.append('bias.png')
            file_names_to_check.append('dark.png')

        # Locate the dark frame file in the folder
        initial_file = None
        for file_name in file_names_to_check:
            if file_name in os.listdir(self.dir_path):
                initial_file = os.path.join(self.dir_path, file_name)
                break            

        # If using UWO files, automatically load the dark file and skip the dialog
        if not force_dialog and self.usingUWOData() and initial_file is not None:
            dark_file = initial_file

        else:
            
            if initial_file is None:
                initial_file = self.dir_path

            # Open the file dialog to select the dark frame file
            dark_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select the dark frame file", 
                initial_file, "Image files (*.png *.jpg *.bmp *.nef *.cr2);;All files (*)")[0]

        if not dark_file:
            return False, None

        print("Loading the dark frame file:", dark_file)

        try:

            # Load the dark
            dark = loadDark(*os.path.split(dark_file), dtype=self.img.data.dtype, \
                                  byteswap=self.img_handle.byteswap)
            
            print("Dark loaded successfully!")

        except Exception as e:

            print("Loading the dark failed with error: " + repr(e))
            print()
            print(*traceback.format_exception(*sys.exc_info()))

            qmessagebox(title='Dark frame error',
                        message='Dark frame could not be loaded!',
                        message_type="error")

            return False, None

        dark = dark.astype(self.img.data.dtype).T

        # Check if the size of the file matches
        if self.img.data.shape != dark.shape:
            print()
            print('Size of the dark frame:', dark.shape)
            print('Size of the image:', self.img.data.shape)
            qmessagebox(title='Dark field file error',
                        message='The size of the dark frame does not match the size of the image!',
                        message_type="error")

            dark = None

        # Check if the dark frame was successfully loaded
        if dark is None:
            qmessagebox(title='Dark field file error',
                        message='The file you selected could not be loaded as a dark field!',
                        message_type="error")

        return dark_file, dark


    def addCentroid(self, frame, x_centroid, y_centroid, mode=1, 
                    background_intensity=0, snr=1, saturated=False):
        """
        Adds or modifies a pick marker at given frame to self.pick_list with given information

        Arguments:
            frame: [int] Frame to add/modify pick to.
            x_centroid: [float] x coordinate of pick.
            y_centroid: [float] y coordinate of pick.

        Keyword arguments:
            mode: [0 or 1] The mode of the pick, 0 is yellow, 1 is red.
            background_intensity: [float] Background intensity of the pick.
            snr: [float] Signal to noise ratio of the pick.
            saturated: [bool] Whether the pick is saturated.

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
                    'photometry_pixels': None,
                    'background_intensity': background_intensity,
                    'snr': snr,
                    'saturated': saturated}
            self.pick_list[frame] = pick

        self.tab.debruijn.modifyRow(frame, mode)

        self.updateGreatCircle()

        # Update Kalman/ASTRA Ready status if instance exists
        if hasattr(self, 'astra_dialog') and self.astra_dialog is not None:
            self.checkASTRACanRun()
            self.checkKalmanCanRun()

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

        # Update Kalman/ASTRA Ready status if instance exists
        if hasattr(self, 'astra_dialog') and self.astra_dialog is not None:
            self.checkASTRACanRun()
            self.checkKalmanCanRun()


    def centroid(self, prev_x_cent=None, prev_y_cent=None):
        """ Find the centroid of the star clicked on the image. """

        self.updateBottomLabel()
        # If the centroid from the previous iteration is given, use that as the centre
        if (prev_x_cent is not None) and (prev_y_cent is not None):
            mouse_x = prev_x_cent
            mouse_y = prev_y_cent

        else:
            mouse_x = self.mouse_x - 0.5
            mouse_y = self.mouse_y - 0.5

        # Check if the mouse was pressed outside the FOV
        if mouse_x is None:
            return None, None, None, None, None, None

        ### Extract part of image around the mouse cursor ###
        ######################################################################################################

        # Outer circle radius
        outer_radius = self.star_aperture_radius*2

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

        # Crop the image - always use avepixel for intensity calculation to match calstars extraction
        # This ensures consistency regardless of whether maxpixel or avepixel is being displayed
        if hasattr(self.img_handle, 'ff') and self.img_handle.ff is not None:
            # For FF files, always use avepixel (same as calstars extraction)
            avepixel_data = self.img_handle.ff.avepixel.T
            # Apply dark and flat if available (same processing as displayed image)
            if self.dark is not None:
                avepixel_data = applyDark(avepixel_data, self.dark)
            if self.flat_struct is not None:
                avepixel_data = applyFlat(avepixel_data, self.flat_struct)
            img_crop_orig = avepixel_data[x_min:x_max, y_min:y_max]
        else:
            # For other input types (videos, images), use the current displayed image
            img_crop_orig = self.img.data[x_min:x_max, y_min:y_max]

        # Perform gamma correction
        img_crop = gammaCorrectionImage(img_crop_orig, self.config.gamma, out_type=np.float32)


        ######################################################################################################

        ### Estimate the background ###
        ######################################################################################################

        # Create an image mask with the same size as the cropped image
        annulus_mask = np.zeros_like(img_crop)
        aperture_mask = np.zeros_like(img_crop)

        # Create a circular mask
        for i in range(img_crop.shape[0]):
            for j in range(img_crop.shape[1]):

                # Calculate distance of pixel from centre of the cropped image
                i_rel = i - img_crop.shape[0]/2
                j_rel = j - img_crop.shape[1]/2
                pix_dist = math.sqrt(i_rel**2 + j_rel**2)

                # Take only those pixels between the inner and the outer circle
                if (pix_dist <= outer_radius) and (pix_dist > self.star_aperture_radius):
                    annulus_mask[i, j] = 1

                # Take only those pixels within the star aperture radius
                if pix_dist <= self.star_aperture_radius:
                    aperture_mask[i, j] = 1


        # Compute the median of the pixels in the aperture mask as the background
        bg_median = np.median(img_crop[annulus_mask == 1])

        # Compute the standard deviation of the pixels in the aperture mask
        bg_std = np.std(img_crop[annulus_mask == 1])


        ######################################################################################################


        ### Check for saturation ###
        ######################################################################################################
        # If 10 or more pixels are saturated (within 2% of the maximum value), mark the pick as saturated

        # Count the number of pixels above the saturation threshold (original non-gramma corrected image)
        # Apply the mask to only include the pixels within the star aperture radius
        saturated_count = np.sum(img_crop_orig[aperture_mask == 1] > self.saturation_threshold)

        # print("Saturation threshold: {:.2f}, count: {:d}".format(self.saturation_threshold, saturated_count))

        # If 2 or more pixels are saturated, mark the pick as saturated
        min_saturated_px_count = 2
        if saturated_count >= min_saturated_px_count:
            saturated = True
        else:
            saturated = False

        ######################################################################################################


        ### Calculate the centroid using a center of mass method ###
        ######################################################################################################
        x_acc = 0
        y_acc = 0
        source_intens = 0
        source_px_count = 0

        for i in range(img_crop.shape[0]):
            for j in range(img_crop.shape[1]):

                # Calculate distance of pixel from centre of the cropped image
                i_rel = i - img_crop.shape[0]/2
                j_rel = j - img_crop.shape[1]/2
                pix_dist = math.sqrt(i_rel**2 + j_rel**2)

                # Take only those pixels between the inner and the outer circle
                if pix_dist <= self.star_aperture_radius:
                    x_acc += i*(img_crop[i, j] - bg_median)
                    y_acc += j*(img_crop[i, j] - bg_median)
                    source_intens += img_crop[i, j] - bg_median
                    source_px_count += 1

        if source_intens > 0:
            x_centroid = x_acc/source_intens + x_min
            y_centroid = y_acc/source_intens + y_min
        else:
            x_centroid = mouse_x
            y_centroid = mouse_y

        ######################################################################################################

        ### Calculate the FWHM ###

        # Convert centroid to cropped image coordinates
        x_centroid_crop = x_centroid - x_min
        y_centroid_crop = y_centroid - y_min

        # Initialize moment accumulators
        moment_x = 0.0
        moment_y = 0.0
        total_flux = 0.0

        for i in range(img_crop.shape[0]):
            for j in range(img_crop.shape[1]):
                i_rel = i - img_crop.shape[0]/2
                j_rel = j - img_crop.shape[1]/2
                pix_dist = math.sqrt(i_rel**2 + j_rel**2)

                if pix_dist <= self.star_aperture_radius:
                    net_flux = img_crop[i, j] - bg_median
                    if net_flux > 0:
                        dx = i - x_centroid_crop
                        dy = j - y_centroid_crop
                        moment_x += net_flux*dx**2
                        moment_y += net_flux*dy**2
                        total_flux += net_flux

        # Compute sigma and FWHM
        fwhm_x = fwhm_y = fwhm = 0
        if total_flux > 0:

            # Compute the standard deviations in x and y directions
            sigma_x = math.sqrt(moment_x/total_flux)
            sigma_y = math.sqrt(moment_y/total_flux)

            # Compute the circular sigma
            sigma = math.sqrt((moment_x + moment_y)/(2*total_flux))

            # Compute a circular FWHM
            fwhm = 2.355*sigma

        ######################################################################################################


        # ######################################################################################################

        # # Left - image, right - horizontal profile of the star
        # plt.figure(figsize=(12, 6))

        # # Plot the image
        # plt.subplot(1, 2, 1)

        # # Plot a background-subtracted cropped image (colormap red for above zero and blue for below zero)
        # img_ax = plt.imshow(img_crop - bg_median, cmap='bwr')

        # # Plot a circle with the radius of the star aperture
        # circle = plt.Circle((img_crop.shape[1]/2, img_crop.shape[0]/2), self.star_aperture_radius, color='black', fill=False, label='Aperture')
        # plt.gca().add_artist(circle)

        # # Plot the centroid location
        # plt.scatter(y_centroid - y_min, x_centroid - x_min, color='black', s=100, marker='x', label='Centroid')

        # # Plot the sigma as the radius around the centroid
        # circle = plt.Circle((y_centroid - y_min, x_centroid - x_min), sigma, color='orange', fill=False, label='1 Sigma')
        # plt.gca().add_artist(circle)

        # # Plot the FWHM as the radius around the centroid
        # circle = plt.Circle((y_centroid - y_min, x_centroid - x_min), fwhm/2, color='green', fill=False, label='FWHM')
        # plt.gca().add_artist(circle)

        # plt.legend()

        # # Add a colorbar on the brightness of the image
        # plt.colorbar(img_ax, label='Brightness')

        # plt.gca().invert_yaxis()
        # plt.gca().invert_xaxis()


        # # Plot the horizontal profile of the star
        # plt.subplot(1, 2, 2)

        # # Compute the horizontal profile of the star
        # # Cut the image horizontally at the centroid (y_centroid)
        # horizontal_profile = img_crop[int(round(x_centroid - x_min)), :]
        # plt.plot(horizontal_profile - bg_median, label='Horizontal profile')

        # # Plot the background level
        # plt.axhline(y=0, color='black', linestyle='--', label='Background level')

        # # Plot the centroid location
        # plt.axvline(x=y_centroid - y_min, color='black', linestyle=':', label='Centroid')

        # # Plot the FWHM as the radius around the centroid
        # plt.axvline(x=y_centroid - y_min - fwhm/2, color='green', linestyle=':', label='FWHM')
        # plt.axvline(x=y_centroid - y_min + fwhm/2, color='green', linestyle=':')

        # # Plot the sigma as the radius around the centroid
        # plt.axvline(x=y_centroid - y_min - sigma, color='orange', linestyle=':', label='1 Sigma')
        # plt.axvline(x=y_centroid - y_min + sigma, color='orange', linestyle=':')

        # plt.legend()


        # plt.show()

        # ######################################################################################################


        # Compute the SNR using the "CCD equation" (Howell et al., 1989)
        snr = signalToNoise(source_intens, source_px_count, bg_median, bg_std)

        # Debug print
        if prev_x_cent is not None:
            print('Centroid at ({:7.2f}, {:7.2f}), FWHM {:5.2f}, intensity {:9d}, SNR {:6.2f}, saturated: {}'.format(
                x_centroid, y_centroid, fwhm, int(source_intens), snr, saturated))

        return x_centroid, y_centroid, fwhm, source_intens, snr, saturated


    def updateGreatCircle(self):
        """ Fits great circle to observations. """

        # Extract picked points
        good_picks = collections.OrderedDict((frame, pick) for frame, pick in self.pick_list.items() 
                                             if (pick['mode'] == 1) and (pick['x_centroid'] is not None))

        # Remove the old great circle
        self.great_circle_line.clear()

        # If there are more than 2 picks, fit a great circle to the picks
        # If ground points are measured, don't fit the great circle
        if (len(good_picks) > 1) and (not self.meas_ground_points):

            # Sort picks by frame
            good_picks = collections.OrderedDict(sorted(good_picks.items(), key=lambda t: t[0]))

            # Extract X, Y data
            frames = np.array(list(good_picks.keys()))
            x_data = [pick['x_centroid'] for frame, pick in good_picks.items()]
            y_data = [pick['y_centroid'] for frame, pick in good_picks.items()]


            # Compute RA/Dec of the pick if the platepar is available
            if self.platepar is not None:

                # Compute time data
                time_data = [jd2Date(datetime2JD(self.img_handle.beginning_datetime \
                    + datetime.timedelta(seconds=frame/self.img_handle.fps))) for frame in frames]

                # Compute measured RA/Dec from image coordinates
                _, ra_data, dec_data, _ = xyToRaDecPP(time_data, x_data, \
                    y_data, np.ones_like(x_data), self.platepar, measurement=True, \
                    extinction_correction=False, precompute_pointing_corr=True)

                cartesian_points = []

                # Convert equatorial coordinates to a unit direction vector in the ECI frame
                for ra, dec in zip(ra_data, dec_data):

                    vect = vectNorm(raDec2Vector(ra, dec))

                    if np.any(np.isnan(vect)):
                        continue

                    cartesian_points.append(vect)

                cartesian_points = np.array(cartesian_points)

                # Fit a great circle through observations
                x_arr, y_arr, z_arr = cartesian_points.T
                coeffs, theta0, phi0 = fitGreatCircle(x_arr, y_arr, z_arr)

                # # Find the great circle phase angle at the middle of the observation
                # ra_mid = np.mean(ra_data)
                # dec_mid = np.mean(dec_data)
                # ra_list_temp = [ra_data[0], ra_mid, ra_data[-1]]
                # dec_list_temp = [dec_data[0], dec_mid, dec_data[-1]]
                # gc_phases = []

                # for ra, dec in zip(ra_list_temp, dec_list_temp):
                #     x, y, z = raDec2Vector(ra, dec)
                #     theta, phi = cartesianToPolar(x, y, z)
                #     gc_phase = (greatCirclePhase(theta, phi, theta0, phi0)[0])%(2*np.pi)

                #     gc_phases.append(gc_phase)

                # gc_phase_first, gc_phase_mid, gc_phase_last = gc_phases

                # # Get the correct angle order (in the clockwise order: first, mid, last)
                # if isAngleBetween(gc_phase_first, gc_phase_last, gc_phase_mid):
                #     gc_phase_first, gc_phase_last = gc_phase_last, gc_phase_first

                # # Generate a list of phase angles which are +/- 45 degrees from the middle phase and at least
                # # 15 deg from the start and end phases
                # gc_phase_min = gc_phase_mid - np.radians(45)
                # gc_phase_max = gc_phase_mid + np.radians(45)
                # if isAngleBetween(gc_phase_first, gc_phase_min, gc_phase_mid):
                #     gc_phase_min = gc_phase_first - np.radians(15)
                # if isAngleBetween(gc_phase_mid, gc_phase_max, gc_phase_last):
                #     gc_phase_max = gc_phase_last + np.radians(15)

                # # Generate a list of phase angles
                # phase_angles = np.linspace(gc_phase_min, gc_phase_max, 1000)

                # Generate a list of phase angles (full circle)
                phase_angles = np.linspace(0, 2*np.pi, 1000)

                
                # Sample the great circle
                x_array, y_array, z_array = greatCircle(phase_angles, theta0, phi0)

                if isinstance(x_array, float):
                    x_array = [x_array]
                    y_array = [y_array]
                    z_array = [z_array]

                # Compute RA/Dec of every points
                ra_array = []
                dec_array = []
                for x, y, z in zip(x_array, y_array, z_array):
                    ra, dec = vector2RaDec(np.array([x, y, z]))

                    ra_array.append(ra)
                    dec_array.append(dec)


                ra_array, dec_array = np.array(ra_array), np.array(dec_array)

                # Compute image coordinates
                x_array, y_array = raDecToXYPP(ra_array, dec_array, \
                    datetime2JD(self.img_handle.currentFrameTime(dt_obj=True)), self.platepar)

                # Remove points outside the image
                filter_arr = (x_array >= 0) & (x_array <= self.platepar.X_res) & (y_array >= 0) \
                    & (y_array <= self.platepar.Y_res)
                x_array = x_array[filter_arr]
                y_array = y_array[filter_arr]

                connect = np.ones_like(x_array)
                if np.any(~filter_arr):

                    # Find the break in pixel coordinates and create breaks there
                    x_diff = np.abs(np.diff(x_array))
                    y_diff = np.abs(np.diff(y_array))
                    if np.max(x_diff) > np.max(y_diff):
                        break_indx = np.argmax(x_diff)
                    else:
                        break_indx = np.argmax(y_diff)

                    # Find the index where the array breaks
                    connect[break_indx] = 0
                    connect[break_indx-1] = 0

                
                self.great_circle_line.setData(x=x_array + 0.5, y=y_array + 0.5, connect=connect)


                ### ###



        return None



    def findClosestCatalogStarIndex(self, pos_x, pos_y):
        """ Finds the index of the closest catalog star on the image to the given image position. """

        min_index = 0
        min_dist = np.inf
        min_type = 'catalog'

        # Find the index of the closest catalog star to the given image coordinates
        for i, (x, y) in enumerate(zip(self.catalog_x_filtered, self.catalog_y_filtered)):

            dist = (pos_x - x)**2 + (pos_y - y)**2

            if dist < min_dist:
                min_dist = dist
                min_index = i


        # If geo points are given, choose from them
        if self.geo_points_obj is not None:
            for i, (x, y) in enumerate(zip(self.geo_x, self.geo_y)):

                # Only take the star if it's visible in the FOV
                if not self.geo_filtered_indices[i]:
                    continue

                dist = (pos_x - x)**2 + (pos_y - y)**2

                if dist < min_dist:
                    min_dist = dist
                    min_index = i
                    min_type = 'geo'


        return min_type, min_index

    def getMinFitStars(self):
        """ Returns the minimum number of stars needed for the fit. """

        #   - if fitting only the pointing and no scale and no distortion, then require 2 stars
        #   - if the scale is also to be fitted but no distortion, then require 3 stars
        #   - if the distortion is also to be fitted, then require 5 stars
        if self.fit_only_pointing and self.fixed_scale:
            min_stars = 2
        elif self.fit_only_pointing and not self.fixed_scale:
            min_stars = 3
        else:
            min_stars = 5

        return min_stars


    def fitPickedStars(self):
        """ Fit stars that are manually picked. The function first only estimates the astrometry parameters
            without the distortion, then just the distortion parameters, then all together.

        """

        # Check if there are enough stars for the fit
        min_stars = self.getMinFitStars()
        if len(self.paired_stars) < min_stars:

            qmessagebox(title='Number of stars', 
                        message="At least {:d} paired stars are needed to do the fit!".format(min_stars), 
                        message_type="warning")

            return self.platepar

        print()
        print("----------------------------------------")
        print("Fitting platepar...")

        # Extract paired catalog stars and image coordinates separately
        img_stars = np.array(self.paired_stars.imageCoords())
        catalog_stars = np.array(self.paired_stars.skyCoords())

        # Get the Julian date of the image that's being fit
        jd = date2JD(*self.img_handle.currentTime())

        # Fit the platepar to paired stars
        self.platepar.fitAstrometry(jd, img_stars, catalog_stars, first_platepar_fit=self.first_platepar_fit,\
            fit_only_pointing=self.fit_only_pointing, fixed_scale=self.fixed_scale)
        self.first_platepar_fit = False

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



        # ### TEST ###

        # print("time:", self.img_handle.currentTime())
        # print("jd:", jd)
        # print("LST:", JD2LST(jd, self.platepar.lon)[0])
        # print()

        # print("RA_J2000, Dec_J2000, RA_date, Dec_date, RA_ref, Dec_ref, Azim_ref, Elev_ref")
        # for cat_coords in catalog_stars:
        #     ra, dec, _ = cat_coords

        #     # Precess to epoch of date
        #     ra_date, dec_date = equatorialCoordPrecession(2451545.0, jd, np.radians(ra), np.radians(dec))
        #     ra_date, dec_date = np.degrees(ra_date), np.degrees(dec_date)

        #     # Compute apparent RA/Dec with the applied refraction
        #     azim, elev = cyTrueRaDec2ApparentAltAz(np.radians(ra), np.radians(dec), jd, np.radians(self.platepar.lat), np.radians(self.platepar.lon), refraction=True)
        #     ra_ref, dec_ref = cyaltAz2RADec(azim, elev, jd, np.radians(self.platepar.lat), np.radians(self.platepar.lon))
        #     azim, elev = np.degrees(azim), np.degrees(elev)
        #     ra_ref, dec_ref = np.degrees(ra_ref), np.degrees(dec_ref)


        #     print("{:>12.6f}, {:>+13.6f}, {:>12.6f}, {:>+13.6f}, {:>12.6f}, {:>+13.6f}, {:>12.6f}, {:>+13.6f}".format(ra, dec, ra_date, dec_date, ra_ref, dec_ref, azim, elev))

        # ### ###



        print()
        print("Image time =", self.img_handle.currentTime(dt_obj=True), "UTC")
        print("Image JD = {:.8f}".format(jd))
        print("Image LST = {:.8f}".format(JD2LST(jd, self.platepar.lon)[0]))

        residuals = []

        print()
        print('Residuals')
        print('----------')
        print(
            ' No,       Img X,       Img Y, RA cat (deg), Dec cat (deg),    Cat X,   Cat Y, RA img (deg), Dec img (deg), Err amin,  Err px, Direction,  FWHM,    Mag, -2.5*LSP, Mag err,    SNR, Saturated')

        # Use zeros for photometry residuals if not available or has wrong length
        # (e.g., after auto-fit before photometry fit, or after RANSAC removed stars)
        if self.photom_fit_resids is not None and len(self.photom_fit_resids) == len(catalog_x):
            photom_resids = self.photom_fit_resids
        else:
            photom_resids = [0.0]*len(catalog_x)

        # Get all coordinates from paired stars
        all_coords = self.paired_stars.allCoords()

        # Calculate the distance and the angle between each pair of image positions and catalog predictions
        for star_no, (cat_x, cat_y, cat_coords, paired_stars, mag_err) in enumerate(
            zip(catalog_x, catalog_y, catalog_stars, all_coords, photom_resids)):

            img_x, img_y, fwhm, sum_intens, snr, saturated = paired_stars[0]
            ra, dec, mag = cat_coords

            # Compute magnitude error from SNR (same formula as used in lightcurve plots and FTPdetectinfo)
            if snr > 0:
                mag_err_random = 2.5*np.log10(1 + 1/snr)
                mag_err = np.sqrt(mag_err_random**2 + self.platepar.mag_lev_stddev**2)
            else:
                mag_err = 0.0

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

            lsp = -2.5*np.log10(sum_intens)

            # Print out the residuals
            print(
                '{:3d}, {:11.6f}, {:11.6f}, {:>12.6f}, {:>+13.6f}, {:8.2f}, {:7.2f}, {:>12.6f}, {:>+13.6f}, {:8.2f}, {:7.2f}, {:+9.1f}, {:5.2f}, {:+6.2f}, {:8.2f}, {:+7.2f}, {:6.1f}, {:9s}'.format(
                    star_no + 1, img_x, img_y, ra, dec, cat_x, cat_y, \
                    ra_img, dec_img, 60*angular_distance, distance, np.degrees(angle),
                    fwhm, mag, lsp, mag_err, snr, str(saturated)
                    )
                )


        # Compute RMSD errors
        rmsd_angular = 60*RMSD([entry[4] for entry in residuals])
        rmsd_img = RMSD([entry[3] for entry in residuals])

        # If the average angular error is larger than 60 arc minutes, report it in degrees
        if rmsd_angular > 60:
            rmsd_angular /= 60
            angular_error_label = 'deg'

        elif rmsd_angular > 0.5:
            angular_error_label = 'arcmin'

        else:
            rmsd_angular *= 60
            angular_error_label = 'arcsec'


        print('RMSD: {:.2f} px, {:.2f} {:s}'.format(rmsd_img, rmsd_angular, angular_error_label))

        # Update RMSD display in the Fit Parameters tab
        self.tab.param_manager.updateRMSD(rmsd_img, rmsd_angular, angular_error_label)

        # Update fit residuals in the station tab when geopoints are used
        if self.geo_points_obj is not None:
            self.tab.geolocation.residuals_label.setText("Residuals:\n{:.2f} px, {:.2f} {:s}".format(rmsd_img,\
                rmsd_angular, angular_error_label))

        # Print the field of view size
        #print("FOV: {:.2f} x {:.2f} deg".format(*computeFOVSize(self.platepar))) 

        ####################

        # Save the residuals
        self.residuals = residuals

        self.updateDistortion()
        self.updateLeftLabels()
        self.updateStars()
        self.updateFitResiduals()
        self.tab.param_manager.updatePlatepar()


    def jumpNextStar(self, miss_this_one=False):

        new_x, new_y, self.max_pixels_between_matched_stars  = self.furthestStar(miss_this_one=miss_this_one)
        self.updateBottomLabel()
        self.old_autopan_x, self.old_autopan_y = self.current_autopan_x, self.current_autopan_y
        self.current_autopan_x, self.current_autopan_y = new_x, new_y
        self.img_frame.setRange(xRange=(new_x + 15, new_x - 15), yRange=(new_y + 15, new_y - 15))
        self.checkParamRange()
        self.platepar.updateRefRADec(preserve_rotation=True)
        self.checkParamRange()
        self.tab.param_manager.updatePlatepar()
        self.updateLeftLabels()
        self.updateStars()

    def showAstrometryFitPlots(self):
        """ Show window with astrometry fit details. """

        # Extract paired catalog stars and image coordinates separately (with SNR and saturation)
        all_coords = list(self.paired_stars.allCoords())
        img_stars = np.array([img_c[:3] for img_c, _ in all_coords])  # x, y, fwhm
        catalog_stars = np.array([sky_c for _, sky_c in all_coords])
        snr_data = [img_c[4] for img_c, _ in all_coords]  # SNR is index 4
        saturated_data = [img_c[5] for img_c, _ in all_coords]  # Saturated is index 5

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
        snr_list = []
        mag_list = []
        saturated_list = []
        fwhm_list = []
        total_error_px = []  # Total error in pixels for SNR/mag/FWHM plots

        # Get image time and Julian date
        img_time = self.img_handle.currentTime()
        jd = date2JD(*img_time)

        # Get RA/Dec of the FOV centre
        ra_centre, dec_centre = self.computeCentreRADec()

        # Calculate the distance and the angle between each pair of image positions and catalog predictions
        for star_no, (cat_x, cat_y, cat_coords, img_c, snr, saturated) in enumerate(zip(
                catalog_x, catalog_y, catalog_stars, img_stars, snr_data, saturated_data)):
            # Compute image coordinates
            img_x, img_y, img_fwhm = img_c
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

            # Collect SNR, magnitude, saturation, FWHM data
            snr_list.append(snr)
            mag_list.append(cat_coords[2])  # Catalog magnitude
            saturated_list.append(saturated)
            fwhm_list.append(img_fwhm)

            # Compute total error in pixels (distance between catalog and image positions)
            total_error_px.append(np.hypot(cat_x - img_x, cat_y - img_y))

        # Init astrometry fit window (3 rows: angular, pixel, SNR/mag/FWHM)
        fig_a, (
            (ax_azim, ax_elev, ax_skyradius),
            (ax_x, ax_y, ax_radius),
            (ax_snr, ax_mag, ax_fwhm)
        ) = plt.subplots(ncols=3, nrows=3, facecolor=None, figsize=(12, 9))

        # Set figure title
        try:
            fig_a.canvas.set_window_title("Astrometry fit")

        except AttributeError:

            fig_a.canvas.manager.window.setWindowTitle("Astrometry fit")

        except:

            # Handle FigureCanvasQTAgg error on some versions of Qt
            print("Failed to set the window title!")

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

        # Equalize Y limits, make them multiples of 5 arcmin, and set a minimum range of 5, 2, or 1 arcmin
        azim_max_ylim = np.max(np.abs(ax_azim.get_ylim()))
        elev_max_ylim = np.max(np.abs(ax_elev.get_ylim()))
        skyradius_max_ylim = np.max(np.abs(ax_skyradius.get_ylim()))
        max_ylim = np.max([azim_max_ylim, elev_max_ylim, skyradius_max_ylim])

        if max_ylim < 0.5:
            max_ylim = 0.5

        elif max_ylim < 2.0:
            max_ylim = 2.0

        elif max_ylim < 5.0:
            max_ylim = 5.0

        else:

            # Make it a multiple of 5 arcmin if errors are large
            max_ylim = np.ceil(max_ylim/5)*5

        
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

        # Plot error vs SNR
        ax_snr.scatter(snr_list, total_error_px, s=2, c='k', zorder=3)

        ax_snr.grid(alpha=0.3)
        ax_snr.set_xlabel("S/N")
        ax_snr.set_ylabel("Error (px)")
        ax_snr.set_xscale('log')
        if len(snr_list) > 0 and min(snr_list) > 0:
            snr_min, snr_max = min(snr_list), max(snr_list)
            snr_range = snr_max - snr_min
            # Find nice tick spacing closest to 6 ticks using 1-2-5 sequence
            nice_spacings = [0.1 * (10 ** p) * m for p in range(-1, 5) for m in [1, 2, 5]]
            target_ticks = 6
            best_spacing = nice_spacings[0]
            best_diff = float('inf')
            for spacing in nice_spacings:
                num_ticks = snr_range / spacing
                if abs(num_ticks - target_ticks) < best_diff:
                    best_diff = abs(num_ticks - target_ticks)
                    best_spacing = spacing
            # Generate tick positions
            tick_start = np.floor(snr_min / best_spacing) * best_spacing
            tick_end = np.ceil(snr_max / best_spacing) * best_spacing
            ticks = np.arange(tick_start, tick_end + best_spacing/2, best_spacing)
            # Filter out zero/negative ticks for log scale
            ticks = ticks[ticks > 0]
            ax_snr.set_xticks(ticks)
            ax_snr.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:g}'))
            ax_snr.xaxis.set_minor_locator(ticker.NullLocator())
            ax_snr.set_xlim([snr_min * 0.8, snr_max * 1.2])

        # Plot error vs magnitude (saturated stars in red)
        mag_arr = np.array(mag_list)
        sat_arr = np.array(saturated_list)
        err_arr = np.array(total_error_px)

        # Plot non-saturated stars in black
        if np.sum(~sat_arr) > 0:
            ax_mag.scatter(mag_arr[~sat_arr], err_arr[~sat_arr], s=2, c='k', zorder=3, label='Normal')
        # Plot saturated stars in red
        if np.sum(sat_arr) > 0:
            ax_mag.scatter(mag_arr[sat_arr], err_arr[sat_arr], s=2, c='r', zorder=4, label='Saturated')
            ax_mag.legend(loc='upper right', fontsize=8, markerscale=3)

        ax_mag.grid()
        ax_mag.set_xlabel("Magnitude")
        ax_mag.set_ylabel("Error (px)")
        ax_mag.invert_xaxis()  # Bright stars (low mag) on right, faint (high mag) on left

        # Plot error vs FWHM
        fwhm_arr = np.array(fwhm_list)
        valid_fwhm = fwhm_arr > 0  # Filter out invalid FWHM values
        if np.sum(valid_fwhm) > 0:
            ax_fwhm.scatter(fwhm_arr[valid_fwhm], err_arr[valid_fwhm], s=2, c='k', zorder=3)

        ax_fwhm.grid()
        ax_fwhm.set_xlabel("FWHM (px)")
        ax_fwhm.set_ylabel("Error (px)")
        if np.sum(valid_fwhm) > 0:
            ax_fwhm.set_xlim([0, np.max(fwhm_arr[valid_fwhm]) * 1.1])

        # Equalize Y limits, make them integers, and set a minimum range of 1 px
        x_max_ylim = np.max(np.abs(ax_x.get_ylim()))
        y_max_ylim = np.max(np.abs(ax_y.get_ylim()))
        radius_max_ylim = np.max(np.abs(ax_radius.get_ylim()))
        snr_max_ylim = np.max(total_error_px) if len(total_error_px) > 0 else 1.0
        max_ylim_px = np.ceil(np.max([x_max_ylim, y_max_ylim, radius_max_ylim, snr_max_ylim]))
        if max_ylim_px < 1:
            max_ylim_px = 1.0
        ax_x.set_ylim([-max_ylim_px, max_ylim_px])
        ax_y.set_ylim([-max_ylim_px, max_ylim_px])
        ax_radius.set_ylim([-max_ylim_px, max_ylim_px])
        ax_snr.set_ylim([0, max_ylim_px])
        ax_mag.set_ylim([0, max_ylim_px])
        ax_fwhm.set_ylim([0, max_ylim_px])

        fig_a.tight_layout()
        fig_a.show()


    def computeIntensitySum(self, star_mask_coeff=3):
        """ Compute the background subtracted sum of intensity of colored pixels. The background is estimated
            as the median of near pixels that are not colored.
            args:
                star_mask_coeff (float): Mask out parts of the image with stars by masking out a region 
                    where star_mask_coeff x stddev > average.
        """

        # Find the pick done on the current frame
        pick = self.getCurrentPick()

        if pick:
            photom_pixels_raw = pick.get('photometry_pixels')

            # If there are no photometry pixels, set the intensity to 0
            if photom_pixels_raw is None:
                pick['intensity_sum'] = 1
                return None

            if isinstance(photom_pixels_raw, np.ndarray):
                has_pixels = photom_pixels_raw.size > 0
            else:
                try:
                    has_pixels = len(photom_pixels_raw) > 0
                except TypeError:
                    has_pixels = bool(photom_pixels_raw)

            if not has_pixels:
                pick['intensity_sum'] = 1
                return None

            photom_pixels = np.asarray(photom_pixels_raw, dtype=np.int64)
            if photom_pixels.size == 0:
                pick['intensity_sum'] = 1
                return None

            if photom_pixels.ndim != 2 or photom_pixels.shape[1] != 2:
                try:
                    photom_pixels = np.reshape(photom_pixels, (-1, 2))
                except ValueError:
                    pick['intensity_sum'] = 1
                    return None

            if photom_pixels.shape[0] == 0:
                pick['intensity_sum'] = 1
                return None

            x_arr_global, y_arr_global = photom_pixels.T

            # Compute the centre of the colored pixels
            x_centre = np.mean(x_arr_global)
            y_centre = np.mean(y_arr_global)

            # Take a window twice the size of the colored pixels
            x_color_size = np.max(x_arr_global) - np.min(x_arr_global)
            y_color_size = np.max(y_arr_global) - np.min(y_arr_global)

            x_min = int(np.floor(x_centre - x_color_size))
            x_max = int(np.ceil(x_centre + x_color_size)) + 1
            y_min = int(np.floor(y_centre - y_color_size))
            y_max = int(np.ceil(y_centre + y_color_size)) + 1

            # Limit the size to be within the bounds
            if x_min < 0: x_min = 0
            if x_max > self.img.data.shape[0]: x_max = self.img.data.shape[0]
            if y_min < 0: y_min = 0
            if y_max > self.img.data.shape[1]: y_max = self.img.data.shape[1]

            if x_max <= x_min:
                x_max = min(self.img.data.shape[0], x_min + 1)
            if y_max <= y_min:
                y_max = min(self.img.data.shape[1], y_min + 1)

            x_arr = (x_arr_global - x_min).astype(np.int64, copy=False)
            y_arr = (y_arr_global - y_min).astype(np.int64, copy=False)

            valid_mask = (
                (x_arr >= 0) & (x_arr < (x_max - x_min)) &
                (y_arr >= 0) & (y_arr < (y_max - y_min))
            )

            if not np.all(valid_mask):
                x_arr = x_arr[valid_mask]
                y_arr = y_arr[valid_mask]
                x_arr_global = x_arr_global[valid_mask]
                y_arr_global = y_arr_global[valid_mask]
                if x_arr.size == 0:
                    pick['intensity_sum'] = 1
                    return None

            pick['photometry_pixels'] = list(map(tuple, np.stack([x_arr_global, y_arr_global], axis=-1)))

            # Take only the colored part
            mask_img = np.ones(self.img.data.shape, dtype=bool)
            mask_img[x_arr_global, y_arr_global] = False
            masked_img = np.ma.masked_array(self.img.data, mask_img)
            crop_img = masked_img[x_min:x_max, y_min:y_max]

            # Perform gamma correction on the colored part
            crop_img = gammaCorrectionImage(
                crop_img, self.config.gamma, 
                bp=0, wp=(2**self.config.bit_depth - 1), 
                out_type=np.float32
                )

            # Mask out the colored in pixels
            mask_img_bg = np.zeros(self.img.data.shape, dtype=bool)
            mask_img_bg[x_arr_global, y_arr_global] = True

            # Take the image where the colored part is masked out and crop the surroundings
            masked_img_bg = np.ma.masked_array(self.img.data, mask_img_bg)
            crop_bg = masked_img_bg[x_min:x_max, y_min:y_max]

            # Perform gamma correction on the background
            crop_bg = gammaCorrectionImage(
                crop_bg, self.config.gamma, 
                bp=0, wp=(2**self.config.bit_depth - 1),
                out_type=np.float32
                )

            # Compute the median background
            background_lvl = np.ma.median(crop_bg)

            # Store the background intensity in the pick
            pick['background_intensity'] = background_lvl


            # If the DFN image is used and a dark has been applied (i.e. the previous image is subtracted),
            #   assume that the background is zero
            if (self.img_handle.input_type == "dfn") and (self.dark is not None):
                background_lvl = 0

            # If the nobg flag is set, assume that the background is zero.
            # This is useful when the background is already subtracted or saturated objects are being
            #  measured
            if self.no_background_subtraction:
                background_lvl = 0


            # If the background level is set to zero, simply sum up the intensity of the colored pixels
            if background_lvl == 0:
                intensity_sum = np.ma.sum(crop_img)

            # Use the peripheral background subtraction method if forced or on static images
            elif self.peripheral_background_subtraction \
                or (self.img_handle.input_type == "dfn") \
                or (hasattr(self.img_handle, "single_image_mode") and  self.img_handle.single_image_mode):

                # Compute the background subtracted intensity sum by using pixels peripheral to the colored 
                # pixels
                # (do as a float to avoid artificially pumping up the magnitude)
                crop_img_nobg = crop_img.astype(float) - background_lvl
                intensity_sum = np.ma.sum(crop_img_nobg)
                intensity_sum = np.abs(intensity_sum)

            # Subtract the background using the avepixel
            else:

                # Get the avepixel image and apply a dark and flat to it if needed
                avepixel = self.img_handle.ff.avepixel.T

                if self.dark is not None:
                    avepixel = applyDark(avepixel, self.dark)
                if self.flat_struct is not None:
                    avepixel = applyFlat(avepixel, self.flat_struct)

                
                ### Create star mask to remove bright stars from affecting the centroid and photometry ###
                
                # Don't allow the star mask if the FR file is being used as the bright fireball track can
                # affect the avepixel significantly
                # Also don't allow on static images and they have the bright fireball track
                if ((self.img_handle.input_type == "ff") and self.img_handle.use_fr_files) \
                    or (self.img_handle.input_type == "dfn") or (self.img_handle.input_type == "images"):

                    star_mask = np.zeros_like(avepixel.copy(), dtype=bool)
                    crop_star_mask = np.zeros((x_max - x_min, y_max - y_min), dtype=bool)
                
                else:

                    # Create the star mask and mask out bright stars from the avepixel
                    star_mask = np.zeros_like(avepixel.copy(), dtype=int)
                    star_mask[avepixel > (np.median(avepixel) + star_mask_coeff*np.std(avepixel))] = 1
                    crop_star_mask = star_mask[x_min:x_max, y_min:y_max]

                ### ###
                
                # Add the star mask & mask_img to the avepixel mask
                avepixel_masked = np.ma.masked_array(avepixel, mask_img | star_mask)
                avepixel_crop = avepixel_masked[x_min:x_max, y_min:y_max]

                # Perform gamma correction on the avepixel crop
                avepixel_crop = gammaCorrectionImage(
                    avepixel_crop, self.config.gamma, 
                    bp=0, wp=(2**self.config.bit_depth - 1),
                    out_type=np.float32
                    )

                background_lvl = np.ma.median(avepixel_crop)

                # Correct the crop_img with star_mask
                crop_mask_img = mask_img[x_min:x_max, y_min:y_max]
                crop_img = np.ma.masked_array(crop_img, crop_mask_img | crop_star_mask)

                # Replace photometry pixels that are masked by a star with the median value of the photom. area
                photom_star_masked_indices = np.where((crop_mask_img == 0) & (crop_star_mask == 1))

                # Apply correction only if the streak intersects a star
                if len(photom_star_masked_indices[0]) > 0:

                    # Calulate masked median
                    masked_stars_streak_median = np.ma.median(crop_img)

                    # Unmask those areas
                    crop_img.mask[photom_star_masked_indices] = False

                    # Replace with median
                    crop_img[photom_star_masked_indices] = masked_stars_streak_median

                # Correct the crop_bg with star_mask
                crop_mask_img_bg = mask_img_bg[x_min:x_max, y_min:y_max]
                crop_bg = np.ma.masked_array(crop_bg, crop_mask_img_bg | crop_star_mask)

                # Subtract the avepixel crop from the data crop
                crop_img_nobg = crop_img.astype(float) - avepixel_crop
                intensity_sum = np.ma.sum(crop_img_nobg)
                intensity_sum = np.abs(intensity_sum)

            # Check if the result is masked
            if np.ma.is_masked(intensity_sum):

                # If the result is masked (i.e. error reading pixels), set the intensity sum to 1
                intensity_sum = 1

            # If the intensity sum is a numpy object, set it to int
            elif isinstance(intensity_sum, np.ndarray):
                intensity_sum = intensity_sum.astype(int)

            else:
                intensity_sum = int(intensity_sum)

            # Set the intensity sum to the pick
            pick['intensity_sum'] = intensity_sum

            ### Measure the SNR of the pick ###

            # Compute the standard deviation of the background
            background_stddev = np.ma.std(crop_bg)

            # Count the number of pixels in the photometric area
            source_px_count = np.ma.sum(~crop_img.mask)

            # Compute the signal to noise ratio using the CCD equation
            snr = signalToNoise(intensity_sum, source_px_count, background_lvl, background_stddev)

            # Set the SNR to the pick
            pick['snr'] = snr

            # Debug print
            print("SNR update: intensity sum = {:8d}, source px count = {:5d}, background lvl = {:8.2f}, background stddev = {:6.2f}, SNR = {:.2f}".format(
                intensity_sum, source_px_count, background_lvl, background_stddev, snr))


            # # Plot the image on which the intensity sum was computed: crop_img, avepixel_crop, and crop_img_nobg
            # plt.figure("Intensity sum computation")
            # plt.clf()
            # plt.subplot(1, 3, 1)
            # plt.imshow(crop_img, cmap='gray', origin='lower')
            # plt.colorbar()
            # plt.title("Crop used for intensity sum computation\nIntensity sum = {:d}, Background = {:.2f}, SNR = {:.2f}".format(
            #     intensity_sum, background_lvl, snr))
            # plt.subplot(1, 3, 2)
            # plt.imshow(avepixel_crop, cmap='gray', origin='lower')
            # plt.colorbar()
            # plt.title("Average pixel crop")
            # plt.subplot(1, 3, 3)
            # plt.imshow(crop_img_nobg, cmap='gray', origin='lower')
            # plt.colorbar()
            # plt.title("Background subtracted crop")
            # plt.show()

            ### Determine if there is any saturation in the measured photometric area

            # If at least 2 pixels are saturated in the photometric area, mark the pick as saturated
            if np.sum(crop_img > self.saturation_threshold) >= 2:
                pick['saturated'] = True
            else:
                pick['saturated'] = False

            ###


            # If the DFN image is used, correct intensity sum for exposure difference
            # Of the total 27 second, the stars are exposed 4.31 seconds, and every fireball dot is exposed
            #    a total of 0.01 seconds. Thus the correction factor is 431
            if (self.img_handle.input_type == "dfn"):
                pick['intensity_sum'] *= 431


            # Make sure the intensity sum is never 0
            if pick['intensity_sum'] <= 0:
                pick['intensity_sum'] = 1


    def computeExposureRatioCorrection(self):
        """ Compute the exposure ratio magnitude correction. """

        if self.exposure_ratio <= 0:
            return 0.0

        return -2.5*np.log10(self.exposure_ratio)


    def showLightcurve(self):
        """ Show the meteor lightcurve. """

        # The DFN light curve can only be computed if the background image is subtracted
        if (self.img_handle.input_type == "dfn") and (self.dark is None):
            
            qmessagebox(title='DFN light curve',
                        message='The DFN light curve can only be computed if the background is subtracted! Load the previous or next image as a dark.',
                        message_type="info")

            return None

        # Compute the intensity sum done on the previous frame
        self.computeIntensitySum()

        # Create the list of picks for saving
        centroids = []
        for frame, pick in self.pick_list.items():

            # Skip None entries
            if (pick['x_centroid'] is None) or (pick['y_centroid'] is None):
                continue

            # Only show real picks, and not gaps
            if pick['mode'] == 0:
                continue

            centroids.append([frame, pick['x_centroid'], pick['y_centroid'],
                              pick['intensity_sum'], pick['snr'], pick['saturated']])

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
        frames, x_centroids, y_centroids, intensities, snr, saturated = np.array(fr_intens).T

        # Convert the saturated array to bool where >0.5 is saturated
        saturated = saturated > 0.5

        # Init plot
        fig_p = plt.figure(facecolor=None)
        ax_p = fig_p.add_subplot(1, 1, 1)

        # If the platepar is available, compute the magnitudes, otherwise show the instrumental magnitude
        if self.platepar is not None:

            time_data = [self.img.img_handle.currentFrameTime(fr) for fr in frames]

            # Compute the magnitudes
            _, _, _, mag_data = xyToRaDecPP(time_data, x_centroids, y_centroids, intensities, self.platepar)

            # Compute the total magnitude error as the combination of the photometric fit errors and SNR
            mag_err_random = 2.5*np.log10(1 + 1/snr)
            mag_err_total = np.sqrt(mag_err_random**2 + self.platepar.mag_lev_stddev**2)

            # Apply exposure ratio correction
            mag_data += self.computeExposureRatioCorrection()

            # Plot the magnitudes
            ax_p.errorbar(frames, mag_data, yerr=mag_err_total, capsize=5, color='k')

            # Mark saturated points in red
            if np.any(saturated):
                ax_p.scatter(frames[saturated], mag_data[saturated], color='r', zorder=3, alpha=0.5)

            if 'BSC' in self.config.star_catalog_file:
                mag_str = "V"

            elif 'gaia' in self.config.star_catalog_file.lower():
                mag_str = 'GAIA G band'

            else:

                # If there are only 4 band ratios, assume BVRI
                if len(self.config.star_catalog_band_ratios) == 4:
                    mag_str = "{:.2f}B + {:.2f}V + {:.2f}R + {:.2f}I".format(*self.config.star_catalog_band_ratios)

                # If there are 7, assume BVRI + G + Bp + Rp. Only take non-zero coefficients
                elif len(self.config.star_catalog_band_ratios) == 7:
                    band_names = ['B', 'V', 'R', 'I', 'G', 'Bp', 'Rp']
                    mag_str = " + ".join(["{:.2f}{}".format(ratio, band) for ratio, band in 
                                          zip(self.config.star_catalog_band_ratios, band_names) if ratio > 0])

            ax_p.set_ylabel("Apparent magnitude ({:s})".format(mag_str))

        else:

            # Compute the instrumental magnitude
            inst_mag = -2.5*np.log10(intensities)

            # Apply exposure ratio correction
            inst_mag += self.computeExposureRatioCorrection()

            # Compute the SNR error
            mag_err_random = 2.5*np.log10(1 + 1/snr)

            # Plot the magnitudes
            ax_p.errorbar(frames, inst_mag, yerr=mag_err_random, capsize=5, color='k')

            # Mark saturated points in red
            ax_p.scatter(frames[saturated], inst_mag[saturated], color='r', zorder=3, alpha=0.5)

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
                    'background_intensity': None,
                    'snr': 1.0,
                    'saturated': False,
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

        x_list = range(mouse_x - int(self.star_aperture_radius), mouse_x \
                       + int(self.star_aperture_radius) + 1)
        y_list = range(mouse_y - int(self.star_aperture_radius), mouse_y \
                       + int(self.star_aperture_radius) + 1)

        for x in x_list:
            for y in y_list:

                # Skip pixels outside the image
                if (x < 0) or (x >= self.img.data.shape[0]) or (y < 0) or (y >= self.img.data.shape[1]):
                    continue

                # Check if the given pixels are within the aperture radius
                if ((x - mouse_x)**2 + (y - mouse_y)**2) <= self.star_aperture_radius**2:
                    pixel_list.append((x, y))

        ##########
        return pixel_list

    def drawPhotometryColoring(self):
        """ Updates image to have the colouring in the current frame """

        def set_region_image(image):
            self.region.setImage(image)
            if hasattr(self, "region_zoom"):
                self.region_zoom.setImage(image)

        blank_mask = np.array([[0]], dtype=np.uint8)
        pick = self.getCurrentPick()

        if not pick:
            set_region_image(blank_mask)
            return

        photom_pixels = pick.get('photometry_pixels')
        if photom_pixels is None:
            set_region_image(blank_mask)
            return

        photom_pixels = np.asarray(photom_pixels)
        if photom_pixels.size == 0:
            set_region_image(blank_mask)
            return

        if photom_pixels.ndim != 2 or photom_pixels.shape[1] != 2:
            try:
                photom_pixels = np.reshape(photom_pixels, (-1, 2))
            except ValueError:
                set_region_image(blank_mask)
                return

        if photom_pixels.shape[0] == 0:
            set_region_image(blank_mask)
            return

        photom_pixels = photom_pixels.astype(int, copy=False)

        # Clip any out-of-bounds pixels to the image extent to avoid indexing errors when drawing
        height, width = self.img.data.shape[:2]
        valid_mask = (
            (photom_pixels[:, 0] >= 0) & (photom_pixels[:, 0] < height) &
            (photom_pixels[:, 1] >= 0) & (photom_pixels[:, 1] < width)
        )

        if not np.any(valid_mask):
            set_region_image(blank_mask)
            return

        photom_pixels = photom_pixels[valid_mask]

        # Create a coloring mask
        mask_img = np.zeros(self.img.data.shape, dtype=np.uint8)
        x_mask, y_mask = photom_pixels.T
        mask_img[x_mask, y_mask] = 255

        set_region_image(mask_img)


    def saveFTPdetectinfo(self, ECSV_saved=True):
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
            ff_name_ftp = constructFFName(self.platepar.station_code, 
                                          self.img_handle.beginning_datetime)

        print(self.img_handle.beginning_datetime.strftime("%Y%m%d_%H%M%S_"))

        # Create the list of picks for saving
        centroids = []
        for frame, pick in sorted(self.pick_list.items(), key=lambda x: x[0]):

            # Make sure to centroid is picked and is not just the photometry
            if pick['x_centroid'] is None:
                continue

            # Only store real picks, and not gaps
            if pick['mode'] == 0:
                continue

            # Normalize the frame number to the actual time
            frame_dt = self.img_handle.currentFrameTime(frame_no=frame, dt_obj=True)
            frame_no = (frame_dt - self.img_handle.beginning_datetime).total_seconds()*self.img_handle.fps


            # Get the rolling shutter corrected (or not, depending on the config) frame number
            if self.config.deinterlace_order == -1:
                frame_no = self.getRollingShutterCorrectedFrameNo(frame_no, pick)

            # If the global shutter is used, the frame number can only be an integer
            if self.config.deinterlace_order == -2:
                frame_no = round(frame_no, 0)

            # If the intensity sum is masked, assume it's 1
            if np.ma.is_masked(pick['intensity_sum']):
                pick['intensity_sum'] = 1

            centroids.append([
                frame_no, 
                pick['x_centroid'], pick['y_centroid'], 
                pick['intensity_sum'], pick['background_intensity'], pick['snr'], pick['saturated']
                ])

        # If there are no centroids, don't save anything
        if len(centroids) == 0:
            
            # If FTP not saved, ECSV was saved
            if ECSV_saved:
                qmessagebox(title='FTPdetectinfo saving error',
                            message='No centroids to save! FTPdetectinfo saving aborted',
                            message_type="info")
                
            # Neither saved
            if not ECSV_saved:
                qmessagebox(title='FTPdetectinfo/ECSV saving error',
                            message='No centroids to save! ECSV & FTPdetectinfo saving aborted',
                            message_type="info")

            return 1
        
        # FTP saved but ECSV not saved
        elif not ECSV_saved:
            qmessagebox(title='ECSV saving error',
                        message='No centroids to save! ECSV saving aborted.',
                        message_type="info")

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

            # Use a modified platepar if ground points are being picked
            pp_tmp = copy.deepcopy(self.platepar)
            if self.meas_ground_points:
                pp_tmp.switchToGroundPicks()

            applyAstrometryFTPdetectinfo(self.dir_path, ftpdetectinfo_name, '', \
                                         UT_corr=pp_tmp.UT_corr, platepar=pp_tmp, 
                                         exp_mag_corr=self.computeExposureRatioCorrection())

            print('Platepar applied to manual picks!')


    def loadSatelliteTracks(self):
        """ Calculates satellite tracks for the current time and location. """
        
        if not self.show_sattracks or not SKYFIELD_AVAILABLE:
            return

        if self.platepar is None:
            return

        print("Loading satellite tracks...")
        
        # Get location from Platepar
        lat = self.platepar.lat
        lon = self.platepar.lon
        elev = self.platepar.elev
        
        # Determine time range
        t_start = None
        t_end = None
        
        try:
            # Use manually provided start time if available
            if hasattr(self, 'beginning_time') and self.beginning_time is not None:
                t_start = self.beginning_time
                if t_start.tzinfo is None:
                    t_start = t_start.replace(tzinfo=datetime.timezone.utc)
                  
                # Determine end time from total frames and FPS, or fallback
                if hasattr(self, 'img_handle') and hasattr(self.img_handle, 'total_frames') and hasattr(self.img_handle, 'fps') and self.img_handle.fps > 0:
                    duration = self.img_handle.total_frames/self.img_handle.fps
                    t_end = t_start + datetime.timedelta(seconds=duration)
                else:
                    t_end = t_start + datetime.timedelta(seconds=60)

            # Use current frame time as start
            elif hasattr(self, 'img_handle'):
                  
                # Get the time of the first frame
                t_start = self.img_handle.currentFrameTime(frame_no=0, dt_obj=True)
                  
                # Ensure timezone is UTC
                if t_start.tzinfo is None:
                    t_start = t_start.replace(tzinfo=datetime.timezone.utc)

                # Determine end time from total frames and FPS
                if hasattr(self.img_handle, 'total_frames') and hasattr(self.img_handle, 'fps') and self.img_handle.fps > 0:
                    duration = self.img_handle.total_frames/self.img_handle.fps
                    t_end = t_start + datetime.timedelta(seconds=duration)
                else:
                    # Fallback
                    t_end = t_start + datetime.timedelta(seconds=60)

            elif hasattr(self, 'current_time'):
                t_start = self.current_time
                if t_start.tzinfo is None:
                    t_start = t_start.replace(tzinfo=datetime.timezone.utc)
                t_end = t_start + datetime.timedelta(seconds=60)

        except Exception as e:
            print(f"Error determining time for satellites: {e}")

        if t_start is None:
            print("Could not determine start time for satellite tracks.")
            return

        if self.tle_file and os.path.exists(self.tle_file):
             
            tle_path_to_load = self.tle_file
             
            # If directory, find the best file
            if os.path.isdir(self.tle_file):
                tle_path_to_load = findClosestTLEFile(self.tle_file, t_start)
             
            if tle_path_to_load:
                print(f"Loading TLEs from file: {tle_path_to_load}")
                try:
                    sats = loadRobustTLEs(tle_path_to_load)
                except Exception as e:
                    print(f"Error loading TLE file: {e}")
                    return
            else:
                # If None returned or path was bad, fallback to standard download/cache
                cache_dir = os.path.join(getRmsRootDir(), ".skyfield_cache")
                sats = loadTLEs(cache_dir, max_age_hours=24)
        else:
            cache_dir = os.path.join(getRmsRootDir(), ".skyfield_cache")
            sats = loadTLEs(cache_dir, max_age_hours=24)
        
        if not sats:
            return
             
        predictor = SatellitePredictor(lat, lon, elev, t_start, t_end)
        
        jd = datetime2JD(t_start)
        
        # Generate FOV polygon by sampling edges
        w = self.platepar.X_res
        h = self.platepar.Y_res
        
        # Check if we can use cached FOV polygon
        fov_poly = []
        if self.fov_poly_cache is not None and self.fov_poly_jd == jd:
            fov_poly = self.fov_poly_cache
            # print("Using cached FOV polygon.")
        else:
            # print("Computing FOV polygon...")
            # Define edges: (x1, y1) -> (x2, y2)
            edges = [
                ((0, 0), (w, 0)),   # Top
                ((w, 0), (w, h)),   # Right
                ((w, h), (0, h)),   # Bottom
                ((0, h), (0, 0))    # Left
            ]
            
            samples_per_side = 10
            
            try:
                for (x_start, y_start), (x_end, y_end) in edges:
                    xs = np.linspace(x_start, x_end, samples_per_side, endpoint=False)
                    ys = np.linspace(y_start, y_end, samples_per_side, endpoint=False)
                    
                    # Prepare inputs
                    n = len(xs)
                    jd_arr = [jd]*n
                    level_arr = [1]*n
                    
                    _, r_arr, d_arr, _ = xyToRaDecPP(jd_arr, xs, ys, level_arr, self.platepar, jd_time=True, extinction_correction=False)
                    
                    for r, d in zip(r_arr, d_arr):
                        fov_poly.append((r, d))
                        
                # Update cache
                self.fov_poly_cache = fov_poly
                self.fov_poly_jd = jd
                
            except Exception as e:
                print(f"Error computing FOV polygon: {e}")
                return

        try:
            self.satellite_tracks = predictor.getSatelliteTracks(self.platepar, fov_poly, sats)
            print(f"Computed {len(self.satellite_tracks)} satellite tracks.")
            
            # Print details to console (matching CLI output)
            print("-"*60)
            print(f"Time Start (SkyFit2): {t_start}")
            print(f"Location: Lat={self.platepar.lat:.4f}, Lon={self.platepar.lon:.4f}, Elev={self.platepar.elev:.1f}m")
            print("-"*60)
            
            for track in self.satellite_tracks:
                name = track['name']
                x = track['x']
                y = track['y']
                ra = track['ra']
                dec = track['dec']
                
                if len(x) > 0:
                    x1, x2 = x[0], x[-1]
                    y1, y2 = y[0], y[-1]
                    r1, r2 = ra[0], ra[-1]
                    d1, d2 = dec[0], dec[-1]
                    
                    print(f"{name}")
                    print(f"      {'begin':>10}, {'end':>10}")
                    print(f"ra    = {r1:10.4f}, {r2:10.4f}")
                    print(f"dec   = {d1:10.4f}, {d2:10.4f}")
                    print(f"x     = {x1:10.2f}, {x2:10.2f}")
                    print(f"y     = {y1:10.2f}, {y2:10.2f}")
                    print("-"*60)

            self.drawSatelliteTracks()
            
        except Exception as e:
            print(f"Error computing satellite tracks: {e}")
            traceback.print_exc()

    def toggleShowSatTracks(self):
        """ Toggle whether to show satellite tracks. """

        self.show_sattracks = not self.show_sattracks
        self.tab.settings.updateShowSatTracks()
        
        if self.show_sattracks and not self.satellite_tracks:
            if SKYFIELD_AVAILABLE:
                self.loadSatelliteTracks()
            else:
                print("Cannot load satellite tracks: Skyfield not available.")
                self.show_sattracks = False
                self.tab.settings.updateShowSatTracks()
                return

        self.drawSatelliteTracks()

    def loadTLEFileDialog(self):
        """ Opens a file dialog to choose a TLE file and loads it. """
        
        # Use directory of current TLE file if available, else CWD
        init_dir = os.getcwd()
        if self.tle_file:
            init_dir = os.path.dirname(self.tle_file)

        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select TLE File",
            init_dir,
            "Text Files (*.txt);;TLE Files (*.tle);;All Files (*)"
        )

        if file_path:
            self.tle_file = file_path
            
            # Update label
            self.tab.settings.updateTLELabel(os.path.basename(self.tle_file))
            
            # Force enable tracks if they were off
            if not self.show_sattracks:
                self.show_sattracks = True
                self.tab.settings.updateShowSatTracks()
            
            # Reload tracks
            if SKYFIELD_AVAILABLE:
                self.loadSatelliteTracks()
                self.drawSatelliteTracks()
            else:
                print("Cannot load satellite tracks: Skyfield not available.")


    def clearTLESelection(self):
        """ Clears the custom TLE file and reloads from default/downloaded. """
        self.tle_file = None
        self.tab.settings.updateTLELabel("latest downloaded")
        
        # Ensure tracks are enabled
        if not self.show_sattracks:
            self.show_sattracks = True
            self.tab.settings.updateShowSatTracks()
             
        if SKYFIELD_AVAILABLE:
            self.loadSatelliteTracks()
            self.drawSatelliteTracks()
        else:
            print("Cannot load satellite tracks: Skyfield not available.")

    def redrawSatelliteTracks(self):
        """ Manually re-computes satellite tracks (prediction + projection). """
        if SKYFIELD_AVAILABLE:
            print("Redrawing satellite tracks...")
            self.loadSatelliteTracks()
            self.drawSatelliteTracks()
        else:
            print("Cannot load satellite tracks: Skyfield not available.")

    def drawSatelliteTracks(self):
        """ Draws satellite tracks on the image. """
        
        # Clear existing
        for curve in self.sat_track_curves:
            self.img_frame.removeItem(curve)
        for label in self.sat_track_labels:
            self.img_frame.removeItem(label)
        for arrow in self.sat_track_arrows:
            self.img_frame.removeItem(arrow)
        for marker in self.sat_markers:
            self.img_frame.removeItem(marker)
        self.sat_track_curves = []
        self.sat_track_labels = []
        self.sat_track_arrows = []
        self.sat_markers = []

        if not self.show_sattracks:
            return

        w = self.platepar.X_res
        h = self.platepar.Y_res

        # Margin to prevent label placement too close to edge
        margin = 100 # pixels

        # Define a list of high-contrast colors suitable for both dark and light backgrounds
        # (R, G, B)
        colors = [
            (255, 0, 0),      # Red
            (0, 255, 0),      # Green
            (60, 100, 255),   # Lighter Blue
            (255, 255, 0),    # Yellow
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Cyan
            (255, 128, 0),    # Orange
            (128, 0, 255),    # Purple
        ]

        for i, track in enumerate(self.satellite_tracks):
            # Cycle through colors
            color = colors[i % len(colors)]
            
            # Draw curve
            # Thicker line (width=8), alpha=0.125 (32)
            # Use color with alpha
            pen_color = color + (32,)
            pen = pg.mkPen(pen_color, width=8)
            
            curve = pg.PlotCurveItem(track['x'], track['y'], pen=pen, clickable=False)
            self.img_frame.addItem(curve)
            self.sat_track_curves.append(curve)

            # Create marker for real-time satellite position
            marker_pen = pg.mkPen(color + (90,), width=5)
            marker = pg.ScatterPlotItem(size=30, pen=marker_pen, brush=None, symbol='o')

            # Initialize with empty points to hide it
            marker.setData([], [])
            self.img_frame.addItem(marker)
            self.sat_markers.append(marker)
            
            # Draw arrows at the beginning, middle, and end
            if len(track['x']) >= 2:
                
                # Indices for arrows
                arrow_indices = [0, len(track['x'])//2, -1]
                
                # Filter indices to ensure they are distinct?
                # For very short tracks start/mid/end might overlap, but that's okay.
                for idx in arrow_indices:
                    
                    # Normalize index
                    if idx < 0:
                        idx = len(track['x']) + idx

                    # Calculate position
                    x_pos = track['x'][idx]
                    y_pos = track['y'][idx]
                    
                    # Calculate direction
                    # If at end, look back to previous point
                    if idx == len(track['x']) - 1:
                        dx = track['x'][idx] - track['x'][idx-1]
                        dy = track['y'][idx] - track['y'][idx-1]
                    else:
                        dx = track['x'][idx+1] - track['x'][idx]
                        dy = track['y'][idx+1] - track['y'][idx]
                        
                    # Calculate angle
                    angle_deg = np.degrees(np.arctan2(dy, dx)) + 180
                    
                    arrow = pg.ArrowItem(pos=(x_pos, y_pos), angle=angle_deg, headLen=40, tipAngle=40, tailLen=0, 
                                         brush=pg.mkBrush(pen_color), pen=pg.mkPen(pen_color))
                    
                    self.img_frame.addItem(arrow)
                    self.sat_track_arrows.append(arrow)

            
            # Smart label placement
            if len(track['x']) > 0:
                x = track['x']
                y = track['y']
                
                # Default to middle point if no better point found
                best_idx = len(x)//2
                
                # Try to find a point well inside the image 
                # (margin from edges)
                for test_pt in range(len(x)):
                    xi, yi = x[test_pt], y[test_pt]
                    if margin < xi < (w - margin) and margin < yi < (h - margin):
                        best_idx = test_pt
                        break
                
                label_x = x[best_idx]
                label_y = y[best_idx]
                
                text = pg.TextItem(track['name'], color=color, anchor=(0, 1))
                
                # Increase text size by 50% and make it bold
                font = QFont()
                font.setBold(True)
                
                # Set the font size
                base_size = 8
                font.setPointSizeF(base_size)
                
                text.setFont(font)
                
                text.setPos(label_x, label_y)
                # Store original position for restoring later
                text.orig_pos = (label_x, label_y)
                
                self.img_frame.addItem(text)
                self.sat_track_labels.append(text)
                
        # Update marker positions if in manual reduction mode
        self.updateSatelliteMarker()


    def updateSatelliteMarker(self):
        """ Updates the satellite marker position based on the current frame time. """
        
        # Only show markers in manual reduction mode
        if self.mode != 'manualreduction':
            for marker in self.sat_markers:
                marker.setData([], [])
            return

        # Get current frame JD
        current_jd = date2JD(*self.img_handle.currentFrameTime())
        
        print("-" * 30)
        print(f"Frame = {self.img.img_handle.current_frame}")

        for i, track in enumerate(self.satellite_tracks):
            if i >= len(self.sat_markers):
                break
                
            marker = self.sat_markers[i]
            
            # Interpolate position
            
            # Try to read the time from the track
            times = track.get('time')
            if times is None or len(times) < 2:
                marker.setData([], [])
                continue
                
            # Interpolate the position of the satellite based on the time and position
            # Make sure that the satellite is visible on the track segment we calculated
            t_min = np.min(times)
            t_max = np.max(times)
            if t_min <= current_jd <= t_max:
                
                x = np.interp(current_jd, times, track['x'])
                y = np.interp(current_jd, times, track['y'])
                
                marker.setData([x], [y])
                print(f"{track['name']:<25}: X = {x:8.2f}, Y = {y:8.2f}")
            else:
                marker.setData([], [])

            # Update label position so it follows the marker
            if i < len(self.sat_track_labels):
                label = self.sat_track_labels[i]
                
                # Check if marker is visible and inside the image
                marker_visible = False
                if t_min <= current_jd <= t_max:
                    
                    # Check if inside the image (with some margin?)
                    # x, y are already computed above
                    w = self.platepar.X_res
                    h = self.platepar.Y_res
                    
                    if 0 <= x <= w and 0 <= y <= h:
                        marker_visible = True
                        
                if marker_visible:
                    # Move label to marker
                    label.setPos(x, y)
                else:
                    # Restore original position
                    if hasattr(label, 'orig_pos'):
                        label.setPos(*label.orig_pos)



    def saveECSV(self):
        """ Save the picks into the GDEF ECSV standard. """

        # If no picks, save nothing and send no-picks to FTPDetectionInfo save
        if [key for key, val in self.pick_list.items() if (val['x_centroid'] is not None)] == []:
            return False

        isodate_format_file = "%Y-%m-%dT%H_%M_%S"
        isodate_format_entry = "%Y-%m-%dT%H:%M:%S.%f"

        # Reference time
        dt_ref = self.img_handle.beginning_datetime

        # ESCV files name
        ecsv_file_name = dt_ref.strftime(isodate_format_file) + '_RMS_' + self.config.stationID + ".ecsv"


        # Compute alt/az pointing
        azim, elev = trueRaDec2ApparentAltAz(self.platepar.RA_d, self.platepar.dec_d, self.platepar.JD, \
            self.platepar.lat, self.platepar.lon, refraction=False)

        # Compute FOV size
        fov_horiz, fov_vert = computeFOVSize(self.platepar)

        if self.img_handle.input_type == 'ff':
            ff_name = self.img_handle.current_ff_file
        else:
            ff_name = "FF_{:s}_".format(self.platepar.station_code) \
                          + self.img_handle.beginning_datetime.strftime("%Y%m%d_%H%M%S_") \
                          + "{:03d}".format(int(self.img_handle.beginning_datetime.microsecond//1000)) \
                          + "_0000000.fits"

        # Get the number of stars in the list
        if self.platepar.star_list is not None:
            n_stars = len(self.platepar.star_list)
        else:
            n_stars = 0

        # Write the meta header
        meta_dict = {
            'obs_latitude': self.platepar.lat,                        # Decimal signed latitude (-90 S to +90 N)
            'obs_longitude': self.platepar.lon,                       # Decimal signed longitude (-180 W to +180 E)
            'obs_elevation': self.platepar.elev,                      # Altitude in metres above MSL. Note not WGS84
            'origin': 'SkyFit2',                                      # The software which produced the data file
            'camera_id': self.config.stationID,                       # The code name of the camera, likely to be network-specific
            'cx' : self.platepar.X_res,                               # Horizontal camera resolution in pixels
            'cy' : self.platepar.Y_res,                               # Vertical camera resolution in pixels
            'photometric_band' : self.mag_band_string,                # The photometric band of the star catalogue
            'image_file' : ff_name,                                   # The name of the original image or video
            'isodate_start_obs': str(dt_ref.strftime(isodate_format_entry)), # The date and time of the start of the video or exposure
            'astrometry_number_stars' : n_stars,                      # The number of stars identified and used in the astrometric calibration
            'mag_label': 'mag_data',                                  # The label of the Magnitude column in the Point Observation data
            'no_frags': 1,                                            # The number of meteoroid fragments described in this data
            'obs_az': azim,                                           # The azimuth of the centre of the field of view in decimal degrees. North = 0, increasing to the East
            'obs_ev': elev,                                           # The elevation of the centre of the field of view in decimal degrees. Horizon =0, Zenith = 90
            'obs_rot': rotationWrtHorizon(self.platepar),             # Rotation of the field of view from horizontal, decimal degrees. Clockwise is positive
            'fov_horiz': fov_horiz,                                   # Horizontal extent of the field of view, decimal degrees
            'fov_vert': fov_vert,                                     # Vertical extent of the field of view, decimal degrees
           }


        # Write the header
        out_str = """# %ECSV 0.9
# ---
# datatype:
# - {name: datetime, datatype: string}
# - {name: ra, unit: deg, datatype: float64}
# - {name: dec, unit: deg, datatype: float64}
# - {name: azimuth, datatype: float64}
# - {name: altitude, datatype: float64}
# - {name: x_image, unit: pix, datatype: float64}
# - {name: y_image, unit: pix, datatype: float64}
# - {name: integrated_pixel_value, datatype: int64}
# - {name: background_pixel_value, datatype: int64}
# - {name: saturated_pixels, datatype: bool}
# - {name: mag_data, datatype: float64}
# - {name: err_minus_mag, datatype: float64}
# - {name: err_plus_mag, datatype: float64}
# - {name: snr, datatype: float64}
# delimiter: ','
# meta: !!omap
"""
        # Add the meta information
        for key in meta_dict:

            value = meta_dict[key]

            if isinstance(value, str):
                value_str = "'{:s}'".format(value)
            else:
                value_str = str(value)

            out_str += "# - {" + "{:s}: {:s}".format(key, value_str) + "}\n"


        out_str += "# schema: astropy-2.0\n"
        out_str += "datetime,ra,dec,azimuth,altitude,x_image,y_image,integrated_pixel_value,background_pixel_value,saturated_pixels,mag_data,err_minus_mag,err_plus_mag,snr\n"

        # Add the data (sort by frame)
        for frame, pick in sorted(self.pick_list.items(), key=lambda x: x[0]):

            # Make sure to centroid is picked and is not just the photometry
            if pick['x_centroid'] is None:
                continue

            # Only store real picks, and not gaps
            if pick['mode'] == 0:
                continue
            
            # Read the SNR and make sure it is not None
            snr = pick['snr']
            if snr is None:

                # If SNR is None, then set it to 0
                snr = 0.0

                # If SNR is None, then set the random error to 0
                mag_err_random = 0

            else:

                # Compute the random error based on the SNR
                mag_err_random = 2.5*np.log10(1 + 1/pick['snr'])

            # Compute the magnitude errors
            mag_err_total = np.sqrt(mag_err_random**2 + self.platepar.mag_lev_stddev**2)

            # Use a modified platepar if ground points are being picked
            pp_tmp = copy.deepcopy(self.platepar)
            if self.meas_ground_points:
                pp_tmp.switchToGroundPicks()

            time_data = [self.img_handle.currentFrameTime(frame_no=frame)]

            # Compute measured RA/Dec from image coordinates
            jd_data, ra_data, dec_data, mag_data = xyToRaDecPP(time_data, [pick['x_centroid']],
                [pick['y_centroid']], [pick['intensity_sum']], pp_tmp, measurement=True)

            jd = jd_data[0]
            ra = ra_data[0]
            dec = dec_data[0]
            mag = mag_data[0]

            # Apply exposure ratio correction
            mag += self.computeExposureRatioCorrection()

            # Compute alt/az (topocentric, i.e. without refraction)
            azim, alt = trueRaDec2ApparentAltAz(ra, dec, jd, pp_tmp.lat, pp_tmp.lon, refraction=False)

            # Normalize the frame number to the actual time
            frame_dt = self.img_handle.currentFrameTime(frame_no=frame, dt_obj=True)
            frame_no = (frame_dt - dt_ref).total_seconds()*self.img_handle.fps

            # Get the rolling shutter corrected (or not, depending on the config) frame number
            if self.config.deinterlace_order == -1:
                frame_no = self.getRollingShutterCorrectedFrameNo(frame_no, pick)

            # If the global shutter is used, the frame number can only be an integer
            if self.config.deinterlace_order == -2:
                frame_no = round(frame_no, 0)

            # Compute the time relative to the reference JD
            t_rel = frame_no/self.img_handle.fps

            # Determine whether to save the raw times that came with in the data
            save_raw_times = False
            if (self.img_handle.input_type == "vid") or (self.img_handle.input_type == "dfn"):
                save_raw_times = True

            if self.img_handle.input_type == "images":
                if self.img_handle.fripon_mode or self.img_handle.uwo_png_mode:
                    save_raw_times = True

            # For UWO .vid files DFN data, don't normalize the time to the FPS, as the time is GPS-synced
            if save_raw_times:
                frame_time = frame_dt
            
            else:
                
                # Compute the datetime of the point
                frame_time = dt_ref + datetime.timedelta(seconds=t_rel)

            # Add an entry to the ECSV file
            entry = [
                frame_time.strftime(isodate_format_entry),
                "{:10.6f}".format(ra), "{:+10.6f}".format(dec),
                "{:10.6f}".format(azim), "{:+10.6f}".format(alt),
                "{:9.3f}".format(pick['x_centroid']), "{:9.3f}".format(pick['y_centroid']), 
                "{:10d}".format(int(pick['intensity_sum'])),
                "{:10d}".format(int(pick['background_intensity'])),
                "{:5s}".format(str(pick['saturated'])),
                "{:+7.2f}".format(mag), "{:+6.2f}".format(-mag_err_total), "{:+6.2f}".format(mag_err_total),
                "{:10.2f}".format(snr)
                ]

            out_str += ",".join(entry) + "\n"



        ecsv_file_path = os.path.join(self.dir_path, ecsv_file_name)

        # Write file to disk
        with open(ecsv_file_path, 'w') as f:
            f.write(out_str)


        print("ESCV file saved to:", ecsv_file_path)

        return True




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
                img_h = self.img.data.shape[1]

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

    def furthestStar(self, miss_this_one=False, min_separation=15):
        """
        Find the star which is furthest away from all other stars that have already been matched.

        Keyword arguments:
            miss_this_one: Return coordinates of a different star at random, but don't mark anything.
            min_separation: Minimum separation in pixels between stars.

        Returns: 
            (x,y) integers of the image location of the furthest star away from all other matched stars.

        """

        # Strategy

        # Working in image coordinates

        # Get two lists marked_x, marked_y of each marked star - getMarkedStars()
        # Get three lists candidate_x, candidate_y of each unmarked star, and the distance to nearest
        # marked star which is more than minimum separation away

        # Get star with the greatest distance to the nearest marked star



        # Return the image coordinates of the star which is furthest away from any marked star
        # Create marked_x, marked_y in image coordinates which is composed of matched stars and unsuitable stars

        # Get all the matched stars in image coordinates



        def getMarkedStars(include_unsuitable=True):

            """

            Returns: a list of stars which are either marked as paired, or bad in image coordinates

            """

            marked_x, marked_y = [], []
            coords_list = self.paired_stars.imageCoords()
            for coords in coords_list:
                marked_x.append(coords[0])
                marked_y.append(coords[1])

            if include_unsuitable:
                coords_list = self.unsuitable_stars.imageCoords()
                for coords in coords_list:
                    marked_x.append(coords[0])
                    marked_y.append(coords[1])

            return marked_x, marked_y
        ##############################################################################################################

        def isDouble(x,y, reference_x_list, reference_y_list, min_separation=5):

            """
            Are x,y coordinates which are very close to, but distinct from all coordinates in reference list

            Args:
                x: image coordinates of star
                y: image coordinates of star
                reference_x_list: list of x image coordinates
                reference_y_list: list of y image coordinates

            Returns:
                [bool] True if star is within min_separation of another star
            """

            for reference_x, reference_y in zip(reference_x_list, reference_y_list):
                # Check if this the reference is the same star
                if reference_x == x and reference_y == y:
                    continue
                if ((reference_x - x) ** 2 + (reference_y - y) ** 2) ** 0.5 < min_separation:
                    return True

            return False
        ##############################################################################################################

        def getVisibleUnmarkedStarsAndDistanceToMarked(marked_x_list, marked_y_list, min_separation=15):

            """
            From the catalogue of filtered stars return a lists of coordinates stars which are not marked,
            and another list which is the distance to the nearest marked star

            Args:
                marked_x_list: list of marked star x coordinates
                marked_y_list: list of marked star y coordinates
                min_separation: minimum separation to be regarded as a different stra

            Returns:
                unmarked_x_list: list of unmarked star x coordinates
                unmarked_y_list: list of unmarked star x coordinates
                dist_nearest_marked_list: distance of the nearest marked star for returned star coordinates


            """

            # Is there a way to get this in image coordinates directly
            visible_ra_list = [star[0] for star in self.catalog_stars_filtered]
            visible_dec_list = [star[1] for star in self.catalog_stars_filtered]

            # Convert all visible star to image coordinates

            visible_x, visible_y = raDecToXYPP(np.array(visible_ra_list), np.array(visible_dec_list),
                                               datetime2JD(self.img_handle.currentFrameTime(dt_obj=True)),
                                               self.platepar)

            # Handle jump when no stars are marked - just pick and return a single random star
            if len(marked_x_list) == 0 or len(marked_y_list) == 0:
                random_star = random.randint(0, len(visible_x) - 1)
                return [visible_x[random_star]], [visible_y[random_star]], [np.inf], [np.inf]

            # Iterate through all visible stars creating a list of stars which are more than
            # min separation from a marked star, and then add coordinates of the visible star
            # and minimum distance to the nearest marked star, which is more than min_separation away
            # If a visible star is too close to an already marked star then ignore this star
            # and do not append to the candidate star list

            candidate_x_list, candidate_y_list, dist_nearest_marked_list = [], [], []


            # Reject stars which are too close to the edge
            edge_margin = 5 # px

            for x, y in zip(visible_x, visible_y):
                ignore_this_star = False

                if isDouble(x,y, visible_x, visible_y):
                    continue

                nearest_pixel_separation = np.inf

                for marked_x, marked_y in zip(marked_x_list, marked_y_list):
                    
                    # calculate cartesian separation
                    pixel_separation = ((marked_x - x) ** 2 + (marked_y - y) ** 2) ** 0.5
                    
                    # If this star is less than minimum separation away
                    if pixel_separation < min_separation or ignore_this_star:
                        # do not use this visible star in any further iteration
                        ignore_this_star = True
                        break
                    
                    # If this star is too close to the edge, do not use it
                    if (x < edge_margin) or (x > self.platepar.X_res - edge_margin) or \
                        (y < edge_margin) or (y > self.platepar.Y_res - edge_margin):

                        ignore_this_star = True
                        break


                    else:
                        if pixel_separation < nearest_pixel_separation:

                            # Update the x, y coordinates and the nearest star by pixel separation
                            nearest_x, nearest_y, nearest_pixel_separation = x, y, pixel_separation


                # Append once for each visible star that is not marked to be ignored
                if not ignore_this_star:
                    candidate_x_list.append(nearest_x)
                    candidate_y_list.append(nearest_y)
                    dist_nearest_marked_list.append(nearest_pixel_separation)

            return candidate_x_list, candidate_y_list, dist_nearest_marked_list, nearest_pixel_separation
        
        ######################################################################################################

        marked_x_list, marked_y_list = getMarkedStars(include_unsuitable=False)
        max_distance_between_paired = maxDistBetweenPoints(marked_x_list, marked_y_list)

        marked_x_list, marked_y_list = getMarkedStars(include_unsuitable=True)
        unmarked_x_list, unmarked_y_list, dist_nearest_marked_list, distance_between_unmarked = \
            getVisibleUnmarkedStarsAndDistanceToMarked(marked_x_list, marked_y_list, 
                                                       min_separation=min_separation)

        if len(dist_nearest_marked_list) == 0:
            print("No stars left to pick")
            return marked_x_list[-1], marked_y_list[-1], max_distance_between_paired

        if miss_this_one:
            # Pick a distance at random
            next_star_index = dist_nearest_marked_list.index(random.choice(dist_nearest_marked_list))
        else:
            # Find the index of this star
            next_star_index = dist_nearest_marked_list.index(max(dist_nearest_marked_list))


        # Return coordinates of next star and maximum pixel distance between marked stars

        return unmarked_x_list[next_star_index], unmarked_y_list[next_star_index], max_distance_between_paired



if __name__ == '__main__':
    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Tool for fitting astrometry plates and photometric calibration.")

    arg_parser.add_argument('input_path', metavar='INPUT_PATH', type=str, nargs='?', default=None,
                            help='Path to the folder with FF or image files, path to a video file, or to a state file.'
                                 ' If images or videos are given, their names must be in the format: YYYYMMDD_hhmmss.uuuuuu'
                                 ' If not provided, a dialog will prompt for selection.')

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

    arg_parser.add_argument('-p', '--geopoints', metavar='GEO_POINTS_PATH', type=str,
                            help="Path to a file with a list of geo coordinates which will be projected on "
                                 "the image as seen from the perspective of the observer.")
    
    arg_parser.add_argument('-n', '--nobg', action="store_true", \
                            help="Do not subtract the background when doing photometry. This is useful when "
                            "calibrating saturated objects, as the background can vary between images and the " 
                            "idea is that the intensity is used as a measure of the radius of the saturated "
                            "object.")
    
    arg_parser.add_argument('--peribg', action="store_true", \
                            help="Compute the background using the periphery around coloured pixels instead "
                            "of the avepixel image. This is useful when the avepixel is contaminated by the "
                            "measured object.")
    
    arg_parser.add_argument('--flipud', action="store_true", \
                            help="Flip the image upside down. Only applied to images and videos.")

    arg_parser.add_argument('--sattracks', action="store_true", \
                            help="Show satellite tracks overlaid on the image (requires internet to download TLEs).")

    arg_parser.add_argument('--tle_file', type=str, default=None,
                            help="Path to a specific TLE file to use for satellite tracks (skips download). "
                            "Alternatively, a directory containing TLE files can be specified. The code will"
                            " automatically select the TLE file closest to the beginning time of the video.")




    arg_parser.add_argument('-m', '--mask', metavar='MASK_PATH', type=str,
                            help="Path to a mask file which will be applied to the star catalog")
    
    arg_parser.add_argument('--flatbiassub', action="store_true", \
        help="Subtract the bias from the flat. False by default.")

    arg_parser.add_argument('--expratio', metavar='EXPOSURE_RATIO', type=float, default=1.0,
                            help="Exposure ratio between stars and meteor segments. Used for static images " 
                            "where the stars are continuously exposed but the meteors/fireballs are chopped "
                            "up by a shutter. For example, a 30 s exposure with meteor segments at 20 FPS "
                            "results in a exposure ratio of 600.")



    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################


    # Parse the beginning time into a datetime object
    if cml_args.timebeg is not None:

        time_formats_to_try = ["%Y%m%d_%H%M%S.%f", "%Y%m%d_%H%M%S", "%Y%m%d-%H%M%S.%f", "%Y%m%d-%H%M%S"]

        beginning_time = None
        for time_format in time_formats_to_try:
            try:
                beginning_time = datetime.datetime.strptime(cml_args.timebeg[0], time_format)
                break
            except ValueError:
                pass

        if beginning_time is None:
            raise ValueError("The beginning time format is not recognized! Please use one of the following formats: "
                                + ", ".join(time_formats_to_try))

    else:
        beginning_time = None

    app = QtWidgets.QApplication(sys.argv)

    # If no input path was provided, prompt for one
    if cml_args.input_path is None:
        # Ask user what type of input to select
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle("SkyFit2 - Select Input")
        msg.setText("What would you like to open?")
        folder_btn = msg.addButton("Folder (FF/images)", QtWidgets.QMessageBox.ActionRole)
        file_btn = msg.addButton("File (video/state)", QtWidgets.QMessageBox.ActionRole)
        cancel_btn = msg.addButton(QtWidgets.QMessageBox.Cancel)
        msg.exec_()

        input_path = None
        if msg.clickedButton() == folder_btn:
            input_path = QtWidgets.QFileDialog.getExistingDirectory(
                None, "Select folder with FF/image files",
                os.path.expanduser("~"),
                QtWidgets.QFileDialog.ShowDirsOnly
            )
        elif msg.clickedButton() == file_btn:
            input_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                None, "Select video or state file",
                os.path.expanduser("~"),
                "All supported (*.state *.mp4 *.avi *.mkv *.mov);;State files (*.state);;Video files (*.mp4 *.avi *.mkv *.mov);;All files (*)"
            )

        if not input_path:
            print("No input path selected. Exiting.")
            sys.exit(0)

        cml_args.input_path = input_path

    # If no config file was provided, prompt for one
    if cml_args.config is None:
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle("SkyFit2 - Select Config")
        msg.setText("No .config file specified.\n\nSelect the config file source:")
        data_btn = msg.addButton("Data Folder", QtWidgets.QMessageBox.ActionRole)
        rms_btn = msg.addButton("RMS Root", QtWidgets.QMessageBox.ActionRole)
        browse_btn = msg.addButton("Browse...", QtWidgets.QMessageBox.ActionRole)
        msg.exec_()

        if msg.clickedButton() == data_btn:
            # Use config from data folder (use '.' to trigger directory search)
            cml_args.config = ['.']
        elif msg.clickedButton() == browse_btn:
            config_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                None, "Select config file",
                os.path.expanduser("~"),
                "Config files (*.config);;All files (*)"
            )
            if config_path:
                cml_args.config = [config_path]
            else:
                print("No config file selected. Using RMS root default.")

    # If the state file was given, load the state
    if cml_args.input_path.endswith('.state'):

        dir_path, state_name = os.path.split(cml_args.input_path)
        config = cr.loadConfigFromDirectory(cml_args.config, cml_args.input_path)

        # Create plate_tool without calling its constructor then calling loadstate
        plate_tool = PlateTool.__new__(PlateTool)
        super(PlateTool, plate_tool).__init__()

        if cml_args.mask is not None:
            print("Given a path to a mask at {}".format(cml_args.mask))
            mask = getMaskFile(os.path.expanduser(cml_args.mask), config)

        elif os.path.exists(os.path.join(config.rms_root_dir, config.mask_file)):
            print("No mask specified loading mask from {}".format(os.path.join(config.rms_root_dir, config.mask_file)))
            mask = getMaskFile(config.rms_root_dir, config)

        elif os.path.exists("mask.bmp"):
            mask = getMaskFile(".", config)

        elif True:
            mask = None

        # If the dimensions of the mask do not match the config file, ignore the mask
        if (mask is not None) and (not mask.checkResolution(config.width, config.height)):
            print("Mask resolution ({:d}, {:d}) does not match the image resolution ({:d}, {:d}). Ignoring the mask.".format(
                mask.width, mask.height, config.width, config.height))
            mask = None

        plate_tool.loadState(dir_path, state_name, beginning_time=beginning_time, mask=mask)

        # Initialize satellite track related attributes for loaded state
        plate_tool.show_sattracks = cml_args.sattracks
        plate_tool.tle_file = cml_args.tle_file
        plate_tool.satellite_tracks = []
        plate_tool.sat_track_curves = []
        plate_tool.sat_track_labels = []
        if plate_tool.show_sattracks:
            plate_tool.loadSatelliteTracks()

    else:

        # Extract the data directory path
        input_path = cml_args.input_path.replace('"', '')
        if os.path.isfile(input_path):
            dir_path = os.path.dirname(input_path)
        else:
            dir_path = input_path

        # Load the config file
        config = cr.loadConfigFromDirectory(cml_args.config, dir_path)


        if cml_args.mask is not None:
            print("Given a path to a mask at {}".format(cml_args.mask))
            mask = getMaskFile(os.path.expanduser(cml_args.mask), config)

        elif os.path.exists(os.path.join(config.rms_root_dir, config.mask_file)):

            print("No mask specified loading mask from {}".format(os.path.join(config.rms_root_dir, config.mask_file)))
            mask = getMaskFile(config.rms_root_dir, config)

        elif os.path.exists("mask.bmp"):
            mask = getMaskFile(".", config)

        else:
            mask = None

        # If the dimensions of the mask do not match the config file, ignore the mask
        if (mask is not None) and (not mask.checkResolution(config.width, config.height)):
            print("Mask resolution ({:d}, {:d}) does not match the image resolution ({:d}, {:d}). Ignoring the mask.".format(
                mask.width, mask.height, config.width, config.height))
            mask = None

        # Init SkyFit
        plate_tool = PlateTool(input_path, config, beginning_time=beginning_time, fps=cml_args.fps, \
            gamma=cml_args.gamma, use_fr_files=cml_args.fr, geo_points_input=cml_args.geopoints, \
            mask=mask, nobg=cml_args.nobg, peribg=cml_args.peribg, flipud=cml_args.flipud, \
            flatbiassub=cml_args.flatbiassub, exposure_ratio=cml_args.expratio, show_sattracks=cml_args.sattracks, \
            tle_file=cml_args.tle_file)



    # Run the GUI app
    sys.exit(app.exec_())
