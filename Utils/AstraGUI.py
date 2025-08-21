from PyQt5.QtWidgets import (
    QDialog, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
    QLineEdit, QGroupBox, QFormLayout, QComboBox, QFileDialog,
    QProgressBar, QTextEdit, QApplication, QWidget, QGridLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from Utils.Astra import ASTRA  
import html, re
from PyQt5 import QtCore


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

    def __init__(self, run_load_callback=None, run_astra_callback=None, run_kalman_callback=None, skyfit_instance=None, parent=None):

        super().__init__(parent)
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

        def addGridFields(field_dict, defaults, title, tooltips=None):
            group = QGroupBox(title)
            layout = QGridLayout()
            tts = tooltips or {}

            def _is_bool_like(val: str) -> bool:
                return isinstance(val, str) and val.strip().lower() in ("true", "false")

            for idx, (key, default) in enumerate(defaults.items()):
                key_html = toHTMLMath(key)
                label = QLabel(key_html.replace('</span></body></html>', ':</span></body></html>'))

                # tooltip
                tt_raw = tts.get(key, "")
                tt_html = toHTMLMath(tt_raw) if tt_raw else ""
                if tt_html:
                    label.setToolTip(tt_html)

                # >>> create the right editor
                if _is_bool_like(default):
                    field = QComboBox()
                    field.addItems(["True", "False"])
                    # set initial selection based on the default
                    field.setCurrentText("True" if str(default).strip().lower() == "true" else "False")
                else:
                    field = QLineEdit(default)

                if tt_html:
                    field.setToolTip(tt_html)

                layout.addWidget(label, idx // 2, (idx % 2) * 2)
                layout.addWidget(field, idx // 2, (idx % 2) * 2 + 1)
                field_dict[key] = field

            group.setLayout(layout)
            return group

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
            "L_max": "1.5", "Verbose": "False", "photom_thresh" : "0.05", 
            "Save Animation": "False", "CE_coeff" : "3"
        }

        # === Kalman Filter Settings ===
        self.kalman_fields = {}
        kalman_defaults = {
            "Monotonicity": "True", "sigma_xy (px)": "0.5", "sigma_vxy (%)": "100", "save results" : "False"
        }

        # === PARAMETER GUIDE ===
        PSO_TT = {
            "w (0-1)": "PSO particle inertia. Higher = more exploration.",
            "c_1 (0-1)": "Cognitive weight (pull to particle’s best).",
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
            "CE_coeff" : "Multiples of fitted length (STD) to place leading edge from center pick"
        }

        KALMAN_TT = {
            "Monotonicity": "Enforce monotonic motion along dominant axis (True/False).",
            "sigma_xy (px)": "STD of position estimate errors (px).",
            "sigma_vxy (%)": "STD of velocity estimate errors (in percent).",
            "save results" : "Save the uncertainties from the kalman filter into a .csv file"
        }

        main_layout.addWidget(
            addGridFields(self.pso_fields, pso_defaults, "PSO PARAMETER SETTINGS", PSO_TT)
        )
        main_layout.addWidget(
            addGridFields(self.astra_fields, astra_defaults, "ASTRA PARAMETER SETTINGS", ASTRA_TT)
        )
        main_layout.addWidget(
            addGridFields(self.kalman_fields, kalman_defaults, "KALMAN FILTER SETTINGS", KALMAN_TT)
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
        self.config = config

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
        else:
            status_text = "READY" if ready else "NOT READY"
            color = "#4CAF50" if ready else "#F44336"  # Green or Red
            enable_btn = bool(ready)
        self.astra_status_dot.setText(status_text)
        self.astra_status_dot.setAlignment(Qt.AlignCenter)
        self.astra_status_dot.setStyleSheet(
            f"background-color: {color}; color: white; border-radius: 6px; min-width: 80px; min-height: 20px; max-height: 40px; font-weight: bold;"
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
        else:
            status_text = "READY" if ready else "NOT READY"
            color = "#4CAF50" if ready else "#F44336"  # Green or Red
            enable_btn = bool(ready)
        self.kalman_status_dot.setText(status_text)
        self.kalman_status_dot.setAlignment(Qt.AlignCenter)
        self.kalman_status_dot.setStyleSheet(
            f"background-color: {color}; color: white; border-radius: 6px; min-width: 80px; min-height: 20px; max-height: 40px; font-weight: bold;"
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

        self.config = {
            "file_path": self.selected_file_label.text(),
            "pso":   {k: _value_of(v) for k, v in self.pso_fields.items()},
            "astra": {k: _value_of(v) for k, v in self.astra_fields.items()},
            "kalman":{k: _value_of(v) for k, v in self.kalman_fields.items()},
        }

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
        self.worker.finished.connect(lambda: self.file_picker_button.setEnabled(True))

        # Restore interactivity if ASTRA/Kalman can run
        self.worker.finished.connect(lambda: self.skyfit_instance.checkASTRACanRun())
        self.worker.finished.connect(lambda: self.skyfit_instance.checkKalmanCanRun())
        self.worker.finished.connect(lambda: self.skyfit_instance.checkPickRevertCanRun())


        self.thread.start()

    def setPreviousPicks(self):
        """Hits parent instance to revert picks to previous state"""
        self.skyfit_instance.reverseASTRAPicks()

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
            "Save Animation": (True, False, bool), "CE_coeff": (0, None, float)
        }

        kalman_ranges_and_types = {
            "Monotonicity": (True, False, bool), "sigma_xy (px)": (0, None, float), "sigma_vxy (%)": (0, 100, float), "save results": (True, False, bool)
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
                if param_type == bool:
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

        self.kalman_worker = KalmanWorker(self.skyfit_instance, self.config)
        self.kalman_worker.progress.connect(self.updateProgress)

        # Restore interactivity
        self.kalman_worker.finished.connect(lambda: self.file_picker_button.setEnabled(True))

        # Restore interactivity if ASTRA/Kalman can run
        self.kalman_worker.finished.connect(lambda: self.skyfit_instance.checkASTRACanRun())
        self.kalman_worker.finished.connect(lambda: self.skyfit_instance.checkKalmanCanRun())
        self.kalman_worker.finished.connect(lambda: self.skyfit_instance.checkPickRevertCanRun())
        self.kalman_worker.finished.connect(lambda: self.updateProgress(100))
        self.kalman_worker.start()

class KalmanWorker(QThread):
    """Worker thread for running Kalman filter."""
    progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, skyfit_instance, config):
        super().__init__()
        self.skyfit_instance = skyfit_instance
        self.config = config

    def run(self):
        self.skyfit_instance.runKalmanFromConfig(self.config, progress_callback=self.progress.emit)
        self.finished.emit()

class AstraWorker(QObject):
    """Worker thread for running ASTRA."""
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    results_ready = pyqtSignal(object)

    def __init__(self, config, skyfit_instance):
        super().__init__()
        self.config = config
        self.skyfit_instance = skyfit_instance

    def run(self):
        # Prepare data
        data_dict = self.skyfit_instance.prepareASTRAData(self.config)

        if data_dict is False:
            self.finished.emit()
            return

        # Run ASTRA here, directly in worker
        astra = ASTRA(**data_dict, progress_callback=self.progress.emit)
        astra.process()

        # Handle results via callback
        self.results_ready.emit(astra)
        self.finished.emit()

def launchASTRAGUI(run_astra_callback=None,
                     run_kalman_callback=None,
                     run_load_callback=None,
                     parent=None,
                     skyfit_instance=None):
    dialog = AstraConfigDialog(
        run_astra_callback=run_astra_callback,
        run_kalman_callback=run_kalman_callback,
        run_load_callback=run_load_callback,
        parent=parent,
        skyfit_instance=skyfit_instance
    )
    dialog.show()
    return dialog

if __name__ == "__main__":
    config = launchASTRAGUI()
    if config:
        print("Returned config from GUI:")
        for section, values in config.items():
            print(f"[{section}]")
            if isinstance(values, dict):
                for k, v in values.items():
                    print(f"  {k}: {v}")
            else:
                print(f"  {values}")
