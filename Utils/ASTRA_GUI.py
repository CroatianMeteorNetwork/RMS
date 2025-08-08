from PyQt5.QtWidgets import (
    QDialog, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
    QLineEdit, QGroupBox, QFormLayout, QComboBox, QFileDialog,
    QProgressBar, QTextEdit, QApplication, QWidget, QGridLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from .ASTRA import ASTRA  


class AstraConfigDialog(QDialog):
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
        pick_method_group = QGroupBox("KICK START MODE")
        pick_layout = QVBoxLayout()
        self.pick_method_combo = QComboBox()
        self.pick_method_combo.addItems(["ECSV / txt", "Manual"])
        self.pick_method_combo.currentIndexChanged.connect(self.toggle_file_button)
        pick_layout.addWidget(self.pick_method_combo)

        self.file_picker_button = QPushButton("SELECT ECSV/TXT FILE")
        self.file_picker_button.clicked.connect(self.select_file)
        pick_layout.addWidget(self.file_picker_button)
        self.selected_file_label = QLabel("No file selected")
        pick_layout.addWidget(self.selected_file_label)

        pick_method_group.setLayout(pick_layout)
        main_layout.addWidget(pick_method_group)

        def add_grid_fields(field_dict, defaults, title):
            group = QGroupBox(title)
            layout = QGridLayout()
            for idx, (key, default) in enumerate(defaults.items()):
                label = QLabel(f"<b>{key}</b>:")
                field = QLineEdit(default)
                layout.addWidget(label, idx // 2, (idx % 2) * 2)
                layout.addWidget(field, idx // 2, (idx % 2) * 2 + 1)
                field_dict[key] = field
            group.setLayout(layout)
            return group

        # === PSO Settings ===
        self.pso_fields = {}
        pso_defaults = {
            "w (0-1)": "0.9", "c1 (0-1)": "0.4", "c2 (0-1)": "0.3",
            "m_iter": "100", "n_par": "100", "Vc (0-1)": "0.3",
            "ftol": "1e-4", "ftol_iter": "25", "expl_c": "3", "P_sigma": "3"
        }
        main_layout.addWidget(add_grid_fields(self.pso_fields, pso_defaults, "PSO PARAMETER SETTINGS"))

        # === ASTRA General Settings ===
        self.astra_fields = {}
        astra_defaults = {
            "O_sigma": "3", "m_SNR": "5",
            "P_c": "1.5", "sigma_i (px)": "2", "sigma_m": "1.2",
            "L_m": "1.5", "VERB": "False", "P_thresh" : "0.65"
        }
        main_layout.addWidget(add_grid_fields(self.astra_fields, astra_defaults, "ASTRA PARAMETER SETTINGS"))

        # === Kalman Filter Settings ===
        self.kalman_fields = {}
        kalman_defaults = {
            "Monotonicity": "True", "Use_Accel": "True", "Med_err (px)": "0.3"
        }
        main_layout.addWidget(add_grid_fields(self.kalman_fields, kalman_defaults, "KALMAN FILTER SETTINGS"))

        # === PARAMETER GUIDE ===
        guide_group = QGroupBox("PARAMETER GUIDE")
        guide_layout = QVBoxLayout()
        self.param_info = QTextEdit()
        self.param_info.setReadOnly(True)
        self.param_info.setHtml(
            "<b> MANUAL PICK MODE GUIDE </b><br>"
            "<b>1.</b> Pick two the leading edge of the first and last frame of the event.<br>"
            "NOTE: It is essential that the line outlined by the first and last picks perfectly intersect the meteor trajectory.<br>"
            "<b>2.</b> Pick three frame-adjacent leading-edge picks at the highest SNR near middle of event.<br>"
            "<b>3.</b> THEN, run ASTRA with the 'RUN ASTRA' button.<br><br>"
            "NOTE ON CHANGING PARAMETERS<br>"
            "Default parameters are optimized to work for most EMCCD data. Paramters are sensitive and may result in large changes in computation time and possible failure. Only change when dealing with extranous data."
            "All extra descriptions on how parameters affect the algorithm are with respect to increasing the value of the parameter.<br><br>"
            "<b>w</b>: PSO intertial weight - Increases parameter exploration<br>"
            "<b>c1</b>: PSO cognitive component (individual best) - Increases local parameter minimum exploration<br>"
            "<b>c2</b>: PSO social component (global best) - Decreases parameter exploration, improves convergence<br>"
            "<b>m_iter</b>: Max PSO iterations - Improves convergence, increases computation time. Must be matched with higher parameter exploration<br>"
            "<b>n_par</b>: Number of particles - Improves exploration, increases computation time. Must be matches with higher parameter exploration<br>"
            "<b>Vc</b>: Fraction of parameter space as max velocity - Improves exploration, must be matched with faster convergence<br>"
            "<b>ftol</b>: Min tolerance for convergence - Improves full local optimization, increases computation time<br>"
            "<b>ftol_iter</b>: Min itterations for convergence - Improves full local optimization, increases computation time<br>"
            "<b>expl_c</b>: Explorative coeffecient (disperses inital particle seeding) - Increases exploration, reduces local minimization<br>"
            "<b>P_sigma</b>: 2nd pass optimizer paramter bound coeffeicent - Improves ability for local minimizer to adjust from PSO result, reduces optimal local solution<br>"
            "<b>O_sigma</b>: Num STD above background to mask as star - Reduces chance for bright meteors to be masked as a star<br>"
            "<b>m_SNR</b>: Minimum SNR value to keep pick - Improves the quality of picks, reduces total picks<br>"
            "<b>P_c</b>: Streak cropping padding coeffeicent - Increases ability for ASTRA to recover from bad fits, increases computation time<br>"
            "<b>sigma_i</b>: Initial Gaussian sigma (height) guess - Improves ASTRA ability to fit large meteors, increases computation time and can throw off algorithm if inaccurate<br>"
            "<b>sigma_m</b>: Coeff to init. sigma guess as max bound - Improves ASTRA ability to fit large meteors, increases computation time and can throw off algorithm if inaccurate<br>"
            "<b>L_m</b>: Coeff to init. length guess as max bound - Improves ASTRA ability to fit large meteors, increases computation time and can throw off algorithm if inaccurate<br>"
            "<b>P_thresh</b>: Fraction of peak meteor brightness to threshold photometry pixels - less noise included in photometry, though possibly less of the meteor as well.<br>"
            "<b>VERB</b>: Verbose -shows testing data<br>"
            "<b>Monotonicity</b>: Enforce monotonic motion<br>"
            "<b>Use_Accel</b>: Enable acceleration model (default const. vel.)<br>"
            "<b>Med_err</b>: Estimated error at median SNR for R matrix - Increases power of Kalman filter, high values may fake data by overfitting<br>"
        )
        guide_layout.addWidget(self.param_info)
        guide_group.setLayout(guide_layout)
        main_layout.addWidget(guide_group)

        # === Progress Bar ===
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(QLabel("Progress:"))
        main_layout.addWidget(self.progress_bar)

        # === Control Buttons ===
        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("LOAD PICKS")
        self.run_astra_btn = QPushButton("RUN ASTRA")
        self.run_kalman_btn = QPushButton("RUN KALMAN")
        self.load_btn.clicked.connect(self.load_picks)
        self.run_astra_btn.clicked.connect(self.start_astra_thread)
        self.run_kalman_btn.clicked.connect(self.start_kalman_thread)
        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.run_astra_btn)
        btn_layout.addWidget(self.run_kalman_btn)
        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)
        self.toggle_file_button()

    def toggle_file_button(self):
        self.file_picker_button.setEnabled(self.pick_method_combo.currentText() == "ECSV / txt")

    def toggle_load_button(self):
        self.load_btn.setEnabled(self.pick_method_combo.currentText() == "ECSV / txt")


    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select the ECSV or TXT file",
            "",
            "ECSV or TXT files (*.ecsv *.txt);;All files (*)"
        )
        if file_path:
            self.selected_file_label.setText(file_path)

    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    def load_picks(self):
        self.store_config()
        if self.load_picks_callback:
            self.load_picks_callback(self.config)

    def store_config(self):
        self.config = {
            "pick_method": self.pick_method_combo.currentText(),
            "file_path": self.selected_file_label.text(),
            "pso": {k: v.text() for k, v in self.pso_fields.items()},
            "astra": {k: v.text() for k, v in self.astra_fields.items()},
            "kalman": {k: v.text() for k, v in self.kalman_fields.items()}
        }
        print("Stored ASTRA Config:", self.config)

    def get_config(self):
        return self.config
    
    def run_astra(self):
        self.store_config()
        if self.run_astra_callback:
            self.run_astra_callback(self.config)

    def run_kalman(self):
        self.store_config()
        if self.run_kalman_callback:
            self.run_kalman_callback(self.config)

    def start_astra_thread(self):
        from PyQt5 import QtCore
        self.store_config()
        config = self.get_config()

        self.thread = QThread()
        self.worker = AstraWorker(config, self.skyfit_instance)
        print('ASTRA Object Created! Processing beginning (30-80 seconds)...')
        self.worker.moveToThread(self.thread)

        self.run_astra_btn.setEnabled(False)
        self.run_kalman_btn.setEnabled(False)
        self.load_btn.setEnabled(False)

        self.worker.progress.connect(self.update_progress)

        self.thread.started.connect(self.worker.run)

        self.worker.results_ready.connect(
            self.skyfit_instance.integrate_astra_results, 
            QtCore.Qt.QueuedConnection
        )

        # Clean up and re-enable UI
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        

        # Restore interactivity
        self.worker.finished.connect(lambda: self.run_astra_btn.setEnabled(True))
        self.worker.finished.connect(lambda: self.run_kalman_btn.setEnabled(True))
        self.worker.finished.connect(lambda: self.load_btn.setEnabled(True))

        self.thread.start()

    def start_kalman_thread(self):
        self.store_config()
        self.config = self.get_config()

        self.run_kalman_btn.setEnabled(False)
        self.run_astra_btn.setEnabled(False)

        self.kalman_worker = KalmanWorker(self.skyfit_instance, self.config)
        self.kalman_worker.progress.connect(self.update_progress)
        self.kalman_worker.finished.connect(lambda: self.run_kalman_btn.setEnabled(True))
        self.kalman_worker.finished.connect(lambda: self.run_astra_btn.setEnabled(True))
        self.kalman_worker.finished.connect(lambda: self.update_progress(100))
        self.kalman_worker.start()

class KalmanWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, skyfit_instance, config):
        super().__init__()
        self.skyfit_instance = skyfit_instance
        self.config = config

    def run(self):
        self.skyfit_instance.run_kalman_from_config(self.config, progress_callback=self.progress.emit)
        self.finished.emit()

class AstraWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    results_ready = pyqtSignal(object)

    def __init__(self, config, skyfit_instance):
        super().__init__()
        self.config = config
        self.skyfit_instance = skyfit_instance

    def run(self):
        # Prepare data
        data_dict = self.skyfit_instance.prepare_astra_data(self.config)

        if data_dict is False:
            self.finished.emit()
            return

        # Run ASTRA here, directly in worker
        from .ASTRA import ASTRA
        astra = ASTRA(data_dict, progress_callback=self.progress.emit)
        astra.process()

        # Handle results via callback
        self.results_ready.emit(astra)
        self.finished.emit()

def launch_astra_gui(run_astra_callback=None,
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
    config = launch_astra_gui()
    if config:
        print("Returned config from GUI:")
        for section, values in config.items():
            print(f"[{section}]")
            if isinstance(values, dict):
                for k, v in values.items():
                    print(f"  {k}: {v}")
            else:
                print(f"  {values}")
