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
            "ftol": "1e-4", "ftol_iter": "25", "expl_c": "3"
        }
        main_layout.addWidget(add_grid_fields(self.pso_fields, pso_defaults, "PSO PARAMETER SETTINGS"))

        # === ASTRA General Settings ===
        self.astra_fields = {}
        astra_defaults = {
            "P_sigma": "3", "O_sigma": "3", "m_SNR": "10",
            "P_c": "1.5", "sigma_i (px)": "2", "sigma_m": "1.2",
            "L_m": "1.5", "VERB": "False"
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
            "<b>1.</b> Pick three frame-adjacent leading-edge picks at the highest SNR near middle of event.<br>"
            "<b>2.</b> Pick two the leading edge of the first and last frame of the event.<br>"
            "NOTE: It is essential that the line outlined by the first and last picks perfectly intersect the meteor trajectory.<br>"
            "<b>3.</b> THEN, run ASTRA with the 'RUN ASTRA' button.<br><br>"
            "<b>w</b>: PSO intertial weight<br>"
            "<b>c1</b>: PSO social component (individual best)<br>"
            "<b>c2</b>: PSO cognitive component (global best)<br>"
            "<b>m_iter</b>: Max PSO iterations<br>"
            "<b>n_par</b>: Number of particles<br>"
            "<b>Vc</b>: Fraction of parameter space as max velocity<br>"
            "<b>tol</b>: Min tolerance for convergence<br>"
            "<b>tol_iter</b>: Min itterations for convergence<br>"
            "<b>expl_c</b>: Explorative coeffecient (inital particle seeding)<br>"
            "<b>P_sigma</b>: 2nd pass optimizer paramter bound coeffeicent<br>"
            "<b>O_sigma</b>: Num STD above background to mask as star<br>"
            "<b>m_SNR</b>: Minimum SNR value to keep pick<br>"
            "<b>P_c</b>: Streak cropping padding coeffeicent<br>"
            "<b>sigma_i</b>: Initial Gaussian sigma (height) guess<br>"
            "<b>sigma_m</b>: Coeff to init. sigma guess as max bound<br>"
            "<b>L_m</b>: Coeff to init. length guess as max bound<br>"
            "<b>VERB</b>: Verbose<br>"
            "<b>Monotonicity</b>: Enforce monotonic motion<br>"
            "<b>Use_Accel</b>: Enable acceleration model (default const. vel.)<br>"
            "<b>Med_err</b>: Estimated error at median SNR for R matrix<br>"
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
        self.run_kalman_btn.clicked.connect(self.run_kalman)
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
        self.store_config()
        self.run_astra_btn.setEnabled(False)
        self.run_kalman_btn.setEnabled(False)

        self.worker_thread = QThread()
        self.worker = AstraWorker(self.config, self.skyfit_instance)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_progress)

        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.worker.finished.connect(lambda: self.run_astra_btn.setEnabled(True))
        self.worker.finished.connect(lambda: self.run_kalman_btn.setEnabled(True))

        self.worker_thread.start()


class AstraWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def __init__(self, config, skyfit_instance):
        super().__init__()
        self.config = config
        self.skyfit_instance = skyfit_instance

    def run(self):
        # Temporarily replace progress callback
        self.skyfit_instance.run_astra_from_config(self.config, progress_callback=self.progress.emit)
        self.finished.emit()


def launch_astra_gui(run_astra_callback=None, run_kalman_callback=None, load_picks_callback=None):
    import sys
    app = QApplication.instance() or QApplication(sys.argv)
    dialog = AstraConfigDialog(run_astra_callback=run_astra_callback, run_kalman_callback=run_kalman_callback, load_picks_callback=load_picks_callback)
    if dialog.exec_():
        return dialog.get_config()
    return None


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
