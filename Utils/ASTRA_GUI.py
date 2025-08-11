from PyQt5.QtWidgets import (
    QDialog, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
    QLineEdit, QGroupBox, QFormLayout, QComboBox, QFileDialog,
    QProgressBar, QTextEdit, QApplication, QWidget, QGridLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from .ASTRA import ASTRA  
import html, re


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

        # === LATEX CONVERSIONS ===
        _GREEK = {
        "alpha":"α","beta":"β","gamma":"γ","delta":"δ","epsilon":"ε","zeta":"ζ","eta":"η",
        "theta":"θ","iota":"ι","kappa":"κ","lambda":"λ","mu":"μ","nu":"ν","xi":"ξ",
        "pi":"π","rho":"ρ","sigma":"σ","tau":"τ","upsilon":"υ","phi":"φ","chi":"χ",
        "psi":"ψ","omega":"ω",
        }
        
        def to_html_math(s: str) -> str:
            # escape first, we’ll insert our own tags
            t = html.escape(s)

            # 1) greek name → symbol (word boundaries; case-sensitive map above)
            for name, sym in _GREEK.items():
                t = re.sub(rf'\b{re.escape(name)}\b', sym, t)

            # 2) underscores → <sub>…</sub>, e.g., sigma_i -> σ<sub>i</sub>, Med_err -> Med<sub>err</sub>
            # handles multiple: a_b_c -> a<sub>b</sub><sub>c</sub>
            def _subber(m):
                base = m.group(1)
                subs = m.group(2)
                # split on underscores inside the suffix and nest subs
                out = base
                for part in subs.split('_'):    
                    out += f'<sub>{part}</sub>'
                return out
            t = re.sub(r'([A-Za-zΑ-Ωα-ω]+)_([A-Za-z0-9_]+)', _subber, t)

            # 3) simple power like ^2 → <sup>2</sup>
            t = re.sub(r'\^([0-9]+)', r'<sup>\1</sup>', t)

            # 4) replace ASCII ranges like (0-1) → [0, 1] (purely cosmetic)
            t = t.replace('(0-1)', '[0, 1]')

            # Tell Qt “this is rich text” by starting with a tag
            return f'<span>{t}</span>'

        def add_grid_fields(field_dict, defaults, title, tooltips=None):
            group = QGroupBox(title)
            layout = QGridLayout()
            tts = tooltips or {}
            for idx, (key, default) in enumerate(defaults.items()):
                key_html = to_html_math(key)                 # pretty label text
                label = QLabel(f"{key_html}:")
                # tooltips: format if present
                tt_raw = tts.get(key, "")
                tt_html = to_html_math(tt_raw) if tt_raw else ""
                if tt_html:
                    label.setToolTip(tt_html)

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
            "w (0-1)": "0.9", "c1 (0-1)": "0.4", "c2 (0-1)": "0.3",
            "m_iter": "100", "n_par": "100", "Vc (0-1)": "0.3",
            "ftol": "1e-4", "ftol_iter": "25", "expl_c": "3", "P_sigma": "3"
        }

        # === ASTRA General Settings ===
        self.astra_fields = {}
        astra_defaults = {
            "O_sigma": "3", "m_SNR": "5",
            "P_c": "1.5", "sigma_i (px)": "2", "sigma_m": "1.2",
            "L_m": "1.5", "VERB": "False", "P_thresh" : "0.65"
        }

        # === Kalman Filter Settings ===
        self.kalman_fields = {}
        kalman_defaults = {
            "Monotonicity": "True", "Use_Accel": "True", "Med_err (px)": "0.3"
        }

        # === PARAMETER GUIDE ===
        PSO_TT = {
            "w (0-1)": "PSO inertia (exploration vs exploitation). Higher = more exploration.",
            "c1 (0-1)": "Cognitive weight (pull to particle’s best).",
            "c2 (0-1)": "Social weight (pull to global best).",
            "m_iter": "Maximum PSO iterations.",
            "n_par": "Number of particles.",
            "Vc (0-1)": "Max velocity as fraction of bound width.",
            "ftol": "Stop when objective change < ftol.",
            "ftol_iter": "Consecutive iters below ftol to stop.",
            "expl_c": "Initial seeding spread coefficient.",
            "P_sigma": "Second-pass bound looseness for local fitting."
        }

        ASTRA_TT = {
            "O_sigma": "Background mask threshold (σ above mean).",
            "m_SNR": "Minimum SNR to keep a pick.",
            "P_c": "Crop padding coefficient.",
            "sigma_i (px)": "Initial Gaussian σ guess (px).",
            "sigma_m": "Max σ multiplier (upper bound).",
            "L_m": "Max length multiplier (upper bound).",
            "P_thresh": "Photometry threshold (fraction of peak).",
            "VERB": "Verbose logging (True/False)."
        }

        KALMAN_TT = {
            "Monotonicity": "Enforce monotonic motion along dominant axis.",
            "Use_Accel": "Use constant-acceleration model (else CV).",
            "Med_err (px)": "R at median SNR (px) — higher trusts model more."
        }

        main_layout.addWidget(
            add_grid_fields(self.pso_fields, pso_defaults, "PSO PARAMETER SETTINGS", PSO_TT)
        )
        main_layout.addWidget(
            add_grid_fields(self.astra_fields, astra_defaults, "ASTRA PARAMETER SETTINGS", ASTRA_TT)
        )
        main_layout.addWidget(
            add_grid_fields(self.kalman_fields, kalman_defaults, "KALMAN FILTER SETTINGS", KALMAN_TT)
        )

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
