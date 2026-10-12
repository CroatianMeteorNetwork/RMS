""" Importing the detection doesn't need the compiled KHT module, which only the KHT line finder uses (the RANSAC
    line finder works without it, and the module can be rebuilt by pyximport when it is first used). """

import os
import subprocess
import sys


REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_detection_import_does_not_load_kht():

    code = "import sys, RMS.Detection; print('RMS.Routines.Kht' in sys.modules)"
    result = subprocess.run([sys.executable, '-c', code], cwd=REPO_DIR, capture_output=True, text=True, timeout=120)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == 'False'
