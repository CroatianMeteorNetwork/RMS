from __future__ import annotations
from pathlib import Path
import sys
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

try:
    import numpy as np
    numpy_includes = [np.get_include()]
except ModuleNotFoundError:
    numpy_includes = []

# --- bootstrap ---------------------------------------------------------------
# This bootstrap code intercepts direct `python setup.py install` commands and 
# redirects them to use pip in editable mode. This handles a critical transition
# scenario:
#
# When RMS_Update.sh runs for the first time after this change:
# 1. The script is already loaded in memory with the old `python setup.py install`
# 2. Git pull updates the script to use `pip install -e .`
# 3. But the in-memory version still executes the old command
# 4. This bootstrap catches that and redirects it to the new pip command
#
# On subsequent runs, RMS_Update.sh will use the new command directly.
#
# How it works:
# - Detects if setup.py was called with 'install' argument
# - Excludes setuptools' internal flag '--old-and-unmanageable' to avoid loops
# - Redirects to pip install with editable mode (-e), no deps, and no build isolation
# - Exits with appropriate return code to maintain script compatibility
#
# This can be removed after a deprecation period once all users have migrated.
from subprocess import check_call, CalledProcessError
if "install" in sys.argv and "--old-and-unmanageable" not in sys.argv:
    try:
        check_call([sys.executable, "-m", "pip", "install",
                    "-e", ".", "--no-deps", "--no-build-isolation"])
    except CalledProcessError as exc:
        sys.exit(exc.returncode)
    sys.exit(0)
# -----------------------------------------------------------------------------    

# C/C++ extension
kht_module = Extension(
    "kht_module",
    sources=[
        "Native/Hough/kht.cpp",
        "Native/Hough/buffer_2d.cpp",
        "Native/Hough/eigen.cpp",
        "Native/Hough/linking.cpp",
        "Native/Hough/peak_detection.cpp",
        "Native/Hough/subdivision.cpp",
        "Native/Hough/voting.cpp",
    ],
    include_dirs=["Native/Hough/"],
    extra_compile_args=["-O3", "-Wall"],
)

# Cython extensions
cython_modules = [
    Extension("RMS.Astrometry.CyFunctions", ["RMS/Astrometry/CyFunctions.pyx"], include_dirs=numpy_includes),
    Extension("RMS.Routines.BinImageCy", ["RMS/Routines/BinImageCy.pyx"], include_dirs=numpy_includes),
    Extension("RMS.Routines.DynamicFTPCompressionCy", ["RMS/Routines/DynamicFTPCompressionCy.pyx"], include_dirs=numpy_includes),
    Extension("RMS.Routines.Grouping3Dcy", ["RMS/Routines/Grouping3Dcy.pyx"], include_dirs=numpy_includes),
    Extension("RMS.Routines.MorphCy", ["RMS/Routines/MorphCy.pyx"], include_dirs=numpy_includes),
    Extension("RMS.CompressionCy", ["RMS/CompressionCy.pyx"], include_dirs=numpy_includes),
    Extension("Utils.SaturationTools", ["Utils/SaturationTools.pyx"], include_dirs=numpy_includes),
]

# Runtime requirements
with open("requirements.txt") as fh:
    requirements = []
    for line in fh:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("git+"):
            requirements.append(stripped)

# Data files
base = Path(__file__).resolve().parent
catalog_files = [str(p.relative_to(base)) for p in (base / "Catalogs").glob("*")]
share_files   = [str(p.relative_to(base)) for p in (base / "share").glob("*") if p.is_file()]
plate_files   = [str(p.relative_to(base)) for p in (base / "share/platepar_templates").glob("*")]

setup(
    name="RMS",
    version="0.1",
    description="Raspberry Pi Meteor Station",
    packages=find_packages(),
    install_requires=requirements,
    data_files=[
        ("Catalogs", catalog_files),
        ("share", share_files),
        ("share/platepar_templates", plate_files),
    ],
    ext_modules=[kht_module] + cythonize(cython_modules),
    include_dirs=numpy_includes,
    include_package_data=True,
)
