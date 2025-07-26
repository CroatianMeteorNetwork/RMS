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
    requirements = [line.strip() for line in fh if line and not line.startswith("git+")]

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
