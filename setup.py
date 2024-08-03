import os
import sys

import numpy

from setuptools import setup, Extension, find_packages

from Cython.Build import cythonize

# Determine find_packages function to use depending on Python version
find_packages_func = find_packages

# If in Python 3.3 or later, load find_namespace_packages()
if sys.version_info >= (3, 3):
    from setuptools import find_namespace_packages
    find_packages_func = find_namespace_packages


kht_module = Extension("kht_module",
                    sources = ["Native/Hough/kht.cpp",
                               "Native/Hough/buffer_2d.cpp",
                               "Native/Hough/eigen.cpp",
                               "Native/Hough/linking.cpp",
                               "Native/Hough/peak_detection.cpp",
                               "Native/Hough/subdivision.cpp",
                               "Native/Hough/voting.cpp"],
                    include_dirs = ["Native/Hough/"],
                    extra_compile_args=["-O3", "-Wall"])



# Read requirements file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()
# drop unsupported git refs for install_requires https://github.com/pypa/setuptools/issues/1052
for requirement in requirements:
    if requirement.startswith("git+"):
        requirements.remove(requirement)

### Add rawpy is running on Windows or Linux (not the Pi) ###

# Check if running on Windows
if 'win' in sys.platform and "darwin" not in sys.platform:
    requirements.append("rawpy")

# Check if running on Linux
else:

    # Check if running on the Pi
    if 'arm' in os.uname()[4]:
        print("Not installing rawpy because it is not available on the Pi...")

    else:
        requirements.append("rawpy")

# Cython modules which will be compiled on setup
cython_modules = [
    Extension('RMS.Astrometry.CyFunctions', sources=['RMS/Astrometry/CyFunctions.pyx'], \
        include_dirs=[numpy.get_include()]),
    Extension('RMS.Routines.BinImageCy', sources=['RMS/Routines/BinImageCy.pyx'], \
        include_dirs=[numpy.get_include()]),
    Extension('RMS.Routines.DynamicFTPCompressionCy', sources=['RMS/Routines/DynamicFTPCompressionCy.pyx'], \
        include_dirs=[numpy.get_include()]),
    Extension('RMS.Routines.Grouping3Dcy', sources=['RMS/Routines/Grouping3Dcy.pyx'], \
        include_dirs=[numpy.get_include()]),
    Extension('RMS.Routines.MorphCy', sources=['RMS/Routines/MorphCy.pyx'], \
        include_dirs=[numpy.get_include()]),
    Extension('RMS.CompressionCy', sources=['RMS/CompressionCy.pyx'], \
        include_dirs=[numpy.get_include()]),
    Extension('Utils.SaturationTools', sources=['Utils/SaturationTools.pyx'], \
        include_dirs=[numpy.get_include()])
    ]


# Get all data files
dir_path = os.path.split(os.path.abspath(__file__))[0]
catalog_files = [os.path.join('Catalogs', file_name) for file_name in os.listdir(os.path.join(dir_path, 'Catalogs'))]
share_files = [os.path.join('share', file_name) for file_name in os.listdir(os.path.join(dir_path, 'share'))]

setup (name = "RMS",
        version = "0.1",
        description = "Raspberry Pi Meteor Station",
        setup_requires=["numpy", 
        # Setuptools 18.0 properly handles Cython extensions.
            'setuptools>=18.0',
            'cython'],
        install_requires=requirements,
        data_files=[('Catalogs', catalog_files), ('share', share_files)],
        ext_modules = [kht_module] + cythonize(cython_modules),
        packages=find_packages_func(),
        include_package_data=True,
        include_dirs=[numpy.get_include()]
        )
