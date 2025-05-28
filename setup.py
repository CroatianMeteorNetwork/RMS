from __future__ import print_function, division, absolute_import

import os
import sys
import subprocess
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


def isPackageInstalled(package_name):
    """Check if a package is installed."""
    try:
        print("Checking if {:s} is installed...".format(package_name))
        subprocess.check_call([sys.executable, '-c', 'import {:s}'.format(package_name)])
        print("{:s} is already installed.".format(package_name))
        return True
    except subprocess.CalledProcessError:
        print("{:s} is not installed.".format(package_name))
        return False


def attemptInstall(package):
    """Attempt to install a package using pip."""
    try:
        print("Attempting to install {:s}...".format(package))
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        print("Successfully installed {:s}.".format(package))
        return True
    except subprocess.CalledProcessError:
        print("Failed to install {:s}.".format(package))
        return False



# Check if TensorFlow is already installed
if not isPackageInstalled('tensorflow'):
    # Attempt to install tflite-runtime
    if not attemptInstall('tflite-runtime'):
        # If tflite-runtime fails, install TensorFlow
        attemptInstall('tensorflow')

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
    if any(arch in os.uname()[4].lower() for arch in ['arm', 'aarch']):
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
catalog_files = [
    os.path.join('Catalogs', file_name) for file_name in os.listdir(os.path.join(dir_path, 'Catalogs'))
    ]
share_files = [
    os.path.join('share', file_name) for file_name in os.listdir(os.path.join(dir_path, 'share')) 
        if os.path.isfile(os.path.join(dir_path, 'share', file_name))
        ]
platepar_templates = [
    os.path.join('share', 'platepar_templates', file_name) 
        for file_name in os.listdir(os.path.join(dir_path, 'share', 'platepar_templates')) 
        if os.path.isfile(os.path.join(dir_path, 'share', 'platepar_templates', file_name))
        ]

setup (name = "RMS",
        version = "0.1",
        description = "Raspberry Pi Meteor Station",
        setup_requires=["numpy", 
        # Setuptools 18.0 properly handles Cython extensions.
            'setuptools>=18.0',
            'cython'],
        install_requires=requirements,
        data_files=[('Catalogs', catalog_files), ('share', share_files), ('share/platepar_templates', platepar_templates)],
        ext_modules = [kht_module] + cythonize(cython_modules),
        packages=find_packages_func(),
        include_package_data=True,
        include_dirs=[numpy.get_include()]
        )
