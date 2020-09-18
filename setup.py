import os
import sys
import subprocess

import numpy

from setuptools import setup, Extension, find_packages

from Cython.Build import cythonize


kht_module = Extension("kht_module",
                    sources = ["Native/Hough/kht.cpp",
                               "Native/Hough/buffer_2d.cpp",
                               "Native/Hough/eigen.cpp",
                               "Native/Hough/linking.cpp",
                               "Native/Hough/peak_detection.cpp",
                               "Native/Hough/subdivision.cpp",
                               "Native/Hough/voting.cpp"],
                    include_dirs = ["Native/Hough/"],
                    extra_compile_args=["-O3", "-Wall"], extra_link_args=["-O3", "-Wall"])



# Read requirements file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Init the submodules (python-dvr)
x = subprocess.call(['git','submodule','update','--init'])


### HANDLE DIFFERENT ONVIF LIBRARIES FOR Py 2 AND 3 ###

# Python 2 uses the 'onvif' library and Python 3 uses the onvif_zeep library
if sys.version_info[0] < 3:

    onvif_str = 'onvif_zeep'
    onvif_proper = 'onvif'

else:
    onvif_str = 'onvif'
    onvif_proper = 'onvif_zeep'


# Strip version numbers from requirements
reqs_stripped = [req.split('==')[0] for req in requirements]
reqs_stripped = [req.split('>=')[0] for req in reqs_stripped]
reqs_stripped = [req.split('<=')[0] for req in reqs_stripped]

if onvif_str in reqs_stripped:
    onvif_index = reqs_stripped.index(onvif_str)

    # Replace the onvif module with the correct module for this version
    requirements[onvif_index] = onvif_proper

### ###


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

setup (name = "RMS",
        version = "0.1",
        description = "Raspberry Pi Meteor Station",
        setup_requires=["numpy", 
        # Setuptools 18.0 properly handles Cython extensions.
            'setuptools>=18.0',
            'cython'],
        install_requires=requirements,
        data_files=[(os.path.join('Catalogs'), catalog_files)],
        ext_modules = [kht_module] + cythonize(cython_modules),
        packages=find_packages(),
        include_package_data=True,
        include_dirs=[numpy.get_include()]
        )
