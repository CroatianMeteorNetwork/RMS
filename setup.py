from setuptools import setup, Extension, find_packages

import os
import sys

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



setup (name = "RMS",
        version = "0.1",
        description = "Raspberry Pi Meteor Station",
        setup_requires=["numpy"],
        install_requires=requirements,
        ext_modules = [kht_module],
        packages=find_packages(),
        include_package_data=True
        )
