from distutils.core import setup, Extension
import os

kht_module = Extension("kht_module",
                    sources = ["Native/Hough/kht.cpp",
                               "Native/Hough/buffer_2d.cpp",
                               "Native/Hough/eigen.cpp",
                               "Native/Hough/linking.cpp",
                               "Native/Hough/peak_detection.cpp",
                               "Native/Hough/subdivision.cpp",
                               "Native/Hough/voting.cpp"],
                    include_dirs = ["Native/Hough/"],
                    extra_compile_args=["-O3"], extra_link_args=["-O3"])

setup (name = "RMS",
       version = "0.0",
       description = "Video meteor station for under 100 bucks",
       ext_modules = [kht_module])