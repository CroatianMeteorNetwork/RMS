#!/usr/bin/env python

import os
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt

from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP, raDecToXYPP
from RMS.Astrometry.Conversions import date2JD, jd2Date
from RMS.Formats.FFfile import validFFName, getMiddleTimeFF, filenameToDatetime
from RMS.Formats.FTPdetectinfo import readFTPdetectinfo
from RMS.Formats.FFfile import read as readFF
from RMS.Formats.Platepar import Platepar
from RMS.Math import angularSeparation
from Utils.ShowerAssociation import showerAssociation
from RMS.Routines.MaskImage import loadMask, MaskStructure


def drawConstellations(platepar, ff_file, separation_deg=90, color_bgra=None):
    if not color_bgra:
        color_bgr = [255, 0, 0, 192]
    img = np.zeros((platepar.Y_res, platepar.X_res, 4), dtype=np.uint8)
    img[:, :, 3] = 0  # Fully transparent
    fps = 25  # TODO get from config
    fftime_jd = date2JD(*getMiddleTimeFF(os.path.basename(ff_file), fps))
    constellations_path = os.path.join(os.path.dirname(__file__), "../share/constellation_lines.csv")
    lines = np.loadtxt(constellations_path, delimiter=",")
    from_ra, from_dec = lines[:, 0], lines[:, 1]
    to_ra, to_dec = lines[:, 2], lines[:, 3]
    from_x, from_y = raDecToXYPP(np.array(from_ra), np.array(from_dec), fftime_jd, platepar)
    ang_sep = np.rad2deg(angularSeparation(np.deg2rad(platepar.RA_d), np.deg2rad(platepar.dec_d), np.deg2rad(from_ra), np.deg2rad(from_dec)))
    to_x, to_y = raDecToXYPP(np.array(to_ra), to_dec, fftime_jd, platepar)
    for i in range(len(to_x)):
        if ang_sep[i] < separation_deg:
            cv2.line(img, (int(round(from_x[i])), int(round(from_y[i]))), (int(round(to_x[i])), int(round(to_y[i]))), color_bgr, 1)

    return img


if __name__ == "__main__":
    import argparse
    import RMS.ConfigReader as cr

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="""Draw constellations""")

    arg_parser.add_argument('platepars_file', help='Full path to a platepars_recalibrated file')
    arg_parser.add_argument('ff_file', help='Full path to an FF file')
    arg_parser.add_argument('-o', '--output', help='Output filename (default: deduce from FF filename)')
    arg_parser.add_argument('-r', '--resolution', help='Resolution (override platepar; this also resets all other platepar parameters)')

    cml_args = arg_parser.parse_args()

    outfilename = cml_args.output
    if outfilename is None:
        outfilename = cml_args.ff_file.rstrip(".fits") + "_constellations.png"

    with open(cml_args.platepars_file, 'r') as f:
        platepars = json.load(f)

    platepar_dict = platepars[os.path.basename(cml_args.ff_file)]
    platepar = Platepar()
    platepar.loadFromDict(platepar_dict)

    if cml_args.resolution is not None:
        platepar.X_res = int(cml_args.resolution)
        platepar.Y_res = int(cml_args.resolution)
        platepar.F_scale *= 0.5
        platepar.refraction = False
        platepar.resetDistortionParameters()

    img = drawConstellations(platepar, cml_args.ff_file)

    cv2.imwrite(outfilename, img)
    print("Wrote", outfilename)
