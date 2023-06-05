""""" Automatically uploads data files based on time and trajectory information given on a website. """

from __future__ import print_function, division, absolute_import

import numpy as np

from RMS.EventMonitor import EventContainer, EventMonitor, convertgmntimetoposix
import RMS.ConfigReader as cr

import os
import json
import time
import datetime
import argparse
import pathlib
from RMS.Formats.Platepar import Platepar
from datetime import datetime
from dateutil import parser
from tqdm import tqdm
import pandas as pd
import numpy as np

def analysetestresults(config):

    #build the dataframe

    allcameras= []
    with open(os.path.expanduser(os.path.join(config.data_dir, "testlog")), 'rt') as logfile:
        readingheader=True
        body = header = ""
        for line in logfile:
            readingheader = True if 'GMN' in line else False
            header = line.replace("'","").replace(" ","") if readingheader else header
            body = line if not readingheader else body
            cameralist = header.split('[')[1].split(']')[0].split(',')

            for camera in cameralist:
                if camera not in allcameras:
                    allcameras.append(camera)

    # Build the data frame for the analysis

    allcameras.sort()

    catlist = ['Used', 'Predicted', 'Predicted and used', 'Predicted not used', 'Used not Predicted']

    df = pd.DataFrame(data = allcameras)
    df.rename(columns={0:'Camera'}, inplace=True)

    for category in catlist:
        df[category] = 0


    trajectoriesanalysed = observations = 0
    predictedcameras = usedcameras =  []
    with open(os.path.expanduser(os.path.join(config.data_dir, "testlog")), 'rt') as logfile:
        readingheader=True
        body = header = ""
        for line in tqdm(logfile):


            # Populate used not predicted
            if 'GMN' in line and not readingheader:
                trajectoriesanalysed += 1
                for camera in usedcameras:

                    if camera not in predictedcameras:

                        df.loc[df['Camera'].isin([camera]), 'Used not Predicted'] += 1  # df.loc[df['Camera'].isin([predictedcamera]), 'Predicted and used'] + 1
            else:
                observations += 1

            readingheader = True if 'GMN' in line else False
            header = line.replace("'","").replace(" ","") if readingheader else header
            body = line if not readingheader else body

            # Populate the Used column
            if readingheader:
                predictedcameras = []
                usedcameras = header.split('[')[1].split(']')[0].split(',')
                df.loc[df['Camera'].isin(usedcameras), 'Used'] += 1  #df.loc[df['Camera'].isin(usedcameralist), 'Used'] + 1

            # Build up the predicted camera list
            if not readingheader:
                predictedcamera = body.split(" ")[3]
                predictedcameras.append(predictedcamera)

            # Populate the Predicted column - the camera was in the prediction
            if not readingheader:
                predictedcamera = body.split(" ")[3]
                df.loc[df['Camera'].isin([predictedcamera]), 'Predicted'] += 1   #df.loc[df['Camera'].isin([predictedcamera]), 'Predicted'] + 1

            # Populate the Predicted and used

            if not readingheader:
                if predictedcamera in usedcameras:
                  df.loc[df['Camera'].isin([predictedcamera]), 'Predicted and used'] += 1 #df.loc[df['Camera'].isin([predictedcamera]), 'Predicted and used'] + 1

            # Populate predicted not used

            if not readingheader:
                if predictedcamera not in usedcameras:
                  df.loc[df['Camera'].isin([predictedcamera]), 'Predicted not used'] += 1 #df.loc[df['Camera'].isin([predictedcamera]), 'Predicted and used'] + 1






    print("Results")
    print("{} trajectories and {} observations were analysed".format(trajectoriesanalysed, observations))
    print(df.to_string())

if __name__ == "__main__":


    arg_parser = argparse.ArgumentParser(description="""Analyse test results from EventMonitor tests. \
        """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
                            help="Path to a config file which will be used instead of the default one.")

    cml_args = arg_parser.parse_args()

    config = cr.loadConfigFromDirectory(cml_args.config, os.path.abspath('eventmonitor'))
    analysetestresults(config)