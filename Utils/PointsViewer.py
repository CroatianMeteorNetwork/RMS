""" Show points detected by the video extractor on a given FF file as a 3D plot. """

from __future__ import print_function, division, absolute_import

import os
import argparse

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from RMS.CLITools import addConfigArgument, loadConfig
from RMS.Formats import FFfile
from RMS import VideoExtraction


def view(dir_path, file_name, config):

    ff = FFfile.read(dir_path, file_name, array=True)

    ve = VideoExtraction.Extractor(config)
    ve.frames = np.empty((256, ff.nrows, ff.ncols))
    ve.compressed = ff.array

    points = np.array(ve.findPoints())

    plot(points, ff.nrows//config.f, ff.ncols//config.f, file_name)

def plot(points, y_dim, x_dim, name):
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    plt.title(name)

    y = points[:,0]
    x = points[:,1]
    z = points[:,2]

    # Plot points in 3D
    ax.scatter(x, y, z)

    # Set axes limits
    ax.set_zlim(0, 255)
    plt.xlim([0, x_dim])
    plt.ylim([0, y_dim])

    ax.set_ylabel("Y")
    ax.set_xlabel("X")
    ax.set_zlabel("Time")

    plt.show()


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="""Show points detected by the video extractor \
on the given FF file as a 3D plot.""")

    arg_parser.add_argument('ff_path', metavar='FF_PATH', type=str,
        help='Path to the FF file to inspect.')

    addConfigArgument(arg_parser)

    cml_args = arg_parser.parse_args()

    dir_path, file_name = os.path.split(os.path.abspath(cml_args.ff_path))

    config = loadConfig(cml_args.config, dir_path)

    view(dir_path, file_name, config)
