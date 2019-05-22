""" Functions to load the shower catalog. """

from __future__ import print_function, division, absolute_import



import os

import numpy as np



def loadShowers(dir_path, file_name):
	""" Loads the given shower CSV file. """


	shower_data = np.genfromtxt(os.path.join(dir_path, file_name), delimiter='|', dtype=None, autostrip=True,
		encoding=None)

	return shower_data





if __name__ == "__main__":


	shower_data = loadShowers("share", "established_showers.csv")

	print(shower_data)