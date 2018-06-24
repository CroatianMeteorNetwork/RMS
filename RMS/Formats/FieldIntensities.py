# RPi Meteor Station
# Copyright (C) 2017  Denis Vida
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

""" Format for saving image intensitites per every field. """

from __future__ import print_function, division, absolute_import

import os

import numpy as np



def saveFieldIntensitiesText(intensity_array, dir_path, file_name, deinterlace=False):
	""" Saves sums of intensities per every field to a comma separated file. 
	
	Arguments:
		intensity_array: [ndarray] Numpy array containing the sums of intensitites per every field.
		dir_path: [str] Path to the directory where the file will be saved.
		file_name: [str] Name of the file in which the data will be saved.
	"""

	file_name = "FS_" + file_name + '_fieldsum.txt'


	with open(os.path.join(dir_path, file_name), 'w') as f:

		# Write header
		f.write(file_name + '\n\n')
		f.write('Frame, Intensity sum\n')

		if deinterlace:
			deinterlace_flag = 2.0
		else:
			deinterlace_flag = 1.0

		for i, value in enumerate(intensity_array):

			# Calculate the half frame
			half_frame = float(i)/deinterlace_flag

			f.write("{:.1f}, {:d}\n".format(half_frame, value))


	return file_name



def saveFieldIntensitiesBin(intensity_array, dir_path, file_name):
	""" Saves sums of intensities per every field to a binary file. 
	
	Arguments:
		intensity_array: [ndarray] Numpy array containing the sums of intensitites per every field.
		dir_path: [str] Path to the directory where the file will be saved.
		file_name: [str] Name of the file in which the data will be saved.
	"""

	file_name = "FS_" + file_name + '_fieldsum.bin'


	with open(os.path.join(dir_path, file_name), 'wb') as fid:

		# Write the number of entries in the header
		np.array(len(intensity_array)).astype(np.uint16).tofile(fid)

		# Write intensities
		for value in intensity_array:
			np.array(value).astype(np.uint32).tofile(fid)


	return file_name



def readFieldIntensitiesBin(dir_path, file_name, deinterlace=False):
	""" Read the field intensities form a binary file.
	
	Arguments:
		dir_path: [str] Path to the directory where the file is located.
		file_name: [str] Name of the file.
	"""

	with open(os.path.join(dir_path, file_name), 'rb') as fid:

		# Read the number of entries
		n_entries = int(np.fromfile(fid, dtype=np.uint16, count = 1))

		intensity_array = np.zeros(n_entries, dtype=np.uint32)
		half_frames = np.zeros(n_entries)

		if deinterlace:
			deinterlace_flag = 2.0
		else:
			deinterlace_flag = 1.0

		# Read individual entries
		for i in range(n_entries):

			# Calculate the half frame
			half_frames[i] = float(i)/deinterlace_flag

			# Read the summmed field intensity
			intensity_array[i] = int(np.fromfile(fid, dtype=np.uint32, count = 1))


		return half_frames, intensity_array



def convertFieldIntensityBinToTxt(dir_path, file_name, deinterlace=False):
	""" Converts the field sum binary file to a text file
	
	Arguments:
		dir_path: [str] Path to the directory where the file is located.
		file_name: [str] Name of the file.

	"""

	# Read the binary file
	half_frames, intensity_array = readFieldIntensitiesBin(dir_path, file_name, deinterlace=deinterlace)

	# Replace the bin with txt
	file_name = file_name.replace('FS_', '')
	file_name = file_name.replace('_fieldsum.bin', '')

	# Save the field sums to a text file
	saveFieldIntensitiesText(intensity_array, dir_path, file_name, deinterlace=deinterlace)

