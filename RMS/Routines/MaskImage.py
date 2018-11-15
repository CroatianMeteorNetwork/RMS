# RPi Meteor Station
# Copyright (C) 2016  Dario Zubovic, Denis Vida
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

from __future__ import division, absolute_import, print_function

import os
import logging

import numpy as np
from scipy import misc


# Get the logger from the main module
log = logging.getLogger("logger")



def loadMask(mask_file):
	""" Load the mask image. """

	# If there is no mask file
	if not os.path.isfile(mask_file):
		return (False, 0)

	# Load the mask file
	mask = misc.imread(mask_file, flatten=0)

	# Convert the RGB image to one channel image (if not already one channel)
	try:
		mask = mask[:,:,0]
	except:
		pass

	return (True, mask)



def maskImage(input_image, mask):
	""" Apply masking to the given image. """

	# If the image dimensions don't agree, dont apply the mask
	if input_image.shape != mask.shape:
		log.warning('Image and mask dimensions do not agree! Skipping masking...')
		return input_image

	# Set all image pixels where the mask is black to 0
	input_image[mask == 0] = np.mean(input_image)

	return input_image



def applyMask(input_image, mask_tuple, ff_flag=False):
	""" Apply a mask to the given image array or FF file. """

	# Check if the loading procedure determined if the mask file exists
	mask_flag, mask = mask_tuple

	if not mask_flag:
		return input_image

	# Apply masking to an FF file
	if ff_flag:
		input_image.maxpixel = maskImage(input_image.maxpixel, mask)
		input_image.avepixel = maskImage(input_image.avepixel, mask)
		input_image.stdpixel = maskImage(input_image.stdpixel, mask)
		#input_image.maxframe = maskImage(input_image.maxframe, mask)

		return input_image

	# Apply the mask to a regular image array
	else:
		return maskImage(input_image, mask)




if __name__ == '__main__':

	mask_file = '../../mask.bmp'

	print(loadMask(mask_file))