
from __future__ import print_function, division, absolute_import

import sys
import os
import numpy as np
import time

from RMS.Detection import thresholdImg, show
from RMS.Formats import FFfile

# Import old morph
from RMS.OLD import MorphologicalOperations as morph

# Cython init
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
from RMS.Routines.MorphCy import morphApply



# Run tests

# Extract file and directory
head, ff_name = os.path.split(sys.argv[1])
ff_path = os.path.abspath(head) + os.sep

# Load the FF bin file
ff = FFfile.read(ff_path, ff_name)

img_thresh = thresholdImg(ff, 1.8, 9)

show('thresh', img_thresh)

# Convert img to integer
img = img_thresh.astype(np.uint8)

# Old morph
img_old = np.copy(img)

t1 = time.clock()
img_old = morph.clean(img_old)
img_old = morph.bridge(img_old)
img_old = morph.close(img_old)
img_old = morph.thin2(img_old)

print('time for old:', time.clock() - t1)

show('old', img_old)

# New morph
t1 = time.clock()
img = morphApply(img, [1, 2, 3, 4])

print('time for new:', time.clock() - t1)
show('new', img)

show('diff', np.abs(img_old - img))

print('diff', np.sum(img_old - img))