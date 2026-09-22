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

from __future__ import absolute_import, division, print_function

import os
import zipfile

import numpy as np
import cv2

from RMS.Logger import getLogger
from RMS.Routines.Image import loadImage

# Get the logger from the main module
log = getLogger("rmslogger")



class MaskStructure(object):
    def __init__(self, img):
        """ Structure for holding the mask. This is used so the mask can be hashed. """
        
        self.img = img

        if img is not None:
            
            # Get the mask resolution
            self.height = img.shape[0]
            self.width = img.shape[1]

        else:
            self.height = None
            self.width = None

    def resetEmpty(self, x_res, y_res):
        """ Reset the mask to an empty array. """

        self.img = np.full((y_res, x_res), 255, dtype=np.uint8)


    def checkResolution(self, x_res, y_res):
        """ Check if the mask has the given resolution. """

        if self.img is not None:

            if (self.img.shape[0] == y_res) and (self.img.shape[1] == x_res):

                return True

            else:

                return False

        return None


    def checkMask(self, x_res, y_res):
        """ Check the if the mask resolution matches and reset it if it doesn't. """

        if self.img is None:
            self.resetEmpty(x_res, y_res)

        elif not self.checkResolution(x_res, y_res):
            print("MASK RESET because the resolution didn't match!")
            self.resetEmpty(x_res, y_res)



def getMaskFile(dir_path, config, file_list=None, default_as_backup=False):
    """ Get the mask file from the given directory. If the mask file is not found, return None.
        It will also check if the mask is a zip file and load it if it is.

    Arguments:
        dir_path: [str] Path to the directory where the mask file is located.
        config: [Config] Configuration object.

    Keyword arguments:
        file_list: [list] List of files in the directory. If None, the files will be listed.
        default_as_backup: [bool] If True, the default mask file will be used as a backup, from the RMS 
            directory.

    Returns:
        mask: [MaskStructure] Mask structure object. If the mask file is not found, None is returned.
    """

    mask = None

    if file_list is None:
        file_list = os.listdir(dir_path)

    # Look through files and if there is mask.bmp or mask.zip, keep track of that then load it
    mask_status = max(
        2*(os.path.splitext(os.path.basename(config.mask_file))[0] == os.path.splitext(os.path.basename(filename))[0]) 
            - filename.endswith('.zip')
               for filename in file_list
    )
    
    # If a mask file is found, load it
    if mask_status > 0:
        
        mask_path = os.path.join(
            dir_path, 
            config.mask_file if mask_status == 2 else os.path.splitext(
                os.path.basename(config.mask_file))[0] + '.zip'
        )
        
        print("Loading mask:", mask_path)
        mask = loadMask(mask_path)

        if mask is None:
            print("  Mask file could not be loaded!")
        else:
            print("  Mask file loaded!")
            

    # If the mask file is not found, use the default mask file as a backup
    if (mask is None) and default_as_backup:

        # Path to the mask file in the config directory
        default_mask_path = os.path.join(config.config_file_path, config.mask_file)

        # Check if the mask file is in the config directory, if not, use the default mask file
        if not os.path.isfile(default_mask_path):

            # Path to the mask in RMS source directory
            default_mask_path = os.path.join(config.rms_root_dir, config.mask_file)

        print("Mask file not found! Using default mask file as a backup:", default_mask_path)

        mask = loadMask(default_mask_path)

        if mask is None:
            print("  Default mask file could not be loaded!")
        else:
            print("  Default mask file loaded!")


    if mask is None:
        print("No mask used!")

    return mask
    

def loadMask(mask_file):
    """ Load the mask image. """

    # If there is no mask file
    if not os.path.isfile(mask_file):
        return None

    # Load the mask file
    try:

        # Load a mask from zip
        if mask_file.endswith('.zip'):

            with zipfile.ZipFile(mask_file, 'r') as archive:
                
                data = archive.read('mask.bmp')
                mask = cv2.imdecode(np.frombuffer(data, np.uint8), 1)

        else:
                
            mask = loadImage(mask_file, flatten=0)
        
    except:
        print("WARNING! The mask file could not be loaded! File path: {:s}".format(mask_file))
        return None

    # Convert the RGB image to one channel image (if not already one channel)
    try:
        mask = mask[:,:,0]
    except:
        pass


    mask_struct = MaskStructure(mask)

    return mask_struct



def maskImage(input_image, mask, image=False):
    """ Apply masking to the given image. 

    Keyword arguments:
        image: [bool] If True, the image for the mask was given, and no the MaskStructure instance.
    """


    if not image:
        mask = mask.img

    # If the image dimensions don't agree, dont apply the mask
    if input_image.shape != mask.shape:
        # log.warning('Image and mask dimensions do not agree! Skipping masking...')
        return input_image

    # Set all image pixels where the mask is black to the mean value of the image
    input_image[mask == 0] = np.mean(input_image[mask > 0])

    return input_image



def applyMask(input_image, mask, ff_flag=False, image=False):
    """ Apply a mask to the given image array or FF file. 
    
    Keyword arguments:
        image: [bool] If True, the image for the mask was given, and not the MaskStructure instance.
    """

    if mask is None:
        return input_image

    # Apply masking to an FF file
    if ff_flag:
        input_image.maxpixel = maskImage(input_image.maxpixel, mask, image=image)
        input_image.avepixel = maskImage(input_image.avepixel, mask, image=image)
        input_image.stdpixel = maskImage(input_image.stdpixel, mask, image=image)
        #input_image.maxframe = maskImage(input_image.maxframe, mask)

        return input_image

    # Apply the mask to a regular image array
    else:
        return maskImage(input_image, mask, image=image)




if __name__ == '__main__':

    mask_file = '../../mask.bmp'

    print(loadMask(mask_file))



# Pixel encoding of the SkyFit2 brush paint layer
PAINT_UNTOUCHED = 0     # transparent, the polygon layer shows through
PAINT_MASKED = 1        # painted, the pixel is masked
PAINT_UNMASKED = 2      # erased, the pixel is unmasked even when inside a polygon



def compositeMaskLayers(mask_polygons, paint_layer, img_width, img_height, masked_value=0, \
    unmasked_value=255):
    """ Render the editable mask layers (polygons plus the brush paint layer) into a single mask image.

        The polygons are filled first, then the paint layer is composited on top so that brush-painted
        pixels mask and brush-erased pixels unmask, overriding any polygon beneath.

    Arguments:
        mask_polygons: [list] List of polygons, each a list of (x, y) image coordinates.
        paint_layer: [ndarray or None] uint8 (img_height, img_width) brush layer with PAINT_UNTOUCHED,
            PAINT_MASKED and PAINT_UNMASKED values. A layer of a different size is resampled to the image
            size with nearest-neighbour interpolation.
        img_width: [int] Image width in pixels.
        img_height: [int] Image height in pixels.

    Keyword arguments:
        masked_value: [int] Value written where the pixel is masked. 0 by default (RMS mask.bmp
            convention).
        unmasked_value: [int] Value written where the pixel is unmasked. 255 by default.

    Return:
        mask: [ndarray] uint8 (img_height, img_width) mask image.
    """

    # Start fully unmasked
    mask = np.full((img_height, img_width), unmasked_value, dtype=np.uint8)

    # Burn in the polygons
    for polygon in mask_polygons:
        pts = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [pts], masked_value)

    # Composite the brush paint layer on top
    if paint_layer is not None:

        # The layer may come from a mask saved for a different image size (nearest-neighbour keeps the
        #   0/1/2 labels intact)
        if paint_layer.shape != mask.shape:
            paint_layer = cv2.resize(paint_layer, (img_width, img_height), interpolation=cv2.INTER_NEAREST)

        mask[paint_layer == PAINT_MASKED] = masked_value
        mask[paint_layer == PAINT_UNMASKED] = unmasked_value

    return mask



def maskRasterResiduals(mask_img, mask_polygons):
    """ Compute the brush paint layer needed to reproduce a mask image exactly from the given polygons.

        Pixels masked in the image but not covered by any polygon become PAINT_MASKED, pixels unmasked in
        the image but inside a polygon become PAINT_UNMASKED.

    Arguments:
        mask_img: [ndarray] uint8 mask image, 0 = masked, 255 = unmasked.
        mask_polygons: [list] List of polygons, each a list of (x, y) image coordinates.

    Return:
        paint_layer: [ndarray or None] uint8 paint layer, or None if the polygons reproduce the image
            exactly.
    """

    img_height, img_width = mask_img.shape[:2]

    # What the polygons alone would reproduce
    polygon_mask = compositeMaskLayers(mask_polygons, None, img_width, img_height)

    paint_layer = np.zeros((img_height, img_width), dtype=np.uint8)
    paint_layer[(mask_img == 0) & (polygon_mask == 255)] = PAINT_MASKED
    paint_layer[(mask_img == 255) & (polygon_mask == 0)] = PAINT_UNMASKED

    if np.any(paint_layer != PAINT_UNTOUCHED):
        return paint_layer

    return None



def binariseMaskImage(mask_img):
    """ Binarise a mask image the way RMS applies it: only 0 is masked, any other value is unmasked.

    Arguments:
        mask_img: [ndarray] uint8 mask image.

    Return:
        binary: [ndarray] uint8 mask image with only 0 (masked) and 255 (unmasked).
    """

    return np.where(mask_img > 0, 255, 0).astype(np.uint8)



def _simplifyContour(contour, epsilon_frac):
    """ Simplify a contour with approxPolyDP and return its points.

    Arguments:
        contour: [ndarray] OpenCV contour.
        epsilon_frac: [float] approxPolyDP tolerance as a fraction of the contour length.

    Return:
        points: [list] List of (x, y) float points.
    """

    epsilon = epsilon_frac*cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    return [(float(pt[0][0]), float(pt[0][1])) for pt in approx]



def _spliceHole(outer, hole):
    """ Join a hole to its outer boundary with a zero-width bridge, giving one keyhole polygon.

        The bridge connects the closest pair of vertices. Both lie on masked pixels (the contours trace
        the masked pixels on either side of the masked ring), so filling the keyhole polygon masks the
        ring and leaves the hole unmasked.

    Arguments:
        outer: [list] (x, y) points of the outer boundary (possibly already holding spliced holes).
        hole: [list] (x, y) points of the hole boundary.

    Return:
        polygon: [list] (x, y) points of the keyhole polygon.
    """

    outer_arr = np.array(outer)
    hole_arr = np.array(hole)

    # Closest pair of vertices
    dist = np.hypot(outer_arr[:, None, 0] - hole_arr[None, :, 0], outer_arr[:, None, 1] - hole_arr[None, :, 1])
    i, j = np.unravel_index(np.argmin(dist), dist.shape)

    # Walk the outer boundary to vertex i, go around the hole starting and ending at vertex j, come back
    #   to vertex i and continue along the outer boundary
    hole_loop = list(hole[j:]) + list(hole[:j]) + [hole[j]]

    return list(outer[:i + 1]) + hole_loop + list(outer[i:])



def decomposeMaskImage(mask_img, epsilon_frac=0.002):
    """ Split a mask image into editable polygons plus a paint layer holding the raster residuals.

        Masked regions are traced as contours and simplified with approxPolyDP. Unmasked holes inside a
        masked region (e.g. the sky disc of an all-sky mask, whose masked ring covers the frame edge) are
        joined to their outer boundary as keyhole polygons, so the polygons alone describe the mask and
        the hole is not left to a frame-sized erase layer. Whatever the simplified polygons don't
        reproduce (brush strokes, boundary pixels rounded away) is captured in the paint layer so the
        mask survives a save/load round trip without pixel loss.

        The image is binarised the way RMS applies masks first: only 0 is masked.

    Arguments:
        mask_img: [ndarray] uint8 mask image, 0 = masked, 255 = unmasked.

    Keyword arguments:
        epsilon_frac: [float] approxPolyDP tolerance as a fraction of the contour length. 0.002 by
            default.

    Return:
        (polygons, paint_layer): [tuple]
            polygons: [list] List of polygons, each a list of (x, y) float image coordinates.
            paint_layer: [ndarray or None] Raster residuals, None if the polygons reproduce the image.
    """

    mask_img = binariseMaskImage(mask_img)

    # Trace the masked (value 0) regions with their holes (two-level hierarchy: the outer boundaries of
    #   the masked regions, and the boundaries of the unmasked holes in them)
    inverted = cv2.bitwise_not(mask_img)
    contours, hierarchy = cv2.findContours(inverted, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    if hierarchy is not None:

        hierarchy = hierarchy[0]

        for idx, contour in enumerate(contours):

            # Only start from the outer boundaries, holes have a parent
            if hierarchy[idx][3] >= 0:
                continue

            points = _simplifyContour(contour, epsilon_frac)
            if len(points) < 3:
                continue

            # Join every hole of this region into the polygon
            child = hierarchy[idx][2]
            while child >= 0:

                hole = _simplifyContour(contours[child], epsilon_frac)
                if len(hole) >= 3:
                    points = _spliceHole(points, hole)

                child = hierarchy[child][0]

            polygons.append(points)

    return polygons, maskRasterResiduals(mask_img, polygons)



def paintBrushSegment(paint_layer, prev_pos, pos, radius, value):
    """ Stamp one brush step onto the paint layer, in place.

        A disc of the brush radius is drawn at the current position, and when there is a previous
        position the gap is filled with a line of width 2*radius so fast drags leave no holes. The
        footprint is thus the same disc along the whole stroke, including at its first point.

    Arguments:
        paint_layer: [ndarray] uint8 (height, width) paint layer, modified in place.
        prev_pos: [tuple or None] (x, y) integer position of the previous step, None at the start of a
            stroke.
        pos: [tuple] (x, y) position of this step, clamped to the layer.
        radius: [int] Brush radius in pixels (at least 1).
        value: [int] PAINT_MASKED or PAINT_UNMASKED.

    Return:
        center: [tuple] The clamped integer (x, y) position, to pass as prev_pos of the next step.
    """

    img_height, img_width = paint_layer.shape[:2]

    radius = max(int(radius), 1)

    # Clamp the brush centre to the image so a cursor dragged off the image doesn't produce huge
    #   coordinates (the disc may still partly overhang the edge, OpenCV clips the drawing)
    center = (int(min(max(round(pos[0]), 0), img_width - 1)), int(min(max(round(pos[1]), 0), img_height - 1)))

    # Fill the gap from the previous position
    if prev_pos is not None:
        cv2.line(paint_layer, tuple(prev_pos), center, value, thickness=2*radius)

    # Always stamp the disc at the current position (the thick line caps are not guaranteed to match it)
    cv2.circle(paint_layer, center, radius, value, -1)

    return center
