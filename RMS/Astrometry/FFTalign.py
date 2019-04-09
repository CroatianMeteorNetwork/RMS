""" Using FFT image registration to find larger offsets between the platepar and the image. """

from __future__ import print_function, division, absolute_import

try:
    import imreg_dft
    IMREG_INSTALLED = True

except ImportError:
    IMREG_INSTALLED = False


import numpy as np
import matplotlib.pyplot as plt

from RMS.Math import rotatePoint



def addPoint(img, xc, yc, radius):
    """ Add a point to the image. """

    img_w = img.shape[1]
    img_h = img.shape[0]

    sigma, mu = 1.0, 0.0

    # Generate a small array with a gaussian
    grid_arr = np.linspace(-radius, radius, 2*radius + 1, dtype=np.int)
    x, y = np.meshgrid(grid_arr, grid_arr)
    d = np.sqrt(x**2 + y**2)
    gauss = 255*np.exp(-((d - mu)**2/(2.0*sigma**2)))

    # Overlay the Gaussian on the image
    for xi, i in enumerate(grid_arr):
        for yj, j in enumerate(grid_arr):

            # Compute the coordinates of the point
            xp = int(i + xc)
            yp = int(j + yc)

            # Check that the point is inside the image
            if (xp >=0) and (xp < img_w) and (yp >= 0) and (yp < img_h):

                # Set the value of the gaussian to the image
                img[yp, xp] = max(gauss[yj, xi], img[yp, xp])


    return img




def constructImage(img_size, point_list, dot_radius):
    """ Construct the image that will be fed into the FFT registration algorithm. """

    # Construct images using given star positions. Map coordinates to img_size x img_size image
    img = np.zeros((img_size, img_size), dtype=np.uint8)

    # Add all given points to the imge
    for point in point_list:
        x, y = point
        img = addPoint(img, x, y, dot_radius)

    return img



def findStarsTransform(config, reference_list, moved_list, img_size=128, dot_radius=2):
    """ Given a list of reference and predicted star positions, return a transform (rotation, scale, \
        translation) between the two lists using FFT image registration. This is achieved by creating a 
        synthetic star image using both lists and searching for the transform using phase correlation.
    
    Arguments:
        config: [Config instance]
        reference_list: [2D list] A list of reference (x, y) star coordinates.
        moved_list: [2D list] A list of moved (x, y) star coordinates.

    Keyword arguments:
        img_size: [int] Power of 2 image size (e.g. 128, 256, etc.) which will be created and fed into the
            FFT registration algorithm.
        dot_radius: [int] The radius of the dot which will be drawn on the synthetic image.

    Return:
        angle, scale, translation_x, translation_y:
            - angle: [float] Angle of rotation (deg).
            - scale: [float] Image scale difference.
            - translation_x: [float]
            - translation_y: [float]
    """

    # If the image registration library is not installed, return nothing
    if not IMREG_INSTALLED:
        print('The imreg_dft library is not installed! Install it by running either:')
        print(' a) pip install imreg_dft')
        print(' b) conda install -c conda-forge imreg_dft')

        return 0.0, 1.0, 0.0, 0.0


    # Set input types
    reference_list = np.array(reference_list).astype(np.float)
    moved_list = np.array(moved_list).astype(np.float)


    # Rescale the image coordinates 2x, so the image fits inside the smaller window
    rescale_factor = 8

    reference_list /= rescale_factor
    moved_list /= rescale_factor

    # Take only those coordinates which are inside img_size/2 distance from the centre, and
    #   shift the coordinates
    shift_x = img_size/2 - config.width/(2*rescale_factor)
    shift_y = img_size/2 - config.height/(2*rescale_factor)

    reference_list[:, 0] += shift_x
    reference_list[:, 1] += shift_y
    moved_list[:, 0] += shift_x
    moved_list[:, 1] += shift_y

    # Construct the reference and moved images
    img_ref = constructImage(img_size, reference_list, dot_radius)
    img_mov = constructImage(img_size, moved_list, dot_radius)

    # plt.imshow(img_ref, cmap='gray')
    # plt.show()

    # plt.imshow(img_mov, cmap='gray')
    # plt.show()

    
    # Run the FFT registration
    res = imreg_dft.imreg.similarity(img_ref, img_mov)

    angle = res['angle']
    scale = res['scale']
    translate = res['tvec']

    # Extract translation and rescale it
    translation_x = rescale_factor*translate[1]
    translation_y = rescale_factor*translate[0]

    return angle, scale, translation_x, translation_y




if __name__ == "__main__":

    class Config(object):
        def __init__(self):
            self.width = 1280
            self.height = 720



    config = Config()


    # Generate some random points as stars
    npoints = 100
    reference_list = np.c_[np.random.randint(-100, config.width + 100, size=npoints), 
                           np.random.randint(-100, config.height + 100, size=npoints)]

    
    # Create a list of shifted stars
    moved_list = np.copy(reference_list)

    # Move the dots
    moved_list[:, 0] += 50
    moved_list[:, 1] += 40

    # Rotate the moved points
    rot_angle = np.radians(25)
    origin = np.array([config.width/2, config.height/2])
    for i, mv_pt in enumerate(moved_list):
        moved_list[i] = rotatePoint(origin, mv_pt, rot_angle)

    # Choose every second star on the moved list
    moved_list = moved_list[::2]


    # Find the transform between the star lists
    print(findStarsTransform(config, reference_list, moved_list, img_size=128, dot_radius=2))