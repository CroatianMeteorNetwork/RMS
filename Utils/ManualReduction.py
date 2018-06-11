"""Tool for manual astrometry and photometry of FF and FR files. """

from __future__ import print_function, division, absolute_import

import os
import sys
import argparse
import math

import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import RMS.ConfigReader as cr
from RMS.Formats.FFfile import read as readFF
from RMS.Formats.FFfile import validFFName
from RMS.Formats.FRbin import read as readFR
from RMS.Formats.FTPdetectinfo import writeFTPdetectinfo
from RMS.Routines import Image



class Pick(object):
    def __init__(self):
        """ Container for picks per every frame. """

        self.frame = None
        self.x_centroid = None
        self.y_centroid = None

        self.photometry_pixels = None
        self.intensity_sum = None



class FireballPickTool(object):
    def __init__(self, config, ff_file, fr_file, first_frame=None):
        """ Tool for manually picking fireball centroids and photometry. """


        self.config = config

        self.ff_file = ff_file
        self.fr_file = fr_file

        # Load the FF file if given
        if self.ff_file is not None:
            self.ff = readFF(*os.path.split(self.ff_file))
        else:
            self.ff = None


        # Load the FR file is given
        if self.fr_file is not None:
            self.fr = readFR(*os.path.split(self.fr_file))
        else:
            self.fr = None


        ###########

        self.img_gamma = 1.0
        self.show_key_help = True

        self.current_image = None

        self.fr_xmin = None
        self.fr_xmax = None
        self.fr_ymin = None
        self.fr_ymax = None

        # Previous zoom
        self.prev_xlim = None
        self.prev_ylim = None

        self.circle_aperture = None
        self.circle_aperture_outer = None
        self.aperture_radius = 5
        self.scroll_counter = 0

        self.centroid_handle = None

        self.photometry_coloring_mode = False
        self.photometry_coloring_color = False
        self.photometry_aperture_radius = 3
        self.photometry_add = True
        self.photometry_coloring_handle = None

        self.pick_list = []

        ###########


        # Each FR bin can have multiple detections, the first one is by default
        self.current_line = 0

        if first_frame is not None:
            self.current_frame = first_frame%256

        else:

            # Set the first frame to the first frame in FR, if given
            if self.fr is not None:
                self.current_frame = self.fr.t[self.current_line][0]

            else:
                self.current_frame = 0



        ### INIT IMAGE ###

        plt.figure(facecolor='black')

        # Init the first image
        self.updateImage()

        self.ax = plt.gca()

        # Set the bacground color to black
        #matplotlib.rcParams['axes.facecolor'] = 'k'

        # Disable standard matplotlib keyboard shortcuts
        plt.rcParams['keymap.save'] = ''
        plt.rcParams['keymap.fullscreen'] = ''
        plt.rcParams['keymap.all_axes'] = ''
        plt.rcParams['keymap.quit'] = ''


        # Register event handlers
        self.ax.figure.canvas.mpl_connect('key_press_event', self.onKeyPress)
        self.ax.figure.canvas.mpl_connect('key_release_event', self.onKeyRelease)
        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.onMouseMotion)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.onMousePress)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.onMouseRelease)
        self.ax.figure.canvas.mpl_connect('scroll_event', self.onScroll)


    def printStatus(self):
        """ Print the status message. """

        print('----------------------------')
        print('Frame:', self.current_frame)
        print('Line:', self.current_line)


    def updateImage(self):
        """ Updates the current plot. """

        # Reset circle patches
        self.circle_aperature = None
        self.circle_aperature_outer = None

        # Reset centroid patch
        self.centroid_handle = None

        # Reset photometry coloring
        self.photometry_coloring_handle = None

        # Save the previous zoom
        if self.current_image is not None:
            
            self.prev_xlim = plt.gca().get_xlim()
            self.prev_ylim = plt.gca().get_ylim()


        plt.clf()
            

        # If FF is given, reconstruct frames
        if self.ff is not None:

            # Take the current frame from FF file
            img = np.copy(self.ff.avepixel)
            frame_mask = np.where(self.ff.maxframe == self.current_frame)
            img[frame_mask] = self.ff.maxpixel[frame_mask]

        # Otherwise, create a blank background with the size enough to fit the FR bin
        else:

            # Get the maximum extent of the meteor frames
            y_size = max(max(np.array(self.fr.yc[i]) + np.array(self.fr.size[i])//2) for i in \
                range(self.fr.lines))
            x_size = max(max(np.array(self.fr.xc[i]) + np.array(self.fr.size[i])//2) for i in \
                range(self.fr.lines))

            # Make the image square
            img_size = max(y_size, x_size)

            img = np.zeros((img_size, img_size), np.uint8)


        # If FR is given, paste the raw frame onto the image
        if self.fr is not None:

            # Compute the index of the frame in the FR bin structure
            frame_indx = self.current_frame - self.fr.t[self.current_line][0]

            # Reconstruct the frame if it is within the bounds
            if frame_indx < self.fr.frameNum[self.current_line]:

                # Get the center position of the detection on the current frame
                yc = self.fr.yc[self.current_line][frame_indx]
                xc = self.fr.xc[self.current_line][frame_indx]

                # # Get the frame number
                # t = self.fr.t[self.current_line][frame_indx]

                # Get the size of the window
                size = self.fr.size[self.current_line][frame_indx]


                # Paste the frames onto the big image
                y_img = np.arange(yc - size//2, yc + size//2)
                x_img = np.arange(xc - size//2,  xc + size//2)

                Y_img, X_img = np.meshgrid(y_img, x_img)

                y_frame = np.arange(len(y_img))
                x_frame = np.arange(len(x_img))

                Y_frame, X_frame = np.meshgrid(y_frame, x_frame)                

                img[Y_img, X_img] = self.fr.frames[self.current_line][frame_indx][Y_frame, X_frame]

                # Save the limits of the FR
                self.fr_xmin = np.min(x_img)
                self.fr_xmax = np.max(x_img)
                self.fr_ymin = np.max(y_img)
                self.fr_ymax = np.min(y_img)

                # Draw a red rectangle around the pasted frame
                rect_x = np.min(x_img)
                rect_y = np.max(y_img)
                rect_w = np.max(x_img) - rect_x
                rect_h = np.min(y_img) - rect_y
                plt.gca().add_patch(mpatches.Rectangle((rect_x, rect_y), rect_w, rect_h, fill=None, edgecolor='red', alpha=0.5))



        # Current image without adjustments
        self.current_image = np.copy(img)


        # Adjust image levels
        img = Image.adjustLevels(img, 0, self.img_gamma, (2**self.config.bit_depth - 1), 
            self.config.bit_depth)

        plt.imshow(img, cmap='gray', vmin=0, vmax=255)

        self.drawText()

        if (self.prev_xlim is not None) and (self.prev_ylim is not None):

            # Restore previous zoom
            plt.xlim(self.prev_xlim)
            plt.ylim(self.prev_ylim)


        # Don't draw the picks in the photometry coloring more
        if not self.photometry_coloring_mode:
            
            # Plot image pick
            self.drawPicks(update_plot=False)


        # Plot the photometry coloring
        self.drawPhotometryColoring(update_plot=False)

        plt.gcf().canvas.draw()



    def drawText(self):
        """ Draw the text on the image. """

        if self.show_key_help:

            # Draw info text

            text_str  = "Frame: {:d}\n".format(self.current_frame)
            text_str += "Gamma: {:.2f}\n".format(self.img_gamma)

            plt.gca().text(10, 10, text_str, color='w', verticalalignment='top', horizontalalignment='left', \
                fontsize=8)

            # Show text on image with instructions
            text_str  = 'Keys:\n'
            text_str += '-----------\n'
            text_str += 'Previous/next frame - Arrow keys left/right\n'
            text_str += 'Previous/next FR line - ,/.\n'
            text_str += 'Img Gamma - U/J\n'
            text_str += 'Zoom in/out - +/-\n'
            text_str += 'Reset view - R\n'
            text_str += 'Save current frame - CTRL + W\n'
            text_str += 'Save FTPdetectinfo - CTRL + S\n'
            text_str += '\n'
            text_str += 'Mouse:\n'
            text_str += '-----------\n'
            text_str += 'Centroid - Left click\n'
            text_str += 'Manual pick - CTRL + Left click\n'
            text_str += 'Photometry coloring add - Shift + Left click'
            text_str += 'Photometry coloring remove - Shift + Right click'
            text_str += '\n'
            text_str += 'Hide/show text - F1'


            plt.gca().text(10, self.current_image.shape[0] - 5, text_str, color='w', 
                verticalalignment='bottom', horizontalalignment='left', fontsize=8)

        else:

            text_str = "Show text - F1"

            plt.gca().text(self.current_image.shape[1]/2, self.current_image.shape[0], text_str, color='w', 
                verticalalignment='top', horizontalalignment='center', fontsize=8, alpha=0.5)



    def onKeyPress(self, event):
        """ Handles key presses. """

        # Cycle frames
        if event.key == 'left':
            self.prevFrame()

        elif event.key == 'right':
            self.nextFrame()


        # Zoom in/out
        elif event.key == '+':

            self.zoomImage(zoom_in=True)

            self.updateImage()

        elif event.key == '-':
            
            self.zoomImage(zoom_in=False)            

            self.updateImage()


        # Previous line
        elif event.key == ',':

            if self.fr is not None:
                
                self.current_line -= 0

                if self.current_line < 0:
                    self.current_line = 0

                self.printStatus()


        elif event.key == '.':

            if self.fr is not None:

                self.current_line += 0

                if self.current_line >= self.fr.lines:
                    self.current_line = self.fr.lines - 1

                self.printStatus()





        # Show/hide keyboard shortcut help
        elif event.key == 'f1':
            self.show_key_help = not self.show_key_help
            self.updateImage()


        # Increase image gamma
        elif event.key == 'u':

            # Increase image gamma by a factor of 1.1x
            self.updateGamma(1.1)

        elif event.key == 'j':

            # Decrease image gamma by a factor of 0.9x
            self.updateGamma(0.9)


        elif event.key == 'r':

            # Reset the plot limits
            plt.xlim(0, self.current_image.shape[1])
            plt.ylim(self.current_image.shape[0], 0)

            self.updateImage()


        elif event.key == 'ctrl+w':

            # Save current frame to disk as image
            self.saveCurrentFrame()


        elif event.key == 'ctrl+s':

            # Save the FTPdetectinfo file
            self.saveFTPdetectinfo()


        elif event.key == 'shift':

            # Toggle the photometry coloring mode
            if not self.photometry_coloring_mode:
                
                self.photometry_coloring_mode = True
                self.updateImage()



    def onKeyRelease(self, event):
        """ Handles key releases. """

        if event.key == 'shift':

            # Toggle the photometry coloring mode
            if self.photometry_coloring_mode:
                
                self.photometry_coloring_mode = False

                # Redraw the centroids
                self.drawPicks()



    def onMouseMotion(self, event):
        """ Called with the mouse is moved. """

        # Check if the mouse is within bounds
        if (event.xdata is not None) and (event.ydata is not None):
            
            # Read mouse position
            self.mouse_x = event.xdata
            self.mouse_y = event.ydata

            # Change the position of the star aperture circle
            self.drawCursorCircle()

            if self.photometry_coloring_mode and self.photometry_coloring_color:
                self.changePhotometry(self.current_frame, self.photometryColoring(), add_photometry=self.photometry_add)

                self.drawPhotometryColoring(update_plot=True)



    def onScroll(self, event):
        """ Change aperture on scroll. """

        self.scroll_counter += event.step


        if self.scroll_counter > 1:
            
            if self.photometry_coloring_mode:
                self.photometry_aperture_radius += 1
            else:
                self.aperture_radius += 1

            self.scroll_counter = 0

        elif self.scroll_counter < -1:

            if self.photometry_coloring_mode:
                self.photometry_aperture_radius -= 1
            else:
                self.aperture_radius -= 1

            self.scroll_counter = 0


        # Check that the centroid aperture is in the proper limits
        if self.aperture_radius < 2:
            self.aperture_radius = 2

        if self.aperture_radius > 50:
            self.aperture_radius = 50

        # Check that the photometry aperture is in the proper limits
        if self.photometry_aperture_radius < 2:
            self.photometry_aperture_radius = 2

        if self.photometry_aperture_radius > 50:
            self.photometry_aperture_radius = 50

        
        self.drawCursorCircle()



    def onMouseRelease(self, event):
        """ Called when the mouse click is released. """

        # Call the same function for mouse movements to update the variables in the background
        self.onMouseMotion(event)


        # Left mouse button, centroid or CTRL is not pressed
        if (event.button == 1) and (event.key != 'shift'):

            # Remove the old centroid scatter plot handle
            if self.centroid_handle is not None:
                self.centroid_handle.remove()

            # Centroid the star around the pressed coordinates
            self.x_centroid, self.y_centroid, _ = self.centroid()

            # If CTRL is pressed, place the pick manually - NOTE: the intensity might be off then!!!
            if event.key == 'control':
                self.x_centroid = self.mouse_x
                self.y_centroid = self.mouse_y


            # Add the centroid to the list
            self.addCentroid(self.current_frame, self.x_centroid, self.y_centroid)

            self.updateImage()


        # Remove centroid on right click
        if (event.button == 3) and (event.key != 'shift'):
            self.removeCentroid(self.current_frame)

            self.updateImage()


        # Photometry coloring - off
        if ((event.button == 1) or (event.button == 3)) and (event.key == 'shift'):
            self.photometry_coloring_color = False



    def onMousePress(self, event):
        """ Called on mouse click press. """

        # Photometry coloring - on
        if ((event.button == 1) or (event.button == 3)) and (event.key == 'shift'):
            self.photometry_coloring_color = True

            # Color photometry pixels
            if (event.button == 1):
                self.photometry_add = True

            # Remove pixels
            else:
                self.photometry_add = False


    def zoomImage(self, zoom_in):
        """ Change the zoom if the image. """

        zoom_factor = 2.0/3

        # Get the current limits of the plot
        xmin, xmax = plt.gca().get_xlim()
        ymin, ymax = plt.gca().get_ylim()

        x_img = self.current_image.shape[1]
        y_img = self.current_image.shape[0]


        # Compute maximum length of side differences
        max_size = max(abs(xmax - xmin), abs(ymin - ymax))

        # Zoom in
        if zoom_in:

            # Compute the new size
            xsize = zoom_factor*max_size
            ysize = zoom_factor*max_size


        else:

            # Compute the new size
            xsize = (1.0/zoom_factor)*max_size
            ysize = (1.0/zoom_factor)*max_size


        # Compute the new limits
        xmin = self.mouse_x - xsize/2
        xmax = self.mouse_x + xsize/2
        ymin = self.mouse_y + ysize/2
        ymax = self.mouse_y - ysize/2


        # Check if the new bounds are within the limits
        if xmin < 0:
            xmin = 0
        elif xmin > x_img:
            xmin = x_img

        if xmax < 0:
            xmax = 0
        elif xmax > x_img:
            xmax = x_img


        if ymin < 0:
            ymin = 0
        elif ymin > y_img:
            ymin = y_img

        if ymax < 0:
            ymax = 0
        elif ymax > y_img:
            ymax = y_img

        # Set the new limits
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)


    def photometryColoring(self):
        """ Color pixels for photometry. """

        pixel_list = []

        mouse_x = int(self.mouse_x)
        mouse_y = int(self.mouse_y)
            
        ### Add all pixels within the aperture to the list for photometry ###

        x_list = range(mouse_x - self.photometry_aperture_radius, mouse_x \
            + self.photometry_aperture_radius + 1)
        y_list = range(mouse_y - self.photometry_aperture_radius, mouse_y \
            + self.photometry_aperture_radius + 1)

        for x in x_list:
            for y in y_list:

                # Skip pixels ourside the image
                if (x < 0) or (x > self.current_image.shape[1]):
                    continue

                if (y < 0) or (y > self.current_image.shape[0]):
                    continue

                # Check if the given pixels are within the aperture radius
                if math.sqrt((x - mouse_x)**2 + (y - mouse_y)**2) <= self.photometry_aperture_radius:
                    pixel_list.append((x, y))


        ##########

        return pixel_list


    def computeIntensitySum(self):
        """ Compute the background subtracted sum of intensity of colored pixels. The background is estimated
            as the median of near pixels that are not colored. 
        """

        # Find the pick done on the current frame
        pick_found = [pick for pick in self.pick_list if pick.frame == self.current_frame]

        if pick_found:

            pick = pick_found[0]

            # If there are no photometry pixels, set the intensity to 0
            if not pick.photometry_pixels:
                
                pick.intensity_sum = 0

                return None


            x_arr, y_arr = np.array(pick.photometry_pixels).T

            # Compute the centre of the colored pixels
            x_centre = np.mean(x_arr)
            y_centre = np.mean(y_arr)

            # Take a window twice the size of the colored pixels
            x_color_size = np.max(x_arr) - np.min(x_arr)
            y_color_size = np.max(y_arr) - np.min(y_arr)

            x_min = int(x_centre - x_color_size)
            x_max = int(x_centre + x_color_size)
            y_min = int(y_centre + y_color_size)
            y_max = int(y_centre - y_color_size)

            # If only the FR file is given, make sure the background is only estimated within the FR window
            if (self.fr is not None) and (self.ff is None):
                x_min = max(x_min, self.fr_xmin)
                x_max = min(x_max, self.fr_xmax)
                y_min = min(y_min, self.fr_ymin)
                y_max = max(y_max, self.fr_ymax)

            # Limit the size to be within the bounds
            if x_min < 0: x_min = 0
            if x_max > self.current_image.shape[1]: x_max = self.current_image.shape[1]
            if y_min > self.current_image.shape[0]: y_min = self.current_image.shape[0]
            if y_max < 0: y_max = 0
            

            # Take only the colored part
            mask_img = np.ones_like(self.current_image)
            mask_img[y_arr, x_arr] = 0
            masked_img = np.ma.masked_array(self.current_image, mask_img)
            crop_img = masked_img[y_max:y_min, x_min:x_max]



            # Mask out the colored in pixels
            mask_img_bg = np.zeros_like(self.current_image)
            mask_img_bg[y_arr, x_arr] = 1

            # Take the image where the colored part is masked out and crop the surroundings
            masked_img_bg = np.ma.masked_array(self.current_image, mask_img_bg)
            crop_bg = masked_img_bg[y_max:y_min, x_min:x_max]


            # Compute the median background
            background_lvl = np.ma.median(crop_bg)

            # Compute the background subtracted intensity sum
            pick.intensity_sum = np.ma.sum(crop_img - background_lvl)



    def drawCursorCircle(self):
        """ Adds a circle around the mouse cursor. """

        # Delete the old aperture circle
        if self.circle_aperture is not None:
            try:
                self.circle_aperture.remove()
            except:
                pass
            self.circle_aperture = None

        if self.circle_aperture_outer is not None:
            try:
                self.circle_aperture_outer.remove()
            except:
                pass
            self.circle_aperture_outer = None


        # If photometry coloring is on, show a purple annulus
        if self.photometry_coloring_mode:

            # Plot a circle of the given radius around the cursor
            self.circle_aperture = mpatches.Circle((self.mouse_x, self.mouse_y),
                self.photometry_aperture_radius, edgecolor='red', fc='red', alpha=0.5)

            plt.gca().add_patch(self.circle_aperture)

        else:

            # Plot a circle of the given radius around the cursor
            self.circle_aperture = mpatches.Circle((self.mouse_x, self.mouse_y),
                self.aperture_radius, edgecolor='yellow', fc='none')

            # Plot a circle of the given radius around the cursor, which is used to determine the region where the
            # background will be taken from
            self.circle_aperture_outer = mpatches.Circle((self.mouse_x, self.mouse_y),
                self.aperture_radius*2, edgecolor='yellow', fc='none', linestyle='dotted')

            plt.gca().add_patch(self.circle_aperture)
            plt.gca().add_patch(self.circle_aperture_outer)


        plt.gcf().canvas.draw()



    def drawPicks(self, update_plot=True):
        """ Plot the picks. """

        # Plot all picks
        for pick in self.pick_list:

            plt.scatter(pick.x_centroid, pick.y_centroid, marker='+', c='y', s=20, lw=1)


        # Find the pick done on the current frame
        pick_found = [pick for pick in self.pick_list if pick.frame == self.current_frame]

        # Plot the pick done on this image a bit larger
        if pick_found:

            pick = pick_found[0]

            # Draw the centroid on the image
            self.centroid_handle = plt.scatter(pick.x_centroid, pick.y_centroid, marker='+', c='y', s=100, 
                lw=2)


            # Update canvas
            if update_plot:
                plt.gcf().canvas.draw()


    def drawPhotometryColoring(self, update_plot=True):
        """ Color pixels which will be used for photometry. """

        # Remove old photometry coloring
        if self.photometry_coloring_handle is not None:
            try:
                self.photometry_coloring_handle.remove()
            except:
                pass

        # Find the pick done on the current frame
        pick_found = [pick for pick in self.pick_list if pick.frame == self.current_frame]

        if pick_found:

            pick = pick_found[0]

            if pick.photometry_pixels is not None:

                # Create a coloring mask
                x_mask, y_mask = np.array(pick.photometry_pixels).T

                mask_img = np.zeros_like(self.current_image)
                mask_img[y_mask, x_mask] = 1

                # Create a RGB overlay
                mask_overlay = np.zeros((self.current_image.shape[0], self.current_image.shape[1], 4))
                mask_overlay[..., 1] = 1 # Green channel
                mask_overlay[..., 3] = 0.3*mask_img # Alpha channel

                self.photometry_coloring_handle = plt.imshow(mask_overlay)

                # Update canvas
                if update_plot:
                    plt.gcf().canvas.draw()



    def addCentroid(self, frame, x_centroid, y_centroid):
        """ Add the centroid to the list of centroids. """

        print('Added centroid at ({:.2f}, {:.2f}) on frame {:d}'.format(x_centroid, y_centroid, frame))

        # Check if there are previous picks on this frame
        prev_pick = [i for i, pick in enumerate(self.pick_list) if pick.frame == frame]
            
        # Update centroids of previous pick if it exists
        if prev_pick:

            i = prev_pick[0]
            
            self.pick_list[i].x_centroid = x_centroid
            self.pick_list[i].y_centroid = y_centroid

        # Create a new pick
        else:

            pick = Pick()

            pick.frame = frame
            pick.x_centroid = x_centroid
            pick.y_centroid = y_centroid

            self.pick_list.append(pick)


    def removeCentroid(self, frame):
        """ Remove the centroid from the list of centroids. """

        # Check if there are previous picks on this frame
        prev_pick = [i for i, pick in enumerate(self.pick_list) if pick.frame == frame]

        if prev_pick:

            i = prev_pick[0]

            pick = self.pick_list[i]

            print('Removed centroid at ({:.2f}, {:.2f}) on frame {:d}'.format(pick.x_centroid, \
                pick.y_centroid, frame))

            # Remove the centroid
            self.pick_list.pop(i)


    def changePhotometry(self, frame, photometry_pixels, add_photometry):
        """ Add/remove photometry pixels of the pick. """

        # Check if there are previous picks on this frame
        prev_pick = [i for i, pick in enumerate(self.pick_list) if pick.frame == frame]
            
        # Update centroids of previous pick if it exists
        if prev_pick:

            i = prev_pick[0]

            # If there's no previous photometry, add an empty list
            if self.pick_list[i].photometry_pixels is None:
                self.pick_list[i].photometry_pixels = []

            if add_photometry:
                
                # Add the photometry pixels to the pick
                self.pick_list[i].photometry_pixels = list(set(self.pick_list[i].photometry_pixels \
                    + photometry_pixels))

            else:

                # Remove the photometry pixels to the pick
                self.pick_list[i].photometry_pixels = [px for px in self.pick_list[i].photometry_pixels if px not in photometry_pixels]


        # Add a new pick
        elif add_photometry:

            pick = Pick()

            pick.frame = frame
            pick.photometry_pixels = photometry_pixels

            self.pick_list.append(pick)



    def centroid(self):
        """ Find the centroid of the object clicked on the image. """

        ### Extract part of image around the mouse cursor ###
        ######################################################################################################

        # Outer circle radius
        outer_radius = self.aperture_radius*2

        x_min = int(round(self.mouse_x - outer_radius))
        if x_min < 0: x_min = 0

        x_max = int(round(self.mouse_x + outer_radius))
        if x_max > self.current_image.shape[1] - 1:
            x_max > self.current_image.shape[1] - 1

        y_min = int(round(self.mouse_y - outer_radius))
        if y_min < 0: y_min = 0

        y_max = int(round(self.mouse_y + outer_radius))
        if y_max > self.current_image.shape[0] - 1:
            y_max > self.current_image.shape[0] - 1

        img_crop = self.current_image[y_min:y_max, x_min:x_max]

        ######################################################################################################


        ### Estimate the background ###
        ######################################################################################################
        bg_acc = 0
        bg_counter = 0
        for i in range(img_crop.shape[0]):
            for j in range(img_crop.shape[1]):

                # Calculate distance of pixel from centre of the cropped image
                i_rel = i - img_crop.shape[0]/2
                j_rel = j - img_crop.shape[1]/2
                pix_dist = math.sqrt(i_rel**2 + j_rel**2)

                # Take only those pixels between the inner and the outer circle
                if (pix_dist <= outer_radius) and (pix_dist > self.aperture_radius):
                    bg_acc += img_crop[i, j]
                    bg_counter += 1

        # Calculate mean background intensity
        bg_intensity = bg_acc/bg_counter

        ######################################################################################################


        ### Calculate the centroid ###
        ######################################################################################################
        x_acc = 0
        y_acc = 0
        intens_acc = 0

        for i in range(img_crop.shape[0]):
            for j in range(img_crop.shape[1]):

                # Calculate distance of pixel from centre of the cropped image
                i_rel = i - img_crop.shape[0]/2
                j_rel = j - img_crop.shape[1]/2
                pix_dist = math.sqrt(i_rel**2 + j_rel**2)

                # Take only those pixels between the inner and the outer circle
                if pix_dist <= self.aperture_radius:
                    x_acc += j*(img_crop[i, j] - bg_intensity)
                    y_acc += i*(img_crop[i, j] - bg_intensity)
                    intens_acc += img_crop[i, j] - bg_intensity


        x_centroid = x_acc/intens_acc + x_min
        y_centroid = y_acc/intens_acc + y_min

        ######################################################################################################

        return x_centroid, y_centroid, intens_acc


    def updateGamma(self, gamma_adj_factor):
        """ Change the image gamma by a given factor. """

        self.img_gamma *= gamma_adj_factor

        # Make sure gamma is in the proper range
        if self.img_gamma < 0.1: self.img_gamma = 0.1
        if self.img_gamma > 10: self.img_gamma = 10

        self.updateImage()



    def prevFrame(self):
        """ Cycle to the previous frame. """

        # Compute the intensity sum done on the previous frame
        self.computeIntensitySum()

        self.current_frame = (self.current_frame - 1)%256

        self.printStatus()

        self.updateImage()



    def nextFrame(self):
        """ Cycle to the next frame. """

        # Compute the intensity sum done on the previous frame
        self.computeIntensitySum()

        self.current_frame = (self.current_frame + 1)%256

        self.printStatus()

        self.updateImage()



    def saveFTPdetectinfo(self):
        """ Saves the picks to a FTPdetectinfo file in the same folder where the first given file is. """

        # Compute the intensity sum done on the previous frame
        self.computeIntensitySum()

        # Extract the save directory
        if self.ff_file is not None:
            dir_path, ff_name_ftp = os.path.split(self.ff_file)

        elif self.fr_file is not None:
            dir_path, ff_name_ftp = os.path.split(self.fr_file)


        # Remove the file extension of the image file
        ff_name_ftp = ff_name_ftp.replace('.bin', '').replace('.fits', '')

        # Create the list of picks for saving
        centroids = []
        for pick in self.pick_list:
            centroids.append([pick.frame, pick.x_centroid, pick.y_centroid, pick.intensity_sum])

        meteor_list = [[ff_name_ftp, 1, 0, 0, centroids]]

        # Create a name for the FTPdetectinfo
        ftpdetectinfo_name = "FTPdetectinfo_" + "_".join(ff_name_ftp.split('_')[1:]) + '_manual.txt'

        # Read the station code for the file name
        station_id = ff_name_ftp.split('_')[1]

        # Take the FPS from the FF file, if available
        fps = None
        if self.ff is not None:
            if hasattr(self.ff, 'fps'):
                fps = self.ff.fps

        if fps is None:
            fps = self.config.fps

        # Write the FTPdetect info
        writeFTPdetectinfo(meteor_list, dir_path, ftpdetectinfo_name, '', station_id, fps)

        print('FTPdetecinfo written to:', os.path.join(dir_path, ftpdetectinfo_name))



    def saveCurrentFrame(self):
        """ Saves the current frame to disk. """

        # Extract the save directory
        if self.ff_file is not None:
            dir_path, ff_name_ftp = os.path.split(self.ff_file)

        elif self.fr_file is not None:
            dir_path, ff_name_ftp = os.path.split(self.fr_file)


        # Remove the file extension of the image file
        ff_name_ftp = ff_name_ftp.replace('.bin', '').replace('.fits', '')

        # Construct the file name
        frame_file_name = ff_name_ftp + "_frame_{:03d}".format(self.current_frame) + '.png'
        frame_file_path = os.path.join(dir_path, frame_file_name)

        # Save the frame to disk
        scipy.misc.imsave(frame_file_path, self.current_image)

        print('Frame {:d} saved to: {:s}'.format(self.current_frame, frame_file_path))





if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Tool for manually picking positions of fireballs on video frames.")

    arg_parser.add_argument('file1', metavar='FILE1', type=str, nargs=1,
                    help='Path to an FF file, or an FR file if an FF file is not available.')

    arg_parser.add_argument('file2', metavar='FILE2', type=str, nargs='*',
                    help='If an FF file was given, an FR file can be given in addition.')

    arg_parser.add_argument('-f', '--firstframe', metavar='FIRST FRAME', type=int, help="First frame to show. Has to be between 0-255.")


    #########################

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()


    # Load the configuration file
    config = cr.parse(".config")


    ff_name = None
    fr_name = None


    file1_name = os.path.split(cml_args.file1[0])[-1]

    # Check if the first file is an FF file
    if validFFName(file1_name):

        # This is an FF file
        ff_name = cml_args.file1[0]

    # This is possibly a FR file then
    else:

        if 'FR' in file1_name:
            fr_name = cml_args.file1[0]


    if cml_args.file2 and (ff_name is None):
        print('The given FF file is not a proper FF file!')
        sys.exit()


    # Check if the second file is a good FR file
    if cml_args.file2:

        file2_name = os.path.split(cml_args.file2[0])[-1]

        if 'FR' in file2_name:

            fr_name = cml_args.file2[0]

        else:
            print('The given FR file is not valid!')
            sys.exit()



    # Make sure there is at least one good file given
    if (ff_name is None) and (fr_name is None):
        print('No valid FF or FR files given!')
        sys.exit()


    # Init the fireball picker
    fireball_picker = FireballPickTool(config, ff_name, fr_name, first_frame=cml_args.firstframe)


    plt.tight_layout()
    plt.show()


