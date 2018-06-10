from __future__ import print_function, division, absolute_import

import os
import sys
import argparse
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import RMS.ConfigReader as cr
from RMS.Formats.FFfile import read as readFF
from RMS.Formats.FFfile import validFFName
from RMS.Formats.FRbin import read as readFR
from RMS.Routines import Image



class Pick(object):
    def __init__(self):
        """ Container for picks per every frame. """

        self.frame = None
        self.x_centroid = None
        self.y_centroid = None

        self.photometry_pixels = None



class FireballPickTool(object):
    def __init__(self, config, ff, fr, first_frame=None):
        """ Tool for manually picking fireball centroids and doing photometry. """


        self.config = config

        self.ff = ff
        self.fr = fr


        ###########

        self.img_gamma = 1.0
        self.show_key_help = True

        self.current_image = None

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
        self.photometry_add = True

        self.pick_list = []

        ###########


        # Each FR bin can have multiple detections, the first one is by default
        self.current_line = 0

        if first_frame is not None:
            self.current_frame = first_frame%256

        else:

            # Set the first frame to the first frame in FR, if given
            if fr is not None:
                self.current_frame = self.fr.t[self.current_line][0]

            else:
                self.current_frame = 0


        print(self.current_frame)




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



    def updateImage(self):
        """ Updates the current plot. """

        # Reset circle patches
        self.circle_aperature = None
        self.circle_aperature_outer = None

        # Reset centroid patch
        self.centroid_handle = None


        # Save the previous zoom
        if self.current_image is not None:
            
            self.prev_xlim = plt.gca().get_xlim()
            self.prev_ylim = plt.gca().get_ylim()

        plt.clf()
            

        # If FF is given, reconstruct frames
        if ff is not None:

            # Take the current frame from FF file
            img = np.copy(self.ff.avepixel)
            frame_mask = np.where(self.ff.maxframe == self.current_frame)
            img[frame_mask] = self.ff.maxpixel[frame_mask]

        # Otherwise, create a blank background
        else:
            img = np.zeros((self.config.height, self.config.width), np.uint8)


        # If FR is given, paste the raw frame onto the image
        if fr is not None:

            # Compute the index of the frame in the FR bin structure
            frame_indx = self.current_frame - fr.t[self.current_line][0]

            # Reconstruct the frame if it is within the bounds
            if frame_indx < fr.frameNum[self.current_line]:

                # Get the center position of the detection on the current frame
                yc = fr.yc[self.current_line][frame_indx]
                xc = fr.xc[self.current_line][frame_indx]

                # # Get the frame number
                # t = fr.t[self.current_line][frame_indx]

                # Get the size of the window
                size = fr.size[self.current_line][frame_indx]


                # Paste the frames onto the big image
                y_img = np.arange(yc - size//2, yc + size//2)
                x_img = np.arange(xc - size//2,  xc + size//2)

                Y_img, X_img = np.meshgrid(y_img, x_img)

                y_frame = np.arange(len(y_img))
                x_frame = np.arange(len(x_img))

                Y_frame, X_frame = np.meshgrid(y_frame, x_frame)                

                img[Y_img, X_img] = self.fr.frames[self.current_line][frame_indx][Y_frame, X_frame]


                # Draw a red rectangle around the pasted frame
                rect_x = np.min(x_img)
                rect_y = np.max(y_img)
                rect_w = np.max(x_img) - rect_x
                rect_h = np.min(y_img) - rect_y
                plt.gca().add_patch(mpatches.Rectangle((rect_x, rect_y), rect_w, rect_h, fill=None, edgecolor='red', alpha=0.5))



        # Current image without adjustments
        self.current_image = np.copy(img)


        # Adjust image levels
        img = Image.adjustLevels(img, 0, self.img_gamma, (2**self.config.bit_depth -1), 
            self.config.bit_depth)

        plt.imshow(img, cmap='gray', vmin=0, vmax=255)

        self.drawText()

        if (self.prev_xlim is not None) and (self.prev_ylim is not None):

            print('setting plot zoom to:', self.prev_xlim, self.prev_ylim)
            # Restore previous zoom
            plt.xlim(self.prev_xlim)
            plt.ylim(self.prev_ylim)


        # Plot image pick
        self.drawPick(update_plot=False)

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
            text_str += 'Img Gamma - U/J\n'
            text_str += 'Reset view - R\n'
            text_str += '\n'
            text_str  = 'Mouse:\n'
            text_str += '-----------\n'
            text_str += 'Centroid - Left click\n'
            text_str += 'Manual pick - CTRL + Left click\n'
            text_str += 'Photometry coloring - Shift + Left click'
            text_str += '\n'
            text_str += 'Hide/show text - F1\n'


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


        elif event.key == 'shift':
            self.photometry_coloring_mode = True



    def onKeyRelease(self, event):
        """ Handles key releases. """

        if event.key == 'shift':
            self.photometry_coloring_mode = False



    def onMouseMotion(self, event):
        """ Called with the mouse is moved. """


        # Read mouse position
        self.mouse_x = event.xdata
        self.mouse_y = event.ydata

        # Change the position of the star aperture circle
        self.drawCursorCircle()

        if self.photometry_coloring_mode:
            self.photometryColoring()


    def onScroll(self, event):
        """ Change aperture on scroll. """

        self.scroll_counter += event.step


        if self.scroll_counter > 1:
            self.aperture_radius += 1
            self.scroll_counter = 0

        elif self.scroll_counter < -1:
            self.aperture_radius -= 1
            self.scroll_counter = 0


        # Check that the star aperture is in the proper limits
        if self.aperture_radius < 2:
            self.aperture_radius = 2

        if self.aperture_radius > 50:
            self.aperture_radius = 50

        # Change the size of the star aperture circle
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

            self.drawPick()


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


    def photometryColoring(self):
        """ Color pixels for photometry. """

        pixel_list = []

        mouse_x = int(self.mouse_x)
        mouse_y = int(self.mouse_y)

        if self.photometry_coloring_color:
            
            ### Add all pixels within the aperture to the list for photometry ###

            x_list = range(mouse_x - self.aperture_radius, mouse_x + self.aperture_radius + 1)
            y_list = range(mouse_y - self.aperture_radius, mouse_y + self.aperture_radius + 1)

            for x in x_list:
                for y in y_list:

                    # Skip pixels ourside the image
                    if (x < 0) or (x > self.current_image.shape[1]):
                        continue

                    if (y < 0) or (y > self.current_image.shape[0]):
                        continue

                    # Check if the given pixels are within the aperture radius
                    if math.sqrt((x - mouse_x)**2 + (y - mouse_y)**2) <= self.aperture_radius:
                        pixel_list.append((x, y))


            ##########

        return pixel_list



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
                self.aperture_radius, edgecolor='red', fc='red', alpha=0.5)

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



    def drawPick(self, update_plot=True):
        """ Plot the pick done on the current frame. """



        # Plot all picks
        for pick in self.pick_list:

            plt.scatter(pick.x_centroid, pick.y_centroid, marker='+', c='y', s=20, lw=1)


        # Find the pick done on this frame
        pick_found = [(i, pick) for i, pick in enumerate(self.pick_list) if pick.frame == self.current_frame]

        # Plot the pick done on this image a bit larger
        if pick_found:

            i, pick = pick_found[0]

            # Draw the centroid on the image
            self.centroid_handle = plt.scatter(pick.x_centroid, pick.y_centroid, marker='+', c='y', s=100, 
                lw=1)


            # Draw colored pixels
            #if pick.photometry_pixels:


            # Update canvas
            if update_plot:
                plt.gcf().canvas.draw()





    def addCentroid(self, frame, x_centroid, y_centroid):
        """ Add the centroid to the list of centroids. """

        # Check if there are previous picks on this frame
        prev_pick = [i for i, pick in enumerate(self.pick_list) if pick.frame == frame]
            
        # Update centroids of previous pick if it exists
        if prev_pick:

            i = prev_pick[0]
            
            self.pick_list[i].x_centroid = x_centroid
            self.pick_list[i].y_centroid = y_centroid

            print('Pick modified!')

        # Create a new pick
        else:

            pick = Pick()

            pick.frame = frame
            pick.x_centroid = x_centroid
            pick.y_centroid = y_centroid

            self.pick_list.append(pick)

            print('Pick added!')


    def removeCentroid(self, frame):
        """ Remove the centroid from the list of centroids. """

        # Check if there are previous picks on this frame
        prev_pick = [i for i, pick in enumerate(self.pick_list) if pick.frame == frame]

        if prev_pick:

            i = prev_pick[0]

            # Remove the centroid
            self.pick_list.pop(i)



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

        self.current_frame = (self.current_frame - 1)%256

        self.updateImage()


    def nextFrame(self):

        self.current_frame = (self.current_frame + 1)%256

        self.updateImage()






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


    # Load the FF file if given
    if ff_name is not None:
        ff = readFF(*os.path.split(ff_name))
    else:
        ff = None


    # Load the FR file is given
    if fr_name is not None:
        fr = readFR(*os.path.split(fr_name))
    else:
        fr = None


    # Init the fireball picker
    fireball_picker = FireballPickTool(config, ff, fr, first_frame=cml_args.firstframe)


    plt.tight_layout()
    plt.show()


