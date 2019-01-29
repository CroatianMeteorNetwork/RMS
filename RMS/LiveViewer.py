""" Showing an image on the screen in parallel to the main program, and update the shown image on the screen
    from the other thread.
"""

from __future__ import print_function, division, absolute_import

import cv2
import numpy as np
import time
import multiprocessing

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

# tkinter import that works on both Python 2 and 3
try:
    import tkinter
except:
    import Tkinter as tkinter


def drawText(ff_array, img_text):
    """ Draws text on the image represented as a numpy array.

    """

    # Convert the array to PIL image
    im = Image.fromarray(np.uint8(ff_array))
    im = im.convert('RGB')
    draw = ImageDraw.Draw(im)

    # Load the default font
    font = ImageFont.load_default()

    # Draw the text on the image, in the upper left corent
    draw.text((0, 0), img_text, (255,255,0), font=font)
    draw = ImageDraw.Draw(im)

    # Convert the type of the image to grayscale, with one color
    try:
        if len(ff_array[0][0]) != 3:
            im = im.convert('L')
    except:
        im = im.convert('L')

    return np.array(im)



class LiveViewer(multiprocessing.Process):
    """ Uses OpenCV to show an image, which can be updated from another thread. """

    def __init__(self, window_name=None):
        """
        Keyword arguments:
            window_name: [str] name (title) of the window

        """

        super(LiveViewer, self).__init__()
        
        self.img_queue = multiprocessing.Queue()
        self.window_name = window_name
        self.first_image = True
        self.run_exited = multiprocessing.Event()

        self.start()



    def start(self):
        """ Start the live viewer window.
        """

        super(LiveViewer, self).start()



    def updateImage(self, img, img_text=None):
        """ Update the image on the screen. 
        
        Arguments:
            img: [ndarray] array with the image
            img_text: [str] text to be written on the image
            
        """

        self.img_queue.put([img, img_text])

        time.sleep(0.1)



    def run(self):
        """ Keep updating the image on the screen from the queue. """


        # Repeat until the live viewer is not stopped from the outside
        while True:

            # Get the next element in the queue (blocking, until next element is available)
            item = self.img_queue.get(block=True)

            # If the 'poison pill' is received, exit the viewer
            if item is None:
                self.run_exited.set()
                return


            img, img_text = item


            # Find the screen size
            root = tkinter.Tk()
            root.withdraw()

            screen_w = root.winfo_screenwidth()
            screen_h = root.winfo_screenheight()

            root.destroy()
            del root

            # If the screen is smaller than the image, resize the image
            if (screen_h < img.shape[0]) or (screen_w < img.shape[1]):

                # Find the ratios between dimentions
                y_ratio = screen_h/img.shape[0]
                x_ratio = screen_w/img.shape[1]

                # Resize the image so that the image fits the screen
                min_ratio = min(x_ratio, y_ratio)

                width_new = int(img.shape[1]*min_ratio)
                height_new = int(img.shape[0]*min_ratio)

                img = cv2.resize(img, (width_new, height_new))



            # Write text on the image if any is given
            if img_text is not None:
                img = drawText(img, img_text)


            # Update the image on the screen
            cv2.imshow(self.window_name, img)

            # If this is the first image, move it to the upper left corner
            if self.first_image:
                
                cv2.moveWindow(self.window_name, 0, 0)

                self.first_image = False


            cv2.waitKey(100)



    def stop(self):
        """ Stop the viewer. """

        self.terminate()

        cv2.destroyAllWindows()

        self.join()




if __name__ == "__main__":

    ### Test the live viewer ###

    # Start the viewer
    live_view = LiveViewer(window_name='Maxpixel')

    # Generate an example image
    img = np.zeros((500, 500))

    for i in range(50):
        i = i*3
        img[i, i] = 128
        live_view.updateImage(img, str(i))
        time.sleep(0.1)



    # img[::5, ::5] = 128

    # # Update the image in the Viewer
    # live_view.updateImage(img.astype(np.uint8), 'test 1')

    # print('updated')

    # time.sleep(5)

    # # Generate another image
    # img[::-2, ::-2] = 128

    # # Upate the image in the viewer
    # live_view.updateImage(img, 'blah 2')

    # time.sleep(2)

    # print('close')



    # Stop the viewer
    live_view.stop()
    del live_view # IMPORTANT! Otherwise the program does not exit

    print('stopped')