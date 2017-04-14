""" Showing an image on the screen in parallel to the main program, and update the shown image on the screen
    from the other thread.
"""

import cv2
import numpy as np
import time
import multiprocessing



class LiveViewer(multiprocessing.Process):
    """ Uses OpenCV to show an image, which can be updated from another thread. """

    def __init__(self):

        super(LiveViewer, self).__init__()
        
        self.img_queue = multiprocessing.Queue()

        self.window_name = None

        self.start()



    def start(self):
        """ Start the live viewer window.
        """

        super(LiveViewer, self).start()



    def updateImage(self, img, window_name):
        """ Update the image on the screen. 
        
        Arguments:
            img: [ndarray] array with the image
            window_name: [str] name of the window
            
        """

        self.img_queue.put([img, window_name])



    def run(self):
        """ Keep updating the image on the screen from the queue. """


        # Repeat until the live viewer is not stopped from the outside
        while True:

            # Get the next element in the queue
            item = self.img_queue.get(True)

            # If the 'poison pill' is received, exit the viewer
            if item is None:
                break


            img, window_name = item

            # Kill the previous window
            cv2.destroyWindow(self.window_name)
            cv2.waitKey()

            self.window_name = window_name

            # Update the image on the screen
            cv2.imshow(self.window_name, img)

            cv2.waitKey(100)



    def stop(self):

        # Put the 'poison pill' in the queue which will exit the viewer
        self.img_queue.put(None)
        self.join()




if __name__ == "__main__":

    ### Test the live viewer ###

    # Start the viewer
    live_view = LiveViewer()

    # Generate an example image
    img = np.zeros((500, 500))
    img[::5, ::5] = 255

    # Update the image in the Viewer
    live_view.updateImage(img.astype(np.uint8), '1')

    print 'updated'

    time.sleep(5)

    # Generate another image
    img[::-2, ::-2] = 1

    # Upate the image in the viewer
    live_view.updateImage(img, '2')

    time.sleep(2)

    print 'close'

    # Stop the viewer
    live_view.stop()

    time.sleep(3)

