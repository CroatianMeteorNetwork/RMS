""" Shows the live stream from the camera. """

from __future__ import print_function, division, absolute_import

import cv2
import time

import RMS.ConfigReader as cr



if __name__ == "__main__":

    # Load the configuration file
    config = cr.parse(".config")

    # Open video device
    vcap = cv2.VideoCapture(config.deviceID)


    t_prev = time.time()
    counter = 0

    first_image = True

    if vcap.isOpened():
        print('Open')

        while(1):
            
            if (time.time() - t_prev) >= 1.0:
                t_prev = time.time()

                # Print the number of frames received in the last second
                print("FPS:", counter)
                counter = -1


            # Get the video frame
            ret, frame = vcap.read()

            # If the connection has been lost, try reconnecting the device
            if not ret:

                print('Connection lost! Reconnecting...')

                while not vcap.isOpened():
                    
                    time.sleep(5)

                    print('Trying to reconnect...')
                    vcap = cv2.VideoCapture(config.deviceID)
                    

            counter += 1

            window_name = 'Live stream'
            cv2.imshow(window_name, frame)

            # If this is the first image, move it to the upper left corner
            if first_image:
                
                cv2.moveWindow(window_name, 0, 0)

                first_image = False

            cv2.waitKey(1)

    else:
        print('Cant open video stream:', config.deviceID)