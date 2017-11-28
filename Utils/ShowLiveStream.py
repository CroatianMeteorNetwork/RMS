""" Shows the live stream from the camera. """

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

    if vcap.isOpened():
        print('Open')

        while(1):

            
            if (time.time() - t_prev) >= 1:
                t_prev = time.time()

                # Print the number of frames received in the last second
                print(counter)
                counter = 0

            ret, frame = vcap.read()

            counter += 1

            cv2.imshow('VIDEO', frame)
            cv2.waitKey(1)

    else:
        print('Cant open')