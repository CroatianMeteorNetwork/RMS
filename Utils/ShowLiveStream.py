""" Shows the live stream from the camera. """

from __future__ import print_function, division, absolute_import

import cv2
import time
import argparse

import RMS.ConfigReader as cr
from RMS.Routines.Image import applyBrightnessAndContrast


def get_device(config):
    """ Get the video device """
    if config.media_backend == 'v4l2':
        vcap = cv2.VideoCapture(config.deviceID, cv2.CAP_V4L2)
        vcap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    else:
        vcap = cv2.VideoCapture(config.deviceID)

    return vcap


if __name__ == "__main__":



    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description=""" Show live stream from the camera.
        """)

    arg_parser.add_argument('-n', '--novideo', action="store_true", help="""Get the frames in the background,
        but don't show them on the screen. """)

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str,
        help="Path to a config file which will be used instead of the default one.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()



    # Load the configuration file

    config = cr.loadConfigFromDirectory(cml_args.config, 'notused')

    # Open video device
    vcap = get_device(config)

    t_prev = time.time()
    counter = 0
    fps = config.fps

    first_image = True

    if vcap.isOpened():
        print('Open')

        while(1):
            
            if (time.time() - t_prev) >= 1.0:
                t_prev = time.time()

                fps = counter

                # Print the number of frames received in the last second
                print("FPS:", counter)
                counter = 0


                # If the FPS is lower than the configured value, this means that the video buffer is full and needs
                # to be emptied
                if fps < config.fps:
                    for i in range(int(config.fps - fps) + 1):
                        ret = vcap.grab()
                        counter += 1
                    

            # Get the video frame
            ret, frame = vcap.read()


            # Apply brightness and contrast corrections if given
            if (config.brightness != 0) or (config.contrast != 0):
                frame = applyBrightnessAndContrast(frame, config.brightness, config.contrast)
            

            # If the connection has been lost, try reconnecting the device
            if not ret:

                print('Connection lost! Reconnecting...')

                while 1:
                    print('Trying to reconnect...')
                    time.sleep(5)

                    # Try reconnecting the device and getting a frame
                    vcap = get_device(config)

                    ret, frame = vcap.read()

                    if ret:
                        break

                continue

                    

            counter += 1


            if not cml_args.novideo:

                window_name = 'Live stream'
                if len(frame.shape) == 3:
                    
                    if frame.shape[2] == 3:

                        # Get green channel
                        frame = frame[:, :, 1]

                    # If UYVY image given, take luma (Y) channel
                    elif config.uyvy_pixelformat and (frame.shape[2] == 2):
                        frame = frame[:, :, 1]

                    else:
                        frame = frame[:, :, 0]

                cv2.imshow(window_name, frame)

                # If this is the first image, move it to the upper left corner
                if first_image:
                    
                    cv2.moveWindow(window_name, 0, 0)

                    first_image = False

                cv2.waitKey(1)

    else:
        print("Can't open video stream:", config.deviceID)
