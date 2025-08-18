#!/usr/bin/env python

# Initial contributor: Richard Bassom, 2019


""" Program to check through a set of exposures. Builds up a "max pixel" image to allow quick visual check.
"""

from __future__ import print_function, division, absolute_import

import os
import sys
import signal
import glob
import argparse
import time
import cv2
import numpy as np
import shutil
import threading

from RMS.Formats.FFfile import read as readFF
from RMS.Formats.FFfile import validFFName

try:
   # Python 2
   import Queue as queue
except:
   # Python 3
   import queue


### CONSTANTS

VERBOSE = False
QUEUE_SIZE = 2

# Timeout in seconds for getting images from image queue
QUEUE_TIMEOUT = 0.2       

# Number of images for a long max image
NUM_LONG_MAX_IMAGES = 150 

# Pixel level difference threshold
THRESHOLD_DIFF = 20       

# Pixel level for a bright image
THRESHOLD_BRIGHT = 40     

# Ratio of total image pixels above brightness threshold to cause a pause
RATIO_BRIGHT_THRESHOLD = 0.5

brightness_check = False

###


class NamedImage():
   image = None
   image_time = None

   def __init__(self, image, image_name):
      self.image = image
      self.image_name = image_name


class ImageProducer(threading.Thread):
   def __init__(self, threadName, queue, filenames):
      threading.Thread.__init__( self, name=threadName )

      self.producerQueue = queue
      self.filenames = filenames

   def run(self):
      """ Loop through image files in the directory """

      self.index = 0
      self.last = -1

      if ignore_last_frame: self.last = -2

      while True:
         self.get_filenames()
         while self.index < len(self.files)+self.last:
            try:
               if VERBOSE: print(self.files[self.index])
               if self.files[self.index].endswith('fits'):
         
                  image = readFF(*os.path.split(self.files[self.index])).maxpixel
                  self.producerQueue.put(NamedImage(image, self.files[self.index]), block=True)

               else:
                  self.producerQueue.put(NamedImage(cv2.imread(self.files[self.index],cv2.IMREAD_COLOR), self.files[self.index]), block=True)

               self.index += 1
            except:
               print("Error getting image")
               self.index = 0
         time.sleep(1)

   def get_filenames(self):
      if self.filenames is None or len(self.filenames) == 0:
         self.files = sorted(glob.glob('*.jpg'))
      else:
         self.files = self.filenames

   def get_index(self):
      return self.index - self.producerQueue.qsize()

   def set_index(self, index):
      self.index = index

   def move_index(self, movement):
      #while not self.producerQueue.empty(): self.producerQueue.get()
      self.index += movement
      if self.index < 0: self.index = 0



class FrameAnalyser():
   def __init__(self, dir_path, img_type):

      # Take FF files if the the image type was not given
      if img_type is None:
         
         # Get all FF files in the given folder
         self.filenames = sorted([os.path.abspath(os.path.join(dir_path, filename)) for filename \
            in os.listdir(dir_path) if validFFName(filename)])

      else:

         # Get all images of the given extension
         self.filenames = sorted([os.path.abspath(os.path.join(dir_path, filename)) for filename \
            in os.listdir(dir_path) if filename.lower().endswith(img_type.lower())])


      # If no files were given, take 
      if (self.filenames is None) or (len(self.filenames) == 0):
         print('No files in the directory that match the pattern!')
         sys.exit()


      self.files = self.filenames


      # Load an image
      for filename in self.filenames:

         if validFFName(os.path.basename(filename)):
            self.im8u = readFF(*os.path.split(filename)).maxpixel

         else:
            self.im8u = cv2.imread(filename, cv2.IMREAD_COLOR)

         break


      if VERBOSE: 
         print(self.im8u.shape)


      self.HEIGHT = self.im8u.shape[0]
      self.WIDTH  = self.im8u.shape[1]

      if VERBOSE: 
         print("Width =", self.WIDTH, "Height = ", self.HEIGHT)

      if self.WIDTH > 2600:   
         self.scale = 0.25
      elif self.WIDTH > 1280:
         self.scale = 0.5
      else:                    
         self.scale = 1.0



      if len(self.im8u.shape) == 3:
         self.im8u_grey      = cv2.cvtColor(self.im8u, cv2.COLOR_BGR2GRAY)
         self.last_im8u      = cv2.cvtColor(self.im8u, cv2.COLOR_BGR2GRAY)
      else:
         self.im8u_grey      = np.copy(self.im8u)
         self.last_im8u      = np.copy(self.im8u)

      self.diff           = np.copy(self.im8u)
      self.prev_image     = np.copy(self.im8u)
      self.short_max_im8u = np.copy(self.im8u)
      self.long_max_im8u  = np.copy(self.im8u)
      self.short_coadd    = np.copy(self.im8u)
      self.short_coadd_scaled = np.copy(self.im8u)
      self.trigger_list   = []

      self.flip = False
      self.contrast = False
      self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

      # Set the font for overlay
      self.font = cv2.FONT_HERSHEY_SIMPLEX

      self.index = 0
      self.pause = False
      self.short_max_im8u.fill(0)

      cv2.imshow('CheckNight', cv2.resize(self.short_max_im8u, (0,0), fx=self.scale, fy=self.scale))
      cv2.moveWindow("CheckNight", 0, 0)
      # cv2.resizeWindow('CheckNight', self.WIDTH/2, self.HEIGHT/2)


   def start(self):

      self.pause = pause_at_start
      self.show_max = True
      self.single_step = False

      # Start the image producer
      self.imageQueue = queue.Queue(maxsize=QUEUE_SIZE)
      self.imageProducer = ImageProducer("Producer", self.imageQueue, self.files)
      self.imageProducer.start()

      # Just display the first bright images
      while (brightness_check):
         try:
            # Get the image from the queue
            self.named_image = self.imageQueue.get(timeout = QUEUE_TIMEOUT)
            self.im8u = self.named_image.image

            # Resize and display the image
            cv2.imshow('CheckNight', cv2.resize(self.im8u, (0,0), fx=self.scale, fy=self.scale))
            
            # Check for key presses
            self.check_key()
            
            while(self.pause): 
               self.check_key()

            # If the latest image is bright continue
            ratio_pixels_above_thresh = np.count_nonzero(self.im8u \
               > THRESHOLD_BRIGHT)/(self.im8u.shape[0]*self.im8u.shape[1])
            
            if ratio_pixels_above_thresh > RATIO_BRIGHT_THRESHOLD: 
               pass
            else: 
               break

         except: 
            pass


      while True:
         try:
            # Get the image from the queue
            self.named_image = self.imageQueue.get(timeout=QUEUE_TIMEOUT)
            self.im8u = self.named_image.image

            # Calculate the diff and pause before display if above threshold
            #self.diff = cv2.subtract(self.im8u, self.prev_image)
            #ratio_pixels_above_thresh = np.count_nonzero(self.diff>THRESHOLD_DIFF)

            # If the latest image is bright, copy it, but ignore it in the max image
            ratio_pixels_above_thresh = np.count_nonzero(self.im8u > THRESHOLD_BRIGHT)/(self.im8u.shape[0] \
               *self.im8u.shape[1]) if brightness_check else 0

            if brightness_check and (ratio_pixels_above_thresh > RATIO_BRIGHT_THRESHOLD):
               if VERBOSE: print("Bright:", self.named_image.image_name, ratio_pixels_above_thresh)
               #self.pause = True
               #while(self.pause): self.check_key()
               #self.trigger_list.append(self.imageProducer.get_currentimagefilename())
               cv2.imwrite("bright-" + self.named_image.image_name, self.named_image.image)
            
            else:
               self.prev_image = np.copy(self.im8u)

               # Update the max image
               cv2.max(self.im8u, self.short_max_im8u, self.short_max_im8u)
               # self.short_max_im8u = np.maximum(self.short_max_im8u, self.im8u)

            # Copy the image, flip if required, and stamp for display
            if self.show_max:
               self.display_image = np.copy(self.short_max_im8u)
            else:
               self.display_image = np.copy(self.im8u)

            if self.flip: self.display_image = cv2.flip(self.display_image, -1)
            self.stamp_image(self.display_image, os.path.basename(self.named_image.image_name))

            # Resize and display the image
            cv2.imshow("CheckNight", cv2.resize(self.display_image, (0,0), fx=self.scale, fy=self.scale))


         except queue.Empty:

            self.pause = True
            while(self.pause): self.check_key()

         except:
            print("Exception: ", sys.exc_info()[0])
            self.pause = True
            while(self.pause): self.check_key()
            # self.finish()

         #print("--- %s seconds ---" % (time.time() - start_time))
         #start_time = time.time()

         # Check for key presses
         while(self.pause):
            self.check_key()

         self.check_key()
         if self.single_step:
             self.pause = True

      self.finish()

   """ Clear the image queue """
   def clear_queue(self):
      try:
         for i in range(QUEUE_SIZE+1):
            self.imageQueue.get(timeout = QUEUE_TIMEOUT)
      except: pass
      time.sleep(0.1)


   """ Check the keyboard for key press events and action the necessary events """
   def check_key(self):
      self.key_pressed64 = cv2.waitKey(1) & 0xffff
      self.key_pressed = self.key_pressed64 & 0xff
      # if self.key_pressed64 != 255: print(hex(self.key_pressed64))
      if (self.key_pressed64 < 0) or (self.key_pressed64 == 255) or (self.key_pressed == 255):
         return    # No key pressed
      elif(self.key_pressed == 0x1b) or (self.key_pressed == 0x71): # <Esc> or q key
         self.finish()
      elif (self.key_pressed == 0x72):  # 'r' reset
         self.short_max_im8u.fill(0)
      elif (self.key_pressed == 0x0A) or (self.key_pressed == 13):  # 'Enter' key - pause and reset
         self.pause = not self.pause
         self.short_max_im8u.fill(0)
      elif (self.key_pressed == 0x73): # 's' key - save image
         cv2.imwrite(self.named_image.image_name + "-saved.png", self.short_max_im8u)
         print("Saved image: ", self.named_image.image_name + "-saved.png")
      elif (self.key_pressed == 0x20): # space key - pause
         self.pause = not self.pause
         self.single_step = False
      elif (self.key_pressed == 0x6d): # m key - toggle max pixel display
         self.short_max_im8u.fill(0)
         self.show_max = not self.show_max
      elif (self.key_pressed64 == 0x055) or (self.key_pressed == 44): # pgup arrow - rewind 100 frames
         self.short_max_im8u.fill(0)
         self.imageProducer.move_index(-100)
         self.clear_queue()
         self.pause = False
      elif (self.key_pressed64 == 0x056) or (self.key_pressed == 45): # pgdown arrow - forward 300 frames
         self.short_max_im8u.fill(0)
         self.imageProducer.move_index(300)
         self.clear_queue()
         self.pause = False
      elif (self.key_pressed == 65361) or (self.key_pressed64 == 0x051) or (self.key_pressed == 2): # left arrow - rewind 100 frames
         self.short_max_im8u.fill(0)
         self.imageProducer.move_index(-100)
         self.clear_queue()
         self.pause = False
      elif (self.key_pressed == 65363) or (self.key_pressed64 == 0x053) or (self.key_pressed == 3): # right arrow - forward 100 frames
         self.short_max_im8u.fill(0)
         self.imageProducer.move_index(100)
         self.clear_queue()
         self.pause = False
      elif (self.key_pressed == 65362) or (self.key_pressed64 == 0x052) or (self.key_pressed == 0): # up arrow - flip display frame
         self.short_max_im8u.fill(0)
         self.flip = not self.flip
      elif (self.key_pressed == 0x63): # 'c' key - change contrast
         if len(self.display_image) == 3:
            cv2.imshow("CheckNight", cv2.resize(self.clahe.apply(cv2.cvtColor(self.display_image, cv2.COLOR_RGB2GRAY)), (0,0), fx=self.scale, fy=self.scale))
         else:
            cv2.imshow("CheckNight", cv2.resize(self.clahe.apply(self.display_image), (0,0), fx=self.scale, fy=self.scale))
         self.contrast = not self.contrast
         self.pause = True
      #elif (self.key_pressed == 0x76): # 'v' key to view latest frame using eog
      #   try:
      #      os.system ('eog ' + self.named_image.image_name)
      #   except:
      #      print("Unable to view image")
      elif (self.key_pressed == 0x2e): # '.' key to single step
         self.single_step = True
         self.pause = False
      else:
          print("Unknown key:", self.key_pressed)



   """ Timestamp an image """
   def timestamp_image(self, im8u):
      cv2.putText(im8u, time.strftime("%Z %Y-%m-%d %H:%M:%S"), (10, self.HEIGHT-10), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255))

   """ Stamp an image """
   def stamp_image(self, im8u, stamp):
      cv2.putText(im8u, stamp, (10, self.HEIGHT-10), cv2.FONT_HERSHEY_PLAIN, 2.0, (255,255,255))

   """ Number an image """
   def number_image(self, im8u, number):
      cv2.putText(im8u, str(number), (10, self.HEIGHT-10), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255))


   def finish(self):
      cv2.destroyAllWindows()

      os._exit(0)

   """ Create a set of long max pixel images from the image set """
   def make_long_frames(self):
      # Start the image producer
      self.imageQueue = queue.Queue(maxsize=2)
      self.imageProducer = ImageProducer("Producer", self.imageQueue)
      self.imageProducer.start()
      self.max_index = 0

      try:
         self.max_image = self.imageQueue.get(timeout = QUEUE_TIMEOUT)
         while True:
            # Get the next image from the queue and update the max
            self.im = self.imageQueue.get(timeout = QUEUE_TIMEOUT)
            cv2.max(self.im, self.max_image, self.max_image)

            self.max_index += 1

            # Write a long image
            if self.max_index >= NUM_LONG_MAX_IMAGES:
               self.max_index = 0
               cv2.imwrite(("long-" + self.imageProducer.get_currentimagefilename()), self.max_image)
               self.max_image.fill(0)

      except:
         cv2.imwrite(("long-" + self.imageProducer.get_currentimagefilename()), self.max_image)
         print("Exception checking file ", sys.exc_info()[0])
         self.finish()


# Main program
if __name__ == "__main__":

   # Construct the argument parser and parse the arguments
   ap = argparse.ArgumentParser(description='Build a max-pixel image from a sequence of images', \
      epilog='Example of use: check_images FF*.fits')

   ap.add_argument("dir_path", nargs=1, help="Path to directory with FF files.")
   ap.add_argument("img_type", nargs='?', help="File type of the images. If not given FF files will be taken.",
      default=None)
   ap.add_argument("-n", "--noninteractive", action='store_false', help="Run non-interactively")
   ap.add_argument("-b", "--brightness_check", action='store_true', help="Brightness check")
   ap.add_argument("-i", "--ignore", action='store_true', help="Ignore first and last frames")
   ap.add_argument("-p", "--pause", action='store_true', help="Pause from start of frames")

   args = vars(ap.parse_args())


   dir_path = args["dir_path"][0]
   img_type = args["img_type"]
   is_interactive = args['noninteractive']
   brightness_check = args['brightness_check']
   ignore_last_frame = args['ignore']
   pause_at_start = args['pause']


   # Print key control help
   print("")
   print("Command control keys")
   print("--------------------")
   print("Space key:     pause")
   print("Enter key:     pause and reset max-pixel image")
   print("'r' key:       reset max-pixel image")
   print("right arrow:   forward 100 frames")
   print("left arrow:    rewind 100 frames")
   print("up arrow:      flip display image")
   print("pgdwn key:     forward 300 frames")
   print("pgup key:      rewind 100 frames")
   print("'.' key:       single step frame forward")
   print("'m' key:       toggle between single image and max-pixel display")
   print("'c' key:       change image contrast settings")
   print("'s' key:       save currently displayed max image")
   print("'q' key:       quit")
   print("--------------------")
   print("")

   # Start the frame analyser
   frame_analyzer = FrameAnalyser(dir_path, img_type)

   if is_interactive: 
      frame_analyzer.start()

   else: 
      frame_analyzer.make_long_frames()

