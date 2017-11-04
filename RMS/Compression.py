# RPi Meteor Station
# Copyright (C) 2015  Dario Zubovic
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

import time
import logging

from math import floor

# New versions of scipy have weave as a separate module
try:
    from scipy import weave
except ImportError:
    import weave


import numpy as np
import multiprocessing



from RMS.VideoExtraction import Extractor
from RMS.Formats import FFfile, FFStruct
from RMS.Formats import FieldIntensities


# Get the logger from the main module
log = logging.getLogger("logger")


class Compressor(multiprocessing.Process):
    """Compress list of numpy arrays (video frames).

        Output is in Four-frame Temporal Pixel (FTP) format. See the Jenniskens et al., 2011 paper about the
        CAMS project for more info.

    """

    running = False
    
    def __init__(self, data_dir, array1, startTime1, array2, startTime2, config, detector=None, 
        live_view=None, flat_struct=None):
        """

        Arguments:
            array1: first numpy array in shared memory of grayscale video frames
            startTime1: float in shared memory that holds time of first frame in array1
            array2: second numpy array in shared memory
            startTime1: float in shared memory that holds time of first frame in array2
            config: configuration class

        Keyword arguments:
            detector: [Detector object] Handle to Detector object used for running star extraction and
                meteor detection.
            live_view: [LiveViewer object] Handle to the LiveViewer object which will show in real time 
                the latest maxpixel on the screen.
            flat_struct: [Flat struct] Structure containing the flat field. None by default.

        """
        
        super(Compressor, self).__init__()
        
        self.data_dir = data_dir
        self.array1 = array1
        self.startTime1 = startTime1
        self.array2 = array2
        self.startTime2 = startTime2
        self.config = config

        self.detector = detector
        self.live_view = live_view
        self.flat_struct = flat_struct

        self.exit = multiprocessing.Event()

        self.run_exited = multiprocessing.Event()
    


    def compress(self, frames):
        """ Compress frames to the FTP-compatible array and extract sums of intensities per every field.

        NOTE: The standard deviation calculation is performed in a non-standard way due to performance 
            concerns. The end result is the same as a proper calculation due to the usage of low-precision
            8-bit unsigned integers, so the difference does not matter.
        
        Arguments:
            frames: [3D ndarray] grayscale frames stored as 3d numpy array
        
        Return:
            [3D ndarray]: in format: (N, y, x) where N is a member of [0, 1, 2, 3]

        """
        
        # Init the output FTP array
        out = np.empty((4, frames.shape[1], frames.shape[2]), np.uint8)

        # Init the field intensity sums array
        fieldsum = np.zeros((2*frames.shape[0]), np.uint32)
        
        code = """
        unsigned int x, y, n, acc, var, max, max_frame, pixel, num_equal, mean;
        unsigned short rand_count = 0;
        
        unsigned int height = Nframes[1];
        unsigned int width = Nframes[2];
        unsigned int frames_num = Nframes[0];
        unsigned int frames_num_minus_one = frames_num - 1;
        unsigned int frames_num_minus_two = frames_num - 2;

        unsigned int half_frame_n;
        
        // Populate the randomN array with 2**16 random numbers from 0 to 255
        unsigned char randomN[65536] = {};
        unsigned int arand = 1;
        for(n=0; n<65536; n++) 
        {
            arand = (arand * 32719 + 3) % 32749;
            randomN[n] = (unsigned char)(32767.0 / (double)(1 + arand % 32767));
        }
            
        for(y=0; y<height; y++) 
        {
            for(x=0; x<width; x++) 
            {
                acc = 0;
                var = 0;
                max = 0;
                
                // Calculate mean, stddev, max, and max frame
                for(n=0; n<frames_num; n++) 
                {
                    pixel = FRAMES3(n, y, x);
                    acc += pixel;
                    var += pixel*pixel;
                    
                    if(max < pixel)
                    {
                        max = pixel;
                        max_frame = n;
                        num_equal = 1;
                    }
                    // Randomize taken frame number for max pixel if there are several frames with the maximum
                    // value
                    else if(max == pixel)
                    { 
                        num_equal++;
                        
                        // rand_count is unsigned short, which means it will overflow back to 0 after 65535
                        rand_count = (rand_count + 1)% 65536L;

                        // Select the frame by random
                        if(num_equal <= randomN[rand_count])
                        {
                            max_frame = n;
                        }
                    }

                    // Calculate the current half frame index
                    half_frame_n = 2*n + (y + deinterlace_order)%2;

                    // Sum intensity per every field
                    FIELDSUM1(half_frame_n) += (unsigned long) pixel;

                }
                
                // Calculate mean
                acc -= max;    // remove max pixel from average
                mean = acc / frames_num_minus_one;
                
                // Calculate stddev
                var -= max*max;     // remove max pixel
                var -= acc*mean;    // subtract average squared sum of all values (acc*mean = acc*acc/frames_num_minus_one)
                var = sqrt(var / frames_num_minus_two);

                // Make sure that the stddev is not 0, to prevent divide by zero afterwards
                if (var == 0)
                {
                    var = 1;
                }
                
                // Output results
                OUT3(0, y, x) = max;
                OUT3(1, y, x) = max_frame;
                OUT3(2, y, x) = mean;
                OUT3(3, y, x) = var;
            }
        }
        """

        deinterlace_order = self.config.deinterlace_order
        
        # Run the weave code
        weave.inline(code, ['frames', 'out', 'fieldsum', 'deinterlace_order'], verbose=2, 
            extra_compile_args=self.config.weaveArgs, extra_link_args=self.config.weaveArgs)

        return out, fieldsum
    


    def saveFF(self, arr, startTime, N):
        """ Write metadata and data array to FF file.
        
        Arguments:
            arr: [3D ndarray] 3D numpy array in format: (N, y, x) where N is [0, 4)
            startTime: [float] seconds and fractions of a second from epoch to first frame
            N: [int] frame counter (ie. 0000512)
        """
        
        # Generate the name for the file
        date_string = time.strftime("%Y%m%d_%H%M%S", time.localtime(startTime))

        # Calculate miliseconds
        millis = int((startTime - floor(startTime))*1000)
        

        filename = str(self.config.stationID).zfill(3) +  "_" + date_string + "_" + str(millis).zfill(3) \
            + "_" + str(N).zfill(7)

        ff = FFStruct.FFStruct()
        ff.array = arr
        ff.nrows = arr.shape[1]
        ff.ncols = arr.shape[2]
        ff.nbits = self.config.bit_depth
        ff.first = N + 256
        ff.camno = self.config.stationID
        
        # Write the FF file
        FFfile.write(ff, self.data_dir, filename, fmt=self.config.ff_format)
        
        return filename
    


    def stop(self):
        """ Stop compression.
        """
        

        self.exit.set()
        log.debug('Compression exit flag set')

            
        log.debug('Joining compression...')


        t_beg = time.time()

        # Wait until everything is done
        while not self.run_exited.is_set():
            
            time.sleep(0.01)

            # Do not wait more than a minute, just terminate the compression thread then
            if (time.time() - t_beg) > 60:
                log.debug('Waitied more than 60 seconds for compression to end, killing it...')
                break


        log.debug('Compression joined!')

        self.terminate()
        self.join()

        # Return the detector and live viewer objects because they were updated in this namespace
        return self.detector, self.live_view
    


    def start(self):
        """ Start compression.
        """
        
        super(Compressor, self).start()
    


    def run(self):
        """ Retrieve frames from list, convert, compress and save them.
        """
        
        n = 0
        
        # Repeat until the compressor is killed from the outside
        while not self.exit.is_set():

            # Block until frames are available
            while self.startTime1.value == 0 and self.startTime2.value == 0: 

                # Exit function if process was stopped from the outside
                if self.exit.is_set():

                    log.debug('Compression run exit')
                    self.run_exited.set()

                    return None

                time.sleep(0.1)

                

            t = time.time()

            
            if self.startTime1.value != 0:

                # Retrieve time of first frame
                startTime = self.startTime1.value 

                # Copy frames
                frames = self.array1 
                self.startTime1.value = 0

            else:

                # Retrieve time of first frame
                startTime = self.startTime2.value 

                # Copy frames
                frames = self.array2 
                self.startTime2.value = 0

            
            log.debug("memory copy: " + str(time.time() - t) + "s")
            t = time.time()
            
            # Run the compression
            compressed, field_intensities = self.compress(frames)
            
            log.debug("compression: " + str(time.time() - t) + "s")
            t = time.time()
            
            # Save the compressed image
            filename = self.saveFF(compressed, startTime, n*256)
            n += 1
            
            log.debug("saving: " + str(time.time() - t) + "s")


            # Save the extracted intensitites per every field
            FieldIntensities.saveFieldIntensitiesBin(field_intensities, self.data_dir, filename)


            # Run the extractor
            extractor = Extractor(self.config, self.data_dir)
            extractor.start(frames, compressed, filename)

            # Fully format the filename (this could not have been done before as the extractor will add
            # the FR prefix)
            filename = "FF_" + filename + "." + self.config.ff_format


            log.debug('Extractor started for: ' + filename)


            # Run the detection on the file, if the detector handle was given
            if self.detector is not None:

                # Add the file to the detector queue
                self.detector.addJob([self.data_dir, filename, self.config, self.flat_struct])


            # Refresh the maxpixel currently shown on the screen
            if self.live_view is not None:

                # Add the image to the image queue
                self.live_view.updateImage(compressed[0], filename + " maxpixel")



        log.debug('Compression run exit')
        self.run_exited.set()


    