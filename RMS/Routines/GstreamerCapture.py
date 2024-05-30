
import os
# Set GStreamer debug level. Use '2' for warnings in production environments.
os.environ['GST_DEBUG'] = '3'


import time
import numpy as np
import multiprocessing

GST_IMPORTED = False
try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    GST_IMPORTED = True

except ImportError as e:
    print('Could not import gi: {}. Using OpenCV.'.format(e))



class GstCaptureTest(multiprocessing.Process):
    def __init__(self, nframes=100):

        super(GstCaptureTest, self).__init__()

        self.exit = multiprocessing.Event()

        self.nframes = nframes

    def startCapture(self):
        self.start()

    def createGSTDevice(self):

        Gst.init(None)

        self.pipeline = Gst.parse_launch(
            "rtspsrc buffer-mode=1 protocols=tcp tcp-timeout=5000000 retry=5 location=rtsp://192.168.42.10:554/user=admin&password=&channel=1&stream=0.sdp/ ! "
            "rtph264depay ! queue ! h264parse ! nvh264dec ! videoconvert ! "
            "video/x-raw,format=BGR ! "
            "queue leaky=downstream max-size-buffers=100 max-size-bytes=0 max-size-time=0 ! "
            "appsink max-buffers=100 drop=true sync=0 name=appsink"
        )

        self.pipeline.set_state(Gst.State.PLAYING)

        return self.pipeline.get_by_name("appsink")
    
    def initStream(self):
        
        print("Initializing stream...", end="")
        self.device = self.createGSTDevice()
        print("done.")

        # Attempt to get a sample and determine the frame shape
        print("Trying to get a sample...", end="")
        sample = self.device.emit("pull-sample")
        if not sample:
            raise ValueError("Could not obtain sample.")
        print("done.")

        buffer = sample.get_buffer()
        ret, map_info = buffer.map(Gst.MapFlags.READ)
        if not ret:
            raise ValueError("Could not obtain frame.")

        # Extract video information from caps
        caps = sample.get_caps()
        if not caps:
            raise ValueError("Sample caps are None.")
            
        structure = caps.get_structure(0)
        if not structure:
            raise ValueError("Could not determine frame shape.")
        
        # Extract width, height, and format, and create frame
        width = structure.get_value('width')
        height = structure.get_value('height')
        self.frame_shape = (height, width, 3)

        frame = np.ndarray(shape=self.frame_shape, buffer=map_info.data, dtype=np.uint8)

        # Unmap the buffer
        buffer.unmap(map_info)

        return True
    

    def deviceIsOpened(self):

        state = self.device.get_state(Gst.CLOCK_TIME_NONE).state
        if state == Gst.State.PLAYING:
            return True
        else:
            return False

    def run(self):

        while not self.initStream():
            print("Trying to initialize stream...")
            time.sleep(1)

        print("Stream initialized.")

        # Wait until the device is opened
        device_opened = False
        for i in range(20):
            time.sleep(1)
            if self.deviceIsOpened():
                device_opened = True
                break

        print("Device opened: ", device_opened)

        # Read the given number of frames
        for i in range(self.nframes):

            frame = self.pull_frame()
            if frame is not None:
                print(i)
                #print(frame)
            

    def pull_frame(self):
        sample = self.device.emit("pull-sample")
        if not sample:
            return None

        buffer = sample.get_buffer()
        ret, map_info = buffer.map(Gst.MapFlags.READ)
        if not ret:
            return None

        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = structure.get_value('width')
        height = structure.get_value('height')
        frame_shape = (height, width, 3)
        frame = np.ndarray(shape=frame_shape, buffer=map_info.data, dtype=np.uint8)
        buffer.unmap(map_info)
        return frame

    def stop(self):
        self.exit.set()
        self.join()





class GstVideoFile():
    def __init__(self, file_path, decoder='decodebin', video_format='GRAY8'):
        """ Initialize the video file stream using GStreamer. 
        
        Arguments:
            file_path: [str] The path to the video file.

        Keyword Arguments:
            decoder: [str] The decoder to use. Default is 'decodebin'. Examples: 'nvh264dec', 'avdec_h264'.
            video_format: [str] The video format to use. Default is 'GRAY8'. Examples: 'BGR', 'GRAY8'.
        
        """

        self.file_path = file_path
        self.decoder = decoder
        self.video_format = video_format

        self.height = None
        self.width = None

        self.frame_shape = None

        self.current_frame = 0

        self.initStream()


    def createGSTDevice(self):

        # Initialize GStreamer
        Gst.init(None)

        pipeline_str = (
            "filesrc location={} ! matroskademux ! h264parse ! {} ! "
            "videoconvert ! video/x-raw,format={} ! "
            "queue leaky=downstream max-size-buffers=100 ! "
            "appsink emit-signals=True max-buffers=100 drop=False sync=0 name=appsink"
            "".format(self.file_path, self.decoder, self.video_format)
        )

        self.pipeline = Gst.parse_launch(pipeline_str)

        print("Gstreamer video pipeline:")
        print(pipeline_str)

        self.pipeline.set_state(Gst.State.PLAYING)

        return self.pipeline.get_by_name("appsink")



    def initStream(self):
        """ Initialize the video stream. """
        
        self.device = self.createGSTDevice()

        # Attempt to get a sample and determine the frame shape
        sample = self.device.emit("pull-sample")
        if not sample:
            raise ValueError("Could not obtain sample.")

        buffer = sample.get_buffer()
        ret, map_info = buffer.map(Gst.MapFlags.READ)
        if not ret:
            raise ValueError("Could not obtain frame.")

        # Extract video information from caps
        caps = sample.get_caps()
        if not caps:
            raise ValueError("Sample caps are None.")
            
        structure = caps.get_structure(0)
        if not structure:
            raise ValueError("Could not determine frame shape.")
        
        # Get the total duration of the video
        self.duration = self.pipeline.query_duration(Gst.Format.TIME)[1]/Gst.SECOND
        
        # Extract width, height, and format, and create frame
        self.width = structure.get_value('width')
        self.height = structure.get_value('height')
        
        # Determine the frame shape, depending on whether the video is grayscale or color
        if self.video_format == 'GRAY8':
            self.frame_shape = (self.height, self.width)
        else:
            self.frame_shape = (self.height, self.width, 3)


        # Get the framerate
        framerate = structure.get_fraction('framerate')
        self.fps = framerate[1]/framerate[0]

        # Calculate total frames
        self.total_frames = int(self.duration*self.fps)

        # Unmap the buffer
        buffer.unmap(map_info)

        return True
    
    
    def read(self):
        """ Read a frame from the video file. """

        sample = self.device.emit("pull-sample")

        if not sample:
            return False, None

        buffer = sample.get_buffer()
        ret, map_info = buffer.map(Gst.MapFlags.READ)

        if not ret:
            return ret, None

        frame = np.ndarray(shape=self.frame_shape, buffer=map_info.data, dtype=np.uint8)

        buffer.unmap(map_info)

        self.current_frame += 1

        return ret, frame


    def restartVideo(self):

        # Set the cursor to the beginning of the video
        self.pipeline.seek_simple(Gst.Format.TIME, Gst.SeekFlags.FLUSH, 0)

        self.current_frame = 0

    
    def get(self, num):
        """ Simulate OpenCVs get method which returns video information. """


        if num == 3:
            return self.width

        elif num == 4:
            return self.height
        
        elif num == 5:
            return self.fps
        
        elif num == 7:
            return self.total_frames

        else:
            return None
        

    def set(self, num, value):
        """ Simulate OpenCVs set method which sets video information. """

        # Set the frame cursor to the given frame number
        if num == 1:

            self.pipeline.seek_simple(Gst.Format.TIME, Gst.SeekFlags.FLUSH, int(value*self.fps*Gst.SECOND))
            self.current_frame = value
            
    

    def frameGenerator(self):
        """ Generator that yields frames from the next video file. """
        
        while True:

            frame = self.read()

            self.current_frame += 1

            if frame is not None:
                yield frame

            else:
                break

    def release(self):
        """ Release the video file. """

        self.pipeline.set_state(Gst.State.NULL)

    
    def get_state(self, gst_param):
        """ Get the state of the video file. """

        return self.pipeline.get_state(gst_param)



if __name__ == "__main__":

    # # Test capture from a camera
    # gst_test = GstCaptureTest()
    # gst_test.startCapture()
    # gst_test.stop()


    # Test capture from a video file
    video_file_path = "/mnt/RMS_data/CapturedFiles/CAWE01_20240530_015840_497533/video_00001.mkv"

    gst_video_file = GstVideoFile(video_file_path, decoder='nvh264dec')

    # Print the video information
    print("Video information:")
    print("Width: ", gst_video_file.width)
    print("Height: ", gst_video_file.height)
    print("Duration: ", gst_video_file.duration, " seconds")
    print("FPS: ", gst_video_file.fps)
    print("Total frames: ", gst_video_file.total_frames)


    # Restart the video to the beginning
    gst_video_file.restartVideo()


    # Go through the video a few times
    for _ in range(3):

        gst_video_file.restartVideo()

        count = 0
        for frame in gst_video_file.frameGenerator():
            #print(count)
            count += 1
            #print(frame)
            #cv2.imshow("frame", frame)
            #cv2.waitKey(1)

    gst_video_file.release()
