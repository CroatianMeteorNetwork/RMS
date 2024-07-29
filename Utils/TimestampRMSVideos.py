""" Take videos collected with RMS (gstreamer) and change their names to include the timestamp of the first frame. """

import os
import sys
import datetime
import shutil
import tarfile
from collections import OrderedDict

import cv2
import numpy as np

from RMS.Formats.FFfile import filenameToDatetime




class FrameTime:
    def __init__(self, frame_dt_data, camera_id):
        """ Initialize the FrameTime object with a dictionary of frame numbers and their corresponding timestamps. 
        
        Arguments:
            frame_dt_data: [dict] a dictionary with frame numbers as keys and datetime.datetime objects as values
            camera_id: [str] the camera ID from which the frame data was collected

        """
        
        self.frame_dt_data = frame_dt_data

        self.camera_id = camera_id

        # Compute the average time difference between frames using all the frame numbers
        keys = np.array(list(frame_dt_data.keys()))
        key_diff = np.diff(keys)
        time_diff = np.diff(list(frame_dt_data.values()))
        time_diff = np.array([t.total_seconds() for t in time_diff])

        self.fps = np.mean(key_diff/time_diff)
        

    def timeFromFrameNumber(self, frame_number):
        """ Compute the timestamp of the given frame number. 
        
        Arguments:
            frame_number: [int] the frame number for which to compute the timestamp

        Returns:
            dt: [datetime.datetime] the timestamp of the given frame number

        """

        # Find the closest frame number smaller than the given frame number in the dictionary
        keys = np.array(list(self.frame_dt_data.keys()))
        key = keys[keys <= frame_number].max()

        # Compute the time at the given frame number
        dt = self.frame_dt_data[key] + datetime.timedelta(seconds=(frame_number - key)/self.fps)

        return dt




def loadFrameTimeData(fs_archive):
    """ Load the frame time data from the FS files in the given archive.

    Arguments:
        fs_archive: [str] the path to the FS archive file

    Returns:
        ft: [FrameTime] the FrameTime object containing the frame time data

    """

    # Extract the camera ID from the archive name
    camera_id = os.path.basename(fs_archive).split("_")[1]

    # Load the list of FS files from the tar.bz2 archive
    frame_time_data = OrderedDict()
    with tarfile.open(fs_archive, "r:bz2") as archive:
        for member in archive.getmembers():
            if member.isfile():
                file_name = member.name.replace("./", "")

                # Make sure the file ends .bin
                if not file_name.endswith(".bin"):
                    continue
            
                # Extract the frame number from the file name
                fn_data = file_name.split("_")
                frame_num = int(fn_data[5])

                frame_time_data[frame_num] = filenameToDatetime(file_name, microseconds=True)


    # Sort the FS files by name
    frame_time_data = OrderedDict(sorted(frame_time_data.items(), key=lambda x: x[0]))

    # Initialize the FrameTime object
    ft = FrameTime(frame_time_data, camera_id)

    return ft


def getVideoFrameCount(video_file):
    """ Get the number of frames in the given video file.

    Arguments:
        video_file: [str] the path to the video file
        
    Returns:
        frame_count: [int] the number of frames in the video file

    """

    # Check that the video file exists
    if not os.path.exists(video_file):
        raise FileNotFoundError("Video file {:s} not found!".format(video_file))

    cap = cv2.VideoCapture(video_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # If the frame count is <= 0, try to get the frame count manually
    if frame_count <= 0:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

    cap.release()

    return frame_count



def timestampRMSVideos(dir_path, rename=False):
    """ Change the names of RMS videos to include the timestamp of the first frame.

    Arguments:
        dir_path: [str] the path to the directory with video_*.mkv files and the FS archive

    Keyword arguments:
        rename: [bool] rename the video files instead of making copies

    """

    # Find the the FS archive in the directory
    fs_archive = None
    for f in os.listdir(dir_path):
        if f.startswith("FS") and f.endswith(".tar.bz2"):
            fs_archive = os.path.join(dir_path, f)
            break

    if fs_archive is None:
        print("No FS archive found in the directory!")
        return None



    # Load the frame time data from the FS files
    ft = loadFrameTimeData(fs_archive)


    # Load the list of all video*.mkv files in the directory
    video_files = [f for f in sorted(os.listdir(dir_path)) if f.startswith("video_") and f.endswith(".mkv")]

    if len(video_files) == 0:
        print("No video files found in the directory!")
        return None


    ### Assume that the first video begins exactly at the time of the first FS file ###
    ### Compute the beginning time of each video file ###
    video_frame_counts = []
    video_start_times = []
    for video_file in video_files:
        
        # Get the number of frames in the video file
        frame_count = getVideoFrameCount(os.path.join(dir_path, video_file))

        # Sum all previous frame counts to get the frame number of the first frame in the video file
        total_frames = sum(video_frame_counts)

        # Get the time of the first frame in the video file
        start_time = ft.timeFromFrameNumber(total_frames)

        video_frame_counts.append(frame_count)
        video_start_times.append(start_time)

        # Generate a video name which includes the timestamp of the first frame
        video_ts_name = "{:s}_{:s}_video.mkv".format(ft.camera_id, start_time.strftime("%Y%m%d_%H%M%S_%f"))
        
        #print("{:s} - total frames: {:d}, start time: {:s}".format(video_file, total_frames, str(start_time)))
        print("fr tot = {:8d}, fr vid = {:d}, {:s} - {:s}".format(total_frames, frame_count, video_file, video_ts_name))

        if rename:
            
            # Rename the video file to the new name
            os.rename(os.path.join(dir_path, video_file), os.path.join(dir_path, video_ts_name))

        else:

            # Copy the video file to the new name
            shutil.copy2(os.path.join(dir_path, video_file), os.path.join(dir_path, video_ts_name))





if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Change the names of RMS videos to include the timestamp of the first frame.")

    parser.add_argument("dir_path", type=str, help="Directory with video_*.mkv files and the FS archive.")

    parser.add_argument("-r", "--rename", action="store_true", help="Rename the video files instead of making copies.")

    cml_args = parser.parse_args()

    ### 

    dir_path = cml_args.dir_path
    
    timestampRMSVideos(dir_path, rename=cml_args.rename)


    