# RPi Meteor Station
# Copyright (C) 2024
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

"""
This script downloads videos from the static part of the www.globalmeteornetwork.org website
and programmatically generates and optionally executes a command to generate a mosaic of videos

It is based on the technique at

https://trac.ffmpeg.org/wiki/Create%20a%20mosaic%20out%20of%20several%20input%20videos

The command template is

ffmpeg
	-i 1.avi -i 2.avi -i 3.avi -i 4.avi
	-filter_complex "
		nullsrc=size=640x480 [base];
		[0:v] setpts=PTS-STARTPTS, scale=320x240 [upperleft];
		[1:v] setpts=PTS-STARTPTS, scale=320x240 [upperright];
		[2:v] setpts=PTS-STARTPTS, scale=320x240 [lowerleft];
		[3:v] setpts=PTS-STARTPTS, scale=320x240 [lowerright];
		[base][upperleft] overlay=shortest=1 [tmp1];
		[tmp1][upperright] overlay=shortest=1:x=320 [tmp2];
		[tmp2][lowerleft] overlay=shortest=1:y=240 [tmp3];
		[tmp3][lowerright] overlay=shortest=1:x=320:y=240
	"
	-c:v libx264 output.mkv
"""

from __future__ import print_function, division, absolute_import

import os
import sys
import logging
import subprocess
import cv2
from RMS.Misc import mkdirP
import time

if sys.version_info[0] < 3:

    import urllib2

    # Fix Python 2 SSL certs
    try:
        import os, ssl
        if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
            getattr(ssl, '_create_unverified_context', None)): 
            ssl._create_default_https_context = ssl._create_unverified_context
    except:
        # Print the error
        print("Error: {}".format(sys.exc_info()[0]))

else:
    import urllib.request


import numpy as np
import requests
import tempfile
import datetime

# Import Cython functions
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
log = logging.getLogger("logger")

def downloadFiles(urls, station_id, working_dir=None, no_download=False, minimum_duration = 20):

    """
    Args:
        urls: a list of URLs to download
        station_id: a list of stationIDs to download, required for naming
        working_dir: the directory to keep files in, if not specified a temporary directory is generated

    Returns:
        working_dir: the directory which was used for working
        video_paths: paths to the downloaded videos
    """

    # either create a working directory, or use the area the user has specified
    if working_dir is None:
        working_dir = tempfile.mkdtemp()
    else:
        working_dir = os.path.expanduser(working_dir)

    # make sure working_dir exists
    mkdirP(working_dir)

    video_paths = []
    for video_url, stationID in zip(urls, station_id):
        file_name = "{:s}.mp4".format(stationID.lower())
        destination_file = os.path.join(working_dir, file_name)

        if no_download and os.path.isfile(destination_file):
            print("Local copy of file {:s} available, not downloading".format(destination_file))
            video_paths.append(destination_file)

        if no_download and not os.path.isfile(destination_file):
                print("Ignoring no_download directive because file for {} did not exist.".format(stationID.upper()))
                no_download = False

        if not no_download or not os.path.isfile(destination_file):

            retry = 0
            while retry < 10:
                temp_dir = tempfile.mktemp()
                mkdirP(temp_dir)
                print("Created directory {:s}".format(temp_dir))
                temp_download_destination_file = os.path.join(temp_dir,file_name)

                try:
                    video = None
                    video = requests.get(video_url, allow_redirects=True)
                    connection_good = True
                except:
                    print("No connection to {:s}".format(video_url))
                    connection_good = False

                if connection_good and video.status_code == 200:
                    print("Downloading {:s}".format(video_url), end="")
                    open(temp_download_destination_file,"wb").write(video.content)
                    video_duration = getVideoDurations([temp_download_destination_file])[0]
                    print(" - video duration is {:.1f} seconds".format(getVideoDurations([destination_file])[0]))
                    if video_duration < minimum_duration:
                        print("This video is short duration, only {:.1f} seconds".format(video_duration))
                        if os.path.exists(destination_file):
                            old_video_duration = getVideoDurations([temp_download_destination_file])[0]
                            if video_duration > old_video_duration:
                                print("However is longer than existing video {:.0f} seconds, so using this video"
                                      .format(old_video_duration))
                                print("Moving downloaded video from {:s} to {:s}"
                                      .format(temp_download_destination_file, destination_file))
                                os.replace(temp_download_destination_file, destination_file)
                                print("Removing directory {:s}".format(temp_dir))
                                os.rmdir(temp_dir)
                            else:
                                print("Keeping original file, which is duration {:.1f} seconds".format(video_duration))
                                print("Deleting {} and removing directory".format(temp_download_destination_file,
                                                                                         temp_dir))
                                os.unlink(temp_download_destination_file)
                                os.rmdir(temp_dir)
                    else:
                        print("Moving downloaded video from {:s} to {:s}".format(temp_download_destination_file,destination_file))
                        os.replace(temp_download_destination_file, destination_file)
                        print("Removing directory {:s}".format(temp_dir))
                        os.rmdir(temp_dir)
                    video_paths.append(destination_file)
                    break


                else:
                    if connection_good:
                        print("- No file found at {:s}, will retry".format(video_url))
                    time.sleep(6)
                    retry += 1
                    print("Removing directory {:s}".format(temp_dir))
                    os.rmdir(temp_dir)

            # if we did not get any connection, exit the loop
            if not connection_good:
                print("Did not get any connection to {:s} - relying on stored files".format(video_url))
                video_paths.append(destination_file)
                break

            if video.status_code != 200:
                print("No file found at {:s} after {:.0f} retries".format(video_url, retry))
                if os.path.exists(destination_file):
                    print("Local copy of {:s} available, continuing with local copy".format(destination_file))
                    video_paths.append(destination_file)
                else:
                    print("Quitting, because no local copy of {:s}, and not available from server".format(destination_file))
                    quit()

    return working_dir, video_paths

def getVideoDurations(paths_to_videos):

    """

    Args:
        paths_to_videos: list of paths to videos

    Returns:
        durations: list of durations in seconds
    """

    video_durations = []
    for path_to_video in paths_to_videos:
        video = cv2.VideoCapture(os.path.expanduser(path_to_video))
        frames,fps = video.get(cv2.CAP_PROP_FRAME_COUNT),video.get(cv2.CAP_PROP_FPS)
        video_durations.append(frames / fps)

    return video_durations

def getDurationCompensationFactors(durations, equalise_durations=True):

    """

    Args:
        durations: a list of durations

    Returns:
        compensation_factors: a list of compensation factors. factor < 1 means slow the video down
                              should never return a factor greater than 1
    """

    compensation_factors = []
    max_duration = max(durations)
    for duration in durations:
        if equalise_durations:
            compensation_factors.append(duration/max_duration)
        else:
            compensation_factors.append(1)
    return compensation_factors

def convertListOfStationIDsToListOfUrls(station_ids):

    """
        Args:
        station_id: a list of stationIDs

    Returns:
        list of urls pointing to the latest static video for that station
    
    e.g converts ["AU000A", "AU000C"] to
    ["https://globalmeteornetwork.org/weblog/AU/AU000A/static/AU000A_timelapse_static.mp4",
    "https://globalmeteornetwork.org/weblog/AU/AU000C/static/AU000C_timelapse_static.mp4"]
    """

    video_url_template = "https://globalmeteornetwork.org/weblog/{:s}/{:s}/static/{:s}_timelapse_static.mp4"
    video_urls = []
    for station in station_ids:
        country_code = station[0:2].upper()
        video_urls.append(video_url_template.format(country_code,station.upper(),station.upper()))

    return video_urls

def generateOutput(output_file, lib="libx264",print_nicely=False):

    """
    Generate the output clause of the ffmpeg statement

    Args:
        output_file: name of the output file to be used
        lib: generation library
        print_nicely: optionally include \n characters, generally for debugging purposes

    Returns:
        output_clause: the string which forms the output part of the ffmpeg statement
    """

    output_clause = " -c:v {} {}".format(lib, os.path.expanduser(output_file))
    output_clause += "\n " if print_nicely else " "
    return output_clause

def generateInputVideo(input_videos, tile_count, print_nicely=False):

    """
    Args:
        input_videos: list of input videos
        tile_count: the count of tiles to be generated
        print_nicely: optionally include \n characters, generally for debugging purposes

    Returns:
        the input clause for the ffmpeg statement
    """

    input,vid_count = "", 0
    for video in input_videos:
        input += "-i  {} ".format(video)
        vid_count += 1
        input += "\n " if print_nicely else " "
        if vid_count > tile_count:
            break

    return input

def generateFilter(duration_compensations, resolution_list, layout_list,print_nicely = False):

    """
    Args:
        video_paths: list of input video paths
        resolution_list: list of resolution  e.g.[x,y]
        layout_list: the list of layout e.g.[3,2]
        print_nicely: optionally include \n characters, generally for debugging purposes

    Returns:
        the filter section for the ffmpeg command
    """

    null_video = "nullsrc=size={}x{}[tmp_0]; ".format(resolution_list[0],resolution_list[1])

    res_tile = []

    res_tile.append(int(resolution_list[0] / layout_list[0]))
    res_tile.append(int(resolution_list[1] / layout_list[1]))

    video_counter,filter = 0, '-filter_complex " '
    filter += null_video
    filter += "\n " if print_nicely else " "
    for duration_compensation in duration_compensations:
        filter += ("[{}:v] setpts=PTS/{}-STARTPTS,scale={}x{}[tile_{}]; "
                   .format(video_counter,duration_compensation,res_tile[0],res_tile[1],video_counter))
        filter += "\n " if print_nicely else " "
        video_counter += 1
        if video_counter == layout_list[0] * layout_list[1]:
            break
    filter += "\n " if print_nicely else " "

    tile_count,x_pos,y_pos = 0,0,0
    for tile_down in range(layout_list[1]):
        for tile_across in range(layout_list[0]):

            filter += "[tmp_{}][tile_{}]overlay=shortest=1:x={}:y={}".format(tile_count,tile_count,x_pos,y_pos)
            tile_count += 1
            if tile_count != layout_list[0] * layout_list[1]:
                filter += "[tmp_{}] ; ".format(tile_count)
            else:
                filter += '" '
            x_pos += res_tile[0]
            filter += "\n " if print_nicely else " "
        x_pos = 0
        y_pos += res_tile[1]

    return filter


def generateCommand(video_paths, resolution, shape, output_filename = "~/mosaic_video.mp4", print_nicely=False,
                        equalise_durations = True):

    """
    Calls the input, filter and output commands and assembled the full ffmpeg command string
    for generating a video mosaic

    Args:
        video_paths: paths to the videos
        resolution: resolution for generated file e.g. [x,y]
        shape: the shape of the mosaic e.g. [3,2]
        output_filename: the output path and filename
        print_nicely: optionally include \n characters, generally for debugging purposes

    Returns:
        command string
    """

    durations = getVideoDurations(video_paths)
    duration_compensations = getDurationCompensationFactors(durations, equalise_durations=equalise_durations)

    ffmpeg_command_string = "ffmpeg -y -r 30  "
    ffmpeg_command_string += generateInputVideo(video_paths, shape[0] * shape[1],print_nicely=print_nicely)
    ffmpeg_command_string += generateFilter(duration_compensations,resolution,shape,print_nicely=print_nicely)
    ffmpeg_command_string += generateOutput(output_filename, print_nicely=print_nicely)

    return ffmpeg_command_string


def videoMosaic(station_ids, x_shape=2, y_shape=2, x_res=1280, y_res=720, equalise_durations=True,
                generate=True, output_file_path="~/mosaic_video.mp4", keep_files=False, working_directory=None,
                no_download=False, show_ffmpeg=False):

    """

    Args:
        station_ids: a list of stationIDs which have been requested to be downloaded, and optionally combined into a montage
        x_shape: number of tiles across e.g. 3
        y_shape: number of tiles down e.g. 2
        x_res: x resolution e.g. 1280
        y_res: y resolution e.g. 720
        generate: execute the command to generate the output
        output_file_path: file path for the generate file
        keep_files: keep the downloaded files
        working_directory: optional user specified directory for working, useful for downloading files

    Returns:
        [ffmpeg_command_string,working_directory]
    """

    if station_ids == None:
        return
    if len(station_ids) == 0:
        return
    if len(station_ids) < x_shape * y_shape:
        print("Too few stationIDs to create video of requested shape {:.f0} x {:.f0}".format(x_shape, y_shape))
        return

    if not working_directory is None and keep_files==False:
        print("Working directory specified therefore keeping files at end of work")
        keep_files=True

    url_list = convertListOfStationIDsToListOfUrls(station_ids)
    video_directory, input_video_paths = downloadFiles(url_list, station_ids, working_directory,
                                                       no_download, minimum_duration=minimum_duration)
    output_file_path = os.path.expanduser(output_file_path)
    ffmpeg_command_string = generateCommand(input_video_paths, [x_res, y_res],
                                            [x_shape, y_shape], output_file_path,
                                            equalise_durations = equalise_durations,
                                            print_nicely = True)

    if show_ffmpeg:
        print("ffmpeg command string \n {:s}".format(ffmpeg_command_string))
    if generate:
        generation_start_time = time.time()
        print("Video generation started at {:s}".format(
            datetime.datetime.fromtimestamp(generation_start_time).strftime('%Y-%m-%d %H:%M:%S')))


        subprocess.call(ffmpeg_command_string.replace("\n", " "),
                        shell=True, stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL )
        generation_end_time = time.time()
        generation_duration = generation_end_time - generation_start_time
        print("Video generation ended at {:s}, duration {:.0f} seconds".format(
            datetime.datetime.fromtimestamp(generation_end_time).strftime('%Y-%m-%d %H:%M:%S'),
                    generation_duration))
    if keep_files:
        print("Downloaded files in {:s}".format(working_directory))
    else:
        for input_video in input_video_paths:
            os.unlink(input_video)
        os.rmdir(video_directory)

    return ffmpeg_command_string, video_directory


def argumentHandler():

    def list_of_strings(arg):
        return arg.split(',')

    description = ""
    description += "Generate an n x n mosaic of videos. Minimum required to generate a video is\n"
    description += " python -m Utils.VideoMosaic \n\n"
    description += "A more comprehensive example is \n"
    description += " python -m Utils.VideoMosaic -c AU000U,AU000V,AU000W,AU000X,AU000Y,AU000Z -r -a -t 8"
    description += "3840 1440 -s 3 2 -o ~/station_video.mpg \n \n"
    description += "which creates a video of 6 cameras, resolution 3840 x 1440, 3 across, 2 down saved in users root as station_video.mpg\n"
    description += " -n inhibits generating the video and only prints the ffmpeg command string"
    description += " -k inhibits deletion of downloaded files"
    description += " -a displays on screen automatically and -t 8 sets a refresh time of 8 hours"
    description += "    press q while video is running to stop the program"

    arg_parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)


    arg_parser.add_argument('-c', '--cameras', metavar='CAMERAS', type=list_of_strings,
                            help="Cameras to use.")

    arg_parser.add_argument('-n', '--no_generate', dest='generate_video', default=True, action="store_false",
                            help="Generate the command string but do not execute")

    arg_parser.add_argument('-d', '--no_download', dest='no_download', default=False, action="store_true",
                            help="Do not download")


    arg_parser.add_argument('-k', '--keep_files', dest='keep_files', default=False, action="store_true",
                            help="Do not delete files at end")

    arg_parser.add_argument('-r', '--res', nargs=2, metavar='RESOLUTION', type=int,
                            help="outputresolution e.g 1280 720")

    arg_parser.add_argument('-s', '--shape', nargs=2, metavar='SHAPE', type=int,
                            help="Number of tiles across, number of tiles down e.g 4 3")

    arg_parser.add_argument('-o', '--output', nargs=1, metavar='OUTPUT', type=str,
                            help="Output filename")

    arg_parser.add_argument('-w', '--working_directory', metavar='WORKING', type=str,
                            help="Working directory to use")

    arg_parser.add_argument('-a', '--automatic', default=False,action="store_true",
                            help="Downloads files, displays on screen, refreshes every 24 hours")

    arg_parser.add_argument('-t', '--time', nargs=1, type=int,
                            help="Number of hours between refreshes, default 24")

    arg_parser.add_argument('-v', '--frame_duration', nargs=1, type=int,
                            help="Set the duration of each frame, default 40ms")

    arg_parser.add_argument('-m', '--minimum_duration', nargs=1, type=int,
                            help="Preferred minimum duration of video to use")

    arg_parser.add_argument('-f', '--show_ffmpeg', dest="show_ffmpeg", default=False, action="store_true",
                            help="Show the ffmpeg command")



    cml_args = arg_parser.parse_args()

    return cml_args




if __name__ == "__main__":

    import argparse

    cml_args = argumentHandler()

    default_camera_list = ["AU000A","AU000C","AU000D","AU000G"]
    cameras = cml_args.cameras if not cml_args.cameras is None else default_camera_list
    generate = cml_args.generate_video if not cml_args.generate_video is None else True
    output = cml_args.output[0] if not cml_args.output is None else "~/mosaic_video.mp4"
    keep_files = cml_args.keep_files if not cml_args.keep_files is None else False
    working_directory = cml_args.working_directory if not cml_args.working_directory is None else cml_args.working_directory
    cycle_hours = cml_args.time[0] if not cml_args.time == None else 24
    no_download = False if cml_args.no_download is None else cml_args.no_download
    frame_duration = 40 if cml_args.frame_duration is None else cml_args.frame_duration
    minimum_duration = 20 if cml_args.minimum_duration is None else cml_args.minimum_duration
    automatic_mode = cml_args.automatic if not cml_args.automatic is None else False
    show_ffmpeg = cml_args.show_ffmpeg if not cml_args.automatic is None else False

    cameras = [camera.upper() for camera in cameras]

    if not cml_args.shape is None:
        x_shape, y_shape = cml_args.shape[0], cml_args.shape[1]
    else:
        x_shape, y_shape = 2,2

    if not cml_args.res is None:
        x_res, y_res = cml_args.res[0], cml_args.res[1]
    else:
        x_res, y_res = 1280, 720


    run_count = 1
    exit_requested = False
    last_run_duration = cycle_hours * 3600
    last_target_run_duration = cycle_hours * 3600

    while run_count > 0 and exit_requested == False:

        this_start_time = time.time()

        target_run_duration = (last_target_run_duration - (last_run_duration - cycle_hours)) * 3600

        print("Start time / target end time  {:s} / {:s}".format(
            datetime.datetime.fromtimestamp(this_start_time).strftime('%Y-%m-%d %H:%M:%S'),
            datetime.datetime.fromtimestamp(this_start_time + target_run_duration).strftime('%Y-%m-%d %H:%M:%S')))


        videoMosaic(cameras, x_shape=x_shape, y_shape=y_shape, generate=generate, x_res=x_res, y_res=y_res,
                     output_file_path=output, keep_files=keep_files, working_directory=working_directory,
                    no_download=no_download, show_ffmpeg=show_ffmpeg)

        if automatic_mode:

            output = os.path.expanduser(output)
            interframe_wait_ms = 25

            #ref https://stackoverflow.com/questions/49949639/fullscreen-a-video-on-opencv

            # play the video
            window_name = "Video Player"
            cap = cv2.VideoCapture(output)
            if not cap.isOpened():
                print("Error: Could not open video.")
                exit()
            exit_requested = False
            while (target_run_duration > (time.time() - this_start_time)
                    and run_count > 0 and not exit_requested):
                print("Run duration target / elapsed {:.2f}/{:.2f} minutes"
                      .format(target_run_duration / 60, (time.time() - this_start_time) / 60))

                cap = cv2.VideoCapture(output)


                cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        #print("Reached end of video")
                        break

                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(interframe_wait_ms) & 0x7F == ord('q'):
                        print("Exit requested.")
                        exit_requested = True
                        break
            run_count -= 1
            run_count = 1 if automatic_mode and not exit_requested else 0
            last_run_duration = time.time() - this_start_time

            cap.release()
            cv2.destroyAllWindows()

        else:
            run_count = 0
        last_target_run_duration = target_run_duration