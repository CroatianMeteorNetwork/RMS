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
from RMS.Misc import mkdirP

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

# Import Cython functions
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
log = logging.getLogger("logger")

def downloadFilesToTmp(urls, station_id, working_dir=None):

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
        print("Downloading from URL {:s}".format(video_url))
        video = requests.get(video_url, allow_redirects=True)
        destination_file = os.path.join(working_dir, "{:s}.mp4".format(stationID.lower()))
        open(destination_file,"wb").write(video.content)
        video_paths.append(destination_file)

    return working_dir, video_paths


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
        country_code = station[0:2]
        video_urls.append(video_url_template.format(country_code,station,station))

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

    output_clause = " -c:v {} {}".format(lib,output_file)
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

def generateFilter(video_paths, resolution_list, layout_list,print_nicely = False):

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
    print(null_video)
    res_tile = []
    res_tile.append(int(resolution_list[0] / layout_list[0]))
    res_tile.append(int(resolution_list[1] / layout_list[1]))

    video_counter,filter = 0, '-filter_complex " '
    filter += null_video
    filter += "\n " if print_nicely else " "
    for video in video_paths:
        filter += "[{}:v] setpts=PTS-STARTPTS,scale={}x{}[tile_{}]; ".format(video_counter,res_tile[0],res_tile[1],video_counter)
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


def generateCommand(video_paths, resolution, shape, output_filename = "~/mosaic_video.mp4", print_nicely=False):

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

    output_filename = os.path.expanduser(output_filename)
    ffmpeg_command_string = "ffmpeg -y -r 30 "
    ffmpeg_command_string += generateInputVideo(video_paths, shape[0] * shape[1],print_nicely=print_nicely)
    ffmpeg_command_string += generateFilter(video_paths,resolution,shape,print_nicely=print_nicely)
    ffmpeg_command_string += generateOutput(output_filename, print_nicely=print_nicely)

    return ffmpeg_command_string


def videoMosaic(stationIDs, x_shape=2, y_shape=2, x_res=1280, y_res=720,
                generate=True, output_file_path="~/mosaic_video.mp4", keep_files=False, working_directory=None):

    """

    Args:
        stationIDs: a list of stationIDs which have been requested to be downloaded, and optionally combined into a montage
        x_shape: number of tiles across e.g. 3
        y_shape: number of tiles down e.g. 2
        x_res: x resolution e.g. 1280
        y_res: y resolution e.g. 720
        generate: execute the command to generate the output
        output_file_path: file path for the generate file
        keep_files: keep the files downloaded into the temporary directory
        working_directory: optional user specified directory for working, useful for downloading files

    Returns:
        [ffmpeg_command_string,working_directory]
    """

    if stationIDs == None:
        return
    if len(stationIDs) == 0:
        return
    if len(stationIDs) < x_shape * y_shape:
        print("Too few stationIDs to create video of requested shape {:.f0} x {:.f0}".format(x_shape, y_shape))
        return

    if not working_directory is None and keep_files==False:
        print("user has specified a directory, keeping files")
        keep_files=True

    url_list = convertListOfStationIDsToListOfUrls(stationIDs)
    video_directory, input_video_paths = downloadFilesToTmp(url_list, stationIDs, working_directory)
    output_file_path = os.path.expanduser(output_file_path)
    ffmpeg_command_string = generateCommand(input_video_paths, [x_res, y_res],
                                            [x_shape, y_shape], output_file_path)
    if generate:
        subprocess.call(ffmpeg_command_string.replace("\n", " "), shell=True)
    if keep_files:
        print("Downloaded files in {:s}".format(working_directory))
    else:
        for input_video in input_video_paths:
            os.unlink(input_video)
        os.rmdir(video_directory)

    return ffmpeg_command_string, video_directory


if __name__ == "__main__":

    import argparse

    def list_of_strings(arg):
        return arg.split(',')

    description = ""
    description += "Generate an n x n mosaic of videos. Minimum required to generate a video is\n"
    description += " python -m Utils.VideoMosaic \n\n"
    description += "A more comprehensive example is \n"
    description += " python -m Utils.VideoMosaic -c AU000U,AU000V,AU000W,AU000X,AU000Y,AU000Z -r "
    description += "3840 1440 -s 3 2 -o ~/station_video.mpg \n \n"
    description += "which creates a video of 6 cameras, resolution 3840 x 1440, 3 across, 2 down saved in users root as station_video.mpg\n"
    description += " -n inhibits generating the video and only prints the ffmpeg command string"
    description += " -k inhibits deletion of downloaded files"

    arg_parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)


    arg_parser.add_argument('-c', '--cameras', metavar='CAMERAS', type=list_of_strings,
                            help="Cameras to use.")

    arg_parser.add_argument('-n', '--no_generate', dest='generate_video', default=True, action="store_false",
                            help="Generate the command string but do not execute")

    arg_parser.add_argument('-k', '--keep_files', dest='keep_files', default=False, action="store_true",
                            help="Do not delete files at end")

    arg_parser.add_argument('-r', '--resolution', nargs=2, metavar='RESOLUTION', type=int,
                            help="outputresolution e.g 1280 720")

    arg_parser.add_argument('-s', '--shape', nargs=2, metavar='SHAPE', type=int,
                            help="Number of tiles across, number of tiles down e.g 4 3")

    arg_parser.add_argument('-o', '--output', nargs=1, metavar='OUTPUT', type=str,
                            help="Output filename")

    arg_parser.add_argument('-w', '--working_directory', metavar='WORKING', type=str,
                            help="Working directory to use")

    cml_args = arg_parser.parse_args()

    if not cml_args.cameras is None:
        cameras = cml_args.cameras
    else:
        cameras = ["AU000A","AU000C","AU000D","AU000G"]

    if not cml_args.shape is None:
        x_shape,y_shape = cml_args.shape[0], cml_args.shape[1]
    else:
        x_shape, y_shape = 2,2

    if not cml_args.resolution is None:
        x_res,y_res = cml_args.resolution[0], cml_args.resolution[1]
    else:
        x_res,y_res = 1280,720

    if not cml_args.generate_video is None:
        generate = cml_args.generate_video
    else:
        generate = True

    if not cml_args.output is None:
        output = cml_args.output[0]
    else:
        output = "~/mosaic_video.mp4"

    if not cml_args.keep_files is None:
        keep_files = cml_args.keep_files
    else:
        keep_files = False

    if not cml_args.working_directory is None:
        working_directory = cml_args.working_directory
        print("Working in {}".format(working_directory))

    # do the work
    print(videoMosaic(cameras, x_shape=x_shape, y_shape=y_shape, generate=generate, x_res=x_res, y_res=y_res,
                      output_file_path=output, keep_files=keep_files, working_directory=working_directory)[0])

