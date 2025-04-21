""" Generate an High Quality MP4 movie from FF files. 
    Contributors: Tioga Gulon
"""

from __future__ import print_function, division, absolute_import

import sys
import os
import platform
import argparse
import subprocess
import shutil
import cv2
import glob
import tarfile
from datetime import datetime

from PIL import ImageFont

from RMS.Formats.FFfile import read as readFF
from RMS.Formats.FFfile import validFFName, filenameToDatetime
from RMS.Misc import mkdirP, RmsDateTime



def generateTimelapse(dir_path, keep_images=False, fps=None, output_file=None, hires=False):
    """ Generate an High Quality MP4 movie from FF files. 
    
    Arguments:
        dir_path: [str] Path to the directory containing the FF files.

    Keyword arguments:
        keep_images: [bool] Keep the temporary images. False by default.
        fps: [int] Frames per second. 30 by default.
        output_file: [str] Output file name. If None, the file name will be the same as the directory name.
        hires: [bool] Make a higher resolution timelapse. False by default.
    """

    # Set the default FPS if not given
    if fps is None:
        fps = 30


    # Make the name of the output file if not given
    if output_file is None:
        mp4_path = os.path.join(dir_path, os.path.basename(dir_path) + ".mp4")

    else:
        mp4_path = os.path.join(dir_path, os.path.basename(output_file))


    # Set the CRF value for the video compression depending on the resolution
    if hires:
        crf = 25

    else:
        crf = 29


    t1 = RmsDateTime.utcnow()

    # Load the font for labeling
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 18)
    except:
        font = ImageFont.load_default()

    # Create temporary directory
    dir_tmp_path = os.path.join(dir_path, "temp_img_dir")

    if os.path.exists(dir_tmp_path):
        shutil.rmtree(dir_tmp_path)
        print("Deleted directory : " + dir_tmp_path)
		
    mkdirP(dir_tmp_path)
    print("Created directory : " + dir_tmp_path)
    
    print("Preparing files for the timelapse...")
    c = 0


    ff_list = [ff_name for ff_name in sorted(os.listdir(dir_path)) if validFFName(ff_name)]

    for file_name in ff_list:

        # Read the FF file
        ff = readFF(dir_path, file_name)

        # Skip the file if it could not be read
        if ff is None:
            continue

        # Get the timestamp from the FF name
        timestamp = filenameToDatetime(file_name).strftime("%Y-%m-%d %H:%M:%S")
		
        # Get id cam from the file name
            # e.g.  FF499_20170626_020520_353_0005120.bin
            # or FF_CA0001_20170626_020520_353_0005120.fits

        file_split = file_name.split('_')

        # Check the number of list elements, and the new fits format has one more underscore
        i = 0
        if len(file_split[0]) == 2:
            i = 1
        camid = file_split[i]

        # Make a filename for the image, continuous count %04d
        img_file_name = 'temp_{:04d}.jpg'.format(c)

        img = ff.maxpixel

        # Draw text to image
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = camid + " " + timestamp + " UTC"
        cv2.putText(img, text, (10, ff.nrows - 6), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # Save the labelled image to disk
        cv2.imwrite(os.path.join(dir_tmp_path, img_file_name), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    
        c = c + 1

        # Print elapsed time
        if c % 30 == 0:
            print("{:>5d}/{:>5d}, Elapsed: {:s}".format(c + 1, len(ff_list), \
                str(RmsDateTime.utcnow() - t1)), end="\r")
            sys.stdout.flush()


    # If running on Linux, use avconv if available
    if platform.system() == 'Linux':

        # If avconv is not found, try using ffmpeg. In case of using ffmpeg,
        # use parameter -nostdin to avoid it being stuck waiting for user input
        software_name = "avconv"
        nostdin = ""
        print("Checking if avconv is available...")
        if os.system(software_name + " --help > /dev/null"):
            software_name = "ffmpeg"
            nostdin =  " -nostdin "
        
        # Construct the command for avconv            
        temp_img_path = os.path.basename(dir_tmp_path) + os.sep + "temp_%04d.jpg"
        com = "cd " + dir_path + ";" \
            + software_name + nostdin + " -v quiet -r "+ str(fps) +" -y -i " + temp_img_path \
            + " -vcodec libx264 -pix_fmt yuv420p -crf " + str(crf) \
            + " -movflags faststart -threads 2 -g 15 -vf \"hqdn3d=4:3:6:4.5,lutyuv=y=gammaval(0.77)\" " \
            + mp4_path

        print("Creating timelapse using {:s}...".format(software_name))
        print(com)
        subprocess.call([com], shell=True)


    # If running on Windows, use ffmpeg.exe
    elif platform.system() == 'Windows':
	
        # ffmpeg.exe path
        root = os.path.dirname(__file__)
        ffmpeg_path = os.path.join(root, "ffmpeg.exe")
	
        # Construct the ecommand for ffmpeg
        temp_img_path = os.path.join(os.path.basename(dir_tmp_path), "temp_%04d.jpg")
        com = ffmpeg_path + " -v quiet -r " + str(fps) + " -i " + temp_img_path \
            + " -c:v libx264 -pix_fmt yuv420p -an -crf " + str(crf) \
            + " -g 15 -vf \"hqdn3d=4:3:6:4.5,lutyuv=y=gammaval(0.77)\" -movflags faststart -y " \
            + mp4_path
		
        print("Creating timelapse using ffmpeg...")
        print(com)
        subprocess.call(com, shell=True, cwd=dir_path)
		
    else :
        print ("generateTimelapse only works on Linux or Windows the video could not be encoded")

    #Delete temporary directory and files inside
    if os.path.exists(dir_tmp_path) and not keep_images:
        shutil.rmtree(dir_tmp_path)
        print("Deleted temporary directory : " + dir_tmp_path)
		
    print("Total time:", RmsDateTime.utcnow() - t1)


def generateTimelapseFromFrames(day_dir, video_path, fps=30, crf=27, cleanup_mode='none', compression='bz2'):
    """
    Generate a timelapse video from frame images and optionally cleanup the
    source directory.

    Keyword arguments:
        day_dir: [str] Directory containing a day of frame image files. day_dir is expected to have
                       subdirectories by the hour of day "00", "01", ..., "23"
        video_path: [str] Output path for the generated video.
        fps: [int] Frames per second for the output video. 30 by default.
        crf: [int] Constant Rate Factor for video compression. 26 by default.
        cleanup_mode: [str] Cleanup mode after video creation.
                      Options: 'none', 'delete', 'tar'. 'none' by default.
        compression: [str] Compression method for tar.
                     Options: 'bz2', 'gz'. 'bz2' by default.
    """

    # Create temporary directory for timestamped images
    # Remove any existing such directory so they don't get listed in image_files
    raw_dir_tmp_path = os.path.join(day_dir, "temp_raw_img_dir")

    if os.path.exists(raw_dir_tmp_path):
        shutil.rmtree(raw_dir_tmp_path)
        print("Deleted directory : " + raw_dir_tmp_path)

    mkdirP(raw_dir_tmp_path)
    print("Created directory : " + raw_dir_tmp_path)
    print("Preparing files for the timelapse...")

    # Combine all required files paths
    image_files = [img for img in sorted(glob.glob(os.path.join(day_dir, "*/*.jpg")))] + \
                  [img for img in sorted(glob.glob(os.path.join(day_dir, "*/*.png")))]

    if len(image_files) == 0:
        print("No images found.")
        return

    # Start timestamping overlay on images
    for index, img_path in enumerate(image_files):

        # Setup timestamp text from filename (img_path example : 'XX0001_20241123_100015_442')
        image_name_parts = os.path.basename(img_path).split('_')
        station_id = image_name_parts[0]
        extracted_time = datetime.strptime(image_name_parts[1] + image_name_parts[2], "%Y%m%d%H%M%S")
        
        # Draw timestamp to image
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        text = station_id + " " + extracted_time.strftime("%Y-%m-%d %H:%M:%S") + " UTC"
        position = (10, image.shape[0] - 10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, text, position, font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # Save the labelled image to disk
        timestamped_img_path = os.path.join(raw_dir_tmp_path, 'temp_{:05d}.jpg'.format(index))
        cv2.imwrite(timestamped_img_path, image, [cv2.IMWRITE_JPEG_QUALITY, 100])

        # Update the new frame path in the image_file list
        image_files[index] = timestamped_img_path


    # Create a text file listing all the images
    list_file_path = os.path.join(day_dir, "filelist.txt")
    with open(list_file_path, 'w') as f:
        for img_path in image_files:
            f.write("file '{0}'\n".format(img_path))

    if platform.system() in ['Linux', 'Darwin']:  # Darwin is macOS
        software_name = "ffmpeg"
    elif platform.system() == 'Windows':
        software_name = os.path.join(os.path.dirname(__file__), "ffmpeg.exe")
    else:
        print("Unsupported platform.")
        return

    # Formulate the ffmpeg command
    encode_command = [
        software_name, "-nostdin", "-f", "concat", "-safe", "0", "-v", "quiet",
        "-r", str(fps), "-y", "-i", list_file_path, "-c:v", "libx264",
        "-crf", str(crf), "-threads", "2", "-g", "15", video_path
    ]

    # Execute the command
    print("Creating timelapse using ffmpeg...")
    subprocess.call(encode_command)

    # Avoid archiving raw frame temp directory
    if os.path.exists(raw_dir_tmp_path):
        shutil.rmtree(raw_dir_tmp_path)
        print("Deleted directory : " + raw_dir_tmp_path)

    # Cleanup process
    if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
        print("Video created successfully at {0}".format(video_path))
        # Cleanup based on the specified mode
        if cleanup_mode == 'delete':
            try:
                shutil.rmtree(day_dir)
                print("Successfully deleted the source directory: {0}".format(day_dir))
            except Exception as e:
                print("Error deleting the source directory: {0}".format(e))

        elif cleanup_mode == 'tar':
            try:
                ext = '.tar.bz2' if compression == 'bz2' else '.tar.gz'

                # Derive the tar file name from video_path, stripping '_timelapse.mp4'
                base_name = os.path.basename(video_path).replace('_timelapse.mp4', '')

                # Construct the full path for the tar file
                tar_path = os.path.join(os.path.dirname(video_path), base_name + ext)

                # Create the tar archive with the correct mode
                mode = 'w:bz2' if compression == 'bz2' else 'w:gz'

                with tarfile.open(tar_path, mode) as tar:
                    tar.add(day_dir, arcname=os.path.basename(day_dir))

                # Remove the original directory
                shutil.rmtree(day_dir)
                print("Successfully created tar archive at: {0}".format(tar_path))

            except Exception as e:
                print("Error creating tar archive: {0}".format(e))
    else:
        print("Video creation failed or resulted in an empty file.")



if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Convert all FF files in a folder to a movie")

    arg_parser.add_argument('dir_path', metavar='DIR_PATH', type=str, \
        help='Path to directory with FF files.')

    arg_parser.add_argument('-fps', '--fps', metavar='FPS', type=int, \
        help='FPS to use for video.')

    arg_parser.add_argument('-x', '--nodel', action="store_true", \
        help="""Do not delete generated JPG file.""")
    
    arg_parser.add_argument('-o', '--output', metavar='OUTPUT', type=str, \
        help='Output video file name.')
    
    arg_parser.add_argument('--hires', action="store_true", \
                            help='Make a higher resolution timelapse. The video file will be larger.')

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################


    dir_path = os.path.normpath(cml_args.dir_path)

    # Generate the timelapse
    generateTimelapse(dir_path, keep_images=cml_args.nodel, fps=cml_args.fps, output_file=cml_args.output, hires=cml_args.hires)
