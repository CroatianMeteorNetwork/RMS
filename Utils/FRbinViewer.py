""" Showing fireball detections from FR bin files. """

# RPi Meteor Station
# Copyright (C) 2017  Dario Zubovic, Denis Vida
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

from __future__ import print_function, absolute_import, division

import os
import sys
import argparse
import platform
import subprocess
import time

import cv2
import numpy as np

import RMS.ConfigReader as cr
from RMS.Formats import FFfile, FRbin
import datetime
from glob import glob
from Utils.ShowerAssociation import showerAssociation
from RMS.Formats.FTPdetectinfo import validDefaultFTPdetectinfo
from RMS.Astrometry.Conversions import datetime2JD


class FrameDisplay:
    """Cross-platform frame display helper with OpenCV/Matplotlib backends."""

    def __init__(self, window_name):
        # Remember the logical window name for whichever backend we end up using.
        self.window_name = window_name

        # Default to OpenCV until we prove that GUI support is not available.
        self.backend = 'cv2'

        # Track backend-specific state so we can tear down windows cleanly.
        self._cv2_window_created = False
        self._mpl_ready = False

        # Keyboard state is reused for both backends.
        self._last_key = None

        # Matplotlib drawing artefacts are initialised lazily when required.
        self._mpl_im = None
        self._mpl_fig = None
        self._mpl_ax = None
        self._plt = None

    def _ensureCv2Window(self):
        """Create an OpenCV window if GUI support is available."""

        # If we already created the window during a previous call, there is nothing else to do.
        if self._cv2_window_created:
            return True

        # Try to create an OpenCV window. This step fails on systems where OpenCV was compiled
        # without GUI support (e.g. headless servers), so we guard it with a try/except.
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self._cv2_window_created = True
            return True
        except cv2.error:
            return False

    def _switchToMatplotlib(self):
        """Fallback initialisation using Matplotlib if OpenCV fails."""

        # When we have already switched to Matplotlib earlier, we can immediately reuse it.
        if self.backend == 'mpl':
            return True

        # Import Matplotlib lazily so that we only pay the import cost when OpenCV GUI support is
        # missing. Import failures mean that we cannot display frames at all.
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return False

        # Activate the Matplotlib backend in interactive mode.
        self.backend = 'mpl'
        self._plt = plt
        plt.ion()

        # Build a simple figure for image display. The result mimics the old OpenCV window so the
        # rest of the viewer logic does not have to change depending on the backend.
        self._mpl_fig, self._mpl_ax = plt.subplots()

        try:
            self._mpl_fig.canvas.manager.set_window_title(self.window_name)
        except Exception:
            # Some Matplotlib backends do not support named windows.
            pass

        # Remove spines/ticks so the image occupies the entire canvas.
        self._mpl_ax.set_axis_off()

        # Capture keyboard events so the rest of the viewer behaves the same.
        self._mpl_fig.canvas.mpl_connect('key_press_event', self._onKeyPress)
        self._mpl_ready = True

        # Close any half-created OpenCV window to avoid zombie handles. This may happen when OpenCV
        # successfully creates a window but rendering fails afterwards.
        if self._cv2_window_created:
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                pass
            self._cv2_window_created = False

        print("OpenCV imshow unavailable; using Matplotlib for display.")
        return True

    def _onKeyPress(self, event):
        """Store the last key pressed in the Matplotlib window."""

        self._last_key = event.key

    def showFrame(self, image):
        """Display a frame using the currently selected backend."""

        # Attempt to show the frame using OpenCV first because that matches the historic viewer.
        if self.backend == 'cv2':
            # Make sure an OpenCV window exists. If creation fails we immediately switch to
            # Matplotlib and retry the same frame.
            if not self._ensureCv2Window():
                if not self._switchToMatplotlib():
                    return False

                return self.showFrame(image)

            try:
                # Regular GUI-enabled builds reach this line and render the frame directly.
                cv2.imshow(self.window_name, image)
                return True
            except cv2.error:
                # Some builds create windows but fail to render frames (e.g. missing GTK backend).
                # In that case we fall back to Matplotlib and retry so playback can continue.
                if not self._switchToMatplotlib():
                    return False

                return self.showFrame(image)

        # Matplotlib initialisation failed earlier and cannot be recovered.
        if not self._mpl_ready and not self._switchToMatplotlib():
            return False

        # Convert BGR images to RGB because Matplotlib expects the latter, otherwise the colours
        # will look swapped. Grayscale frames can be rendered as-is.
        display_data = image
        if image.ndim == 3 and image.shape[2] == 3:
            display_data = image[:, :, ::-1]

        # First Matplotlib draw creates the image artist. Subsequent draws only update the data,
        # which is cheaper and avoids flicker.
        if self._mpl_im is None:
            kwargs = {}
            if display_data.ndim == 2 or (display_data.ndim == 3 and display_data.shape[2] == 1):
                kwargs['cmap'] = 'gray'
                kwargs['vmin'] = 0
                kwargs['vmax'] = 255

            self._mpl_im = self._mpl_ax.imshow(display_data, **kwargs)
        else:
            # Update the existing artist to reflect the new frame without rebuilding the figure.
            self._mpl_im.set_data(display_data)

        # Flush the Matplotlib event queue so window contents update on screen.
        self._mpl_fig.canvas.draw_idle()
        self._plt.pause(0.001)
        return True

    def waitKey(self, delay_ms):
        """Wait for keyboard input matching the behaviour of cv2.waitKey."""

        # When OpenCV owns the window we simply forward to cv2.waitKey so native shortcuts work.
        if self.backend == 'cv2':
            return cv2.waitKey(delay_ms) & 0xFF

        # Matplotlib is not initialised, which means the viewer is effectively headless.
        if not self._mpl_ready:
            return -1

        self._last_key = None

        # Match cv2.waitKey semantics: delay 0 waits indefinitely, otherwise we poll until the
        # timeout expires. We keep the pause interval small so keyboard input feels responsive.
        if delay_ms == 0:
            while self._last_key is None:
                self._plt.pause(0.05)
        else:
            end_time = time.time() + (delay_ms / 1000.0)
            while time.time() < end_time and self._last_key is None:
                remaining = end_time - time.time()
                self._plt.pause(min(0.05, remaining))

        key = self._last_key
        self._last_key = None

        # Matplotlib supplies keys as strings. For single-character inputs we convert them to ASCII
        # codes so the rest of the viewer can stay unchanged.
        if not key:
            return -1

        if len(key) == 1:
            return ord(key)

        return -1

    def moveWindow(self, x_coord, y_coord):
        """Relocate the active window so behaviour matches the old viewer."""

        # If OpenCV created the window we use the native moveWindow helper to position it. Some
        # headless builds raise errors here as well, so we catch them to keep playback running.
        if self.backend == 'cv2' and self._cv2_window_created:
            try:
                cv2.moveWindow(self.window_name, x_coord, y_coord)
            except cv2.error:
                pass
        # Matplotlib does not expose a cross-backend API for moving windows. Instead we reach into
        # the manager and, when the underlying toolkit provides a window object, we apply the
        # geometry change. Backends without such access simply ignore the request.
        elif self.backend == 'mpl' and self._mpl_ready:
            try:
                manager = self._mpl_fig.canvas.manager
                window = getattr(manager, 'window', None)
                if window is not None:
                    window.wm_geometry(f"+{x_coord}+{y_coord}")
            except Exception:
                pass

    def destroyWindow(self):
        """Destroy whichever window type is currently active."""

        # Release native OpenCV windows if they were created so we do not leave zombie GUI handles
        # behind when the viewer exits or switches to Matplotlib.
        if self.backend == 'cv2' and self._cv2_window_created:
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                pass

            self._cv2_window_created = False
        # When Matplotlib drives the display we close the figure to unblock the event loop and free
        # resources. Closing a non-initialised figure would raise, so we only do this when ready.
        elif self.backend == 'mpl' and self._mpl_ready:
            self._plt.close(self._mpl_fig)
            self._mpl_ready = False

def view(dir_path, ff_path, fr_path, config, save_frames=False, extract_format=None, hide=False,
        avg_background=False, split=False, add_timestamp=False, add_frame_number=False, append_ff_to_video=False,
         add_shower_name=False, associations={}):
    """ Shows the detected fireball stored in the FR file. 
    
    Arguments:
        dir_path: [str] Current directory.
        ff: [str] path to the FF bin file
        fr: [str] path to the FR bin file
        config: [conf object] configuration structure

    Keyword arguments:
        save_frames: [bool] Save FR frames to disk. False by default.
        extract_format: [str] Format of saved images. E.g. png, jpg, mp4.
        hide: [bool] Don't show frames on the screen.
        avg_background: [bool] Avepixel as background. False by default, in which case the maxpixel will be
            used.
        split: [bool] Split the video into multiple videos, one for each line. False by default.
        add_timestamp: [bool] Add timestamp to the image. False by default.
        add_frame_number: [bool] Add frame number to the image. False by default.
        append_ff_to_video: [bool] Append image with meteor to video

    """

    if extract_format is None:
        extract_format = 'png'
    
    name = fr_path
    fr = FRbin.read(dir_path, fr_path)

    print('------------------------')
    print('Showing file:', fr_path)


    if ff_path is None:
        #background = np.zeros((config.height, config.width), np.uint8)

        # Get the maximum extent of meteor frames
        y_size = max([max(np.array(fr.yc[i]) + np.array(fr.size[i])//2) for i in range(fr.lines)])
        x_size = max([max(np.array(fr.xc[i]) + np.array(fr.size[i])//2) for i in range(fr.lines)])

        # Make the image square
        img_size = max(y_size, x_size)

        background = np.zeros((img_size, img_size), np.uint8)
        add_timestamp = False
        add_shower_name = False

    else:
        ff_file = FFfile.read(dir_path, ff_path)
        if avg_background:
            background = ff_file.avepixel
        else:
            background = ff_file.maxpixel
        if append_ff_to_video:
            meteor_image = np.copy(ff_file.maxpixel)
        if add_timestamp:
            station_name, file_timestamp = getStationNameAndTimestampfromFile(ff_path)

    print("Number of lines:", fr.lines)
    
    # Track whether the first frame has already been positioned on screen.
    first_image = True
    wait_time = 2*int(1000.0/config.fps)

    # Viewer state for interactive playback.
    pause_flag = False
    display = None

    # Prepare the display helper unless the caller explicitly hid the window.
    if not hide:
        try:
            display = FrameDisplay(name)
        except RuntimeError as exc:
            print(str(exc))
            hide = True

    # if the file format was mp4, lets make a video from the data
    makevideo = False
    if extract_format == 'mp4':
        makevideo = True
        extract_format = 'png'

    videos = []
    
    if split: # legacy mode, one video per line
        for current_line in range(fr.lines):
            frames = []
            for z in range(fr.frameNum[current_line]):
                frames.append([(current_line, z)])
            videos.append(frames)
    else: # regular mode, sort frames in time producing single video
        clips = dict()
        start = 10000
        end = -1
        
        for current_line in range(fr.lines):
            for z in range(fr.frameNum[current_line]):
                t = fr.t[current_line][z]
                if not t in clips:
                    clips[t] = []
                clips[t].append((current_line, z))
                start = min(start, t)
                end = max(end, t)

        videos.append([])
        
        for t in range(start, end + 1):
            if t in clips:
                videos[0].append(clips[t])

    video_num = 0
    for video in videos:

        print('Frame,  Y ,  X , size')
        framefiles=[] # array to hold names of frames for later deletion

        frame_num = 0

        # Track the first frame
        first_frame = np.inf
        # calculate shower name
        showerNameTitle = ""
        if add_shower_name:
            showerNameTitle = getMeteorShowerTitle(video, fr, ff_path, associations, config.fps)

        for frame in video:

            img = np.copy(background)

            for current_line, z in frame:
                
                # Get the center position of the detection on the current frame
                yc = fr.yc[current_line][z]
                xc = fr.xc[current_line][z]

                # Get the frame number
                t = fr.t[current_line][z]

                # Get the size of the window
                size = fr.size[current_line][z]
                
                print("  {:3d}, {:3d}, {:3d}, {:d}".format(t, yc, xc, size))

                # Set the first frame
                if t < first_frame:
                    first_frame = t
                
                # Paste the frames onto the big image
                y_img = np.arange(yc - size//2, yc + size//2)
                x_img = np.arange(xc - size//2, xc + size//2)

                Y_img, X_img = np.meshgrid(y_img, x_img)

                y_frame = np.arange(len(y_img))
                x_frame = np.arange(len(x_img))

                Y_frame, X_frame = np.meshgrid(y_frame, x_frame)                

                img[Y_img, X_img] = fr.frames[current_line][z][Y_frame, X_frame]

            # Add frame number
            if add_frame_number:

                # Put the name of the FR file, followed by the frame number
                # Put a black shadow
                title = fr_path + " frame = {:3d}".format(t)
                addTextToImage(img, title, 10, 20)

            # Add timestamp
            if add_timestamp:
                addTimestampToImage(img, getTimestampTitle(station_name, file_timestamp, t, config.fps))
            # Add meteor shower name
            if add_shower_name:
                addShowerNameToImage(add_timestamp, img, showerNameTitle)

            # Save frame to disk
            if save_frames or makevideo:
                frame_file_name = fr_path.replace('.bin', '') \
                    + "_line_{:02d}_frame_{:03d}.{:s}".format(video_num, frame_num, extract_format)
                cv2.imwrite(os.path.join(dir_path, frame_file_name), img)
                framefiles.append(frame_file_name)
                img_patt = os.path.join(dir_path, fr_path.replace('.bin', '')
                    + "_line_{:02d}_frame_%03d.{:s}".format(video_num, extract_format))

            frame_num += 1

            if not hide:

                # Resize large frames so they fit comfortably on screen.
                display_image = resizeImageIfNeed(img)
                # Abort preview if the backend fails so we can continue processing.
                if not display or not display.showFrame(display_image):
                    if display:
                        display.destroyWindow()
                        display = None
                    print("Unable to display frames; continuing without preview.")
                    hide = True
                    first_image = False
                    continue

                # If this is the first image, move it to the upper left corner
                if first_image:
                    display.moveWindow(0, 0)
                    first_image = False


                if pause_flag:
                    wait_time = 0
                else:
                    wait_time = 2*int(1000.0/config.fps)

                # Space key: pause display.
                # 1: previous file.
                # 2: next line.
                # q: Quit.
                key = display.waitKey(wait_time)

                if key == ord("1"):
                    if display:
                        display.destroyWindow()
                    return -1

                elif key == ord("2"): 
                    break

                elif key == ord(" "): 
                    
                    # Pause/unpause video
                    pause_flag = not pause_flag

                elif key == ord("q"):
                    os._exit(0)


        if makevideo is True:
            if append_ff_to_video and ff_path is not None:
                # add duration of 1.5 sec
                frameCount = int(config.fps * 1.5)
                timestamp_title = ""
                if add_timestamp:
                    timestamp_title = getTimestampTitle(station_name, file_timestamp, 0, config.fps)
                saveFramesForMeteorImage(meteor_image, fr_path, add_timestamp, frame_num - 1, frameCount, video_num,
                                         extract_format, framefiles, dir_path, add_shower_name, timestamp_title, showerNameTitle)

            root = os.path.dirname(__file__)
            ffmpeg_path = os.path.join(root, "ffmpeg.exe")
            
            mp4_path = os.path.join(dir_path, fr_path.replace('.bin', '') + '_line_{:02d}.mp4'.format(video_num))

            # If running on Windows, use ffmpeg.exe
            if platform.system() == 'Windows':
                com = ffmpeg_path + " -y -f image2 -pattern_type sequence -framerate " + str(config.fps) + " -start_number " + str(0) + " -i " + img_patt +" " + mp4_path
                

            else:
                software_name = "avconv"
                if os.system(software_name + " --help > /dev/null"):
                    software_name = "ffmpeg"
                    # Construct the ecommand for ffmpeg           
                    com = software_name + " -y -f image2 -pattern_type sequence -framerate " + str(config.fps) + " -start_number " + str(0) + " -i " + img_patt +" -pix_fmt yuv420p " + mp4_path
                else:
                    com = "cd " + dir_path + ";" \
                        + software_name + " -v quiet -r 30 -y -start_number " + str(0) + " -i " + img_patt \
                        + " -vcodec libx264 -pix_fmt yuv420p -crf 25 -movflags faststart -g 15 -vf \"hqdn3d=4:3:6:4.5,lutyuv=y=gammaval(0.97)\" " \
                        + mp4_path
            
            # Print the command
            print("Command:")
            print(com)

            # Run the command
            subprocess.call(com, shell=True, cwd=dir_path)

            # Delete frames unless the user specified to keep them
            if not save_frames:
                for frame in framefiles:
                    os.remove(os.path.join(dir_path, frame))

        video_num += 1

    if not hide and display:
        display.destroyWindow()


def saveFramesForMeteorImage(meteorImage, frPath, addTimestamp, lastFrameNumber, frameCount, videoNumber,
                             format,
                             frameFiles, folder, addShowerName, timestampTitle, showerNameTitle):
    # Add timestamp
    if addTimestamp:
        addTimestampToImage(meteorImage, timestampTitle)
    # Add meteor shower name
    if addShowerName:
        addShowerNameToImage(addTimestamp, meteorImage, showerNameTitle)
    # append frames for 1.5 second
    for frameNumber in range(frameCount):
        frameFileName = frPath.replace('.bin', '') \
                        + "_line_{:02d}_frame_{:03d}.{:s}".format(videoNumber, lastFrameNumber + frameNumber + 1,
                                                                  format)
        cv2.imwrite(os.path.join(folder, frameFileName), meteorImage)
        frameFiles.append(frameFileName)


def addTimestampToImage(image, title):
    height = image.shape[0]
    addTextToImage(image, title, 15, height - 20)


def addShowerNameToImage(withTimestamp, image, title):
    height = image.shape[0]
    # if no timestamp - move to the left
    if withTimestamp:
        addTextToImage(image, title, 340, height - 20)
    else:
        addTextToImage(image, title, 15, height - 20)


def getMeteorShowerTitle(video, frFile, ffPath, associations, fps):
    title = "Meteor shower : Unknown"
    # use time offset 100mls (~2 frames) in case if video capture started later or sopped earlier
    mls = 100
    # convert seconds to days
    timeOffset = mls/1000/(24*60*60)
    # get start and end time of video
    frFrameTimeStart, frFrameTimeEnd = getVideoStartAndEndTime(video, frFile, ffPath, fps)
    # get all available meteors for current FF file
    fileName = os.path.basename(ffPath)
    meteorsForFile = [key for key in associations if key[0].startswith(fileName)]
    # search first suitable by time range
    for meteor in meteorsForFile:
        frameTimes = associations[meteor][0].jd_array
        meteorTimeStart = frameTimes[0]
        meteorTimeEnd = frameTimes[-1]
        # meteor time should be inside video time +- 100 mls for error
        if meteorTimeStart >= frFrameTimeStart - timeOffset and meteorTimeEnd <= frFrameTimeEnd + timeOffset:
            shower = associations[meteor][1]
            if shower is not None:
                title = "Meteor shower : [{:s}] - {:s}".format(shower.name, shower.name_full)
            else:
                title = "Meteor shower : Sporadic"
            break

    return title


def getVideoStartAndEndTime(video, frFile, ffPath, fps):
    video_line = video[0][0][0]
    frStartFrame = frFile.t[video_line][0]
    frEndFrame = frFile.t[video_line][-1]
    frDate = FFfile.filenameToDatetime(ffPath)
    # calculate time of min and max frame
    frFrameTimeStart = datetime2JD(frDate + datetime.timedelta(seconds=float(frStartFrame)/fps))
    frFrameTimeEnd = datetime2JD(frDate + datetime.timedelta(seconds=float(frEndFrame)/fps))

    return frFrameTimeStart, frFrameTimeEnd


def addTextToImage(image, title, x, y):
    cv2.putText(image, title, (x, y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=[0, 0, 0],
                lineType=cv2.LINE_AA, thickness=2)
    cv2.putText(image, title, (x, y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=[255, 255, 255],
                lineType=cv2.LINE_AA, thickness=1)



# Resize image to fit window to screen (for images larger than 1280x720)
# By default resize to HD (with=1280 same as for regular camera resolution)
def resizeImageIfNeed(image, width=1280):

    (h, w) = image.shape[:2]
    #  Resize only if image larger than required
    if w > width:
        r = width / float(w)
        dim = (width, int(h * r))
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    else:
        return image


def getTimestampTitle(stationName, startDate, frameNumber, fps):
    timestamp = startDate + datetime.timedelta(seconds=(frameNumber/fps))
    return stationName + ' ' + (timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-5]) + ' UTC'

def getStationNameAndTimestampfromFile(ff_path):
    fileName = os.path.basename(ff_path)
    stationName = fileName.split('_')[1]
    return stationName, FFfile.filenameToDatetime(fileName)


def loadShowerAssociations(folder, configuration):
    associations = {}
    # Get FTP file so we can filter by shower
    ftp_list = glob(os.path.join(folder, 'FTPdetectinfo_{}*.txt'.format(configuration.stationID)))
    ftp_list = [x for x in ftp_list if validDefaultFTPdetectinfo(os.path.basename(x))]

    if len(ftp_list) < 1:
        print('Unable to find FTPdetect file in {}'.format(folder))
        # return empty list to finish mp4 generation if FTPdetect file not found
        return associations
    ftp_file = ftp_list[0]

    print('Performing shower association using {}'.format(ftp_file))

    associations_per_dir, _ = showerAssociation(configuration, [ftp_file],
                                                shower_code=None, show_plot=False, save_plot=False, plot_activity=False)
    associations.update(associations_per_dir)

    return associations


if __name__ == "__main__":

    # COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="""Show reconstructed fireball detections from FR files.
        Key mapping:
            Space: pause display.
            1: previous file.
            2: next line.
            q: Quit.
            """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('dir_path', nargs=1, metavar='DIR_PATH', type=str,
        help='Path to the directory which contains FR bin files.')

    arg_parser.add_argument('-e', '--extract', action="store_true",
        help="Save frames from FR files to disk.")

    arg_parser.add_argument('-a', '--avg', action="store_true",
        help="Average pixel as the background instead of maxpixel.")

    arg_parser.add_argument('-x', '--hide', action="store_true",
        help="Do not show frames on the screen.")
    
    arg_parser.add_argument('-f', '--extractformat', metavar='EXTRACT_FORMAT', help="""Image format for extracted files. png by default. """)

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str,
        help="Path to a config file which will be used instead of the default one.")

    arg_parser.add_argument('-s', '--split', action="store_true", help="Use legacy mode where lines are displayed one-by-one.")

    arg_parser.add_argument("-t", "--timestamp", action="store_true", help="Show timestamp on the image.")

    arg_parser.add_argument("-n", "--framenumber", action="store_true", help="Show frame number on the image.")

    arg_parser.add_argument("-m", "--append_ff_to_video", action="store_true", help="Append image with meteor to video")

    arg_parser.add_argument("-w", "--add_shower_name", action="store_true", help="Show shower name on image")


    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################
    dir_path = os.path.abspath(cml_args.dir_path[0])

    # Load the configuration file
    config = cr.loadConfigFromDirectory(cml_args.config, 'notused')

    

    # Get the list of FR bin files (fireball detections) in the given directory
    fr_list = [fr for fr in os.listdir(dir_path) if fr[0:2]=="FR" and fr.endswith('bin')]
    fr_list = sorted(fr_list)

    if not fr_list:

        print("No files found!")
        sys.exit()

    # Get the list of FF bin files (compressed video frames)
    ff_list = [ff for ff in os.listdir(dir_path) if FFfile.validFFName(ff)]
    ff_list = sorted(ff_list)

    add_shower_name=cml_args.add_shower_name
    associations = {}
    if add_shower_name:
        associations = loadShowerAssociations(dir_path, config)
        # if no meteor information - skip adding name
        if not associations:
            print("Shower Associations not loaded, skipping add shower name")
            add_shower_name = False

    i = 0

    while True:

        # Break the loop if at the end
        if i >= len(fr_list):
            break

        fr = fr_list[i]

        ff_match = None

        # Strip extensions
        fr_name = ".".join(fr.split('.')[:-1]).replace('FR', '').strip("_")

        # Find the matching FF bin to the given FR bin
        for ff in ff_list:

            # Strip extensions
            ff_name = ".".join(ff.split('.')[:-1]).replace('FF', "").strip("_")


            if ff_name[2:] == fr_name[2:]:
                ff_match = ff
                break
        
        # View the fireball detection
        retval = view(dir_path, ff_match, fr, config, save_frames=cml_args.extract,
            extract_format=cml_args.extractformat, hide=cml_args.hide, avg_background=cml_args.avg,
            split=cml_args.split, add_timestamp=cml_args.timestamp, add_frame_number=cml_args.framenumber,
                      append_ff_to_video=cml_args.append_ff_to_video, add_shower_name=add_shower_name,
                      associations=associations)

        # Return to previous file
        if retval == -1:
            i -= 2

        if i < 0:
            i = 0

        i += 1
