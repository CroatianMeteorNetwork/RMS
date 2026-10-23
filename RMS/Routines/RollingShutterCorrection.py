""" Functions for correction the rolling shutter effect.

A rolling shutter exposes and reads the rows of the sensor one after the other, from the top of the image to the
bottom, so a row is recorded later than the row above it. The time of a frame is the time of its first (top) row,
and a point at the row y is recorded later by y/n_rows of the readout time, the time the sensor takes to read all
its rows. The corrected time of the point is then, in frames:

    frame + readout_frames*y/n_rows

where readout_frames is the readout time in frames (readout time times the frame rate).

The readout time depends on the sensor and its mode (a sensor which can run at 30 fps, recording at 10 fps, reads
its rows in a third of a frame), so it is measured and set in seconds with rolling_shutter_readout in the config
(see readoutFrames); 0 (the default) makes no correction. It is the row time of the sensor mode times the number
of rows (from the sensor specification), or it is measured, e.g. against another camera with a global shutter: the
time offset of the points of a moving object grows linearly with their row, and the readout time is the offset from
the top row to the bottom row.

An image cropped in the capture (video_crop, or the region of interest roi_up/roi_down) still comes from all the rows
of the sensor: its first row is the row 'top' of the sensor, which reads n_rows + top + bottom rows in the readout
time (see sensorRows). correctFrameTime applies both from the config.
"""

from __future__ import print_function, division, absolute_import


def readoutFrames(config, fps=None):
    """ The readout time of the rows of the sensor, in frames.

    Arguments:
        config: [Config] Configuration object, with the measured readout time rolling_shutter_readout (seconds, 0
            for none).

    Keyword arguments:
        fps: [float] Frame rate of the input. None by default (the frame rate in the config).

    Return:
        [float] Readout time in frames, 0 without a measured readout time (no correction).
    """

    # The readout time (s) times the frame rate of the input; a missing or negative readout time makes no correction
    return max(getattr(config, 'rolling_shutter_readout', 0.0), 0.0)*(fps if fps else config.fps)


# The inputs which come from the frames processed by the RMS capture, after the crop of the video and the region of
#   interest: FF files and frame blocks. Videos (the raw stream, saved before the crop), image folders (from any
#   camera) and the files of other capture systems (e.g. UWO .vid) are taken as not cropped
CAPTURE_INPUT_TYPES = ('ff', 'fblk')


def croppedInput(input_type):
    """ Whether an input type (FrameInterface input_type) comes from the cropped frames of the RMS capture (see
        CAPTURE_INPUT_TYPES).
    """

    return input_type in CAPTURE_INPUT_TYPES


def sensorRows(config, n_rows, cropped=True):
    """ The row of the sensor at the top of the image, and the number of rows the sensor reads in the readout
        time, in rows of the image. The capture can remove rows which the sensor still reads: a crop of the video
        (video_crop, e.g. 'top=10 bottom=20 left=15 right=15') and the region of interest (roi_up, roi_down). An
        image which was not cropped (a raw video saved before the crop, or the file of another capture system), or
        taller than the processed frames (height_device without the region of interest), is the whole sensor.

    Arguments:
        config: [Config] Configuration object, with video_crop, roi_up, roi_down and height_device (height of
            the delivered frames before the region of interest, as in the config file).
        n_rows: [int] Number of rows of the image.

    Keyword arguments:
        cropped: [bool] The image comes from the processed frames of the RMS capture (see croppedInput), not
            from a raw video saved before the crop or another capture system. True by default.

    Return:
        (top, rows): [tuple] The sensor row of the first image row, and the number of rows read by the sensor.
    """

    # The rows cropped from the video at the top and at the bottom (pixels of the delivered frames, after any
    #   video_scale). The capture crops the video only with the GStreamer backend and a well-formed crop (see
    #   BufferedCapture.validVideoCrop)
    crop = {'top': 0, 'bottom': 0}
    tokens = [token.partition('=') for token in (getattr(config, 'video_crop', None) or '').split()]
    valid = all(sep and (key in ('top', 'bottom', 'left', 'right')) and val.isdigit() for key, sep, val in tokens)
    if valid and (getattr(config, 'media_backend', 'gst') == 'gst'):
        for key, _, val in tokens:
            if key in crop:
                crop[key] = int(val)

    # The rows outside the region of interest of the delivered frames (config.height is not used: inputs of
    #   another size overwrite it)
    roi_top, roi_bottom = 0, 0
    height_device = getattr(config, 'height_device', None) or n_rows
    roi_up, roi_down = getattr(config, 'roi_up', -1), getattr(config, 'roi_down', -1)
    if 0 <= roi_up < roi_down <= height_device:
        roi_top, roi_bottom = roi_up, height_device - roi_down

    # All rows removed above and below the image
    top = crop['top'] + roi_top
    bottom = crop['bottom'] + roi_bottom

    # Without removed rows, for an image which was not cropped or one taller than the processed frames, the image
    #   is the whole sensor
    if (not cropped) or ((top + bottom) == 0) or (n_rows > (height_device - roi_top - roi_bottom)):
        return 0, n_rows

    return top, n_rows + top + bottom


def correctFrameTime(frame, y_centr, n_rows, config, fps=None, cropped=True):
    """ Correct the frame of a point at the row y_centr of the image for the rolling shutter, with the measured
        readout time and the crop of the capture from the config.

    Arguments:
        frame: [float or ndarray] Frame of the point(s).
        y_centr: [float or ndarray] Row of the point(s) on the image.
        n_rows: [int] Number of rows of the image.
        config: [Config] Configuration object (rolling_shutter_readout, video_crop, the region of interest, fps).

    Keyword arguments:
        fps: [float] Frame rate of the input. None by default (the frame rate in the config).
        cropped: [bool] The image comes from the processed frames of the RMS capture (see croppedInput). True by
            default.

    Return:
        [float or ndarray] The corrected frame(s).
    """

    # The sensor row of the top of the image and the rows read by the sensor, then the time of the sensor row of the
    #   point within the readout
    top, rows = sensorRows(config, n_rows, cropped=cropped)

    return correctRollingShutterTemporal(frame, y_centr + top, rows, readout_frames=readoutFrames(config, fps))


def correctRollingShutterTemporal(frame, y_centr, n_rows, readout_frames=1.0):
    """ Given the frame index, height of the meteor on the image, total height of the image, correct
        frame timestamps to compensate for the rolling shutter effect.

    Source: Kukic et al. 2018 "Correction for meteor centroids observed using rolling shutter cameras", WGN
    
    Arguments:
        frame: [list] Frame on which the meteor was recorded.
        y_centr: [list] Y coordinate of the meteor centroid (row of the sensor).
        n_rows: [float] Number of rows the sensor reads in the readout time (the vertical size of the image if
            it is not cropped, see sensorRows).

    Keyword arguments:
        readout_frames: [float] Time in which the sensor reads all its rows, in frames (see readoutFrames). 1 by
            default (the whole frame period).
    """

    # Compute the time offset in fractional frames
    delta_t = readout_frames*y_centr/n_rows

    # Compute the corrected timestamp
    fr_corrected = frame + delta_t

    return fr_corrected





def correctRollingShutterTemporalList(frames_list, height_list, n_rows, readout_frames=1.0):
    """ Given the list of frame indices, height of the meteor on the image, total height of the image, correct
        frame timestamps to compensate for the rolling shutter effect.

    Source: Kukic et al. 2018 "Correction for meteor centroids observed using rolling shutter cameras", WGN
    
    Arguments:
        frames_list: [list] A list of frames on which the meteor was recorded.
        height_list: [list] A list of Y coordinates of meteor centroids.
        n_rows: [float] Vertical size of the image in pixels.

    Keyword arguments:
        readout_frames: [float] Time in which the sensor reads all its rows, in frames (see readoutFrames). 1 by
            default (the whole frame period).
    """

    # Go through all frames
    corrected_frame_times = []


    # Check if the list of frames has the same number of elements as the list of Y image coordinates
    if len(frames_list) != len(height_list):
        raise ValueError("The length of the list of frames and the list of Y image coordinates must be the same!")


    # Compute the corrected timestamp for every centroid
    for fr, y_centr in zip(frames_list, height_list):

        fr_corrected = correctRollingShutterTemporal(fr, y_centr, n_rows, readout_frames=readout_frames)

        corrected_frame_times.append(fr_corrected)


    return corrected_frame_times





if __name__ == "__main__":


    import matplotlib.pyplot as plt

    # Test the rolling shutter correction
    
    frames_list = map(float, list(range(0, 21)))
    height_list = [36*x for x in list(range(0, 21))]

    n_rows = 720

    # Perform the timestamp correction
    corrected_frame_times = correctRollingShutterTemporalList(frames_list, height_list, n_rows)


    plt.scatter(frames_list, height_list, label='Original')
    plt.scatter(corrected_frame_times, height_list, label='Corrected')

    plt.xlabel('Frame')
    plt.ylabel('Y coordinate')

    plt.legend()

    plt.show()



