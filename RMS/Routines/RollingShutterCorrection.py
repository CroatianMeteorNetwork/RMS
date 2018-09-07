""" Functions for correction the rolling shutter effect. """

from __future__ import print_function, division, absolute_import


def correctRollingShutterTemporal(frame, y_centr, n_rows):
    """ Given the frame index, height of the meteor on the image, total height of the image, correct
        frame timestamps to compensate for the rolling shutter effect.

    Source: Kukic et al. 2018 "Correction for meteor centroids observed using rolling shutter cameras", WGN
    
    Arguments:
        frame: [list] Frame on which the meteor was recorded.
        y_centr: [list] Y coordinate of the meteor centroid.
        n_rows: [float] Vertical size of the image in pixels.
    """

    # Compute the time offset in fractional frames
    delta_t = y_centr/n_rows

    # Compute the corrected timestamp
    fr_corrected = frame + delta_t

    return fr_corrected





def correctRollingShutterTemporalList(frames_list, height_list, n_rows):
    """ Given the list of frame indices, height of the meteor on the image, total height of the image, correct
        frame timestamps to compensate for the rolling shutter effect.

    Source: Kukic et al. 2018 "Correction for meteor centroids observed using rolling shutter cameras", WGN
    
    Arguments:
        frames_list: [list] A list of frames on which the meteor was recorded.
        height_list: [list] A list of Y coordinates of meteor centroids.
        n_rows: [float] Vertical size of the image in pixels.
    """

    # Go through all frames
    corrected_frame_times = []


    # Check if the list of frames has the same number of elements as the list of Y image coordinates
    if len(frames_list) != len(height_list):
        raise ValueError("The lenght of the list of frames and the list of Y image coordinates must be the same!")


    # Compute the corrected timestamp for every centroid
    for fr, y_centr in zip(frames_list, height_list):

        fr_corrected = correctRollingShutterTemporal(fr, y_centr, n_rows)

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



