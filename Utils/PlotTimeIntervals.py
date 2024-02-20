""" Plot the intervals between timestamps from FF file and scores the timing performance.

This script analyzes the timestamps in the files located in the specified folder.
The optional fps argument allows specifying the frames per second for the analysis.
If fps is not provided, the default value of 25 is used.

"""

from __future__ import print_function, division, absolute_import

import os
import argparse
import tarfile
import math
import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import RMS.ConfigReader as cr


def plotFFTimeIntervals(dir_path, fps=25.0, ff_block_size=256, ma_window_size=50):
    """ Plot the intervals between timestamps from FF file and scores the timing performance.
    
    Arguments:
        dir_path: [str] The path to the folder containing the files to analyze. FS files will be recursively
            in all directories within dir_path.
    
    Keyword Arguments:
        fps: [float] The expected frames per second. (default: 25.0)
        ff_ff_block_size: [int] The number of frames in each FF file. (default: 256)
        ma_window_size: [int] The window size for the moving average. (default: 50)

    Returns:
        jitter_quality: [float] The jitter quality score.
        dropped_frame_rate: [float] The dropped frame rate score.
        plot_filename: [str] The filename of the plot that was saved.

    """

    # Extract the subdir_name from dir_path
    subdir_name = os.path.basename(dir_path.rstrip('/\\'))

    # Find the FS*.tar.bz2 file in the specified directory
    try:
        tar_file_path = None
        for file in os.listdir(dir_path):
            if file.endswith('.tar.bz2') and file.startswith('FS'):
                tar_file_path = os.path.join(dir_path, file)
                break

        if not tar_file_path:
            print("Tar file not found.")
            return None, None, None

    except FileNotFoundError:
        print("Directory not found: {}".format(dir_path))
        return None, None, None

    timestamps = []

    # Open the tar file
    with tarfile.open(tar_file_path, 'r:bz2') as tar:

        # Iterate through its members
        for member in tar.getmembers():

            # Check if the current member is a .bin file
            if member.isfile() and member.name.endswith('.bin'):

                try:
                    # Extract timestamp from the file name
                    file_name_parts = member.name.split('_')
                    timestamp_str = file_name_parts[2] + file_name_parts[3] + file_name_parts[4].split('.')[0]
                    timestamp = datetime.datetime.strptime(timestamp_str, '%Y%m%d%H%M%S%f')
                    timestamps.append(timestamp)

                except ValueError:
                    print("Skipping file with incorrect format: {}".format(member.name))

    if len(timestamps) < 2:
        print("Insufficient timestamps. At least two timestamps are required.")
        return None, None, None

    timestamps.sort()

    # Calculate intervals, starting from the second timestamp as the first is unreliable
    intervals = [(timestamps[i + 1] - timestamps[i]).total_seconds() for i in range(1, len(timestamps) - 1)]
    timestamps = timestamps[1:-1]

    # Convert timestamps to a NumPy array for boolean indexing
    timestamps_np = np.array(timestamps)
    intervals_np = np.array(intervals)
    
    # Calculate minimum, maximum, average, median, and expected intervals
    min_interval = min(intervals) if intervals else None
    max_interval = max(intervals) if intervals else None
    #mean_interval = np.mean(intervals) if intervals else None
    median_interval = np.median(intervals_np)
    std_intervals_seconds = np.std(intervals_np)
    std_intervals_frames = std_intervals_seconds*fps
    expected_interval = ff_block_size/fps

    # Calculate average fps
    # average_fps = ff_block_size/mean_interval
    median_fps = ff_block_size/median_interval

    
    ### Scoring ###

    # Set frame tolerance threshold
    tolerance = 1

    # Don't compute scores if there's not enough data
    if len(intervals) > ma_window_size + 10:
        # Calculate moving average
        moving_avg = np.convolve(intervals_np, np.ones(ma_window_size), 'valid')/ma_window_size

        # Insert padding at the start to line up with timestamps
        padding = np.full(ma_window_size - 1, expected_interval)
        padded_moving_avg = np.concatenate([padding, moving_avg])

        # Set the threshold above which to tag for possible dropped frames
        threshold = (ff_block_size + tolerance)/fps

        # Create a boolean mask where the moving average exceeds the threshold
        above_threshold = padded_moving_avg > threshold
        # Calculate Jitter Quality
        count_within_tolerance = sum(1 for interval in intervals if abs(interval - expected_interval) <= tolerance/fps)
        jitter_quality = (count_within_tolerance/len(intervals))*100 if intervals else 0
        # Calculate Dropped Frame Rate
        dropped_frame_rate = (np.sum(above_threshold)/len(padded_moving_avg))*100

    else:
        jitter_quality = 100
        dropped_frame_rate = 0
        above_threshold = above_threshold = np.full(intervals_np.shape, False, dtype=bool)



    ### Plotting ###
    
    # Only tag long intervals for plotting
    above_expected_interval = intervals_np > expected_interval
    combined_condition = above_threshold & above_expected_interval

    # Separate data points based on the condition for plotting
    timestamps_below_threshold = timestamps_np[~combined_condition]
    intervals_below_threshold = intervals_np[~combined_condition]
    timestamps_above_threshold = timestamps_np[combined_condition]
    intervals_above_threshold = intervals_np[combined_condition]

    # Calculating the lower and upper interval values for plotting
    lower_interval = (ff_block_size - 1)/fps
    upper_interval = (ff_block_size + 1)/fps

    # Set up two subplots one on top of the other. The top one is 2x the size of the bottom one
    fig, (ax_inter, ax_res) = plt.subplots(
        nrows=2, ncols=1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]}
        )

    # Plot grey points (below threshold)
    ax_inter.scatter(timestamps_below_threshold, intervals_below_threshold, 
                label='Intervals, max ({:.3f}s), min ({:.3f}s)'.format(max_interval, min_interval), 
                c='gray', s=10, alpha=0.5, zorder=3)

    # Plot red points (above threshold) at the highest z-order
    ax_inter.scatter(timestamps_above_threshold, intervals_above_threshold, 
                label='Possible Dropped Frames', 
                c='red', s=10, alpha=0.5, zorder=5)

    # Plot Expected and Median lines
    ax_inter.axhline(y=expected_interval, color='lime', linestyle='-', 
                label='Expected ({:.3f}s), ({:.2f} fps)'.format(expected_interval, fps), zorder=4)
    # ax_inter.axhline(y=mean_interval, color='blue', linestyle='--', label='Mean ({:.3f}s), ({:.2f} fps)'.format(mean_interval, average_fps), zorder=4)
    ax_inter.axhline(y=median_interval, color='green', linestyle='--', 
                label='Median ({:.3f} +/- {:.3f} s), ({:.2f} +/- {:.2f} fps)'.format(
                    median_interval, std_intervals_seconds, median_fps, std_intervals_frames), zorder=4)

    # Determine grid interval dynamically
    difference_range = max_interval - min_interval
    raw_interval = difference_range/11

    # Round the interval up to the nearest 0.1, 1, or 10
    if raw_interval < 0.1:
        grid_interval = math.ceil(raw_interval*10)/10
        grid_color = 'grey'
        rounded_min_interval = np.floor(min_interval*10)/10
        rounded_max_difference = np.ceil(max_interval*10)/10

        # Only draw lower and upper interval lines if the scale is fine enough
        ax_inter.axhline(y=lower_interval, color='lime', linestyle='--', 
                    label='-1/fps Interval ({:.3f}s)'.format(lower_interval), zorder=4)
        ax_inter.axhline(y=upper_interval, color='lime', linestyle='--', 
                    label='+1/fps Interval ({:.3f}s)'.format(upper_interval), zorder=4)

    elif raw_interval < 1:
        grid_interval = math.ceil(raw_interval)
        grid_color = '#C0A040'
        rounded_min_interval = np.floor(min_interval)
        rounded_max_difference = np.ceil(max_interval)

    else:
        grid_interval = math.ceil(raw_interval/10)*10
        grid_color = '#FFBF00'
        rounded_min_interval = np.floor(min_interval/10)*10
        rounded_max_difference = np.ceil(max_interval/10)*10

    minimum_allowed_interval = 0.1
    grid_interval = max(grid_interval, minimum_allowed_interval)

    # Draw Horizontal grid lines
    y_ticks = np.arange(rounded_min_interval, rounded_max_difference + grid_interval, grid_interval)
    ax_inter.set_yticks(y_ticks)
    ax_inter.grid(axis='y', color=grid_color, linestyle='-', alpha=0.7, zorder=0)

    # Set vertical grid to appear every two hours
    ax_inter.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax_inter.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_inter.grid(axis='x', linestyle='-', alpha=0.7, zorder=0)

    # Labeling
    ax_inter.set_ylabel('Intervals (seconds)')

    # Title and subtitle
    plot_title = subdir_name.replace("_detected", "")
    fig.suptitle('Timestamp Intervals - {}'.format(plot_title))

    subtitle_text = 'Jitter Quality (intervals within +/-1 frame): {:.1f}%'.format(jitter_quality)
    subtitle_text_2 = 'Dropped Frame Rate (intervals >{} frames late within {} FF files): {:.1f}%'.format(
        tolerance, ma_window_size, dropped_frame_rate)

    plt.figtext(0.5, 0.945, subtitle_text, ha='center', va='top', fontsize=8)
    plt.figtext(0.5, 0.920, subtitle_text_2, ha='center', va='top', fontsize=8)

    # Legend
    ax_inter.legend(fontsize='x-small')



    # Plot the residuals from the expected interval
    residuals = intervals_np - expected_interval
    ax_res.scatter(timestamps, residuals, c='gray', s=1, zorder=3)

    # Enable the grid
    ax_res.grid(alpha=0.7, zorder=0)

    # Limit the plot to +/- 2 frames in the Y axis
    ax_res.set_ylim(-2/fps, 2/fps)

    # Draw a horizontal line at 0
    ax_res.axhline(y=0, color='lime', linestyle='-', zorder=4)

    # Plot the median
    median_residual = median_interval - expected_interval
    ax_res.axhline(y=median_residual, color='green', linestyle='--', zorder=4)


    ax_res.set_ylabel('Residuals (seconds)')
    ax_res.set_xlabel('Time (UTC)')


    plt.tight_layout()

    # Move the upper edge of ax_inter a bit down to make room for the text
    plt.subplots_adjust(top=0.90, hspace=0.05)

    # Save the plot in the dir_path
    suffix = "_ff_intervals_flagged.png" if dropped_frame_rate > 0.1 else "_ff_intervals.png"
    plot_filename = os.path.join(dir_path, '{}{}'.format(plot_title, suffix))
    plt.savefig(plot_filename, dpi=150)
    plt.close()

    return jitter_quality, dropped_frame_rate, plot_filename


if __name__ == '__main__':

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Generate interval plots on all subdirectories where a FS*tar.bz2 is found. FPS will be taken from config files unless it is manually overridden.")

    arg_parser.add_argument('dir_path', metavar='DIR_PATH', type=str, \
        help='Path to directory with FS*tar.bz2 files.')

    arg_parser.add_argument('--fps', metavar='FPS', type=float, default=25.0, required=False, \
        help='Expected fps (default: taken from config file, 25.0 if config not found).')

     # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    for root, dirs, files in os.walk(cml_args.dir_path):
        for file in files:
            if file.endswith('.tar.bz2') and file.startswith('FS'):
                try:
                    config = cr.loadConfigFromDirectory('.config', root)
                    fps = config.fps
                except:
                    fps = cml_args.fps
                tar_file_path = os.path.join(root, file)
                print("Processing {}".format(tar_file_path))
                plotFFTimeIntervals(root, fps)
