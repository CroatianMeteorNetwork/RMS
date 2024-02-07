""" Plot the intervals between timestamps from FF file and scores the timing performance.
Usage:
  python -m Utils.PlotTimeIntervals /path/to/directory --fps 25

Arguments:
  <dir_path>  The path to the folder containing the files to analyze.
  [fps]          Frames per second. Optional argument. Default is 25.

This script analyzes the timestamps in the files located in the specified folder.
The optional fps argument allows specifying the frames per second for the analysis.
If fps is not provided, the default value of 25 is used.

 """
import os
import argparse
import tarfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
from datetime import datetime
import RMS.ConfigReader as cr


def analyze_timestamps(dir_path, fps=25.0):

    # Extract the subdir_name from dir_path
    subdir_name = os.path.basename(dir_path.rstrip('/\\'))

    # Find the FS*.tar.bz2 file in the specified directory
    tar_file_path = None
    for file in os.listdir(dir_path):
        if file.endswith('.tar.bz2') and file.startswith('FS'):
            tar_file_path = os.path.join(dir_path, file)
            break

    if not tar_file_path:
        print("Tar file not found.")
    else:
        # Open the tar file
        with tarfile.open(tar_file_path, 'r:bz2') as tar:
            timestamps = []
            # Iterate through its members
            for member in tar.getmembers():
                # Check if the current member is a .bin file
                if member.isfile() and member.name.endswith('.bin'):
                    try:
                        # Extract timestamp from the file name
                        file_name_parts = member.name.split('_')
                        timestamp_str = file_name_parts[2] + file_name_parts[3] + file_name_parts[4].split('.')[0]
                        timestamp = datetime.strptime(timestamp_str, '%Y%m%d%H%M%S%f')
                        timestamps.append(timestamp)
                    except ValueError:
                        print("Skipping file with incorrect format: {}".format(member.name))


    timestamps.sort()

    # set the number of frames in each FF file
    block_size = 256

    # Calculate intervals, starting from the second timestamp as the first is unreliable
    intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(1, len(timestamps) - 1)]
    timestamps = timestamps[1:-1]

    # Convert timestamps to a NumPy array for boolean indexing
    timestamps_np = np.array(timestamps)
    intervals_np = np.array(intervals)
    
    # Calculate minimum, maximum, average, median, and expected intervals
    min_interval = min(intervals) if intervals else None
    max_interval = max(intervals) if intervals else None
    #mean_interval = np.mean(intervals) if intervals else None
    median_interval = np.median(intervals_np) if intervals_np.size > 0 else None
    expected_interval = block_size / fps

    # Calculate average fps
    # average_fps = block_size / mean_interval
    median_fps = block_size / median_interval

    # Set the window size for the moving average
    window_size = 50

    # Calculate moving average
    moving_avg = np.convolve(intervals_np, np.ones(window_size), 'valid') / window_size

    # Insert padding at the start to line up with timestamps
    padding = np.full(window_size-1, expected_interval)
    padded_moving_avg = np.concatenate([padding, moving_avg])

    # Set the threshold above which to tag for possible dropped frames
    tolerance = 1
    threshold = (block_size + tolerance) / fps

    # Create a boolean mask where the moving average exceeds the threshold
    above_threshold = padded_moving_avg > threshold

    

    ### Scoring ###

    # Don't compute scores if there's not enough data
    if len(intervals) > 60:
        # Calculate Jitter Quality
        count_within_tolerance = sum(1 for interval in intervals if abs(interval - expected_interval) <= tolerance / fps)
        jitter_quality = (count_within_tolerance / len(intervals)) * 100 if intervals else 0
        # Calculate Dropped Frame Rate
        dropped_frame_rate = (np.sum(above_threshold) / len(padded_moving_avg)) * 100

    else:
        jitter_quality = None
        dropped_frame_rate = None



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
    lower_interval = (block_size - 1) / fps
    upper_interval = (block_size + 1) / fps

    plt.figure()

    # Plot grey points (below threshold)
    plt.scatter(timestamps_below_threshold, intervals_below_threshold, label='Intervals, max ({:.3f}s), min ({:.3f}s)'.format(max_interval, min_interval), c='gray', s=10, alpha=0.5, zorder=3)

    # Plot red points (above threshold) at the highest z-order
    plt.scatter(timestamps_above_threshold, intervals_above_threshold, label='Possible Dropped Frames', c='red', s=10, alpha=0.5, zorder=5)

    # Plot Expected and Median lines
    plt.axhline(y=expected_interval, color='lime', linestyle='-', label='Expected ({:.3f}s), ({:.2f} fps)'.format(expected_interval, fps), zorder=4)
    # plt.axhline(y=mean_interval, color='blue', linestyle='--', label='Mean ({:.3f}s), ({:.2f} fps)'.format(mean_interval, average_fps), zorder=4)
    plt.axhline(y=median_interval, color='green', linestyle='--', label='Median ({:.3f}s), ({:.2f} fps)'.format(median_interval, median_fps), zorder=4)

    # Determine grid interval dynamically
    difference_range = max_interval - min_interval
    raw_interval = difference_range / 11

    # Round the interval up to the nearest 0.1, 1, or 10
    if raw_interval < 0.1:
        grid_interval = math.ceil(raw_interval * 10) / 10
        grid_color = 'grey'
        rounded_min_interval = np.floor(min_interval * 10) / 10
        rounded_max_difference = np.ceil(max_interval * 10) / 10

        # Only draw lower and upper interval lines if the scale is fine enough
        plt.axhline(y=lower_interval, color='lime', linestyle='--', label='-1/fps Interval ({:.3f}s)'.format(lower_interval), zorder=4)
        plt.axhline(y=upper_interval, color='lime', linestyle='--', label='+1/fps Interval ({:.3f}s)'.format(upper_interval), zorder=4)

    elif raw_interval < 1:
        grid_interval = math.ceil(raw_interval)
        grid_color = '#C0A040'
        rounded_min_interval = np.floor(min_interval)
        rounded_max_difference = np.ceil(max_interval)

    else:
        grid_interval = math.ceil(raw_interval / 10) * 10
        grid_color = '#FFBF00'
        rounded_min_interval = np.floor(min_interval / 10) * 10
        rounded_max_difference = np.ceil(max_interval / 10) * 10

    # Ensure grid_interval is not too small
    minimum_allowed_interval = 0.1
    grid_interval = max(grid_interval, minimum_allowed_interval)

    # Draw Horizontal grid lines
    y_ticks = np.arange(rounded_min_interval, rounded_max_difference + grid_interval, grid_interval)
    plt.yticks(y_ticks)
    plt.grid(axis='y', color=grid_color, linestyle='-', alpha=0.7, zorder=0)

    # Set vertical grid to appear every two hours
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.grid(axis='x', linestyle='-', alpha=0.7, zorder=0)

    # Labeling
    plt.xlabel('Time (UTC)')
    plt.ylabel('Intervals (seconds)')

    # Title and subtitle
    plt.title('Timestamp Intervals - {}'.format(subdir_name), pad=30)
    subtitle_text = 'Jitter Quality (intervals within +/-1 frame): {:.1f}%'.format(round(jitter_quality, 1))
    subtitle_text_2 = 'Dropped Frame Rate (intervals >{} frames late within {} FF files): {:.1f}%'.format(tolerance, window_size, dropped_frame_rate)

    plt.figtext(0.5, 0.925, subtitle_text, ha='center', va='top', fontsize=8)
    plt.figtext(0.5, 0.895, subtitle_text_2, ha='center', va='top', fontsize=8)

    # Legend
    plt.legend(fontsize='x-small')
    plt.tight_layout()

    # Save the plot in the dir_path
    plot_filename = os.path.join(dir_path, '{}_intervals_scores_{:.0f}-{:.0f}.png'.format(subdir_name, jitter_quality, dropped_frame_rate))
    plt.savefig(plot_filename, format='png', dpi=150)
    plt.close()

    return jitter_quality, dropped_frame_rate, plot_filename


if __name__ == '__main__':

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Generate interval plots on all subfoler where a FS*tar.gz2 is found.")

    arg_parser.add_argument('dir_path', metavar='DIR_PATH', type=str, \
        help='Path to directory with FS*tar.bz2 files.')

    arg_parser.add_argument('--fps', metavar='FPS', type=float, default=25.0, required=False, \
        help='Expected fps (default: 25.0).')

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
                analyze_timestamps(root, fps)
