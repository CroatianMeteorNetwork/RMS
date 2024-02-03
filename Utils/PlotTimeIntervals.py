""" Plot the intervals between timestamps from FF file and scores the variability.
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


def calculate_score(differences, alpha=1.5):
    """
    Calculate a score using an exponential decay function based on the standard deviation.
    A higher standard deviation results in a lower score.
    The alpha parameter controls the rate of decay.
    """
    current_std_dev = np.std(differences)
    
    score = 1000 * np.exp(-alpha * current_std_dev)

    return int(round(score))


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
    # Calculate differences, starting from the second timestamp as the first is unreliable
    differences = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(1, len(timestamps) - 1)]

    if len(differences) > 10:
        score = calculate_score(differences)
    else:
        score = None
    
    min_difference = min(differences) if differences else None
    max_difference = max(differences) if differences else None
    average_difference = sum(differences) / len(differences) if differences else None

    # Calculate average fps
    block_size = 256
    average_fps = block_size / average_difference
    expected_interval = block_size / fps

    # Plotting
    plt.figure(figsize=(12, 6))


    plt.scatter(timestamps[1:-1], differences, label= 'Intervals, max ({:.3f}s), min ({:.3f}s)'.format(max_difference, min_difference), c='gray', s=10, alpha=0.5)

    # Expected and Average lines
    plt.axhline(y=expected_interval, color='green', linestyle='-', label='Expected Interval ({:.3f}s), ({:.1f} fps)'.format(expected_interval, fps))
    plt.axhline(y=average_difference, color='blue', linestyle='--', label='Average Interval ({:.3f}s), ({:.1f} fps)'.format(average_difference, average_fps))

    # Setting gridlines
    # Determine grid interval dynamically
    difference_range = max_difference - min_difference
    raw_interval = difference_range / 11


    # Round the interval up to the nearest 0.1, 1, or 10
    if raw_interval < 0.1:
        grid_interval = math.ceil(raw_interval * 10) / 10
        grid_color = 'grey'
        rounded_min_difference = np.floor(min_difference * 10) / 10
        rounded_max_difference = np.ceil(max_difference * 10) / 10

    elif raw_interval < 1:
        grid_interval = math.ceil(raw_interval)
        grid_color = '#FFBF00'
        rounded_min_difference = np.floor(min_difference)
        rounded_max_difference = np.ceil(max_difference)

    else:
        grid_interval = math.ceil(raw_interval / 10) * 10
        grid_color = 'red'
        rounded_min_difference = np.floor(min_difference / 10) * 10
        rounded_max_difference = np.ceil(max_difference / 10) * 10

    # Ensure grid_interval is not too small
    minimum_allowed_interval = 0.1
    grid_interval = max(grid_interval, minimum_allowed_interval)


    y_ticks = np.arange(rounded_min_difference, rounded_max_difference + grid_interval, grid_interval)
    plt.yticks(y_ticks)
    plt.grid(axis='y', color=grid_color, linestyle='-', alpha=0.7)

    # Vertical grid every hour
    plt.gca().xaxis.set_major_locator(mdates.HourLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.grid(axis='x', linestyle='-', alpha=0.7)

    # Labeling
    plt.xlabel('Timestamp')
    plt.ylabel('Time Difference (seconds)')
    plt.title('Timestamp Intervals {} - Score: {}'.format(subdir_name, score))
    plt.legend()

    # Save the plot in the dir_path
    plot_filename = os.path.join(dir_path, '{}_intervals_score_{}.png'.format(subdir_name, score))
    plt.savefig(plot_filename, format='png', dpi=300)
    plt.close()

    return score, plot_filename


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
