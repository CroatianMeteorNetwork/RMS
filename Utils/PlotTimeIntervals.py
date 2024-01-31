""" Plot the intervals between timestamps from FF file and scores the variability. """
import os
import tarfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
from datetime import datetime
from RMS.ConfigReader import Config
import sys


def calculate_score(differences, alpha=1.5):
    """
    Calculate a score using an exponential decay function based on the standard deviation.
    A higher standard deviation results in a lower score.
    The alpha parameter controls the rate of decay.
    """
    current_std_dev = np.std(differences)
    
    score = 1000 * np.exp(-alpha * current_std_dev)

    return int(round(score))


def analyze_timestamps(folder_path, fps=25):

    # Extract the subdir_name from folder_path
    subdir_name = os.path.basename(folder_path.rstrip('/\\'))

    # Find the FS*.tar.bz2 file in the specified directory
    tar_file_path = None
    for file in os.listdir(folder_path):
        if file.endswith('.tar.bz2') and file.startswith('FS'):
            tar_file_path = os.path.join(folder_path, file)
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
                        print(f"Skipping file with incorrect format: {member.name}")

    timestamps.sort()
    differences = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps) - 1)]
    df = pd.DataFrame({'Timestamp': timestamps[:-1], 'Difference': differences})

    if len(differences) > 10:
        score = calculate_score(differences)
    else:
        score = None

    # Calculate mean and standard deviation
    mean_diff = df['Difference'].mean()
    std_diff = df['Difference'].std()

    # Define outlier threshold based on standard deviation (e.g., 2 or 3 sigma)
    sigma = 10
    lower_bound = mean_diff - (sigma * std_diff)
    upper_bound = mean_diff + (sigma * std_diff)

    # Identify outliers
    df['Outlier'] = (df['Difference'] < lower_bound) | (df['Difference'] > upper_bound)


    # Calculate average excluding outliers
    average_difference = df[~df['Outlier']]['Difference'].mean()

    # Calculate average fps
    block_size = 256
    average_fps = block_size / average_difference
    expected_interval = block_size / fps

    # Plotting
    plt.figure(figsize=(12, 6))

    # Separate scatter plots for normal points and outliers
    normal_points = df[~df['Outlier']]
    outlier_points = df[df['Outlier']]

    plt.scatter(normal_points['Timestamp'], normal_points['Difference'], label='Normal', c='gray', s=10, alpha=0.5)
    plt.scatter(outlier_points['Timestamp'], outlier_points['Difference'], label='Outliers', c='red', s=10, alpha=0.5)

    # Expected and Average lines
    plt.axhline(y=expected_interval, color='green', linestyle='-', label=f'Expected Interval from .config fpsv({expected_interval:.4f}s)')
    plt.axhline(y=average_difference, color='blue', linestyle='--', label=f'Average Interval ({average_difference:.4f}s), Average ({average_fps:.1f} fps)')

    # Find the minimum difference and round down to nearest 0.1
    min_difference = df['Difference'].min()

    # Find the maximum difference and round up to nearest 0.1
    max_difference = df['Difference'].max()

    # Setting gridlines
    # Determine grid interval dynamically
    difference_range = df['Difference'].max() - df['Difference'].min()
    raw_interval = difference_range / 11


    # Round the interval up to the nearest 0.1, 1, or 10
    if raw_interval < 0.1:
        grid_interval = math.ceil(raw_interval * 10) / 10
        grid_color = 'grey'
        rounded_min_difference = np.floor(min_difference * 10) / 10
        rounded_max_difference = np.ceil(max_difference * 10) / 10

    elif raw_interval < 1:
        grid_interval = math.ceil(raw_interval)
        grid_color = '#FFBF00'  # Corrected color format
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
    plt.title(f'Frame Intervals {subdir_name} - Score: {score}')
    plt.legend()

    # Save the plot in the folder_path
    plot_filename = os.path.join(folder_path, f'{subdir_name}_intervals_score_{score}.png')
    plt.savefig(plot_filename, format='png', dpi=300)
    #plt.show()

    return score, plot_filename



def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    analyze_timestamps(folder_path)

if __name__ == "__main__":
    main()
