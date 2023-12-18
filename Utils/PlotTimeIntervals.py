import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import sys

def analyze_timestamps(folder_path):
    timestamps = []
    for file in os.listdir(folder_path):
        if file.endswith('.jpg'):
            try:
                timestamp_str = file.split('_')[2] + file.split('_')[3] + file.split('_')[4].split('.')[0]
                timestamp = datetime.strptime(timestamp_str, '%Y%m%d%H%M%S%f')
                timestamps.append(timestamp)
            except ValueError:
                print(f"Skipping file with incorrect format: {file}")

    timestamps.sort()
    differences = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps) - 1)]
    df = pd.DataFrame({'Timestamp': timestamps[:-1], 'Difference': differences})

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

    # Plotting
    plt.figure(figsize=(12, 6))

    # Scatter plot for differences
    plt.scatter(df['Timestamp'], df['Difference'], label='Normal', c=df['Outlier'].map({True: 'red', False: 'gray'}), s=10, alpha=0.5)

    # Expected and Average lines
    plt.axhline(y=5.12, color='green', linestyle='-', label='Expected Interval (5.12s)')
    plt.axhline(y=average_difference, color='blue', linestyle='--', label=f'Average Interval ({average_difference:.4f}s)')

    # Find the minimum difference and round down to nearest 0.1
    min_difference = df['Difference'].min()
    rounded_min_difference = np.floor(min_difference * 10) / 10

    # Find the maximum difference and round up to nearest 0.1
    max_difference = df['Difference'].max()
    rounded_max_difference = np.ceil(max_difference * 10) / 10

    # Setting gridlines
    # Horizontal grid every 0.1 seconds
    y_ticks = np.arange(rounded_min_difference, rounded_max_difference + 0.1, 0.1)
    plt.yticks(y_ticks)
    plt.grid(axis='y', linestyle='-', alpha=0.7)

    # Vertical grid every hour
    plt.gca().xaxis.set_major_locator(mdates.HourLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.grid(axis='x', linestyle='-', alpha=0.7)

    # Labeling
    plt.xlabel('Timestamp')
    plt.ylabel('Time Difference (seconds)')
    plt.title('Time Differences Between Consecutive Video Frames')
    plt.legend()

    # Save the plot in the folder_path
    plot_filename = os.path.join(folder_path, 'time_difference_plot.png')
    plt.savefig(plot_filename, format='png', dpi=300)
    plt.show()

    return df



def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    analyze_timestamps(folder_path)

if __name__ == "__main__":
    main()
