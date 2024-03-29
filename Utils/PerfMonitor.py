"""
A monitoring tool for long term logging of system performance metrics and key settings on RMS systems.
It records write speeds for system and data drives, gathers system info (OS version, architecture, model),
and checks disk space. Metrics are logged to a CSV file for analysis.
"""

from __future__ import print_function, division, absolute_import


import os
import csv
import time
import platform
import shutil
import logging

from threading import Lock
from multiprocessing import Manager

# Get the logger from the main module
log = logging.getLogger("logger")

class PerfMonitor:
    def __init__(self, night_data_dir_name, config):
        """Initializes the performance monitoring tool.

        Arguments:
            night_data_dir_name: [str] Name of the directory where night data is stored.
            config: [Config instance]
        """

        self.manager = Manager()
        self.data_entries = self.manager.dict()
        self.log_file_path = './perfMonitorLog.csv'

        self.fieldnames = [
            'data_dir_name',
            'model',
            'os_version',
            'architecture',
            'system_drive_desc',
            'system_drive_speed',
            'data_drive_desc',
            'data_drive_speed',
            'res',
            'calc_fps',
            'media_backend',
            'media_backend_ovr',
            'live_maxpixel',
            'live_jpg',
            'slideshow',
            'hdu_compress',
            'fireball_detection',
            'jitter_quality',
            'dropped_frame_rate',
            'total_gb',
            'used_gb',
            'free_gb'
        ]

        self.night_data_dir_name = night_data_dir_name
        self.config = config
        self.lock = Lock()

        self.getRunInfo()


    def cleanNonAscii(self, text):
        """Removes non-ASCII characters from the given text.

    Arguments:
        text: [str] The string from which to remove non-ASCII characters.
    Return:
        [str] The cleaned string, containing only ASCII characters.
    """

        return ''.join(char for char in text if ord(char) < 128)


    def updateEntry(self, key, value):
        """Updates a specific performance metric in the data entries dictionary.

        Arguments:
            key: [str] The name of the performance metric to update.
            value: [various] The new value for the specified metric.
        """

        with self.lock:
            if self.night_data_dir_name not in self.data_entries:
                # Initialize a new entry if this is the first value for this data_dir_name
                entry = {fieldname: None for fieldname in self.fieldnames}
            else:
                # Retrieve the existing entry to modify it
                entry = self.data_entries[self.night_data_dir_name]

            # Update the entry
            entry[key] = value

            # Re-assign the modified entry back to the managed dictionary
            self.data_entries[self.night_data_dir_name] = entry


    def logToCsv(self):
        """Logs the collected performance metrics to a CSV file.

        Ensures each metric is correctly logged under its corresponding header.
        """
        # Check if the file exists to determine if we need to write headers
        file_exists = os.path.isfile(self.log_file_path)

        with self.lock:
            with open(self.log_file_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)

                # Write the header only if the file did not exist before
                if not file_exists:
                    writer.writeheader()

                for data_dir, entry in self.data_entries.items():
                    # Ensure 'data_dir_name' is part of the entry for logging
                    entry['data_dir_name'] = data_dir
                    writer.writerow(entry)

        # Clear entries after logging
        self.data_entries.clear()


    def writeTest(self, file_path='./', block_size=1024*1024, num_blocks=100):
        """
        Perform a write performance test by writing a specific number of blocks
        of data to a temporary file and measure the time taken.

        Arguments:
            file_path: [str] Path to the temporary file for the write test.
            block_size: [int] Size of each block in bytes.
            num_blocks: [int] Number of blocks to write.

        Return:
            speed_mbps: [float] Write speed in MB/s.
        """

        # Full path to the test file
        filename = 'tempWriteTestFile'
        full_file_path = os.path.join(os.path.abspath(file_path), filename)

        data = os.urandom(block_size)

        # Sleep for 5 seconds before starting the write test
        time.sleep(5)

        start_time = time.time()

        with open(full_file_path, 'wb') as file:
            for _ in range(num_blocks):
                file.write(data)

        end_time = time.time()
        duration = end_time - start_time
        bytes_written = block_size*num_blocks
        speed_mbps = (bytes_written/1024/1024)/duration

        # Log the result
        if file_path == './':
            self.updateEntry('system_drive_speed', round(speed_mbps, 2))
        else:
            self.updateEntry('data_drive_speed', round(speed_mbps, 2))
        # Clean up the temporary file
        os.remove(full_file_path)

        return speed_mbps


    def getSystemInfo(self):
        """Gathers and updates basic system information, such as OS version and architecture.

        Return:
            [dict] Collected system information including OS version and architecture.
        """

        info = {
            "os_version": platform.platform(),
            "architecture": platform.machine(),
        }

        # Update entries with the gathered information
        self.updateEntry('os_version', info['os_version'])
        self.updateEntry('architecture', info['architecture'])

        return info


    def getModel(self):
        """Retrieves the model information of the system.

        Attempts to read the system's model from specific files. Falls back to 'Unknown Model' if not found.

        Return:
            [str] The model information of the system or 'Unknown Model'.
        """

        try:
            with open("/proc/device-tree/model", "r") as model_file:
                # Read the model information
                model_info = model_file.read().strip()
                model_info = self.cleanNonAscii(model_info)

                # Log the result
                self.updateEntry('model', model_info)

                return model_info
        except IOError:
            try:
                with open("/sys/firmware/devicetree/base/model", "r") as model_file:
                    model_info = model_file.read().strip()
                    model_info = self.cleanNonAscii(model_info)

                    return model_info
            except IOError:
                pass  # Model file could not be read

        return "Unknown Model"


    def checkFreeSpace(self, path='/'):
        """Checks and logs the free disk space of the given path.

        Arguments:
            path: [str] The filesystem path to check disk space for. Defaults to root '/'.

        Return:
            [tuple] Total, used, and free disk space in GB.
        """

        # Get disk usage statistics
        total, used, free = shutil.disk_usage(path)

        # Convert bytes to GB for easier reading
        total_gb = total / 1024 / 1024 / 1024
        used_gb = used / 1024 / 1024 / 1024
        free_gb = free / 1024 / 1024 / 1024

        self.updateEntry('total_gb', round(total_gb, 2))
        self.updateEntry('used_gb', round(used_gb, 2))
        self.updateEntry('free_gb', round(free_gb, 2))

        return total_gb, used_gb, free_gb


    def getRunInfo(self):
        """Performs all set tests (write speed, system info) and updates entries with the results.

        Logs the results of write tests, system information gathering, and disk space checks.
        """

        # Perform a system drive performance test and update PerfMonitor
        write_speed_mbps = self.writeTest()
        log.info("Logged System Drive write speed of {:.2f} MB/s".format(write_speed_mbps))

        # Perform a data drive performance test and update PerfMonitor
        write_speed_mbps = self.writeTest(file_path=self.config.data_dir)
        log.info("Logged Data Drive write speed of {:.2f} MB/s".format(write_speed_mbps))

        # Gather basic system information and update PerfMonitor
        model = self.getModel()
        log.info("Model: {}".format(model))

        info = self.getSystemInfo()
        info_str = ', '.join(f'{key}: {value}' for key, value in info.items())
        log.info("System Information: {}".format(info_str))

        # Gather config settings and update PerfMonitor
        system_drive_description = getattr(self.config, 'system_drive_description', None)
        self.updateEntry('system_drive_desc', system_drive_description)

        data_drive_description = getattr(self.config, 'data_drive_description', None)
        self.updateEntry('data_drive_desc', data_drive_description)

        live_maxpixel_value = getattr(self.config, 'live_maxpixel_enable', None)
        self.updateEntry('live_maxpixel', live_maxpixel_value)

        live_jpg_value = getattr(self.config, 'live_jpg', None)
        self.updateEntry('live_jpg', live_jpg_value)

        slideshow_value = getattr(self.config, 'slideshow_enable', None)
        self.updateEntry('slideshow', slideshow_value)

        hdu_compress_value = getattr(self.config, 'hdu_compress', None)
        self.updateEntry('hdu_compress', hdu_compress_value)

        fireball_detection_value = getattr(self.config, 'enable_fireball_detection', None)
        self.updateEntry('fireball_detection', fireball_detection_value)
