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
        self.manager = Manager()
        self.data_entries = self.manager.dict()
        self.log_file_path = './perfMonitorLog.csv'

        self.fieldnames = [
            'data_dir_name',
            'model',
            'os_version',
            'architecture',
            'system_drive_speed',
            'system_drive_desc',
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


    def updateEntry(self, key, value):
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
        file_path = os.path.join(os.path.abspath(file_path), filename)

        data = os.urandom(block_size)

        # Sleep for 5 seconds before starting the write test
        time.sleep(5)

        start_time = time.time()

        with open(file_path, 'wb') as file:
            for _ in range(num_blocks):
                file.write(data)

        end_time = time.time()
        duration = end_time - start_time
        bytes_written = block_size*num_blocks
        speed_mbps = (bytes_written/1024/1024)/duration

        # Log the result
        if file_path == './':
            self.updateEntry('system_drive_speed', speed_mbps)
        else:
            self.updateEntry('data_drive_speed', speed_mbps)
        # Clean up the temporary file
        os.remove(file_path)

        return speed_mbps


    def getSystemInfo(self):
        # Gather basic system information
        info = {
            "os_version": platform.platform(),
            "architecture": platform.machine(),
        }

        # Update entries with the gathered information
        self.updateEntry('os_version', info['os_version'])
        self.updateEntry('architecture', info['architecture'])

        return info


    def getModel(self):
        try:
            with open("/proc/device-tree/model", "r") as model_file:
                # Read the model information
                model_info = model_file.read().strip()

                # Log the result
                self.updateEntry('model', model_info)

                return model_info
        except IOError:
            try:
                with open("/sys/firmware/devicetree/base/model", "r") as model_file:
                    model_info = model_file.read().strip()
                    return model_info
            except IOError:
                pass  # Model file could not be read

        return "Unknown Model"


    def checkFreeSpace(self, path='/'):
        """
        Check and print the free disk space for the given path in a human-readable format.

        :param path: Path to check disk space for. Defaults to root '/'.
        """
        # Get disk usage statistics
        total, used, free = shutil.disk_usage(path)
        
        # Convert bytes to GB for easier reading
        total_gb = total / 1024 / 1024 / 1024
        used_gb = used / 1024 / 1024 / 1024
        free_gb = free / 1024 / 1024 / 1024
        
        self.updateEntry('total_gb', total_gb)
        self.updateEntry('used_gb', used_gb)
        self.updateEntry('free_gb', free_gb)

        return total_gb, used_gb, free_gb


    def getRunInfo(self):
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



