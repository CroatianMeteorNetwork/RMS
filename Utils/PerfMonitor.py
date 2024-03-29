import os
import csv
import time
import platform

from threading import Lock

class PerfMonitor:
    def __init__(self, night_data_dir_name):
        self.data_entries = {}
        self.log_file_path = './perfMonitorLog.csv'
        self.fieldnames = ['data_dir_name', 'write_speed_mbps', 'res', 'calc_fps', 'media_backend', 'media_backend_ovr',
                           'live_maxpixel', 'live_jpg', 'slideshow', 'hdu_compress', 'fireball_detection',
                           'jitter_quality', 'dropped_frame_rate', 'os_version', 'architecture', 'cpu']
        self.night_data_dir_name = night_data_dir_name
        self.lock = Lock()


    def updateEntry(self, key, value):
        with self.lock:
            if self.night_data_dir_name not in self.data_entries:
                # Initialize a new entry if this is the first value for this data_dir_name
                self.data_entries[self.night_data_dir_name] = {fieldname: None for fieldname in self.fieldnames}
            self.data_entries[self.night_data_dir_name][key] = value


    def logToCsv(self):
        with self.lock:
            with open(self.log_file_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writeheader()
                for data_dir, entry in self.data_entries.items():
                    entry['data_dir_name'] = data_dir  # Ensure the key is part of the entry
                    writer.writerow(entry)
            # Clear entries after logging if they should not be logged again
            self.data_entries.clear()


    def writeTest(self, file_path, block_size=1024*1024, num_blocks=100):
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
        self.updateEntry('write_speed_mbps', speed_mbps)
        # Clean up the temporary file
        os.remove(file_path)

        return speed_mbps


    def getSystemInfo(self):
        # Gather basic system information
        info = {
            "os_version": platform.platform(),
            "architecture": platform.machine(),
            "cpu": platform.processor(),
        }

        # Update entries with the gathered information
        self.updateEntry('os_version', info['os_version'])
        self.updateEntry('architecture', info['architecture'])
        self.updateEntry('cpu', info['cpu'])

        return info
