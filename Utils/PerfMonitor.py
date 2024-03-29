import os
import csv
import time


def writeTest(file_path, block_size, num_blocks):
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

    data = os.urandom(block_size)
    start_time = time.time()

    with open(file_path, 'wb') as file:
        for _ in range(num_blocks):
            file.write(data)

    end_time = time.time()
    duration = end_time - start_time
    bytes_written = block_size*num_blocks
    speed_mbps = (bytes_written/1024/1024)/duration

    # Clean up the temporary file
    os.remove(file_path)

    return speed_mbps


def logToCsv(log_file_path, data):
    """
    Log data to a CSV file.

    Arguments:
        log_file_path: [path] Path to the log CSV file.
        data: Dictionary containing the test results to log.
    """
    file_exists = os.path.isfile(log_file_path)
    with open(log_file_path, 'a', newline='') as csvfile:
        fieldnames = ['timestamp', 'write_speed_mbps']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)
