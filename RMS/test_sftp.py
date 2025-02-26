#!/usr/bin/env python3
import os
import sys
import time
import math
import random
import string
import logging
import paramiko
import multiprocessing

print(f"Python version: {sys.version}")
print(f"Paramiko version: {paramiko.__version__}")

#############################################
# CONFIGURE YOUR REMOTE HOST HERE
#############################################
HOSTNAME = "gmn.uwo.ca"
USERNAME = "lbusquin"
PORT     = 22
RSA_KEY  = os.path.expanduser("~/.ssh/id_rsa")
REMOTE_DIR = "/home/lbusquin/files/upload_test"


#############################################
# TEST SETTINGS
#############################################
NUM_UPLOADERS = 3         # How many processes each uploading a big file
UPLOAD_FILE_MB = 200      # Size (MB) of each test file
NUM_CPU_BURNERS = 2       # How many processes to just burn CPU
NUM_DISK_WRITERS = 2      # How many processes to repeatedly write random data
TEST_DURATION = 120       # How long disk writers and CPU burners run (seconds)

#############################################
# LOGGING SETUP
#############################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(processName)s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

def create_large_local_file(filename, size_mb):
    """Create a file of ~size_mb MB of random binary data."""
    block_size = 1024 * 1024  # 1MB
    log.info(f"[create_large_local_file] Creating {filename} (~{size_mb} MB)...")
    with open(filename, "wb") as f:
        for _ in range(size_mb):
            f.write(os.urandom(block_size))
    log.info(f"[create_large_local_file] Done creating {filename}.")


def upload_file_sshclient(local_file, remote_filename):
    """Paramiko SSHClient-based upload of one file."""
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        log.info(f"[upload_file_sshclient] Connecting to {HOSTNAME}:{PORT}")
        ssh.connect(HOSTNAME, port=PORT, username=USERNAME, key_filename=RSA_KEY)
        log.info("[upload_file_sshclient] SSH connect() succeeded.")

        sftp = ssh.open_sftp()
        remote_path = os.path.join(REMOTE_DIR, remote_filename)

        local_size = os.path.getsize(local_file)
        log.info(f"[upload_file_sshclient] Starting put({local_file} -> {remote_path}), size={local_size} bytes")

        start_time = time.time()
        sftp.put(local_file, remote_path)
        elapsed = time.time() - start_time
        log.info(f"[upload_file_sshclient] Upload finished in {elapsed:.1f} sec")

        # Verify
        rstat = sftp.stat(remote_path)
        if rstat.st_size != local_size:
            log.error(f"Size mismatch! local={local_size}, remote={rstat.st_size}")
        else:
            log.info("[upload_file_sshclient] Upload verified OK.")

        sftp.close()
        ssh.close()
    except Exception as e:
        log.error("[upload_file_sshclient] Exception:", exc_info=True)


def uploader_process(file_size_mb):
    """
    Each uploader process makes its own large file, then uploads it to the server.
    """
    pid = os.getpid()
    filename = f"uploader_{pid}_{file_size_mb}MB.bin"
    create_large_local_file(filename, file_size_mb)

    # Upload
    remote_filename = os.path.basename(filename)
    upload_file_sshclient(filename, remote_filename)

    # Clean up local file
    if os.path.exists(filename):
        os.remove(filename)
        log.info(f"[uploader_process] Removed local test file {filename}")


def cpu_burner(duration=30):
    """
    Burns CPU for `duration` seconds doing random math loops.
    """
    start = time.time()
    ops = 0
    log.info(f"[cpu_burner] Starting CPU burn for {duration} sec...")
    while time.time() - start < duration:
        dummy = 0
        for i in range(10000):
            dummy += math.sin(i) * math.sqrt(i)
        ops += 1
    log.info(f"[cpu_burner] Completed {ops} big loops in {duration} sec")


def disk_writer(duration=30):
    """
    Continuously writes random data to a temp file for `duration` seconds, 
    then deletes the file.
    """
    start = time.time()
    temp_name = f"disk_writer_{os.getpid()}.bin"
    log.info(f"[disk_writer] Writing to {temp_name} for {duration} sec...")

    f = open(temp_name, "wb")
    while time.time() - start < duration:
        f.write(os.urandom(1024 * 1024))  # 1MB write
        f.flush()
    f.close()

    if os.path.exists(temp_name):
        os.remove(temp_name)
    log.info(f"[disk_writer] Done, removed {temp_name}")


def main():
    log.info("Starting torture test with:")
    log.info(f"   - {NUM_UPLOADERS} uploaders, each uploading ~{UPLOAD_FILE_MB}MB file")
    log.info(f"   - {NUM_CPU_BURNERS} CPU burners")
    log.info(f"   - {NUM_DISK_WRITERS} disk writers")
    log.info(f"   - Duration of CPU/Disk tasks: {TEST_DURATION} seconds")

    processes = []

    # Start uploaders
    for _ in range(NUM_UPLOADERS):
        p = multiprocessing.Process(target=uploader_process, args=(UPLOAD_FILE_MB,))
        p.start()
        processes.append(p)

    # Start CPU burners
    for _ in range(NUM_CPU_BURNERS):
        p = multiprocessing.Process(target=cpu_burner, args=(TEST_DURATION,))
        p.start()
        processes.append(p)

    # Start disk writers
    for _ in range(NUM_DISK_WRITERS):
        p = multiprocessing.Process(target=disk_writer, args=(TEST_DURATION,))
        p.start()
        processes.append(p)

    # Wait for all to finish
    for p in processes:
        p.join()
        log.info(f"[main] Process {p.name} joined, exit code={p.exitcode}")

    log.info("[main] All processes completed.")


if __name__ == "__main__":
    main()