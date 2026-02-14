""" Functions for uploading files to server via SFTP. """

import logging
import os
import ctypes
import multiprocessing
import time
from queue import Empty
from multiprocessing import Manager

import paramiko

from RMS.Logger import LoggingManager, getLogger
from RMS.Misc import mkdirP, UTCFromTimestamp, runWithTimeout

# Suppress Paramiko internal errors before they appear in logs
getLogger("paramiko.transport").setLevel(logging.CRITICAL)
getLogger("paramiko.auth_handler").setLevel(logging.CRITICAL)

QueueEmpty = Empty

# Get the logger from the main module
log = logging.getLogger("rmslogger")


def existsRemoteDirectory(sftp,path):
    """ Check if a directory exists on the remote server.

    Arguments:
        sftp: [paramiko.SFTPClient object] Connection handle.
        path: [str] Path to the directory to be checked.

    Returns:
        [bool] True if the directory exists, False otherwise.
    """

    try:
        # Get files in directory above target
        listing = sftp.listdir(os.path.dirname(path))

        # Is the required directory name in the filelist
        if os.path.basename(path) in listing:

            # Is the required directory name actually a directory
            sftp_return = str(sftp.stat(path))

            if sftp_return[0] == 'd':
                return True
            else:
                log.error("{} must be a directory, but was not.".format(path))
                log.error("stat returns {}".format(sftp_return))
                return False
        else:
            return False
        
    except Exception:
        log.error("Failure whilst checking that directory {} exists".format(path))
        return False

def createRemoteDirectory(sftp, path):
    """ Recursively create a directory tree on the remote server.
    
    Arguments:
        sftp: [paramiko.SFTPClient object] Connection handle.
        path: [str] Path to the directory to be created.

    Return:
        [bool] True if successful, False otherwise.
    """

    # Check if the path is absolute
    is_abspath = False
    if path.startswith('/') or path.startswith('\\'):
        is_abspath = True

    try:
        # Split the path into segments
        folders = []
        while path not in ('', '/', '\\'):
            path, folder = os.path.split(path)
            if folder:
                folders.append(folder)
        
        # Reverse the list to create from top to bottom
        folders.reverse()

        # Recursively create directory tree
        path = ''
        for i, folder in enumerate(folders):

            # Join the path (if it's the first folder, don't add a slash in front to avoid make it absolute)
            if (i == 0) and (not is_abspath):
                path = folder
            else:
                path = path + '/' + folder

            # Check if the directory exists
            try:
                sftp.stat(path)
                log.debug("Directory '{}' already exists.".format(path))

            except FileNotFoundError:

                sftp.mkdir(path)
                log.debug("Directory '{}' created.".format(path))
            
            except Exception as e:
                log.error("Unable to stat directory '{}': {}".format(path, e))
                return False
        
        return True
    

    except Exception as e:

        # Log the exception (assuming a logging setup is in place)
        log.error("Unable to create directory '{0}': {1}".format(path, e))
        return False


def getSSHClient(hostname,
                 port=22,
                 username=None,
                 key_filename=None,
                 timeout=300,
                 banner_timeout=300,
                 auth_timeout=300,
                 keepalive_interval=30,
                 hard_timeout=600):
    """
    Establishes an SSH connection and returns an SSH client.
    Handles key-based authentication first, then falls back to the SSH agent.
    Returns an SSH client or None.

    The hard_timeout parameter provides a backstop timeout that wraps the entire
    connection attempt, in case paramiko's internal timeouts don't trigger properly
    (e.g., due to network edge cases or half-open connections).
    """
    log.debug("Paramiko version: {}".format(paramiko.__version__))
    log.debug("Establishing SSH connection to: {}:{}...".format(hostname, port))

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    def doConnectWithKey():
        """Inner function to attempt SSH connection with key file."""
        ssh.connect(
            hostname,
            port=port,
            username=username,
            key_filename=key_filename,
            timeout=timeout,
            banner_timeout=banner_timeout,
            auth_timeout=auth_timeout,
            look_for_keys=False,
            allow_agent=False  # Avoid potential hangs from broken SSH agent (paramiko #2147)
        )
        return True

    # Try key_filename first if provided
    if key_filename:
        # Use hard timeout wrapper around ssh.connect()
        success, result, exception = runWithTimeout(doConnectWithKey, timeout=hard_timeout)

        if not success:
            log.error("SSH connection timed out after {} seconds (hard timeout)".format(hard_timeout))
            try:
                ssh.close()
            except Exception:
                pass
            return None

        if exception is not None:
            # Handle specific exceptions
            if isinstance(exception, paramiko.SSHException):
                log.warning("SSH error with provided key: {}".format(str(exception)))
            elif isinstance(exception, ValueError):
                log.warning("Key validation error.")
            elif isinstance(exception, paramiko.AuthenticationException):
                log.warning("Server rejected our key - it may not be authorized")
            elif isinstance(exception, IOError):
                log.warning("IO error with key file: {}".format(str(exception)))
            else:
                log.warning("Unexpected error with key file: {}".format(str(exception)))

            # Close the client before attempting agent auth to ensure clean state
            try:
                ssh.close()
            except Exception as e:
                # Log cleanup failures at debug level; do not interrupt fallback auth.
                log.debug("Error while closing SSH client after key auth failure: %s", e)
        else:
            # Connection successful
            log.debug("SSHClient connected successfully (key file).")

            transport = ssh.get_transport()
            if transport and keepalive_interval > 0:
                transport.set_keepalive(keepalive_interval)
                log.debug("Keepalive set to {} seconds".format(keepalive_interval))

            return ssh

    # Try agent-based authentication if key auth fails or was not attempted
    # Create a fresh SSHClient to ensure clean state
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    def doConnectWithAgent():
        """Inner function to attempt SSH connection with agent fallback."""
        ssh.connect(
            hostname,
            port=port,
            username=username,
            look_for_keys=True,
            timeout=timeout,
            banner_timeout=banner_timeout,
            auth_timeout=auth_timeout
        )
        return True

    # Use hard timeout wrapper around ssh.connect()
    success, result, exception = runWithTimeout(doConnectWithAgent, timeout=hard_timeout)

    if not success:
        log.error("SSH connection timed out after {} seconds (hard timeout)".format(hard_timeout))
        try:
            ssh.close()
        except Exception as e:
            log.debug("Failed to close SSH client after timeout: %s", e)
        return None

    if exception is not None:
        if isinstance(exception, paramiko.AuthenticationException):
            log.warning("Agent authentication failed. No valid authorized keys found.")
        else:
            log.warning("SSH connection failed during agent fallback: {}".format(str(exception)))
        return None

    log.debug("SSHClient connected via agent fallback.")

    transport = ssh.get_transport()
    if transport and keepalive_interval > 0:
        transport.set_keepalive(keepalive_interval)
        log.debug("Keepalive set to {} seconds".format(keepalive_interval))

    return ssh

def getSFTPClient(ssh, sftp_timeout=300):
    """
    Opens an SFTP session from an established SSH client.
    If SFTP fails or times out, logs the error and returns None.

    Arguments:
        ssh: [paramiko.SSHClient] An established SSH client.
        sftp_timeout: [int] Maximum time in seconds to wait for SFTP session to open.
            Default is 300 seconds (5 minutes) to accommodate slow connections.
    """
    if ssh is None:
        log.error("Cannot open SFTP session: SSH client is None.")
        return None

    log.debug("Attempting to open SFTP connection...")

    # Use hard timeout wrapper around open_sftp() to prevent indefinite hangs
    success, sftp, exception = runWithTimeout(ssh.open_sftp, timeout=sftp_timeout)

    if not success:
        log.error("SFTP connection timed out after {} seconds".format(sftp_timeout))
        return None

    if exception is not None:
        log.error("Failed to open SFTP connection: {}".format(exception))
        return None

    log.debug("SFTP connection established.")
    return sftp


def getSSHAndSFTP(hostname, sftp_timeout=300, **kwargs):
    """
    Wrapper function that returns both SSH and SFTP clients.
    If SSH fails, SFTP is not attempted.

    Arguments:
        hostname: [str] Server name or IP address.
        sftp_timeout: [int] Maximum time in seconds to wait for SFTP session to open.
        **kwargs: Additional arguments passed to getSSHClient.
    """
    ssh = getSSHClient(hostname, **kwargs)
    sftp = getSFTPClient(ssh, sftp_timeout=sftp_timeout) if ssh else None
    return ssh, sftp


# Define a class to track progress and share values between the callback and the main thread
class ProgressTracker(object):
    def __init__(self, total_bytes):
        self.total_bytes = total_bytes
        self.uploaded_bytes = 0
        self.last_percent = 0
        self.start_time = time.time()


def uploadSFTP(hostname, username, dir_local, dir_remote, file_list, port=22,
               rsa_private_key=os.path.expanduser('~/.ssh/id_rsa'),
               allow_dir_creation=False,
               connect_timeout=300,
               banner_timeout=300,
               auth_timeout=300,
               keepalive_interval=30):
    """ Upload the given list of files using SFTP with progress reporting.
        The upload only supports uploading files from one local directory to one remote directory.
        The files are uploaded only if they do not already exist on the server, or if they are of 
        different size than the local files.

        The RSA private key is used for authentication. If the key is not found, the function will try to
        use the keys from the SSH agent (if available). Passphrase-protected keys are not supported.

    Arguments:
        hostname: [str] Server name or IP address.
        username: [str] Username used for connecting to the server.
        dir_local: [str] Path to the local directory where the local files are located.
        dir_remote: [str] Path on the server where the files will be stored. It can be relative to the user's
            home directory, or an absolute path.
        file_list: [list or strings] A list of files to be uploaded to the server. These should only be
            file names, not full paths. The full path is constructed from the dir_local (on the local 
            machine) and the dir_remote (on the server).

    Keyword arguments:
        port: [int] SSH port. 22 by default.
        rsa_private_key: [str] Path to the SSH private key. ~/.ssh/id_rsa by default.
        allow_dir_creation: [bool] Create a remote directory if it doesn't exist. False by default.

    Return:
        [bool] True if upload successful, false otherwise.
    """

    # If the file list is empty, don't do anything
    if not file_list:
        log.info('No files to upload!')
        return True

    # Connect and use paramiko SFTP to negotiate SSH2 across the connection
    # The whole thing is in a try block because if an error occurs, the connection will be closed at the end

    ssh = None
    sftp = None

    try:
        # Connect with timeouts
        ssh, sftp = getSSHAndSFTP(
            hostname,
            port=port,
            username=username,
            key_filename=rsa_private_key,
            timeout=connect_timeout,
            banner_timeout=banner_timeout,
            auth_timeout=auth_timeout,
            keepalive_interval=keepalive_interval
        )

        # Optionally ensure remote directory exists
        if allow_dir_creation:
            log.debug("Checking/creating remote dir '{}'".format(dir_remote))
            if not existsRemoteDirectory(sftp, dir_remote):
                createRemoteDirectory(sftp, dir_remote)

        # Verify remote dir
        try:
            sftp.stat(dir_remote)
        except Exception as e:
            log.error("Remote directory '{}' does not exist or is not accessible: {}".format(dir_remote, e))
            return False

        # Go through all files
        for fname in file_list:

            # Path to the local file
            local_file = os.path.join(dir_local, fname)

            # Get the size of the local file
            local_file_size = os.lstat(local_file).st_size

            # Path to the remote file
            remote_file = dir_remote + '/' + os.path.basename(fname)

            # Check if the remote file already exists and skip it if it has the same size as the local file
            try:
                remote_info = sftp.lstat(remote_file)
                
                # If the remote and the local file are of the same size, skip it
                if local_file_size == remote_info.st_size:
                    log.debug("The file '{}' already exists on the server and is the same size. Skipping.".format(remote_file))
                    continue
            
            except IOError as e:
                # Means remote file doesn't exist yet, so proceed
                pass
            
            # Initialize the progress tracker
            tracker = ProgressTracker(local_file_size)

            update_interval = 1  # Update progress every 1% for large files or 5% for small files
            if tracker.total_bytes > 100*1024*1024:  # For files over 100MB, update more frequently
                update_interval = 0.5

            # Define a callback function for progress reporting
            def progressCallback(bytes_transferred, _):
                """ Callback function to report upload progress. 
                    It updates the tracker and prints the progress to the console.
                """

                tracker.uploaded_bytes = bytes_transferred
                
                # Calculate percentage
                if tracker.total_bytes > 0:
                    percent_complete = round(100.0*tracker.uploaded_bytes/tracker.total_bytes, 1)
                    
                    # Only update when the percentage changes by at least update_interval
                    # Also prevent duplicate 100% messages
                    if (percent_complete >= tracker.last_percent + update_interval and tracker.last_percent < 100.0) \
                        or (percent_complete == 100.0 and tracker.last_percent != 100.0):
                        
                        elapsed_time = time.time() - tracker.start_time
                        
                        # Calculate transfer speed
                        if elapsed_time > 0:
                            transfer_rate = tracker.uploaded_bytes/elapsed_time/1024  # KB/s
                            
                            # Format as MB/s if over 1024 KB/s
                            if transfer_rate > 1024:
                                transfer_rate_str = "{:.2f} MB/s".format(transfer_rate/1024)
                            else:
                                transfer_rate_str = "{:.2f} KB/s".format(transfer_rate)
                            
                            # Estimate time remaining
                            if percent_complete > 0 and percent_complete < 100.0:
                                time_remaining = (elapsed_time/percent_complete)*(100 - percent_complete)
                                # Format time remaining
                                if time_remaining > 60:
                                    time_str = "{:.1f} min remaining".format(time_remaining/60)
                                else:
                                    time_str = "{:.0f} sec remaining".format(time_remaining)
                                
                                print('[{:.1f}%] Uploading: {} ({}/{}) @ {} - {}'.format(
                                    percent_complete,
                                    os.path.basename(local_file),
                                    formatSize(tracker.uploaded_bytes),
                                    formatSize(tracker.total_bytes),
                                    transfer_rate_str,
                                    time_str,
                                    end=''
                                ))
                            else:
                                # At 100%, show "complete" instead of remaining time
                                if percent_complete == 100.0:
                                    print('[100.0%] Upload complete: {} ({}/{}) @ {}'.format(
                                        os.path.basename(local_file),
                                        formatSize(tracker.uploaded_bytes),
                                        formatSize(tracker.total_bytes),
                                        transfer_rate_str
                                    ))
                                else:
                                    print('[{:.1f}%] Uploading: {} ({}/{}) @ {}'.format(
                                        percent_complete,
                                        os.path.basename(local_file),
                                        formatSize(tracker.uploaded_bytes),
                                        formatSize(tracker.total_bytes),
                                        transfer_rate_str,
                                        end=''
                                    ))
                        
                        else:
                            print('[{:.1f}%] Uploading: {} ({}/{})'.format(
                                percent_complete,
                                os.path.basename(local_file),
                                formatSize(tracker.uploaded_bytes),
                                formatSize(tracker.total_bytes), 
                                end=''
                            ))
                        
                        tracker.last_percent = percent_complete
            
            # Upload the file to the server if it isn't already there
            log.info('Starting upload of ' \
                     + local_file + ' ({}) to '.format(formatSize(local_file_size)) + remote_file)
            sftp.put(local_file, remote_file, callback=progressCallback)
            log.debug("Upload completed, verifying...")

            # Check that the size of the remote file is correct, indicating a successful upload
            remote_info = sftp.lstat(remote_file)
            
            # If the remote and the local file are of the same size, skip it
            if local_file_size != remote_info.st_size:
                log.error('File verification failed: local size {} != remote size {}'.format(
                    formatSize(local_file_size), formatSize(remote_info.st_size)))
                return False

            log.debug("File upload verified: {:s}".format(remote_file))
            
        return True

    except Exception as e:
        log.error("Exception during SFTP upload: {}".format(e), exc_info=True)
        return False

    finally:
        # Close SFTP and SSH if open
        if sftp is not None:
            log.debug("Closing SFTP channel")
            sftp.close()
        if ssh is not None:
            log.debug("Closing SSH client connection")
            ssh.close()

# Helper function to format file sizes in human-readable format
def formatSize(size_bytes):
    """Format a size in bytes into a human-readable string"""
    if size_bytes < 1024:
        return "{} B".format(size_bytes)
    elif size_bytes < 1024*1024:
        return "{:.2f} KB".format(size_bytes/1024)
    elif size_bytes < 1024*1024*1024:
        return "{:.2f} MB".format(size_bytes/(1024*1024))
    else:
        return "{:.2f} GB".format(size_bytes/(1024*1024*1024))

class UploadManager(multiprocessing.Process):
    def __init__(self, config):
        """ Uploads all processed data which has not yet been uploaded to the server. The files will be tried 
            to be uploaded every 15 minutes, until successful. 
        
        """


        super(UploadManager, self).__init__()

        self.config = config

        # These will be defined in .run()
        self._mgr = Manager()
        self.file_queue      = self._mgr.Queue()
        self.file_queue_lock = self._mgr.Lock()

        # Construct the path to the queue backup file
        self.upload_queue_file_path = os.path.join(self.config.data_dir, self.config.upload_queue_file)

        self.exit = multiprocessing.Event()
        self.upload_in_progress = multiprocessing.Value(ctypes.c_bool, False)

        # These timing variables must be shared between processes using multiprocessing.Value() because
        # 
        # - When upload_manager.start() is called, it creates a NEW process with its own memory space
        # - The parent process (StartCapture) and child process (UploadManager) have separate instances
        # - Parent calls addFiles() and delayNextUpload() to control timing
        # - Child runs the upload loop and checks these timing variables
        # - Without shared memory, parent's timing changes are invisible to child process
        # - This caused upload delays to be ignored and coordination issues during shutdown
        #
        # Using multiprocessing.Value() creates shared memory that both processes can access
        # Value format: 'd' = double precision float (for timestamp storage)
        # Convention: 0.0 = None/not set, >0.0 = unix timestamp
        
        # Time when the upload was run last
        self.last_runtime = multiprocessing.Value('d', 0.0)
        self.last_runtime_lock = multiprocessing.Lock()

        # Time when the next upload should be run (used for delaying the upload)
        self.next_runtime = multiprocessing.Value('d', 0.0)
        self.next_runtime_lock = multiprocessing.Lock() 

        


    def start(self):
        """ Starts the upload manager. """

        super(UploadManager, self).start()



    def stop(self, timeout=60):
        """ Stops the upload manager.
        
        Keyword arguments:
            timeout: [int] Maximum time to wait for the upload manager to stop, in seconds. Default is 60 seconds.
        """

        self.exit.set()
        self.join(timeout)
        if not self.is_alive():
            log.info("UploadManager stopped successfully.")
            return
        
        log.warning("UploadManager did not stop within the timeout period of {} seconds.".format(timeout))
        self.terminate()

        short_wait = 5
        self.join(short_wait)
        if self.is_alive():
            log.error(
                "UploadManager still alive after terminate() & %d more seconds. "
                "It may be stuck in a non-interruptible blocking call.",
                short_wait
            )
        else:
            log.info("UploadManager terminated (after forced terminate).")

        # Always join to reap zombie (returns instantly if already dead)
        self.join()



    def addFiles(self, file_list):
        """ Adds a list of files to be uploaded to the queue. """
        
        # Load the existing items in the queue (can't be under lock, as it would block the queue)
        existing_items = set(self.getFileList())

        # Add new files to the queue
        with self.file_queue_lock:

            new_files = [f for f in file_list if f not in existing_items]

            for file_name in new_files:
                self.file_queue.put(file_name)

        time.sleep(0.1)

        # Save only if new files were added
        if new_files:

            self.saveQueue(overwrite=False)

            # Make sure the data gets uploaded right away
            with self.last_runtime_lock:
                self.last_runtime.value = 0.0


    def getFileList(self):
        """ Safely get a snapshot of all items in the queue, preserving order, with thread/process lock. """

        items = []

        with self.file_queue_lock:

            # Drain the queue
            while True:
                try:
                    item = self.file_queue.get_nowait()
                    items.append(item)
                except Empty:
                    break
                except Exception as e:
                    log.error("Unexpected error while draining file_queue: {}".format(e), exc_info=True)
                    break

            # Restore the queue
            for item in items:
                self.file_queue.put(item)

        return items


    def loadQueue(self):
        """ Load a list of files to be uploaded from a file. """

        # Check if the queue file exists, if not, create it
        if not os.path.exists(self.upload_queue_file_path):
            self.saveQueue(overwrite=True)
            return None


        # Phase 1: read file from disk without holding the lock
        filenames = []
        if os.path.exists(self.upload_queue_file_path):

            with open(self.upload_queue_file_path) as f:
                
                for file_name in f:

                    file_name = file_name.replace('\n', '').replace('\r', '')

                    # Skip empty names
                    if len(file_name) == 0:
                        continue

                    # Make sure the file for upload exists
                    if not os.path.isfile(file_name):
                        log.warning("Local file not found: {:s}".format(file_name))
                        log.warning("Skipping it...")
                        continue

                    # Add the file name to the list
                    filenames.append(file_name)


        # Phase 2: compare with what's already in the queue
        existing = set(self.getFileList())   # getFileList briefly acquires + releases the lock
        to_enqueue = [fn for fn in filenames if fn not in existing]

        # Phase 3: add only missing files under the lock
        if to_enqueue:
            with self.file_queue_lock:
                for fn in to_enqueue:
                    self.file_queue.put(fn)



    def saveQueue(self, overwrite=False):
        """ Save the list of file to upload to disk, for bookkeeping in case of a power failure. 
    
        Keyword arguments:
            overwrite: [bool] If True, the holding file will be overwritten. Otherwise (default), the entries
                that are not in the file will be added at the end of the file.
        """

        # Convert the queue to a list
        file_list = self.getFileList()

        # Make the data directory if it doesn't exist
        mkdirP(self.config.data_dir)

        # If overwrite is true, save the queue to the holding file completely
        if overwrite:

            # Create the queue file
            with open(self.upload_queue_file_path, 'w') as f:
                for file_name in file_list:
                    f.write(file_name + '\n')

        else:

            # Load the list from the file and make sure to write only the entries not already in the file

            # Get a list of entries in the holding file
            existing_list = []
            try:
                with open(self.upload_queue_file_path) as f:
                    for file_name in f:
                        file_name = file_name.replace('\n', '').replace('\r', '')
                        existing_list.append(file_name)
            except FileNotFoundError:
                log.warning("Upload queue file not found: {:s}".format(self.upload_queue_file_path))
                log.warning("Creating a new upload queue file.")

                # If the file does not exist, create it
                with open(self.upload_queue_file_path, 'w') as f:
                    pass

            # Save to disk only those entires which are not already there
            with open(self.upload_queue_file_path, 'a') as f:
                for file_name in file_list:
                    if file_name not in existing_list and file_name is not None:
                        f.write(file_name + '\n')




    def uploadData(self, retries=5):
        """ Pulls the upload list from a file, tries to upload the file, and if it fails it saves the list of 
            failed files to disk. 

        Keyword arguments:
            retries: [int] Number of tried to upload a file before giving up. Default is 5.
        """

        # Skip uploading if the upload is already in progress
        if self.upload_in_progress.value:
            return


        # Set flag that the upload as in progress
        self.upload_in_progress.value = True

        try:
            # Read the file list from disk
            self.loadQueue()

            tries = 0

            # Go through every file and upload it to server
            while True:

                # Get a file from the queue
                with self.file_queue_lock:
                    try:
                        file_name = self.file_queue.get(timeout=1)
                    except QueueEmpty:
                        break  # nothing left to do

                if not os.path.isfile(file_name):
                    log.warning("Local file not found: {:s}".format(file_name))
                    log.warning("Skipping it...")
                    continue

                # Separate the path to the file and the file name
                data_path, f_name = os.path.split(file_name)

                # Upload the file via SFTP (use the lowercase version of the station ID as the username)
                upload_status = uploadSFTP(self.config.hostname, self.config.stationID.lower(), data_path, \
                    self.config.remote_dir, [f_name], rsa_private_key=self.config.rsa_private_key, 
                    port=self.config.host_port)

                # If the upload was successful, rewrite the holding file, which will remove the uploaded file
                if upload_status:
                    log.info('Upload successful!')
                    self.saveQueue(overwrite=True)
                    tries = 0

                # If the upload failed, put the file back on the list and wait a bit
                else:

                    log.warning('Uploading failed! Retry {:d} of {:d}'.format(tries + 1, retries))

                    tries += 1 
                    with self.file_queue_lock:
                        self.file_queue.put(file_name)

                    # Progressive retry delay: 15s * 2^(attempt-1), capped at 8 minutes
                    delay = min(15*(2**(tries - 1)), 480)
                    if delay < 60:
                        log.info("Waiting %.0f s before next retry...", delay)
                    else:
                        log.info("Waiting %.1f min before next retry...", delay/60)
                    time.sleep(delay)

                # Check if the upload was tried too many times
                if tries >= retries:
                    break

        finally:
            # Set the flag that the upload is done
            self.upload_in_progress.value = False


    def delayNextUpload(self, delay=0):
        """ Delay the upload by the given number of seconds from now. Zero by default. """

        # Set the next run time using a delay
        with self.next_runtime_lock:
            self.next_runtime.value = time.time() + delay

            if delay > 0:
                # Log the delay
                next_time_str = UTCFromTimestamp.utcfromtimestamp(self.next_runtime.value).strftime('%Y-%m-%d %H:%M:%S')
                log.info("Upload delayed for {:.1f} min until {:s} UTC".format(delay/60, next_time_str))
            else:
                # Log that the upload will run immediately
                log.info("Upload will run immediately")



    def run(self):
        """ Try uploading the files every 15 minutes. """

        # Load the file queue from disk
        self.loadQueue()

        with self.last_runtime_lock:
            self.last_runtime.value = 0.0

        while not self.exit.is_set():

            with self.last_runtime_lock:

                # Check if the upload should be run (if 15 minutes are up)
                if self.last_runtime.value > 0:
                    if (time.time() - self.last_runtime.value) < 15*60:
                        time.sleep(1)
                        continue

            with self.next_runtime_lock:

                # Check if the upload delay is up
                if self.next_runtime.value > 0:
                    if time.time() < self.next_runtime.value:
                        time.sleep(1)
                        continue

            with self.last_runtime_lock:
                self.last_runtime.value = time.time()

            # Run the upload procedure
            self.uploadData()

            time.sleep(0.1)




if __name__ == "__main__":

    # Set up a fake config file
    class FakeConf(object):
        def __init__(self):

            self.username = 'dvida'

            # remote hostname where SSH server is running
            self.hostname = 'gmn.uwo.ca'
            self.host_port = 22
            self.remote_dir = 'files'
            self.stationID = 'dvida'
            self.rsa_private_key = os.path.expanduser("~/.ssh/id_rsa")

            self.upload_queue_file = 'FILES_TO_UPLOAD.inf'

            self.data_dir = os.path.join(os.path.expanduser('~'), 'RMS_data')
            self.log_dir = 'logs'


    config = FakeConf()

    dir_local = '/home/dvida/Desktop'

    # # Test uploading a single file
    # uploadSFTP(
    #     config.hostname, config.stationID, 
    #     dir_local, dir_remote, file_list, 
    #     rsa_private_key=config.rsa_private_key
    #     )
    

    # Test directly uploading files and remote directory creation
    dir_local = "C:\\temp\\dir2\\dir3"
    remote_dir = "files/upload_test/dir2/dir3"
    uploadSFTP(
        config.hostname, config.stationID, 
        dir_local, remote_dir, 
        ['test.txt'], 
        rsa_private_key=config.rsa_private_key,
        allow_dir_creation=True
        )
    
    sys.exit()


    ### Test the upload manager ###

    # Init the logger
    log_manager = LoggingManager()
    log_manager.initLogging(config)

    up = UploadManager(config)
    up.start()

    time.sleep(2)

    up.addFiles([os.path.join(dir_local, 'test.txt')])

    time.sleep(1)

    up.addFiles([os.path.join(dir_local, 'test2.txt')])
    up.addFiles([os.path.join(dir_local, 'test3.txt')])

    up.uploadData()

    up.stop()

    ### ###