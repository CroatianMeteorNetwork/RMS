""" Functions for uploading files to server via SFTP. """

from __future__ import print_function, division, absolute_import

import sys
import os
import ctypes
import multiprocessing
import time
import datetime
import logging

import binascii
import paramiko

# Suppress Paramiko internal errors before they appear in logs
logging.getLogger("paramiko.transport").setLevel(logging.CRITICAL)
logging.getLogger("paramiko.auth_handler").setLevel(logging.CRITICAL)

try:
    # Python 2
    import Queue

except:
    # Python 3
    import queue as Queue



from RMS.Misc import mkdirP, RmsDateTime

# Map FileNotFoundError to IOError in Python 2 as it does not exist
if sys.version_info[0] < 3:
    FileNotFoundError = IOError

# Get the logger from the main module
log = logging.getLogger("logger")


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
        
    except:
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
                print("Directory '{}' already exists.".format(path))

            except FileNotFoundError:

                sftp.mkdir(path)
                print("Directory '{}' created.".format(path))
            
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
                 keepalive_interval=30):
    """
    Establishes an SSH connection and returns an SSH client.
    Handles key-based authentication first, then falls back to the SSH agent.
    Returns an SSH client or None.
    """
    log.info("Paramiko version: {}".format(paramiko.__version__))
    log.info("Establishing SSH connection to: {}:{}...".format(hostname, port))

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Try key_filename first if provided
    if key_filename:
        try:
            ssh.connect(
                hostname,
                port=port,
                username=username,
                key_filename=key_filename,
                timeout=timeout,
                banner_timeout=banner_timeout,
                auth_timeout=auth_timeout,
                look_for_keys=False
            )
            log.info("SSHClient connected successfully (key file).")

            transport = ssh.get_transport()
            if transport and keepalive_interval > 0:
                transport.set_keepalive(keepalive_interval)
                log.info("Keepalive set to {} seconds".format(keepalive_interval))

            return ssh

        except paramiko.SSHException as e:
            log.warning("SSH error with provided key: {}".format(str(e)))
        except ValueError:
            log.warning("Key validation error.")
        except paramiko.AuthenticationException:
            log.warning("Server rejected our key - it may not be authorized")
        except IOError as e:
            log.warning("IO error with key file: {}".format(str(e)))
        except Exception as e:
            log.warning("Unexpected error with key file: {}".format(str(e)))

    # Try agent-based authentication if key auth fails
    try:
        ssh.connect(
            hostname,
            port=port,
            username=username,
            look_for_keys=True,
            timeout=timeout,
            banner_timeout=banner_timeout,
            auth_timeout=auth_timeout
        )
        log.info("SSHClient connected via agent fallback.")
        return ssh

    except paramiko.AuthenticationException:
        log.warning("Agent authentication failed. No valid authorized keys found.")
    except Exception as e:
        log.warning("SSH connection failed during agent fallback: {}".format(str(e)))
    
    return None

def getSFTPClient(ssh, 
                 window_size=None,
                 max_packet_size=None, 
                 rekey_bytes=None):
    """
    Opens an SFTP session from an established SSH client with configured transfer parameters.
    
    Arguments:
        ssh: [paramiko.SSHClient] Established SSH client connection
        
    Keyword Arguments:
        window_size: [int] Window size for the SFTP connection. Default is None (use Paramiko default)
        max_packet_size: [int] Maximum packet size for SFTP. Default is None (use Paramiko default)
        rekey_bytes: [int] Number of bytes before rekeying. Default is None (use Paramiko default)
        
    Returns:
        [paramiko.SFTPClient or None] SFTP client if successful, None otherwise
    """
    if ssh is None:
        log.error("Cannot open SFTP session: SSH client is None.")
        return None

    log.debug("Attempting to open SFTP connection...")
    try:
        # Configure transport parameters if specified
        transport = ssh.get_transport()
        
        if transport:
            # Apply window size if specified (controls how much data can be in-flight)
            if window_size is not None:
                orig_window_size = transport.window_size
                transport.window_size = window_size
                log.info(f"SFTP window size set from {orig_window_size} to {window_size} bytes")
            
            # Configure packet size if specified
            if max_packet_size is not None:
                orig_max_packet_size = transport.packetizer.MAX_PACKET_SIZE
                transport.packetizer.MAX_PACKET_SIZE = max_packet_size
                log.info(f"SFTP max packet size set from {orig_max_packet_size} to {max_packet_size} bytes")
                
            # Configure rekey frequency if specified
            if rekey_bytes is not None:
                orig_rekey_bytes = transport.packetizer.REKEY_BYTES
                transport.packetizer.REKEY_BYTES = rekey_bytes
                log.info(f"SFTP rekey bytes set from {orig_rekey_bytes} to {rekey_bytes} bytes")

        # Open SFTP session
        sftp = ssh.open_sftp()
        log.info("SFTP connection established with configured parameters.")
        return sftp

    except Exception as e:
        log.error("Failed to open SFTP connection: {}".format(e))
        return None


def getSSHAndSFTP(hostname, **kwargs):
    """
    Wrapper function that returns both SSH and SFTP clients with configured transfer parameters.
    
    Arguments:
        hostname: [str] Hostname or IP address of the SSH server
        
    Keyword Arguments:
        port: [int] SSH port number
        username: [str] SSH username
        key_filename: [str] Path to private key file
        timeout: [int] Connection timeout in seconds
        banner_timeout: [int] SSH banner timeout in seconds
        auth_timeout: [int] Authentication timeout in seconds
        keepalive_interval: [int] Keepalive interval in seconds
        window_size: [int] SFTP window size in bytes
        max_packet_size: [int] SFTP maximum packet size in bytes
        rekey_bytes: [int] SFTP rekey frequency in bytes
        
    Returns:
        [tuple] (ssh_client, sftp_client) - Both can be None if connection failed
    """
    # Extract SFTP-specific parameters
    sftp_params = {}
    for param in ['window_size', 'max_packet_size', 'rekey_bytes']:
        if param in kwargs:
            sftp_params[param] = kwargs.pop(param)
    
    # Establish SSH connection
    ssh = getSSHClient(hostname, **kwargs)
    
    # Establish SFTP connection with the configured parameters
    sftp = getSFTPClient(ssh, **sftp_params) if ssh else None
    
    return ssh, sftp


def uploadSFTP(hostname, username, dir_local, dir_remote, file_list, port=22,
               rsa_private_key=os.path.expanduser('~/.ssh/id_rsa'),
               allow_dir_creation=False,
               connect_timeout=300,
               banner_timeout=300,
               auth_timeout=300,
               keepalive_interval=30,
               window_size=32768,         # Default reduced from 2**31 to 32KB for ADSL
               max_packet_size=32768,     # Default reduced from ~35000 to 32KB
               rekey_bytes=10*1024*1024,  # Default 10MB before rekeying
               block_size=8192):          # Default reduced from 32768 to 8KB
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
        window_size: [int] Window size for SFTP transport (bytes in flight before ACK).
        max_packet_size: [int] Maximum size of individual SFTP packets.
        rekey_bytes: [int] Number of bytes before rekeying the connection.
        block_size: [int] Block size for file transfers (smaller = slower but gentler).

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
        # Log bandwidth optimization parameters
        log.info(f"SFTP optimization parameters:")
        log.info(f"  - Window size: {format_size(window_size)}")
        log.info(f"  - Max packet size: {format_size(max_packet_size)}")
        log.info(f"  - Rekey bytes: {format_size(rekey_bytes)}")
        log.info(f"  - Block size: {format_size(block_size)}")
        
        # Connect with timeouts and SFTP optimization parameters
        ssh, sftp = getSSHAndSFTP(
            hostname,
            port=port,
            username=username,
            key_filename=rsa_private_key,
            timeout=connect_timeout,
            banner_timeout=banner_timeout,
            auth_timeout=auth_timeout,
            keepalive_interval=keepalive_interval,
            window_size=window_size,
            max_packet_size=max_packet_size,
            rekey_bytes=rekey_bytes
        )

        # Optionally ensure remote directory exists
        if allow_dir_creation:
            log.info("Checking/creating remote dir '{}'".format(dir_remote))
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
                    log.info("The file '{}' already exists on the server and is the same size. Skipping.".format(remote_file))
                    continue
            
            except IOError as e:
                # Means remote file doesn't exist yet, so proceed
                pass
            
            # Define a callback function for progress reporting
            total_bytes = local_file_size
            uploaded_bytes = 0
            start_time = time.time()
            last_percent = 0
            update_interval = 1  # Update progress every 1% for large files or 5% for small files
            if total_bytes > 100 * 1024 * 1024:  # For files over 100MB, update more frequently
                update_interval = 0.5

            def progress_callback(bytes_transferred, _):
                nonlocal uploaded_bytes, last_percent, start_time
                uploaded_bytes = bytes_transferred
                
                # Calculate percentage
                if total_bytes > 0:
                    percent_complete = round(100.0 * uploaded_bytes / total_bytes, 1)
                    
                    # Only update when the percentage changes by at least update_interval
                    # Also prevent duplicate 100% messages
                    if (percent_complete >= last_percent + update_interval and last_percent < 100.0) or (percent_complete == 100.0 and last_percent != 100.0):
                        elapsed_time = time.time() - start_time
                        
                        # Calculate transfer speed
                        if elapsed_time > 0:
                            transfer_rate = uploaded_bytes / elapsed_time / 1024  # KB/s
                            
                            # Format as MB/s if over 1024 KB/s
                            if transfer_rate > 1024:
                                transfer_rate_str = "{:.2f} MB/s".format(transfer_rate / 1024)
                            else:
                                transfer_rate_str = "{:.2f} KB/s".format(transfer_rate)
                            
                            # Estimate time remaining
                            if percent_complete > 0 and percent_complete < 100.0:
                                time_remaining = (elapsed_time / percent_complete) * (100 - percent_complete)
                                # Format time remaining
                                if time_remaining > 60:
                                    time_str = "{:.1f} min remaining".format(time_remaining / 60)
                                else:
                                    time_str = "{:.0f} sec remaining".format(time_remaining)
                                
                                print('[{:.1f}%] Uploading: {} ({}/{}) @ {} - {}'.format(
                                    percent_complete,
                                    os.path.basename(local_file),
                                    format_size(uploaded_bytes),
                                    format_size(total_bytes),
                                    transfer_rate_str,
                                    time_str
                                ))
                            else:
                                # At 100%, show "complete" instead of remaining time
                                if percent_complete == 100.0:
                                    print('[100.0%] Upload complete: {} ({}/{}) @ {}'.format(
                                        os.path.basename(local_file),
                                        format_size(uploaded_bytes),
                                        format_size(total_bytes),
                                        transfer_rate_str
                                    ))
                                else:
                                    print('[{:.1f}%] Uploading: {} ({}/{}) @ {}'.format(
                                        percent_complete,
                                        os.path.basename(local_file),
                                        format_size(uploaded_bytes),
                                        format_size(total_bytes),
                                        transfer_rate_str
                                    ))
                        
                        else:
                            print('[{:.1f}%] Uploading: {} ({}/{})'.format(
                                percent_complete,
                                os.path.basename(local_file),
                                format_size(uploaded_bytes),
                                format_size(total_bytes)
                            ))
                        
                        last_percent = percent_complete
            
            # Upload the file to the server if it isn't already there
            log.info('Starting upload of ' + local_file + ' ({}) to '.format(format_size(local_file_size)) + remote_file)
            
            # Use the optimized block size for the transfer
            sftp.put(local_file, remote_file, callback=progress_callback, block_size=block_size)
            
            log.info("Upload completed, verifying...")

            # Check that the size of the remote file is correct, indicating a successful upload
            remote_info = sftp.lstat(remote_file)
            
            # If the remote and the local file are of the same size, skip it
            if local_file_size != remote_info.st_size:
                log.error('File verification failed: local size {} != remote size {}'.format(
                    format_size(local_file_size), format_size(remote_info.st_size)))
                return False

            log.info("File upload verified: {:s}".format(remote_file))
            
        return True

    except Exception as e:
        log.error("Exception during SFTP upload: {}".format(e), exc_info=True)
        return False

    finally:
        # Close SFTP and SSH if open
        if sftp is not None:
            log.info("Closing SFTP channel")
            sftp.close()
        if ssh is not None:
            log.info("Closing SSH client connection")
            ssh.close()

# Helper function to format file sizes in human-readable format
def format_size(size_bytes):
    """Format a size in bytes into a human-readable string"""
    if size_bytes < 1024:
        return "{} B".format(size_bytes)
    elif size_bytes < 1024 * 1024:
        return "{:.2f} KB".format(size_bytes / 1024)
    elif size_bytes < 1024 * 1024 * 1024:
        return "{:.2f} MB".format(size_bytes / (1024 * 1024))
    else:
        return "{:.2f} GB".format(size_bytes / (1024 * 1024 * 1024))

class UploadManager(multiprocessing.Process):
    def __init__(self, config):
        """ Uploads all processed data which has not yet been uploaded to the server. The files will be tried 
            to be uploaded every 15 minutes, until successful. 
        
        """


        super(UploadManager, self).__init__()

        self.config = config

        self.file_queue = Queue.Queue()
        self.exit = multiprocessing.Event()
        self.upload_in_progress = multiprocessing.Value(ctypes.c_bool, False)

        # Time when the upload was run last
        self.last_runtime = None
        self.last_runtime_lock = multiprocessing.Lock()

        # Time when the next upload should be run (used for delaying the upload)
        self.next_runtime = None
        self.next_runtime_lock = multiprocessing.Lock() 

        # Construct the path to the queue backup file
        self.upload_queue_file_path = os.path.join(self.config.data_dir, self.config.upload_queue_file)

        # Load the list of files to upload, and have not yet been uploaded
        self.loadQueue()
        
        # Set default SFTP throttling parameters
        self.sftp_window_size = self.config.window_size
        self.sftp_max_packet_size = self.config.max_packet_size
        self.sftp_rekey_bytes = self.config.rekey_bytes
        self.sftp_block_size = self.config.block_size



    def start(self):
        """ Starts the upload manager. """

        super(UploadManager, self).start()



    def stop(self):
        """ Stops the upload manager. """

        self.exit.set()
        self.join()



    def addFiles(self, file_list):
        """ Adds a list of files to be uploaded to the queue. """

        # Add the files to the queue
        for file_name in file_list:
            self.file_queue.put(file_name)

        time.sleep(0.1)

        # Write the queue to disk
        self.saveQueue()

        # Make sure the data gets uploaded right away
        with self.last_runtime_lock:
            self.last_runtime = None



    def loadQueue(self):
        """ Load a list of files to be uploaded from a file. """

        # Check if the queue file exists, if not, create it
        if not os.path.exists(self.upload_queue_file_path):
            self.saveQueue(overwrite=True)
            return None


        # Read the queue file
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


                # Add the file if it was not already in the queue
                if not file_name in self.file_queue.queue:
                    self.file_queue.put(file_name)



    def saveQueue(self, overwrite=False):
        """ Save the list of file to upload to disk, for bookkeeping in case of a power failure. 
    
        Keyword arguments:
            overwrite: [bool] If True, the holding file will be overwritten. Otherwise (default), the entries
                that are not in the file will be added at the end of the file.
        """

        # Convert the queue to a list
        file_list = [file_name for file_name in self.file_queue.queue]

        # If overwrite is true, save the queue to the holding file completely
        if overwrite:

            # Make the data directory if it doesn't exist
            mkdirP(self.config.data_dir)

            # Create the queue file
            with open(self.upload_queue_file_path, 'w') as f:
                for file_name in file_list:
                    f.write(file_name + '\n')

        else:

            # Load the list from the file and make sure to write only the entries not already in the file

            # Get a list of entries in the holding file
            existing_list = []
            with open(self.upload_queue_file_path) as f:
                for file_name in f:
                    file_name = file_name.replace('\n', '').replace('\r', '')
                    existing_list.append(file_name)

            # Save to disk only those entires which are not already there
            with open(self.upload_queue_file_path, 'a') as f:
                for file_name in file_list:
                    if file_name not in existing_list:
                        f.write(file_name + '\n')




    def uploadData(self, retries=5):
        """ Pulls the upload list from a file, tries to upload the file, and if it fails it saves the list of 
            failed files to disk. 

        Keyword arguments:
            retries: [int] Number of tried to upload a file before giving up.
        """

        # Skip uploading if the upload is already in progress
        if self.upload_in_progress.value:
            return


        # Set flag that the upload as in progress
        self.upload_in_progress.value = True

        # Read the file list from disk
        self.loadQueue()

        tries = 0

        # Go through every file and upload it to server
        while self.file_queue.qsize() > 0:

            # Get a file from the queue
            file_name = self.file_queue.get()
            if not os.path.isfile(file_name):
                log.warning("Local file not found: {:s}".format(file_name))
                log.warning("Skipping it...")
                continue

            # Separate the path to the file and the file name
            data_path, f_name = os.path.split(file_name)

            # Upload the file via SFTP (use the lowercase version of the station ID as the username)
            upload_status = uploadSFTP(
                self.config.hostname, 
                self.config.stationID.lower(), 
                data_path, 
                self.config.remote_dir, 
                [f_name], 
                rsa_private_key=self.config.rsa_private_key, 
                port=self.config.host_port,
                window_size=self.sftp_window_size,
                max_packet_size=self.sftp_max_packet_size,
                rekey_bytes=self.sftp_rekey_bytes,
                block_size=self.sftp_block_size
            )

            # If the upload was successful, rewrite the holding file, which will remove the uploaded file
            if upload_status:
                log.info('Upload successful!')
                self.saveQueue(overwrite=True)
                tries = 0

            # If the upload failed, put the file back on the list and wait a bit
            else:

                log.warning('Uploading failed! Retry {:d} of {:d}'.format(tries + 1, retries))

                tries += 1 
                self.file_queue.put(file_name)

                # Given the network a moment to recover between attempts
                time.sleep(10)

            # Check if the upload was tried too many times
            if tries >= retries:
                break

        # Set the flag that the upload is done
        self.upload_in_progress.value = False


    def delayNextUpload(self, delay=0):
        """ Delay the upload by the given number of seconds from now. Zero by default. """

        # Set the next run time using a delay
        with self.next_runtime_lock:
            self.next_runtime = RmsDateTime.utcnow() + datetime.timedelta(seconds=delay)

            log.info("Upload delayed for {:.1f} min until {:s}".format(delay/60, str(self.next_runtime)))



    def run(self):
        """ Try uploading the files every 15 minutes. """

        with self.last_runtime_lock:
            self.last_runtime = None

        while not self.exit.is_set():

            with self.last_runtime_lock:

                # Check if the upload should be run (if 15 minutes are up)
                if self.last_runtime is not None:
                    if (RmsDateTime.utcnow() - self.last_runtime).total_seconds() < 15*60:
                        time.sleep(1)
                        continue

            with self.next_runtime_lock:

                # Check if the upload delay is up
                if self.next_runtime is not None:
                    if (RmsDateTime.utcnow() - self.next_runtime).total_seconds() < 0:
                        time.sleep(1)
                        continue

            with self.last_runtime_lock:
                self.last_runtime = RmsDateTime.utcnow()

            # Run the upload procedure
            self.uploadData()

            time.sleep(0.1)




if __name__ == "__main__":

    from RMS.Logger import initLogging

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

            # SFTP optimization parameters for ADSL connections
            self.sftp_window_size = 32768        # Default 32KB (reduced from 2**31)
            self.sftp_max_packet_size = 32768    # Default 32KB (reduced from ~35000)
            self.sftp_rekey_bytes = 10*1024*1024 # Default 10MB before rekeying
            self.sftp_block_size = 8192          # Default 8KB (reduced from default)

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
        allow_dir_creation=True,
        # Using the ADSL-optimized parameters 
        window_size=config.sftp_window_size,
        max_packet_size=config.sftp_max_packet_size,
        rekey_bytes=config.sftp_rekey_bytes,
        block_size=config.sftp_block_size
        )
    
    sys.exit()


    ### Test the upload manager ###

    # Init the logger
    initLogging(config)

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