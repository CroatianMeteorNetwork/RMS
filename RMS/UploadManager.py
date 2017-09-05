""" Functions for uploading files to server via SFTP. """

from __future__ import print_function, division, absolute_import

import os
import ctypes
import paramiko
import multiprocessing
import time
import datetime

try:
    # Python 2
    import Queue

except:
    # Python 3
    import queue as Queue



def _agentAuth(transport, username, rsa_private_key):
    """ Attempt to authenticate to the given transport using any of the private keys available from an SSH 
        agent or from a local private RSA key file (assumes no pass phrase).

    Arguments:
        transport: [paramiko.Transport object] Connection handle.
        username: [str] Username which will be used to connect to the host.
        rsa_private_key: [str] Path to the RSA private key on the system.

    Return:
        [bool] True if successfull, False otherwise.
    """

    # Try loading the private key
    try:
        ki = paramiko.RSAKey.from_private_key_file(rsa_private_key)

    except Exception, e:
        print('Failed loading', rsa_private_key, e)

    # Find all available keys
    agent = paramiko.Agent()
    agent_keys = agent.get_keys() + (ki,)

    if len(agent_keys) == 0:
        return False

    # Try a key until finding the one which works
    for key in agent_keys:
        print('Trying ssh-agent key', key.get_fingerprint().encode('hex'))

        # Try the key to authenticate
        try:
            transport.auth_publickey(username, key)
            print('... success!')
            return True

        except paramiko.SSHException, e:
            print('... failed!', e)

    return False



def uploadSFTP(hostname, username, dir_local, dir_remote, file_list, port=22, 
        rsa_private_key=os.path.expanduser('~/.ssh/id_rsa')):
    """ Upload the given list of files using SFTP. 

    Arguments:
        hostname: [str] Server name or IP address.
        username: [str] Username used for connecting to the server.
        dir_local: [str] Path to the local directory where the local files are located.
        dir_remove: [str] Path on the server where the files will be stored.
        file_list: [list or strings] A list of files to the uploaded to the server.

    Ketword arguments:
        port: [int] SSH port. 22 by default.
        rsa_private_key: [str] Path to the SSH private key. ~/.ssh/id_rsa by defualt.

    Return:
        [bool] True if upload successful, false otherwise.
    """

    # Connect and use paramiko Transport to negotiate SSH2 across the connection
    # The whole thing is in a try block because if an error occurs, the connection will be closed at the end
    try:

        print('Establishing SSH connection to:', hostname, port, '...')

        # Connect to host
        t = paramiko.Transport((hostname, port))
        t.start_client()

        # Authenticate the connection
        auth_status = _agentAuth(t, username, rsa_private_key)
        if not auth_status:
            return False

        # Open new SFTP connection
        sftp = paramiko.SFTPClient.from_transport(t)

        # Go through all files
        for fname in file_list:

            # Path to the local file
            local_file = os.path.join(dir_local, fname)

            # Get the size of the local file
            local_file_size = os.lstat(local_file).st_size

            # Path to the remove file
            remote_file = dir_remote + '/' + os.path.basename(fname)

            # Check if the remote file already exists and skip it if it has the same size as the local file
            try:
                remote_info = sftp.lstat(remote_file)
                
                # If the remote and the local file are of the same size, skip it
                if local_file_size == remote_info.st_size:
                    continue
            
            except IOError, e:
                pass
            
            # Upload the file to the server if it isn't already there
            print('Copying', local_file, 'to ', remote_file)
            sftp.put(local_file, remote_file)

        t.close()

        return True

    except Exception, e:
        print('*** Caught exception: %s: %s' % (e.__class__, e))
        try:
            t.close()
        except:
            pass

        return False




class UploadManager(multiprocessing.Process):
    def __init__(self, data_path, config):
        """ Uploads all processed data which has not yet been uploaded to the server. The files will be tied 
            to uploaded every 15 minutes, until successfull. """


        super(UploadManager, self).__init__()

        self.data_path = data_path
        self.config = config

        self.file_queue = Queue.Queue()
        self.exit = multiprocessing.Event()
        self.upload_in_progress = multiprocessing.Value(ctypes.c_bool, False)

        # Load the list of files to upload, and have not yet been uploaded
        self.loadQueue()



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

        # Write the queue to disk
        self.saveQueue()

        # Upload the data
        self.uploadData()



    def loadQueue(self):
        """ Load a list of files to be uploaded from a file. """

        # Check if the queue file exists, if not, create it
        if not os.path.exists(self.config.upload_queue_file):
            self.saveQueue(overwrite=True)
            return None


        # Read the queue file
        with open(self.config.upload_queue_file) as f:
            
            for file_name in f:

                file_name = file_name.replace('\n', '').replace('\r', '')

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

            with open(self.config.upload_queue_file, 'w') as f:
                for file_name in file_list:
                    f.write(file_name + '\n')

        else:

            # Load the list from the file and make sure to write only the entries not already in the file

            # Get a list of entries in the holding file
            existing_list = []
            with open(self.config.upload_queue_file) as f:
                for file_name in f:
                    file_name = file_name.replace('\n', '').replace('\r', '')
                    existing_list.append(file_name)

            # Save to disk only those entires which are not already there
            with open(self.config.upload_queue_file, 'a') as f:
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

            # Upload the file via SFTP
            upload_status = uploadSFTP(self.config.hostname, self.config.stationID, self.data_path, '.', \
                [file_name], rsa_private_key=self.config.rsa_private_key)

            # If the upload was successful, rewrite the holding file, which will remove the uploaded file
            if upload_status:
                self.saveQueue(overwrite=True)
                tries = 0

            # If the upload failed, put the file back on the list and wait a bit
            else:

                print('Uploading failed! Retry {:d} of {:d}'.format(tries + 1, retries))

                tries += 1 
                self.file_queue.put(file_name)

                time.sleep(2)

            # Check if the upload was tried too many times
            if tries >= retries:
                break

        # Set the flag that the upload is done
        self.upload_in_progress.value = False



    def run(self):
        """ Try uploading the files every 15 minutes. """

        last_runtime = None
        while not self.exit.is_set():

            # Check if the upload should be run
            if last_runtime is not None:
                if (datetime.datetime.now() - last_runtime).total_seconds() < 15*60:
                    continue

            last_runtime = datetime.datetime.now()

            # Run the upload procedure
            self.uploadData()

            time.sleep(0.1)


            

            




if __name__ == "__main__":

    # Set up a fake config file
    class FakeConf(object):
        def __init__(self):

            self.username = 'pi'

            # remote hostname where SSH server is running
            self.hostname = 'minorid.localnet' 
            self.host_port = 22
            self.stationID = 'pi'
            self.rsa_private_key = r"/home/dvida/.ssh/id_rsa"

            self.upload_queue_file = 'FILES_TO_UPLOAD.inf'


    config = FakeConf()

    dir_local='/home/dvida/Desktop'
    dir_remote = "."


    #uploadSFTP(config.hostname, config.stationID, dir_local, dir_remote, file_list, rsa_private_key=config.rsa_private_key)

    up = UploadManager(dir_local, config)
    up.start()

    time.sleep(2)

    up.addFiles(['test.txt'])

    time.sleep(1)

    up.addFiles(['test2.txt'])    
    up.addFiles(['test3.txt'])    


    up.stop()


