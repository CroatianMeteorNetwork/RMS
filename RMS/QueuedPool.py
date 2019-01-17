from __future__ import print_function

import os
import logging
import traceback
import time
import functools
import multiprocessing
import multiprocessing.dummy

from RMS.Pickling import savePickle, loadPickle
from RMS.Misc import randomCharacters, isListKeyInDict, listToTupleRecursive


class SafeValue(object):
    """ Thread safe value. Uses locks. 
    
    Source: http://eli.thegreenplace.net/2012/01/04/shared-counter-with-pythons-multiprocessing
    """

    def __init__(self, initval=0):
        self.val = multiprocessing.Value('i', initval)
        self.lock = multiprocessing.Lock()


    def increment(self):
        with self.lock:
            self.val.value += 1


    def decrement(self):
        with self.lock:
            self.val.value -= 1


    def set(self, n):
        with self.lock:
            self.val.value = n


    def value(self):
        with self.lock:
            return self.val.value



class BackupContainer(object):
    def __init__(self, inputs, outputs):
        """ Container for storing the inputs and outputs of a certain worker function. This container is 
            saved to disk when the worker function finishes, and can be later restored. 
        """

        # Take the input list a tuple, as it has to be imutable to be a dictionary key
        self.inputs = listToTupleRecursive(inputs)

        self.outputs = outputs



class QueuedPool(object):
    def __init__(self, func, cores=None, log=None, delay_start=0, worker_timeout=2000, backup_dir='.'):
        """ Provides capability of creating a pool of workers which will process jobs in a given queue, and 
        the input queue can be updated in another thread. 

        The workers will process the queue until the pool is deliberately closed. All results are stored in an 
        output queue. It is also possible to change the number of workers in a pool during runtime.

        The default worker timeout time is 1000 seconds.

        Arguments:
            func: [function] Worker function to which the arguments from the queue will be passed

        Keyword arguments:
            cores: [int] Number of CPU cores to use. None by default. If negative, then the number of cores 
                to be used will be the total number available, minus the given number.
            log: [logging handle] A logger object which will be used for logging.
            delay_start: [float] Number of seconds to wait after init before the workers start workings.
            worker_timeout: [int] Number of seconds to wait before the queue is killed due to a worker getting 
                stuck.
            backup_dir: [str] Path to the directory where result backups will be held.

        """


        # If the cores are not given, use all available cores
        if cores is None:
            cores = multiprocessing.cpu_count()


        # If cores are negative, use the total available cores minus the given number
        if cores < 0:

            cores = multiprocessing.cpu_count() + cores

            if cores < 1:
                cores = 1

            if cores > multiprocessing.cpu_count():
                cores = multiprocessing.cpu_count()


        self.cores = SafeValue(cores)
        self.log = log

        self.start_time = time.time()
        self.delay_start = delay_start
        self.worker_timeout = worker_timeout

        # Initialize queues (for some reason queues from Manager need to be created, otherwise they are 
        # blocking when using get_nowait)
        manager = multiprocessing.Manager()
        self.input_queue = manager.Queue()
        self.output_queue = manager.Queue()

        self.func = func
        self.pool = None

        self.total_jobs = SafeValue()
        self.results_counter = SafeValue()
        self.active_workers = SafeValue()
        self.kill_workers = multiprocessing.Event()


        ### Backing up results

        self.bkup_dir = backup_dir
        self.bkup_file_prefix = 'rms_queue_bkup_'
        self.bkup_file_extension = '.pickle'
        self.bkup_dict = {}

        # Load all previous backup files in the given directory, if any
        self.loadBackupFiles()

        ### ###


    def printAndLog(self, *args):
        """ Print and log the given message. """

        message = " ".join(list(map(str, args)))
        print(message)

        if self.log is not None:
            self.log.info(message)


    def saveBackupFile(self, inputs, outputs):
        """ Save the pair of inputs and outputs to disk, so if the script breaks, previous results can be
            loaded in and processing can continue from that point.
        """

        # Create a backup object
        bkup_obj = BackupContainer(inputs, outputs)

        # Create a name for the backup file
        bkup_file_name = self.bkup_file_prefix + str(self.results_counter.value()) + "_" + randomCharacters(9) \
            + self.bkup_file_extension

        # Save the backup to disk
        savePickle(bkup_obj, self.bkup_dir, bkup_file_name)



    def _listBackupFiles(self):
        """ Returns a list of all backup files in the backup folder. """

        bkup_file_list = []

        for file_name in os.listdir(self.bkup_dir):

            # Check if this is the backup file
            if file_name.startswith(self.bkup_file_prefix) and file_name.endswith(self.bkup_file_extension):

                bkup_file_list.append(file_name)

        return bkup_file_list



    def loadBackupFiles(self):
        """ Load previous backup files. """

        # Load all backup files in a dictionary
        for file_name in self._listBackupFiles():

            # Load the backup file
            bkup_obj = loadPickle(self.bkup_dir, file_name)

            if bkup_obj is None:
                continue

            # Get the inputs
            bkup_inputs = listToTupleRecursive(bkup_obj.inputs)

            # Make sure the value-key pair does not exist
            key_status, _ = isListKeyInDict(bkup_inputs, self.bkup_dict)
            if not key_status:

                # Add the pair of inputs vs. outputs to the lookup dictionary
                self.bkup_dict[bkup_inputs] = bkup_obj.outputs


        # Print and log how many previous files have been loaded
        print_str = "Loaded {:d} backed up results...".format(len(self.bkup_dict))
        self.printAndLog(print_str)



    def deleteBackupFiles(self):
        """ Delete all backup files in the backup folder. """

        # Go though all backup files
        for file_name in self._listBackupFiles():

            # Remove the backup file
            os.remove(os.path.join(self.bkup_dir, file_name))



    def _workerFunc(self, func):
        """ A wrapper function for the given worker function. Handles the queue operations. """
        

        # Wait until delay has passed
        while (self.start_time + self.delay_start) > time.time():
            time.sleep(0.1)

        self.active_workers.increment()

        while True:

            # Get the function arguments (block until available)
            args = self.input_queue.get(True)

            # The 'poison pill' for killing the worker when closing is requested
            if args is None:
                break

            # First do a lookup in the dictionary if this set of inputs have already been processed
            read_from_backup = False
            args_tpl = listToTupleRecursive(args)

            # Check if the key exists. Return the found key as object instances might differ
            key_status, key = isListKeyInDict(args_tpl, self.bkup_dict)

            if key_status:

                # Load the results from backup
                result = self.bkup_dict[key]

                read_from_backup = True

                self.printAndLog('Result loaded from backup for input: {:s}'.format(str(args)))


            # Process the inputs if they haven't been processed already
            else:

                # Catch errors in workers and handle them softly
                try:

                    # Call the original worker function and collect results
                    result = func(*args)

                except:
                    tb = traceback.format_exc()

                    self.printAndLog(tb)

                    result = None


            # Save the results to an output queue
            self.output_queue.put(result)
            self.results_counter.increment()

            time.sleep(0.1)

            # Back up the result to disk, if it was not already in the backup
            if not read_from_backup:
                self.saveBackupFile(args, result)

            # Exit if exit is requested
            if self.kill_workers.is_set():
                self.printAndLog('Worker killed!')
                break

        self.active_workers.decrement()



    def startPool(self, cores=None):
        """ Start the pool with the given worker function and number of cores. """

        if cores is not None:
            self.cores.set(cores)


        self.printAndLog('Using {:d} cores'.format(self.cores.value()))

        # Initialize the pool of workers with the given number of worker cores
        # Comma in the argument list is a must!
        self.pool = multiprocessing.Pool(self.cores.value(), self._workerFunc, (self.func, ))



    def closePool(self):
        """ Wait until all jobs are done and close the pool. """

        if self.pool is not None:

            c = 0

            prev_output_qsize = 0
            output_qsize_last_change = time.time()

            # Wait until the input queue is empty, then close the pool
            while True:

                c += 1

                if c%1000 == 0:
                    self.printAndLog('-----')
                    self.printAndLog('Queue size:', self.output_queue.qsize())
                    self.printAndLog('Total jobs:', self.total_jobs.value())
                    self.printAndLog('Active workers:', self.active_workers.value())


                # Keep track of the changes of the output queue size
                if self.output_queue.qsize() != prev_output_qsize:
                    prev_output_qsize = self.output_queue.qsize()
                    output_qsize_last_change = time.time()


                # If the queue has been idle for too long, kill it
                if (time.time() - output_qsize_last_change) > self.worker_timeout:
                    self.printAndLog('One of the workers got stuck longer then {:d} seconds, killing multiprocessing...'.format(self.worker_timeout))

                    self.printAndLog('Terminating pool...')
                    self.pool.terminate()

                    self.printAndLog('Joining pool...')
                    self.pool.join()

                    self.active_workers.set(0)

                    break

                
                # If all jobs are done, close the pool
                if self.output_queue.qsize() >= self.total_jobs.value():

                    self.printAndLog('Inserting poison pills...')

                    # Insert the 'poison pill' to the queue, to kill all workers
                    for i in range(self.active_workers.value()):
                        self.printAndLog('Inserting pill', i + 1)
                        self.input_queue.put(None)


                    time.sleep(0.1)


                    # Wait until the pills are 'swallowed'
                    while self.input_queue.qsize():
                        self.printAndLog('Swallowing pills...', self.input_queue.qsize())
                        time.sleep(0.1)


                    # Close the pool and wait for all threads to terminate
                    self.printAndLog('Closing pool...')
                    self.pool.close()

                    self.printAndLog('Terminating pool...')
                    self.pool.terminate()

                    self.printAndLog('Joining pool...')
                    self.pool.join()

                    break

                else:
                    time.sleep(0.1)



    def updateCoreNumber(self, cores=None):
        """ Update the number of cores/workers used by the pool.

        Arguments:
            cores: [int] Number of CPU cores to use. None by default.

        """

        # Kill the workers
        self.kill_workers.set()

        # Wait until all workers have exited
        loop_start = time.time()
        while self.active_workers.value() > 0:
            self.printAndLog('Active workers:', self.active_workers.value())
            time.sleep(0.1)

            # Break the loop if waiting for more than 100 s
            if abs(time.time() - loop_start) > 100:
                break

        # Join the previous pool
        self.printAndLog('Closing pool...')
        self.pool.close()
        self.printAndLog('Terminating pool...')
        self.pool.terminate()
        self.printAndLog('Joining pool...')
        self.pool.join()

        self.kill_workers.clear()

        # If cores were not given, use all available cores
        if cores is None:
            cores = multiprocessing.cpu_count()

        self.printAndLog('Setting new number of cores to:', cores)
        self.cores.set(cores)

        # Init a new pool
        self.printAndLog('Starting new pool...')
        self.startPool()



    def addJob(self, job, wait_time=0.05):
        """ Add a job to the input queue. Job can be a list of arguments for the worker function. If a list is
            not given, the arguments will be wrapped in the list.

        """

        # Track the total number of jobs received
        self.total_jobs.increment()

        if not isinstance(job, list):
            job = [job]

        self.input_queue.put(job)

        time.sleep(wait_time)


    def allDone(self):
        """ If all jobs are done, return True.
        """

        if self.output_queue.qsize() == self.total_jobs.value():
            return True

        else:
            return False



    def getResults(self):
        """ Get the results from the output queue and store them in a list. The output list will be returned. 
        """

        results = []

        # Get all elements in the output queue
        if not self.output_queue.empty():
            while True:
                try:
                    results.append(self.output_queue.get_nowait())
                except:
                    break
            
        return results



def exampleWorker(in_str, in_num):
    """ An example worker function. """

    print('Got:', in_str, in_num)


    # Wait for in_num seconds
    t1 = time.time()
    while True:
        if time.time() - t1 > in_num:
            break


    # Cause an error in the worker when input is 0
    5/in_num


    return in_str + " " + str(in_num) + " X"




if __name__ == "__main__":

    # # Init logging
    # logging.basicConfig()
    # log = logging.getLogger('logger')
    # log.setLevel(logging.INFO)
    # log.setLevel(logging.DEBUG)

    log = None

    # Initialize the pool with only one core, timeout of only 10 seconds
    workpool = QueuedPool(exampleWorker, cores=1, log=log, worker_timeout=10)

    # Give the pool something to do
    for i in range(1, 3):

        workpool.addJob(["hello", i])

        time.sleep(0.1)


    # Start the pool
    workpool.startPool()

    time.sleep(2)

    print('Updating cores...')

    # Use all available cores
    #workpool.updateCoreNumber(3)

    print('Adding more jobs...')

    # Give the pool some more work to do
    for i in range(1, 4):
        workpool.addJob(["test1", i])
        time.sleep(0.05)


    workpool.addJob(["long time", 99])

    print('Closing the pool...')

    # Wait for everything to finish and close the pool
    workpool.closePool()


    # Print out the results
    results = workpool.getResults()

    print(results)

    # Delete all backed up files
    workpool.deleteBackupFiles()