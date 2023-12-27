from __future__ import print_function

import os
import sys
import logging
import traceback
import time
import functools
import multiprocessing
import multiprocessing.dummy

from RMS.Pickling import savePickle, loadPickle
from RMS.Misc import randomCharacters, isListKeyInDict, listToTupleRecursive


from errno import EPIPE

# Python 3
try:
    broken_pipe_exception = BrokenPipeError

# Python 2
except NameError:
    broken_pipe_exception = IOError



class SafeValue(object):
    """ Thread safe value. Uses locks. 
    
    Source: http://eli.thegreenplace.net/2012/01/04/shared-counter-with-pythons-multiprocessing
    """

    def __init__(self, initval=0, minval=None, maxval=None):
        
        self.val = multiprocessing.Value('i', initval)
        self.lock = multiprocessing.Lock()

        self.minval = minval
        self.maxval = maxval


    def increment(self):
        with self.lock:
            self.val.value += 1

            if self.maxval is not None:
                if self.val.value > self.maxval:
                    self.val.value = self.maxval


    def decrement(self):
        with self.lock:
            self.val.value -= 1

            if self.minval is not None:
                if self.val.value < self.minval:
                    self.val.value = self.minval


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
    def __init__(self, func, cores=None, log=None, delay_start=0, worker_timeout=2000, backup_dir='.', \
        input_queue_maxsize=None, low_priority=False, func_extra_args=None, func_kwargs=None, 
        worker_wait_inbetween_jobs=0.1, print_state=True):
        """ Provides capability of creating a pool of workers which will process jobs in a given queue, and 
        the input queue can be updated in another thread. 

        The workers will process the queue until the pool is deliberately closed. All results are stored in an 
        output queue. It is also possible to change the number of workers in a pool during runtime.

        The default worker timeout time is 2000 seconds.

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
            input_queue_maxsize: [int] Maximum size of the input queue. Used to conserve memory. Can be set
                to the number of cores, optimally. None by default, meaning there is no size limit.
            low_priority: [bool] If True, the child processes will run with a lower priority, i.e. larger
                'niceness' (available only on Unix).
            func_extra_args: [tuple] Extra arguments for the worker function. Can be used when there
                arguments are the same for all function calls to conserve memory if they are large. None by
                default.
            func_kwargs: [dict] Extra keyword arguments for the worker function. Can be used when there
                arguments are the same for all function calls to conserve memory if they are large. None by
                default.
            worker_wait_inbetween_jobs: [float] Wait this number of seconds after finished a job and putting
                the result in the output queue. 0.1 s by default.
            print_state: [bool] Print state of workers during execution, True by default.
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


        if func_extra_args is None:
            func_extra_args = ()

        if func_kwargs is None:
            func_kwargs = {}


        self.cores = SafeValue(cores, minval=1, maxval=multiprocessing.cpu_count())
        self.log = log

        self.start_time = time.time()
        self.delay_start = delay_start
        self.worker_timeout = worker_timeout
        self.low_priority = low_priority
        self.func_extra_args = func_extra_args
        self.func_kwargs = func_kwargs
        self.worker_wait_inbetween_jobs = worker_wait_inbetween_jobs

        # Initialize queues (for some reason queues from Manager need to be created, otherwise they are 
        # blocking when using get_nowait)
        manager = multiprocessing.Manager()

        # Only init with maxsize if given, otherwise it return a TypeErorr when fed data from Compressor
        if input_queue_maxsize is None:
            self.input_queue = manager.Queue()
        else:
            self.input_queue = manager.Queue(maxsize=input_queue_maxsize)

        self.output_queue = manager.Queue()

        self.func = func
        self.pool = None

        self.total_jobs = SafeValue(minval=0)
        self.results_counter = SafeValue(minval=0)
        self.active_workers = SafeValue(minval=0, maxval=multiprocessing.cpu_count())
        self.available_workers = SafeValue(self.cores.value(), minval=0, maxval=multiprocessing.cpu_count())
        self.kill_workers = multiprocessing.Event()


        ### Backing up results

        self.bkup_dir = backup_dir
        self.bkup_file_prefix = 'rms_queue_bkup_'
        self.bkup_file_extension = '.pickle'
        self.bkup_dict = {}

        # Load all previous backup files in the given directory, if any
        if self.bkup_dir is not None:
            self.loadBackupFiles()

        ### ###

        self.print_state = print_state


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

        if self.bkup_dir is not None:

            # Go though all backup files
            for file_name in self._listBackupFiles():

                # Remove the backup file
                os.remove(os.path.join(self.bkup_dir, file_name))



    def _workerFunc(self, func):
        """ A wrapper function for the given worker function. Handles the queue operations. """
        
        # Set lower priority, if given
        if self.low_priority:

            # Try setting the process niceness (available only on Unix systems)
            try:
                os.nice(20)
                self.printAndLog('Set low priority for processing thread!')
            except Exception as e:
                self.printAndLog('Setting niceness failed with message:\n' + repr(e))


        # Wait until delay has passed
        while (self.start_time + self.delay_start) > time.time():
            time.sleep(0.1)

        self.active_workers.increment()

        
        input_ret_failures = 0

        while True:

            # Get the function arguments (block until available, handle possible errors)
            try:
                args = self.input_queue.get(True)
            
            except:
                tb = traceback.format_exc()
                self.printAndLog('Failed retrieving inputs...')
                self.printAndLog(tb)

                if input_ret_failures > 5:
                    self.printAndLog("Too many failures to get inputs for QueuedPool, assuming all inputs were processed...")
                    break

                input_ret_failures += 1
                time.sleep(1.0)
                continue


            # The 'poison pill' for killing the worker when closing is requested
            if args is None:
                break

            self.available_workers.decrement()

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
                    all_args = tuple(args) + tuple(self.func_extra_args)
                    result = func(*all_args, **self.func_kwargs)

                except:
                    tb = traceback.format_exc()

                    self.printAndLog(tb)

                    result = None


            # Save the results to an output queue
            self.output_queue.put(result)
            self.results_counter.increment()
            self.available_workers.increment()
            time.sleep(self.worker_wait_inbetween_jobs)

            # Back up the result to disk, if it was not already in the backup
            if (not read_from_backup) and (self.bkup_dir is not None):
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
            all_workers_idle_time = None

            # Wait until the input queue is empty, then close the pool
            while True:

                c += 1

                if c%500 == 0 and self.print_state:
                    self.printAndLog('-----')
                    self.printAndLog('Cores in use:', self.cores.value())
                    self.printAndLog('Active worker threads:', self.active_workers.value())
                    self.printAndLog('Idle worker threads:', self.available_workers.value())
                    self.printAndLog('Total jobs:', self.total_jobs.value())
                    self.printAndLog('Finished jobs:', self.output_queue.qsize())


                # Keep track of the changes of the output queue size
                if self.output_queue.qsize() != prev_output_qsize:
                    prev_output_qsize = self.output_queue.qsize()
                    output_qsize_last_change = time.time()

                # If the output queue size is zero, meaning no jobs are done, increase the time by the delay
                #   time
                if prev_output_qsize == 0:
                    worker_timeout = self.worker_timeout + self.delay_start

                else:
                    worker_timeout = self.worker_timeout

                # If the queue has been idle for too long, kill it
                if (time.time() - output_qsize_last_change) > worker_timeout:
                    self.printAndLog('One of the workers got stuck longer than {:.1f} seconds, killing multiprocessing...'.format(float(worker_timeout)))

                    self.printAndLog('Terminating pool...')
                    self.pool.terminate()

                    self.printAndLog('Joining pool...')
                    self.pool.join()

                    self.active_workers.set(0)

                    break


                

                # If all workers are idle, set the last idle time
                if all_workers_idle_time is None:
                    if self.available_workers.value() >= self.cores.value():
                        all_workers_idle_time = time.time()
                        self.printAndLog('All workers are idle!')
                else:
                    
                    # Check if the workers are still idle
                    if self.available_workers.value() < self.cores.value():
                        self.printAndLog('All workers are NOT idle anymore!')
                        all_workers_idle_time = None


                # If all workers have been idle for more than 100 seconds, terminate the queue
                if all_workers_idle_time is not None:
                    if (time.time() - all_workers_idle_time) > 100:

                        self.printAndLog('All workers were idle for more than 100 seconds, killing multiprocessing...')

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


                    time.sleep(0.5)


                    # Wait until the poison pills are 'swallowed' (do this until timeout)
                    timeout = 60 # seconds
                    timeout_count = 0
                    while (self.input_queue.qsize() > 0) and (self.active_workers.value() > 0):
                        self.printAndLog('Swallowing pills...', self.input_queue.qsize())
                        time.sleep(1)

                        timeout_count += 1

                        # If all workers are idle after timeout, break the swallowing loop
                        if timeout_count > timeout:
                            if self.available_workers.value() >= self.cores.value():
                                break


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



    def addJob(self, job, wait_time=0.1, repeated=False):
        """ Add a job to the input queue. Job can be a list of arguments for the worker function. If a list is
            not given, the arguments will be wrapped in the list.

        """

        if not isinstance(job, list):
            job = [job]

        # Add a job to the queue
        try:
            
            self.input_queue.put(job)
            time.sleep(wait_time/2.0)

            # Track the total number of jobs received
            self.total_jobs.increment()

        # Sometimes the pipe gets broken, so try handling it gracefully
        except broken_pipe_exception as exc:

            self.printAndLog("Pipe IOError caught, trying to handle it gracefully...")

            if broken_pipe_exception == IOError:
                if exc.errno != EPIPE:
                    raise


            # Try adding the job to processing queue again
            if not repeated:
                time.sleep(0.1)
                self.addJob(job, wait_time=wait_time, repeated=True)
                time.sleep(0.1)
                return None

            else:
                self.printAndLog("Error! Failed adding the job to processing list the second time...")
                raise broken_pipe_exception("Input queue pipe broke!")


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