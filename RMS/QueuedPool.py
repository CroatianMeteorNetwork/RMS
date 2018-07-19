from __future__ import print_function

import logging
import traceback
import multiprocessing
import time


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



class QueuedPool(object):
    """ Provides capability of creating a pool of workers which will process jobs in a given queue, and the 
        input queue can be updated in another thread. 

        The workers will process the queue until the pool is deliberately closed. All results are stored in an 
        output queue. It is also possible to change the number of workers in a pool during runtime.

    Arguments:
        func: [function] Worker function to which the arguments from the queue will be passed

    Keyword arguments:
        cores: [int] Number of CPU cores to use. None by default. If negative, then the number of cores to be 
            used will be the total number available, minus the given number.
        log: [logging handle] A logger object which will be used for logging.
        delay_start: [float] Number of seconds to wait after init before the workers start workings.

    """

    def __init__(self, func, cores=None, log=None, delay_start=0):


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

        # Initialize queues (for some reason queues from Manager need to be created, otherwise they are 
        # blocking when using get_nowait)
        manager = multiprocessing.Manager()
        self.input_queue = manager.Queue()
        self.output_queue = manager.Queue()

        self.func = func
        self.pool = None

        self.total_jobs = SafeValue()
        self.active_workers = SafeValue()
        self.kill_workers = multiprocessing.Event()



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


            # Catch errors in workers and handle them softly

            try:
                
                # Call the original worker function and collect results
                result = func(*args)

            except:
                tb = traceback.format_exc()

                print(tb)
                
                if self.log is not None:
                    self.log.error(tb)

                result = None

            # Save the results to an output queue
            self.output_queue.put(result)

            # Exit if exit is requested
            if self.kill_workers.is_set():
                break

        self.active_workers.decrement()



    def startPool(self, cores=None):
        """ Start the pool with the given worker function and number of cores. """

        if cores is not None:
            self.cores.set(cores)


        if self.log is not None:
            self.log.info('Using {:d} cores'.format(self.cores.value()))

        # Initialize the pool of workers with the given number of worker cores
        # Comma in the argument list is a must!
        self.pool = multiprocessing.Pool(self.cores.value(), self._workerFunc, (self.func, ))



    def closePool(self):
        """ Wait until all jobs are done and close the pool. """

        if self.pool is not None:

            c = 0

            # Wait until the input queue is empty, then close the pool
            while True:

                c += 1

                if c%1000 == 0:
                    print('-----')
                    print('Queue size:', self.output_queue.qsize())
                    print('Total jobs:', self.total_jobs.value())
                
                # If all jobs are done, close the pool
                if self.output_queue.qsize() >= self.total_jobs.value():

                    print('Inserting poison pills...')

                    # Insert the 'poison pill' to the queue, to kill all workers
                    for i in range(self.cores.value()):
                        print('Inserting pill', i + 1)
                        self.input_queue.put(None)


                    # Wait until the pills are 'swallowed'
                    while self.input_queue.qsize():
                        print('Swallowing pills...', self.input_queue.qsize())
                        time.sleep(0.1)


                    # Close the pool and wait for all threads to terminate
                    print('Closing pool...')
                    self.pool.close()
                    print('Terminating pool...')
                    self.pool.terminate()
                    print('Joining pool...')
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
        while self.active_workers.value() > 0:
            time.sleep(0.1)

        # Join the previous pool
        self.pool.close()
        self.pool.terminate()
        self.pool.join()

        self.kill_workers.clear()

        # If cores were not given, use all available cores
        if cores is None:
            cores = multiprocessing.cpu_count()

        self.cores.set(cores)

        # Init a new pool
        self.startPool()



    def addJob(self, job):
        """ Add a job to the input queue. Job can be a list of arguments for the worker function. If a list is
            not given, the arguments will be wrapped in the list.

        """

        # Track the total number of jobs received
        self.total_jobs.increment()

        if not isinstance(job, list):
            job = [job]

        self.input_queue.put(job)


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

    t1 = time.time()
    while True:
        if time.time() - t1 > 2:
            break


    # Cause an error in the worker
    5/in_num


    return in_str + " " + str(in_num) + " X"




if __name__ == "__main__":

    # Init logging
    logging.basicConfig()
    log = logging.getLogger('logger')
    log.setLevel(logging.INFO)
    log.setLevel(logging.DEBUG)

    # Initialize the pool with only one core
    workpool = QueuedPool(exampleWorker, cores=1, log=log)

    # Give the pool something to do
    for i in range(1, 3):

        workpool.addJob(["hello", i])

        time.sleep(0.1)

        workpool.addJob(["world", i])


    # Start the pool
    workpool.startPool()

    time.sleep(2)

    print('Updating cores...')

    # Use all available cores
    workpool.updateCoreNumber(3)

    print('Adding more jobs...')

    # Give the pool some more work to do
    for i in range(0, 4):
        workpool.addJob(["test1", i])
        time.sleep(0.05)
        workpool.addJob(["test2", i])


    print('Closing the pool...')

    # Wait for everything to finish and close the pool
    workpool.closePool()


    # Print out the results
    results = workpool.getResults()

    print(results)