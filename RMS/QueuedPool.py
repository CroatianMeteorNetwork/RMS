from __future__ import print_function

import multiprocessing
import time


class QueuedPool(object):
    """ Provides capability of creating a pool of workers which will process jobs in a given queue, and the 
        input queue can be updated in another thread. 

        The workers will process the queue until the pool is deliberately closed. All results are stored in an 
        output queue. It is also possible to change the number of workers in a pool during runtime.

    Arguments:
        func: [function] Worker function to which the arguments from the queue will be passed

    Keyword arguments:
        cores: [int] Number of CPU cores to use. None by default.

    """
    def __init__(self, func, cores=None):

        # If the cores are not given, use all available cores
        if cores is None:
            cores = multiprocessing.cpu_count()

        self.cores = cores

        # Initialize queues
        self.input_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()

        self.func = func
        self.pool = None

        # Start the pool with the given parameters - this will wait until the input queue is given jobs
        self.startPool()



    def _workerFunc(self, func, input_queue, output_queue):
        """ A wrapper function for the given worker function. Handles the queue operations. """
        
        while True:

            # Get the function arguments
            args = input_queue.get(True)

            # The 'poison pill' for killing the worker when closing is requested
            if args is None:
                break

            # Call the original worker function and collect results
            result = func(*args)

            # Save the results to an output queue
            output_queue.put(result)



    def startPool(self):
        """ Start the pool with the given worker function and number of cores. """

        # Initialize the pool of workers with the given number of worker cores
        # Comma in the argument list is a must!
        self.pool = multiprocessing.Pool(self.cores, self._workerFunc, (self.func, self.input_queue, 
            self.output_queue, ))



    def closePool(self):
        """ Wait until all jobs are done and close the pool. """

        if self.pool is not None:

            # Wait until the input queue is empty, then close the pool
            while True:
                
                # If all jobs are done, close the pool
                if self.input_queue.qsize() == 0:

                    # Insert the 'poison pill' to the queue, to kill all workers
                    for i in range(self.cores + 1):
                        self.input_queue.put(None)

                    # Close the pool and wait for all threads to terminate
                    self.pool.close()
                    self.pool.join()

                    break

                else:
                    time.sleep(0.01)



    def updateCoreNumber(self, cores=None):
        """ Update the number of cores/workers used by the pool.

        Arguments:
            cores: [int] Number of CPU cores to use. None by default.

        """

        if cores is None:
            cores = multiprocessing.cpu_count()

        self.cores = cores

        self.startPool()



    def addJob(self, job):
        """ Add a job to the input queue. Job can be a list of arguments for the worker function. If a list is
            not given, the arguments will be wrapped in the list.

        """

        if not isinstance(job, list):
            job = [job]

        self.input_queue.put(job)



    def getResults(self):
        """ Get the results from the output queue and store them in a list. The output list will be returned. 
        """

        results = []

        # Get all elements in the output queue
        while workpool.output_queue.qsize():
            results.append(workpool.output_queue.get())

        return results



def exampleWorker(in_str, in_num):
    """ An example worker function. """

    print('Got:', in_str, in_num)

    # Simulate processing
    time.sleep(0.5)

    return in_str + " " + str(in_num) + " X"



if __name__ == "__main__":

    # Initialize the pool with only one core
    workpool = QueuedPool(exampleWorker, cores=1)

    # Give the pool something to do
    for i in range(2):

        workpool.addJob(["hello", i])

        time.sleep(0.1)

        workpool.addJob(["world", i])


    time.sleep(2)


    # Use all available cores
    workpool.updateCoreNumber()


    # Give the pool some more work to do
    for i in range(4):
        workpool.addJob(["test1", i])
        workpool.addJob(["test2", i])


    # Wait for everything to finish and close the pool
    workpool.closePool()


    # Print out the results
    results = workpool.getResults()

    print(results)