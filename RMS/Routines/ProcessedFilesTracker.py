""" Track processed files by maintaining a record of the last processed file, identified by a sorted key. """

import os


class ProcessedFilesTracker(object):
    """ Track which files have been processed by persisting the key of the last processed file to disk.

        Files are considered "processed" if their key compares less than the last recorded key, using a
        caller-supplied comparison function. By default, keys are the base filenames compared
        lexicographically.
    """

    def __init__(self, key_func=None, directory=".", tracker_file_name="last_processed_file.txt",
                 log_func=None):
        """ Initialize the tracker, loading any previously recorded state from disk.

        Arguments:
            key_func: [callable] A function (a, b) -> bool that returns True if key a is "less than" key b.
                Used to determine whether a file has already been processed. Default is a lexicographic
                comparison of os.path.basename.
            directory: [str] Directory in which the tracker state file is stored. Does not have to be the
                same directory as the files being tracked.
            tracker_file_name: [str] Name of the state file. If relative, it is placed inside *directory*;
                an absolute path may also be given. Default is "last_processed_file.txt".
            log_func: [callable or None] Optional logging function, e.g. log.info.
        """

        # Default comparison: lexicographic on base filename
        if key_func is None:
            key_func = lambda a, b: os.path.basename(a) < os.path.basename(b)

        self.key_func = key_func
        self.directory = directory
        self.tracker_file_name = tracker_file_name
        self.log_func = log_func

        # Resolve the full path to the tracker file
        if os.path.isabs(tracker_file_name):
            self.tracker_file_path = tracker_file_name
        else:
            self.tracker_file_path = os.path.join(directory, tracker_file_name)

        # Load the last processed key from disk (if available)
        if os.path.exists(self.tracker_file_path):
            with open(self.tracker_file_path, 'r') as tf:
                self.last_file_path = tf.read().strip()

            if self.log_func is not None:
                self.log_func("Initialized ProcessedFilesTracker with last processed file: "
                              "{}".format(self.last_file_path))
        else:
            self.last_file_path = None

            if self.log_func is not None:
                self.log_func("Initialized ProcessedFilesTracker with no last processed file.")


    def isProcessed(self, file_path):
        """ Check whether a file has already been processed.

        Arguments:
            file_path: [str] Key (typically a filename) to check.

        Return:
            [bool] True if the file compares less than the last processed key.
        """

        if self.last_file_path is None:
            if self.log_func is not None:
                self.log_func("File {} is not processed (no prior record).".format(file_path))
            return False

        processed = self.key_func(file_path, self.last_file_path)

        if self.log_func is not None:
            self.log_func("File {} is {}".format(file_path, "processed" if processed else "not processed"))

        return processed


    def markProcessed(self, file_path):
        """ Mark a file as processed only if it advances past the current high-water mark.

        Arguments:
            file_path: [str] Key of the file to mark.
        """

        # Only update if this key is at or beyond the current high-water mark
        if self.last_file_path is None or not self.key_func(file_path, self.last_file_path):
            self.setProcessed(file_path)


    def setProcessed(self, file_path):
        """ Unconditionally set the last processed key and persist it to disk.

        Arguments:
            file_path: [str] Key to record.
        """

        self.last_file_path = file_path

        with open(self.tracker_file_path, 'w') as tf:
            tf.write(file_path + '\n')

        if self.log_func is not None:
            self.log_func("Marked file {} as processed.".format(file_path))


    def clear(self):
        """ Reset the tracker, removing the persisted state file. """

        self.last_file_path = None

        if os.path.exists(self.tracker_file_path):
            os.remove(self.tracker_file_path)

        if self.log_func is not None:
            self.log_func("Cleared processed files tracker.")