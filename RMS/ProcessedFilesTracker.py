
"""
    A class to track processed files by maintaining a record of the last processed file
    itentifies by some soerted key.  The default behavious is a string comparison on
    the kwy using the simplistic scenario of the key being the filename.
"""

import os

class ProcessedFilesTracker():

 
    def __init__(self, key_func=lambda a,b: os.path.basename(a) < os.path.basename(b), directory=".", tracker_file_name="last_processed_file.txt", log_func=None):
        """ 
        Arguments:
            key_func: callable[[str, str], bool] A function that takes two file keys and returns true if the first key is "less than" the second key.  
                                            This is used to determine if a file has been processed by comparing its key to the last processed file key.
                                            By default, it is a simple string comparison of the base file name of the provided file paths.
            directory [string] : The directory in which the tracker file is written.  This can be but is not required to be the source directory of the files being processed.
            tracker_file_name: [str] The path name of the file in which to store the key of the last processed file.  By default, it is "last_processed_file.txt" in the directory of the files being processed.
                                 If provided and the pathname is relative, it will be placed in the spefied directory.  An absolute pathname can be provided.

        """                                    
        self.key_func = key_func
        self.directory = directory
        self.tracker_file_name = tracker_file_name
        self.log_func = log_func
        self.tracker_file_path = os.path.join(directory, tracker_file_name) if not os.path.isabs(tracker_file_name) else tracker_file_name

        # Initialize the last processed file key from the tracker file
        if os.path.exists(self.tracker_file_path):
            with open(self.tracker_file_path, 'r') as tf:
                self.last_file_path = tf.read().strip()
                if self.log_func is not None:
                    self.log_func(f"Initialized ProcessedFilesTracker with last processed file: {self.last_file_path}")
        else:   
            self.last_file_path = None
            if self.log_func is not None:
                self.log_func("Initialized ProcessedFilesTracker with no last processed file.")

    def isProcessed(self, file_path):
        if self.last_file_path is None:
            if self.log_func is not None:
                self.log_func(f"File {file_path} is not processed because there is no last processed file.")
            return False
        processed = self.key_func(file_path, self.last_file_path)
        if self.log_func is not None:
            self.log_func(f"File {file_path} is {'processed' if processed else 'not processed'}")
        return processed

    def markProcessed(self, file_path):
        if self.last_file_path is None or not self.key_func(file_path, self.last_file_path):
            self.last_file_path = file_path
            with open(self.tracker_file_path, 'w') as tf:
                tf.write(file_path + '\n')
            if self.log_func is not None:
                self.log_func(f"Marked file {file_path} as processed.")

    def clear(self):
        self.last_file_path = None      
        if os.path.exists(self.tracker_file_path):
            os.remove(self.tracker_file_path)
        if self.log_func is not None:
            self.log_func("Cleared processed files tracker.")
    