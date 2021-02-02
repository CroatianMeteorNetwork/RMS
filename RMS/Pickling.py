""" Functions for pickling and unpickling Python objects. """

from __future__ import print_function, absolute_import

import os
import sys
import pickle


from RMS.Misc import mkdirP




def savePickle(obj, dir_path, file_name):
    """ Dump the given object into a file using Python 'pickling'. The file can be loaded into Python
        ('unpickled') afterwards for further use.

    Arguments:
    	obj: [object] Object which will be pickled.
        dir_path: [str] Path of the directory where the pickle file will be stored.
        file_name: [str] Name of the file where the object will be stored.

    """

    mkdirP(dir_path)

    with open(os.path.join(dir_path, file_name), 'wb') as f:
        pickle.dump(obj, f, protocol=2)



def loadPickle(dir_path, file_name):
    """ Loads pickle file.
	
	Arguments:
		dir_path: [str] Path of the directory where the pickle file will be stored.
        file_name: [str] Name of the file where the object will be stored.

    """

    with open(os.path.join(dir_path, file_name), 'rb') as f:

        # Try loading the Pickle file

        try:
            # Python 2
            if sys.version_info[0] < 3:
                return pickle.load(f)

            # Python 3
            else:
                return pickle.load(f, encoding='latin1')

        except (IOError, EOFError, TypeError, KeyError):
            
            print('The pickle file was corruped and could not be loaded:', os.path.join(dir_path, file_name))
            return None