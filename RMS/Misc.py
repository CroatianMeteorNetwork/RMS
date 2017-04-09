import os
import errno


def mkdirP(path):
    """ Makes a directory and handles all errors.
    """

    # Try to make a directory
    try:
        os.makedirs(path)

    # If it already exist, do nothing
    except OSError, exc:
        if exc.errno == errno.EEXIST:
            pass

    # Raise all other errors
	else: 
		raise