
import os
import shutil
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



def archiveDir(source_dir, file_list, dest_dir, zipname, delete_dest_dir=False):
    """ Move the given file list from the source directory to the destination directory, ZIP the destination 
        directory and save it as a .zip file.

    Arguments:
        source_dir: [str] Path to the directory from which the files will be taken and archived.
        file_list: [list] A list of files from the source_dir which will be archived.
        dest_dir: [str] Path to the archive directory which will be zipped.
        zipname: [str] Name of the zip file which will be created.

    Keyword arguments:
        delete_dest_dir: [bool] Delete the destination directory after zipping. False by defualt.

    """

    # Make the archive directory
    mkdirP(dest_dir)

    # Copy the files from the source to the archive directory
    for file_name in file_list:
        shutil.copy2(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))

    # Zip the archive directory
    shutil.make_archive(os.path.join(dest_dir, zipname), 'zip', dest_dir)

    # Delete the archive directory after zipping
    if delete_dest_dir:
        shutil.rmtree(dest_dir)