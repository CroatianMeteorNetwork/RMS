
import os
import shutil
import errno
import logging

# Get the logger from the main module
log = logging.getLogger("logger")


def mkdirP(path):
    """ Makes a directory and handles all errors.
    """

    # Try to make a directory
    try:
        os.makedirs(path)

    # If it already exist, do nothing
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            pass

    # Raise all other errors
    else: 
        raise



def archiveDir(source_dir, file_list, dest_dir, compress_file, delete_dest_dir=False, extra_files=None):
    """ Move the given file list from the source directory to the destination directory, compress the 
        destination directory and save it as a .bz2 file. BZ2 compression is used as ZIP files have a limit
        of 2GB in size.

    Arguments:
        source_dir: [str] Path to the directory from which the files will be taken and archived.
        file_list: [list] A list of files from the source_dir which will be archived.
        dest_dir: [str] Path to the archive directory which will be compressed.
        compress_file: [str] Name of the compressed file which will be created.

    Keyword arguments:
        delete_dest_dir: [bool] Delete the destination directory after compression. False by default.
        extra_files: [list] A list of extra files (with fill paths) which will be be saved to the night 
            archive.

    Return:
        archive_name: [str] Full name of the archive.

    """

    # Make the archive directory
    mkdirP(dest_dir)

    # Copy the files from the source to the archive directory
    for file_name in file_list:
        shutil.copy2(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))

    # Copy the additional files to the archive directory
    if extra_files is not None:
        for file_name in extra_files:
            shutil.copy2(file_name, os.path.join(dest_dir, os.path.basename(file_name)))


    # Compress the archive directory
    archive_name = shutil.make_archive(os.path.join(dest_dir, compress_file), 'bztar', dest_dir, logger=log)

    # Delete the archive directory after compression
    if delete_dest_dir:
        shutil.rmtree(dest_dir)


    return archive_name