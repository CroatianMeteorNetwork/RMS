""" Selecting and zipping files with detections. """


import os

from RMS.Misc import archiveDir





def selectFiles(dir_path, ff_detected):
    """ Make a list of all files which should be zipped in the given night directory. 
    
        In the list are included:
            - all TXT files
            - all FR bin files and their parent FF bin files
            - all FF bin files with detections

    Arguments:
        dir_path: [str] Path to the night directory.
        ff_detected: [list] A list of FF bin file with detections on them.

    Return:
        selected_files: [list] A list of files selected for compression.

    """


    selected_list = []

    # Go through all files in the night directory
    for file_name in os.listdir(dir_path):

        # Take all .txt files
        if '.txt' in file_name:
            selected_list.append(file_name)


        # Take all field sum files
        if ('FF' in file_name) and ('fieldsum' in file_name):
            selected_list.append(file_name)


        # Take all FR bin files, and their parent FF bin files
        if ('FR' in file_name) and ('.bin' in file_name):

            fr_split = file_name.split('_')

            # FR file identifier which it shares with the FF bin file
            fr_id = '_'.join(fr_split[1:3])

            ff_match = None

            # Locate the parent FF bin file
            for ff_file_name in os.listdir(dir_path):

                if ('FF' in ff_file_name) and ('.bin' in ff_file_name) and (fr_id in ff_file_name):
                    
                    ff_match = ff_file_name
                    break


            # Add the FR bin file and it's parent FF file to the list
            selected_list.append(file_name)

            if ff_match is not None:
                selected_list.append(ff_match)


        # Add FF file which contain detections to the list
        if file_name in ff_detected:
            selected_list.append(file_name)


    # Take only the unique elements in the list, sorted by name
    selected_list = sorted(list(set(selected_list)))


    return selected_list



def archiveDetections(captured_path, archived_path, ff_detected):


    # Get the list of files to archive
    file_list = selectFiles(captured_path, ff_detected)


    if file_list:

        # Create the archive ZIP in the parent directory of the archive directory
        archive_name = os.path.join(os.path.abspath(os.path.join(archived_path, os.pardir)), 
            os.path.basename(captured_path) + '_detected')

        # Archive the files
        archiveDir(captured_path, file_list, archived_path, archive_name)



if __name__ == "__main__":

    ### Test the archive function

    captured_path = "/home/dvida/RMS_data/20170410_002933_521633"

    archived_path = "/home/dvida/RMS_data/ArchivedFiles/20170410_002933_521633"

    ff_detected = ['FF499_20170410_003723_921_0014080.bin', 'FF499_20170410_003732_463_0014336.bin']


    archiveDetections(captured_path, archived_path, ff_detected)