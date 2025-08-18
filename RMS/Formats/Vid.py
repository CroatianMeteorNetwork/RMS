""" Loading and handling *.vid files. """

from __future__ import print_function, division, absolute_import

import os
import numpy as np


class VidStruct(object):
    """ Structure for storing vid file info. """

    def __init__(self):

        # List of video frames - this is used only in the parent Vid structure, while the child structures
        # have this empty, but they contain image data in the img_data variable
        self.frames = None

        self.magic = 0

        # Bytes for a single image
        self.seqlen = 0

        # Header length in bytes
        self.headlen = 0

        self.flags = 0
        self.seq = 0

        # UNIX time
        self.ts = 0
        self.tu = 0

        # Station number
        self.station_id = 0

        # Image dimensions in pixels
        self.wid = 0
        self.ht = 0

        # Image depth in bits
        self.depth = 0

        # Mirror pointing for centre of frame
        self.hx = 0
        self.hy = 0

        # Stream number
        self.str_num = 0
        self.reserved0 = 0

        # Exposure time in milliseconds
        self.exposure = 0

        self.reserved2 = 0

        self.text = 0

        # Image data
        self.img_data = None




def readFrame(st, fid, metadata_only=False):
    """ Read in the information from the next frame, save them to the given structure and return the image 
        data.

    Arguments:
        st: [Vid structure]
        fid: [file handle] File handle to the vid file. Make sure it was open in the 'rb' mode.

    Keyword arguments:
        metadata_only: [bool] Only read the metadata, but not the whole frame. False by default
    """

    # Get the current position in the file
    file_pos = fid.tell()

    # Check if the end of file (EOF) is reached
    if not fid.read(1):
        return None

    fid.seek(file_pos)


    #### Read the header ###
    ##########################################################################################################

    st.magic = int(np.fromfile(fid, dtype=np.uint32, count=1))

    # Size of one frame in bytes
    st.seqlen = int(np.fromfile(fid, dtype=np.uint32, count=1))

    # Header length in bytes
    st.headlen = int(np.fromfile(fid, dtype=np.uint32, count=1))

    st.flags = int(np.fromfile(fid, dtype=np.uint32, count=1))
    st.seq = int(np.fromfile(fid, dtype=np.uint32, count=1))

    # Beginning UNIX time
    st.ts = int(np.fromfile(fid, dtype=np.int32, count=1))
    st.tu = int(np.fromfile(fid, dtype=np.int32, count=1))

    # Station number
    st.station_id = int(np.fromfile(fid, dtype=np.int16, count=1))

    # Image dimensions
    st.wid = int(np.fromfile(fid, dtype=np.int16, count=1))
    st.ht = int(np.fromfile(fid, dtype=np.int16, count=1))

    # Image depth
    st.depth = int(np.fromfile(fid, dtype=np.int16, count=1))

    st.hx = int(np.fromfile(fid, dtype=np.uint16, count=1))
    st.hy = int(np.fromfile(fid, dtype=np.uint16, count=1))

    # Camera stream identifier, where 0 = 'A', 1 = 'B', etc
    st.str_num = int(np.fromfile(fid, dtype=np.uint16, count=1))

    st.reserved0 = int(np.fromfile(fid, dtype=np.uint16, count=1))
    st.exposure = int(np.fromfile(fid, dtype=np.uint32, count=1))
    st.reserved2 = int(np.fromfile(fid, dtype=np.uint32, count=1))
    
    st.text = np.fromfile(fid, dtype=np.uint8, count=64).tostring().decode("ascii").replace('\0', '')

    ##########################################################################################################

    if not metadata_only:

        # Rewind the file to the beginning of the frame
        fid.seek(file_pos)

        # Read one whole frame
        fr = np.fromfile(fid, dtype=np.uint16, count=st.seqlen//2)

        # Set the values of the first row to 0
        fr[:st.ht] = 0

        # Reshape the frame to the proper image size
        img = fr.reshape(st.ht, st.wid)

    else:

        # If only the metadata was read, set the image to None
        img = None


    return img




def readVid(dir_path, file_name):
    """ Read in a *.vid file. 
    
    Arguments:
        dir_path: [str] path to the directory where the *.vid file is located
        file_name: [str] name of the *.vid file

    Return:
        [VidStruct object]
    """

    # Open the file for binary reading
    fid = open(os.path.join(dir_path, file_name), 'rb')

    # Init the vid struct
    vid = VidStruct()

    # Read the info from the first frame
    readFrame(vid, fid, metadata_only=True)

    # Reset the file pointer to the beginning
    fid.seek(0)
    
    vid.frames = []

    # Read in the frames
    while True:

        # Init a new frame structure
        frame = VidStruct()

        # Read one frame
        #fr = np.fromfile(fid, dtype=np.uint16, count=vid.seqlen//2)
        frame.img_data = readFrame(frame, fid)

        # Check if we have reached the end of file
        if frame.img_data is None:
            break

        # Reshape the frame and add it to the frame list
        vid.frames.append(frame)

    fid.close()


    return vid



if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # Vid file path
    dir_path = "../../MirfitPrepare/20160929_050928_mir"
    file_name = "ev_20160929_050928A_01T.vid"

    # Read in the *.vid file
    vid = readVid(dir_path, file_name)

    frame_num = 125

    # Show one frame of the vid file
    plt.imshow(vid.frames[frame_num].img_data, cmap='gray', vmin=0, vmax=255)
    plt.show()


