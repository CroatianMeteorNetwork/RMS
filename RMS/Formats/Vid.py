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




# Layout of the header of a frame: (field, dtype, byte offset), as read by readFrame
VID_HEADER_FIELDS = [('magic', np.uint32, 0), ('seqlen', np.uint32, 4), ('headlen', np.uint32, 8),
                     ('flags', np.uint32, 12), ('seq', np.uint32, 16), ('ts', np.int32, 20), ('tu', np.int32, 24),
                     ('station_id', np.int16, 28), ('wid', np.int16, 30), ('ht', np.int16, 32),
                     ('depth', np.int16, 34), ('hx', np.uint16, 36), ('hy', np.uint16, 38),
                     ('str_num', np.uint16, 40), ('reserved0', np.uint16, 42), ('exposure', np.uint32, 44),
                     ('reserved2', np.uint32, 48)]
VID_TEXT_OFFSET = 52
VID_TEXT_LENGTH = 64


def readFrameFromBuffer(st, buf, offset, metadata_only=False):
    """ Read the frame which begins at the given byte offset of a vid file loaded into memory, the same way as
        readFrame reads it from the file: save its header to the given structure and return the image data.

    Arguments:
        st: [Vid structure]
        buf: [ndarray] uint8 array with the contents of the vid file.
        offset: [int] Byte offset of the beginning of the frame.

    Keyword arguments:
        metadata_only: [bool] Only read the metadata, but not the whole frame. False by default

    Return:
        [ndarray] The image (a copy), None if the frame is beyond the end of the data or incomplete (or only
            the metadata was read).
    """

    # The end of the data, or a header which is not complete
    if offset + VID_TEXT_OFFSET + VID_TEXT_LENGTH > len(buf):
        return None

    # The header fields, at their fixed byte offsets (in the byte order of the machine, as np.fromfile reads)
    for name, dtype, pos in VID_HEADER_FIELDS:
        setattr(st, name, int(np.frombuffer(buf, dtype=dtype, count=1, offset=offset + pos)[0]))

    text = buf[offset + VID_TEXT_OFFSET:offset + VID_TEXT_OFFSET + VID_TEXT_LENGTH]
    st.text = text.tobytes().decode("ascii").replace('\0', '')

    if metadata_only:
        return None

    # The whole frame (an incomplete last frame is not read)
    if offset + st.seqlen > len(buf):
        return None

    fr = np.frombuffer(buf, dtype=np.uint16, count=st.seqlen//2, offset=offset).copy()

    # Set the values of the first row to 0 (the header is stored there), as readFrame does
    fr[:st.ht] = 0

    return fr.reshape(st.ht, st.wid)


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


