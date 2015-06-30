import cv2
import numpy as np
import sys

# FFbin handling stolen from FF_bin_suite.py in CMN_binViewer

class ff_struct:
    """ Default structure for a FF*.bin file.
    """
    def __init__(self):
        self.nrows = 0
        self.ncols = 0
        self.nbits = 0
        self.first = 0
        self.camno = 0
        self.maxpixel = 0
        self.maxframe = 0
        self.avepixel = 0
        self.stdpixel = 0
        
def readFF(filename):
    """Function for reading FF bin files.
    Returns a structure that allows access to individual parameters of the image
    e.g. print readFF("FF300_20140802_205545_600_0090624.bin").nrows to print out the number of rows
    e.g. print readFF("FF300_20140802_205545_600_0090624.bin").maxpixel to print out the array of nrows*ncols numbers which represent the image
    INPUTS:
        filename: file name from the file to be read
    """

    fid = open(filename, 'rb')
    ff = ff_struct()
    ff.nrows = np.fromfile(fid, dtype=np.uint32, count = 1)
    ff.ncols = np.fromfile(fid, dtype=np.uint32, count = 1)
    ff.nbits = np.fromfile(fid, dtype=np.uint32, count = 1)
    ff.first = np.fromfile(fid, dtype=np.uint32, count = 1)
    ff.camno = np.fromfile(fid, dtype=np.uint32, count = 1)

    N = ff.nrows * ff.ncols

    ff.maxpixel = np.reshape (np.fromfile(fid, dtype=np.uint8, count = N), (ff.nrows, ff.ncols))
    ff.maxframe = np.reshape (np.fromfile(fid, dtype=np.uint8, count = N), (ff.nrows, ff.ncols))
    ff.avepixel = np.reshape (np.fromfile(fid, dtype=np.uint8, count = N), (ff.nrows, ff.ncols))
    ff.stdpixel = np.reshape (np.fromfile(fid, dtype=np.uint8, count = N), (ff.nrows, ff.ncols))

    return ff

class fr_struct:
    """ Default structure for a FF*.bin file.
    """
    def __init__(self):
        self.lines = 0
        self.frameNum = []
        self.yc = []
        self.xc = []
        self.t = []
        self.size = []
        self.frames = []
        
def readFR(filename):
    fid = open(filename, 'rb')
    fr = fr_struct()
    fr.lines = np.fromfile(fid, dtype=np.uint32, count = 1)
    
    for i in range(fr.lines):
        frameNum = np.fromfile(fid, dtype=np.uint32, count = 1)
        yc = []
        xc = []
        t = []
        size = []
        frames = []
        
        for z in range(frameNum):
            yc.append(np.fromfile(fid, dtype=np.uint32, count = 1))
            xc.append(np.fromfile(fid, dtype=np.uint32, count = 1))
            t.append(np.fromfile(fid, dtype=np.uint32, count = 1))
            size.append(np.fromfile(fid, dtype=np.uint32, count = 1))
            frames.append(np.reshape(np.fromfile(fid, dtype=np.uint8, count = size[-1]**2), (size[-1], size[-1])))
        
        fr.frameNum.append(frameNum)
        fr.yc.append(yc)
        fr.xc.append(xc)
        fr.t.append(t)
        fr.size.append(size)
        fr.frames.append(frames)

    return fr

def view(ff, fr):
    background = ff.maxpixel
    
    print "Number of lines:", fr.lines
    
    for i in range(fr.lines):
        for z in range(fr.frameNum[i]):
            yc = fr.yc[i][z]
            xc = fr.xc[i][z]
            t = fr.t[i][z]
            size = fr.size[i][z]
            
            print "Center coords:", yc, xc, t, "size:", size
            
            y2 = 0
            for y in range(yc - size/2, yc + size/2):
                x2 = 0
                for x in range(xc - size/2,  xc + size/2):
                    background[y, x] = fr.frames[i][z][y2, x2]
                    x2 += 1
                y2 += 1
            
            cv2.imshow("view", background)
            cv2.waitKey(200)
            

if __name__ == "__main__":
    if len(sys.argv) == 2:
        view(readFF("FF"+sys.argv[1]), readFR("FR"+sys.argv[1]))