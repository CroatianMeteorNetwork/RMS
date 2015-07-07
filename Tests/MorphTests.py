import sys
import numpy as np
from RMS.VideoExtractor import Extractor
from scipy import weave
import cv2
import time
from os import uname

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

def convertToBinaryImage(points, height, width):
    image = np.zeros((height, width), np.bool_)
    
    for point in points:
        image[point[0], point[1]] = True
    
    return image

def clean(image):
    code = """
    bool p2, p3, p4, p5, p6, p7, p8, p9;
    unsigned int y, x;
    
    for(y = 1; y < Nimage[0]-1; y++) {
        for(x = 1; x < Nimage[1]-1; x++) {
            if(!IMAGE2(y, x)){
                continue;
            }
            
            p2 = IMAGE2(y-1, x);
            p3 = IMAGE2(y-1, x+1);
            p4 = IMAGE2(y, x+1);
            p5 = IMAGE2(y+1, x+1);
            p6 = IMAGE2(y+1, x);
            p7 = IMAGE2(y+1, x-1);
            p8 = IMAGE2(y, x-1);
            p9 = IMAGE2(y-1, x-1);

            if(!p2 && !p3 && !p4 && !p5 && !p6 && !p7 && !p8 && !p9) {
                IMAGE2(y, x) = 0;
            }
        }
    }
    """
    
    args = []
    if uname()[4] == "armv7l":
       args = ["-O3", "-mfpu=neon", "-mfloat-abi=hard", "-fdump-tree-vect-details", "-funsafe-loop-optimizations", "-ftree-loop-if-convert-stores"]
    
    
    weave.inline(code, ['image'], extra_compile_args=args, extra_link_args=args)
    
    return image

def close(image):
    kernel = np.ones((3, 3), np.uint8)
    
    image = cv2.morphologyEx(image.astype(np.uint8)*255, cv2.MORPH_CLOSE, kernel)
    
    return image.astype(np.bool_)

def thinning(image):
    image = thin(image)
    previous = None
    
    while not np.array_equal(image, previous):
        previous = np.copy(image)
        image = thin(image)
    
    return image

def thin(image):
    #Zhang-Suen algorithm    
    code = """
    bool p2, p3, p4, p5, p6, p7, p8, p9;
    unsigned int y, x, A, B, m1_1, m2_1, m1_2, m2_2;
    
    for(y = 1; y < Nimage[0]-1; y++) {
        for(x = 1; x < Nimage[1]-1; x++) {
            if(!MASK2(y, x)){
                continue;
            }
            
            p2 = IMAGE2(y-1, x  );
            p3 = IMAGE2(y-1, x+1);
            p4 = IMAGE2(y  , x+1);
            p5 = IMAGE2(y+1, x+1);
            p6 = IMAGE2(y+1, x  );
            p7 = IMAGE2(y+1, x-1);
            p8 = IMAGE2(y  , x-1);
            p9 = IMAGE2(y-1, x-1);

            A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) + 
                (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) + 
                (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
            B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
            m1_1 = (p2 * p4 * p6);
            m2_1 = (p4 * p6 * p8);
            
            m1_2 = (p2 * p4 * p8);
            m2_2 = (p2 * p6 * p8);

            if(A == 1 && (B >= 2 && B <= 6) && ((m1_1 == 0 && m2_1 == 0) || (m1_2 == 0 && m2_2 == 0))) {
                MASK2(y, x) = 0;
            }
        }
    }
    """
    
    args = []
    if uname()[4] == "armv7l":
       args = ["-O3", "-mfpu=neon", "-mfloat-abi=hard", "-fdump-tree-vect-details", "-funsafe-loop-optimizations", "-ftree-loop-if-convert-stores"]
       
    mask = np.ones((image.shape[0], image.shape[1]), np.bool_)
    
    weave.inline(code, ['image', 'mask'], extra_compile_args=args, extra_link_args=args)
    
    image = np.bitwise_and(image, mask)
    
    return image


if __name__ == "__main__":
    ff = readFF(sys.argv[1])
    compressed = np.empty((4, ff.nrows, ff.ncols), np.uint8)
    compressed[0] = ff.maxpixel
    compressed[1] = ff.maxframe
    compressed[2] = ff.avepixel
    compressed[3] = ff.stdpixel
    frames = np.empty((256, ff.nrows, ff.ncols), np.uint8)
    
    ve = Extractor()
    ve.frames = frames
    ve.compressed = compressed
    ve.f = 1
    ve.min_level = 0
    ve.min_points = 0
    ve.max_points_per_frame = 1000000
    ve.max_per_frame_factor = 1000
    ve.max_points = 1000000
    ve.k1 = 3
    
    points = ve.findPoints()
    
    img = convertToBinaryImage(points, ff.nrows, ff.ncols)
    
    cv2.imshow("binary image", img.astype(np.uint8)*255)
    cv2.waitKey(0)
    
    t = time.time()
    img = clean(img)
    print "Time for cleaning:", (time.time() - t)/20
    
    cv2.imshow("cleaned image", img.astype(np.uint8)*255)
    cv2.waitKey(0)
    
    t = time.time()
    img = close(img)
    print "Time for closing:", (time.time() - t)/20
    
    cv2.imshow("closed image", img.astype(np.uint8)*255)
    cv2.waitKey(0)
    
    t = time.time()
    img = thinning(img)
    print "Time for thinning:", (time.time() - t)/20
    
    cv2.imshow("thinned image", img.astype(np.uint8)*255)
    cv2.waitKey(0)