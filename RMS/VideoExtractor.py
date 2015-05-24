from multiprocessing import Process
import numpy as np
from scipy import weave
import cv2
from RMS.Compression import Compression

class Extractor(Process):
    def __init__(self):
        super(Extractor, self).__init__()
    
    @staticmethod
    def scale(frames, compressed):
        out = np.empty((256, frames.shape[1]/16, frames.shape[2]/16), np.int32)
        
        code = """
        unsigned int x, y, n, pixel;
    
        for(n=0; n<Nframes[0]; n++) {
            for(y=0; y<Nframes[1]; y++) {
                for(x=0; x<Nframes[2]; x++) {
                    pixel = FRAMES3(n, y, x);
                    if(pixel - COMPRESSED3(2, y, x) > COMPRESSED3(3, y, x)) {
                        OUT3(n, y/16, x/16) += pixel;
                    }
               }
            }
        }
        """
        
        weave.inline(code, ['frames', 'compressed', 'out'])
        return out
    
    @staticmethod
    def extract(frames, arr):
        out = np.zeros((256, 80, 80), np.uint8)
        pos = np.zeros((256, 2), np.uint8)
        
        code = """
        unsigned int x, y, x2, y2, n, pixel, max, maxX, maxY;
        float num_equal;
        unsigned int rand_count = 0;
    
        for(n=0; n<Narr[0]; n++) {
            max = 0;
            
            for(y=0; y<Narr[1]; y++) {
                for(x=0; x<Narr[2]; x++) {
                    pixel = ARR3(n, y, x);
                    if(pixel > 40960 && pixel > max) {
                        max = pixel;
                        maxY = y;
                        maxX = x;
                    }
                }
            }
            
            if(max > 0) {
                maxY = maxY * 16 + 8;
                maxX = maxX * 16 + 8;
                
                if(maxY < 16) {
                    maxY = 16;
                } else if(maxY > Nframes[1]+15) {
                    maxY = Nframes[1]+15;
                }
                if(maxX < 16) {
                    maxX = 16;
                } else if(maxX > Nframes[2]+15) {
                    maxX = Nframes[2]+15;
                }
                
                num_equal = 0;
                max = 160; //40960 = 160 * 16 * 16
                
                for(y=maxY-16; y<maxY+16; y++) {
                    for(x=maxX-16; x<maxX+16; x++) {
                        pixel = FRAMES3(n, y, x);
                        if(pixel > max) {
                            max = pixel;
                            maxY = y;
                            maxX = x;
                        }
                    }
                }
                
                if(maxY < 40) {
                    maxY = 40;
                } else if(maxY > Nframes[1]+39) {
                    maxY = Nframes[1]+39;
                }
                if(maxX < 40) {
                    maxX = 40;
                } else if(maxX > Nframes[2]+39) {
                    maxX = Nframes[2]+39;
                }
                
                POS2(n, 0) = maxY;
                POS2(n, 1) = maxX;
                
                y2 = 0, x2 = 0;
                for(y=maxY-40; y<maxY+40; y++) {
                    for(x=maxX-40; x<maxX+40; x++) {
                        if(y<0 || x<0 || y>=Nframes[1] || x>=Nframes[2]) {
                            pixel = 0;
                        } else {
                            pixel = FRAMES3(n, y, x);
                        }
                        OUT3(n, y2, x2) = pixel;
                        x2++;
                    }
                    y2++;
                }
            }
        }
        """
        
        weave.inline(code, ['frames', 'arr', 'out', 'pos'])
        return (out, pos)
    
    
        
    
if __name__ ==  "__main__":
    cap = cv2.VideoCapture("/home/pi/RMS/m20050320_012752.wmv")
    
    frames = np.empty((256, 480, 640), np.uint8)
    
    for i in range(224):
        ret, frame = cap.read()
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        frames[i] = gray
    
    cap.release()
    
    comp = Compression(None, None, None, None, 0)
    compressed = comp.convert(frames)
    compressed = comp.compress(compressed)
    
    print np.amax(compressed[3])
    print np.amin(compressed[3])
    
    arr = Extractor.scale(frames, compressed)
    
    print np.amax(arr[100])
    
    out, pos = Extractor.extract(frames, arr)
    
    for i in range(out.shape[0]):
        print i
        cv2.imshow("bla", out[i])
        cv2.waitKey(100)
    