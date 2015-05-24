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
        out = np.zeros((frames.shape[0], frames.shape[1]/16, frames.shape[2]/16), np.int32)
        
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
        out = np.zeros((frames.shape[0], 80, 80), np.uint8)
        pos = np.zeros((frames.shape[0], 2), np.uint8)
        
        code = """
        unsigned int x, y, x2, y2, n, pixel, max, max_x, max_y;
        float max_x2, max_y2, num_equal;
    
        for(n=0; n<Narr[0]; n++) {
            max = 0;
            
            for(y=0; y<Narr[1]; y++) {
                for(x=0; x<Narr[2]; x++) {
                    pixel = ARR3(n, y, x);
                    if(pixel > 45056 && pixel > max) {
                        max = pixel;
                        max_y = y;
                        max_x = x;
                    }
                }
            }
            
            if(max > 0) {
                max_y = max_y * 16 + 8;
                max_x = max_x * 16 + 8;
                
                if(max_y < 16) {
                    max_y = 16;
                } else if(max_y > Nframes[1]+15) {
                    max_y = Nframes[1]+15;
                }
                if(max_x < 16) {
                    max_x = 16;
                } else if(max_x > Nframes[2]+15) {
                    max_x = Nframes[2]+15;
                }
                
                max = 176; //45056 = 176 * 16 * 16
                max_y2 = max_y;
                max_x2 = max_x;
                num_equal = 1;
                
                for(y=max_y-16; y<max_y+16; y++) {
                    for(x=max_x-16; x<max_x+16; x++) {
                        pixel = FRAMES3(n, y, x);
                        if(pixel > max) {
                            max = pixel;
                            max_y2 = y;
                            max_x2 = x;
                            num_equal = 1;
                        } else if(pixel == max) {
                            max_y2 += y;
                            max_x2 += x;
                            num_equal++;
                        }
                    }
                }
                
                max_y = max_y2/num_equal;
                max_x = max_x2/num_equal;
                
                if(max_y < 40) {
                    max_y = 40;
                } else if(max_y > Nframes[1]+39) {
                    max_y = Nframes[1]+39;
                }
                if(max_x < 40) {
                    max_x = 40;
                } else if(max_x > Nframes[2]+39) {
                    max_x = Nframes[2]+39;
                }
                
                POS2(n, 0) = max_y;
                POS2(n, 1) = max_x;
                
                y2 = 0;
                for(y=max_y-40; y<max_y+40; y++) {
                    x2 = 0;
                    for(x=max_x-40; x<max_x+40; x++) {
                        OUT3(n, y2, x2) = FRAMES3(n, y, x);
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
    
    frames = np.empty((224, 480, 640), np.uint8)
    
    for i in range(224):
        ret, frame = cap.read()
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        frames[i] = gray
    
    cap.release()
    
    comp = Compression(None, None, None, None, 0)
    compressed = comp.convert(frames)
    compressed = comp.compress(compressed)
    
    print compressed.shape
    
    arr = Extractor.scale(frames, compressed)
    
    print arr.shape
    
    out, pos = Extractor.extract(frames, arr)
    
    print out.shape
    
    background = compressed[2]
    
    for i in range(out.shape[0]):
        frame = out[i]
        position = pos[i]
        
        y2 = 0
        for y in range(position[0]-40, position[0]+40):
            x2 = 0
            for x in range(position[1]-40, position[1]+40):
                pixel = frame[y2, x2]
                if pixel > 0:
                    background[y, x] = pixel
                x2 = x2 + 1
            y2 = y2 + 1
        
        print i
        cv2.imshow("bla", background)
        cv2.waitKey(50)
    