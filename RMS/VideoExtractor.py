from multiprocessing import Process
import numpy as np
from scipy import weave, stats
import cv2
from RMS.Compression import Compression
from math import floor
import time
import matplotlib.pyplot as plt
import statsmodels.api as sm

class Extractor(Process):
    def __init__(self):
        super(Extractor, self).__init__()
    
    @staticmethod
    def scale(frames, compressed):
        out = np.zeros((frames.shape[0], floor(frames.shape[1]//16), floor(frames.shape[2]//16)), np.uint16)
        
        code = """
        unsigned int x, y, n, pixel;
    
        for(n=0; n<Nframes[0]; n++) {
            for(y=0; y<Nframes[1]; y++) {
                for(x=0; x<Nframes[2]; x++) {
                    pixel = FRAMES3(n, y, x);
                    if(pixel - COMPRESSED3(2, y, x) >= 0.95 * COMPRESSED3(3, y, x)) {
                        OUT3(n, y/16, x/16) += pixel;
                    }
               }
            }
        }
        """
        
        weave.inline(code, ['frames', 'compressed', 'out'])
        return out
    
    @staticmethod
    def findCenters(frames, arr):
        position = np.zeros((3, frames.shape[0]), np.uint16)
        
        code = """
        unsigned int x, y, n, pixel, max, max_x, max_y;
        float max_x2, max_y2, num_equal;
        
        unsigned int i = 0;
        
        for(n=0; n<Narr[0]; n++) {
            max = 0;
            max_y = 0;
            max_x = 0;
            
            for(y=0; y<Narr[1]; y++) {
                for(x=0; x<Narr[2]; x++) {
                    pixel = ARR3(n, y, x);
                    if(pixel > 45000 && pixel >= max) {
                        max = pixel;
                        max_y = y;
                        max_x = x;
                    }
                }
            }
            
            if(max > 0) {
                max_y = max_y * 16 + 8;
                max_x = max_x * 16 + 8;
                
                max = 160;
                max_y2 = max_y;
                max_x2 = max_x;
                num_equal = 1;
                
                for(y=max_y-16; y<max_y+16; y++) {
                    for(x=max_x-16; x<max_x+16; x++) {
                        if(!(y<0 || x<0 || y>=Nframes[1] || x>=Nframes[2])) {
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
                }
                
                max_y = max_y2/num_equal;
                max_x = max_x2/num_equal;
                
                POSITION2(0, i) = max_y;
                POSITION2(1, i) = max_x;
                POSITION2(2, i) = n;
                i++;
            }
        }
        """
        
        weave.inline(code, ['frames', 'arr', 'position'])
    
        y = np.trim_zeros(position[0], 'b')
        x = np.trim_zeros(position[1], 'b')
        z = np.trim_zeros(position[2], 'b')
        return x, y, z
    
    @staticmethod
    def extract(frames, alphaZX, betaZX, alphaZY, betaZY, firstFrame, lastFrame):
        firstFrame -= 15
        if firstFrame < 0:
            firstFrame = 0
        lastFrame += 16 #15 + 1 because index starts from 0
        if lastFrame > frames.shape[0]:
            lastFrame = frames.shape[0]
        
        out = np.empty((frames.shape[0], 80, 80), np.uint16)
        pos = np.zeros((frames.shape[0], 2), np.uint16)
        
        code = """
        unsigned int x, y, x2, y2, n;
        float max_x, max_y;
        
        for(n=firstFrame; n<lastFrame; n++) {
            max_x = alphaZX + betaZX * n;
            if(max_x < 40) {
                max_x = 40;
            } else if(max_x >= Nframes[2]) {
                max_x = Nframes[2] - 1;
            }
            
            max_y = alphaZY + betaZY * n;
            if(max_y < 40) {
                max_y = 40;
            } else if(max_y >= Nframes[1]) {
                max_y = Nframes[1] - 1;
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
        """
        
        weave.inline(code, ['frames', 'alphaZX', 'betaZX', 'alphaZY', 'betaZY', 'firstFrame', 'lastFrame', 'out', 'pos'])
        
        return pos, out
    
if __name__ ==  "__main__":
    cap = cv2.VideoCapture("/home/dario/Videos/m20050320_012752.wmv")
    
    frames = np.empty((224, 480, 640), np.uint8)
    
    for i in range(224):
        ret, frame = cap.read()
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        frames[i] = gray
    
    cap.release()
    
    comp = Compression(None, None, None, None, 0)
    converted = comp.convert(frames)
    compressed = comp.compress(converted)
    
    t = time.time()
    
    arr = Extractor.scale(frames, compressed)
    
    print "scale: " + str(time.time() - t)
    t = time.time()
    
    x, y, z = Extractor.findCenters(frames, arr)
    
    print "extract: " + str(time.time() - t)
    t = time.time()
    
    regressionZX = stats.linregress(z, x)
    regressionZY = stats.linregress(z, y)
    
    print "regression: " + str(time.time() - t)
    t = time.time()
    
    alphaZX = np.asscalar(regressionZX[1])
    betaZX = np.asscalar(regressionZX[0])
    alphaZY = np.asscalar(regressionZY[1])
    betaZY = np.asscalar(regressionZY[0])
    firstFrame = np.asscalar(z[0])
    lastFrame = np.asscalar(z[z.size-1])
    pos, extracted = Extractor.extract(frames, alphaZX, betaZX, alphaZY, betaZY, firstFrame, lastFrame)
    
    print "extraction: " + str(time.time() - t)
    t = time.time()
    
    background = compressed[2]
    
    for i in range(extracted.shape[0]):
        output = extracted[i]
        position = pos[i]
        
        y = 0
        x = 0
        y2 = 0
        for y in range(position[0]-40, position[0]+39):
            x2 = 0
            for x in range(position[1]-40, position[1]+39):
                pixel = output[y2, x2]
                if pixel > 0:
                    background[y, x] = pixel
                x2 = x2 + 1
            y2 = y2 + 1
        
        cv2.imshow("bla", background)
        cv2.waitKey(50)