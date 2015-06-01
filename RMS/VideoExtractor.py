from multiprocessing import Process
import numpy as np
from scipy import weave, stats
import cv2
from RMS.Compression import Compression
from math import floor
import time

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
        position = np.zeros((3, frames.shape[0]), np.uint16) # x, y, frameNum (z)
        size = np.zeros((2, frames.shape[0]), np.uint8) # half-height, half-width
        
        code = """
        unsigned int x, y, n, pixel, max, max_x, max_y;
        float max_x2, max_y2, dist_x, dist_y, num_equal;
        
        unsigned int i = 0;
        
        for(n=0; n<Narr[0]; n++) {
            max = 0;
            max_y = 0;
            max_x = 0;
            
            // find brightest segment brighter than threshold 
            
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
                // find center
                
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
                
                max_y = max_y2 / num_equal;
                max_x = max_x2 / num_equal;
                
                POSITION2(0, i) = max_y;
                POSITION2(1, i) = max_x;
                POSITION2(2, i) = n;
                
                i++;
                
                // find size
                
                dist_x = 0;
                dist_y = 0;
                num_equal = 0;
                
                for(y=max_y-64; y<max_y+64; y++) {
                    for(x=max_x-64; x<max_x+64; x++) {
                        if(!(y<0 || x<0 || y>=Nframes[1] || x>=Nframes[2])) {
                            pixel = FRAMES3(n, y, x);
                            if(pixel == max) {
                                dist_y += abs(max_y - y);
                                dist_x += abs(max_x - x);
                                num_equal++;
                            }
                        }
                    }
                }
                
                dist_y = dist_y / num_equal * 2;
                if(dist_y < 8) {
                    dist_y = 8;
                }
                dist_x = dist_x / num_equal * 2;
                if(dist_x < 8) {
                    dist_x = 8;
                }
                
                SIZE2(0, n) = dist_y;
                SIZE2(1, n) = dist_x;
            }
        }
        """
        
        weave.inline(code, ['frames', 'arr', 'position', 'size'])
    
        y = np.trim_zeros(position[0], 'b')
        x = np.trim_zeros(position[1], 'b')
        z = np.trim_zeros(position[2], 'b')
        
        return x, y, z, size
    
    @staticmethod
    def extract(frames, size, alphaZX, betaZX, alphaZY, betaZY, firstFrame, lastFrame):
        firstFrame -= 15
        if firstFrame < 0:
            firstFrame = 0
        lastFrame += 16 #15 + 1 because index starts from 0
        if lastFrame > frames.shape[0]:
            lastFrame = frames.shape[0]
        
        out = np.zeros((frames.shape[0], 256, 256), np.uint8) # y, x
        pos = np.zeros((frames.shape[0], 2), np.uint16) # y, x
        outsize = np.zeros((frames.shape[0], 2), np.uint8) # half-height, half-width
        
        code = """
        unsigned int x, y, x2, y2, n, i, half_width, half_height;
        float max_x, max_y, k_width = 0, k_height = 0,
        first_forward_frame = 0, first_forward_width = 0,  first_forward_height= 0,
        first_backwards_frame = 0, first_backwards_width = 0, first_backwards_height = 0;
        
        for(n=firstFrame; n<lastFrame; n++) {
            
            // calculate size
            
            half_height = SIZE2(0, n);
            if(half_height != 0) { // size is available in SIZE2 array
                half_width = SIZE2(1, n);
                
            } else if(n < first_forward_frame) { // size missing from SIZE2 array
                // find size from coefficients
                half_height = first_backwards_height + k_height * (n - first_backwards_frame + 1);
                half_width = first_backwards_width + k_width * (n - first_backwards_frame + 1);
                
            } else { //size & coefficients missing
                // find coefficients for extrapolating between previous and next defined size
                
                first_forward_width = 0;
                first_backwards_width = 0;
                for(i=n+1; i<lastFrame; i++) {
                    first_forward_height = SIZE2(0, i);
                    if(first_forward_height != 0) {
                        first_forward_width = SIZE2(1, i);
                        first_forward_frame = i;
                        break;
                    }
                }
                for(i=n-1; i>=firstFrame; i--) {
                    first_backwards_height = SIZE2(0, i);
                    if(first_backwards_height != 0) {
                        first_backwards_width = SIZE2(1, i);
                        first_backwards_frame = i;
                        break;
                    }
                }
                if(first_forward_width == 0) { // we are dealing with frames at end of the sequence
                    first_forward_frame = 0; // don't use coefficients
                    half_height = first_backwards_height; // and copy size from last frame with defined size
                    half_width = first_backwards_width;
                } else if(first_backwards_width == 0) { //we are dealing with frames at start of the sequence 
                    first_forward_frame = 0; // don't use coefficients
                    half_height = first_forward_height; // and copy size from next frame with defined size
                    half_width = first_forward_width;
                } else {
                    k_height = (first_forward_height - first_backwards_height) / (first_forward_frame - first_backwards_frame);
                    k_width = (first_forward_width - first_backwards_width) / (first_forward_frame - first_backwards_frame);
                    
                
                    // find size from coefficients
                    half_height = first_backwards_height + k_height * (n - first_backwards_frame + 1);
                    half_width = first_backwards_width + k_width * (n - first_backwards_frame + 1);
                }
            }
            
            OUTSIZE2(n, 0) = half_height;
            OUTSIZE2(n, 1) = half_width;
            
            // calculate center from regression coefficients
            
            max_x = alphaZX + betaZX * n;
            if(max_x < half_width) {
                max_x = half_width;
            } else if(max_x >= Nframes[2]-half_width) {
                max_x = Nframes[2] - half_width - 1;
            }
            
            max_y = alphaZY + betaZY * n;
            if(max_y < half_height) {
                max_y = half_height;
            } else if(max_y >= Nframes[1] - half_height) {
                max_y = Nframes[1] - half_height - 1;
            }
            
            POS2(n, 0) = max_y;
            POS2(n, 1) = max_x;
            
            // extract part of frame specified by position (max_x, max_y) and size (half_width, half_height)
            
            y2 = 0;
            for(y=max_y-half_height; y<max_y+half_height; y++) {
                x2 = 0;
                for(x=max_x-half_width; x<max_x+half_width; x++) {
                    OUT3(n, y2, x2) = FRAMES3(n, y, x);
                    x2++;
                }
                y2++;
            }
        }
        """
        
        weave.inline(code, ['frames', 'size', 'alphaZX', 'betaZX', 'alphaZY', 'betaZY', 'firstFrame', 'lastFrame', 'outsize', 'out', 'pos'])
        
        return pos, outsize, out
    
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
    
    x, y, z, size = Extractor.findCenters(frames, arr)
    
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
    
    pos, size, extracted = Extractor.extract(frames, size, alphaZX, betaZX, alphaZY, betaZY, firstFrame, lastFrame)
    
    print "extraction: " + str(time.time() - t)
    t = time.time()
    
    background = compressed[2]
    
    for i in range(extracted.shape[0]):
        output = extracted[i]
        position = pos[i]
        hh = size[i, 0]
        hw = size[i, 1]
        
        y = 0
        x = 0
        y2 = 0
        for y in range(position[0]-hh, position[0]+hh-1):
            x2 = 0
            for x in range(position[1]-hw, position[1]+hw-1):
                pixel = output[y2, x2]
                if pixel > 0:
                    background[y, x] = pixel
                x2 = x2 + 1 
            y2 = y2 + 1
        
        cv2.imshow("bla", background)
        cv2.waitKey(50)