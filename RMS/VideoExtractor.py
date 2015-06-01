from multiprocessing import Process
import numpy as np
from scipy import weave, stats
import cv2
from RMS.Compression import Compression
from math import floor
import time
import statsmodels.api as sm

class Extractor(Process):
    def __init__(self):
        super(Extractor, self).__init__()
    
    def scale(self, frames, compressed):
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
    
    def findCenters(self, frames, arr):
        position = np.zeros((3, frames.shape[0]), np.uint16) # x, y, frameNum (z)
        size = np.zeros((frames.shape[0]), np.uint8) # half-height, half-width
        
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
                
                for(y=max_y-128; y<max_y+128; y++) {
                    for(x=max_x-128; x<max_x+128; x++) {
                        if(!(y<0 || x<0 || y>=Nframes[1] || x>=Nframes[2])) {
                            pixel = FRAMES3(n, y, x);
                            if(pixel >= 0.9 * max) {
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
                
                if(dist_y > dist_x) {
                    SIZE1(n) = dist_y;
                } else {
                    SIZE1(n) = dist_x;
                }
            }
        }
        """
        
        weave.inline(code, ['frames', 'arr', 'position', 'size'])
    
        y = np.trim_zeros(position[0], 'b')
        x = np.trim_zeros(position[1], 'b')
        z = np.trim_zeros(position[2], 'b')
        
        return x, y, z, size
    
    def extract(self, frames, size, alphaZX, betaZX, alphaZY, betaZY, firstFrame, lastFrame):
        firstFrame -= 15
        if firstFrame < 0:
            firstFrame = 0
        lastFrame += 26 #15 + 10 + 1; 1 because index starts from 0, 10 because we are more interested in the end of meteor
        if lastFrame > frames.shape[0]:
            lastFrame = frames.shape[0]
        
        out = np.zeros((frames.shape[0], 256, 256), np.uint8) # y, x
        pos = np.zeros((frames.shape[0], 2), np.uint16) # y, x
        outsize = np.zeros((frames.shape[0]), np.uint8)
        
        code = """
        unsigned int x, y, x2, y2, n, i, half_size;
        float max_x, max_y, k = 0,
        next_frame = 0, next_size = 0,
        previous_frame = 0, previous_size = 0;
        
        for(n=firstFrame; n<lastFrame; n++) {
            
            // calculate size
            
            half_size = SIZE1(n);
            if(half_size != 0) {
                // size from SIZE1 array is fine
                
            } else if(n < next_frame) { // size missing from SIZE1 array
                // find size from coefficients
                half_size = previous_size + k * (n - previous_frame + 1);
                
            } else { //size & coefficients missing
                // find coefficients for extrapolating between previous and next defined size
                
                next_frame = 0;
                previous_frame = 0;
                for(i=n+1; i<lastFrame; i++) {
                    next_size = SIZE1(i);
                    if(next_size != 0) {
                        next_frame = i;
                        break;
                    }
                }
                for(i=n-1; i>=firstFrame; i--) {
                    previous_size = SIZE1(i);
                    if(previous_size != 0) {
                        previous_frame = i;
                        break;
                    }
                }
                if(next_frame == 0) { // we are dealing with frames at end of the sequence
                    next_frame = lastFrame;
                    k = 0; // k=0 means that we will copy size from last frame with defined size
                } else if(previous_frame == 0) { //we are dealing with frames at start of the sequence 
                    k = 0; // same here
                    previous_size = next_size; // but we are copying from next frame with defined size
                } else {
                    k = (next_size - previous_size) / (next_frame - previous_frame);
                }
                
                // find size from coefficients
                half_size = previous_size + k * (n - previous_frame + 1);
            }
            
            OUTSIZE1(n) = half_size * 2;
            
            // calculate center from regression coefficients
            
            max_x = alphaZX + betaZX * n;
            if(max_x < half_size) {
                max_x = half_size;
            } else if(max_x >= Nframes[2]-half_size) {
                max_x = Nframes[2] - half_size - 1;
            }
            
            max_y = alphaZY + betaZY * n;
            if(max_y < half_size) {
                max_y = half_size;
            } else if(max_y >= Nframes[1] - half_size) {
                max_y = Nframes[1] - half_size - 1;
            }
            
            POS2(n, 0) = max_y;
            POS2(n, 1) = max_x;
            
            // extract part of frame specified by position (max_x, max_y) and size (half_width, half_height)
            
            y2 = 0;
            for(y=max_y-half_size; y<max_y+half_size; y++) {
                x2 = 0;
                for(x=max_x-half_size; x<max_x+half_size; x++) {
                    OUT3(n, y2, x2) = FRAMES3(n, y, x);
                    x2++;
                }
                y2++;
            }
        }
        """
        
        weave.inline(code, ['frames', 'size', 'alphaZX', 'betaZX', 'alphaZY', 'betaZY', 'firstFrame', 'lastFrame', 'outsize', 'out', 'pos'])
        
        return pos, outsize, out
    
    def save(self, position, size, extracted, fileName):
        file = "FR" + fileName + ".bin"
        
        with open(file, "wb") as f:
            f.write(struct.pack('I', extracted.shape[0]))               # frames num
            
            for n in range(extracted.shape[0]):
                f.write(struct.pack('I', position[n, 0]))               # y of center
                f.write(struct.pack('I', position[n, 1]))               # x of center
                f.write(struct.pack('I', size[n]))          
                np.resize(extracted[n], (size[n], size[n])).tofile(f)   # frame
        
    
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
    
    extractor = Extractor()
    
    t = time.time()
    
    arr = extractor.scale(frames, compressed)
    
    print "scale: " + str(time.time() - t)
    t = time.time()
    
    x, y, z, size = extractor.findCenters(frames, arr)
    
    print "find centers: " + str(time.time() - t)
    t = time.time()
    
    constant = sm.add_constant(z)
    regressionZX = sm.RLM(x, constant, M=sm.robust.norms.HuberT()).fit().params
    regressionZY = sm.RLM(y, constant, M=sm.robust.norms.HuberT()).fit().params
    
    alphaZX = np.asscalar(regressionZX[0])
    betaZX = np.asscalar(regressionZX[1])
    alphaZY = np.asscalar(regressionZY[0])
    betaZY = np.asscalar(regressionZY[1])
    firstFrame = np.asscalar(z[0])
    lastFrame = np.asscalar(z[z.size-1])
    
    print "regression: " + str(time.time() - t)
    t = time.time()
    
    pos, size, extracted = extractor.extract(frames, size, alphaZX, betaZX, alphaZY, betaZY, firstFrame, lastFrame)
    
    print "extraction: " + str(time.time() - t)
    t = time.time()
    
    background = compressed[2]
    
    for i in range(extracted.shape[0]):
        output = extracted[i]
        position = pos[i]
        
        y = 0
        x = 0
        y2 = 0
        for y in range(position[0]-size[i]/2, position[0]+size[i]/2-1):
            x2 = 0
            for x in range(position[1]-size[i]/2, position[1]+size[i]/2-1):
                pixel = output[y2, x2]
                if pixel > 0:
                    background[y, x] = pixel
                x2 = x2 + 1 
            y2 = y2 + 1
        
        cv2.imshow("bla", background)
        cv2.waitKey(50)