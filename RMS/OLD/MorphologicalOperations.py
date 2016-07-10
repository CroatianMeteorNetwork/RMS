# RPi Meteor Station
# Copyright (C) 2015  Dario Zubovic
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import cv2
from scipy import weave
import numpy as np
from RMS import ConfigReader

config = ConfigReader.Config()

def repeat(func, image, N):
    """ Repeat specified morphological operation N number of times.
    
    @param func: morphological operation to be repeated
    @param image: input image
    @param N: number of times to repeat func or None in which case it's repeated until result of operation doesn't changes
    
    @return image
    """
    
    if N == None:
        # repeat until previous iteration and current are same
        image = func(image)
        previous = None
        
        while not np.array_equal(image, previous):
            previous = np.copy(image)
            image = func(image)
        
    else:
        # repeat N times
        for _ in range(N):
            func(image)
    
    return image

def clean(image):
    """ Clean isolated pixels, as in:
     0  0  0      0  0  0
     0  1  0  =>  0  0  0
     0  0  0      0  0  0
    
    @param image: input image
    
    @return cleaned image
    """
    
    code = """
    bool p2, p3, p4, p5, p6, p7, p8, p9;
    unsigned int y, x;
    
    unsigned int ym = Nimage[0] - 1;
    unsigned int xm = Nimage[1] - 1;
    
    for(y = 0; y < Nimage[0]; y++) {
        for(x = 0; x < Nimage[1]; x++) {
            if(!IMAGE2(y, x)){
                continue;
            }
            
            p2 = y==0 ? 0 : IMAGE2(y-1, x);
            p3 = y==0 || x==xm ? 0 : IMAGE2(y-1, x+1);
            p4 = x==xm ? 0 : IMAGE2(y, x+1);
            p5 = y==ym || x==xm ? 0 : IMAGE2(y+1, x+1);
            p6 = y==ym ? 0 : IMAGE2(y+1, x);
            p7 = y==ym || x==0 ? 0 : IMAGE2(y+1, x-1);
            p8 = x==0 ? 0 : IMAGE2(y, x-1);
            p9 = y==0 || x==0 ? 0 : IMAGE2(y-1, x-1);

            if(!p2 && !p3 && !p4 && !p5 && !p6 && !p7 && !p8 && !p9) {
                IMAGE2(y, x) = 0;
            }
        }
    }
    """
    
    weave.inline(code, ['image'], extra_compile_args=config.weaveArgs, extra_link_args=config.weaveArgs)
    
    return image

def spur(image):
    """ Remove pixels that have only one neighbor, as in:
     0  0  0  0      0  0  0  0
     0  1  0  0  =>  0  0  0  0
     0  0  1  1      0  0  1  1
     0  0  0  1      0  0  0  1
    
    @param image: input image
    
    @return cleaned image
    """
    
    code = """
    bool p2, p3, p4, p5, p6, p7, p8, p9;
    unsigned int y, x;
    
    unsigned int ym = Nimage[0] - 1;
    unsigned int xm = Nimage[1] - 1;
    
    for(y = 0; y < Nimage[0]; y++) {
        for(x = 0; x < Nimage[1]; x++) {
            if(!IMAGE2(y, x)){
                continue;
            }
            
            p2 = y==0 ? 0 : IMAGE2(y-1, x);
            p3 = y==0 || x==xm ? 0 : IMAGE2(y-1, x+1);
            p4 = x==xm ? 0 : IMAGE2(y, x+1);
            p5 = y==ym || x==xm ? 0 : IMAGE2(y+1, x+1);
            p6 = y==ym ? 0 : IMAGE2(y+1, x);
            p7 = y==ym || x==0 ? 0 : IMAGE2(y+1, x-1);
            p8 = x==0 ? 0 : IMAGE2(y, x-1);
            p9 = y==0 || x==0 ? 0 : IMAGE2(y-1, x-1);

            if((p2 && !p3 && !p4 && !p5 && !p6 && !p7 && !p8 && !p9) ||
               (!p2 && p3 && !p4 && !p5 && !p6 && !p7 && !p8 && !p9) ||
               (!p2 && !p3 && p4 && !p5 && !p6 && !p7 && !p8 && !p9) ||
               (!p2 && !p3 && !p4 && p5 && !p6 && !p7 && !p8 && !p9) ||
               (!p2 && !p3 && !p4 && !p5 && p6 && !p7 && !p8 && !p9) ||
               (!p2 && !p3 && !p4 && !p5 && !p6 && p7 && !p8 && !p9) ||
               (!p2 && !p3 && !p4 && !p5 && !p6 && !p7 && p8 && !p9) ||
               (!p2 && !p3 && !p4 && !p5 && !p6 && !p7 && !p8 && p9)) {
                MASK2(y, x) = 0;
            }
        }
    }
    """
    
    mask = np.ones_like(image)
    weave.inline(code, ['image', 'mask'], extra_compile_args=config.weaveArgs, extra_link_args=config.weaveArgs)
    
    np.bitwise_and(image, mask, image)
    
    return image

def bridge(image):
    """ Connect pixels on opposite sides, if other pixels are 0, as in:
     0  0  1      0  0  1
     0  0  0  =>  0  1  0
     1  0  0      1  0  0
    
    @param image: input image
    
    @return bridged image
    """
    
    code = """
    bool p2, p3, p4, p5, p6, p7, p8, p9;
    unsigned int y, x;
    
    for(y = 1; y < Nimage[0]-1; y++) {
        for(x = 1; x < Nimage[1]-1; x++) {
            if(IMAGE2(y, x) && MASK2(y, x)){
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
            
            if((p2 && !p3 && !p4 && !p5 && p6 && !p7 && !p8 && !p9) ||
               (!p2 && !p3 && !p4 && p5 && !p6 && !p7 && !p8 && p9) ||
               (!p2 && !p3 && p4 && !p5 && !p6 && !p7 && p8 && !p9) ||
               (!p2 && p3 && !p4 && !p5 && !p6 && p7 && !p8 && !p9)){
                MASK2(y, x) = 1;
            }
        }
    }
    """
    
    mask = np.zeros_like(image)
    weave.inline(code, ['image', 'mask'], extra_compile_args=config.weaveArgs, extra_link_args=config.weaveArgs)
    
    np.bitwise_or(image, mask, image)
    
    return image

def thin(image):
    """ Zhang-Suen fast thinning algorithm, modified to only single pass.
        
    @param image: input image
    
    @return thinned image
    """
    
    code = """
    bool p2, p3, p4, p5, p6, p7, p8, p9;
    unsigned int y, x, A, B, m1, m2;
    
    bool first = true;
    
    do {
        // one thinning pass
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
                
                if(first) {
                    m1 = (p2 * p4 * p6);
                    m2 = (p4 * p6 * p8);
                } else {
                    m1 = (p2 * p4 * p8);
                    m2 = (p2 * p6 * p8);
                }
                
                if(A == 1 && (B >= 2 && B <= 6) && (m1 == 0 && m2 == 0)) {
                    MASK2(y, x) = 0;
                }
            }
        }
        
        // image &= mask; and reset mask to ones if first pass
        for(y = 1; y < Nimage[0]-1; y++) {
            for(x = 1; x < Nimage[1]-1; x++) {
                IMAGE2(y, x) &= MASK2(y, x);
                
                if(first) { // reset MASK2
                    MASK2(y, x) = 1;
                }
            }
        }
    
        first = false;
        
    } while(first);
    """
    
    mask = np.ones((image.shape[0], image.shape[1]), np.bool_)
    weave.inline(code, ['image', 'mask'], extra_compile_args=config.weaveArgs, extra_link_args=config.weaveArgs)
    
    return image


def _thinningIteration(im, iter):
    I, M = im, np.zeros(im.shape, np.uint8)
    expr = """
    for (int i = 1; i < NI[0]-1; i++) {
        for (int j = 1; j < NI[1]-1; j++) {
            int p2 = I2(i-1, j);
            int p3 = I2(i-1, j+1);
            int p4 = I2(i, j+1);
            int p5 = I2(i+1, j+1);
            int p6 = I2(i+1, j);
            int p7 = I2(i+1, j-1);
            int p8 = I2(i, j-1);
            int p9 = I2(i-1, j-1);
            int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                     (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                     (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                     (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
            int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
            int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
            int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);
            if (A == 1 && B >= 2 && B <= 6 && m1 == 0 && m2 == 0) {
                M2(i,j) = 1;
            }
        }
    } 
    """

    weave.inline(expr, ["I", "iter", "M"])
    return (I & ~M)


def thin2(src):
    """ Zhang-Suen fast thinning algorithm.

    Source: https://github.com/bsdnoobz/zhang-suen-thinning/blob/master/thinning.py
    """
    
    src = src.astype(np.uint8)
    dst = src.copy()
    prev = np.zeros(src.shape[:2], np.uint8)
    diff = None

    while True:
        dst = _thinningIteration(dst, 0)
        dst = _thinningIteration(dst, 1)
        diff = np.absolute(dst - prev)
        prev = dst.copy()
        if np.sum(diff) == 0:
            break

    return dst



def close(image): # TODO: weave implementation on np.bool_ instead of OpenCV
    """ Morphological closing (dilation followed by erosion).
    
    @param image: input image
    
    @return cleaned image
    """
    
    kernel = np.ones((3, 3), np.uint8)
    
    image = cv2.morphologyEx(image.astype(np.uint8)*255, cv2.MORPH_CLOSE, kernel)
    
    return image.astype(np.bool_)