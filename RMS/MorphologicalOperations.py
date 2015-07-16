import cv2
from scipy import weave
import numpy as np
from RMS import ConfigReader

config = ConfigReader.Config()

def repeat(func, image, N):
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

    mask = np.ones((image.shape[0], image.shape[1]), np.bool_)    
    weave.inline(code, ['image', 'mask'], extra_compile_args=config.weaveArgs, extra_link_args=config.weaveArgs)
    
    image = np.bitwise_and(image, mask)
    
    return image

def skeleton(image): # TODO: weave implementation on np.bool_ instead of OpenCV
    image = image.astype(np.uint8)*255
    size = np.size(image)
    skel = np.zeros_like(image)
    
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    done = False
     
    while(not done):
        eroded = cv2.erode(image, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(image, temp)
        skel = cv2.bitwise_or(skel, temp)
        image = eroded.copy()
        
        if cv2.countNonZero(image)==0:
            done = True
     
    return skel.astype(np.bool_)

def close(image): # TODO: weave implementation on np.bool_ instead of OpenCV
    kernel = np.ones((3, 3), np.uint8)
    
    image = cv2.morphologyEx(image.astype(np.uint8)*255, cv2.MORPH_CLOSE, kernel)
    
    return image.astype(np.bool_)