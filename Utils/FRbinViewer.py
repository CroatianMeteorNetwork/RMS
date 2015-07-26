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
import numpy as np
import sys, os
from RMS.Formats import FFbin, FRbin

def view(ff, fr):
    if ff == None:
        background = np.zeros((576, 720), np.uint8)
    else:
        background = FFbin.read(sys.argv[1], ff).maxpixel
    
    name = fr
    fr = FRbin.read(sys.argv[1], fr)
    
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
            
            cv2.imshow(name, background)
            cv2.waitKey(200)
    
    cv2.destroyWindow(name)
            

if __name__ == "__main__":
    fr_list = [fr for fr in os.listdir(sys.argv[1]) if fr[0:2]=="FR" and fr[-3:]=="bin"]
    if(len(fr_list) == None):
        print "No files found!"
        sys.exit()
    
    ff_list = [ff for ff in os.listdir(sys.argv[1]) if ff[0:2]=="FF" and ff[-3:]=="bin"]
    
    for fr in fr_list:
        ffbin = None
        for ff in ff_list:
            if ff[2:] == fr[2:]:
                ffbin = ff
                break
        
        view(ffbin, fr)