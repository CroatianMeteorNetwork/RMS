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

from RMS import BufferedCapture
from RMS import Compression
from multiprocessing import Manager

if __name__ == "__main__":
    manager = Manager()
    framesList = manager.list()
    
    bc = BufferedCapture(framesList)
    
    c = Compression(framesList, 499)
    
    bc.startCapture()
    c.start()
    
    raw_input("Press Enter to stop...")
    
    bc.stopCapture()
    c.stop()