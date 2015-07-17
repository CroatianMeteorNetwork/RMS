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

from ConfigParser import RawConfigParser

class Config:
    def __init__(self):
        self.weaveArgs = ["-O3"]
        
        ##### VideoExtraction
        self.f = 16                    # subsampling factor
        
        # params for Extractor.findPoints()
        self.min_level = 40            # ignore pixel if below this level
        self.min_pixels = 8            # minimum number of pixels required to add event point
        self.k1 = 4                    # k1 factor for thresholding
        self.max_points_per_frame = 30 # ignore frame if there are too many points in it (ie. flare)
        self.max_per_frame_factor = 10 # multiplied with median number of points for flare detection
        self.max_points = 190          # if there are too many points in total after flare removal, randomize them
        self.min_frames = 4            # minimum number of frames
        
        # params for Extractor.testPoints()
        self.min_points = 5            # minimum number of event points
        
        # params for Extractor.extract()
        self.before = 0.15             # percentage of frames to extrapolate before detected start of meteor trail
        self.after = 0.3               # percentage of frames to extrapolate after detected end of meteor trail
        self.limitForSize = 0.90
        self.minSize = 40
        self.maxSize = 192
        
        # params for Grouping3D
        self.distance_treshold = 70    # maximum distance between the line and the point to be takes as a part of the same line
        self.gap_treshold = 130        # maximum allowed gap between points
        self.line_distance_const = 4   # constant that determines the influence of average point distance on the line quality
        self.point_ratio_treshold = 0.7# ratio of how many points must be close to the line before considering searching for another line
        self.max_lines = 5             # maximum number of lines

def parse(filename):
    parser = RawConfigParser()
    parser.read(filename)
    
    config = Config()
    
    if parser.has_section("Build"):
        parseArgs(config, parser)
    
    if parser.has_section("Compression"):
        parseCompression(config, parser)
    
    if parser.has_section("VideoExtraction"):
        parseExtractionParams(config, parser)
    
    return config

def parseArgs(config, parser):
    if parser.has_option("Build", "weave"):
        config.weaveArgs = parser.get("Build", "weave").split()
        
    if parser.has_option("Build", "extension"):
        config.weaveArgs = parser.get("Build", "extension").split()

def parseCompression(config, parser):
    pass

def parseExtractionParams(config, parser):
    if parser.has_option("VideoExtraction", "subsampling_size"):
        config.f = parser.getint("VideoExtraction", "subsampling_size")
    
    if parser.has_option("VideoExtraction", "minimal_level"):
        config.min_level = parser.getint("VideoExtraction", "minimal_level")
    
    if parser.has_option("VideoExtraction", "minimum_pixels"):
        config.min_pixels = parser.getint("VideoExtraction", "minimum_pixels")
    
    if parser.has_option("VideoExtraction", "k1"):
        config.k1 = parser.getint("VideoExtraction", "k1")
    
    if parser.has_option("VideoExtraction", "max_points_per_frame"):
        config.max_points_per_frame = parser.getint("VideoExtraction", "max_points_per_frame")
    
    if parser.has_option("VideoExtraction", "max_per_frame_factor"):
        config.max_per_frame_factor = parser.getint("VideoExtraction", "max_per_frame_factor")
    
    if parser.has_option("VideoExtraction", "max_points"):
        config.max_points = parser.getint("VideoExtraction", "max_points")
    
    if parser.has_option("VideoExtraction", "min_frames"):
        config.min_frames = parser.getint("VideoExtraction", "min_frames")
    
    if parser.has_option("VideoExtraction", "min_points"):
        config.min_points = parser.getint("VideoExtraction", "min_points")
    
    if parser.has_option("VideoExtraction", "extend_before"):
        config.before = parser.getfloat("VideoExtraction", "extend_before")
    
    if parser.has_option("VideoExtraction", "extend_after"):
        config.after = parser.getfloat("VideoExtraction", "extend_after")
    
    if parser.has_option("VideoExtraction", "min_window_size"):
        config.minSize = parser.getint("VideoExtraction", "min_window_size")
    
    if parser.has_option("VideoExtraction", "max_window_size"):
        config.maxSize = parser.getint("VideoExtraction", "max_window_size")
    
    if parser.has_option("VideoExtraction", "threshold_for_size"):
        config.limitForSize = parser.getfloat("VideoExtraction", "threshold_for_size")
    
    if parser.has_option("VideoExtraction", "distance_treshold"):
        config.distance_treshold = parser.getint("VideoExtraction", "distance_treshold")
    
    if parser.has_option("VideoExtraction", "gap_treshold"):
        config.gap_treshold = parser.getint("VideoExtraction", "gap_treshold")
    
    if parser.has_option("VideoExtraction", "line_distance_const"):
        config.line_distance_const = parser.getint("VideoExtraction", "line_distance_const")
    
    if parser.has_option("VideoExtraction", "point_ratio_treshold"):
        config.point_ratio_treshold = parser.getfloat("VideoExtraction", "point_ratio_treshold")        
    
    if parser.has_option("VideoExtraction", "max_lines"):
        config.max_lines = parser.getint("VideoExtraction", "max_lines")