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
from RMS.Routines import Grouping3D

class Config:
    def __init__(self):
        ##### System
        self.stationID = 499
        
        ##### Capture
        self.width = 720
        self.height = 576
        self.deviceID = 0
        self.deinterlace_order = 1
        
        self.weaveArgs = ["-O3"]
        
        ##### VideoExtraction
        self.f = 16                    # subsampling factor
        self.max_time = 25             # maximum time for line finding
        
        # params for Extractor.findPoints()
        self.min_level = 40            # ignore pixel if below this level
        self.min_pixels = 8            # minimum number of pixels required to add event point
        self.k1 = 4                    # k1 factor for thresholding
        self.max_points_per_frame = 30 # ignore frame if there are too many points in it (ie. flare)
        self.max_per_frame_factor = 10 # multiplied with median number of points for flare detection
        self.max_points = 190          # if there are too many points in total after flare removal, randomize them
        self.min_frames = 4            # minimum number of frames
        
        # params for Extractor.testPoints()
        self.min_points = 4            # minimum number of event points
        
        # params for Extractor.extract()
        self.before = 0.15             # percentage of frames to extrapolate before detected start of meteor trail
        self.after = 0.3               # percentage of frames to extrapolate after detected end of meteor trail
        self.limitForSize = 0.90
        self.minSize = 40
        self.maxSize = 192
        
        # params for Grouping3D
        self.distance_treshold = 4900  # maximum distance between the line and the point to be takes as a part of the same line
        self.gap_treshold = 16900      # maximum allowed gap between points
        self.line_minimum_frame_range = 3   # minimum range of frames that a line should cover (eliminates flash detections)
        self.line_distance_const = 4   # constant that determines the influence of average point distance on the line quality
        self.point_ratio_treshold = 0.7# ratio of how many points must be close to the line before considering searching for another line
        self.max_lines = 5             # maximum number of lines


        # KHT detection parameters
        self.k1_det = 1.5 # weight for stddev in thresholding for faint meteor detection
        self.j1 = 9 # absolute levels above average in thresholding for faint meteor detection
        self.max_white_ratio = 0.07 # maximum ratio of white to all pixels on a thresholded image (used to avoid searching on very messed up images)
        self.time_window_size = 64 # size of the time window which will be slided over the time axis
        self.time_slide = 32 # subdivision size of the time axis (256 will be divided into 256/time_slide parts)
        self.max_lines_det = 30 # maximum number of lines to be found on the time segment with KHT
        self.line_min_dist = 40 # Minimum distance between KHT lines in Cartesian space to merge them (used for merging similar lines after KHT)
        self.stripe_width = 20 # width of the stripe around the line
        self.kht_lib_path = "build/lib.linux-x86_64-2.7/kht_module.so" # path to the compiled KHT module

        self.max_points_det = 600 # maximumum number of points during 3D line search in faint meteor detection (used to minimize runtime)
        self.distance_treshold_det = 50**2 # maximum distance between the line and the point to be takes as a part of the same line
        self.gap_treshold_det = 500**2 # maximum allowed gap between points
        self.min_pixels_det = 10 # minimum number of pixels in a strip
        self.line_minimum_frame_range_det = 4 # minimum number of frames per one detection
        self.line_distance_const_det = 4 # constant that determines the influence of average point distance on the line quality
        self.max_time_det = 10 # maximum time in seconds for which line finding algorithm can run

        self.vect_angle_thresh = 20 # angle similarity between 2 lines in a stripe to be merged
        self.frame_extension = 3 # how many frames to check during centroiding before and after the initially determined frame range

        self.centroids_max_deviation = 2 # maximum deviation of a centroid point from a LSQ fitted line (if above max, it will be rejected)
        self.centroids_max_distance =  30 # maximum distance in pixels between centroids (used for filtering spurious centroids)

def parse(filename):
    parser = RawConfigParser()
    parser.read(filename)
    
    config = Config()
    
    parseSystem(config, parser)
    
    if parser.has_section("Capture"):
        parseCapture(config, parser)
    
    if parser.has_section("Build"):
        parseBuildArgs(config, parser)
    
    if parser.has_section("Compression"):
        parseCompression(config, parser)
    
    if parser.has_section("VideoExtraction"):
        parseExtraction(config, parser)

    if parser.has_section("Detection"):
        parseDetection(config, parser)
    
    return config

def parseSystem(config, parser):
    if not parser.has_section("System"):
        raise RuntimeError("Not configurated!")
    
    try:
        config.stationID = parser.getint("System", "stationID")
    except NoOptionError:
        raise RuntimeError("Not configurated!")

def parseCapture(config, parser):
    if parser.has_option("Capture", "width"):
        config.width = parser.getint("Capture", "width")
       
    if parser.has_option("Capture", "height"):
        config.height = parser.getint("Capture", "height")
    
    if parser.has_option("Capture", "device"):
        config.deviceID = parser.getint("Capture", "device")

    if parser.has_option("Capture", "deinterlace_order"):
        config.deinterlace_order = parser.getint("Capture", "deinterlace_order")

def parseBuildArgs(config, parser):
    if parser.has_option("Build", "weave"):
        config.weaveArgs = parser.get("Build", "weave").split()
        
    if parser.has_option("Build", "extension"):
        config.extension = parser.get("Build", "extension").split()

def parseCompression(config, parser):
    pass

def parseExtraction(config, parser):
    if parser.has_option("VideoExtraction", "subsampling_size"):
        config.f = parser.getint("VideoExtraction", "subsampling_size")
        
    if parser.has_option("VideoExtraction", "max_time"):
        config.max_time = parser.getint("VideoExtraction", "max_time")
    
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
        config.distance_treshold = parser.getint("VideoExtraction", "distance_treshold")**2
    
    config.distance_treshold = Grouping3D.normalizeParameter(config.distance_treshold, config)
    
    if parser.has_option("VideoExtraction", "gap_treshold"):
        config.gap_treshold = parser.getint("VideoExtraction", "gap_treshold")**2
    
    config.gap_treshold = Grouping3D.normalizeParameter(config.gap_treshold, config)

    if parser.has_option("VideoExtraction", "line_minimum_frame_range"):
        config.line_minimum_frame_range = parser.getint("VideoExtraction", "line_minimum_frame_range")
    
    if parser.has_option("VideoExtraction", "line_distance_const"):
        config.line_distance_const = parser.getint("VideoExtraction", "line_distance_const")
    
    if parser.has_option("VideoExtraction", "point_ratio_treshold"):
        config.point_ratio_treshold = parser.getfloat("VideoExtraction", "point_ratio_treshold")        
    
    if parser.has_option("VideoExtraction", "max_lines"):
        config.max_lines = parser.getint("VideoExtraction", "max_lines")
    
    if parser.has_option("VideoExtraction", "min_lines"):
        config.max_lines = parser.getint("VideoExtraction", "max_lines")


def parseDetection(config, parser):
    
    if parser.has_option("Detection", "k1_det"):
        config.k1_det = parser.getfloat("Detection", "k1_det")

    if parser.has_option("Detection", "j1"):
        config.j1 = parser.getint("Detection", "j1")

    if parser.has_option("Detection", "max_white_ratio"):
        config.max_white_ratio = parser.getfloat("Detection", "max_white_ratio")

    if parser.has_option("Detection", "time_window_size"):
        config.time_window_size = parser.getint("Detection", "time_window_size")

    if parser.has_option("Detection", "time_slide"):
        config.time_slide = parser.getint("Detection", "time_slide")

    if parser.has_option("Detection", "max_lines_det"):
        config.max_lines_det = parser.getint("Detection", "max_lines_det")

    if parser.has_option("Detection", "line_min_dist"):
        config.line_min_dist = parser.getint("Detection", "line_min_dist")

    if parser.has_option("Detection", "distance_treshold_det"):
        config.distance_treshold_det = parser.getint("Detection", "distance_treshold_det")**2

    config.distance_treshold_det = Grouping3D.normalizeParameter(config.distance_treshold_det, config)

    if parser.has_option("Detection", "gap_treshold_det"):
        config.gap_treshold_det = parser.getint("Detection", "gap_treshold_det")**2

    config.gap_treshold_det = Grouping3D.normalizeParameter(config.gap_treshold_det, config)

    if parser.has_option("Detection", "min_pixels_det"):
        config.min_pixels_det = parser.getint("Detection", "min_pixels_det")

    if parser.has_option("Detection", "line_minimum_frame_range_det"):
        config.line_minimum_frame_range_det = parser.getint("Detection", "line_minimum_frame_range_det")

    if parser.has_option("Detection", "line_distance_const_det"):
        config.line_distance_const_det = parser.getint("Detection", "line_distance_const_det")

    if parser.has_option("Detection", "max_time_det"):
        config.max_time_det = parser.getint("Detection", "max_time_det")

    if parser.has_option("Detection", "stripe_width"):
        config.stripe_width = parser.getint("Detection", "stripe_width")

    if parser.has_option("Detection", "max_points_det"):
        config.max_points_det = parser.getint("Detection", "max_points_det")
    
    if parser.has_option("Detection", "kht_lib_path"):
        config.kht_lib_path = parser.get("Detection", "kht_lib_path")

    if parser.has_option("Detection", "vect_angle_thresh"):
        config.vect_angle_thresh = parser.getint("Detection", "vect_angle_thresh")

    if parser.has_option("Detection", "frame_extension"):
        config.frame_extension = parser.getint("Detection", "frame_extension")

    if parser.has_option("Detection", "centroids_max_deviation"):
        config.centroids_max_deviation = parser.getfloat("Detection", "centroids_max_deviation")

    if parser.has_option("Detection", "centroids_max_distance"):
        config.centroids_max_distance = parser.getint("Detection", "centroids_max_distance")

