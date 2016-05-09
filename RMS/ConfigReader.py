# RPi Meteor Station
# Copyright (C) 2016  Dario Zubovic, Denis Vida
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
        self.fps = 25.0
        self.fov_w = 64.0
        self.fov_h = 48.0
        self.deinterlace_order = 1
        
        self.weaveArgs = ["-O3"]
        
        ##### FireballDetection
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

        ##### MeteorDetection
        
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

        # 3D line finding for meteor detection
        self.max_points_det = 600 # maximumum number of points during 3D line search in faint meteor detection (used to minimize runtime)
        self.distance_treshold_det = 50**2 # maximum distance between the line and the point to be takes as a part of the same line
        self.gap_treshold_det = 500**2 # maximum allowed gap between points
        self.min_pixels_det = 10 # minimum number of pixels in a strip
        self.line_minimum_frame_range_det = 4 # minimum number of frames per one detection
        self.line_distance_const_det = 4 # constant that determines the influence of average point distance on the line quality
        self.max_time_det = 10 # maximum time in seconds for which line finding algorithm can run

        # 3D line merging parameters
        self.vect_angle_thresh = 20 # angle similarity between 2 lines in a stripe to be merged
        self.frame_extension = 3 # how many frames to check during centroiding before and after the initially determined frame range

        # Centroid filtering parameters
        self.centroids_max_deviation = 2 # maximum deviation of a centroid point from a LSQ fitted line (if above max, it will be rejected)
        self.centroids_max_distance =  30 # maximum distance in pixels between centroids (used for filtering spurious centroids)


        ##### StarExtraction

        # Extraction parameters
        self.max_global_intensity = 150 # maximum mean intensity of an image before it is discared as too bright
        self.border = 10 #  apply a mask on the detections by removing all that are too close to the given image border (in pixels)
        self.neighborhood_size = 10 # size of the neighbourhood for the maximum search (in pixels)
        self.intensity_threshold = 5 # a threshold for cutting the detections which are too faint (0-255)

        # PSF fit and filtering
        self.segment_radius = 4 # radius (in pixels) of image segment around the detected star on which to perform the fit
        self.roundness_threshold = 0.5 # minimum ratio of 2D Gaussian sigma X and sigma Y to be taken as a stars (hot pixels are narrow, while stars are round)
        self.max_feature_ratio = 0.8 # maximum ratio between 2 sigma of the star and the image segment area


        ##### Calibration
        self.star_catalog_path = 'Catalogs'
        self.star_catalog_file = 'BSC5'

        self.platepar_name = 'platepar_cmn2010.cal'

        self.catalog_extraction_radius = 40.0 #deg
        self.catalog_mag_limit = 4.5

        self.calstars_files_N = 100 # How many calstars FF files to evaluate

        self.calstars_min_stars = 500 # Minimum number of stars to use

        self.stars_NN_radius = 10.0 # deg
        self.refinement_star_NN_radius = 0.125 #deg
        self.rotation_param_range = 5.0 # deg
        self.min_matched_stars = 7
        self.max_initial_iterations = 20

        self.min_estimation_value = 0.4 # Minimum estimation parameter when to stop the iteration


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
    
    if parser.has_section("FireballDetection"):
        parseFireballDetection(config, parser)

    if parser.has_section("MeteorDetection"):
        parseMeteorDetection(config, parser)

    if parser.has_section("StarExtraction"):
        parseStarExtraction(config, parser)

    if parser.has_section("Calibration"):
        parseCalibration(config, parser)
    
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

    if parser.has_option("Capture", "fps"):
        config.fps = parser.getfloat("Capture", "fps")

    if parser.has_option("Capture", "fov_w"):
        config.fov_w = parser.getfloat("Capture", "fov_w")

    if parser.has_option("Capture", "fov_h"):
        config.fov_h = parser.getfloat("Capture", "fov_h")

    if parser.has_option("Capture", "deinterlace_order"):
        config.deinterlace_order = parser.getint("Capture", "deinterlace_order")

def parseBuildArgs(config, parser):
    if parser.has_option("Build", "weave"):
        config.weaveArgs = parser.get("Build", "weave").split()
        
    if parser.has_option("Build", "extension"):
        config.extension = parser.get("Build", "extension").split()

def parseCompression(config, parser):
    pass

def parseFireballDetection(config, parser):
    if parser.has_option("FireballDetection", "subsampling_size"):
        config.f = parser.getint("FireballDetection", "subsampling_size")
        
    if parser.has_option("FireballDetection", "max_time"):
        config.max_time = parser.getint("FireballDetection", "max_time")
    
    if parser.has_option("FireballDetection", "minimal_level"):
        config.min_level = parser.getint("FireballDetection", "minimal_level")
    
    if parser.has_option("FireballDetection", "minimum_pixels"):
        config.min_pixels = parser.getint("FireballDetection", "minimum_pixels")
    
    if parser.has_option("FireballDetection", "k1"):
        config.k1 = parser.getint("FireballDetection", "k1")
    
    if parser.has_option("FireballDetection", "max_points_per_frame"):
        config.max_points_per_frame = parser.getint("FireballDetection", "max_points_per_frame")
    
    if parser.has_option("FireballDetection", "max_per_frame_factor"):
        config.max_per_frame_factor = parser.getint("FireballDetection", "max_per_frame_factor")
    
    if parser.has_option("FireballDetection", "max_points"):
        config.max_points = parser.getint("FireballDetection", "max_points")
    
    if parser.has_option("FireballDetection", "min_frames"):
        config.min_frames = parser.getint("FireballDetection", "min_frames")
    
    if parser.has_option("FireballDetection", "min_points"):
        config.min_points = parser.getint("FireballDetection", "min_points")
    
    if parser.has_option("FireballDetection", "extend_before"):
        config.before = parser.getfloat("FireballDetection", "extend_before")
    
    if parser.has_option("FireballDetection", "extend_after"):
        config.after = parser.getfloat("FireballDetection", "extend_after")
    
    if parser.has_option("FireballDetection", "min_window_size"):
        config.minSize = parser.getint("FireballDetection", "min_window_size")
    
    if parser.has_option("FireballDetection", "max_window_size"):
        config.maxSize = parser.getint("FireballDetection", "max_window_size")
    
    if parser.has_option("FireballDetection", "threshold_for_size"):
        config.limitForSize = parser.getfloat("FireballDetection", "threshold_for_size")
    
    if parser.has_option("FireballDetection", "distance_treshold"):
        config.distance_treshold = parser.getint("FireballDetection", "distance_treshold")**2
    
    config.distance_treshold = Grouping3D.normalizeParameter(config.distance_treshold, config)
    
    if parser.has_option("FireballDetection", "gap_treshold"):
        config.gap_treshold = parser.getint("FireballDetection", "gap_treshold")**2
    
    config.gap_treshold = Grouping3D.normalizeParameter(config.gap_treshold, config)

    if parser.has_option("FireballDetection", "line_minimum_frame_range"):
        config.line_minimum_frame_range = parser.getint("FireballDetection", "line_minimum_frame_range")
    
    if parser.has_option("FireballDetection", "line_distance_const"):
        config.line_distance_const = parser.getint("FireballDetection", "line_distance_const")
    
    if parser.has_option("FireballDetection", "point_ratio_treshold"):
        config.point_ratio_treshold = parser.getfloat("FireballDetection", "point_ratio_treshold")        
    
    if parser.has_option("FireballDetection", "max_lines"):
        config.max_lines = parser.getint("FireballDetection", "max_lines")
    
    if parser.has_option("FireballDetection", "min_lines"):
        config.max_lines = parser.getint("FireballDetection", "max_lines")


def parseMeteorDetection(config, parser):
    
    if parser.has_option("MeteorDetection", "k1_det"):
        config.k1_det = parser.getfloat("MeteorDetection", "k1_det")

    if parser.has_option("MeteorDetection", "j1"):
        config.j1 = parser.getint("MeteorDetection", "j1")

    if parser.has_option("MeteorDetection", "max_white_ratio"):
        config.max_white_ratio = parser.getfloat("MeteorDetection", "max_white_ratio")

    if parser.has_option("MeteorDetection", "time_window_size"):
        config.time_window_size = parser.getint("MeteorDetection", "time_window_size")

    if parser.has_option("MeteorDetection", "time_slide"):
        config.time_slide = parser.getint("MeteorDetection", "time_slide")

    if parser.has_option("MeteorDetection", "max_lines_det"):
        config.max_lines_det = parser.getint("MeteorDetection", "max_lines_det")

    if parser.has_option("MeteorDetection", "line_min_dist"):
        config.line_min_dist = parser.getint("MeteorDetection", "line_min_dist")

    if parser.has_option("MeteorDetection", "distance_treshold_det"):
        config.distance_treshold_det = parser.getint("MeteorDetection", "distance_treshold_det")**2

    config.distance_treshold_det = Grouping3D.normalizeParameter(config.distance_treshold_det, config)

    if parser.has_option("MeteorDetection", "gap_treshold_det"):
        config.gap_treshold_det = parser.getint("MeteorDetection", "gap_treshold_det")**2

    config.gap_treshold_det = Grouping3D.normalizeParameter(config.gap_treshold_det, config)

    if parser.has_option("MeteorDetection", "min_pixels_det"):
        config.min_pixels_det = parser.getint("MeteorDetection", "min_pixels_det")

    if parser.has_option("MeteorDetection", "line_minimum_frame_range_det"):
        config.line_minimum_frame_range_det = parser.getint("MeteorDetection", "line_minimum_frame_range_det")

    if parser.has_option("MeteorDetection", "line_distance_const_det"):
        config.line_distance_const_det = parser.getint("MeteorDetection", "line_distance_const_det")

    if parser.has_option("MeteorDetection", "max_time_det"):
        config.max_time_det = parser.getint("MeteorDetection", "max_time_det")

    if parser.has_option("MeteorDetection", "stripe_width"):
        config.stripe_width = parser.getint("MeteorDetection", "stripe_width")

    if parser.has_option("MeteorDetection", "max_points_det"):
        config.max_points_det = parser.getint("MeteorDetection", "max_points_det")
    
    if parser.has_option("MeteorDetection", "kht_lib_path"):
        config.kht_lib_path = parser.get("MeteorDetection", "kht_lib_path")

    if parser.has_option("MeteorDetection", "vect_angle_thresh"):
        config.vect_angle_thresh = parser.getint("MeteorDetection", "vect_angle_thresh")

    if parser.has_option("MeteorDetection", "frame_extension"):
        config.frame_extension = parser.getint("MeteorDetection", "frame_extension")

    if parser.has_option("MeteorDetection", "centroids_max_deviation"):
        config.centroids_max_deviation = parser.getfloat("MeteorDetection", "centroids_max_deviation")

    if parser.has_option("MeteorDetection", "centroids_max_distance"):
        config.centroids_max_distance = parser.getint("MeteorDetection", "centroids_max_distance")


def parseStarExtraction(config, parser):

    if parser.has_option("StarExtraction", "max_global_intensity"):
        config.max_global_intensity = parser.getint("StarExtraction", "max_global_intensity")

    if parser.has_option("StarExtraction", "border"):
        config.border = parser.getint("StarExtraction", "border")

    if parser.has_option("StarExtraction", "neighborhood_size"):
        config.neighborhood_size = parser.getint("StarExtraction", "neighborhood_size")

    if parser.has_option("StarExtraction", "intensity_threshold"):
        config.intensity_threshold = parser.getint("StarExtraction", "intensity_threshold")

    if parser.has_option("StarExtraction", "segment_radius"):
        config.segment_radius = parser.getint("StarExtraction", "segment_radius")

    if parser.has_option("StarExtraction", "roundness_threshold"):
        config.roundness_threshold = parser.getfloat("StarExtraction", "roundness_threshold")

    if parser.has_option("StarExtraction", "max_feature_ratio"):
        config.max_feature_ratio = parser.getfloat("StarExtraction", "max_feature_ratio")


def parseCalibration(config, parser):

    if parser.has_option("Calibration", "star_catalog_path"):
        config.star_catalog_path = parser.get("Calibration", "star_catalog_path")

    if parser.has_option("Calibration", "star_catalog_file"):
        config.star_catalog_file = parser.get("Calibration", "star_catalog_file")

    if parser.has_option("Calibration", "platepar_name"):
        config.platepar_name = parser.get("Calibration", "platepar_name")

    if parser.has_option("Calibration", "catalog_extraction_radius"):
        config.catalog_extraction_radius = parser.getfloat("Calibration", "catalog_extraction_radius")

    if parser.has_option("Calibration", "catalog_mag_limit"):
        config.catalog_mag_limit = parser.getfloat("Calibration", "catalog_mag_limit")

    if parser.has_option("Calibration", "calstars_files_N"):
        config.calstars_files_N = parser.getint("Calibration", "calstars_files_N")

    if parser.has_option("Calibration", "calstars_min_stars"):
        config.calstars_min_stars = parser.getint("Calibration", "calstars_min_stars")

    if parser.has_option("Calibration", "stars_NN_radius"):
        config.stars_NN_radius = parser.getfloat("Calibration", "stars_NN_radius")

    if parser.has_option("Calibration", "refinement_star_NN_radius"):
        config.refinement_star_NN_radius = parser.getfloat("Calibration", "refinement_star_NN_radius")

    if parser.has_option("Calibration", "rotation_param_range"):
        config.rotation_param_range = parser.getfloat("Calibration", "rotation_param_range")

    if parser.has_option("Calibration", "min_matched_stars"):
        config.min_matched_stars = parser.getint("Calibration", "min_matched_stars")

    if parser.has_option("Calibration", "max_initial_iterations"):
        config.max_initial_iterations = parser.getint("Calibration", "max_initial_iterations")

    if parser.has_option("Calibration", "min_estimation_value"):
        config.min_estimation_value = parser.getfloat("Calibration", "min_estimation_value")