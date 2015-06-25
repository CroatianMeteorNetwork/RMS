from multiprocessing import Process
import numpy as np
from scipy import weave, stats
import cv2
from RMS.Compression import Compression
from math import floor, sqrt
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class Extractor(Process):
    factor = 0
    
    def __init__(self):
        super(Extractor, self).__init__()
    
    def findPoints(self, frames, compressed, min_level=40, min_points=24, k1=1.8, max_per_frame_factor=20, f=16):
        """Treshold and subsample frames then extract points.
        
        @param frames: numpy array, for example (256, 576, 720), with all frames
        @param average: average frame (or median)
        @param stddev: standard deviation frame
        @return: (y, x, z) of found points
        """
     
        count = np.zeros((frames.shape[0], floor(frames.shape[1]//f), floor(frames.shape[2]//f)), np.int16)
        pointsy = np.empty((frames.shape[0]*floor(frames.shape[1]//f)*floor(frames.shape[2]//f)), np.uint16)
        pointsx = np.empty((frames.shape[0]*floor(frames.shape[1]//f)*floor(frames.shape[2]//f)), np.uint16)
        pointsz = np.empty((frames.shape[0]*floor(frames.shape[1]//f)*floor(frames.shape[2]//f)), np.uint16)
        
        code = """
        unsigned int x, y, x2, y2, n, i, max;
        unsigned int num = 0, acc = 0;
        
        unsigned int counter[Nframes[0]] = {0}; // counts threshold passers per frame
        
        for(y=0; y<Nframes[1]; y++) {
            for(x=0; x<Nframes[2]; x++) {
                max = COMPRESSED3(0, y, x);
                if((max > min_level) && (max >= COMPRESSED3(2, y, x) + k1 * COMPRESSED3(3, y, x))) {
                    n = COMPRESSED3(1, y, x);
                    
                    y2 = y/f; // subsample frame in f*f squares
                    x2 = x/f;
                    
                    if(COUNT3(n, y2, x2) >= min_points) { // check if there is enough of threshold passers inside of this square
                        POINTSY1(num) = y2;
                        POINTSX1(num) = x2;
                        POINTSZ1(num) = n;
                        counter[n]++;
                        num++;
                        COUNT3(n, y2, x2) = -1; //don't repeat this number
                    } else if(COUNT3(n, y2, x2) != -1) { // increase counter if not enough threshold passers and this number isn't written already
                        COUNT3(n, y2, x2) += 1;
                    }
                }
            }    
        }
        
        for(n=0; n<Nframes[0]; n++) {
            acc += counter[n];
            
            if(counter[n] > max_per_frame_factor * acc / (n+1)) {
                // flare detected, filter it out
                acc -= counter[n];
                
                for(i=num-1; i-- >0; ) {
                    if(POINTSZ1(i) == n) {
                        // remove this point by replacing it with last point in array and lowering num variable
                        num--;
                        POINTSY1(i) = POINTSY1(num);
                        POINTSX1(i) = POINTSX1(num);
                        POINTSZ1(i) = POINTSZ1(num);
                    }
                }
            }
        }
        
        return_val = num; // output length of POINTS arrays
        """
        
        length = weave.inline(code, ['frames', 'compressed', 'min_level', 'min_points', 'k1', 'max_per_frame_factor', 'f', 'count', 'pointsy', 'pointsx', 'pointsz'])
        
        # cut away extra long array
        y = pointsy[0 : length]
        x = pointsx[0 : length]
        z = pointsz[0 : length]
        
        # sort by frame number
        indices = np.argsort(z) # quicksort
        y = y[indices].astype(np.float)
        x = x[indices].astype(np.float)
        z = z[indices].astype(np.float)
        
        return y, x, z
    
    def testPoints(self, points, min_points=5, gap_treshold=70):
        """ Test if points are interesting (ie. something is detected).
        
        @return: true if video should be further checked for meteors, false otherwise
        """
        
        y, x, z = points
        
        # check if there is enough points
        if(len(y) < min_points):
            return False
        
        # check how many points are close to each other (along the time line)
        code = """
        unsigned int distance, i, count = 0,
        y_dist, x_dist, z_dist,
        y_prev = 0, x_prev = 0, z_prev = 0;
        
        for(i=1; i<Ny[0]; i++) {
            y_dist = Y1(i) - y_prev;
            x_dist = X1(i) - x_prev;
            z_dist = Z1(i) - z_prev;
            
            distance = sqrt(y_dist*y_dist + z_dist*z_dist + z_dist*z_dist);
            
            if(distance < gap_treshold) {
                count++;
            }
            
            y_prev = Y1(i);
            x_prev = X1(i);
            z_prev = Z1(i);
        }
        
        return_val = count;
        """
        
        count = weave.inline(code, ['gap_treshold', 'y', 'x', 'z'])
        
        return count >= min_points

    def normalizeParameter(self, param):
        """ Normalize detection parameter to be size independent.
        
        @param param: parameter to be normalized
        @return: normalized parameter
        """
    
        return param * self.factor
    
    ###TODO
    def extract(self, frames, coefficients, before=3, after=7, f=16, limitForSize=0.85, minSize=80, maxSize=192):
        clips = []
        
        for coeff in coefficients:
            point, slopeXZ, slopeYZ, firstFrame, lastFrame = coeff
            slopeXZ = float(slopeXZ)
            slopeYZ = float(slopeYZ)
            firstFrame = int(firstFrame)
            lastFrame = int(lastFrame)
            
            out = np.zeros((frames.shape[0], maxSize, maxSize), np.uint8)
            sizepos = np.empty((frames.shape[0], 4), np.uint16) # y, x, size
            
            code = """
                int x_m, x_p, x_t, y_m, y_p, y_t, k,
                first_frame = firstFrame - before,
                last_frame = lastFrame + after,
                half_max_size = maxSize / 2,
                half_f = f / 2;
                unsigned int x, y, i, x2, y2, num = 0,
                max, pixel, limit, max_width, max_height, size, half_size, num_equal;
                
                if(first_frame < 0) {
                    first_frame = 0;
                }
                if(last_frame >= Nframes[0]) {
                    last_frame = Nframes[0] - 1;
                }
                
                for(i = first_frame; i < last_frame; i++) {
                    // calculate point at current time
                    k = i - firstFrame;
                    y_t = (POINT1(0) + slopeYZ * k) * f + half_f;
                    x_t = (POINT1(1) + slopeXZ * k) * f + half_f;
                    
                    if(y_t < 0 || x_t < 0 || y_t >= Nframes[1] || x_t >= Nframes[2]) {
                        // skip if out of bounds
                        continue;
                    }
                    
                    SIZEPOS2(num, 0) = y_t; 
                    SIZEPOS2(num, 1) = x_t; 
                    SIZEPOS2(num, 2) = i;
                    
                    // calculate boundaries for finding max value
                    y_m = y_t - half_f, y_p = y_t + half_f, 
                    x_m = x_t - half_f, x_p = x_t + half_f;
                    if(y_m < 0) {
                        y_m = 0;
                    }
                    if(x_m < 0) {
                        x_m = 0;
                    }
                    if(y_p >= Nframes[1]) {
                        y_p = Nframes[1] - 1;
                    }
                    if(x_p >= Nframes[2]) {
                        x_p = Nframes[2] - 1;
                    }
                    
                    // find max value
                    max = 0;
                    for(y=y_m; y<y_p; y++) {
                        for(x=x_m; x<x_p; x++) {
                            pixel = FRAMES3(i, y, x);
                            if(pixel > max) {
                                max = pixel;
                            }
                        }
                    }
                    
                    // calculate boundaries for finding size
                    y_m = y_t - half_max_size, y_p = y_t + half_max_size, 
                    x_m = x_t - half_max_size, x_p = x_t + half_max_size;
                    if(y_m < 0) {
                        y_m = 0;
                    }
                    if(x_m < 0) {
                        x_m = 0;
                    }
                    if(y_p >= Nframes[1]) {
                        y_p = Nframes[1] - 1;
                    }
                    if(x_p >= Nframes[2]) {
                        x_p = Nframes[2] - 1;
                    }
                    
                    // calculate mean distance from center
                    max_width = 0, max_height = 0, num_equal = 0,
                    limit = limitForSize * max;
                    for(y=y_m; y<y_p; y++) {
                        for(x=x_m; x<x_p; x++) {
                            if(FRAMES3(i, y, x) > limit) {
                                max_height += abs(y_t - y);
                                max_width += abs(x_t - x);
                                num_equal++;
                            }
                        }
                    }
                    
                    // calculate size
                    if(max_height > max_width) {
                        size = max_height / num_equal;
                    } else {
                        size = max_width / num_equal;
                    }
                    if(size < minSize) {
                        size = minSize;
                    }
                    SIZEPOS2(num, 3) = size;
                    half_size = size / 2;
                    
                    // calculate boundaries for frame extraction (cropping)
                    y_m = y_t - half_size, y_p = y_t + half_size, 
                    x_m = x_t - half_size, x_p = x_t + half_size;
                    if(y_m < 0) {
                        y_m = 0;
                    }
                    if(x_m < 0) {
                        x_m = 0;
                    }
                    if(y_p >= Nframes[1]) {
                        y_p = Nframes[1] - 1;
                    }
                    if(x_p >= Nframes[2]) {
                        x_p = Nframes[2] - 1;
                    }
                    
                    // crop part of frame
                    y2 = 0, x2 = 0;
                    for(y=y_m; y<y_p; y++) {
                        y2++;
                        x2 = 0;
                        for(x=x_m; x<x_p; x++) {
                            OUT3(num, y2, x2) = FRAMES3(i, y, x);
                            x2++;
                        }
                    }
                    
                    num++;
                }
                
                return_val = num;                
            """
            
            length = weave.inline(code, ['frames', 'point', 'slopeXZ', 'slopeYZ', 'firstFrame', 'lastFrame', 'before', 'after', 'f', 'limitForSize', 'minSize', 'maxSize', 'sizepos', 'out'])
            
            out = out[:length]
            sizepos = sizepos[:length]
            
            clips.append([out, sizepos])
        
        return clips
    
    ###TODO
    def save(self, position, size, extracted, fileName):
        file = "FR" + fileName + ".bin"
        
        with open(file, "wb") as f:
            f.write(struct.pack('I', extracted.shape[0]))               # frames num
            
            for n in range(extracted.shape[0]):
                f.write(struct.pack('I', position[n, 0]))               # y of center
                f.write(struct.pack('I', position[n, 1]))               # x of center
                f.write(struct.pack('I', size[n]))          
                np.resize(extracted[n], (size[n], size[n])).tofile(f)   # frame

#################

def line3DDistance_simple(x1, y1, z1, x2, y2, z2, x0, y0, z0):
    """ Distance from line to a point in 3D using simple operations. """

    # Original function:
    # np.linalg.norm(np.cross((point0 - point1), (point0 - point2))) / np.linalg.norm(point2 - point1)

    # Length of vector in the numerator
    dx1 = x0 - x1
    dy1 = y0 - y1
    dz1 = z0 - z1

    dx2 = x0 - x2
    dy2 = y0 - y2
    dz2 = z0 - z2

    n_len = sqrt((dx1*dy2 - dx2*dy1)**2+(dx2*dz1 - dx1*dz2)**2 + (dy1*dz2 - dy2*dz1)**2)

    # Length of denominator vector
    d_len = point3DDistance(x1, y1, z1, x2, y2, z2)

    return n_len / d_len

def point3DDistance(x1, y1, z1, x2, y2, z2):
    """ Distance from two points in 3D space. """

    return sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def getAllPoints(point_list, x1, y1, z1, x2, y2, z2):
    """ Returns all points describing a particular line. """

    def propagateLine(propagation_list):
        """ Finds all points present on a line starting from a point on that line. """

        x_prev, y_prev, z_prev = x1, y1, z1

        for point3 in propagation_list:

            y3, x3, z3 = point3

            line_dist = line3DDistance_simple(x1, y1, z1, x2, y2, z2, x3, y3, z3)

            if line_dist < distance_treshold:

                # Calculate the gap from the previous point and reject the solution if the point is too far
                if point3DDistance(x_prev, y_prev, z_prev, x3, y3, z3) > gap_treshold:

                    if point3DDistance(x_prev, y_prev, z_prev, x2, y2, z2) > gap_treshold:
                        counter = 0

                    break

                x_prev, y_prev, z_prev = x3, y3, z3

                line_points.append(point3)


    point1 = [y1, x1, z1]
    point2 = [y2, x2, z2]

    line_points = []

    # Spread point cloud forward
    point1_index = point_list.index(point1)

    propagateLine(point_list[point1_index:])

    # Spread point cloud backwards

    propagateLine(reversed(point_list[:point1_index]))

    return line_points


def find3DLines(point_list):
    """ Iteratively finds N straight lines in 3D space. """

    results_list = []
    for i, point1 in enumerate(point_list):
        for point2 in point_list[i:]:

            if point1 == point2:
                continue

            x1, y1, z1 = point1
            x2, y2, z2 = point2

            # Include 2 points that make the line in the count
            counter = 2

            # Track average distance from the line
            line_dist_sum = 0

            x_prev, y_prev, z_prev = x1, y1, z1

            for point3 in point_list:

                if point1 == point3:
                    continue

                x3, y3, z3 = point3

                # Check if the distance between the line and the point is close enough
                
                line_dist = line3DDistance_simple(x1, y1, z1, x2, y2, z2, x3, y3, z3)

                if line_dist < distance_treshold:

                    # Calculate the gap from the previous point and reject the solution if the point is too far
                    if point3DDistance(x_prev, y_prev, z_prev, x3, y3, z3) > gap_treshold:

                        # Reject solution (reset counter) if the last point is too far
                        if point3DDistance(x2, y2, z2, x_prev, y_prev, z_prev) > gap_treshold:
                            counter = 0

                        break

                    counter += 1
                    line_dist_sum += line_dist

                    x_prev, y_prev, z_prev = x3, y3, z3


            # Skip if too little points were found
            if (counter-2) < minimum_points:
                continue

            # Average distance between points and the line
            line_dist_avg = line_dist_sum / float(counter - 2)

            # calculate a parameter for line quality
            # larger average distance = less quality
            line_quality = counter - line_distance_const * line_dist_avg
            results_list.append((point1, point2, counter, line_quality))

    # Return empty if no good match was found
    if not results_list:
        return None

    # Line with the best quality    
    max_line = max(results_list, key=lambda x: x[3])
    print max_line

    line_list.append([max_line])

    point_num = len(point_list)

    # ratio of points inside and and all points
    line_ratio = max_line[2] / float(point_num)
    print line_ratio
    print 'Remaining points: ', point_num

    # if there are more lines on the image
    if line_ratio < point_ratio_treshold and point_num > 10:
        point_list = remove3DPoints(point_list, max_line[0], max_line[1])

        # Recursively find lines until the condition is met
        find3DLines(point_list)

    return line_list

def remove3DPoints(point_list, point1, point2):
    """ Removes points from the point list that belong to the given line.

    Dominant line is given as coordinates of point1 and point2."""

    x1, y1, z1 = point1
    x2, y2, z2 = point2

    line_points = getAllPoints(point_list, y1, x1, z1, y2, x2, z2)

    return [x for x in point_list if x not in line_points]

def normalizeParameter(param, y_dim, x_dim):
    """ Normalizes detection parameters to be size independant. """

    return param * sqrt(y_dim * x_dim) / sqrt(720*576)

def checkPointsDistance(point_list):
    """ Check quickly how many points are close to each other (along the time line) to determine if there are any meteors on the image. 

    Returns True of False, whether the image should be further checked for meteors. """
    
    # Check for minimum amout of points on the image
    if len(point_list) >= minimum_img_points:

        close_points_count = 0

        # Check distances between points
        y_prev, x_prev, z_prev = point_list[0]
        for y, x, z in point_list[1:]:
            if point3DDistance(y, x, z, y_prev, x_prev, z_prev) < gap_treshold:
                close_points_count += 1
            y_prev, x_prev, z_prev = y, x, z

        print 'Close points: ', close_points_count

        # Check minimum close point count
        if close_points_count > minimum_points:
            return True
        

    return False

def findCoefficients(event_points, line_list, before=5, after=15):
    coeff = []
    
    for i, detected_line in enumerate(line_list):
        detected_line = detected_line[0]
    
        # get detected points
        x1, x2 = detected_line[0][1], detected_line[1][1]
        y1, y2 = detected_line[0][0], detected_line[1][0]
        z1, z2 = detected_line[0][2], detected_line[1][2]
        detected_points = getAllPoints(event_points, x1, y1, z1, x2, y2, z2)
        
        # sort by frame number
        detected_points = sorted(detected_points, key=lambda x:x[2])
        
        point1 = np.array(detected_points[0])
        point2 = np.array(detected_points[-1])
        point3 = point2 - point1
        
        slope1 = point3[1]/point3[2]
        slope2 = point3[0]/point3[2]
        
        coeff.append([point1, slope1, slope2, point1[2], point2[2]])
        #first point, slope of XZ, slope of YZ, first frame, last frame
        
    return coeff
#################

if __name__ ==  "__main__":
    cap = cv2.VideoCapture("/home/dario/Videos/m20050320_012752.wmv")
    
    frames = np.empty((200, 480, 640), np.uint8)
    for i in range(200):
        ret, frame = cap.read()
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        frames[i] = gray
        
    cap.release()
    
    comp = Compression(None, None, None, None, 0)
    compressed = comp.compress(frames)
    
    extractor = Extractor()
    
    extractor.factor = sqrt(frames.shape[1]*frames.shape[2]) / sqrt(720*576) # calculate normalization factor
    
    bigT = t = time.time()
    
    points = extractor.findPoints(frames, compressed)
    print points[0].shape
    print "time for thresholding and subsampling: ", time.time() - t
    t = time.time()
    
    should_continue = extractor.testPoints(points)
    
    if not should_continue:
        print "nothing found, not extracting anything"
        
    print "time for test: ", time.time() - t
    
    y_dim = frames.shape[1]/16
    x_dim = frames.shape[2]/16
    
    event_points = []
    for i in range(len(points[0])):
        event_points.append([points[0][i], points[1][i], points[2][i]])
    print "points per frame:", stats.itemfreq(points[2])
    
    print "poinnnt", event_points

    print len(event_points), 'points'


    ############################
    # Define parameters

    # Minimum points on the image to even consider processing it further
    minimum_img_points = 6

    # Max distance between the line and the point to be takes as a part of the same line
    distance_treshold = 60 #px
    distance_treshold = normalizeParameter(distance_treshold, y_dim, x_dim)

    # Constant that determines the influence of average point distance on the line quality
    # Larger constant "punishes" the quality more with larger average distance
    line_distance_const = 3

    # Maximum gap between consecutive points allowed
    gap_treshold = 100
    gap_treshold = normalizeParameter(gap_treshold, y_dim, x_dim)

    # Minimum points required to form a line
    minimum_points = 4

    # Ratio of how many points must be close to the line before considering searching for another line
    point_ratio_treshold = 0.9

    ###########################
    line_list = []

    # Find lines in 3D space and store them to line_list
    time1 = time.clock()
    find3DLines(list(event_points))

    print 'Time for finding lines:', time.clock() - time1

    t = time.time()
    coeff = findCoefficients(event_points, line_list)
    print "Time for calculating coefficients", time.time() - t
    
    t = time.time()
    ex = extractor.extract(frames, coeff)
    print "Time for extracting frames", time.time() - t
    
    

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    points = np.array(event_points)
    xs = points[:,1]
    ys = points[:,0]
    zs = points[:,2]
    # Plot points in 3D
    ax.scatter(xs, ys, zs, s = 5)
    # Define line colors to use
    ln_colors = ['r', 'g', 'y', 'k', 'm', 'c']
    # Plot detected lines in 3D
    for i, detected_line in enumerate(line_list):
        detected_line = detected_line[0]
        xs = [detected_line[0][1], detected_line[1][1]]
        ys = [detected_line[0][0], detected_line[1][0]]
        zs = [detected_line[0][2], detected_line[1][2]]
        ax.plot(xs, ys, zs, c = ln_colors[i%6])

    # Plot grouped points
    
    for i, detected_line in enumerate(line_list):
        detected_line = detected_line[0]

        x1, x2 = detected_line[0][1], detected_line[1][1]

        y1, y2 = detected_line[0][0], detected_line[1][0]

        z1, z2 = detected_line[0][2], detected_line[1][2]

        detected_points = getAllPoints(event_points, x1, y1, z1, x2, y2, z2)

        if not detected_points:
            continue

        detected_points = np.array(detected_points)

        xs = detected_points[:,1]
        ys = detected_points[:,0]
        zs = detected_points[:,2]

        ax.scatter(xs, ys, zs, c = ln_colors[i%6], s = 40)

    # Set axes limits
    ax.set_zlim(0, 255)
    plt.xlim([0,x_dim])
    plt.ylim([0,y_dim])
    plt.show()
    
    print "extracted", ex
    
    background = compressed[0]
    
    for clip in ex:
        out = clip[0]
        posSize = clip[1]
        
        for i, crop in enumerate(out):
            y_pos = posSize[i, 0]
            x_pos = posSize[i, 1]
            z_pos = posSize[i, 2]
            size = posSize[i, 3]
            
            y2 = 0
            for y in range(y_pos - size/2, y_pos + size/2):
                y2 += 1
                x2 = 0
                for x in range(x_pos - size/2, x_pos + size/2):
                    pixel = crop[y2, x2]
                    if pixel > 0:
                        background[y, x] = pixel
                    x2 += 1
                
            
            cv2.imshow("bla", background)
            cv2.waitKey(150)