"""
Refactored Sequential RANSAC for line detection.
Optimized for speed and robustness on sparse, intermittent data.
"""

import numpy as np
import logging
log = logging.getLogger("logger")

def getPolarLine(x1, y1, x2, y2, img_w, img_h):
    """ 
    Calculate polar line coordinates (rho, theta) given 2 points.
    Coordinates are CENTER-BASED (origin at image center), consistent with
    getStripeIndices, plotLines, and the rest of the RMS pipeline.
    
    Arguments:
        x1, y1, x2, y2: Point coordinates in image (top-left origin) coords.
        img_w, img_h: Image dimensions, used to compute the center.
        
    Return:
        rho: Perpendicular distance from image center to line.
        theta_deg: Angle of the normal vector in degrees.
    """
    if abs(x1 - x2) < 1e-9 and abs(y1 - y2) < 1e-9:
        return 0.0, 0.0

    # Center the coordinates
    cx = img_w/2.0
    cy = img_h/2.0
    x1c = x1 - cx
    y1c = y1 - cy
    x2c = x2 - cx
    y2c = y2 - cy

    # Line direction vector
    dx = x2c - x1c
    dy = y2c - y1c
    
    # Normal vector (-dy, dx) gives the direction of rho
    nx = -dy
    ny = dx
    
    # Normalize
    norm = np.sqrt(nx*nx + ny*ny)
    if norm == 0: return 0.0, 0.0
    nx /= norm
    ny /= norm
    
    # Rho is the dot product of any centered point on the line with the normal
    rho = x1c*nx + y1c*ny
    theta = np.arctan2(ny, nx)

    # Standardize rho >= 0
    if rho < 0:
        rho = -rho
        theta += np.pi
    
    return rho, np.degrees(theta)


def fitLine(x, y, img_w, img_h, weights=None):
    """ 
    Fit a line to a set of points using Weighted PCA (Total Least Squares).
    Coordinates are CENTER-BASED (origin at image center).
    
    Arguments:
        x, y: [ndarray] Coordinates in image (top-left origin) coords.
        img_w, img_h: Image dimensions, used to compute the center.
        weights: [ndarray] Optional weights for each point.
        
    Return:
        (rho, theta_deg): Fitted line parameters in center-based coords.
    """
    # Center the coordinates
    cx = img_w/2.0
    cy = img_h/2.0
    x_centered = x - cx
    y_centered = y - cy
    pts = np.column_stack((x_centered, y_centered))
    
    if len(pts) < 2:
        return 0.0, 0.0

    # Weighted Mean
    if weights is None:
        mean = np.mean(pts, axis=0)
        centered = pts - mean
        # Covariance matrix (2x2)
        cov = np.dot(centered.T, centered)
    else:
        w_sum = np.sum(weights)
        if w_sum == 0: return 0.0, 0.0
        
        # Weighted mean
        mean = np.average(pts, axis=0, weights=weights)
        centered = pts - mean
        
        # Weighted covariance
        weighted_centered = centered*weights[:, np.newaxis]
        cov = np.dot(weighted_centered.T, centered)

    # Eigen decomposition to find the normal vector (smallest eigenvalue)
    vals, vecs = np.linalg.eigh(cov)
    
    # The normal is the eigenvector corresponding to the smallest eigenvalue
    nx, ny = vecs[:, 0]
    
    rho = mean[0]*nx + mean[1]*ny
    theta = np.arctan2(ny, nx)

    if rho < 0:
        rho = -rho
        theta += np.pi
        
    return rho, np.degrees(theta)


def _lineDistances(points, cx, cy, rho, theta_deg):
    """ Perpendicular distances of points from a line, and their positions along it (center-based coords).

    Arguments:
        points: [ndarray] Nx2 array of (x, y) image coordinates.
        cx, cy: [float] Image center.
        rho, theta_deg: [float] Line parameters.

    Return:
        dists, t: [tuple of ndarrays] Distances from the line and positions along it.
    """

    theta = np.radians(theta_deg)
    ct, st = np.cos(theta), np.sin(theta)
    x_c = points[:, 0] - cx
    y_c = points[:, 1] - cy

    return np.abs(x_c*ct + y_c*st - rho), -x_c*st + y_c*ct


def _bestSegment(points, cx, cy, rho, theta_deg, distance_thresh, max_gap):
    """ Split the inliers of a line at gaps larger than max_gap and return the segment with the most points.

    Arguments:
        points: [ndarray] Nx2 array of (x, y) image coordinates.
        cx, cy: [float] Image center.
        rho, theta_deg: [float] Line parameters.
        distance_thresh: [float] Maximum distance of an inlier from the line (px).
        max_gap: [float] Maximum gap along the line within a segment (px).

    Return:
        (segment, score): [tuple] Indices of the points of the segment, sorted along the line (empty if there
            are no inliers), and its score: the sum of the weights 1 - d/distance_thresh of its points, where d
            is their distance from the line.
    """

    dists, t = _lineDistances(points, cx, cy, rho, theta_deg)
    inliers = np.where(dists < distance_thresh)[0]
    if len(inliers) == 0:
        return inliers, 0.0

    inliers = inliers[np.argsort(t[inliers])]
    splits = np.where(np.diff(t[inliers]) > max_gap)[0] + 1
    segments = np.split(inliers, splits)
    segment = max(segments, key=len)

    return segment, float(np.sum(1.0 - dists[segment]/distance_thresh))


def findLines(img, max_lines, min_pixels, distance_thresh, min_line_length, max_gap, max_iterations=1000,
    debug=False):
    """ Find line segments in the image using Sequential RANSAC with gap bridging.

    A line is hypothesized through two points, the second one drawn from the neighbourhood of the first one,
    so that both are likely on the same track even when most points are noise. The inliers of the line are
    split at gaps and the segment with the most points is scored. The best segment is refined by fitting the
    line to its points and taking the inliers of the fitted line again, and its points are removed before
    searching for the next line. Scoring the points (not the segment length) keeps sparse noise points which
    happen to extend the segment from deciding the line. Every point counts by how close it is to the line
    (1 - d/distance_thresh): with a plain count of the inliers, a line tilted so that a few outliers are just
    within the distance, while the points of the track are still inliers, would win over the line along the
    track.

    Arguments:
        img: [ndarray] 2D numpy array (uint8), image where >0 are points.
        max_lines: [int] Maximum number of lines to find.
        min_pixels: [int] Minimum number of inliers to accept a line.
        distance_thresh: [float] Maximum distance (px) from line to be an inlier.
        min_line_length: [float] Minimum length of a line segment.
        max_gap: [float] Maximum gap size allowed within a line segment.

    Keyword arguments:
        max_iterations: [int] Maximum RANSAC iterations per line search.
        debug: [bool] If True, print debug information.

    Return:
        lines: [list] List of (rho, theta, x_start, y_start, x_end, y_end) tuples.
    """

    h, w = img.shape
    cx = w/2.0
    cy = h/2.0

    # Extract points (image coordinates, top-left origin)
    y_idxs, x_idxs = np.nonzero(img)
    points = np.column_stack((x_idxs, y_idxs)).astype(np.float64)

    # Random point sampling with a fixed seed, so the same image always gives the same lines
    rng = np.random.RandomState(0)

    # The second point is drawn within this distance of the first one, far enough from it to define the
    #   direction well
    sample_radius = max(3*max_gap, 2*min_line_length)
    min_sample_dist_sq = max(3.0, 0.25*min_line_length)**2

    log.debug("RANSAC: Starting with {:d} points.".format(len(points)))

    found_lines = []

    # Stop after this many searches in a row without a line, and after a bounded number of searches in total
    #   (segments too short to be a line are removed and searched again)
    consecutive_failures = 0
    MAX_FAILURES = 10
    searches = 0

    while (len(points) >= min_pixels) and (len(found_lines) < max_lines) \
            and (consecutive_failures < MAX_FAILURES) and (searches < 4*max_lines):

        searches += 1

        best_segment = None
        best_score = 0.0
        best_model = None

        for _ in range(max_iterations):

            # Sample a point and a second one from its neighbourhood
            p1 = points[rng.randint(len(points))]
            near = np.where((np.abs(points[:, 0] - p1[0]) < sample_radius)
                            & (np.abs(points[:, 1] - p1[1]) < sample_radius))[0]
            p2 = points[near[rng.randint(len(near))]]
            if (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 <= min_sample_dist_sq:
                continue

            rho, theta_deg = getPolarLine(p1[0], p1[1], p2[0], p2[1], w, h)
            segment, score = _bestSegment(points, cx, cy, rho, theta_deg, distance_thresh, max_gap)

            if (len(segment) >= min_pixels) and ((best_segment is None) or (score > best_score)):
                best_segment = segment
                best_score = score
                best_model = (rho, theta_deg)

        if best_segment is None:
            consecutive_failures += 1
            log.debug("  Failed to find line (Attempt {:d}/{:d})".format(consecutive_failures, MAX_FAILURES))
            continue

        # Refine: fit the line to the points of the segment (weighted by their distance from the current line)
        #   and take the segment of the fitted line, until it settles
        rho, theta_deg = best_model
        for _ in range(3):

            dists, _ = _lineDistances(points[best_segment], cx, cy, rho, theta_deg)
            weights = np.maximum(0, 1.0 - dists/distance_thresh)
            rho_ref, theta_ref = fitLine(points[best_segment, 0], points[best_segment, 1], w, h,
                weights=weights)

            segment, _ = _bestSegment(points, cx, cy, rho_ref, theta_ref, distance_thresh, max_gap)
            if len(segment) < min_pixels:
                break

            rho, theta_deg, best_segment = rho_ref, theta_ref, segment

        # Extent of the segment along the line
        _, t = _lineDistances(points[best_segment], cx, cy, rho, theta_deg)
        t_min, t_max = np.min(t), np.max(t)

        # Remove the points covered by the segment: close to the line and within its extent
        all_dists, all_t = _lineDistances(points, cx, cy, rho, theta_deg)
        to_remove = (all_dists < 1.5*distance_thresh) & (all_t >= t_min - max_gap/2.0) \
            & (all_t <= t_max + max_gap/2.0)
        to_remove[best_segment] = True
        points = points[~to_remove]

        # A dense cluster which is too short is not a line, it is removed and the search continues
        if (t_max - t_min) <= min_line_length:
            log.debug("  Segment too short: {:.1f} px, removed {:d} points".format(t_max - t_min,
                np.sum(to_remove)))
            continue

        consecutive_failures = 0

        # End points in image coordinates. The point on the line closest to the center is (rho*cos, rho*sin),
        #   the direction along the line is (-sin, cos)
        theta = np.radians(theta_deg)
        x_line = rho*np.cos(theta) + cx
        y_line = rho*np.sin(theta) + cy
        found_lines.append((rho, theta_deg,
                            x_line - t_min*np.sin(theta), y_line + t_min*np.cos(theta),
                            x_line - t_max*np.sin(theta), y_line + t_max*np.cos(theta)))

        log.debug("  Found Line: rho={:.1f}, theta={:.1f}, len={:.1f}, {:d} points, {:d} remaining".format(
            rho, theta_deg, t_max - t_min, len(best_segment), len(points)))

    # Final Merge Pass
    if len(found_lines) > 1:
        found_lines = mergeSegments(found_lines, distance_thresh, debug=debug)

    return found_lines


def mergeSegments(lines, distance_thresh, angle_thresh=10.0, overlap_fraction=0.1, debug=False):
    """
    Merge collinear segments.
    Refactored to sort by length, ensuring small fragments merge into large lines.
    """
    if len(lines) <= 1: return lines

    # Convert to dictionary objects for mutable state
    # We calculate a 'direction vector' for each line to help with projection
    pool = []
    for (rho, theta, x1, y1, x2, y2) in lines:
        length = np.hypot(x2-x1, y2-y1)
        pool.append({
            'params': (rho, theta, x1, y1, x2, y2),
            'length': length,
            'p1': np.array([x1, y1]),
            'p2': np.array([x2, y2]),
            'alive': True
        })

    # Sort by length descending (Merge small into big)
    pool.sort(key=lambda x: x['length'], reverse=True)
    
    merged_count = 0
    
    for i in range(len(pool)):
        if not pool[i]['alive']: continue
        
        L1 = pool[i]
        theta1 = L1['params'][1]
        
        # Direction vector of L1
        v1 = L1['p2'] - L1['p1']
        v1 /= (np.linalg.norm(v1) + 1e-9)
        
        for j in range(i+1, len(pool)):
            if not pool[j]['alive']: continue
            
            L2 = pool[j]
            theta2 = L2['params'][1]
            
            # 1. Angle Check (Handle 0/360 wrap)
            diff = abs(theta1 - theta2)
            diff = min(diff, 360 - diff)
            # Also check for 180 flips (lines can be antiparallel)
            if diff > angle_thresh and abs(diff - 180) > angle_thresh:
                continue
                
            # 2. Distance Check (Point-to-Line)
            # Check if L2's midpoint is close to L1's infinite line
            mid2 = (L2['p1'] + L2['p2'])/2
            # Perpendicular distance: |det(v1, mid2-p1)|
            v_rel = mid2 - L1['p1']
            perp_dist = abs(v1[0]*v_rel[1] - v1[1]*v_rel[0])
            
            if perp_dist > distance_thresh*4.0: # Generous merge threshold
                continue
                
            # 3. Longitudinal Overlap Check
            # Project everything onto L1's line
            # t = dot(p - p1, v1)
            t1_a, t1_b = 0, L1['length']
            t2_a = np.dot(L2['p1'] - L1['p1'], v1)
            t2_b = np.dot(L2['p2'] - L1['p1'], v1)
            
            min2, max2 = min(t2_a, t2_b), max(t2_a, t2_b)
            
            # Check overlap
            overlap_start = max(t1_a, min2)
            overlap_end = min(t1_b, max2)
            overlap_len = max(0, overlap_end - overlap_start)
            
            # Check gap (if not overlapping)
            # gap is distance between intervals
            if min2 > t1_b: gap = min2 - t1_b
            elif max2 < t1_a: gap = t1_a - max2
            else: gap = 0
            
            # Merge if overlapping OR gap is small (bridging)
            if overlap_len > 0 or gap < distance_thresh*8.0:
                # MERGE: Extend L1
                new_t_min = min(t1_a, min2)
                new_t_max = max(t1_b, max2)
                
                # Update L1
                L1['p1'] = L1['p1'] + v1*new_t_min # Note: this shifts origin, but v1 stays valid
                L1['p2'] = L1['p1'] + v1*(new_t_max - new_t_min)
                L1['length'] = new_t_max - new_t_min
                
                # Update params tuple (keep rho/theta, update endpoints)
                L1['params'] = (L1['params'][0], L1['params'][1], 
                                L1['p1'][0], L1['p1'][1], L1['p2'][0], L1['p2'][1])
                
                L2['alive'] = False
                merged_count += 1
                
    if merged_count > 0:
        log.debug(f"MergeSegments: Merged {merged_count} segments.")

    return [item['params'] for item in pool if item['alive']]