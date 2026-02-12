"""
Refactored Sequential RANSAC for line detection.
Optimized for speed and robustness on sparse, intermittent data.
"""

import numpy as np
import math
import random

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


def findLines(img, max_lines, min_pixels, distance_thresh, min_line_length, max_gap, max_iterations=1000, debug=False):
    """
    Find lines in the image using Sequential RANSAC with gap bridging.
    
    Arguments:
        img: [ndarray] 2D numpy array (uint8), image where >0 are points.
        max_lines: [int] Maximum number of lines to find.
        min_pixels: [int] Minimum number of inliers to accept a line.
        distance_thresh: [float] Maximum distance (px) from line to be an inlier.
        min_line_length: [float] Minimum length of a line segment.
        max_gap: [float] Maximum gap size allowed within a line segment.
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
    points = np.column_stack((x_idxs, y_idxs)).astype(np.float32)

    found_lines = []
    
    if debug:
        print(f"RANSAC: Starting with {len(points)} points.")

    # Safety counter to prevent infinite loops if we can't find a valid line
    consecutive_failures = 0
    MAX_FAILURES = 10

    while len(points) >= min_pixels and len(found_lines) < max_lines:
        
        if consecutive_failures >= MAX_FAILURES:
            if debug: print("RANSAC: Stopping due to consecutive failures.")
            break

        best_model = None       # (rho, theta)
        best_score = 0.0        # Length of the best segment
        best_segment_mask = None # Boolean mask of points belonging to the best segment

        # --- RANSAC Iterations ---
        for i in range(max_iterations):
            # 1. Sample
            if len(points) < 2: break

            # 1. Sample 2 random points
            # Try up to 5 times to find a pair with good separation
            valid_pair = False
            for _ in range(5):
                idx = np.random.choice(len(points), 2, replace=False)
                p1, p2 = points[idx]

                # Check distance squared
                dist_sq = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

                if dist_sq > 3**2: # Minimum 3 pixels apart
                    valid_pair = True
                    break
                    
            if not valid_pair: continue

            # 2. Model (center-based coordinates)
            rho, theta_deg = getPolarLine(p1[0], p1[1], p2[0], p2[1], w, h)
            theta_rad = np.radians(theta_deg)
            ct, st = np.cos(theta_rad), np.sin(theta_rad)

            # 3. Distance Check (Vectorized) — center points first
            # dist = |x_c*cos + y_c*sin - rho|
            pts_cx = points[:, 0] - cx
            pts_cy = points[:, 1] - cy
            dists = np.abs(pts_cx*ct + pts_cy*st - rho)
            
            # Fast filter: check total inlier count first
            inlier_mask = dists < distance_thresh
            if np.sum(inlier_mask) < min_pixels:
                continue

            # 4. Connectivity Check (Segments)
            # Project inliers onto the line: t = -x_c*sin + y_c*cos
            inlier_pts = points[inlier_mask]
            inlier_cx = inlier_pts[:, 0] - cx
            inlier_cy = inlier_pts[:, 1] - cy
            t_vals = -inlier_cx*st + inlier_cy*ct
            
            # Sort t_vals to find gaps
            sort_idx = np.argsort(t_vals)
            t_sorted = t_vals[sort_idx]
            
            # Calculate gaps
            gaps = np.diff(t_sorted)
            # Find indices where gaps exceed max_gap
            split_indices = np.where(gaps > max_gap)[0] + 1
            
            # Split into segments based on gaps
            # We need the sizes of segments to find the best one
            if len(split_indices) == 0:
                # No gaps, one big segment
                length = t_sorted[-1] - t_sorted[0]
                if length > best_score:
                    best_score = length
                    best_model = (rho, theta_deg)
                    best_segment_mask = inlier_mask
            else:
                # Check each sub-segment
                segment_starts = np.insert(split_indices, 0, 0)
                segment_ends = np.append(split_indices, len(t_sorted))
                
                for k in range(len(segment_starts)):
                    start = segment_starts[k]
                    end = segment_ends[k]
                    seg_len_pts = end - start
                    
                    if seg_len_pts < min_pixels: continue
                    
                    # Length in pixels
                    length = t_sorted[end-1] - t_sorted[start]
                    
                    if length > best_score:
                        best_score = length
                        best_model = (rho, theta_deg)
                        
                        # We need to construct the mask for JUST this segment
                        # Recover original indices for this segment
                        # inlier_mask -> True indices
                        flat_indices = np.where(inlier_mask)[0]
                        # sort_idx maps sorted t -> inlier_pts
                        segment_local_indices = sort_idx[start:end]
                        # map back to global points
                        segment_global_indices = flat_indices[segment_local_indices]
                        
                        # Create a specific mask for this segment
                        new_mask = np.zeros(len(points), dtype=bool)
                        new_mask[segment_global_indices] = True
                        best_segment_mask = new_mask

        # --- End RANSAC Loop ---

        if best_model is not None and best_score > min_line_length:
            consecutive_failures = 0
            
            # Refine the model using Weighted PCA on the best segment
            segment_pts = points[best_segment_mask]
            
            # Calculate weights based on distance to the initial model (center-based)
            rho_init, theta_init_deg = best_model
            theta_rad = np.radians(theta_init_deg)
            seg_cx = segment_pts[:,0] - cx
            seg_cy = segment_pts[:,1] - cy
            dists = np.abs(seg_cx*np.cos(theta_rad) + seg_cy*np.sin(theta_rad) - rho_init)
            
            # Linear weight decay
            weights = np.maximum(0, 1.0 - (dists/distance_thresh))
            
            # Refit (center-based)
            rho_ref, theta_ref_deg = fitLine(segment_pts[:, 0], segment_pts[:, 1], w, h, weights=weights)
            theta_ref_rad = np.radians(theta_ref_deg)
            
            # Recalculate extent on refined model (center-based)
            seg_cx = segment_pts[:, 0] - cx
            seg_cy = segment_pts[:, 1] - cy
            t_vals = -seg_cx*np.sin(theta_ref_rad) + seg_cy*np.cos(theta_ref_rad)
            t_min, t_max = np.min(t_vals), np.max(t_vals)
            
            # Calculate endpoints in center-based coords, then convert to image coords
            # Point on line closest to center: (rho*cos, rho*sin)
            # Direction along line: (-sin, cos)
            xc_line = rho_ref*np.cos(theta_ref_rad)
            yc_line = rho_ref*np.sin(theta_ref_rad)
            
            # Endpoints in center-based coords
            x_start_c = xc_line - t_min*np.sin(theta_ref_rad)
            y_start_c = yc_line + t_min*np.cos(theta_ref_rad)
            x_end_c = xc_line - t_max*np.sin(theta_ref_rad)
            y_end_c = yc_line + t_max*np.cos(theta_ref_rad)
            
            # Convert to image (top-left) coords for output
            x_start = x_start_c + cx
            y_start = y_start_c + cy
            x_end = x_end_c + cx
            y_end = y_end_c + cy
            
            found_lines.append((rho_ref, theta_ref_deg, x_start, y_start, x_end, y_end))
            
            if debug:
                print(f"  Found Line: rho={rho_ref:.1f}, theta={theta_ref_deg:.1f}, len={best_score:.1f}")

            # --- Removal Step ---
            # Remove points that are "covered" by this segment.
            # Criteria: Close to infinite line AND within the t-range of the segment.
            
            # 1. Distance to refined line (center-based)
            all_cx = points[:, 0] - cx
            all_cy = points[:, 1] - cy
            all_dists = np.abs(all_cx*np.cos(theta_ref_rad) + all_cy*np.sin(theta_ref_rad) - rho_ref)
            
            # 2. Projection onto refined line (center-based)
            all_t = -all_cx*np.sin(theta_ref_rad) + all_cy*np.cos(theta_ref_rad)
            
            # 3. Buffer logic
            t_buffer = max_gap*0.5 
            
            # Mask of points to remove
            to_remove = (all_dists < distance_thresh*1.5) & \
                        (all_t >= t_min - t_buffer) & \
                        (all_t <= t_max + t_buffer)
            
            points = points[~to_remove]
            
            if debug:
                print(f"  Removed {np.sum(to_remove)} points. Remaining: {len(points)}")
                
        else:
            consecutive_failures += 1
            if debug: print(f"  Failed to find line (Attempt {consecutive_failures}/{MAX_FAILURES})")

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
                
    if debug and merged_count > 0:
        print(f"MergeSegments: Merged {merged_count} segments.")

    return [item['params'] for item in pool if item['alive']]