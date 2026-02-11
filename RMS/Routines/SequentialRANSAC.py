"""
Refactored Sequential RANSAC for line detection.
Optimized for speed and robustness on sparse, intermittent data.
"""

import numpy as np
import math
import random

def getPolarLine(x1, y1, x2, y2, img_w=None, img_h=None):
    """ 
    Calculate polar line coordinates (rho, theta) given 2 points.
    Uses standard image coordinates (0,0 at top-left).
    
    Arguments:
        x1, y1, x2, y2: Point coordinates.
        img_w, img_h: (Unused) Kept for API compatibility.
        
    Return:
        rho: Perpendicular distance from origin to line.
        theta_deg: Angle of the normal vector in degrees.
    """
    if abs(x1 - x2) < 1e-9 and abs(y1 - y2) < 1e-9:
        return 0.0, 0.0

    # Line direction vector
    dx = x2 - x1
    dy = y2 - y1
    
    # Normal vector (-dy, dx) gives the direction of rho
    nx = -dy
    ny = dx
    
    # Normalize
    norm = np.sqrt(nx*nx + ny*ny)
    if norm == 0: return 0.0, 0.0
    nx /= norm
    ny /= norm
    
    # Rho is the dot product of any point on the line with the normal
    rho = x1 * nx + y1 * ny
    theta = np.arctan2(ny, nx)

    # Standardize rho >= 0
    if rho < 0:
        rho = -rho
        theta += np.pi
    
    return rho, np.degrees(theta)


def fitLine(x, y, img_w=None, img_h=None, weights=None):
    """ 
    Fit a line to a set of points using Weighted PCA (Total Least Squares).
    
    Arguments:
        x, y: [ndarray] Coordinates.
        img_w, img_h: (Unused) Kept for API compatibility.
        weights: [ndarray] Optional weights for each point.
        
    Return:
        (rho, theta_deg): Fitted line parameters.
    """
    pts = np.column_stack((x, y))
    
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
        # Multiply each row by its weight
        weighted_centered = centered * weights[:, np.newaxis]
        cov = np.dot(weighted_centered.T, centered)

    # Eigen decomposition to find the normal vector (smallest eigenvalue)
    # eigh is faster and more stable for symmetric matrices than eig
    vals, vecs = np.linalg.eigh(cov)
    
    # The normal is the eigenvector corresponding to the smallest eigenvalue
    # eigh sorts eigenvalues in ascending order, so index 0 is smallest
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
    
    # Extract points
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

            # 2. Model
            rho, theta_deg = getPolarLine(p1[0], p1[1], p2[0], p2[1])
            theta_rad = np.radians(theta_deg)
            ct, st = np.cos(theta_rad), np.sin(theta_rad)

            # 3. Distance Check (Vectorized)
            # dist = |x*cos + y*sin - rho|
            # Note: We use the standard coordinates directly.
            dists = np.abs(points[:, 0] * ct + points[:, 1] * st - rho)
            
            # Fast filter: check total inlier count first
            inlier_mask = dists < distance_thresh
            if np.sum(inlier_mask) < min_pixels:
                continue

            # 4. Connectivity Check (Segments)
            # Project inliers onto the line: t = -x*sin + y*cos
            inlier_pts = points[inlier_mask]
            t_vals = -inlier_pts[:, 0] * st + inlier_pts[:, 1] * ct
            
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
            
            # Calculate weights based on distance to the initial model
            rho_init, theta_init_deg = best_model
            theta_rad = np.radians(theta_init_deg)
            dists = np.abs(segment_pts[:,0]*np.cos(theta_rad) + segment_pts[:,1]*np.sin(theta_rad) - rho_init)
            
            # Linear weight decay
            weights = np.maximum(0, 1.0 - (dists / distance_thresh))
            
            # Refit
            rho_ref, theta_ref_deg = fitLine(segment_pts[:, 0], segment_pts[:, 1], weights=weights)
            theta_ref_rad = np.radians(theta_ref_deg)
            
            # Recalculate extent on refined model
            t_vals = -segment_pts[:, 0] * np.sin(theta_ref_rad) + segment_pts[:, 1] * np.cos(theta_ref_rad)
            t_min, t_max = np.min(t_vals), np.max(t_vals)
            
            # Calculate endpoints
            # x = x0 + t*ux, y = y0 + t*uy
            # Normal is (cos, sin). Direction is (-sin, cos)
            # Projection center point closest to origin is (rho*cos, rho*sin)
            xc = rho_ref * np.cos(theta_ref_rad)
            yc = rho_ref * np.sin(theta_ref_rad)
            
            x_start = xc - t_min * np.sin(theta_ref_rad)
            y_start = yc + t_min * np.cos(theta_ref_rad)
            x_end = xc - t_max * np.sin(theta_ref_rad)
            y_end = yc + t_max * np.cos(theta_ref_rad)
            
            found_lines.append((rho_ref, theta_ref_deg, x_start, y_start, x_end, y_end))
            
            if debug:
                print(f"  Found Line: rho={rho_ref:.1f}, theta={theta_ref_deg:.1f}, len={best_score:.1f}")

            # --- Removal Step ---
            # Remove points that are "covered" by this segment.
            # Criteria: Close to infinite line AND within the t-range of the segment.
            
            # 1. Distance to refined line
            all_dists = np.abs(points[:, 0]*np.cos(theta_ref_rad) + points[:, 1]*np.sin(theta_ref_rad) - rho_ref)
            
            # 2. Projection onto refined line
            all_t = -points[:, 0] * np.sin(theta_ref_rad) + points[:, 1] * np.cos(theta_ref_rad)
            
            # 3. Buffer logic
            t_buffer = max_gap * 0.5 
            
            # Mask of points to remove
            to_remove = (all_dists < distance_thresh * 1.5) & \
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
            mid2 = (L2['p1'] + L2['p2']) / 2
            # Perpendicular distance: |det(v1, mid2-p1)|
            v_rel = mid2 - L1['p1']
            perp_dist = abs(v1[0]*v_rel[1] - v1[1]*v_rel[0])
            
            if perp_dist > distance_thresh * 4.0: # Generous merge threshold
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
            if overlap_len > 0 or gap < distance_thresh * 8.0:
                # MERGE: Extend L1
                new_t_min = min(t1_a, min2)
                new_t_max = max(t1_b, max2)
                
                # Update L1
                L1['p1'] = L1['p1'] + v1 * new_t_min # Note: this shifts origin, but v1 stays valid
                L1['p2'] = L1['p1'] + v1 * (new_t_max - new_t_min)
                L1['length'] = new_t_max - new_t_min
                
                # Update params tuple (keep rho/theta, update endpoints)
                L1['params'] = (L1['params'][0], L1['params'][1], 
                                L1['p1'][0], L1['p1'][1], L1['p2'][0], L1['p2'][1])
                
                L2['alive'] = False
                merged_count += 1
                
    if debug and merged_count > 0:
        print(f"MergeSegments: Merged {merged_count} segments.")

    return [item['params'] for item in pool if item['alive']]