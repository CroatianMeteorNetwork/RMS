"""
Refactored Sequential RANSAC for line detection.
"""

import numpy as np
import random
import math


def getPolarLine(x1, y1, x2, y2, img_w, img_h):
    """ Calculate polar line coordinates (rho, theta) given 2 points.
        Coordinate system starts in the image center.
    """
    x0 = float(img_w)/2
    y0 = float(img_h)/2

    dx = float(x2 - x1)
    dy = float(y2 - y1)

    # Calculate polar line coordinates
    theta = -np.arctan2(dx, dy)
    rho = (dy*x0 - dx*y0 + x2*y1 - y2*x1)/np.sqrt(dy**2 + dx**2)
    
    # Correct for quadrant
    if rho > 0:
        theta += np.pi
    else:
        rho = -rho

    return rho, np.degrees(theta)


def fitLine(x, y, img_w, img_h, weights=None):
    """ Fit a line to a set of points using orthogonal regression (PCA) and return (rho, theta).
        This minimizes the sum of squared perpendicular distances to the line.
        Supports weighted fitting for density-aware refinement.

    Arguments:
        x: [ndarray] X coordinates of the points.
        y: [ndarray] Y coordinates of the points.
        img_w: [int] Image width (used for coordinate centering).
        img_h: [int] Image height (used for coordinate centering).

    Keyword arguments:
        weights: [ndarray] Optional per-point weights. If None, all points are weighted equally.

    Return:
        (rho, theta_deg): [tuple] Line parameters in polar form (distance from center, angle in degrees).
    """
    
    if weights is None:
        # Calculate centroids
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        # Calculate covariance matrix components
        dx = x - x_mean
        dy = y - y_mean
        
        sxx = np.sum(dx*dx)
        syy = np.sum(dy*dy)
        sxy = np.sum(dx*dy)

    else:
        # Weighted centroids
        w_sum = np.sum(weights)
        if w_sum == 0:
            return 0, 0 # Should not happen
            
        x_mean = np.sum(x*weights)/w_sum
        y_mean = np.sum(y*weights)/w_sum
        
        # Weighted covariance matrix components
        dx = x - x_mean
        dy = y - y_mean
        
        sxx = np.sum(weights*dx*dx)
        syy = np.sum(weights*dy*dy)
        sxy = np.sum(weights*dx*dy)


    # Calculate the angle of the principal axis
    # The formula 0.5*atan2(-2*sxy, syy-sxx) minimizes the sum of squared perpendicular distances.
    # Thus, it gives the angle of the NORMAL vector to the line.
    theta_normal = 0.5*np.arctan2(-2*sxy, syy - sxx)
    
    # We need to generate a second point along the line to use getPolarLine.
    # The line direction is perpendicular to the normal.
    theta_line = theta_normal + np.pi/2
    
    x1 = x_mean
    y1 = y_mean
    
    x2 = x_mean + np.cos(theta_line)
    y2 = y_mean + np.sin(theta_line)
    
    return getPolarLine(x1, y1, x2, y2, img_w, img_h)


def findLines(img, max_lines, min_pixels, distance_thresh, min_line_length, max_gap, max_iterations=1000, debug=False):
    """
    Find lines in the image using Sequential RANSAC.

    Arguments:
        img: [ndarray] 2D numpy array (uint8), image where >0 are points.
             Or a list of (y, x) points if already extracted? 
             Let's assume input is the image as standard in Detection.py.
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

    img_h, img_w = img.shape
    
    # Extract points from the image
    # Note: nonzero returns (row_indices, col_indices) -> (y, x)
    y_idxs, x_idxs = np.nonzero(img)
    points = np.column_stack((x_idxs, y_idxs)) # pairs of (x, y)

    found_lines = []

    if debug:
        print(f"RANSAC: Starting with {len(points)} points. Max lines: {max_lines}")

    # Keep searching as long as we have enough points and haven't found enough lines
    while len(points) >= min_pixels and len(found_lines) < max_lines:
        
        best_line = None
        best_inliers_indices = []
        best_score = 0 # Length of the segment

        current_max_iterations = max_iterations
        
        # Adaptive RANSAC or fixed iterations
        for i in range(current_max_iterations):
            
            # 1. Sample 2 random points
            if len(points) < 2:
                break
                
            sample_indices = np.random.choice(len(points), 2, replace=False)
            p1 = points[sample_indices[0]]
            p2 = points[sample_indices[1]]

            # 2. Compute model (rho, theta)
            x1, y1 = p1
            x2, y2 = p2
            
            # Avoid singular lines (same point)
            if x1 == x2 and y1 == y2:
                continue

            rho, theta_deg = getPolarLine(x1, y1, x2, y2, img_w, img_h)
            theta_rad = np.radians(theta_deg)
            
            # 3. Compute distances of all points to the line
            # Distance = |x*cos(theta) + y*sin(theta) - rho| -- Wait, coordinate system!
            # getPolarLine uses center-based coords.
            # Points are in top-left based coords (0..w, 0..h).
            # Convert points to center-based for distance calculation
            
            x0 = img_w/2.0
            y0 = img_h/2.0
            
            pts_centered_x = points[:, 0] - x0
            pts_centered_y = points[:, 1] - y0
            
            # Hough formula: rho = x*cos(theta) + y*sin(theta)
            # Distance = rho_calc - rho_model
            
            dists = np.abs(pts_centered_x*np.cos(theta_rad) + pts_centered_y*np.sin(theta_rad) - np.abs(rho))
            
            # 4. Find inliers
            inlier_mask = dists < distance_thresh
            inlier_count = np.sum(inlier_mask)

            if inlier_count < min_pixels:
                continue
                
            # Optimize: Only check connectivity if inlier count is promising (e.g. better than current best)
            # But "score" is length, not count. A short dense line might have more points than a long sparse one? 
            # We usually care about length for meteors.
            
            # Let's verify connectivity
            potential_inliers = points[inlier_mask]
            
            # Project points onto the line to find extent along the line
            # The direction of the line is (-sin(theta), cos(theta))
            # Position along line `t` = -x*sin(theta) + y*cos(theta)
            
            t_vals = -pts_centered_x[inlier_mask]*np.sin(theta_rad) + pts_centered_y[inlier_mask]*np.cos(theta_rad)
            
            # Sort by position
            sorted_indices = np.argsort(t_vals)
            t_sorted = t_vals[sorted_indices]
            
            # Find gaps
            gaps = np.diff(t_sorted)
            
            # Find split indices where gap > max_gap
            split_indices = np.where(gaps > max_gap)[0] + 1
            segments = np.split(sorted_indices, split_indices)
            
            # Find the best segment (longest)
            max_len = 0
            best_segment_indices = None
            
            for seg_indices in segments:
                if len(seg_indices) < 2:
                    continue
                    
                # Calculate length of this segment
                # t_start = t_sorted[np.where(sorted_indices == seg_indices[0])[0][0]] # This is circular logic
                # seg_indices contains indices into t_sorted's parent array (the inliers)
                # But we split sorted_indices, which are indices into t_vals (and thus potential_inliers)
                
                # We need to map back to which sorted slice this is.
                # simpler: iterate over the splits of t_sorted directly
                pass 
            
            # Re-doing segment logic simpler
            t_segments = np.split(t_sorted, split_indices)
            
            longest_segment_idx = -1
            max_seg_length = 0
            
            for k, t_seg in enumerate(t_segments):
                if len(t_seg) < 2:
                    continue
                length = t_seg[-1] - t_seg[0]
                
                # Check if the segment has enough points and is the longest
                if len(t_seg) >= min_pixels and length > max_seg_length:
                    max_seg_length = length
                    longest_segment_idx = k
            
            if longest_segment_idx == -1:
                continue
                
            # Check against threshold
            if max_seg_length < min_line_length:
                continue
                
            # If this is the best line so far
            if max_seg_length > best_score:
                best_score = max_seg_length
                
                # Get the inliers belonging to this segment
                # We need the indices in the original 'points' array
                
                # segments[longest_segment_idx] are indices into t_vals/potential_inliers
                # potential_inliers are points[inlier_mask]
                
                # Recover original indices
                original_indices = np.where(inlier_mask)[0]
                segment_local_indices = segments[longest_segment_idx]
                
                best_inliers_indices = original_indices[segment_local_indices]
                best_line = (rho, theta_deg)


        # After iterations, if we found a line
        if best_line is not None:
            
            if debug:
                print(f"RANSAC: Found line {len(found_lines)+1}. Rho: {best_line[0]:.2f}, Theta: {best_line[1]:.2f}, Length: {best_score:.2f}, Inliers: {len(best_inliers_indices)}")

            # Refit line to optimized inliers for better precision
            if len(best_inliers_indices) > 2:
                inlier_points = points[best_inliers_indices]
                
                # Calculate weights based on distance to the initial line
                # Recalculate distances of inliers to the current best_line
                rho_init, theta_init_deg = best_line
                theta_init_rad = np.radians(theta_init_deg)
                
                x0 = img_w/2.0
                y0 = img_h/2.0
                pts_centered_x = inlier_points[:, 0] - x0
                pts_centered_y = inlier_points[:, 1] - y0
                
                dists = np.abs(pts_centered_x*np.cos(theta_init_rad) + pts_centered_y*np.sin(theta_init_rad) - np.abs(rho_init))
                
                # Weighting scheme:
                # 1.0 at distance 0
                # 0.0 at distance >= distance_thresh
                # Linear decay: w = 1 - (d/distance_thresh)
                # Clip negative weights (shouldn't happen for inliers, but safe to clip)
                
                weights = 1.0 - (dists/distance_thresh)
                weights = np.maximum(weights, 0)
                
                # --- Density Weighting ---
                # Calculate local density (neighbor count) for each inlier.
                # Points in dense blobs should have higher weights than stragglers.
                
                # Radius for density check (e.g. 2.5 px covers explicit neighbors)
                density_radius_sq = 2.5**2
                
                # Optimization: For small N, broadcasting is fine. 
                # For large N (>1000), this might be slow, but inlier count usually < 500.
                
                # (N, 1, 2) - (1, N, 2) -> (N, N, 2)
                diffs = inlier_points[:, None, :] - inlier_points[None, :, :]
                dists_sq = np.sum(diffs**2, axis=2)
                
                # Count neighbors within radius
                neighbor_counts = np.sum(dists_sq <= density_radius_sq, axis=1)
                
                # Normalize density weights? 
                # Linear weighting by count seems appropriate. 
                # A blob pixel has maybe 5-8 neighbors. A stray has 1-2.
                # Combined with distance weight, this should snap to the blob.
                
                # To be more aggressive against outliers, use squared density
                weights *= (neighbor_counts**2)
                
                # Perform weighted fitting
                rho_refined, theta_refined_deg = fitLine(inlier_points[:, 0], inlier_points[:, 1], img_w, img_h, weights=weights)
                
                if debug:
                    print(f"  Refined Rho: {rho_refined:.2f}, Theta: {theta_refined_deg:.2f}")

                best_line = (rho_refined, theta_refined_deg) 

            # Calculate line extent (start and end points)
            rho, theta_deg = best_line
            theta_rad = np.radians(theta_deg)
            x0 = img_w/2.0
            y0 = img_h/2.0
            
            # Center inlier points
            pts_centered_x = inlier_points[:, 0] - x0
            pts_centered_y = inlier_points[:, 1] - y0
            
            # Project to line parameter t
            # x_c = rho*cos - t*sin
            # y_c = rho*sin + t*cos
            # t = -x_c*sin + y_c*cos
            t_vals = -pts_centered_x*np.sin(theta_rad) + pts_centered_y*np.cos(theta_rad)
            
            t_min = np.min(t_vals)
            t_max = np.max(t_vals)
            
            # Reproject to image coordinates
            # x = x0 + rho*cos - t*sin
            # y = y0 + rho*sin + t*cos
            
            x_start = x0 + rho*np.cos(theta_rad) - t_min*np.sin(theta_rad)
            y_start = y0 + rho*np.sin(theta_rad) + t_min*np.cos(theta_rad)
            
            x_end = x0 + rho*np.cos(theta_rad) - t_max*np.sin(theta_rad)
            y_end = y0 + rho*np.sin(theta_rad) + t_max*np.cos(theta_rad)

            found_lines.append((rho, theta_deg, x_start, y_start, x_end, y_end))
            
            # Remove inliers from points
            # Re-calculate distances for all points to the found line to 
            # remove everything within a larger radius (cleanup)
            
            # Use 1.5x the distance threshold as requested
            removal_thresh = 1.5*distance_thresh
            
            # Center points
            pts_centered_x = points[:, 0] - x0
            pts_centered_y = points[:, 1] - y0
            
            # Distances
            dists = np.abs(pts_centered_x*np.cos(theta_rad) + pts_centered_y*np.sin(theta_rad) - np.abs(rho))
            
            # Keep points outside the removal threshold
            keep_mask = dists > removal_thresh
            points = points[keep_mask]
            
            if debug:
                print(f"RANSAC: Points remaining: {len(points)}")
            
        else:
            # If no line found after max_iterations, we probably exhausted the image
            if debug:
                print("RANSAC: No more lines found.")
            break

    # Merge overlapping segments (thick lines detected multiple times)
    if len(found_lines) > 1:
        found_lines = mergeSegments(found_lines, distance_thresh, debug=debug)
            
    return found_lines


def mergeSegments(lines, distance_thresh, angle_thresh=15.0, overlap_fraction=0.25, debug=False):
    """
    Merge line segments that likely represent the same thick line detected multiple times.
    
    Two segments are merged if:
      1. Their angles are similar (within angle_thresh degrees).
      2. Their perpendicular distance is less than 2*distance_thresh (their "strips" overlap).
      3. They have longitudinal overlap along the line direction (at least overlap_fraction of the shorter segment).
    
    Merged segments keep the parameters of the segment with more inliers (longer segment),
    and extend the endpoints to cover the union of both segments.
    
    Arguments:
        lines: [list] List of (rho, theta, x_start, y_start, x_end, y_end) tuples.
        distance_thresh: [float] The RANSAC distance threshold, used as "half-width" of the line strip.
        angle_thresh: [float] Maximum angular difference (degrees) to consider two lines as parallel.
        overlap_fraction: [float] Minimum fraction of the shorter segment that must overlap longitudinally.
        debug: [bool] If True, print debug information.
    
    Return:
        merged: [list] Merged list of (rho, theta, x_start, y_start, x_end, y_end) tuples.
    """
    
    if len(lines) <= 1:
        return lines
    
    # Convert to list of mutable entries: [rho, theta, x1, y1, x2, y2, alive]
    entries = []
    for line in lines:
        rho, theta, x1, y1, x2, y2 = line
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        entries.append({
            'rho': rho, 'theta': theta,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'length': length,
            'alive': True
        })
    
    # The strip overlap threshold: if perpendicular distance < 2*distance_thresh, the strips overlap
    perp_dist_thresh = 2*distance_thresh
    
    # Greedy merge: iterate and merge pairs
    merged_any = True
    while merged_any:
        merged_any = False
        
        for i in range(len(entries)):
            if not entries[i]['alive']:
                continue
                
            for j in range(i + 1, len(entries)):
                if not entries[j]['alive']:
                    continue
                
                li = entries[i]
                lj = entries[j]
                
                # 1. Check angular similarity (handle 0/360 wraparound)
                dtheta = abs(li['theta'] - lj['theta'])
                # Lines at theta and theta+180 are the same line with opposite rho sign
                dtheta = min(dtheta, abs(dtheta - 180), abs(dtheta - 360))
                
                if dtheta > angle_thresh:
                    continue
                
                # 2. Check perpendicular distance between the two lines
                # Use the midpoint of segment j and compute its distance to line i
                # Use polar line distance formula (center-based coords)
                theta_i_rad = np.radians(li['theta'])
                
                # Midpoints
                mx_j = (lj['x1'] + lj['x2'])/2.0
                my_j = (lj['y1'] + lj['y2'])/2.0
                
                # We don't have img_w/h here, but we can compute perpendicular distance
                # directly from the endpoints of line i.
                # Direction vector of line i
                dx_i = li['x2'] - li['x1']
                dy_i = li['y2'] - li['y1']
                len_i = li['length']
                
                if len_i < 1e-6:
                    continue
                
                # Perpendicular distance from midpoint of j to the infinite line through i
                # |cross product|/|direction|
                perp_dist = abs(dx_i*(li['y1'] - my_j) - dy_i*(li['x1'] - mx_j))/len_i
                
                if perp_dist > perp_dist_thresh:
                    continue
                
                # Also check distance from midpoint of i to line j
                mx_i = (li['x1'] + li['x2'])/2.0
                my_i = (li['y1'] + li['y2'])/2.0
                dx_j = lj['x2'] - lj['x1']
                dy_j = lj['y2'] - lj['y1']
                len_j = lj['length']
                
                if len_j < 1e-6:
                    continue
                    
                perp_dist2 = abs(dx_j*(lj['y1'] - my_i) - dy_j*(lj['x1'] - mx_i))/len_j
                
                if perp_dist2 > perp_dist_thresh:
                    continue
                
                # 3. Check longitudinal overlap
                # Project all 4 endpoints onto the direction of the longer segment
                if len_i >= len_j:
                    ref = li
                else:
                    ref = lj
                
                # Direction unit vector of reference line
                ref_dx = ref['x2'] - ref['x1']
                ref_dy = ref['y2'] - ref['y1']
                ref_len = ref['length']
                ux = ref_dx/ref_len
                uy = ref_dy/ref_len
                
                # Project all endpoints onto this direction
                # Use ref start as origin
                def proj(px, py):
                    return (px - ref['x1'])*ux + (py - ref['y1'])*uy
                
                ti1 = proj(li['x1'], li['y1'])
                ti2 = proj(li['x2'], li['y2'])
                tj1 = proj(lj['x1'], lj['y1'])
                tj2 = proj(lj['x2'], lj['y2'])
                
                # Get intervals [min, max] for each segment
                si_min, si_max = min(ti1, ti2), max(ti1, ti2)
                sj_min, sj_max = min(tj1, tj2), max(tj1, tj2)
                
                # Compute overlap
                overlap_start = max(si_min, sj_min)
                overlap_end = min(si_max, sj_max)
                overlap_len = max(0, overlap_end - overlap_start)
                
                # Minimum overlap required: overlap_fraction of the shorter segment
                shorter_len = min(si_max - si_min, sj_max - sj_min)
                
                if shorter_len < 1e-6 or overlap_len < overlap_fraction*shorter_len:
                    continue
                
                # --- Merge! ---
                if debug:
                    print(f"  Merging lines: rho={li['rho']:.2f},theta={li['theta']:.2f} "
                          f"+ rho={lj['rho']:.2f},theta={lj['theta']:.2f}")
                
                # Keep parameters of the longer/denser segment (primary)
                # Extend endpoints to the union
                if len_i >= len_j:
                    primary, secondary = li, lj
                else:
                    primary, secondary = lj, li
                
                # Find the union extent along the reference direction
                union_min = min(si_min, sj_min)
                union_max = max(si_max, sj_max)
                
                # Reproject union extent back to image coordinates
                new_x1 = ref['x1'] + union_min*ux
                new_y1 = ref['y1'] + union_min*uy
                new_x2 = ref['x1'] + union_max*ux
                new_y2 = ref['y1'] + union_max*uy
                
                # Update primary with merged extent
                primary['x1'] = new_x1
                primary['y1'] = new_y1
                primary['x2'] = new_x2
                primary['y2'] = new_y2
                primary['length'] = np.sqrt((new_x2 - new_x1)**2 + (new_y2 - new_y1)**2)
                
                # Kill secondary
                secondary['alive'] = False
                merged_any = True
                
                if debug:
                    print(f"    -> rho={primary['rho']:.2f}, theta={primary['theta']:.2f}, "
                          f"length={primary['length']:.2f}")
    
    # Collect surviving entries
    merged = []
    for e in entries:
        if e['alive']:
            merged.append((e['rho'], e['theta'], e['x1'], e['y1'], e['x2'], e['y2']))
    
    if debug and len(merged) < len(lines):
        print(f"  Merged {len(lines)} lines -> {len(merged)} lines")
    
    return merged
