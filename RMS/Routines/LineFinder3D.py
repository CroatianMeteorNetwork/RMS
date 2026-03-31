"""
Module for finding line segments in 3D point clouds (x, y, frame) using a robust 
Sequential RANSAC algorithm. Supports handling noise, configurable gaps in spatial 
and frame dimensions, and minimum constraints. Also provides capability to match 
a found line to a parametrized reference line.
"""

import numpy as np
import math

def fitLine3D(points, weights=None):
    """
    Fit a 3D line to a set of points using Weighted PCA (Total Least Squares).
    
    Arguments:
        points: [ndarray] Nx3 array of (x, y, scaled_frame) coordinates.
        weights: [ndarray] Optional weights for each point.
        
    Return:
        (mean, direction):
            mean: [ndarray] 1x3 point on the line.
            direction: [ndarray] 1x3 normalized direction vector.
    """
    if len(points) < 2:
        return np.zeros(3), np.array([1.0, 0.0, 0.0])

    if weights is None:
        mean = np.mean(points, axis=0)
        centered = points - mean
        cov = np.dot(centered.T, centered)
    else:
        w_sum = np.sum(weights)
        if w_sum == 0:
            return np.zeros(3), np.array([1.0, 0.0, 0.0])
        
        mean = np.average(points, axis=0, weights=weights)
        centered = points - mean
        weighted_centered = centered*weights[:, np.newaxis]
        cov = np.dot(weighted_centered.T, centered)

    # Eigen decomposition
    # The normal is the eigenvector corresponding to the LARGEST eigenvalue for a line fit
    # (PCA primary principal component)
    vals, vecs = np.linalg.eigh(cov)
    direction = vecs[:, np.argmax(vals)]
    
    # Ensure direction vector always points in positive time (frame) direction
    if direction[2] < 0:
        direction = -direction

    return mean, direction

def pointToLineDist3D(points, point_on_line, direction):
    """
    Calculate 3D perpendicular distance from points to a 3D line.
    
    Arguments:
        points: [ndarray] Nx3 array of coordinates.
        point_on_line: [ndarray] Any point on the line (1x3).
        direction: [ndarray] Normalized direction vector (1x3).
        
    Return:
        dists: [ndarray] Distances for each point.
    """
    vecs = points - point_on_line
    # Cross product of vector to line and line direction gives area of parallelogram
    # Since direction is normalized, norm of cross product is the height (distance)
    cross_prods = np.cross(vecs, direction)
    dists = np.linalg.norm(cross_prods, axis=1)
    return dists

def findLines3D(points, max_lines, min_points, dist_thresh, max_gap_frame, max_gap_spatial, 
                min_frames=10, frame_scale=None, img_w=None, img_h=None, 
                max_iterations=1000, min_ang_vel_px_per_frame=None, debug=False):
    """
    Find lines in 3D point cloud (x, y, frame) using Sequential 3D RANSAC.
    
    Arguments:
        points: [ndarray] Nx3 numpy array (x, y, frame).
        max_lines: [int] Maximum number of lines to find.
        min_points: [int] Minimum number of inliers to accept a line.
        dist_thresh: [float] Maximum distance (3D) to line to be an inlier.
        max_gap_frame: [float] Maximum gap size in frame dimension.
        max_gap_spatial: [float] Maximum spatial gap size (pixels) unscaled.
        min_frames: [float] Minimum spanning frames for a valid line.
        frame_scale: [float] Normalization ratio for frame dimension. If None,
            calculated so image diagonal equals 256 frames.
        img_w, img_h: [float] Image dimensions. Used for default frame_scale calculate.
        max_iterations: [int] Maximum RANSAC iterations per line search.
        min_ang_vel_px_per_frame: [float] Minimum angular velocity in px/frame. Lines moving
            slower than this are rejected but their points are still removed to avoid
            re-finding the same slow-moving object. None to disable.
        debug: [bool] Print verbose logs.
        
    Return:
        lines: [list] List of tuples ((x_start, y_start, f_start), (x_end, y_end, f_end))
    """
    points = np.asarray(points).astype(np.float64)
    if len(points) < min_points:
        return []

    # Automatic frame_scale if not provided
    if frame_scale is None:
        if img_w is None or img_h is None:
            min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
            min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
            img_w = max_x - min_x
            img_h = max_y - min_y
        
        img_diag = np.sqrt(img_w**2 + img_h**2)
        if img_diag == 0:
            img_diag = 256.0
        frame_scale = img_diag/256.0
        
    if debug:
        print(f"findLines3D: Using frame_scale = {frame_scale:.4f}")

    # Scale the frame dimension
    scaled_points = points.copy()
    scaled_points[:, 2] *= frame_scale

    found_lines = []
    
    # Internal mask: True means the point is still available to be matched
    available_mask = np.ones(len(points), dtype=bool)

    consecutive_failures = 0
    MAX_FAILURES = 10

    while np.sum(available_mask) >= min_points and len(found_lines) < max_lines:
        if consecutive_failures >= MAX_FAILURES:
            break

        pool_indices = np.where(available_mask)[0]
        pts_pool = scaled_points[pool_indices]
        
        best_model = None
        best_inlier_count = 0
        best_segment_indices = None # Global indices of points in best segment

        for i in range(max_iterations):
            if len(pts_pool) < 2:
                break
                
            # Randomly sample 2 points, prefer those further apart in time for stability
            p1_idx = np.random.choice(len(pts_pool))
            p1 = pts_pool[p1_idx]
            
            # Use distances in frame dimension to weight second point selection
            dt = np.abs(pts_pool[:, 2] - p1[2])
            if np.sum(dt) == 0:
                continue
            probs = dt / np.sum(dt)
            p2_idx = np.random.choice(len(pts_pool), p=probs)
            p2 = pts_pool[p2_idx]
            
            # Need strict difference in frames to define temporal direction
            if abs(p1[2] - p2[2]) < 1e-6:
                continue
                
            direction = p2 - p1
            direction_norm = np.linalg.norm(direction)
            if direction_norm < 1e-6:
                continue
            direction /= direction_norm

            # Invert direction if it points backwards in time
            if direction[2] < 0:
                direction = -direction

            # 3D Distance check
            dists = pointToLineDist3D(pts_pool, p1, direction)
            inlier_mask_pool = (dists < dist_thresh)
            
            if np.sum(inlier_mask_pool) < min_points:
                continue

            inlier_pts = pts_pool[inlier_mask_pool]
            inlier_global_idx = pool_indices[inlier_mask_pool]

            # Project onto line to parameterize (using dot product with direction)
            # t = (p - p1) • dir
            t_vals = np.dot(inlier_pts - p1, direction)
            
            # Sort by projection parameter t
            sort_idx = np.argsort(t_vals)
            t_sorted = t_vals[sort_idx]
            inlier_pts_sorted = inlier_pts[sort_idx]
            inlier_global_idx_sorted = inlier_global_idx[sort_idx]
            
            # Evaluate gaps in both frame (unscaled) and spatial dimensions
            # Revert frame scale for gap check
            unscaled_frames = inlier_pts_sorted[:, 2]/frame_scale
            frame_gaps = np.diff(unscaled_frames)
            
            spatial_xy = inlier_pts_sorted[:, :2]
            spatial_gaps = np.linalg.norm(np.diff(spatial_xy, axis=0), axis=1)

            # Split points if either gap exceeds threshold
            split_indices = np.where((np.abs(frame_gaps) > max_gap_frame) | 
                                     (spatial_gaps > max_gap_spatial))[0] + 1
                                     
            segment_starts = np.insert(split_indices, 0, 0)
            segment_ends = np.append(split_indices, len(t_sorted))
            
            for k in range(len(segment_starts)):
                start = segment_starts[k]
                end = segment_ends[k]
                seg_count = end - start
                
                if seg_count < min_points:
                    continue
                    
                # Check min frames span
                frames_in_seg = unscaled_frames[start:end]
                frame_span = np.max(frames_in_seg) - np.min(frames_in_seg)
                
                if frame_span < min_frames:
                    continue
                    
                # Calculate a combined score of points and frame span
                # This prevents picking short dense clusters over long thin lines (meteors)
                combined_score = seg_count * (1.0 + frame_span / 100.0)
                
                if combined_score > best_inlier_count:
                    best_inlier_count = combined_score
                    best_model = (p1, direction)
                    best_segment_indices = inlier_global_idx_sorted[start:end]

        if best_model is not None:
            # We found a valid segment
            consecutive_failures = 0
            
            # Refine model
            segment_pts_scaled = scaled_points[best_segment_indices]
            
            # Calculate weights (linear decay based on distance)
            dists = pointToLineDist3D(segment_pts_scaled, best_model[0], best_model[1])
            weights = np.maximum(0, 1.0 - (dists/dist_thresh))
            
            # Re-fit
            mean_pt, ref_direction = fitLine3D(segment_pts_scaled, weights=weights)
            
            # Project to find endpoints
            t_vals = np.dot(segment_pts_scaled - mean_pt, ref_direction)
            t_min, t_max = np.min(t_vals), np.max(t_vals)
            
            pt_start_scaled = mean_pt + ref_direction*t_min
            pt_end_scaled = mean_pt + ref_direction*t_max
            
            # Unscale frames
            pt_start = pt_start_scaled.copy()
            pt_end = pt_end_scaled.copy()
            pt_start[2] /= frame_scale
            pt_end[2] /= frame_scale
            
            if pt_start[2] > pt_end[2]:
                # Ensure start is temporally before end
                pt_start, pt_end = pt_end, pt_start

            # Check angular velocity in px/frame before accepting the line
            df = pt_end[2] - pt_start[2]
            if df > 1e-6:
                spatial_dist = math.sqrt((pt_end[0] - pt_start[0])**2 + (pt_end[1] - pt_start[1])**2)
                ang_vel_pxf = spatial_dist / df
            else:
                ang_vel_pxf = 0.0

            # If the line is too slow, skip it but still remove points below
            if min_ang_vel_px_per_frame is not None and ang_vel_pxf < min_ang_vel_px_per_frame:
                if debug:
                    print(f"Rejected slow line spanning {pt_start[2]:.1f}-{pt_end[2]:.1f} frames, "
                          f"{len(best_segment_indices)} pts, "
                          f"ang_vel={ang_vel_pxf:.2f} px/f < min {min_ang_vel_px_per_frame:.2f} px/f")
            else:
                found_lines.append((tuple(pt_start), tuple(pt_end), points[best_segment_indices]))
                if debug:
                    print(f"Found line spanning {pt_start[2]:.1f}-{pt_end[2]:.1f} frames, "
                          f"{len(best_segment_indices)} pts, ang_vel={ang_vel_pxf:.2f} px/f")
                
            # Remove points within a buffer zone around the line segment
            t_buffer = (t_max - t_min)*0.1 # 10% buffer
            pool_indices = np.where(available_mask)[0]
            pool_scaled = scaled_points[pool_indices]
            
            ref_dists = pointToLineDist3D(pool_scaled, mean_pt, ref_direction)
            ref_t = np.dot(pool_scaled - mean_pt, ref_direction)
            
            to_remove_local = (ref_dists < dist_thresh*1.5) & \
                              (ref_t >= t_min - t_buffer) & \
                              (ref_t <= t_max + t_buffer)
                              
            to_remove_global = pool_indices[to_remove_local]
            available_mask[to_remove_global] = False
            
        else:
            consecutive_failures += 1
            if debug:
                print(f"Failed to find line (Attempt {consecutive_failures}/{MAX_FAILURES})")

    return found_lines


def selectClosestLine(found_lines, ref_line, ref_has_frames=True, weights=(1.0, 1.0, 1.0), debug=False):
    """
    Selects the best matching line from 'found_lines' against 'ref_line'.
    
    Arguments:
        found_lines: [list] List of lines, each is (start_pt, end_pt) tuple of 3D coords.
        ref_line: [tuple] The reference line.
                 If ref_has_frames=True, it's ((x1, y1, f1), (x2, y2, f2)).
                 If ref_has_frames=False, it's (x1, y1, x2, y2) representing a 2D segment.
        ref_has_frames: [bool] Whether reference line has time dimension.
        weights: [tuple] Weights for (location_diff, orientation_diff, frame_diff).
        
    Return:
        best_line: Return the closest found line segment, or None if list is empty.
    """
    if not found_lines:
        return None

    if ref_has_frames:
        ref_start, ref_end = np.array(ref_line[0]), np.array(ref_line[1])
        ref_mid = (ref_start + ref_end)/2.0
        # 3D vector
        ref_vec = ref_end - ref_start
        ref_len = np.linalg.norm(ref_vec)
        ref_dir = ref_vec/ref_len if ref_len > 0 else np.array([1, 0, 0])
        # 2D direction for orientation metric
        ref_dir_2d = ref_vec[:2]
        ref_len_2d = np.linalg.norm(ref_dir_2d)
        ref_dir_2d = ref_dir_2d/ref_len_2d if ref_len_2d > 0 else np.array([1, 0])
    else:
        x1, y1, x2, y2 = ref_line
        ref_start = np.array([x1, y1])
        ref_end = np.array([x2, y2])
        ref_mid = (ref_start + ref_end)/2.0
        ref_vec = ref_end - ref_start
        ref_len = np.linalg.norm(ref_vec)
        ref_dir_2d = ref_vec/ref_len if ref_len > 0 else np.array([1, 0])

    if debug:
        print("\n--- 3D Line Similarity Search ---")
        if ref_has_frames:
            ref_start, ref_end = np.array(ref_line[0]), np.array(ref_line[1])
            df_ref = ref_end[2] - ref_start[2]
            if abs(df_ref) > 1e-6:
                mx_ref = (ref_end[0] - ref_start[0])/df_ref
                my_ref = (ref_end[1] - ref_start[1])/df_ref
                cx_ref = ref_start[0] - mx_ref*ref_start[2]
                cy_ref = ref_start[1] - my_ref*ref_start[2]
                print(f"Target line: ({ref_start[0]:.2f}, {ref_start[1]:.2f}, {ref_start[2]:.2f}) -> ({ref_end[0]:.2f}, {ref_end[1]:.2f}, {ref_end[2]:.2f})")
                print(f"  dx/df: {mx_ref:.2f}, dy/df: {my_ref:.2f}, x0: {cx_ref:.2f}, y0: {cy_ref:.2f}")
            else:
                print(f"Target line: {ref_start} -> {ref_end} (Static/Zero frame span)")
        else:
            x1, y1, x2, y2 = ref_line
            print(f"Target line (2D): ({x1:.2f}, {y1:.2f}) -> ({x2:.2f}, {y2:.2f})")

    best_line = None
    min_score = float('inf')

    for i, line in enumerate(found_lines):
        cand_start = np.array(line[0])
        cand_end = np.array(line[1])
        cand_mid = (cand_start + cand_end)/2.0
        cand_vec = cand_end - cand_start
        
        cand_len_3d = np.linalg.norm(cand_vec)
        cand_dir = cand_vec/cand_len_3d if cand_len_3d > 0 else np.array([1, 0, 0])
        
        cand_dir_2d = cand_vec[:2]
        cand_len_2d = np.linalg.norm(cand_dir_2d)
        cand_dir_2d = cand_dir_2d/cand_len_2d if cand_len_2d > 0 else np.array([1, 0])

        # 1. Location penalty: distance between midpoints (2D or 3D depending on ref)
        if ref_has_frames:
            loc_diff = np.linalg.norm(cand_mid - ref_mid)
        else:
            loc_diff = np.linalg.norm(cand_mid[:2] - ref_mid)

        # 2. Orientation penalty: 1 - |cos(theta)| in 2D
        # This ignores directionality (180 deg opposite is okay)
        cos_theta = abs(np.dot(cand_dir_2d, ref_dir_2d))
        orient_diff = 1.0 - cos_theta

        # 3. Frame offset penalty (only if ref has frames)
        frame_diff = 0.0
        if ref_has_frames:
            # Difference in start frame or velocity could be checked.
            # Easiest is average frame distance
            frame_diff = abs(cand_mid[2] - ref_mid[2])

        score = weights[0]*loc_diff + weights[1]*orient_diff*100.0 + weights[2]*frame_diff
        
        if debug:
            df_cand = cand_end[2] - cand_start[2]
            if abs(df_cand) > 1e-6:
                mx_cand = (cand_end[0] - cand_start[0])/df_cand
                my_cand = (cand_end[1] - cand_start[1])/df_cand
                cx_cand = cand_start[0] - mx_cand*cand_start[2]
                cy_cand = cand_start[1] - my_cand*cand_start[2]
                
                print(f"  Candidate {i}: ({cand_start[0]:.2f}, {cand_start[1]:.2f}, {cand_start[2]:.2f}) -> ({cand_end[0]:.2f}, {cand_end[1]:.2f}, {cand_end[2]:.2f})")
                print(f"    dx/df: {mx_cand:.2f}, dy/df: {my_cand:.2f}, x0: {cx_cand:.2f}, y0: {cy_cand:.2f}")
                print(f"    loc_diff={loc_diff:.2f}, orient_diff={orient_diff:.4f}, frame_diff={frame_diff:.2f}, score={score:.2f}")
            else:
                print(f"  Candidate {i}: Static/Zero frame span. score={score:.2f}")
            
        if score < min_score:
            min_score = score
            best_line = line

    if debug and best_line is not None:
        print(f"--- Best Candidate Choose: {best_line[0]} -> {best_line[1]} (Score={min_score:.2f})")

    return best_line


def stitch3DLines(found_lines, ref_line, ref_has_frames=True, vect_angle_thresh=10.0, dist_thresh=2.0, 
                  frame_scale=None, debug=False):
    """
    Selects the best matching line (seed) from RANSAC results and then 'stitches' other segments 
    that are collinear extensions of the same track.
    
    Arguments:
        found_lines: [list] List of formatted lines: [(start_pt, end_pt, pts_count, quality, f_min, f_max), ...]
        ref_line: [tuple] The reference line (2D or 3D).
        ref_has_frames: [bool] Whether reference line has time dimension.
        vect_angle_thresh: [float] Maximum 3D angle (degrees) to merge extensions.
        dist_thresh: [float] Maximum 3D distance to the infinite line to merge extensions.
        frame_scale: [float] Normalization ratio for frame dimension. If None, calculated from first found line.
        debug: [bool] Verbose output.
        
    Return:
        stitched_line: A single merged line segment in the same format as found_lines.
    """
    if not found_lines:
        return None

    # 1. Find the seed line
    seed_line = selectClosestLine(found_lines, ref_line, ref_has_frames=ref_has_frames, debug=debug)
    if seed_line is None:
        return None

    if len(found_lines) == 1:
        return seed_line

    # Setup frame scaling for 3D distance checks
    if frame_scale is None:
        # Estimate frame scale if not provided
        # Use a default 100/1 ratio if we can't estimate well
        frame_scale = 1.0
        for l in found_lines:
            dx = l[1][0] - l[0][0]
            dy = l[1][1] - l[0][1]
            df = l[1][2] - l[0][2]
            if abs(df) > 1:
                frame_scale = math.sqrt(dx**2 + dy**2) / abs(df)
                break

    def get_scaled_line(line):
        p1 = np.array(line[0])
        p2 = np.array(line[1])
        p1[2] *= frame_scale
        p2[2] *= frame_scale
        return p1, p2

    seed_p1_sc, seed_p2_sc = get_scaled_line(seed_line)
    seed_vec_sc = seed_p2_sc - seed_p1_sc
    seed_len_sc = np.linalg.norm(seed_vec_sc)
    seed_dir_sc = seed_vec_sc / seed_len_sc if seed_len_sc > 0 else np.array([1, 0, 0])

    stitched_segments = [seed_line]
    other_lines = [l for l in found_lines if l is not seed_line]
    
    if debug:
        print(f"\n--- 3D Line Stitching (Seed: {seed_line[0]} -> {seed_line[1]}) ---")
        print(f"  Frame scale: {frame_scale:.4f}")

    for cand in other_lines:
        cand_p1_sc, cand_p2_sc = get_scaled_line(cand)
        cand_vec_sc = cand_p2_sc - cand_p1_sc
        cand_len_sc = np.linalg.norm(cand_vec_sc)
        cand_dir_sc = cand_vec_sc / cand_len_sc if cand_len_sc > 0 else np.array([1, 0, 0])
        
        # Check directionality (3D angle)
        cos_theta = abs(np.dot(cand_dir_sc, seed_dir_sc))
        angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        
        if angle > vect_angle_thresh:
            if debug:
                print(f"  Candidate {cand[0]} rejected: angle {angle:.1f} > {vect_angle_thresh}")
            continue
            
        # Check distance to infinite line (using scaled coords)
        d1 = pointToLineDist3D(cand_p1_sc.reshape(1,3), seed_p1_sc, seed_dir_sc)[0]
        d2 = pointToLineDist3D(cand_p2_sc.reshape(1,3), seed_p1_sc, seed_dir_sc)[0]
        
        if d1 > dist_thresh or d2 > dist_thresh:
            if debug:
                print(f"  Candidate {cand[0]} rejected: dists ({d1:.1f}, {d2:.1f}) > {dist_thresh}")
            continue
            
        stitched_segments.append(cand)
        if debug:
            print(f"  Candidate {cand[0]} -> {cand[1]} merged! (angle={angle:.1f}, dists={d1:.1f}, {d2:.1f})")

    if len(stitched_segments) == 1:
        return seed_line

    # Combine all segments
    all_starts = np.array([l[0] for l in stitched_segments])
    all_ends = np.array([l[1] for l in stitched_segments])
    
    f_min = min(np.min(all_starts[:, 2]), np.min(all_ends[:, 2]))
    f_max = max(np.max(all_starts[:, 2]), np.max(all_ends[:, 2]))
    
    total_pts = sum(l[2] for l in stitched_segments)
    avg_quality = sum(l[3] for l in stitched_segments) / len(stitched_segments)
    
    # Use the seed's linear model to project endpoints back to unscaled space
    # seed_dir_sc is in scaled space. We need dx/df and dy/df in unscaled space.
    # dx = seed_vec_sc[0], dy = seed_vec_sc[1], df = seed_vec_sc[2]/frame_scale
    seed_p1 = np.array(seed_line[0])
    seed_p2 = np.array(seed_line[1])
    diff = seed_p2 - seed_p1
    
    if abs(diff[2]) > 1e-6:
        m_x = diff[0] / diff[2]
        m_y = diff[1] / diff[2]
        
        x_min = seed_p1[0] + (f_min - seed_p1[2]) * m_x
        y_min = seed_p1[1] + (f_min - seed_p1[2]) * m_y
        
        x_max = seed_p1[0] + (f_max - seed_p1[2]) * m_x
        y_max = seed_p1[1] + (f_max - seed_p1[2]) * m_y
        
        new_start = (x_min, y_min, f_min)
        new_end = (x_max, y_max, f_max)
    else:
        new_start = (seed_p1[0], seed_p1[1], f_min)
        new_end = (seed_p1[0], seed_p1[1], f_max)

    return [new_start, new_end, total_pts, avg_quality, int(round(f_min)), int(round(f_max))]


def find3DLines(stripe_points, current_time, config, fireball_detection=False):
    """
    Wrapper for finding 3D lines that mimics the IO format of the older Grouping3D 
    approach. Designed as a drop-in replacement for Detection logic.
    """
    # Compute minimum angular velocity in px/frame from config
    # ang_vel_min is in deg/s, convert to px/frame:
    #   px/s = (deg/s) / scale, where scale = avg(fov_h/height, fov_w/width)
    #   px/frame = px/s / fps
    scale = (config.fov_h / float(config.height) + config.fov_w / float(config.width)) / 2.0
    if scale > 0 and config.fps > 0:
        min_ang_vel_px_per_frame = config.ang_vel_min / scale / config.fps
    else:
        min_ang_vel_px_per_frame = None

    lines = findLines3D(
        points=stripe_points,
        max_lines=config.ransac3d_max_lines,
        min_points=config.ransac3d_min_points,
        dist_thresh=config.ransac3d_distance_thresh,
        max_gap_frame=config.ransac3d_max_gap_frame,
        max_gap_spatial=config.ransac3d_max_gap_spatial,
        min_frames=config.ransac3d_min_frames,
        frame_scale=None,
        img_w=config.width,
        img_h=config.height,
        max_iterations=config.ransac3d_iterations,
        min_ang_vel_px_per_frame=min_ang_vel_px_per_frame,
        debug=False
    )
    
    formatted_lines = []
    
    for line in lines:
        start_pt, end_pt, _ = line
        f_min = min(start_pt[2], end_pt[2])
        f_max = max(start_pt[2], end_pt[2])
        
        # Estimate points length roughly to satisfy formatting
        pts_count = int(f_max - f_min) + 1 
        
        # Old DL format: [(x1,y1,z1), (x2,y2,z2), counter, quality, f_first, f_last]
        formatted_line = [
            start_pt, 
            end_pt, 
            pts_count,
            1.0, 
            int(round(f_min)), 
            int(round(f_max))
        ]
        formatted_lines.append(formatted_line)
        
    return formatted_lines


def getAllPoints(point_list, x1, y1, z1, x2, y2, z2, config, fireball_detection=False):
    """
    Drop-in replacement for Grouping3D.getAllPoints.
    Extracts all points from point_list that fall closely on the line segment
    defined by (x1, y1, z1) to (x2, y2, z2). 
    """
    points = np.asarray(point_list).astype(np.float64)
    if len(points) == 0:
        return np.array([])
        
    p1 = np.array([x1, y1, z1])
    p2 = np.array([x2, y2, z2])
    
    direction = p2 - p1
    dir_norm = np.linalg.norm(direction)
    if dir_norm < 1e-6:
        return np.array([])
    direction /= dir_norm
    
    # Distance threshold uses 3D RANSAC params if detection, otherwise standard thresh
    dist_thresh = config.ransac3d_distance_thresh if not fireball_detection else config.distance_threshold
    
    img_diag = np.sqrt(config.width**2 + config.height**2)
    frame_scale = img_diag/256.0
    
    scaled_p1 = p1.copy()
    scaled_p1[2] *= frame_scale
    scaled_p2 = p2.copy()
    scaled_p2[2] *= frame_scale
    
    scaled_dir = scaled_p2 - scaled_p1
    sc_dir_norm = np.linalg.norm(scaled_dir)
    if sc_dir_norm > 0:
        scaled_dir /= sc_dir_norm
    else:
        scaled_dir = np.array([1.0, 0.0, 0.0])
        
    scaled_pts = points.copy()
    scaled_pts[:, 2] *= frame_scale
    
    dists = pointToLineDist3D(scaled_pts, scaled_p1, scaled_dir)
    inlier_mask = dists < dist_thresh
    
    if np.sum(inlier_mask) == 0:
        return np.array([])
        
    t_vals = np.dot(scaled_pts[inlier_mask] - scaled_p1, scaled_dir)
    t_end = np.dot(scaled_p2 - scaled_p1, scaled_dir)
    
    t_min = min(0.0, t_end)
    t_max = max(0.0, t_end)
    
    buffer = 5.0*frame_scale
    t_mask = (t_vals >= t_min - buffer) & (t_vals <= t_max + buffer)
    
    final_inliers = points[inlier_mask][t_mask]
    return final_inliers

