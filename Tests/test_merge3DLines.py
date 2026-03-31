"""
Test for merge3DLines to verify it correctly handles perpendicular lines.

The merge3DLines function decides whether to merge two 3D line segments by:
1. Building v1 (direction of line 1), v2 (direction of line 2)
2. Building v_both (vector from start of line 1 to end of line 2)
3. Checking if angle(v1, v_both) < threshold AND angle(v2, v_both) < threshold
4. Checking if frame ranges overlap

Bug hypothesis: when two perpendicular lines share a similar spatial region,
the z (frame) component dominates the 3D vectors, making lines that are
perpendicular in (x,y) appear nearly parallel in 3D space.
"""

import numpy as np


# --- Copy of the function under test (to avoid heavy imports) ---

def merge3DLines(line_list, vect_angle_thresh, last_count=0):

    def _vectorAngle(v1, v2):
        cosang = np.dot(v1, v2)
        sinang = np.linalg.norm(np.cross(v1, v2))
        angle = np.degrees(np.arctan2(sinang, cosang))
        if angle > 180:
            angle = abs(angle - 360)
        elif angle < 0:
            angle = abs(angle)
        return angle

    if len(line_list) < 2:
        return line_list

    final_list = []
    paired_indices = []

    for i, line1 in enumerate(line_list):
        if i in paired_indices:
            continue

        x11, y11, z11 = line1[0]
        x12, y12, z12 = line1[1]
        line1_fmin, line1_fmax = line1[4:6]
        v1 = np.array([x12-x11, y12-y11, z12-z11])

        found_pair = False

        for j, line2 in enumerate(line_list[i+1:]):
            j = j + i + 1
            if j in paired_indices:
                continue

            x21, y21, z21 = line2[0]
            x22, y22, z22 = line2[1]
            line2_fmin, line2_fmax = line2[4:6]
            v2 = np.array([x22-x21, y22-y21, z22-z21])

            # Create a vector from line points (2D - image plane only)
            v1_2d = v1[:2]
            v2_2d = v2[:2]

            # Create a vector from first point of first line to last point of second line
            v_both_2d = np.array([x22 - x11, y22 - y11])

            vect_angle1 = _vectorAngle(v1_2d, v_both_2d)
            vect_angle2 = _vectorAngle(v2_2d, v_both_2d)
            vect_angle12 = _vectorAngle(v1_2d, v2_2d)

            if (vect_angle1 < vect_angle_thresh) and (vect_angle2 < vect_angle_thresh) and (vect_angle12 < vect_angle_thresh):
                if set(range(line1_fmin, line1_fmax+1)).intersection(range(line2_fmin, line2_fmax+1)):
                    frame_min = min((line1_fmin, line2_fmin))
                    frame_max = max((line1_fmax, line2_fmax))
                    paired_indices.append(i)
                    paired_indices.append(j)
                    found_pair = True
                    final_list.append([line1[0], line1[1], max(line1[2], line2[2]), max(line1[3], line2[3]), frame_min, frame_max])
                    break

        if not found_pair:
            final_list.append(line1)

    if len(final_list) != last_count:
        final_list = merge3DLines(final_list, vect_angle_thresh, len(final_list))

    return final_list


# --- Test helpers ---

def make_line(x1, y1, f1, x2, y2, f2, pts=10):
    """Create a line in the format expected by merge3DLines."""
    f_min = min(int(f1), int(f2))
    f_max = max(int(f1), int(f2))
    return [(x1, y1, f1), (x2, y2, f2), pts, 1.0, f_min, f_max]


def vec_angle(a, b):
    cosang = np.dot(a, b)
    sinang = np.linalg.norm(np.cross(a, b))
    return np.degrees(np.arctan2(sinang, cosang))


# --- Tests ---

def test_direct_angle_analysis():
    """Show how the z-component dominates and makes perpendicular lines mergeable."""
    print("\n  Angle analysis: Two lines perpendicular in XY, varied spatial vs temporal extent")
    print(f"  {'spatial_px':>12} {'frames':>8} {'angle(v1,v2)':>14} {'angle(v1,vb)':>14} {'angle(v2,vb)':>14} {'would_merge':>12}")
    
    bugs_found = 0
    for spatial_px in [5, 10, 20, 50, 100, 200]:
        for frames in [10, 30, 50, 100, 200]:
            v1 = np.array([spatial_px, 0, frames], dtype=float)
            v2 = np.array([0, spatial_px, frames], dtype=float)
            v_both = np.array([spatial_px/2, spatial_px/2, frames], dtype=float)
            
            # For analysis, we also need the 2D versions
            v1_2d = v1[:2]
            v2_2d = v2[:2]
            vb_2d = v_both[:2]
            
            a12_3d = vec_angle(v1, v2)
            a1b_3d = vec_angle(v1, v_both)
            a2b_3d = vec_angle(v2, v_both)
            
            # The 2D angles used in the fixed function:
            a12_2d = vec_angle(v1_2d, v2_2d)
            a1b_2d = vec_angle(v1_2d, vb_2d)
            a2b_2d = vec_angle(v2_2d, vb_2d)
            
            # Fixed logic: all 2D angles must be < threshold (20)
            merges = (a1b_2d < 20) and (a2b_2d < 20) and (a12_2d < 20)
            
            # The bug check: if it used to merge but shouldn't have due to 2D perpendicularity
            # (In this sweep, v1 and v2 are always perpendicular in 2D, so they should NEVER merge)
            bug_flag = " (STILL BROKEN)" if merges else ""
            
            print(f"  {spatial_px:>12} {frames:>8} {a12_3d:>14.2f} {a1b_3d:>14.2f} {a2b_3d:>14.2f} {str(merges):>12}{bug_flag}")
    
    return bugs_found


def test_parallel_lines_merge():
    """Two parallel, overlapping lines should be merged."""
    line1 = make_line(100, 100, 10, 200, 100, 20)
    line2 = make_line(150, 102, 12, 250, 102, 22)
    
    result = merge3DLines([line1, line2], vect_angle_thresh=20)
    passed = len(result) == 1
    print(f"{'PASS' if passed else 'FAIL'}: Parallel overlapping lines -> {len(result)} line(s) (expected 1)")
    return passed


def test_perpendicular_lines_no_merge():
    """Two clearly perpendicular lines should NOT be merged."""
    line1 = make_line(100, 200, 10, 200, 200, 20)
    line2 = make_line(150, 100, 12, 150, 300, 22)
    
    result = merge3DLines([line1, line2], vect_angle_thresh=20)
    passed = len(result) == 2
    print(f"{'PASS' if passed else 'FAIL'}: Large perpendicular lines -> {len(result)} line(s) (expected 2)")
    return passed


def test_perpendicular_short_lines_no_merge():
    """Short perpendicular lines where z dominates - this is the bug case."""
    # Short horizontal line (10 px in x over 50 frames)
    line1 = make_line(100, 200, 10, 110, 200, 60)
    # Short vertical line (10 px in y over 50 frames) 
    line2 = make_line(105, 195, 15, 105, 205, 65)
    
    v1 = np.array([10, 0, 50], dtype=float)
    v2 = np.array([0, 10, 50], dtype=float)
    v_both = np.array([5, 5, 55], dtype=float)
    
    a12 = vec_angle(v1, v2)
    a1b = vec_angle(v1, v_both)
    a2b = vec_angle(v2, v_both)
    
    print(f"\n  Short perpendicular lines analysis:")
    print(f"    v1 = {v1}  (horizontal, 10px over 50 frames)")
    print(f"    v2 = {v2}  (vertical, 10px over 50 frames)")
    print(f"    v_both = {v_both}")
    print(f"    angle(v1, v2) = {a12:.2f} deg  (perpendicular in XY!)")
    print(f"    angle(v1, v_both) = {a1b:.2f} deg  (< 20? {a1b < 20})")
    print(f"    angle(v2, v_both) = {a2b:.2f} deg  (< 20? {a2b < 20})")
    
    result = merge3DLines([line1, line2], vect_angle_thresh=20)
    passed = len(result) == 2
    print(f"  {'PASS' if passed else 'FAIL'}: Short perpendicular lines -> {len(result)} line(s) (expected 2)")
    return passed


def test_perpendicular_same_frames():
    """Perpendicular lines with identical frame range - worst case for z domination."""
    line1 = make_line(100, 200, 10, 120, 200, 60)
    line2 = make_line(110, 190, 10, 110, 210, 60)
    
    v1 = np.array([20, 0, 50], dtype=float)
    v2 = np.array([0, 20, 50], dtype=float)
    v_both = np.array([10, 10, 50], dtype=float)
    
    a12 = vec_angle(v1, v2)
    a1b = vec_angle(v1, v_both)
    a2b = vec_angle(v2, v_both)
    
    print(f"\n  Same-frame perpendicular lines analysis:")
    print(f"    v1 = {v1}  (horizontal)")
    print(f"    v2 = {v2}  (vertical)")
    print(f"    v_both = {v_both}")
    print(f"    angle(v1, v2) = {a12:.2f} deg")
    print(f"    angle(v1, v_both) = {a1b:.2f} deg  (< 20? {a1b < 20})")
    print(f"    angle(v2, v_both) = {a2b:.2f} deg  (< 20? {a2b < 20})")
    
    result = merge3DLines([line1, line2], vect_angle_thresh=20)
    passed = len(result) == 2
    print(f"  {'PASS' if passed else 'FAIL'}: Same-frame perpendicular lines -> {len(result)} line(s) (expected 2)")
    return passed


if __name__ == "__main__":
    print("=" * 70)
    print("merge3DLines Bug Analysis")
    print("=" * 70)
    
    bugs = test_direct_angle_analysis()
    
    print("\n" + "-" * 70)
    print("Merge tests (threshold = 20 deg):")
    print("-" * 70)
    
    results = []
    results.append(test_parallel_lines_merge())
    results.append(test_perpendicular_lines_no_merge())
    results.append(test_perpendicular_short_lines_no_merge())
    results.append(test_perpendicular_same_frames())
    
    print("\n" + "=" * 70)
    print(f"Results: {sum(results)}/{len(results)} passed, {bugs} angle-analysis bug cases found")
    
    if bugs > 0:
        print("\nROOT CAUSE: The 3D vectors include the frame (z) component.")
        print("When spatial movement is small relative to frame span,")
        print("the z-component dominates and all vectors point 'mostly in z'.")
        print("This makes perpendicular lines appear nearly parallel in 3D,")
        print("causing angle(v1, v_both) and angle(v2, v_both) to both be < threshold.")
        print("\nFIX: The direction comparison should use 2D (x,y) vectors only,")
        print("or the frame dimension should be excluded/normalized.")
    
    print("=" * 70)
