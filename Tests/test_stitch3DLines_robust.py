import numpy as np
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from RMS.Routines.LineFinder3D import stitch3DLines

def test_robust_stitch():
    # User Case Segments
    # Each: (start_pt, end_pt, pts_count, quality, f_min, f_max)
    
    # 0: (1651.68, 511.43, 316.86) -> (1547.29, 1080.37, 625.84) [309 pts, f:317-626]
    l0 = [(1651.68, 511.43, 316.86), (1547.29, 1080.37, 625.84), 309, 1.0, 317, 626]
    
    # 1: (1745.64, 7.86, 79.92) -> (1653.35, 507.45, 316.86) [237 pts, f:80-317]
    l1 = [(1745.64, 7.86, 79.92), (1653.35, 507.45, 316.86), 237, 1.0, 80, 317]
    
    # 2 (Seed): (1656.70, 482.05, 307.90) -> (1623.76, 656.92, 401.00) [94 pts, f:308-401]
    l2 = [(1656.70, 482.05, 307.90), (1623.76, 656.92, 401.00), 94, 1.0, 308, 401]
    
    # 3: (1580.93, 895.95, 531.00) -> (1548.30, 1073.54, 628.93) [98 pts, f:531-629]
    l3 = [(1580.93, 895.95, 531.00), (1548.30, 1073.54, 628.93), 98, 1.0, 531, 629]
    
    found_lines = [l0, l1, l2, l3]
    
    # Seed matches 2nd segment best
    ref_line = (1656.70, 482.05, 1623.76, 656.92)
    
    print("\nTesting Robust Stitching (User's Case)...")
    # frame_scale from log was 8.6051
    # User's threshold 15.0
    stitched = stitch3DLines(found_lines, ref_line, ref_has_frames=False, frame_scale=8.6051, dist_thresh=15.0, debug=True)
    
    if stitched:
        print(f"\nFinal Stitched Line Frames: {stitched[4]} to {stitched[5]}")
        print(f"Total points: {stitched[2]} (Expected: {309+237+94+98} = 738)")
        
        # Verify frames cover 80 to 629
        assert stitched[4] == 80
        assert stitched[5] == 629
        assert stitched[2] == 738
        print("\nTEST PASSED: Long track successfully stitched!")
    else:
        print("\nTEST FAILED: No stitched line found.")

if __name__ == "__main__":
    test_robust_stitch()
