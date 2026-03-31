import numpy as np
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from RMS.Routines.LineFinder3D import stitch3DLines

def test_stitch():
    # Segment 1: Frame 10 to 50, (100, 100) to (140, 140)
    # dx/df = 1, dy/df = 1
    l1 = [
        (100.0, 100.0, 10.0), 
        (140.0, 140.0, 50.0), 
        41, 1.0, 10, 50
    ]
    
    # Segment 2: Frame 60 to 100, (150, 150) to (190, 190)
    # dx/df = 1, dy/df = 1
    # This is a direct extension of l1
    l2 = [
        (150.0, 150.0, 60.0), 
        (190.0, 190.0, 100.0), 
        41, 1.0, 60, 100
    ]

    # Segment 3: Frame 110 to 130, (203.5, 203.5, 110) to (223.5, 223.5, 130)
    # This has a 3.5px offset from the line y=x
    l3_offset = [
        (203.5, 203.5, 110.0),
        (223.5, 223.5, 130.0),
        21, 1.0, 110, 130
    ]
    
    # Segment 4: Some noise line far away
    l4_noise = [
        (500.0, 500.0, 10.0), 
        (510.0, 510.0, 20.0), 
        11, 0.5, 10, 20
    ]
    
    found_lines = [l1, l2, l3_offset, l4_noise]
    
    # Reference line (2D) that matches the meteor track
    ref_line = (100.0, 100.0, 200.0, 200.0)
    
    print("\nTesting Stitching (with 5.0 threshold)...")
    stitched = stitch3DLines(found_lines, ref_line, ref_has_frames=False, dist_thresh=5.0, debug=True)
    
    if stitched:
        print(f"\nStitched Line: {stitched[0]} -> {stitched[1]}")
        print(f"Frames: {stitched[4]} to {stitched[5]}")
        print(f"Total points: {stitched[2]}")
        
        # Verify frames are 10 to 130
        assert stitched[4] == 10
        assert stitched[5] == 130
        # Verify points are 41 + 41 + 21 = 103
        assert stitched[2] == 103
        print("\nTEST PASSED!")
    else:
        print("\nTEST FAILED: No stitched line found.")

if __name__ == "__main__":
    test_stitch()
