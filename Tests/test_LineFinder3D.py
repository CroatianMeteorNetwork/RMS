import unittest
import numpy as np
import sys

from RMS.Routines.LineFinder3D import fitLine3D, findLines3D, selectClosestLine

PLOT_RESULTS = False

def plot_result(title, points, lines, ref_line=None, best_line=None):
    if not PLOT_RESULTS:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots.")
        return
        
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    
    # Plot points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='k', marker='.', s=10, alpha=0.5, label='Points')
    
    # Plot found lines
    for i, line in enumerate(lines):
        color = 'b'
        lw = 2
        label = f'Line {i+1}' if i == 0 else None
        
        if best_line is not None and line == best_line:
            color = 'g'
            lw = 4
            label = 'Best Line'
            
        x_vals = [line[0][0], line[1][0]]
        y_vals = [line[0][1], line[1][1]]
        z_vals = [line[0][2], line[1][2]]
        ax.plot(x_vals, y_vals, z_vals, color=color, linewidth=lw, label=label)
        
    if ref_line is not None:
        x_vals = [ref_line[0][0], ref_line[1][0]]
        y_vals = [ref_line[0][1], ref_line[1][1]]
        z_vals = [ref_line[0][2], ref_line[1][2]]
        ax.plot(x_vals, y_vals, z_vals, color='r', linewidth=2, linestyle='--', label='Ref Line')
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Frame')
    ax.legend()
    plt.show()

class TestLineFinder3D(unittest.TestCase):
    
    def test_ideal_conditions(self):
        """ Test line extraction without noise """
        # Create perfect straight line along x, y, frame
        frames = np.arange(0, 50, 1)
        x = 100 + frames * 2
        y = 50 + frames * 1.5
        
        points = np.column_stack((x, y, frames))
        
        lines = findLines3D(points, max_lines=1, min_points=10, dist_thresh=2.0, 
                            max_gap_frame=3, max_gap_spatial=10.0, min_frames=10, 
                            frame_scale=1.0)
                            
        self.assertEqual(len(lines), 1)
        line = lines[0]
        
        # Verify starts and ends
        # Note: the endpoints might be slightly different due to PCA extending them 
        # based on min/max projected coordinates, but they should be very close.
        start_pt = line[0]
        end_pt = line[1]
        
        self.assertAlmostEqual(start_pt[2], 0, delta=1.0)
        self.assertAlmostEqual(end_pt[2], 49, delta=1.0)
        
        # Check coordinates 
        self.assertAlmostEqual(start_pt[0], 100, delta=1.0)
        self.assertAlmostEqual(end_pt[0], 100 + 49 * 2, delta=1.0)
        
        plot_result("Ideal Conditions", points, lines)

    def test_noise_and_confusing_lines(self):
        """ Test multiple noisy intersecting lines """
        np.random.seed(42)
        
        # Line 1: The target line
        f1 = np.arange(10, 60, 1)
        x1 = 50 + f1 * 1.0 + np.random.normal(0, 0.5, len(f1))
        y1 = 50 + f1 * 2.0 + np.random.normal(0, 0.5, len(f1))
        
        # Line 2: Intersecting confusing line
        f2 = np.arange(20, 70, 1)
        x2 = 120 - f2 * 1.5 + np.random.normal(0, 1.0, len(f2))
        y2 = 30 + f2 * 1.5 + np.random.normal(0, 1.0, len(f2))
        
        # Line 3: Short confusing line (might be rejected or found as 3rd)
        f3 = np.arange(30, 45, 1)
        x3 = 80 + f3 * 0.5 + np.random.normal(0, 0.5, len(f3))
        y3 = 90 - f3 * 0.5 + np.random.normal(0, 0.5, len(f3))
        
        # Random scatter noise
        noise_f = np.random.uniform(0, 100, 200)
        noise_x = np.random.uniform(0, 200, 200)
        noise_y = np.random.uniform(0, 200, 200)
        
        pts1 = np.column_stack((x1, y1, f1))
        pts2 = np.column_stack((x2, y2, f2))
        pts3 = np.column_stack((x3, y3, f3))
        pts_noise = np.column_stack((noise_x, noise_y, noise_f))
        
        all_pts = np.vstack((pts1, pts2, pts3, pts_noise))
        
        # Shuffle
        np.random.shuffle(all_pts)
        
        lines = findLines3D(all_pts, max_lines=5, min_points=10, dist_thresh=3.0, 
                            max_gap_frame=5, max_gap_spatial=15.0, min_frames=10, 
                            frame_scale=1.0, max_iterations=2000)
                            
        self.assertGreaterEqual(len(lines), 2)
        
        # Define reference of Line 1
        ref_line = ((50 + 35*1.0, 50 + 35*2.0, 35), (50 + 55*1.0, 50 + 55*2.0, 55))
        
        # Find closest
        best = selectClosestLine(lines, ref_line, ref_has_frames=True)
        self.assertIsNotNone(best)
        
        # Verify it matched Line 1 (approximate direction)
        vec_best = np.array(best[1]) - np.array(best[0])
        self.assertGreater(vec_best[0], 0) # dx is positive
        self.assertGreater(vec_best[1], 0) # dy is positive
        
        plot_result("Noise and Confusing Lines", all_pts, lines, ref_line=ref_line, best_line=best)

    def test_gap_bridging_and_splitting(self):
        """ Test bridging small gaps and splitting on large gaps """
        # Line with a small gap (bridges) and big gap (splits)
        # Seg 1: frames 0-10
        # Gap: frames 11-13 (size 3) -> should bridge if max_gap_frame=5
        # Seg 2: frames 14-30
        # Gap: frames 31-40 (size 10) -> should split if max_gap_frame=5
        # Seg 3: frames 41-60
        f_seg1 = np.arange(0, 11)
        f_seg2 = np.arange(14, 31)
        f_seg3 = np.arange(41, 61)
        f_all = np.concatenate((f_seg1, f_seg2, f_seg3))
        
        x_all = 10 + f_all * 2.0
        y_all = 20 + f_all * 0.5
        
        points = np.column_stack((x_all, y_all, f_all))
        
        lines = findLines3D(points, max_lines=5, min_points=10, dist_thresh=2.0, 
                            max_gap_frame=5, max_gap_spatial=15.0, min_frames=10, 
                            frame_scale=1.0)
                            
        # Should find 2 lines: Seg1+Seg2 (merged), and Seg3
        self.assertEqual(len(lines), 2)
        
        len_1 = lines[0][1][2] - lines[0][0][2]
        len_2 = lines[1][1][2] - lines[1][0][2]
        
        # One should be size ~30 (frames 0 to 30), other size ~19 (frames 41 to 60)
        lengths = sorted([len_1, len_2])
        self.assertAlmostEqual(lengths[0], 19, delta=1.0) # Seg3
        self.assertAlmostEqual(lengths[1], 30, delta=1.0) # Seg1+Seg2
        
        plot_result("Gap Bridging and Splitting", points, lines)

    def test_min_frames_constraint(self):
        """ Test that segments failing min_frames and min_points are rejected """
        # Valid length segment
        f_valid = np.arange(0, 20)
        x_valid = 10 + f_valid * 1.0
        y_valid = 10 + f_valid * 1.0
        
        # Too short span segment (only 5 frames, min_frames=10)
        f_short = np.arange(30, 35)
        # Make sure it has enough points (min_points=10)
        f_short = np.repeat(f_short, 3) 
        x_short = 50 + f_short * 1.0
        y_short = 50 - f_short * 1.0
        
        points = np.column_stack((np.concatenate((x_valid, x_short)), 
                                  np.concatenate((y_valid, y_short)), 
                                  np.concatenate((f_valid, f_short))))
                                  
        lines = findLines3D(points, max_lines=5, min_points=10, dist_thresh=2.0, 
                            max_gap_frame=5, max_gap_spatial=15.0, min_frames=10, 
                            frame_scale=1.0)
                            
        # Only the valid line should be found
        self.assertEqual(len(lines), 1)
        self.assertAlmostEqual(lines[0][0][2], 0, delta=1.0)
        
        plot_result("Min Frames Constraint", points, lines)

if __name__ == '__main__':
    if '--plots' in sys.argv:
        PLOT_RESULTS = True
        sys.argv.remove('--plots')
    unittest.main()
