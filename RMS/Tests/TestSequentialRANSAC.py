import sys
import unittest
import numpy as np
import math
from RMS.Routines import SequentialRANSAC

SHOW_PLOT = False

class TestSequentialRANSAC(unittest.TestCase):

    def setUp(self):
        self.img_w = 128
        self.img_h = 96
        self.img = np.zeros((self.img_h, self.img_w), dtype=np.uint8)

    def draw_line(self, rho, theta, value=255):
        """ Helper to draw a line on self.img """
        # rho = x*cos(theta) + y*sin(theta)
        # Using image centered coordinates for rho/theta
        x0 = self.img_w / 2.0
        y0 = self.img_h / 2.0
        
        theta_rad = np.radians(theta)
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)
        
        for y in range(self.img_h):
            for x in range(self.img_w):
                xc = x - x0
                yc = y - y0
                dist = abs(xc*cos_t + yc*sin_t - abs(rho))
                if dist < 1.0:
                    self.img[y, x] = value

    def plot_results(self, lines, title):
        if not SHOW_PLOT:
            return
            
        import matplotlib.pyplot as plt
            
        plt.figure()
        plt.title(title)
        plt.imshow(self.img, cmap='gray')
        
        # Plot points
        y_idxs, x_idxs = np.where(self.img > 0)
        plt.scatter(x_idxs, y_idxs, c='red', s=1)
        
        # Plot lines
        for line in lines:
            rho = line[0]
            theta = line[1]
            
            # If extent is provided
            if len(line) == 6:
                _, _, x1, y1, x2, y2 = line
                plt.plot([x1, x2], [y1, y2], 'g-', linewidth=2)
                
                # Plot start/end points
                plt.plot(x1, y1, 'bo')
                plt.plot(x2, y2, 'yo')
                
            else:
                x0 = self.img_w / 2.0
                y0 = self.img_h / 2.0
                
                theta_rad = np.radians(theta)
                cos_t = np.cos(theta_rad)
                sin_t = np.sin(theta_rad)
                
                # Compute intersection with image boundaries for plotting
                pt1 = None
                pt2 = None
                
                # Check intersection with left/right borders (x=0, x=w)
                if abs(sin_t) > 1e-5:
                    # x = 0
                    x_c = 0 - x0
                    y_c = (rho - x_c*cos_t)/sin_t
                    y = y_c + y0
                    if 0 <= y <= self.img_h:
                        pt1 = (0, y)
                        
                    # x = w
                    x_c = self.img_w - x0
                    y_c = (rho - x_c*cos_t)/sin_t
                    y = y_c + y0
                    if 0 <= y <= self.img_h:
                        if pt1 is None: pt1 = (self.img_w, y)
                        else: pt2 = (self.img_w, y)

                # Check intersection with top/bottom borders (y=0, y=h)
                if abs(cos_t) > 1e-5:
                    # y = 0
                    y_c = 0 - y0
                    x_c = (rho - y_c*sin_t)/cos_t
                    x = x_c + x0
                    if 0 <= x <= self.img_w:
                         if pt1 is None: pt1 = (x, 0)
                         elif pt2 is None: pt2 = (x, 0)
                         
                    # y = h
                    y_c = self.img_h - y0
                    x_c = (rho - y_c*sin_t)/cos_t
                    x = x_c + x0
                    if 0 <= x <= self.img_w:
                         if pt1 is None: pt1 = (x, self.img_h)
                         elif pt2 is None: pt2 = (x, self.img_h)

                if pt1 is not None and pt2 is not None:
                    plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'g-', linewidth=2)
                
        plt.show()

    def test_single_line(self):
        # Create a line
        rho_gt = 50
        theta_gt = 45
        self.draw_line(rho_gt, theta_gt)
        
        lines = SequentialRANSAC.findLines(self.img, max_lines=1, min_pixels=10, 
                                           distance_thresh=2.0, min_line_length=10, max_gap=10)
        
        self.assertEqual(len(lines), 1)
        rho = lines[0][0]
        theta = lines[0][1]
        
        # Check if detected line is close to ground truth
        self.assertAlmostEqual(abs(rho), abs(rho_gt), delta=5)
        self.assertAlmostEqual(theta, theta_gt, delta=5)
        
        self.plot_results(lines, "test_single_line")

    def test_broken_line(self):
        # Draw a line with a gap
        # Just manually set points
        points = []
        for x in range(10, 40):
            points.append((x, x)) # Diagonal
        for x in range(50, 80): # Gap of 10px
            points.append((x, x))
            
        for x, y in points:
            self.img[y, x] = 255
            
        # If max_gap > 10, it should bridge it
        lines = SequentialRANSAC.findLines(self.img, max_lines=1, min_pixels=10, 
                                           distance_thresh=2.0, min_line_length=10, max_gap=15)
        
        self.assertEqual(len(lines), 1)
        
        # If max_gap < 10, it should typically find 2 lines (or 1 if lines=1)
        # Reset image
        self.img = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
        for x, y in points:
            self.img[y, x] = 255
            
        lines = SequentialRANSAC.findLines(self.img, max_lines=2, min_pixels=10, 
                                           distance_thresh=2.0, min_line_length=10, max_gap=5)
        
        self.assertTrue(len(lines) >= 1) 
        # RANSAC is random, so it might pick one segment or the other first.
        # Ideally it finds both if we ask for 2.
        
        self.plot_results(lines, "test_broken_line")

    def test_noise(self):
        # Draw line
        self.draw_line(0, 0)
        
        # Add noise
        noise = np.random.randint(0, 255, (self.img_h, self.img_w), dtype=np.uint8)
        mask = noise > 250 # sparse noise
        self.img[mask] = 255
        
        lines = SequentialRANSAC.findLines(self.img, max_lines=1, min_pixels=50, 
                                           distance_thresh=2.0, min_line_length=50, max_gap=20)
        
        self.assertEqual(len(lines), 1)
        self.assertEqual(len(lines), 1)
        rho = lines[0][0]
        theta = lines[0][1]
        self.assertAlmostEqual(rho, 0, delta=5)
        
        # Handle 0 vs 180 degree ambiguity
        diff = abs(theta - 0)
        diff = min(diff, abs(360 - diff))
        # Check against 180 deg too
        diff2 = abs(theta - 180)
        diff2 = min(diff2, abs(360 - diff2))
        diff3 = abs(theta - (-180))
        diff3 = min(diff3, abs(360 - diff3))
        
        best_diff = min(diff, diff2, diff3)

        self.assertLess(best_diff, 5)

        self.plot_results(lines, "test_noise")

    def test_line_extent(self):
        # Draw a specific line segment
        p1 = (20, 20)
        p2 = (80, 80)
        
        # Bresenham-like drawing or just simple loop
        for i in range(20, 81):
            self.img[i, i] = 255
            
        lines = SequentialRANSAC.findLines(self.img, max_lines=1, min_pixels=10,
                                           distance_thresh=2.0, min_line_length=10, max_gap=10)
                                           
        self.assertEqual(len(lines), 1)
        rho, theta, x1, y1, x2, y2 = lines[0]
        
        # Check coordinates. Note order might be swapped.
        dist1 = np.sqrt((x1-p1[0])**2 + (y1-p1[1])**2)
        dist2 = np.sqrt((x2-p2[0])**2 + (y2-p2[1])**2)
        
        dist1_swap = np.sqrt((x1-p2[0])**2 + (y1-p2[1])**2)
        dist2_swap = np.sqrt((x2-p1[0])**2 + (y2-p1[1])**2)
        
        match_direct = (dist1 < 5) and (dist2 < 5)
        match_swap = (dist1_swap < 5) and (dist2_swap < 5)
        
        self.assertTrue(match_direct or match_swap, f"Endpoints {(x1, y1)}, {(x2, y2)} do not match {p1}, {p2}")
        
        self.plot_results(lines, "test_line_extent")

    def test_blob_vs_outlier(self):
        # Create a dense blob that should dominate the fit
        # Blob centered at (50, 50), radius 3
        # 40-50 pixels
        blob_center = (50, 50)
        for y in range(47, 54):
            for x in range(47, 54):
                if (x-50)**2 + (y-50)**2 <= 9:
                    self.img[y, x] = 255
                    
        # Create a small outlier "tail" that pulls the naive fit
        # Gap of 10px, then 2 pixels
        # Placed such that it extends the length but is slightly off-center
        # Blob is at 50,50. Outlier at 70, 52.
        # Line from 50,50 to 70,52 has angle ~5.7 degrees.
        # But if we just fit the blob, angle should be 0 (or undefined/isotropic).
        # Wait, a circular blob has no preferred direction.
        # Let's make the blob slightly elliptical along X axis to give it a preferred direction (0 deg).
        for x in range(45, 56): # Length 11
             self.img[50, x] = 255
             self.img[49, x] = 255
             self.img[51, x] = 255
             
        # Now we have a thick horizontal line segment.
        # Add outlier at (70, 55). 
        # Ideal line is horizontal (theta=0/180 -> normal theta=90/270).
        # Outlier at (70, 55) (5px off axis) would pull the line if unweighted.
        
        self.img[55, 70] = 255
        self.img[55, 71] = 255
        
        # Standard unweighted fit might traverse from blob center to outlier.
        # Weighted fit should stick to the blob (horizontal).
        
        lines = SequentialRANSAC.findLines(self.img, max_lines=1, min_pixels=10, 
                                           distance_thresh=5.0, min_line_length=10, max_gap=20)
        
        self.assertEqual(len(lines), 1)
        rho, theta, _, _, _, _ = lines[0]
        
        # Check angle. Horizontal line -> normal is 0 or 180 (for x=0 line) or 90/270 (for y=0 line).
        # Line y=50. Normal is vertical. Theta should be 90 or 270.
        
        # Check if theta is close to 90 or 270
        diff1 = abs(theta - 90)
        diff2 = abs(theta - 270)
        best_diff = min(diff1, diff2)
        
        # If outlier pulled it, angle would be arctan(5/20) ~ 14 deg.
        # So theta would be 90 - 14 = 76.
        
        print(f"Blob test: Theta={theta:.2f}")
        
        # With density weighting, it should be very close to 90.
        self.assertLess(best_diff, 5, "Line was pulled by outlier!")
        
        self.plot_results(lines, "test_blob_vs_outlier")

    def test_thick_line_merge(self):
        # Draw two parallel lines close together (simulating a thick line)
        # Line 1: y = x (diagonal)
        for i in range(10, 90):
            self.img[i, i] = 255
        
        # Line 2: y = x + 3 (parallel, 3px away)
        for i in range(10, 90):
            y = i + 3
            x = i
            if 0 <= y < self.img_h and 0 <= x < self.img_w:
                self.img[y, x] = 255
        
        # With distance_thresh=5, these should be detected as overlapping
        # and merged into a single line
        lines = SequentialRANSAC.findLines(self.img, max_lines=5, min_pixels=10, 
                                           distance_thresh=5.0, min_line_length=30, max_gap=10)
        
        # Should merge into 1 line
        self.assertEqual(len(lines), 1, f"Expected 1 merged line, got {len(lines)}")
        
        self.plot_results(lines, "test_thick_line_merge")

if __name__ == '__main__':
    if '--plot' in sys.argv:
        SHOW_PLOT = True
        sys.argv.remove('--plot')
        
    unittest.main()
