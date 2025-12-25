import os
import numpy as np
import matplotlib.pyplot as plt
import copy
import unittest
from unittest.mock import MagicMock

try:
    from RMS.Utils.Astra import ASTRA
    module_prefix = 'RMS.Utils.Astra'
except ImportError:
    from Utils.Astra import ASTRA
    module_prefix = 'Utils.Astra'

class MockConfig:
    def __init__(self):
        self.bit_depth = 8
        self.gamma = 1.0
        self.stationID = "TEST"

class MockImageSequence:
    def __init__(self, frames):
        self.frames = frames
        self.current_frame = 0
        self.avepixel = np.mean(frames, axis=0) # Simple average
        self.frame_dt_dict = {i: 0.04*i for i in range(len(frames))} # Dummy timestamps

    def setFrame(self, frame_num):
        if 0 <= frame_num < len(self.frames):
            self.current_frame = frame_num
        else:
            raise ValueError(f"Frame {frame_num} out of bounds")

    def loadFrame(self):
        return self.frames[self.current_frame].astype(np.float32)

    def loadChunk(self):
        return self # Dummy return

def twoDGaussian(coords, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y, z = coords
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()

def meteorSimulate(img_w, img_h, frame_num, psf_sigma, speed=1):
    frames = np.zeros((frame_num, img_h, img_w), np.float64)
    
    # Linear meteor motion
    slope = 0.5 # dy/dx
    intercept = img_h/ 4
    
    start_x = img_w/ 4
    end_x = 3*img_w/ 4
    
    # Ground truth trajectory
    traj_x = np.linspace(start_x, end_x, frame_num)
    traj_y = slope*traj_x + intercept
    
    x_indices, y_indices = np.meshgrid(np.arange(0, img_w), np.arange(0, img_h))

    picks = []

    for i in range(frame_num):
        # Simulate moving meteor
        x = traj_x[i]
        y = traj_y[i]
        
        # Add to picks (every frame in range to ensure consecutive triplets)
        if 10 <= i <= frame_num - 10:
             picks.append({'frame': i, 'x_centroid': x, 'y_centroid': y})
        
        intens = 200.0
        
        # Generate PSF
        gauss_values = twoDGaussian((x_indices, y_indices, 255), intens, x, y, psf_sigma, psf_sigma, 0.0, 0.0)
        gauss_values = gauss_values.reshape(img_h, img_w)
        
        frames[i] += gauss_values
        
        # Add noise
        frames[i] += np.random.normal(0, 5, frames[i].shape)
        frames[i] += 30 # Background
        frames[i] = np.clip(frames[i], 0, 255)

    return frames, picks, (slope, intercept)

def test_astra_run():
    # Setup
    img_w, img_h = 640, 480
    n_frames = 50
    frames, picks_list, gt_params = meteorSimulate(img_w, img_h, n_frames, psf_sigma=1.5)
    
    img_obj = MockImageSequence(frames)
    
    pick_dict = {p['frame']: p for p in picks_list}
    
    dummy_config = MockConfig()
    
    astra_config = {
        'astra': {
            'star_thresh': 3,
            'min SNR': 5,
            'P_crop': 1.5,
            'sigma_init (px)': 1.5,
            'sigma_max': 1.2,
            'L_max': 1.5,
            'pick_offset': 'center'
        },
        'pso': {},
        'kalman': {}
    }
    
    # Initialize ASTRA
    astra = ASTRA(img_obj, pick_dict, astra_config, ".", dummy_config, dark=None, flat=None)
    
    # Disable plotting to avoid IO
    astra.save_animation = False
    astra.verbose = True # To see print output in logs
    
    # Mock Image module functions used in processImageData correction
    with unittest.mock.patch(f'{module_prefix}.Image') as mock_image:
        mock_image.gammaCorrectionImage.side_effect = lambda img, *args, **kwargs: img
        mock_image.applyDark.side_effect = lambda img, *args, **kwargs: img
        mock_image.applyFlat.side_effect = lambda img, *args, **kwargs: img
        
        print("Running processImageData...")
        astra.processImageData()

        print("Running cropAllMeteorFrames...")
        astra.cropAllMeteorFrames()
    
    print("Planned Trajectory Length:", len(astra.planned_trajectory))
    
    # Verification
    # Check if robust line params match ground truth (approx)
    vx, vy, x0, y0 = astra.robust_line_params
    estimated_slope = vy/vx
    gt_slope = gt_params[0]
    
    # Note: cv2.fitLine returns normalized vector. Slope is vy/vx.
    # We should handle vertical lines if necessary, but here slope is 0.5.
    
    print(f"Ground Truth Slope: {gt_slope}")
    print(f"Estimated Slope: {estimated_slope}")
    
    if abs(estimated_slope - gt_slope) > 0.1:
        print("FAIL: Slope estimation is too far off.")
    else:
        print("PASS: Slope estimation is accurate.")

    # Check kinematic constraints (Leash)
    # The planned trajectory should follow the GT line closely.
    max_dev = 0
    for pt in astra.planned_trajectory:
        px, py = pt
        expected_y = gt_slope*px + gt_params[1] # Approximate check using slope/intercept is tricky because x0/y0 form line equation
        
        # Better: distance to GT line y = mx + c => mx - y + c = 0
        dist = abs(gt_slope*px - py + gt_params[1])/ np.sqrt(gt_slope**2 + 1)
        max_dev = max(max_dev, dist)
        
    print(f"Max deviation from GT line: {max_dev:.4f} pixels")
    
    if max_dev > 5.0:
         print("FAIL: Trajectory deviates significantly from ground truth.")
    else:
         print("PASS: Trajectory follows ground truth.")

if __name__ == "__main__":
    test_astra_run()
