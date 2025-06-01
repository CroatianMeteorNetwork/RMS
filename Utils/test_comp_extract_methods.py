import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import scipy.optimize as opt

### Generate Synthetic Star Field ###
def generate_synthetic_image(shape=(100, 100), num_stars=10, noise_level=5):
    """Creates a synthetic star field with Gaussian PSFs."""
    np.random.seed(42)  # For reproducibility
    img = np.zeros(shape, dtype=np.float32)

    # Random star positions
    star_positions = np.random.uniform(10, shape[0] - 10, size=(num_stars, 2))

    # Add Gaussian stars
    for y, x in star_positions:
        y, x = int(y), int(x)
        sigma = 2.0  # Gaussian width
        for i in range(-5, 6):
            for j in range(-5, 6):
                if 0 <= y + i < shape[0] and 0 <= x + j < shape[1]:
                    img[y + i, x + j] += np.exp(-((i**2 + j**2) / (2 * sigma**2)))

    # Normalize and add noise
    img /= img.max()
    img = (img * 255).astype(np.float32)
    img += np.random.normal(0, noise_level, shape)  # Add Gaussian noise
    return img, star_positions

### Old Star Extraction Method ###
def extractStars_old(img):
    """Old star extraction method with convolve and max filtering."""
    img_filtered = ndimage.filters.convolve(img, weights=np.full((2, 2), 1.0 / 4))  # Mean filter
    
    # Local maxima detection
    data_max = ndimage.maximum_filter(img_filtered, 10)
    maxima = (img_filtered == data_max)
    
    # Extract star positions
    labeled, num_objects = ndimage.label(maxima)
    raw_positions = np.array(ndimage.center_of_mass(img, labeled, range(1, num_objects + 1)))
    
    # Fit PSF (Gaussian) to refine coordinates
    fitted_positions = fitPSF(img, raw_positions)
    
    return raw_positions, fitted_positions

### New Star Extraction Method ###
def extractStars_new(img):
    """New star extraction method with median background subtraction."""
    img_median = np.median(img)
    img_filtered = ndimage.filters.convolve(img, weights=np.full((2, 2), 1.0 / 4))  # Mean filter
    
    # Local maxima detection
    data_max = ndimage.maximum_filter(img_filtered, 10)
    maxima = (img_filtered == data_max)
    
    # Extract star positions
    labeled, num_objects = ndimage.label(maxima)
    raw_positions = np.array(ndimage.center_of_mass(img - img_median, labeled, range(1, num_objects + 1)))
    
    # Fit PSF (Gaussian) to refine coordinates
    fitted_positions = fitPSF(img, raw_positions)
    
    return raw_positions, fitted_positions

### PSF Fitting Function ###
def fitPSF(img, positions):
    """Fits a 2D Gaussian to refine star positions."""
    refined_positions = []
    
    for y, x in positions:
        y, x = int(y), int(x)
        segment = img[max(0, y - 5):y + 6, max(0, x - 5):x + 6]  # 11x11 region around the star

        # Create a coordinate grid
        y_grid, x_grid = np.indices(segment.shape)

        try:
            # Fit a Gaussian PSF
            initial_guess = (30.0, 5, 5, 1.0, 1.0, 0.0, np.mean(segment))
            popt, _ = opt.curve_fit(twoDGaussian, (y_grid, x_grid), segment.ravel(), p0=initial_guess, maxfev=200)
            refined_positions.append([y - 5 + popt[2], x - 5 + popt[1]])  # Adjust for segment offset
        except RuntimeError:
            refined_positions.append([y, x])  # Fallback to original position if fitting fails
    
    return np.array(refined_positions)

### 2D Gaussian Function ###
def twoDGaussian(params, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """Defines a 2D Gaussian function."""
    x, y = params
    a = (np.cos(theta)**2) / (2 * sigma_x**2) + (np.sin(theta)**2) / (2 * sigma_y**2)
    b = -(np.sin(2*theta)) / (4 * sigma_x**2) + (np.sin(2*theta)) / (4 * sigma_y**2)
    c = (np.sin(theta)**2) / (2 * sigma_x**2) + (np.cos(theta)**2) / (2 * sigma_y**2)
    g = offset + amplitude * np.exp(-(a * ((x - xo)**2) + 2*b*(x - xo)*(y - yo) + c * ((y - yo)**2)))
    return g.ravel()

### Compare Results ###
def compare_results(true_positions, raw_old, fitted_old, raw_new, fitted_new):
    """Compare raw detections and fitted positions for shifts."""
    
    def compute_shift(detected, reference):
        """Find nearest neighbors and compute shifts."""
        shifts = []
        for ref in reference:
            distances = np.linalg.norm(detected - ref, axis=1)
            nearest_idx = np.argmin(distances)
            shifts.append(detected[nearest_idx] - ref)
        return np.array(shifts)

    shift_raw_old = compute_shift(raw_old, true_positions)
    shift_fitted_old = compute_shift(fitted_old, true_positions)
    shift_raw_new = compute_shift(raw_new, true_positions)
    shift_fitted_new = compute_shift(fitted_new, true_positions)

    print("Mean shift (raw old method):", np.mean(shift_raw_old, axis=0))
    print("Mean shift (fitted old method):", np.mean(shift_fitted_old, axis=0))
    print("Mean shift (raw new method):", np.mean(shift_raw_new, axis=0))
    print("Mean shift (fitted new method):", np.mean(shift_fitted_new, axis=0))

    plt.scatter(true_positions[:, 1], true_positions[:, 0], marker='o', label="True Stars", color='green')
    plt.scatter(fitted_old[:, 1], fitted_old[:, 0], marker='x', label="Fitted Old", color='red')
    plt.scatter(fitted_new[:, 1], fitted_new[:, 0], marker='+', label="Fitted New", color='blue')
    
    plt.legend()
    plt.gca().invert_yaxis()
    plt.title("Comparison of Star Extraction")
    plt.show()

### Run Test ###
img, true_positions = generate_synthetic_image()
raw_old, fitted_old = extractStars_old(img)
raw_new, fitted_new = extractStars_new(img)

compare_results(true_positions, raw_old, fitted_old, raw_new, fitted_new)