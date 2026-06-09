""" KD-tree based star matching. Replaces the brute-force Cython matchStars with
    scipy.spatial.cKDTree for O(N log M) performance instead of O(N*M).
"""

import numpy as np
from scipy.spatial import cKDTree


def matchStars(stars_list, cat_x_array, cat_y_array, cat_good_indices, max_radius):
    """ Match image stars to catalog stars using a KD-tree for fast nearest-neighbor lookup.

    Arguments:
        stars_list: [ndarray] (N, 4+) array of detected stars, columns (y, x, ...).
        cat_x_array: [ndarray] full catalog X coordinates in image space.
        cat_y_array: [ndarray] full catalog Y coordinates in image space.
        cat_good_indices: [ndarray] indices into cat_x/cat_y of valid catalog stars.
        max_radius: [float] maximum match distance in pixels.

    Return:
        matched_indices: [ndarray] (K, 3) array of (image_star_index, catalog_star_index, distance).
    """

    if len(cat_good_indices) == 0 or len(stars_list) == 0:
        return np.empty((0, 3), dtype=np.float64)

    # Build KD-tree from catalog star pixel positions
    cat_coords = np.column_stack([cat_x_array[cat_good_indices], cat_y_array[cat_good_indices]])
    tree = cKDTree(cat_coords)

    # Query nearest catalog star for each image star
    img_coords = np.column_stack([stars_list[:, 1], stars_list[:, 0]])  # (x, y)
    dist, local_idx = tree.query(img_coords, k=1, distance_upper_bound=max_radius)

    # Filter to stars that had a match within max_radius
    matched = np.isfinite(dist)
    img_inds = np.where(matched)[0]
    cat_inds = cat_good_indices[local_idx[matched]]
    distances = dist[matched]

    matched_indices = np.column_stack([img_inds, cat_inds, distances]).astype(np.float64)

    return matched_indices
