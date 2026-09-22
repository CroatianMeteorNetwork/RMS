""" KD-tree based star matching. Replaces the brute-force Cython matchStars with
    scipy.spatial.cKDTree for O(N log M) performance instead of O(N*M).
"""

# The MIT License

# Copyright (c) 2016 Denis Vida

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from __future__ import print_function, division, absolute_import

import numpy as np
from scipy.spatial import cKDTree


def matchStars(stars_list, cat_x_array, cat_y_array, cat_good_indices, max_radius):
    """ Match image stars to catalog stars using a KD-tree for fast nearest-neighbour lookup.

        Matching semantics are the same as the brute-force Cython matchStars this replaces: every image
        star is independently assigned its nearest catalog star, so the matching is NOT one-to-one - two
        image stars may claim the same catalog star. Callers (matchStarsResiduals) only use the pairs to
        compute residuals and a match count, so duplicates are tolerated. The only difference to the
        Cython version is that a match exactly at max_radius is accepted (cKDTree's
        distance_upper_bound is inclusive) where the old code used a strict "<"; this is a measure-zero
        boundary case.

    Arguments:
        stars_list: [ndarray] (N, 4+) array of detected stars, columns (y, x, ...).
        cat_x_array: [ndarray] Full catalog X coordinates in image space.
        cat_y_array: [ndarray] Full catalog Y coordinates in image space.
        cat_good_indices: [ndarray] Indices into cat_x/cat_y of valid catalog stars.
        max_radius: [float] Maximum match distance in pixels.

    Return:
        matched_indices: [ndarray] (K, 3) array of (image_star_index, catalog_star_index, distance).
    """

    # Nothing to match against, or nothing to match
    if len(cat_good_indices) == 0 or len(stars_list) == 0:
        return np.empty((0, 3), dtype=np.float64)

    # Build the KD-tree from the catalog star pixel positions
    cat_coords = np.column_stack([cat_x_array[cat_good_indices], cat_y_array[cat_good_indices]])
    tree = cKDTree(cat_coords)

    # Query the nearest catalog star for each image star (stars_list is stored as (y, x))
    img_coords = np.column_stack([stars_list[:, 1], stars_list[:, 0]])  # (x, y)
    dist, local_idx = tree.query(img_coords, k=1, distance_upper_bound=max_radius)

    # Keep only the stars that had a match within max_radius (unmatched queries return inf)
    matched = np.isfinite(dist)
    img_inds = np.where(matched)[0]
    cat_inds = cat_good_indices[local_idx[matched]]
    distances = dist[matched]

    # Pack the pairs in the same (image index, catalog index, distance) layout the Cython version used
    matched_indices = np.column_stack([img_inds, cat_inds, distances]).astype(np.float64)

    return matched_indices
