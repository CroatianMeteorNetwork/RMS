""" Tests for the declared SciPy minimum version required by RMS.ExtractStars. """

from __future__ import print_function, division, absolute_import

import os
import re

import numpy as np
from scipy.spatial import cKDTree


def testRequirementsDeclaresScipyAtLeast16():
    """ requirements.txt must declare scipy >= 1.6 for cKDTree.query_pairs(output_type='ndarray'). """

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    req_path = os.path.join(repo_root, "requirements.txt")

    scipy_spec = None
    with open(req_path, "r") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].strip()
            if line.lower().startswith("scipy"):
                scipy_spec = line
                break

    assert scipy_spec is not None, "scipy entry missing from requirements.txt"

    match = re.search(r">=\s*(\d+)\.(\d+)", scipy_spec)
    assert match is not None, "Expected >=X.Y floor in {!r}".format(scipy_spec)

    major, minor = int(match.group(1)), int(match.group(2))
    assert (major, minor) >= (1, 6), "Expected scipy>=1.6, found {!r}".format(scipy_spec)


def testCKDTreeQueryPairsSupportsNdarrayOutput():
    """ Verify the runtime SciPy supports the ndarray output_type used in RMS.ExtractStars. """

    pts = np.array([[0.0, 0.0], [0.5, 0.0], [10.0, 10.0]], dtype=np.float64)
    pairs = cKDTree(pts).query_pairs(r=1.0, output_type='ndarray')
    assert isinstance(pairs, np.ndarray)
    assert pairs.shape == (1, 2)
    assert tuple(pairs[0]) == (0, 1)
