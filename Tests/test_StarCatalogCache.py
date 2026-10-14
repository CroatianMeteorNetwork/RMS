""" Tests of the cache of the decompressed GMN star catalog (RMS.Formats.StarCatalog.loadGMNStarCatalog). """

from __future__ import print_function, division, absolute_import

import os
import struct
import zlib

import numpy as np

import RMS.Formats.StarCatalog as StarCatalog


# The bundled catalog, from which the catalogs of the test are made
CATALOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Catalogs',
                       'GMN_StarCatalog_LM9.0.bin')


def _writeCatalog(path, first, n):
    """ Write a GMN catalog of n rows of the bundled catalog, from the row first. """

    # The header (size, rows, columns and the column names) and the rows of the bundled catalog
    with open(CATALOG, 'rb') as f:
        header_size, num_rows, num_columns = struct.unpack('<III', f.read(12))
        names = f.read(header_size - 12)
        data = np.frombuffer(zlib.decompress(f.read()), dtype=StarCatalog.gmnCatalogDtype(num_columns),
                             count=num_rows)

    # The same header with the new number of rows, and the compressed rows
    with open(path, 'wb') as f:
        f.write(struct.pack('<III', header_size, n, num_columns) + names)
        f.write(zlib.compress(data[first:first + n].tobytes()))


def test_catalogs_of_the_same_name_in_different_directories(tmp_path):
    """ Two catalogs with the same file name in different directories (e.g. the configs of two cameras) are
        cached apart, each returning its own stars. """

    # Two catalogs of different stars, of the same name in two directories
    paths = []
    for i, first in enumerate((0, 500)):
        os.makedirs(str(tmp_path/str(i)))
        path = str(tmp_path/str(i)/'GMN_StarCatalog_LM9.0.bin')
        _writeCatalog(path, first, 300)
        paths.append(path)

    # The stars of each, loaded one after the other
    ra = []
    for path in paths:
        data = StarCatalog.loadGMNStarCatalog(path, catalog_file='GMN_StarCatalog_LM9.0.bin')[0]
        ra.append(np.sort(data[:, 0]))

    # Each its own stars, not the stars of the first one loaded
    assert len(ra[0]) == len(ra[1]) == 300
    assert not np.array_equal(ra[0], ra[1])
