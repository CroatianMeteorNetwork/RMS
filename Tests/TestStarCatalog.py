""" Tests for the GMN star catalog format detection. """

from __future__ import print_function, division, absolute_import

import pytest

np = pytest.importorskip("numpy")

from RMS.Formats.StarCatalog import gmnCatalogDtype, GMN_CATALOG_DTYPE_V1, GMN_CATALOG_DTYPE_V2


def testGmnCatalogDtypeV1():
    """ 18 declared columns select the legacy v1 layout. """

    assert gmnCatalogDtype(18) is GMN_CATALOG_DTYPE_V1
    assert len(GMN_CATALOG_DTYPE_V1.names) == 18


def testGmnCatalogDtypeV2():
    """ 20 declared columns select the v2 layout with the extra name columns. """

    assert gmnCatalogDtype(20) is GMN_CATALOG_DTYPE_V2
    assert len(GMN_CATALOG_DTYPE_V2.names) == 20


@pytest.mark.parametrize("num_columns", [0, 17, 19, 21, 25])
def testGmnCatalogDtypeUnknownRaises(num_columns):
    """ Any other column count is an unknown format and must fail loudly instead of misdecoding. """

    with pytest.raises(ValueError):
        gmnCatalogDtype(num_columns)
