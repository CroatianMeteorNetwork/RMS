""" Tests for the GMN star catalog format detection. """

from __future__ import print_function, division, absolute_import

import pytest

np = pytest.importorskip("numpy")

from RMS.Formats.StarCatalog import gmnCatalogDtype, GMN_CATALOG_DTYPE_V1, GMN_CATALOG_DTYPE_V2


def testGmnCatalogDtypeV1():
    """ 18 declared columns select the legacy v1 layout. """

    # The returned dtype must be the v1 object itself, not a copy with the same fields
    assert gmnCatalogDtype(18) is GMN_CATALOG_DTYPE_V1
    assert len(GMN_CATALOG_DTYPE_V1.names) == 18


def testGmnCatalogDtypeV2():
    """ 20 declared columns select the v2 layout with the extra name columns. """

    # The returned dtype must be the v2 object itself, not a copy with the same fields
    assert gmnCatalogDtype(20) is GMN_CATALOG_DTYPE_V2
    assert len(GMN_CATALOG_DTYPE_V2.names) == 20


@pytest.mark.parametrize("num_columns", [0, 17, 19, 21, 25])
def testGmnCatalogDtypeUnknownRaises(num_columns):
    """ Any other column count is an unknown format and must fail loudly instead of misdecoding. """

    # Silently falling back to one of the known layouts would misread every column
    with pytest.raises(ValueError):
        gmnCatalogDtype(num_columns)


def _writeCatalog(path, num_columns, payload):
    """ Write a GMN-style catalog file: header (size, rows, columns, names) followed by the payload. """

    import struct

    header_size = 64
    with open(path, 'wb') as f:
        f.write(struct.pack('<III', header_size, 1, num_columns))
        f.write(b'\0'*(header_size - 12))
        f.write(payload)


def _runLoadGMNCatalog(monkeypatch, tmp_path, full_name, real_loader_error=None):
    """ Run loadGMNCatalog with the fallback load and the download stubbed, return the calls made. """

    import RMS.Formats.StarCatalog as StarCatalog

    calls = {'downloads': 0}
    real_loader = StarCatalog.loadGMNStarCatalog

    def loader(file_path, catalog_file='', **kwargs):
        if catalog_file == 'fallback.bin':
            return 'fallback'
        if real_loader_error is not None:
            raise real_loader_error
        return real_loader(file_path, catalog_file=catalog_file, **kwargs)

    def download(url, dir_path, file_name):
        calls['downloads'] += 1
        return False

    monkeypatch.setattr(StarCatalog, 'loadGMNStarCatalog', loader)
    monkeypatch.setattr(StarCatalog, 'downloadCatalog', download)

    calls['result'] = StarCatalog.loadGMNCatalog(str(tmp_path), True, full_name, 'http://x', 'fallback.bin',
                                                 {})

    return calls


def testUnsupportedCatalogVersionIsKept(monkeypatch, tmp_path):
    """ An intact catalog with an unknown column count is a newer version: keep it, do not re-download. """

    import zlib

    full_name = 'future_version_catalog.bin'
    _writeCatalog(str(tmp_path/full_name), 25, zlib.compress(b'\0'*1000))

    calls = _runLoadGMNCatalog(monkeypatch, tmp_path, full_name)

    assert calls['result'] == 'fallback'
    assert calls['downloads'] == 0
    assert (tmp_path/full_name).exists()


def testCorruptCatalogWithBadHeaderIsRepaired(monkeypatch, tmp_path):
    """ A garbled header with data that does not decompress is corruption: delete and re-download. """

    full_name = 'corrupt_catalog.bin'
    _writeCatalog(str(tmp_path/full_name), 25, b'not zlib data')

    calls = _runLoadGMNCatalog(monkeypatch, tmp_path, full_name)

    assert calls['result'] == 'fallback'
    assert calls['downloads'] == 1
    assert not (tmp_path/full_name).exists()


def testUnreadableCatalogIsNotDeleted(monkeypatch, tmp_path):
    """ A permission error does not mean the catalog is corrupt, so the file must be kept. """

    full_name = 'no_permission_catalog.bin'
    (tmp_path/full_name).write_bytes(b'data')

    calls = _runLoadGMNCatalog(monkeypatch, tmp_path, full_name,
                               real_loader_error=PermissionError("Permission denied"))

    assert calls['result'] == 'fallback'
    assert calls['downloads'] == 0
    assert (tmp_path/full_name).exists()
