from mock import patch

from turbodbc.cursor import _has_arrow_support

import pytest

# Skip all parquet tests if we can't import pyarrow.parquet
pa = pytest.importorskip('pyarrow')

# Ignore these with pytest ... -m 'not parquet'
pyarrow = pytest.mark.pyarrow


@pyarrow
def test_has_arrow_support_fails():
    with patch('__builtin__.__import__', side_effect=ImportError):
        assert _has_arrow_support() == False

@pyarrow
def test_has_arrow_support_succeeds():
    assert _has_arrow_support() == True
