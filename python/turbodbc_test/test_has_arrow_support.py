from mock import patch

from turbodbc.cursor import _has_arrow_support

import pytest

# Skip all parquet tests if we can't import pyarrow.parquet
pa = pytest.importorskip('pyarrow')

# Ignore these with pytest ... -m 'not parquet'
pyarrow = pytest.mark.pyarrow

# Python 2/3 compatibility
_IMPORT_FUNCTION_NAME = "{}.__import__".format(six.moves.builtins.__name__)


@pyarrow
def test_has_arrow_support_fails():
    with patch(_IMPORT_FUNCTION_NAME, side_effect=ImportError):
        assert _has_arrow_support() == False


@pyarrow
def test_has_arrow_support_succeeds():
    assert _has_arrow_support() == True
