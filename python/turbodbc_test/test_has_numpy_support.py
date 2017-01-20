import six

from mock import patch

from turbodbc.cursor import _has_numpy_support


# Python 2/3 compatibility
_IMPORT_FUNCTION_NAME = "{}.__import__".format(six.moves.builtins.__name__)


def test_has_numpy_support_fails():
    with patch(_IMPORT_FUNCTION_NAME, side_effect=ImportError):
        assert _has_numpy_support() == False


def test_has_numpy_support_succeeds():
    assert _has_numpy_support() == True