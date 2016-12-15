from mock import patch

from turbodbc.cursor import _has_arrow_support


def test_has_arrow_support_fails():
    with patch('__builtin__.__import__', side_effect=ImportError):
        assert _has_arrow_support() == False

def test_has_arrow_support_succeeds():
    assert _has_arrow_support() == True
