from unittest import TestCase

from turbodbc import connect


dsn = "Exasol R&D test database"


class TestCursorUnsupportedFeatures(TestCase):
    """
    Test optional features mentioned in PEP-249 "behave" as specified 
    """
    def test_callproc_unsupported(self):
        cursor = connect(dsn).cursor()

        with self.assertRaises(AttributeError):
            cursor.callproc()

    def test_nextset_unsupported(self):
        cursor = connect(dsn).cursor()

        with self.assertRaises(AttributeError):
            cursor.nextset()