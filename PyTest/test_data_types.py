from unittest import TestCase

from pydbc.data_types import DataType, STRING, BINARY, NUMBER, DATETIME, ROWID

class TestDataTypes(TestCase):
    def test_equality(self):
        a = DataType(0)
        same_as_a = DataType(0)
        other_as_a = DataType(1)

        self.assertTrue(a == a)
        self.assertTrue(a == same_as_a)
        self.assertFalse(a == other_as_a)

    def test_inequality(self):
        a = DataType(0)
        same_as_a = DataType(0)
        other_as_a = DataType(1)

        self.assertFalse(a != a)
        self.assertFalse(a != same_as_a)
        self.assertTrue(a != other_as_a)

    def test_constants(self):
        self.assertNotEqual(STRING, BINARY)
        self.assertNotEqual(STRING, NUMBER)
        self.assertNotEqual(STRING, DATETIME)
        self.assertNotEqual(STRING, ROWID)

        self.assertNotEqual(BINARY, NUMBER)
        self.assertNotEqual(BINARY, DATETIME)
        self.assertNotEqual(BINARY, ROWID)

        self.assertNotEqual(NUMBER, DATETIME)
        self.assertNotEqual(NUMBER, ROWID)

        self.assertNotEqual(DATETIME, ROWID)