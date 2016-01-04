from unittest import TestCase

from pydbc import connect, DatabaseError
from pydbc.connection import Connection


def has_method(_object, method_name):
    return hasattr(_object, method_name) and callable(getattr(_object, method_name))

class TestConnect(TestCase):

    def test_returns_connection_when_successful(self):
        valid_dsn = "Exasol R&D test database"
        connection = connect(valid_dsn)
        self.assertTrue(isinstance(connection, Connection))

    def test_raises_on_error(self):
        invalid_dsn = 'This data source does not exist'
        with self.assertRaises(DatabaseError):
            connect(invalid_dsn)
