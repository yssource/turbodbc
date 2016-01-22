from unittest import TestCase

from turbodbc import connect, DatabaseError
from turbodbc.connection import Connection

VALID_DSN = "Exasol R&D test database"


class TestConnect(TestCase):

    def test_returns_connection_when_successful(self):
        connection = connect(VALID_DSN)
        self.assertTrue(isinstance(connection, Connection))

    def test_returns_connection_with_explicit_dsn(self):
        connection = connect(dsn=VALID_DSN)
        self.assertTrue(isinstance(connection, Connection))

    def test_raises_on_invalid_dsn(self):
        invalid_dsn = 'This data source does not exist'
        with self.assertRaises(DatabaseError):
            connect(invalid_dsn)

    def test_raises_on_invalid_additional_option(self):
        with self.assertRaises(DatabaseError):
            connect(dsn=VALID_DSN, exauid='invalid user')

    def test_buffer_sizes_default_values(self):
        connection = connect("Exasol R&D test database",
                             rows_to_buffer=317,
                             parameter_sets_to_buffer=123)

        self.assertEqual(connection.impl.rows_to_buffer, 317)
        self.assertEqual(connection.impl.parameter_sets_to_buffer, 123)