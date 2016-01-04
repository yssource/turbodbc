from unittest import TestCase

from pydbc import connect, InterfaceError


dsn = "Exasol R&D test database"


class TestConnection(TestCase):
    def test_closing_connection_closes_cursor(self):
        connection = connect(dsn)
        cursor = connection.cursor()
        connection.close()

        with self.assertRaises(InterfaceError):
            cursor.execute("SELECT 42")