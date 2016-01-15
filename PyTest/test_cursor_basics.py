from unittest import TestCase

from turbodbc import connect, InterfaceError


dsn = "Exasol R&D test database"


class TestCursorBasics(TestCase):

    def test_new_cursor_properties(self):
        connection = connect(dsn)
        cursor = connection.cursor()

        # https://www.python.org/dev/peps/pep-0249/#rowcount
        self.assertEqual(cursor.rowcount, -1)
        self.assertIsNone(cursor.description)
        self.assertEqual(cursor.arraysize, 1)

    def test_closed_cursor_raises_when_used(self):
        connection = connect(dsn)
        cursor = connection.cursor()

        cursor.close()

        with self.assertRaises(InterfaceError):
            cursor.execute("SELECT 42")

        with self.assertRaises(InterfaceError):
            cursor.executemany("SELECT 42")

        with self.assertRaises(InterfaceError):
            cursor.fetchone()

        with self.assertRaises(InterfaceError):
            cursor.fetchmany()

        with self.assertRaises(InterfaceError):
            cursor.fetchall()

        with self.assertRaises(InterfaceError):
            cursor.next()

    def test_closing_twice_is_ok(self):
        connection = connect(dsn)
        cursor = connection.cursor()

        cursor.close()
        cursor.close()
