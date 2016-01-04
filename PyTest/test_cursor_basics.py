from unittest import TestCase

from pydbc import connect, Error


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

        with self.assertRaises(Error):
            cursor.execute("SELECT 42")

        with self.assertRaises(Error):
            cursor.execute_many("SELECT 42")

        with self.assertRaises(Error):
            cursor.fetchone()

        with self.assertRaises(Error):
            cursor.fetchmany()

        with self.assertRaises(Error):
            cursor.fetchall()

        with self.assertRaises(Error):
            cursor.next()

    def test_closing_twice_is_ok(self):
        connection = connect(dsn)
        cursor = connection.cursor()

        cursor.close()
        cursor.close()

    def test_close_connection_before_cursor(self):
        connection = connect(dsn)
        cursor = connection.cursor()
        connection.close()

        with self.assertRaises(Error):
            cursor.execute("Oh Boy!")