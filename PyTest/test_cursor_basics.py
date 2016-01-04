from unittest import TestCase

from pydbc import connect, Error

dsn = "Exasol R&D test database"

class TestCursorBasics(TestCase):

    def test_cursor_setup_teardown(self):
        connection = connect(dsn)
        cursor = connection.cursor()

        # https://www.python.org/dev/peps/pep-0249/#rowcount
        self.assertEqual(cursor.rowcount, -1 , 'no query has been performed, rowcount expected to be -1')
        self.assertIsNone(cursor.description, 'no query has been performed, description expected to be None')
        self.assertEqual(cursor.arraysize, 1, 'wrong default for attribute arraysize')

        cursor.close()

        with self.assertRaises(Error):
            cursor.execute("Oh Boy!")

        connection.close()

    def test_close_connection_before_cursor(self):
        connection = connect(dsn)
        cursor = connection.cursor()
        connection.close()

        with self.assertRaises(Error):
            cursor.execute("Oh Boy!")