from unittest import TestCase
import pyodbc as pydbc
#import pydbc

dsn = "DSN=PostgreSQL R&D test database"
#dsn = "PostgreSQL R&D test database"

class TestConnect(TestCase):

    def test_connect(self):
       connection = pydbc.connect(dsn)

       self.assertTrue(hasattr(connection, 'close') and callable(getattr(connection, 'close'))
                  , "close method mising")
       self.assertTrue(hasattr(connection, 'commit') and callable(getattr(connection, 'commit'))
                  , "commit method mising")
       self.assertTrue(hasattr(connection, 'cursor') and callable(getattr(connection, 'cursor'))
                  , "cursor method mising")

       connection.close()

       #FIXME: according to PEP249 this should throw an Error or subclass
       with self.assertRaises(BaseException):
            # after closing a connection, all calls should raise an Error or subclass
            connection.connect("Oh Boy!")

    def test_cursor_setup_teardown(self):
        connection = pydbc.connect(dsn)
        cursor = connection.cursor()

        # https://www.python.org/dev/peps/pep-0249/#rowcount
        self.assertEqual(cursor.rowcount, -1 , 'no query has been performed, rowcount expected to be -1')
        self.assertIsNone(cursor.description, 'no query has been performed, description expected to be None')

        cursor.close()

        #FIXME: according to PEP249 this should throw an Error or subclass
        with self.assertRaises(BaseException):
            cursor.execute("Oh Boy!")

        connection.close()

    def test_close_connection_before_cursor(self):
        connection = pydbc.connect(dsn)
        cursor = connection.cursor()
        connection.close()

        #FIXME: according to PEP249 this should throw an Error or subclass
        with self.assertRaises(BaseException):
            connection.execute("Oh Boy!")

class TestResultSet(TestCase):

    def setUp(self):
        self.connection = pydbc.connect(dsn)
        self.cursor = self.connection.cursor()

    def tearDown(self):
        self.cursor.close()
        self.connection.close()

    def test_single_row_integer_result(self):
        result = self.cursor.execute("select 42")
        row = result.fetchone()
        self.assertItemsEqual(row, [42])

if __name__ == '__main__':
    from unittest import main
    main()


