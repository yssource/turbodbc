from unittest import TestCase
import pydbc

dsn = "PostgreSQL R&D test database"
#dsn = "Exasol R&D test database"

class TestConnect(TestCase):

    def test_connect(self):
       connection = pydbc.connect(dsn)

       self.assertTrue(hasattr(connection, 'close') and callable(getattr(connection, 'close'))
                  , "close method missing")
       self.assertTrue(hasattr(connection, 'commit') and callable(getattr(connection, 'commit'))
                  , "commit method missing")
       self.assertTrue(hasattr(connection, 'cursor') and callable(getattr(connection, 'cursor'))
                  , "cursor method missing")

       connection.close()

       with self.assertRaises(pydbc.Error):
            # after closing a connection, all calls should raise an Error or subclass
            connection.cursor()

    def test_connect_error(self):
        self.assertRaises(pydbc.Error, pydbc.connect, "Oh Boy!")
        
 

    def test_cursor_setup_teardown(self):
        connection = pydbc.connect(dsn)
        cursor = connection.cursor()

        # https://www.python.org/dev/peps/pep-0249/#rowcount
        self.assertEqual(cursor.rowcount, -1 , 'no query has been performed, rowcount expected to be -1')
        self.assertIsNone(cursor.description, 'no query has been performed, description expected to be None')

        cursor.close()

        with self.assertRaises(pydbc.Error):
            cursor.execute("Oh Boy!")

        connection.close()

    def test_close_connection_before_cursor(self):
        connection = pydbc.connect(dsn)
        cursor = connection.cursor()
        connection.close()

        with self.assertRaises(pydbc.Error):
            cursor.execute("Oh Boy!")

class TestDQL(TestCase):

    def setUp(self):
        self.connection = pydbc.connect(dsn)
        self.cursor = self.connection.cursor()

    def tearDown(self):
        self.cursor.close()
        self.connection.close()

    def test_single_row_integer_result(self):
        self.cursor.execute("select 42")
        self.assertEqual(self.cursor.rowcount, 1)
        row = self.cursor.fetchone()
        self.assertItemsEqual(row, [42])
        row = self.cursor.fetchone()
        self.assertIsNone(row)

    def test_single_row_string_result(self):
        self.cursor.execute("select 'Oh Boy!'")
        self.assertEqual(self.cursor.rowcount, 1)
        row = self.cursor.fetchone()
        self.assertItemsEqual(row, ['Oh Boy!'])
        row = self.cursor.fetchone()
        self.assertIsNone(row)

    def test_single_row_multiple_integer_result(self):
        self.cursor.execute("select 40, 41, 42, 43")
        self.assertEqual(self.cursor.rowcount, 1)
        row = self.cursor.fetchone()
        self.assertItemsEqual(row, [40, 41, 42, 43])
        row = self.cursor.fetchone()
        self.assertIsNone(row)

class TestDML(TestCase):

    def setUp(self):
        self.connection = pydbc.connect(dsn)
        self.cursor = self.connection.cursor()

    def tearDown(self):
        self.cursor.close()
        self.connection.close()

    def test_multiple_row_single_integer_result(self):
        self.cursor.execute("delete from test_integer")
        self.cursor.execute("insert into test_integer values (1)")
        self.cursor.execute("insert into test_integer values (2)")
        self.cursor.execute("select * from test_integer order by a")
        self.assertEqual(self.cursor.rowcount, 2)
        row = self.cursor.fetchone()
        self.assertItemsEqual(row, [1])
        row = self.cursor.fetchone()
        self.assertItemsEqual(row, [2])

    def test_multiple_row_single_string_result(self):
        self.cursor.execute("delete from test_string")
        self.cursor.execute("insert into test_string values ('Oh Boy!')")
        self.cursor.execute("insert into test_string values ('py(o)dbc')")
        self.cursor.execute("select * from test_string order by a")
        self.assertEqual(self.cursor.rowcount, 2)
        row = self.cursor.fetchone()
        self.assertItemsEqual(row, [1])
        row = self.cursor.fetchone()
        self.assertItemsEqual(row, [2])


if __name__ == '__main__':
    from unittest import main
    main()


