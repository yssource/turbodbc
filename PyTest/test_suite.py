from unittest import TestCase
import pydbc

dsn = "PostgreSQL R&D test database"
dsn = "Exasol R&D test database"
#dsn = "MySQL R&D test database"

def has_method(_object, method_name):
    return hasattr(_object, method_name) and callable(getattr(_object, method_name))


class TestConnect(TestCase):
 
    def test_connect(self):
       connection = pydbc.connect(dsn)
 
       self.assertTrue(has_method(connection, 'close'), "close method missing")
       self.assertTrue(has_method(connection, 'commit'), "commit method missing")
       self.assertTrue(has_method(connection, 'cursor'), "cursor method missing")
 
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
        self.assertEqual(cursor.arraysize, 1, 'wrong default for attribute arraysize')
 
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
 
    def test_single_row_NULL_result(self):
        self.cursor.execute("select NULL")
        self.assertIn(self.cursor.rowcount, [-1, 1])
        row = self.cursor.fetchone()
        self.assertItemsEqual(row, [None])
        row = self.cursor.fetchone()
        self.assertIsNone(row)
 
    def test_single_row_integer_result(self):
        self.cursor.execute("select 42")
        self.assertIn(self.cursor.rowcount, [-1, 1])
        row = self.cursor.fetchone()
        self.assertItemsEqual(row, [42])
        row = self.cursor.fetchone()
        self.assertIsNone(row)
 
    def test_single_row_string_result(self):
        self.cursor.execute("select 'Oh Boy!'")
        self.assertIn(self.cursor.rowcount, [-1, 1])
        row = self.cursor.fetchone()
        self.assertItemsEqual(row, ['Oh Boy!'])
        row = self.cursor.fetchone()
        self.assertIsNone(row)
         
    def test_single_row_double_result(self):
        self.cursor.execute("select a from test_read_double")
        self.assertIn(self.cursor.rowcount, [-1, 1])
        row = self.cursor.fetchone()
        self.assertItemsEqual(row, [3.14])
        row = self.cursor.fetchone()
        self.assertIsNone(row)
         
    def test_single_row_large_numeric_result(self):
        self.cursor.execute("select -1234567890123.123456789")
        self.assertIn(self.cursor.rowcount, [-1, 1])
        row = self.cursor.fetchone()
        self.assertItemsEqual(row, ['-1234567890123.123456789'])
        row = self.cursor.fetchone()
        self.assertIsNone(row)
 
    def test_single_row_multiple_integer_result(self):
        self.cursor.execute("select 40, 41, 42, 43")
        self.assertIn(self.cursor.rowcount, [-1, 1])
        row = self.cursor.fetchone()
        self.assertItemsEqual(row, [40, 41, 42, 43])
        row = self.cursor.fetchone()
        self.assertIsNone(row)
 
    def test_multiple_row_iterate_result(self):
        self.cursor.execute("delete from test_integer")
        for i in xrange(1,10):
            self.cursor.execute("insert into test_integer values("+str(i)+")")
        self.cursor.execute("select * from test_integer order by a")
        for element in enumerate(self.cursor, start=1):
            self.assertItemsEqual([element[0]], element[1])
             
 
 
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
        self.assertIn(self.cursor.rowcount, [-1, 2])
        row = self.cursor.fetchone()
        self.assertItemsEqual(row, [1])
        row = self.cursor.fetchone()
        self.assertItemsEqual(row, [2])
         
    def test_multiple_row_single_boolean_result(self):
        self.cursor.execute("delete from test_bool")
        self.cursor.execute("insert into test_bool values (FALSE)")
        self.cursor.execute("insert into test_bool values (TRUE)")
        self.cursor.execute("select * from test_bool order by a")
        self.assertIn(self.cursor.rowcount, [-1, 2])
        row = self.cursor.fetchone()
        self.assertItemsEqual(row, [False])
        row = self.cursor.fetchone()
        self.assertItemsEqual(row, [True])
 
    def test_multiple_row_single_string_result(self):
        value_1 = 'Oh Boy!'
        value_2 = 'py(o)dbc'
        self.cursor.execute("delete from test_string")
        self.cursor.execute("insert into test_string values ('{}')".format(value_1))
        self.cursor.execute("insert into test_string values ('{}')".format(value_2))
        self.cursor.execute("select * from test_string order by a")
        self.assertIn(self.cursor.rowcount, [-1, 2])
        row = self.cursor.fetchone()
        self.assertItemsEqual(row, [value_1])
        row = self.cursor.fetchone()
        self.assertItemsEqual(row, [value_2])
 
    def test_fetchall_rows_single_integer_result(self):
        self.cursor.execute("delete from test_integer")
        self.cursor.execute("insert into test_integer values (1)")
        self.cursor.execute("insert into test_integer values (2)")
        self.cursor.execute("insert into test_integer values (3)")
        self.cursor.execute("select * from test_integer order by a")
        rows = self.cursor.fetchall()
        self.assertItemsEqual([[1],[2],[3]], (list(r) for r in rows))
         
    def test_fetchmany_rows_single_integer_result(self):
        self.cursor.execute("delete from test_integer")
        self.cursor.execute("insert into test_integer values (1)")
        self.cursor.execute("insert into test_integer values (2)")
        self.cursor.execute("insert into test_integer values (3)")
        self.cursor.execute("select * from test_integer order by a")
        rows = self.cursor.fetchmany(2)
        self.assertItemsEqual([[1],[2]], (list(r) for r in rows))
        rows = self.cursor.fetchmany(2)
        self.assertItemsEqual([[3]], (list(r) for r in rows))


class TestParameterBasedInsert(TestCase):

    def setUp(self):
        self.connection = pydbc.connect(dsn)
        self.cursor = self.connection.cursor()

    def tearDown(self):
        self.cursor.close()
        self.connection.close()

    def test_single_row_single_integer_column(self):
        self.cursor.execute("delete from test_integer")
        parameters = [1]
        self.cursor.execute("insert into test_integer values (?)", parameters)
         
        self.cursor.execute("select * from test_integer order by a")
        rows = self.cursor.fetchall()
        self.assertItemsEqual([parameters], (list(r) for r in rows))

    def test_multiple_rows_single_integer_column(self):
        self.cursor.execute("delete from test_integer")
        expected = [[1], [2], [3]]
        self.cursor.execute_many("insert into test_integer values (?)", expected)
        
        self.cursor.execute("select * from test_integer order by a")
        rows = self.cursor.fetchall()
        self.assertItemsEqual(expected, (list(r) for r in rows))


if __name__ == '__main__':
    from unittest import main
    main()


