from unittest import TestCase
import pydbc


class SelectBaseTestCase(object):

    def setUp(self):
        self.connection = pydbc.connect(self.dsn)
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


class TestSelectExasol(SelectBaseTestCase, TestCase):
    dsn = "Exasol R&D test database"

class TestSelectPostgreSQL(SelectBaseTestCase, TestCase):
    dsn = "PostgreSQL R&D test database"

class TestSelectMySQL(SelectBaseTestCase, TestCase):
    dsn = "MySQL R&D test database"