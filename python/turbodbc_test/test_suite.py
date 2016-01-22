from unittest import TestCase
import turbodbc

#dsn = "PostgreSQL R&D test database"
dsn = "Exasol R&D test database"
#dsn = "MySQL R&D test database"


class TestDML(TestCase):

    def setUp(self):
        self.connection = turbodbc.connect(dsn)
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


if __name__ == '__main__':
    from unittest import main
    main()
