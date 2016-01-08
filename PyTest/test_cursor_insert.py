import pydbc

from cursor_test_case import CursorTestCase
from query_fixture import query_fixture


class InsertTests(object):

    def test_single_row(self):
        to_insert = [1]

        with query_fixture(self.cursor, self.fixtures, 'INSERT INTEGER') as table_name:
            self.cursor.execute("INSERT INTO {} VALUES (?)".format(table_name), to_insert)
            self.cursor.execute("SELECT a FROM {}".format(table_name))
            inserted = [list(row) for row in self.cursor.fetchall()]
            self.assertItemsEqual([to_insert], inserted)

    def test_string_column(self):
        to_insert = [['hi'], ['there'], ['test case']]

        with query_fixture(self.cursor, self.fixtures, 'INSERT STRING') as table_name:
            self.cursor.execute_many("INSERT INTO {} VALUES (?)".format(table_name), to_insert)
            self.cursor.execute("SELECT a FROM {}".format(table_name))
            inserted = [list(row) for row in self.cursor.fetchall()]
            self.assertItemsEqual(to_insert, inserted)

    def test_integer_column(self):
        to_insert = [[1], [2], [3]]

        with query_fixture(self.cursor, self.fixtures, 'INSERT INTEGER') as table_name:
            self.cursor.execute_many("INSERT INTO {} VALUES (?)".format(table_name), to_insert)
            self.cursor.execute("SELECT a FROM {}".format(table_name))
            inserted = [list(row) for row in self.cursor.fetchall()]
            self.assertItemsEqual(to_insert, inserted)

    def test_double_column(self):
        to_insert = [[1.23], [2.71], [3.14]]
 
        with query_fixture(self.cursor, self.fixtures, 'INSERT DOUBLE') as table_name:
            self.cursor.execute_many("INSERT INTO {} VALUES (?)".format(table_name), to_insert)
            self.cursor.execute("SELECT a FROM {}".format(table_name))
            inserted = [list(row) for row in self.cursor.fetchall()]
            self.assertItemsEqual(to_insert, inserted)

    def test_null(self):
        to_insert = [[None]]

        with query_fixture(self.cursor, self.fixtures, 'INSERT INTEGER') as table_name:
            self.cursor.execute_many("INSERT INTO {} VALUES (?)".format(table_name), to_insert)
            self.cursor.execute("SELECT a FROM {}".format(table_name))
            inserted = [list(row) for row in self.cursor.fetchall()]
            self.assertItemsEqual(to_insert, inserted)

    def test_mixed_data_columns(self):
        # second column has mixed data types in the same column
        # first column makes sure values of "good" columns are not affected
        to_insert = [[23, 1.23],
                     [42, 2]]
 
        with query_fixture(self.cursor, self.fixtures, 'INSERT MIXED') as table_name:
            self.cursor.execute_many("INSERT INTO {} VALUES (?, ?)".format(table_name), to_insert)
            self.cursor.execute("SELECT a, b FROM {}".format(table_name))
            inserted = [list(row) for row in self.cursor.fetchall()]
            self.assertItemsEqual(to_insert, inserted)

    def test_no_parameter_list(self):
        with query_fixture(self.cursor, self.fixtures, 'INSERT INTEGER') as table_name:
            self.cursor.execute_many("INSERT INTO {} VALUES (?)".format(table_name))
            self.cursor.execute("SELECT a FROM {}".format(table_name))
            inserted = [list(row) for row in self.cursor.fetchall()]
            self.assertEqual(0, len(inserted))

    def test_empty_parameter_list(self):
        to_insert = []

        with query_fixture(self.cursor, self.fixtures, 'INSERT INTEGER') as table_name:
            self.cursor.execute_many("INSERT INTO {} VALUES (?)".format(table_name), to_insert)
            self.cursor.execute("SELECT a FROM {}".format(table_name))
            inserted = [list(row) for row in self.cursor.fetchall()]
            self.assertItemsEqual(to_insert, inserted)

    def test_number_of_rows_exceeds_buffer_size(self):
        with query_fixture(self.cursor, self.fixtures, 'INSERT INTEGER') as table_name:
            numbers = 123
            to_insert = [[i] for i in xrange(numbers)]
            self.cursor.execute_many("INSERT INTO {} VALUES (?)".format(table_name), to_insert)

            self.cursor.execute("SELECT a FROM {}".format(table_name))
            retrieved = self.cursor.fetchall()
            actual_sum = sum([row[0] for row in retrieved])
            expected_sum = sum(xrange(numbers))
            self.assertEqual(expected_sum, actual_sum)


# Actual test cases

class TestCursorInsertExasol(InsertTests, CursorTestCase):
    dsn = "Exasol R&D test database"
    fixture_file_name = 'query_fixtures_exasol.json'


class TestCursorInsertPostgreSQL(InsertTests, CursorTestCase):
    dsn = "PostgreSQL R&D test database"
    fixture_file_name = 'query_fixtures_postgresql.json'


class TestCursorInsertMySQL(InsertTests, CursorTestCase):
    dsn = "MySQL R&D test database"
    fixture_file_name = 'query_fixtures_mysql.json'
